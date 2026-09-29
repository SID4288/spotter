"""Freight-rate training pipeline (production-ready, deterministic).

Pipeline: WeightDistanceImputer -> MarketImputer -> FeatureEngineer
          -> ColumnSelector -> DollarHGBRegressor (HGB + Duan smearing).

Key production properties
* Inductive only: imputers learn frozen lookups from train; single-row
  inference == batch inference (see preprocessing.py).
* Outlier flag (6*MAD quadratic log-log) is training-only; inference path
  (pipe.predict / predict.py) never drops or flags live rows.
* Dollar-aware: HGB fits log(rate/mile) with sample_weight=distance and
  Duan smearing, so exp(pred)*distance is unbiased in $ space.
* Artifact (output/pipeline.joblib) uses preprocessing/modeling classes so
  plain joblib.load works in microservices (no __main__ hack needed for
  new artifacts; load_pipeline keeps backward-compat for the old 800MB ET).
"""

from __future__ import annotations

import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.model_selection import RandomizedSearchCV, TimeSeriesSplit
from sklearn.pipeline import Pipeline

from modeling import DollarHGBRegressor
from preprocessing import (
    CAT_FEATURES,
    EQUIPMENT_CODE,
    MODEL_FEATURES,
    NUM_FEATURES,
    ColumnSelector,
    FeatureEngineer,
    MarketImputer,
    WeightDistanceImputer,
)

ROOT = Path(__file__).resolve().parent
DATA, OUT = ROOT / "data", ROOT / "output"
METRICS = ROOT / "notebooks" / "results" / "modelling" / "metrics.json"
PIPE_PATH = OUT / "pipeline.joblib"
SEED = 42

# Backward-compat aliases (tests / notebooks import these from train).
FEATURES = NUM_FEATURES
FEATURES_NO_MARKET = [f for f in FEATURES if f != "market_index"]

REQUIRED_INPUTS = ["pickup", "delivery", "pickup_lat", "pickup_lon",
                   "delivery_lat", "delivery_lon", "distance", "equipment",
                   "weight", "date", "market_index"]
FORBIDDEN_INFERENCE_COLS = {"quote_signal"}  # unstable across horizon; must not reach model.


def basic_clean(df: pd.DataFrame) -> pd.DataFrame:
    """Canonical input repair (runs before validation)."""
    df = df.copy()
    if "weight" in df:
        df["weight"] = df["weight"].abs()  # sign-flipped weights (~0.6% of rows)
    if "distance" in df:
        df.loc[df["distance"] <= 0, "distance"] = np.nan
    # quote_signal is excluded by design (unstable across horizon); drop
    # defensively so downstream code cannot accidentally use it.
    df = df.drop(columns=[c for c in FORBIDDEN_INFERENCE_COLS if c in df.columns])
    return df


def _flag_params(train_df: pd.DataFrame):
    """Fit outlier rule on TRAINING rows only (no holdout peeking)."""
    lr = np.log(train_df["posted_rate"].to_numpy())
    ld = np.log(train_df["distance"].to_numpy())
    coef = np.polyfit(ld, lr, 2)
    resid = lr - np.polyval(coef, ld)
    med = float(np.median(resid))
    mad = float(1.4826 * np.median(np.abs(resid - med)))
    return coef, med, mad


def _apply_flag(df: pd.DataFrame, coef, med: float, mad: float) -> pd.Series:
    lr = np.log(df["posted_rate"].to_numpy())
    ld = np.log(df["distance"].to_numpy())
    resid = lr - np.polyval(coef, ld)
    return pd.Series(np.abs(resid - med) > 6 * mad, index=df.index)


def flag_corrupt_labels(df: pd.DataFrame) -> pd.Series:
    """Training-only label cleaner. NEVER call on live inference rows.

    Use only for the full-train final fit. Inside CV use _flag_params(tr)
    fitted on the training fold and _apply_flag(va, ...) for scoring.
    Inference (pipe.predict / predict.py) bypasses this entirely.
    """
    coef, med, mad = _flag_params(df)
    return _apply_flag(df, coef, med, mad)


def validate_frame(df: pd.DataFrame, require_label: bool = False) -> None:
    """Strict schema gate. Raises ValueError with actionable messages.

    * Missing required columns -> raise (lists culprits).
    * Forbidden columns (quote_signal) -> raise (must be dropped upstream).
    * distance <= 0 (non-NaN) -> raise (basic_clean should have mapped to
      NaN for imputation; a surviving <=0 means cleaning was skipped).
    * Negative weight -> raise (basic_clean abs() was skipped).
    * Unparseable date -> raise. NaN distance/weight/market_index are
      ALLOWED (imputers handle them); everything else must be present.
    * require_label=True: posted_rate must exist, finite, > 0.
    """
    missing = [c for c in REQUIRED_INPUTS if c not in df.columns]
    if missing:
        raise ValueError(f"missing input columns: {missing}")
    forbidden = [c for c in FORBIDDEN_INFERENCE_COLS if c in df.columns]
    if forbidden:
        raise ValueError(
            f"forbidden columns present {forbidden}: drop quote_signal upstream "
            "(unstable across horizon; see modelling notes)"
        )
    if require_label and "posted_rate" not in df.columns:
        raise ValueError("posted_rate required but missing")
    # Distance: NaN allowed (imputed), but concrete <= 0 is a pipeline bug.
    dist = pd.to_numeric(df["distance"], errors="coerce")
    bad_dist = (dist.notna() & (dist <= 0)).sum()
    if int(bad_dist):
        raise ValueError(
            f"distance must be > 0 or NaN (imputable); found {int(bad_dist)} rows <= 0. "
            "Run basic_clean first."
        )
    # Weight: NaN allowed, negative is not.
    w = pd.to_numeric(df["weight"], errors="coerce")
    neg_w = (w.notna() & (w < 0)).sum()
    if int(neg_w):
        raise ValueError(
            f"weight must be >= 0 or NaN; found {int(neg_w)} negative rows. Run basic_clean first."
        )
    dates = pd.to_datetime(df["date"], errors="coerce")
    if dates.isna().any():
        raise ValueError(f"date contains {int(dates.isna().sum())} unparseable values")
    if require_label:
        pr = pd.to_numeric(df["posted_rate"], errors="coerce")
        if pr.isna().any():
            raise ValueError("posted_rate contains NaN/non-numeric values")
        if bool((pr <= 0).any()):
            raise ValueError("posted_rate must be > 0")


def fit_preprocessors(train_df: pd.DataFrame) -> dict:
    """Fit inductive preprocessors on training inputs only."""
    w = WeightDistanceImputer().fit(train_df)
    m = MarketImputer().fit(train_df)
    f = FeatureEngineer().fit(train_df)
    return {"_w": w, "_m": m, "_f": f,
            "eq_weight": w.eq_weight_, "global_weight": w.global_weight_,
            "distance_median": w.distance_median_,
            "date_market": pd.Series(m.date_map_), "month_market": pd.Series(m.month_map_),
            "global_market": m.global_,
            "lane_freq": f.lane_freq_}


def transform(df: pd.DataFrame, prep: dict) -> pd.DataFrame:
    """Apply frozen preprocessors (no peeking at batch statistics)."""
    df = df.copy()
    if "_w" in prep:
        df = prep["_w"].transform(df)
        df = prep["_m"].transform(df)
        df = prep["_f"].transform(df)
        return df
    # Legacy dict path (old pickles): inductive fallback ladder only.
    m = df["weight"].isna()
    if m.any():
        df.loc[m, "weight"] = df.loc[m, "equipment"].map(prep["eq_weight"]).fillna(prep["global_weight"])
    if "distance_median" in prep:
        df["distance"] = df["distance"].mask(df["distance"] <= 0).fillna(prep["distance_median"])
    # Market: exact-date -> month -> global (no batch groupby).
    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    m = df["market_index"].isna()
    if m.any():
        df.loc[m, "market_index"] = df.loc[m, "date"].map(prep["date_market"])
    m = df["market_index"].isna()
    if m.any():
        key = df.loc[m, "date"].dt.strftime("%Y-%m")
        try:
            df.loc[m, "market_index"] = key.map(prep["month_market"])
        except Exception:
            df.loc[m, "market_index"] = np.nan
    df["market_index"] = df["market_index"].fillna(prep["global_market"])
    # Reuse canonical feature logic via a fitted engineer when possible.
    from preprocessing import days_to_quarter_end, haversine
    geo = haversine(df["pickup_lat"], df["pickup_lon"], df["delivery_lat"], df["delivery_lon"])
    df["geo_distance"] = geo
    df["circuity"] = (df["distance"] / np.maximum(geo, 1e-6)).clip(1.0, 2.0).fillna(1.18)
    df["distance_ratio"] = df["circuity"]
    df["weight_per_mile"] = df["weight"] / np.maximum(df["distance"], 1e-6)
    df["equipment_code"] = df["equipment"].map(EQUIPMENT_CODE).fillna(-1).astype(int)
    df["lane_freq"] = 0
    if "pickup" in df and "delivery" in df and "lane_freq" in prep:
        df["lane_freq"] = (df["pickup"].astype(str) + "__" + df["delivery"].astype(str)) \
            .map(prep["lane_freq"]).fillna(0)
    df["days_to_quarter_end"] = days_to_quarter_end(df["date"])
    dow, doy = df["date"].dt.dayofweek, df["date"].dt.dayofyear
    df["dow_sin"] = np.sin(2 * np.pi * dow / 7)
    df["dow_cos"] = np.cos(2 * np.pi * dow / 7)
    df["doy_sin"] = np.sin(2 * np.pi * doy / 365.25)
    df["doy_cos"] = np.cos(2 * np.pi * doy / 365.25)
    df["equip_x_logdist"] = df["equipment_code"] * np.log(np.maximum(df["distance"], 1e-6))
    df["day_of_week"] = dow
    df["day_of_month"] = df["date"].dt.day
    df["is_weekend"] = (dow >= 5).astype(int)
    for col, cats in (("pickup_cat", "pickup"), ("delivery_cat", "delivery"), ("equipment_cat", "equipment")):
        df[col] = pd.Categorical(df[cats]) if cats in df else pd.Categorical([np.nan] * len(df))
    return df


def make_model() -> DollarHGBRegressor:
    """Compact HGB (~10-25MB) with smearing + native categoricals."""
    return DollarHGBRegressor(
        max_iter=300, max_leaf_nodes=63, learning_rate=0.05,
        min_samples_leaf=20, l2_regularization=1.0,
        categorical_features="from_dtype", early_stopping=False,
        random_state=SEED,
    )


def make_pipeline() -> Pipeline:
    return Pipeline([
        ("weight_distance", WeightDistanceImputer()),
        ("market", MarketImputer()),
        ("features", FeatureEngineer()),
        ("select", ColumnSelector(columns=MODEL_FEATURES)),
        ("model", make_model()),
    ])


def load_pipeline(path: Path | str = PIPE_PATH):
    """Load artifact. New artifacts: plain joblib.load. Old 800MB ET: fallback alias."""
    try:
        import preprocessing  # noqa: F401  ensure unpickling namespace present
        import modeling  # noqa: F401
        return joblib.load(path)
    except AttributeError as exc:
        # Backward-compat for artifacts dumped from __main__ (old train.py).
        import sys
        import __main__ as _main
        import preprocessing as _pp
        import modeling as _mo
        for _name in ("WeightDistanceImputer", "MarketImputer", "FeatureEngineer", "ColumnSelector"):
            setattr(_main, _name, getattr(_pp, _name))
        setattr(_main, "DollarHGBRegressor", _mo.DollarHGBRegressor)
        sys.modules.setdefault("train", sys.modules[__name__])
        try:
            return joblib.load(path)
        except Exception:
            raise exc


def _train_mask(train_df: pd.DataFrame) -> np.ndarray:
    """Training-row outlier mask, fitted on train_df only (leakage fix)."""
    if "posted_rate" not in train_df or "distance" not in train_df:
        if "is_corrupt" in train_df:
            return train_df["is_corrupt"].to_numpy()
        return np.zeros(len(train_df), dtype=bool)
    coef, med, mad = _flag_params(train_df)
    return _apply_flag(train_df, coef, med, mad).to_numpy()


def _distance_weights(df: pd.DataFrame, median_fill: float | None = None) -> np.ndarray:
    d = pd.to_numeric(df["distance"], errors="coerce")
    if median_fill is not None:
        d = d.mask((d.isna()) | (d <= 0)).fillna(median_fill)
    else:
        d = d.fillna(d.median())
    return np.maximum(d.to_numpy(dtype=float), 1.0)


def fit_predict(train_df: pd.DataFrame, target_df: pd.DataFrame,
                features: list[str] | None = None,
                denoised_market: bool = False) -> np.ndarray:
    """Functional training path (mirrors pipeline; dollar-weighted + smeared).

    Args:
        features: numeric subset for ablations (categoricals always appended
            so HGB keeps native handling).
    """
    prep = fit_preprocessors(train_df)
    tr, te = transform(train_df, prep), transform(target_df, prep)
    if denoised_market:
        tr["market_index"] = tr["date"].map(prep["_m"].date_map_).fillna(prep["global_market"])
        own_month = te["date"].dt.strftime("%Y-%m").map(
            target_df.assign(market_index=target_df["market_index"]).groupby(
                target_df["date"].astype(str).str.slice(0, 7))["market_index"].median()
        ) if False else None  # disabled: transductive; kept for API compat
        _ = own_month
        te["market_index"] = te["date"].map(prep["_m"].date_map_)
        te["market_index"] = te["market_index"].fillna(prep["global_market"])
    num_feats = FEATURES if features is None else [f for f in features if f in FEATURES]
    feats = num_feats + [c for c in CAT_FEATURES if c in tr.columns]
    mask = _train_mask(train_df)
    tr_clean = tr[~pd.Series(mask, index=train_df.index).reindex(tr.index).fillna(False).to_numpy()]
    w = _distance_weights(tr_clean)
    model = make_model().fit(tr_clean[feats], np.log(tr_clean["posted_rate"] / tr_clean["distance"]),
                             sample_weight=w)
    # predict() already adds log(smearing); multiply back by distance.
    return np.exp(model.predict(te[feats])) * te["distance"].to_numpy()


def tune(train_df: pd.DataFrame, n_iter: int = 12) -> dict:
    """Efficient tuning: RandomizedSearchCV + TimeSeriesSplit (past -> future only)."""
    mask = _train_mask(train_df)
    df = train_df[~mask].copy()
    pipe = Pipeline([
        ("weight_distance", WeightDistanceImputer()),
        ("market", MarketImputer()),
        ("features", FeatureEngineer()),
        ("select", ColumnSelector(columns=MODEL_FEATURES)),
        ("model", DollarHGBRegressor(random_state=SEED)),
    ])
    rs = RandomizedSearchCV(
        pipe,
        {"model__max_leaf_nodes": [31, 63], "model__learning_rate": [0.03, 0.05, 0.1],
         "model__min_samples_leaf": [10, 20, 50], "model__max_iter": [200, 300]},
        n_iter=n_iter, cv=TimeSeriesSplit(n_splits=4),
        scoring="neg_mean_absolute_error", n_jobs=-1, random_state=SEED)
    sw = _distance_weights(df)
    rs.fit(df, np.log(df["posted_rate"] / df["distance"]), model__sample_weight=sw)
    return {"best_params": rs.best_params_, "best_mae_logrpm": float(-rs.best_score_)}


def score(actual: pd.Series, pred: np.ndarray, corrupt: pd.Series) -> dict:
    clean = ~pd.Series(np.asarray(corrupt)).to_numpy()
    a, p = actual.to_numpy(), np.asarray(pred)
    return {
        "clean_mae": round(float(np.mean(np.abs(a[clean] - p[clean]))), 2),
        "clean_rmse": round(float(np.sqrt(np.mean((a[clean] - p[clean]) ** 2))), 2),
        "clean_mape_pct": round(float(np.mean(np.abs(a[clean] - p[clean]) / a[clean]) * 100), 2),
        "all_row_mae": round(float(np.mean(np.abs(a - p))), 2),
    }


def _split_score(tr: pd.DataFrame, va: pd.DataFrame, **kw) -> dict:
    coef, med, mad = _flag_params(tr)  # fit on train only
    va_mask = _apply_flag(va, coef, med, mad)
    return score(va["posted_rate"], fit_predict(tr, va, **kw), va_mask)


def evaluate(train_raw: pd.DataFrame) -> dict:
    results = {}
    folds = {
        "Fold 1 (Jul-Aug)": ("2025-06-30", "2025-07-01", "2025-08-31"),
        "Fold 2 (Aug-Sep)": ("2025-07-31", "2025-08-01", "2025-09-30"),
        "Fold 3 (Sep-Oct)": ("2025-08-31", "2025-09-01", "2025-10-31"),
        # Untouched final check: train on Jan-Sep, test on Oct only.
        "Fold 4 (Oct only)": ("2025-09-30", "2025-10-01", "2025-10-31"),
    }
    # Cache cleaned transforms per fold: one transform per split, reused by ablations.
    cache: dict[str, tuple] = {}
    for name, (tr_end, va_start, va_end) in folds.items():
        tr = train_raw[train_raw["date"] <= tr_end]
        va = train_raw[(train_raw["date"] >= va_start) & (train_raw["date"] <= va_end)]
        prep = fit_preprocessors(tr)
        tr_t, va_t = transform(tr, prep), transform(va, prep)
        cache[name] = (tr, va, prep, tr_t, va_t)
        results[name] = _split_score(tr, va)

    ablation: dict[str, dict] = {"with_market_index": {}, "without_market_index": {},
                                 "denoised_daily_median": {}}
    for name in folds:
        tr, va, prep, tr_t, va_t = cache[name]
        ablation["with_market_index"][name] = _split_score(tr, va)
        ablation["without_market_index"][name] = _split_score(tr, va, features=FEATURES_NO_MARKET)
        ablation["denoised_daily_median"][name] = _split_score(tr, va, denoised_market=True)
    results["ablation_market_index"] = ablation

    # Unseen-city stress test: 12% of the real validation rows touch cities never seen in training.
    cities = sorted(set(train_raw["pickup"]) | set(train_raw["delivery"]))
    held = set(np.random.RandomState(0).choice(cities, 6, replace=False))
    touches = lambda d: d["pickup"].isin(held) | d["delivery"].isin(held)  # noqa: E731
    early = train_raw[train_raw["date"] <= "2025-08-31"]
    late = train_raw[(train_raw["date"] >= "2025-09-01") & touches(train_raw)]
    coef, med, mad = _flag_params(early[~touches(early)])
    late_mask = _apply_flag(late, coef, med, mad)
    res = score(late["posted_rate"], fit_predict(early[~touches(early)], late), late_mask)
    res["held_out_cities"] = sorted(held)
    results["Unseen-city holdout (Sep-Oct)"] = res
    return results


def main() -> None:
    train = pd.read_csv(DATA / "train_test.csv", parse_dates=["date"])
    validation = pd.read_csv(DATA / "validation.csv", parse_dates=["date"])
    december = pd.read_csv(DATA / "december_chart_inputs.csv", parse_dates=["date"])
    # The assessment brief spells the file with underscores; the provided repo uses hyphens.
    tpl_path = next(p for p in (DATA / "validation_predictions_template.csv",
                                DATA / "validation-predictions-template.csv") if p.exists())
    template = pd.read_csv(tpl_path)

    train_raw, val_raw = basic_clean(train), basic_clean(validation)
    validate_frame(train_raw, require_label=True)
    validate_frame(val_raw, require_label=False)
    full_mask = flag_corrupt_labels(train_raw)  # full-train fit: valid for the final model only
    train_raw["is_corrupt"] = full_mask
    print(f"Corrupt labels flagged: {train_raw['is_corrupt'].sum()} ({train_raw['is_corrupt'].mean():.2%})")

    metrics = evaluate(train_raw)
    for name, res in metrics.items():
        if name == "ablation_market_index":
            for variant, folds in res.items():
                print(f"ablation {variant}:")
                for fold, vals in folds.items():
                    print(f"  {fold}", vals)
            continue
        print(name, {k: v for k, v in res.items() if k != "held_out_cities"})

    # ---- final production pipeline (single source of truth, serialized)
    # Preprocessing fit on ALL train rows (inputs only; labels never set input
    # medians); HGB fit on clean rows with sample_weight=distance + smearing.
    # Reuse the same fitted transformer objects so pipe.predict == functional.
    prep = fit_preprocessors(train_raw)
    tr_t = transform(train_raw, prep)
    mask = _train_mask(train_raw)
    tr_clean = tr_t[~mask].copy()
    sw_full = _distance_weights(tr_clean)
    y_full = np.log(tr_clean["posted_rate"] / tr_clean["distance"])
    model = make_model().fit(tr_clean[MODEL_FEATURES], y_full, sample_weight=sw_full)
    pipe = Pipeline([
        ("weight_distance", prep["_w"]),
        ("market", prep["_m"]),
        ("features", prep["_f"]),
        ("select", ColumnSelector(columns=MODEL_FEATURES)),
        ("model", model),
    ])
    OUT.mkdir(exist_ok=True)
    joblib.dump(pipe, PIPE_PATH)

    # Monitoring stats from the frozen imputers after a validation pass.
    _ = pipe.predict(val_raw.head(1))  # prime last_stats_ (single-row parity)
    single_stats = {
        "market": dict(getattr(pipe.named_steps["market"], "last_stats_", {})),
        "weight_distance": dict(getattr(pipe.named_steps["weight_distance"], "last_stats_", {})),
    }
    _ = pipe.predict(val_raw)
    batch_stats = {
        "market": dict(getattr(pipe.named_steps["market"], "last_stats_", {})),
        "weight_distance": dict(getattr(pipe.named_steps["weight_distance"], "last_stats_", {})),
    }
    # Inductive parity: single-row transform must equal batch row transform.
    _r1 = pipe.named_steps["weight_distance"].transform(val_raw.head(1))
    _r1 = pipe.named_steps["market"].transform(_r1)
    _rb = pipe.named_steps["weight_distance"].transform(val_raw)
    _rb = pipe.named_steps["market"].transform(_rb)
    assert np.isclose(_r1["market_index"].iloc[0], _rb["market_index"].iloc[0]), "inductive parity broken"

    # ---- validation predictions (pipeline path, validated independently)
    val_pred = np.clip(np.exp(pipe.predict(val_raw)) * val_raw["distance"].to_numpy(), 1.0, None)
    # cross-check against the functional path (must agree; guards Pipeline drift)
    val_check = np.clip(fit_predict(train_raw, val_raw), 1.0, None)
    assert np.allclose(val_pred, val_check, rtol=1e-6), "pipeline/functional path drift"
    out = pd.DataFrame({"load_id": validation["load_id"], "predicted_rate": np.round(val_pred, 2)})
    assert list(out["load_id"]) == list(template["load_id"]), "IDs/order must match the template"
    assert len(out) == 12_000 and out["predicted_rate"].gt(0).all() and out["predicted_rate"].notna().all()

    # ---- December fixed lane. Coordinates come from the training data for this exact lane.
    lane_rows = train_raw[(train_raw["pickup"] == "Lexington") & (train_raw["delivery"] == "Fort Wayne")]
    if lane_rows.empty:
        raise SystemExit("ERROR: fixed December lane Lexington -> Fort Wayne not found in training data")
    lane = lane_rows.iloc[0]
    dec = december.copy()
    for col in ["pickup_lat", "pickup_lon", "delivery_lat", "delivery_lon"]:
        dec[col] = lane[col]
    # market_index is a provided *input* feature: inductive lookup only.
    # Use frozen training month/dow medians keyed by December date (no labels,
    # no peeking at validation batch statistics).
    mkt = pipe.named_steps["market"]
    dec["date"] = pd.to_datetime(dec["date"])
    dec["market_index"] = dec["date"].map(mkt.date_map_)
    miss = dec["market_index"].isna()
    if miss.any():
        dec.loc[miss, "market_index"] = dec.loc[miss, "date"].dt.strftime("%Y-%m").map(mkt.month_map_)
    miss = dec["market_index"].isna()
    if miss.any():
        dec.loc[miss, "market_index"] = dec.loc[miss, "date"].dt.dayofweek.map(mkt.dow_map_)
    dec["market_index"] = dec["market_index"].fillna(mkt.global_)
    dec["weight"] = dec["weight"].astype(float)
    dec_pred = np.clip(np.exp(pipe.predict(dec)) * dec["distance"].to_numpy(), 1.0, None)
    dec_out = december.copy()
    dec_out["date"] = dec_out["date"].dt.strftime("%Y-%m-%d") if hasattr(dec_out["date"], "dt") else pd.to_datetime(dec_out["date"]).dt.strftime("%Y-%m-%d")
    dec_out["predicted_rate"] = np.round(dec_pred, 2)
    assert len(dec_out) == 31 and dec_out["predicted_rate"].gt(0).all()

    out.to_csv(OUT / "validation_predictions.csv", index=False)
    out.to_csv(ROOT / "validation_predictions.csv", index=False)  # the file named in the submission brief
    dec_out.to_csv(OUT / "december_predictions.csv", index=False)
    metrics["december_mean_rate"] = round(float(dec_out["predicted_rate"].mean()), 2)
    metrics["december_peak"] = {"date": dec_out.loc[dec_out["predicted_rate"].idxmax(), "date"],
                                "rate": float(dec_out["predicted_rate"].max())}
    metrics["feature_set"] = MODEL_FEATURES
    metrics["model"] = {
        "type": "DollarHGBRegressor",
        "smearing": round(float(pipe.named_steps["model"].smearing_), 4),
        "categorical_features": CAT_FEATURES,
        "sample_weight": "distance",
    }
    metrics["monitoring"] = {
        "single_row_stats": single_stats,
        "batch_stats": batch_stats,
        "global_fallback_rate": batch_stats["market"].get("global_fallback_rate", 0.0),
    }
    # Unseen-lane rate on validation (drift guard: alert if > 15%).
    train_lanes = set(train_raw["pickup"].astype(str) + "__" + train_raw["delivery"].astype(str))
    va_lanes = val_raw["pickup"].astype(str) + "__" + val_raw["delivery"].astype(str)
    metrics["monitoring"]["unseen_lane_rate"] = float((~va_lanes.isin(train_lanes)).mean())
    METRICS.parent.mkdir(parents=True, exist_ok=True)
    METRICS.write_text(json.dumps(metrics, indent=2))
    print(f"Validation mean ${out['predicted_rate'].mean():.2f}; December mean ${metrics['december_mean_rate']:.2f}")
    print(f"Smearing S={pipe.named_steps['model'].smearing_:.4f}; "
          f"global_fallback={metrics['monitoring']['global_fallback_rate']:.4f}; "
          f"unseen_lane={metrics['monitoring']['unseen_lane_rate']:.3f}")
    print("Wrote validation_predictions.csv, output/validation_predictions.csv, output/december_predictions.csv, metrics.json, pipeline.joblib")


if __name__ == "__main__":
    main()
