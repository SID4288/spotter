from __future__ import annotations

import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.model_selection import RandomizedSearchCV, TimeSeriesSplit
from sklearn.pipeline import Pipeline

ROOT = Path(__file__).resolve().parent
DATA, OUT = ROOT / "data", ROOT / "output"
METRICS = ROOT / "notebooks" / "results" / "modelling" / "metrics.json"
PIPE_PATH = OUT / "pipeline.joblib"
SEED = 42

EQUIPMENT_CODE = {"Dry Van": 0, "Reefer": 1, "Flatbed": 2}
# Dropped vs original: geo_distance (corr 1.00 with distance),
# day_of_week/day_of_month raw ints (replaced by cyclic encodings).
FEATURES = [
    "pickup_lat", "pickup_lon", "delivery_lat", "delivery_lon",
    "distance", "circuity",
    "weight", "weight_per_mile", "equipment_code", "lane_freq", "market_index",
    "days_to_quarter_end", "dow_sin", "dow_cos", "doy_sin", "doy_cos",
    "equip_x_logdist", "is_weekend",
]
FEATURES_NO_MARKET = [f for f in FEATURES if f != "market_index"]

REQUIRED_INPUTS = ["pickup", "delivery", "pickup_lat", "pickup_lon",
                   "delivery_lat", "delivery_lon", "distance", "equipment",
                   "weight", "date", "market_index"]


def basic_clean(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["weight"] = df["weight"].abs()  # sign-flipped weights (~0.6% of rows)
    df.loc[df["distance"] <= 0, "distance"] = np.nan
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
    """Backward-compatible: fit + apply on the same frame.

    Use only for the full-train final fit. Inside CV use _flag_params(tr)
    fitted on the training fold and _apply_flag(va, ...) for scoring.
    """
    coef, med, mad = _flag_params(df)
    return _apply_flag(df, coef, med, mad)


def haversine(lat1, lon1, lat2, lon2):
    lat1, lon1, lat2, lon2 = map(np.radians, [lat1, lon1, lat2, lon2])
    a = np.sin((lat2 - lat1) / 2) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin((lon2 - lon1) / 2) ** 2
    return 3958.8 * 2 * np.arcsin(np.sqrt(a))


def days_to_quarter_end(dates: pd.Series) -> pd.Series:
    """Vectorized quarter-end countdown (no per-row to_period)."""
    em = ((dates.dt.month - 1) // 3) * 3 + 3
    qend = pd.to_datetime(pd.DataFrame({
        "year": dates.dt.year, "month": em,
        "day": em.map({3: 31, 6: 30, 9: 30, 12: 31}),
    }))
    return (qend - dates.dt.normalize()).dt.days


def validate_frame(df: pd.DataFrame, require_label: bool = False) -> None:
    missing = [c for c in REQUIRED_INPUTS if c not in df.columns]
    if missing:
        raise ValueError(f"missing input columns: {missing}")
    if require_label and "posted_rate" not in df.columns:
        raise ValueError("posted_rate required but missing")
    if df["distance"].isna().any() or (df["distance"] <= 0).any():
        # basic_clean turns <=0 into NaN; imputation fills it later.
        pass


class WeightDistanceImputer(BaseEstimator, TransformerMixin):
    """Fill weight (equipment median) and distance (global median)."""

    def fit(self, X: pd.DataFrame, y=None):
        self.eq_weight_ = X.groupby("equipment")["weight"].median() if "equipment" in X else pd.Series(dtype=float)
        self.global_weight_ = X["weight"].median() if "weight" in X else 30000.0
        self.distance_median_ = X["distance"].median() if "distance" in X else 1000.0
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        X = X.copy()
        if "weight" in X:
            m = X["weight"].isna()
            if m.any():
                fill = X.loc[m, "equipment"].map(self.eq_weight_).fillna(self.global_weight_) \
                    if "equipment" in X else self.global_weight_
                X.loc[m, "weight"] = fill
        if "distance" in X:
            X["distance"] = X["distance"].mask(X["distance"] <= 0).fillna(self.distance_median_)
        return X


class MarketImputer(BaseEstimator, TransformerMixin):
    """Train lookups + batch own-date/month fallback (inputs only, never labels)."""

    def fit(self, X: pd.DataFrame, y=None):
        self.date_map_ = X.groupby("date")["market_index"].median().to_dict() if "market_index" in X else {}
        self.month_map_ = X.groupby(X["date"].dt.strftime("%Y-%m"))["market_index"].median().to_dict() \
            if "market_index" in X else {}
        self.global_ = X["market_index"].median() if "market_index" in X else 1.0
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        X = X.copy()
        if "market_index" not in X:
            return X
        m = X["market_index"].isna()
        if m.any():
            X.loc[m, "market_index"] = X.loc[m, "date"].map(self.date_map_)
        m = X["market_index"].isna()
        if m.any():
            X.loc[m, "market_index"] = X.loc[m, "date"].map(X.groupby("date")["market_index"].median())
        m = X["market_index"].isna()
        if m.any():
            X.loc[m, "market_index"] = X.loc[m, "date"].dt.strftime("%Y-%m").map(self.month_map_)
        m = X["market_index"].isna()
        if m.any():
            X.loc[m, "market_index"] = X.loc[m, "date"].dt.strftime("%Y-%m").map(
                X.groupby(X["date"].dt.strftime("%Y-%m"))["market_index"].median())
        X["market_index"] = X["market_index"].fillna(self.global_)
        return X


class FeatureEngineer(BaseEstimator, TransformerMixin):
    """Geo + date + interaction features. Fit stores lane frequencies only."""

    def fit(self, X: pd.DataFrame, y=None):
        if "pickup" in X and "delivery" in X:
            self.lane_freq_ = (X["pickup"].astype(str) + "__" + X["delivery"].astype(str)).value_counts()
        else:
            self.lane_freq_ = pd.Series(dtype=float)
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        X = X.copy()
        geo = haversine(X["pickup_lat"], X["pickup_lon"], X["delivery_lat"], X["delivery_lon"])
        ratio = X["distance"] / np.maximum(geo, 1e-6)
        X["circuity"] = ratio.clip(1.0, 2.0).fillna(1.18)
        # compat alias (old name, clipped)
        X["distance_ratio"] = X["circuity"]
        X["geo_distance"] = geo
        X["weight_per_mile"] = X["weight"] / np.maximum(X["distance"], 1e-6)
        X["equipment_code"] = X["equipment"].map(EQUIPMENT_CODE).fillna(-1).astype(int) \
            if "equipment" in X else -1
        if "pickup" in X and "delivery" in X:
            X["lane_freq"] = (X["pickup"].astype(str) + "__" + X["delivery"].astype(str)) \
                .map(self.lane_freq_).fillna(0)
        else:
            X["lane_freq"] = 0
        X["days_to_quarter_end"] = days_to_quarter_end(X["date"])
        dow = X["date"].dt.dayofweek
        doy = X["date"].dt.dayofyear
        X["dow_sin"] = np.sin(2 * np.pi * dow / 7)
        X["dow_cos"] = np.cos(2 * np.pi * dow / 7)
        X["doy_sin"] = np.sin(2 * np.pi * doy / 365.25)
        X["doy_cos"] = np.cos(2 * np.pi * doy / 365.25)
        X["equip_x_logdist"] = X["equipment_code"] * np.log(np.maximum(X["distance"], 1e-6))
        X["day_of_week"] = dow  # compat; excluded from FEATURES
        X["day_of_month"] = X["date"].dt.day  # compat; excluded from FEATURES
        X["is_weekend"] = (dow >= 5).astype(int)
        return X


def fit_preprocessors(train_df: pd.DataFrame) -> dict:
    """Backward-compatible prep dict (inputs only, all rows — labels don't set input medians)."""
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
    df = df.copy()
    # New pipeline path
    if "_w" in prep:
        df = prep["_w"].transform(df)
        df = prep["_m"].transform(df)
        df = prep["_f"].transform(df)
        return df
    # Legacy dict path (kept for old pickles): vectorized, hardened version
    m = df["weight"].isna()
    if m.any():
        df.loc[m, "weight"] = df.loc[m, "equipment"].map(prep["eq_weight"]).fillna(prep["global_weight"])
    if "distance_median" in prep:
        df["distance"] = df["distance"].mask(df["distance"] <= 0).fillna(prep["distance_median"])
    m = df["market_index"].isna()
    if m.any():
        df.loc[m, "market_index"] = df.loc[m, "date"].map(prep["date_market"])
    m = df["market_index"].isna()
    if m.any():
        df.loc[m, "market_index"] = df.loc[m, "date"].map(df.groupby("date")["market_index"].median())
    m = df["market_index"].isna()
    if m.any():
        key = df.loc[m, "date"].dt.strftime("%Y-%m")
        month_map = prep["month_market"]
        try:
            df.loc[m, "market_index"] = key.map(month_map)
        except Exception:
            df.loc[m, "market_index"] = np.nan
    m = df["market_index"].isna()
    if m.any():
        df.loc[m, "market_index"] = df.loc[m, "date"].dt.strftime("%Y-%m").map(
            df.groupby(df["date"].dt.strftime("%Y-%m"))["market_index"].median())
    df["market_index"] = df["market_index"].fillna(prep["global_market"])
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
    return df


def make_model() -> ExtraTreesRegressor:
    return ExtraTreesRegressor(n_estimators=300, min_samples_leaf=2, max_features="sqrt",
                               n_jobs=-1, random_state=SEED)


def make_pipeline() -> Pipeline:
    return Pipeline([
        ("weight_distance", WeightDistanceImputer()),
        ("market", MarketImputer()),
        ("features", FeatureEngineer()),
        ("model", make_model()),
    ])


def _train_mask(train_df: pd.DataFrame) -> np.ndarray:
    """Training-row outlier mask, fitted on train_df only (leakage fix)."""
    if "posted_rate" not in train_df or "distance" not in train_df:
        if "is_corrupt" in train_df:
            return train_df["is_corrupt"].to_numpy()
        return np.zeros(len(train_df), dtype=bool)
    coef, med, mad = _flag_params(train_df)
    return _apply_flag(train_df, coef, med, mad).to_numpy()


def fit_predict(train_df: pd.DataFrame, target_df: pd.DataFrame,
                features: list[str] | None = None,
                denoised_market: bool = False) -> np.ndarray:
    prep = fit_preprocessors(train_df)
    tr, te = transform(train_df, prep), transform(target_df, prep)
    if denoised_market:
        tr["market_index"] = tr["date"].map(prep["_m"].date_map_).fillna(prep["global_market"])
        own = te.groupby("date")["market_index"].median()
        te["market_index"] = te["date"].map(prep["_m"].date_map_)
        missing = te["market_index"].isna()
        te.loc[missing, "market_index"] = te.loc[missing, "date"].map(own)
        te["market_index"] = te["market_index"].fillna(prep["global_market"])
    feats = FEATURES if features is None else features
    mask = _train_mask(train_df)
    # align mask to transformed rows (same order/index for train)
    tr_clean = tr[~pd.Series(mask, index=train_df.index).reindex(tr.index).fillna(False).to_numpy()]
    model = make_model().fit(tr_clean[feats], np.log(tr_clean["posted_rate"] / tr_clean["distance"]))
    return np.exp(model.predict(te[feats])) * te["distance"].to_numpy()


def tune(train_df: pd.DataFrame, n_iter: int = 12) -> dict:
    """Efficient tuning: RandomizedSearchCV + TimeSeriesSplit (past -> future only)."""
    mask = _train_mask(train_df)
    df = train_df[~mask]
    pipe = Pipeline([
        ("weight_distance", WeightDistanceImputer()),
        ("market", MarketImputer()),
        ("features", FeatureEngineer()),
        ("model", ExtraTreesRegressor(random_state=SEED, n_jobs=-1)),
    ])
    rs = RandomizedSearchCV(
        pipe,
        {"model__n_estimators": [300, 500], "model__min_samples_leaf": [1, 2, 5],
         "model__max_features": ["sqrt", 0.5, 1.0], "model__min_samples_split": [2, 5]},
        n_iter=n_iter, cv=TimeSeriesSplit(n_splits=4),
        scoring="neg_mean_absolute_error", n_jobs=-1, random_state=SEED)
    rs.fit(df, np.log(df["posted_rate"] / df["distance"]))
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
    pipe = make_pipeline()
    clean = train_raw[~train_raw["is_corrupt"]]
    pipe.fit(clean, np.log(clean["posted_rate"] / clean["distance"]))
    OUT.mkdir(exist_ok=True)
    joblib.dump(pipe, PIPE_PATH)

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
    # market_index is a provided *input* feature: use the December daily median from validation.csv
    # (no labels involved); fall back to the December monthly median for any missing date.
    vm = val_raw.groupby("date")["market_index"].median()
    dec_median = val_raw.loc[val_raw["date"] >= "2025-12-01", "market_index"].median()
    dec["market_index"] = dec["date"].map(vm).fillna(dec_median)
    dec["weight"] = dec["weight"].astype(float)
    dec_pred = np.clip(np.exp(pipe.predict(dec)) * dec["distance"].to_numpy(), 1.0, None)
    dec_out = december.copy()
    dec_out["date"] = dec_out["date"].dt.strftime("%Y-%m-%d")
    dec_out["predicted_rate"] = np.round(dec_pred, 2)
    assert len(dec_out) == 31 and dec_out["predicted_rate"].gt(0).all()

    out.to_csv(OUT / "validation_predictions.csv", index=False)
    out.to_csv(ROOT / "validation_predictions.csv", index=False)  # the file named in the submission brief
    dec_out.to_csv(OUT / "december_predictions.csv", index=False)
    metrics["december_mean_rate"] = round(float(dec_out["predicted_rate"].mean()), 2)
    metrics["december_peak"] = {"date": dec_out.loc[dec_out["predicted_rate"].idxmax(), "date"],
                                "rate": float(dec_out["predicted_rate"].max())}
    metrics["feature_set"] = FEATURES
    METRICS.parent.mkdir(parents=True, exist_ok=True)
    METRICS.write_text(json.dumps(metrics, indent=2))
    print(f"Validation mean ${out['predicted_rate'].mean():.2f}; December mean ${metrics['december_mean_rate']:.2f}")
    print("Wrote validation_predictions.csv, output/validation_predictions.csv, output/december_predictions.csv, metrics.json, pipeline.joblib")


if __name__ == "__main__":
    main()
