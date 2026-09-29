"""Inductive preprocessing transformers for freight-rate prediction.

Production contract
-------------------
* All transformers are **inductive**: ``fit`` learns lookup tables from the
  training split only; ``transform`` maps those frozen tables onto incoming
  rows. No ``groupby`` on the inference frame, so single-row inference
  behaves identically to batch inference.
* Missing-value handling ends in a global fallback constant and records
  per-call statistics in ``last_stats_`` for monitoring (fallback-rate
  alerts). ``transform`` never raises on missing values; schema violations
  raise in :func:`train.validate_frame` instead.
* Classes live in this module (not ``__main__``) so ``pipeline.joblib``
  deserializes with plain ``joblib.load`` in any microservice that ships
  this file.

Target convention: model fits ``log(posted_rate / distance)``; final rate
is ``exp(log_rpm_pred) * distance``. Smearing correction lives in
``modeling.DollarHGBRegressor``.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin

EQUIPMENT_CODE = {"Dry Van": 0, "Reefer": 1, "Flatbed": 2}

# Numeric model inputs (kept stable for backward-compat with tests/metrics).
NUM_FEATURES = [
    "pickup_lat", "pickup_lon", "delivery_lat", "delivery_lon",
    "distance", "circuity",
    "weight", "weight_per_mile", "equipment_code", "lane_freq", "market_index",
    "days_to_quarter_end", "dow_sin", "dow_cos", "doy_sin", "doy_cos",
    "equip_x_logdist", "is_weekend",
]
# Native HGB categoricals (pandas ``category`` dtype, ``from_dtype`` mode).
CAT_FEATURES = ["pickup_cat", "delivery_cat", "equipment_cat"]
MODEL_FEATURES = NUM_FEATURES + CAT_FEATURES

# Backward-compat alias (old code/tests import FEATURES).
FEATURES = NUM_FEATURES


def haversine(lat1, lon1, lat2, lon2):
    """Great-circle miles; inputs are per-city constants (proxy geography)."""
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


class WeightDistanceImputer(BaseEstimator, TransformerMixin):
    """Fill weight (equipment median -> global) and distance (train median).

    Inductive: statistics frozen at ``fit``; ``transform`` never looks at
    other rows of the inference batch. Tracks ``last_stats_`` with
    ``weight_fallback_rate`` / ``distance_fallback_rate`` for monitoring.
    """

    def fit(self, X: pd.DataFrame, y=None):
        X = X.copy()
        if "equipment" in X and "weight" in X:
            self.eq_weight_ = X.groupby("equipment")["weight"].median()
        else:
            self.eq_weight_ = pd.Series(dtype=float)
        self.global_weight_ = float(X["weight"].median()) if "weight" in X else 30000.0
        self.distance_median_ = float(X["distance"].median()) if "distance" in X else 1000.0
        self.last_stats_ = {"n": int(len(X)), "n_weight_imputed": 0, "n_distance_imputed": 0}
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        X = X.copy()
        n = len(X)
        n_w = n_d = 0
        if "weight" in X:
            # Treat sign-flips defensively; canonical fix lives in basic_clean.
            X["weight"] = X["weight"].astype(float)
            m = X["weight"].isna()
            if m.any():
                fill = X.loc[m, "equipment"].map(self.eq_weight_) if "equipment" in X else np.nan
                X.loc[m, "weight"] = pd.Series(fill).fillna(self.global_weight_).to_numpy()
                n_w = int(m.sum())
        if "distance" in X:
            X["distance"] = X["distance"].astype(float)
            bad = X["distance"].isna() | (X["distance"] <= 0)
            if bad.any():
                X.loc[bad, "distance"] = self.distance_median_
                n_d = int(bad.sum())
        self.last_stats_ = {
            "n": int(n),
            "n_weight_imputed": n_w,
            "n_distance_imputed": n_d,
            "weight_fallback_rate": float(n_w / max(n, 1)),
            "distance_fallback_rate": float(n_d / max(n, 1)),
        }
        return X


class MarketImputer(BaseEstimator, TransformerMixin):
    """Inductive market_index repair with a monitored fallback ladder.

    Fit (training only):
      * ``date_map_``: exact-date median (useful for in-sample CV).
      * ``month_map_``: ``YYYY-MM`` median — primary forward signal.
      * ``dow_map_``: day-of-week median — weekly seasonality backstop.
      * ``global_``: overall median — terminal fallback.

    Transform (any batch size incl. single row):
      missing -> date hit -> month hit -> dow hit -> global.
      No ``groupby`` on the inference frame. Unseen Nov/Dec dates
      therefore fall through to month/dow/global deterministically.

    ``last_stats_`` records hits per level and ``global_fallback_rate``
    (alert if > 2%).
    """

    def fit(self, X: pd.DataFrame, y=None):
        df = X.copy()
        df["date"] = pd.to_datetime(df["date"], errors="coerce")
        mi = df["market_index"] if "market_index" in df else pd.Series(dtype=float)
        self.date_map_ = df.groupby("date")["market_index"].median().to_dict() if len(df) else {}
        month_key = df["date"].dt.strftime("%Y-%m")
        self.month_map_ = df.groupby(month_key)["market_index"].median().to_dict() if len(df) else {}
        self.dow_map_ = df.groupby(df["date"].dt.dayofweek)["market_index"].median().to_dict() if len(df) else {}
        self.global_ = float(mi.median()) if mi.notna().any() else 1.0
        self.last_stats_ = {"n": int(len(df)), "n_missing": 0, "global_fallback_rate": 0.0}
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        X = X.copy()
        if "market_index" not in X:
            return X
        X["date"] = pd.to_datetime(X["date"], errors="coerce")
        n = len(X)
        missing_mask = X["market_index"].isna()
        n_missing = int(missing_mask.sum())
        n_date = n_month = n_dow = n_global = 0
        if n_missing:
            idx = X[missing_mask].index
            # Level 1: exact training date.
            v = X.loc[idx, "date"].map(self.date_map_)
            hit = v.notna()
            X.loc[idx[hit], "market_index"] = v[hit].to_numpy()
            n_date = int(hit.sum())
            # Level 2: training month.
            still = X["market_index"].isna()
            if still.any():
                sidx = X[still].index
                v = X.loc[sidx, "date"].dt.strftime("%Y-%m").map(self.month_map_)
                hit = v.notna()
                X.loc[sidx[hit], "market_index"] = v[hit].to_numpy()
                n_month = int(hit.sum())
            # Level 3: day-of-week seasonality.
            still = X["market_index"].isna()
            if still.any():
                sidx = X[still].index
                v = X.loc[sidx, "date"].dt.dayofweek.map(self.dow_map_)
                hit = v.notna()
                X.loc[sidx[hit], "market_index"] = v[hit].to_numpy()
                n_dow = int(hit.sum())
            # Level 4: global terminal fallback.
            still = X["market_index"].isna()
            if still.any():
                X.loc[X[still].index, "market_index"] = self.global_
                n_global = int(still.sum())
        X["market_index"] = X["market_index"].fillna(self.global_).astype(float)
        self.last_stats_ = {
            "n": int(n),
            "n_missing": n_missing,
            "n_date_hit": n_date,
            "n_month_hit": n_month,
            "n_dow_hit": n_dow,
            "n_global_fallback": n_global,
            "global_fallback_rate": float(n_global / max(n, 1)),
            "missing_rate": float(n_missing / max(n, 1)),
        }
        return X


class FeatureEngineer(BaseEstimator, TransformerMixin):
    """Geo + date + interaction features. Fit stores lane freq + cat vocab.

    * ``lane_freq_``: train-only ``pickup__delivery`` counts; unseen lanes
      map to 0 (explicit drift signal, monitored at >15%).
    * ``*_cats_``: fixed category vocabularies; unseen cities/equipment
      become ``NaN`` codes at transform so HGB (``from_dtype``) routes them
      to its missing bin instead of an arbitrary leaf.
    * Never drops rows; outlier logic lives outside inference path.
    """

    def fit(self, X: pd.DataFrame, y=None):
        if "pickup" in X and "delivery" in X:
            self.lane_freq_ = (X["pickup"].astype(str) + "__" + X["delivery"].astype(str)).value_counts()
        else:
            self.lane_freq_ = pd.Series(dtype=float)
        self.pickup_cats_ = sorted(X["pickup"].dropna().unique().tolist()) if "pickup" in X else []
        self.delivery_cats_ = sorted(X["delivery"].unique().tolist()) if "delivery" in X else []
        self.equipment_cats_ = sorted(X["equipment"].dropna().unique().tolist()) if "equipment" in X else []
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        X = X.copy()
        geo = haversine(X["pickup_lat"], X["pickup_lon"], X["delivery_lat"], X["delivery_lon"])
        ratio = X["distance"] / np.maximum(geo, 1e-6)
        X["circuity"] = ratio.clip(1.0, 2.0).fillna(1.18)
        X["distance_ratio"] = X["circuity"]  # compat alias
        X["geo_distance"] = geo
        X["weight_per_mile"] = X["weight"] / np.maximum(X["distance"], 1e-6)
        X["equipment_code"] = X["equipment"].map(EQUIPMENT_CODE).fillna(-1).astype(int) \
            if "equipment" in X else -1
        if "pickup" in X and "delivery" in X:
            X["lane_freq"] = (X["pickup"].astype(str) + "__" + X["delivery"].astype(str)) \
                .map(self.lane_freq_).fillna(0)
        else:
            X["lane_freq"] = 0
        X["days_to_quarter_end"] = days_to_quarter_end(pd.to_datetime(X["date"]))
        dow = pd.to_datetime(X["date"]).dt.dayofweek
        doy = pd.to_datetime(X["date"]).dt.dayofyear
        X["dow_sin"] = np.sin(2 * np.pi * dow / 7)
        X["dow_cos"] = np.cos(2 * np.pi * dow / 7)
        X["doy_sin"] = np.sin(2 * np.pi * doy / 365.25)
        X["doy_cos"] = np.cos(2 * np.pi * doy / 365.25)
        X["equip_x_logdist"] = X["equipment_code"] * np.log(np.maximum(X["distance"], 1e-6))
        X["day_of_week"] = dow  # compat; excluded from model
        X["day_of_month"] = pd.to_datetime(X["date"]).dt.day  # compat; excluded
        X["is_weekend"] = (dow >= 5).astype(int)
        # Native HGB categoricals: fixed vocab -> unseen becomes NaN.
        if "pickup" in X:
            X["pickup_cat"] = pd.Categorical(X["pickup"], categories=self.pickup_cats_)
        else:
            X["pickup_cat"] = pd.Categorical([np.nan] * len(X))
        if "delivery" in X:
            X["delivery_cat"] = pd.Categorical(X["delivery"], categories=self.delivery_cats_)
        else:
            X["delivery_cat"] = pd.Categorical([np.nan] * len(X))
        if "equipment" in X:
            X["equipment_cat"] = pd.Categorical(X["equipment"], categories=self.equipment_cats_)
        else:
            X["equipment_cat"] = pd.Categorical([np.nan] * len(X))
        return X


class ColumnSelector(BaseEstimator, TransformerMixin):
    """Select model columns so strings/dates never reach the estimator."""

    def __init__(self, columns=None):
        self.columns = columns

    def fit(self, X, y=None):
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        return X[list(self.columns)]
