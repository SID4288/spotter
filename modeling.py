"""Dollar-aware HGB wrapper with Duan smearing correction.

Why
---
* Business impact is in dollars, but fitting ``log(rate/mile)`` is
  statistically convenient (stabilizes long-haul variance). Naive
  ``exp(pred) * distance`` is systematically low because
  ``E[exp(Y)] > exp(E[Y])``.
* Duan (1983) smearing: ``E[rate] = distance * exp(mu) * E[exp(resid)]``.
  We estimate ``S = mean(exp(resid_train))`` (sample-weighted when
  ``sample_weight`` is given) and return ``mu + log(S)`` from ``predict``,
  so downstream code ``exp(pred) * distance`` is unbiased without changes.
* ``sample_weight=distance`` support ensures long-haul dollar variance is
  penalized during HGB training (optimization in dollar-space).
* Native categoricals: inner HGB uses ``categorical_features="from_dtype"``,
  so ``pickup_cat/delivery_cat/equipment_cat`` (pandas ``category`` dtype)
  get native splits; unseen levels (NaN) go to the missing bin plus the
  explicit ``lane_freq==0`` signal.

Size: HGB (~300 trees x <=63 leaves) serializes to ~5-25MB vs 800MB ET.
"""

from __future__ import annotations

import numpy as np
from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.ensemble import HistGradientBoostingRegressor


class DollarHGBRegressor(BaseEstimator, RegressorMixin):
    """HGB on log-RPM with distance weighting + smearing correction."""

    def __init__(
        self,
        max_iter: int = 300,
        max_leaf_nodes: int = 63,
        learning_rate: float = 0.05,
        min_samples_leaf: int = 20,
        l2_regularization: float = 1.0,
        categorical_features: str = "from_dtype",
        early_stopping: bool = False,
        random_state: int = 42,
    ):
        self.max_iter = max_iter
        self.max_leaf_nodes = max_leaf_nodes
        self.learning_rate = learning_rate
        self.min_samples_leaf = min_samples_leaf
        self.l2_regularization = l2_regularization
        self.categorical_features = categorical_features
        self.early_stopping = early_stopping
        self.random_state = random_state

    def _make_inner(self) -> HistGradientBoostingRegressor:
        return HistGradientBoostingRegressor(
            max_iter=self.max_iter,
            max_leaf_nodes=self.max_leaf_nodes,
            learning_rate=self.learning_rate,
            min_samples_leaf=self.min_samples_leaf,
            l2_regularization=self.l2_regularization,
            categorical_features=self.categorical_features,
            early_stopping=self.early_stopping,
            random_state=self.random_state,
        )

    def fit(self, X, y, sample_weight=None):
        y = np.asarray(y, dtype=float)
        self.inner_ = self._make_inner()
        self.inner_.fit(X, y, sample_weight=sample_weight)
        resid = y - self.inner_.predict(X)
        # Guard against corrupt-label residuals leaking into S: clip exp
        # residuals to [0.2, 5.0] (matches observed 0.2x / 2-5x modes) so a
        # few bad rows cannot explode the global multiplier. Training rows
        # passed here should already be de-corrupted via _train_mask.
        exp_resid = np.clip(np.exp(resid), 0.2, 5.0)
        if sample_weight is not None:
            w = np.asarray(sample_weight, dtype=float)
            self.smearing_ = float(np.average(exp_resid, weights=w))
        else:
            self.smearing_ = float(np.mean(exp_resid))
        # Clamp to sane band; log(1.0)=0 means no correction.
        self.smearing_ = float(np.clip(self.smearing_, 0.9, 1.5))
        self.log_smearing_ = float(np.log(self.smearing_))
        return self

    def predict(self, X):
        return self.inner_.predict(X) + self.log_smearing_

    # sklearn compat: expose inner attrs used by tests/deploy.
    @property
    def n_features_in_(self):
        return self.inner_.n_features_in_
