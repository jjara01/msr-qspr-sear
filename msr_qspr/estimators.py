"""Physically anchored regressors and Ridge initialization for gradient boosting.

The solubility anchor fits z = log10(MSR) - beta * s, with
s = log10(WaterSolubility_g_L / Cont_MW_g_mol) the log10 molar water solubility,
and predicts z_hat + beta * s. Inputs arrive after training-fitted median
imputation and the log10 transform of WaterSolubility; Cont_MW remains in g/mol.
"""

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, RegressorMixin, clone
from sklearn.linear_model import Ridge
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.utils.validation import check_is_fitted

SOLUBILITY_FEATURE = 'WaterSolubility'
MOLECULAR_WEIGHT_FEATURE = 'Cont_MW'


def standardized_ridge(alpha=1.0):
    """Ridge on training-standardized inputs, usable as an estimator or boosting initialization."""
    return Pipeline([('scaler', StandardScaler()), ('ridge', Ridge(alpha=alpha))])


class SolubilityAnchoredRegressor(RegressorMixin, BaseEstimator):
    """Learn log10(MSR) relative to an imposed multiple of log10 molar water solubility.

    The anchor columns are located by name when fitting on the ordered predictor
    frame and by stored position afterwards, so fitted-space arrays predict identically.
    """

    def __init__(self, estimator=None, anchor_coefficient=0.0):
        self.estimator = estimator
        self.anchor_coefficient = anchor_coefficient

    def _offset(self, values):
        solubility = values[:, self.solubility_index_]
        molecular_weight = values[:, self.molecular_weight_index_]
        if not np.isfinite(solubility).all() or not (molecular_weight > 0).all():
            raise ValueError('The solubility anchor requires finite log10 solubility and positive molecular weight.')
        return self.anchor_coefficient * (solubility - np.log10(molecular_weight))

    def _values(self, X):
        if isinstance(X, pd.DataFrame):
            if list(X.columns) != self.feature_names_in_.tolist():
                raise ValueError('Anchored predictors differ from the fitted feature order.')
        values = np.asarray(X, dtype=float)
        if values.ndim != 2 or values.shape[1] != self.n_features_in_:
            raise ValueError('Anchored predictors have an unexpected shape.')
        return values

    def fit(self, X, y):
        if self.estimator is None or self.anchor_coefficient not in (0.0, 0.5, 1.0):
            raise ValueError('An inner estimator and a declared anchor coefficient are required.')
        if not isinstance(X, pd.DataFrame):
            raise ValueError('The solubility anchor must be fitted on named predictors.')
        names = list(X.columns)
        if SOLUBILITY_FEATURE not in names or MOLECULAR_WEIGHT_FEATURE not in names:
            raise ValueError('The solubility anchor requires WaterSolubility and Cont_MW inputs.')
        self.feature_names_in_ = np.asarray(names, dtype=object)
        self.n_features_in_ = len(names)
        self.solubility_index_ = names.index(SOLUBILITY_FEATURE)
        self.molecular_weight_index_ = names.index(MOLECULAR_WEIGHT_FEATURE)
        values = self._values(X)
        target = np.asarray(y, dtype=float) - self._offset(values)
        self.estimator_ = clone(self.estimator).fit(values, target)
        return self

    def anchor_offset(self, X):
        """Return the imposed offset beta * log10 molar solubility for each row."""
        check_is_fitted(self, 'estimator_')
        return self._offset(self._values(X))

    def predict(self, X):
        check_is_fitted(self, 'estimator_')
        values = self._values(X)
        return self.estimator_.predict(values) + self._offset(values)
