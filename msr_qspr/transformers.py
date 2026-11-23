"""Serializable dataframe transformations fitted only on training observations."""

from msr_qspr.config import MISSINGNESS_THRESHOLD, CORRELATION_THRESHOLD
from typing import Dict, List, Optional, Tuple
import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.compose import ColumnTransformer
from sklearn.feature_selection import VarianceThreshold
from sklearn.preprocessing import OneHotEncoder

class DropMostlyNaN(BaseEstimator, TransformerMixin):
    """Drop columns whose missing fraction exceeds threshold."""
    def __init__(self, threshold: float = MISSINGNESS_THRESHOLD):
        self.threshold = float(threshold)
        self.keep_cols_: Optional[Tuple[str, ...]] = None

    def fit(self, X, y=None):
        X_df = X if isinstance(X, pd.DataFrame) else pd.DataFrame(X)
        frac = X_df.isna().mean()
        keep = frac.index[frac <= self.threshold].tolist()
        if not keep:
            raise ValueError("DropMostlyNaN removed all columns.")
        self.keep_cols_ = tuple(keep)
        return self

    def transform(self, X):
        if self.keep_cols_ is None:
            raise RuntimeError("DropMostlyNaN.transform called before fit.")
        X_df = X if isinstance(X, pd.DataFrame) else pd.DataFrame(X)
        cols = [c for c in self.keep_cols_ if c in X_df.columns]
        if not cols:
            raise ValueError("DropMostlyNaN found no known columns at transform time.")
        return X_df[cols].copy()

    def get_feature_names_out(self, input_features=None):
        if self.keep_cols_ is None:
            raise RuntimeError("DropMostlyNaN.get_feature_names_out called before fit.")
        return np.asarray(self.keep_cols_, dtype=object)


class DataFrameImputer(BaseEstimator, TransformerMixin):
    """Median-impute numeric-like columns."""
    def __init__(self, fill_constant: float = 0.0):
        self.fill_constant = float(fill_constant)
        self.fill_values_: Optional[Dict[str, float]] = None

    def fit(self, X, y=None):
        X_df = X.copy()
        fill: Dict[str, float] = {}
        for c in X_df.columns:
            col = pd.to_numeric(X_df[c], errors="coerce").replace([np.inf, -np.inf], np.nan)
            if col.notna().sum() == 0:
                continue
            med = np.nanmedian(col.values)
            if not np.isfinite(med):
                med = self.fill_constant
            fill[c] = float(med)
        self.fill_values_ = fill
        return self

    def transform(self, X):
        if self.fill_values_ is None:
            raise RuntimeError("DataFrameImputer.transform called before fit.")
        X_df = X.copy()
        for c, v in self.fill_values_.items():
            X_df[c] = (
                pd.to_numeric(X_df[c], errors="coerce")
                .replace([np.inf, -np.inf], np.nan)
                .fillna(v)
            )
        return X_df


class SafeLog10(BaseEstimator, TransformerMixin):
    """Apply conventional log10 to finite, positive post-imputation values."""
    def __init__(self, columns: Optional[List[str]]):
        self.columns = columns
        self.columns_in_: Optional[Tuple[str, ...]] = None

    def fit(self, X, y=None):
        X_df = X if isinstance(X, pd.DataFrame) else pd.DataFrame(X)
        cols_param = [] if self.columns is None else list(self.columns)
        cols = [c for c in cols_param if c in X_df.columns]
        self.columns_in_ = tuple(cols)
        return self

    def transform(self, X):
        if self.columns_in_ is None:
            raise RuntimeError("SafeLog10.transform called before fit.")
        X_df = X.copy()
        for c in self.columns_in_:
            col = pd.to_numeric(X_df[c], errors="coerce")
            if not (np.isfinite(col) & (col > 0)).fillna(False).all():
                raise ValueError(
                    f"SafeLog10 requires finite, strictly positive values in column {c!r} "
                    "after median imputation."
                )
            X_df[c] = np.log10(col)
        return X_df


class DataFrameVarianceThreshold(VarianceThreshold):
    """VarianceThreshold that returns a DataFrame and keeps column names."""
    def fit(self, X, y=None):
        self._is_df = isinstance(X, pd.DataFrame)
        if self._is_df:
            self._cols_in = list(X.columns)
        return super().fit(X, y)

    def transform(self, X):
        Xt = super().transform(X)
        if getattr(self, "_is_df", False):
            keep = super().get_support()
            cols = [c for c, k in zip(self._cols_in, keep) if k]
            return pd.DataFrame(Xt, columns=cols, index=getattr(X, "index", None))
        return Xt

    def get_feature_names_out(self, input_features=None):
        keep = super().get_support()
        cols = [c for c, k in zip(self._cols_in, keep) if k]
        return np.asarray(cols, dtype=object)


class CorrelationFilter(BaseEstimator, TransformerMixin):
    """Greedy filter on absolute Pearson correlation."""
    def __init__(self, threshold: float = CORRELATION_THRESHOLD):
        self.threshold = float(threshold)
        self.keep_cols_: Optional[Tuple[str, ...]] = None

    def fit(self, X, y=None):
        X_df = X if isinstance(X, pd.DataFrame) else pd.DataFrame(X)
        Xn = X_df.apply(pd.to_numeric, errors="coerce")
        corr = Xn.corr(method="pearson").abs()
        keep: List[str] = []
        dropped: set = set()
        for c in corr.columns:
            if c in dropped:
                continue
            keep.append(c)
            high = corr.index[(corr[c] >= self.threshold) & (corr.index != c)].tolist()
            dropped.update(high)
        if not keep:
            raise ValueError("CorrelationFilter removed all columns.")
        self.keep_cols_ = tuple(keep)
        return self

    def transform(self, X):
        if self.keep_cols_ is None:
            raise RuntimeError("CorrelationFilter.transform called before fit.")
        X_df = X if isinstance(X, pd.DataFrame) else pd.DataFrame(X)
        return X_df[list(self.keep_cols_)].copy()

    def get_feature_names_out(self, input_features=None):
        if self.keep_cols_ is None:
            raise RuntimeError("CorrelationFilter.get_feature_names_out called before fit.")
        return np.asarray(self.keep_cols_, dtype=object)


class ColumnSubset(BaseEstimator, TransformerMixin):
    """Keep a fixed, ordered subset of columns by name.

    If a requested column is missing at transform time, it is created and
    filled with zeros. This makes split-wise pipelines robust to missing
    one-hot levels.
    """
    def __init__(self, columns: Optional[List[str]] = None, fill_value: float = 0.0):
        self.columns = columns
        self.fill_value = float(fill_value)
        self.columns_: Optional[Tuple[str, ...]] = None

    def fit(self, X, y=None):
        X_df = X if isinstance(X, pd.DataFrame) else pd.DataFrame(X)
        cols_param = list(X_df.columns) if self.columns is None else list(self.columns)
        self.columns_ = tuple(cols_param)
        return self

    def transform(self, X):
        if self.columns_ is None:
            raise RuntimeError("ColumnSubset.transform called before fit.")
        X_df = X if isinstance(X, pd.DataFrame) else pd.DataFrame(X)
        out = X_df.copy()
        for c in self.columns_:
            if c not in out.columns:
                from msr_qspr.config import CATEGORICAL
                if not any(c.startswith(cat + "_") for cat in CATEGORICAL):
                    raise ValueError(f"Required selected predictor is missing: {c}")
                out[c] = self.fill_value
        return out[list(self.columns_)].copy()

    def get_feature_names_out(self, input_features=None):
        cols = self.columns_ if self.columns_ is not None else (self.columns or [])
        return np.asarray(cols, dtype=object)


class DataFrameColumnTransformer(BaseEstimator, TransformerMixin):
    """Wrap ColumnTransformer and return a DataFrame."""
    def __init__(self, ct: ColumnTransformer):
        self.ct = ct
        self._feature_names_out: Optional[List[str]] = None

    def fit(self, X, y=None):
        self.ct.fit(X, y)
        try:
            names = self.ct.get_feature_names_out()
        except Exception:
            Xt = self.ct.transform(X)
            n_out = Xt.shape[1] if hasattr(Xt, "shape") else Xt.toarray().shape[1]
            names = np.array([f"f_{i}" for i in range(n_out)], dtype=object)
        self._feature_names_out = [str(n) for n in names]
        # Public fitted attribute lets sklearn validate a preprocessing-only Pipeline.
        self.feature_names_out_ = np.asarray(self._feature_names_out, dtype=object)
        return self

    def transform(self, X):
        Xt = self.ct.transform(X)
        if not isinstance(Xt, np.ndarray):
            Xt = Xt.toarray()
        Xt = Xt.astype(float, copy=False)
        return pd.DataFrame(Xt, columns=self._feature_names_out, index=getattr(X, "index", None))

    def get_feature_names_out(self, input_features=None):
        if self._feature_names_out is None:
            raise RuntimeError("DataFrameColumnTransformer.get_feature_names_out called before fit.")
        return np.asarray(self._feature_names_out, dtype=object)


def make_encoder_df(categorical_cols: List[str]) -> DataFrameColumnTransformer:
    ct = ColumnTransformer(
        transformers=[
            ("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False), categorical_cols),
        ],
        remainder="passthrough",
        verbose_feature_names_out=False,
    )
    return DataFrameColumnTransformer(ct)
