from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.compose import ColumnTransformer
from sklearn.feature_selection import VarianceThreshold
from sklearn.preprocessing import OneHotEncoder


class DropMostlyNaN(BaseEstimator, TransformerMixin):
    """Drop columns whose missing fraction exceeds threshold."""
    def __init__(self, threshold: float = 0.95):
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
            col = pd.to_numeric(X_df[c], errors="coerce")
            if np.isfinite(col).sum() == 0:
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
            X_df[c] = pd.to_numeric(X_df[c], errors="coerce").fillna(v)
        return X_df


class SafeLog10(BaseEstimator, TransformerMixin):
    """Log10-transform selected columns with per-column epsilon."""
    def __init__(self, columns: Optional[List[str]]):
        self.columns = columns
        self.columns_in_: Optional[Tuple[str, ...]] = None
        self.eps_: Optional[Dict[str, float]] = None

    def fit(self, X, y=None):
        X_df = X if isinstance(X, pd.DataFrame) else pd.DataFrame(X)
        cols_param = [] if self.columns is None else list(self.columns)
        cols = [c for c in cols_param if c in X_df.columns]
        eps: Dict[str, float] = {}
        for c in cols:
            col = pd.to_numeric(X_df[c], errors="coerce")
            min_pos = col[col > 0].min()
            if min_pos is not None and np.isfinite(min_pos):
                val = float(min_pos) / 10.0
            else:
                val = 1.0
            eps[c] = max(1e-12, val)
        self.columns_in_ = tuple(cols)
        self.eps_ = eps
        return self

    def transform(self, X):
        if self.columns_in_ is None or self.eps_ is None:
            raise RuntimeError("SafeLog10.transform called before fit.")
        X_df = X.copy()
        for c in self.columns_in_:
            X_df[c] = np.log10(pd.to_numeric(X_df[c], errors="coerce") + self.eps_[c])
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
    def __init__(self, threshold: float = 0.95):
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
