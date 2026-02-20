from typing import List, Tuple, Dict, Any

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
import matplotlib.pyplot as plt
import seaborn as sns

from config import SEED, TEST_SIZE, N_BINS, DATA_FIGURES_DIR
from plot_style import set_plot_style, get_discrete_colors
from transformers import (
    DropMostlyNaN,
    DataFrameImputer,
    SafeLog10,
    DataFrameVarianceThreshold,
    CorrelationFilter,
    make_encoder_df,
)


TARGET_COL = "MSR"

# Columns kept as metadata, not used as features
META_COLS: List[str] = [
    "SampleID",
    "Contaminant",
    "Surfactant",
    "SMILES_contaminant",
    "SMILES_surfactant",
]

# For special handling in preprocessing
CATEGORICAL: List[str] = [
    "ContaminantType",
    "SurfactantType",
]

LOG10: List[str] = [
    "CMC",
    "WaterSolubility",
]


def build_design_matrices(
    df_full: pd.DataFrame,
) -> Tuple[pd.DataFrame, pd.Series, pd.DataFrame]:
    """Split full dataframe into X (features), y (target), and metadata."""
    if TARGET_COL not in df_full.columns:
        raise KeyError(f"Target column '{TARGET_COL}' not found in dataframe.")

    meta_cols = [c for c in META_COLS if c in df_full.columns]

    y = df_full[TARGET_COL].copy()
    meta = df_full[meta_cols].copy()

    exclude = set(meta_cols + [TARGET_COL])
    feature_cols = [c for c in df_full.columns if c not in exclude]
    X = df_full[feature_cols].copy()

    return X, y, meta


def _save_target_ecdf_cv_holdout(
    y_cv_log: pd.Series,
    y_holdout_log: pd.Series,
    filename: str = "target_ecdf_cv_holdout.pdf",
) -> None:
    """Save ECDF plot comparing CV vs holdout target distributions."""
    set_plot_style()
    DATA_FIGURES_DIR.mkdir(parents=True, exist_ok=True)

    y_cv_arr = np.sort(y_cv_log.to_numpy(dtype=float))
    y_ho_arr = np.sort(y_holdout_log.to_numpy(dtype=float))

    n_cv = y_cv_arr.size
    n_ho = y_ho_arr.size

    ecdf_cv = np.arange(1, n_cv + 1) / float(n_cv)
    ecdf_ho = np.arange(1, n_ho + 1) / float(n_ho)

    palette = get_discrete_colors(2)
    c_cv = palette[0]
    c_ho = palette[1]

    fig, ax = plt.subplots(figsize=(4.8, 4.0))

    ax.step(
        y_cv_arr,
        ecdf_cv,
        where="post",
        label="CV (90%)",
        color=c_cv,
        linewidth=1.4,
    )
    ax.step(
        y_ho_arr,
        ecdf_ho,
        where="post",
        label="Holdout (10%)",
        color=c_ho,
        linestyle="--",
        linewidth=1.4,
    )

    ax.set_xlabel("log10(MSR)")
    ax.set_ylabel("ECDF")
    ax.legend()

    fig.tight_layout()
    fig.savefig(DATA_FIGURES_DIR / filename)
    plt.close(fig)


def create_cv_holdout_split(
    df_full: pd.DataFrame,
    test_size: float = TEST_SIZE,
    n_bins: int = N_BINS,
) -> Dict[str, Any]:
    """Create 90/10 CV / holdout split with stratification on MSR bins."""
    X, y, meta = build_design_matrices(df_full)

    bins, bin_edges = pd.qcut(
        y,
        q=n_bins,
        labels=False,
        duplicates="drop",
        retbins=True,
    )

    X_cv, X_holdout, y_cv, y_holdout, bins_cv, bins_holdout, meta_cv, meta_holdout = train_test_split(
        X,
        y,
        bins,
        meta,
        test_size=test_size,
        random_state=SEED,
        stratify=bins,
    )

    X_cv = X_cv.reset_index(drop=True)
    X_holdout = X_holdout.reset_index(drop=True)
    y_cv = y_cv.reset_index(drop=True)
    y_holdout = y_holdout.reset_index(drop=True)
    bins_cv = pd.Series(bins_cv, name="bin").reset_index(drop=True)
    bins_holdout = pd.Series(bins_holdout, name="bin").reset_index(drop=True)
    meta_cv = meta_cv.reset_index(drop=True)
    meta_holdout = meta_holdout.reset_index(drop=True)

    y_cv_log = pd.Series(
        np.log10(y_cv.to_numpy(dtype=float)),
        name="log10_MSR_cv",
    )
    y_holdout_log = pd.Series(
        np.log10(y_holdout.to_numpy(dtype=float)),
        name="log10_MSR_holdout",
    )

    _save_target_ecdf_cv_holdout(y_cv_log, y_holdout_log)

    return {
        "X_cv": X_cv,
        "X_holdout": X_holdout,
        "y_cv": y_cv,
        "y_holdout": y_holdout,
        "y_cv_log": y_cv_log,
        "y_holdout_log": y_holdout_log,
        "bins_cv": bins_cv,
        "bins_holdout": bins_holdout,
        "meta_cv": meta_cv,
        "meta_holdout": meta_holdout,
        "bin_edges": bin_edges,
    }


def infer_feature_roles(X: pd.DataFrame) -> Tuple[List[str], List[str]]:
    """Infer categorical and to log10 columns from X."""
    categorical_cols = [c for c in CATEGORICAL if c in X.columns]
    log10_cols = [c for c in LOG10 if c in X.columns]
    return categorical_cols, log10_cols


def build_preprocessor(X: pd.DataFrame) -> Tuple[Pipeline, List[str], List[str]]:
    """
    Build the preprocessing pipeline for features.

    Returns the fitted roles (categorical/log10) together with the pipeline.
    The pipeline:
    - drops columns with too many NaNs
    - imputes numeric-like columns
    - applies log10 to selected columns
    - one-hot encodes categoricals and passes the rest
    - removes zero-variance features
    - filters highly correlated features
    """
    categorical_cols, log10_cols = infer_feature_roles(X)

    encoder = make_encoder_df(categorical_cols)

    preprocessor = Pipeline(
        steps=[
            ("drop_sparse", DropMostlyNaN(threshold=0.95)),
            ("imputer", DataFrameImputer()),
            ("log10", SafeLog10(columns=log10_cols)),
            ("ohe", encoder),
            ("var", DataFrameVarianceThreshold(threshold=0.0)),
            ("corr", CorrelationFilter(threshold=0.95)),
        ]
    )

    return preprocessor, categorical_cols, log10_cols
