"""
app_core.py — MSR prediction engine for the Streamlit app.

Loads all artifacts once at startup and exposes MSRPredictor.predict().
Replicates the exact AD computation from diagnostics.run_applicability_domain.
"""
from __future__ import annotations

import warnings
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import joblib
import numpy as np
import pandas as pd
import shap
from sklearn.base import clone
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.neighbors import NearestNeighbors
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parent


def _resolve_artifacts_dir(root: Path) -> Path:
    """Resolve artifact directory for app loading."""
    env_dir = os.getenv("MSR_ARTIFACTS_DIR")
    if env_dir:
        return Path(env_dir)

    return root / "artifacts"


ARTIFACTS = _resolve_artifacts_dir(ROOT)

_REQUIRED_ARTIFACTS = [
    Path("final_model/models/xgboost_final_model.joblib"),
    Path("data/full_dataset_with_descriptors.pkl"),
]

# Mirror config.py constants (must match exactly for reproducibility)
SEED = 42
TEST_SIZE = 0.10
N_BINS = 5
K_NEIGHBORS = 7
AD_ERROR_QUANTILE = 0.90
Q_REF = 0.95  # structural distance quantile used at deployment

# Columns treated as metadata (excluded from feature matrix)
_META_COLS = ["SampleID", "Contaminant", "Surfactant", "SMILES_contaminant", "SMILES_surfactant"]


# ---------------------------------------------------------------------------
# Result dataclasses
# ---------------------------------------------------------------------------

@dataclass
class ADCriterion:
    name: str
    description: str
    value: float
    threshold: float
    pass_: bool
    value_label: str = ""
    threshold_label: str = ""


@dataclass
class ADResult:
    structural: ADCriterion
    y_range: ADCriterion
    local_error: ADCriterion
    overall: bool
    neighbor_indices: np.ndarray
    neighbor_dists: np.ndarray


@dataclass
class PredictionResult:
    log10_msr: float
    msr: float
    ad: ADResult
    shap_values: np.ndarray
    shap_base: float
    feature_names: list
    feature_values: np.ndarray


# ---------------------------------------------------------------------------
# Main predictor class
# ---------------------------------------------------------------------------

class MSRPredictor:
    """
    Loads all necessary artifacts and computes:
      - log10(MSR) prediction via the trained XGBoost pipeline
      - Applicability Domain (3 criteria, exact replication of diagnostics.py)
      - SHAP explanation for the prediction
    """

    def __init__(self, root: Path = ROOT):
        self.root = root
        self.artifacts = _resolve_artifacts_dir(root)
        self._load()

    # ------------------------------------------------------------------
    # Initialisation
    # ------------------------------------------------------------------

    _CACHE_NAME = "app_startup_cache.joblib"

    def _load(self) -> None:
        """Load model, training reference, AD reference, and SHAP explainer."""

        missing = [self.artifacts / rel for rel in _REQUIRED_ARTIFACTS if not (self.artifacts / rel).exists()]
        if missing:
            missing_str = "\n".join(f"- {p}" for p in missing)
            raise FileNotFoundError(
                "Required app artifacts are missing.\n"
                f"Artifacts directory: {self.artifacts}\n"
                "Missing files:\n"
                f"{missing_str}\n"
                "Run `python msr_pipeline.py` from repository root or set MSR_ARTIFACTS_DIR to a valid artifacts directory."
            )

        # --- 1. Model pipeline ---
        model_path = self.artifacts / "final_model/models/xgboost_final_model.joblib"
        self.pipe = joblib.load(model_path)

        # Preprocessor = all steps except the final estimator
        self.preproc = Pipeline(self.pipe.steps[:-1])
        self.model = self.pipe.steps[-1][1]  # XGBRegressor

        # Feature names from the ColumnSubset step (the 30 final descriptors)
        from transformers import ColumnSubset
        self.feature_names = []  # type: list
        for _, step in self.pipe.steps[:-1]:
            if isinstance(step, ColumnSubset):
                self.feature_names = list(step.columns_ or step.columns or [])
                break
        if not self.feature_names:
            self.feature_names = [f"feature_{i}" for i in range(30)]

        # --- 2. Try loading pre-computed cache ---
        cache_path = self.artifacts / self._CACHE_NAME
        if self._try_load_cache(cache_path):
            return

        # --- 3. Full computation (first run only) ---
        self._compute_references()

        # --- 4. Save cache for next startup ---
        self._save_cache(cache_path)

    def _try_load_cache(self, cache_path: Path) -> bool:
        """Load pre-computed reference data from cache. Returns True on success."""
        if not cache_path.exists():
            return False
        try:
            cache = joblib.load(cache_path)
            self._X_cv_pre = cache["X_cv_pre"]
            self._y_cv_log = cache["y_cv_log"]
            self._meta_cv = cache["meta_cv"]
            self._feature_stats = cache["feature_stats"]
            self._abs_err_cv = cache["abs_err_cv"]
            self._scaler = cache["scaler"]
            self._nn = cache["nn"]
            self._mean_dist_train = cache["mean_dist_train"]
            self._local_err_train = cache["local_err_train"]
            self._dist_threshold = cache["dist_threshold"]
            self._err_threshold = cache["err_threshold"]
            self._y_min = cache["y_min"]
            self._y_max = cache["y_max"]
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                self._shap_explainer = shap.TreeExplainer(self.model)
            return True
        except Exception:
            return False

    def _save_cache(self, cache_path: Path) -> None:
        """Persist derived reference data for fast subsequent startups."""
        try:
            joblib.dump(
                {
                    "X_cv_pre": self._X_cv_pre,
                    "y_cv_log": self._y_cv_log,
                    "meta_cv": self._meta_cv,
                    "feature_stats": self._feature_stats,
                    "abs_err_cv": self._abs_err_cv,
                    "scaler": self._scaler,
                    "nn": self._nn,
                    "mean_dist_train": self._mean_dist_train,
                    "local_err_train": self._local_err_train,
                    "dist_threshold": self._dist_threshold,
                    "err_threshold": self._err_threshold,
                    "y_min": self._y_min,
                    "y_max": self._y_max,
                },
                cache_path,
                compress=3,
            )
        except Exception:
            pass  # non-critical; next startup will recompute

    def _compute_references(self) -> None:
        """Full computation of training references, OOF errors, kNN, and SHAP."""

        # --- Training data: replicate the CV/holdout split exactly ---
        df_full = pd.read_pickle(self.artifacts / "data/full_dataset_with_descriptors.pkl")

        meta_cols_present = [c for c in _META_COLS if c in df_full.columns]
        drop_for_X = [c for c in meta_cols_present + ["MSR"] if c in df_full.columns]

        y_all = df_full["MSR"].copy()
        X_all = df_full.drop(columns=drop_for_X)
        meta_all = df_full[meta_cols_present].copy() if meta_cols_present else pd.DataFrame(index=df_full.index)

        bins_all = pd.qcut(y_all, q=N_BINS, labels=False, duplicates="drop")

        X_cv, _, y_cv, _, _, _, meta_cv, _ = train_test_split(
            X_all, y_all, bins_all, meta_all,
            test_size=TEST_SIZE, random_state=SEED, stratify=bins_all,
        )

        X_cv = X_cv.reset_index(drop=True)
        y_cv = y_cv.reset_index(drop=True)
        meta_cv = meta_cv.reset_index(drop=True)

        self._y_cv_log = np.log10(y_cv.to_numpy(dtype=float))
        self._meta_cv = meta_cv

        # --- Preprocess CV features ---
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            X_cv_pre = np.asarray(self.preproc.transform(X_cv), dtype=float)

        self._X_cv_pre = X_cv_pre  # (127, 30) in scaled descriptor space
        self._feature_stats = pd.DataFrame(
            {
                "feature": self.feature_names,
                "train_mean": X_cv_pre.mean(axis=0),
                "train_std": X_cv_pre.std(axis=0),
                "train_min": X_cv_pre.min(axis=0),
                "train_max": X_cv_pre.max(axis=0),
            }
        )

        # --- Out-of-fold CV errors (exact replica of diagnostics.py logic) ---
        y_cv_series = pd.Series(self._y_cv_log)
        strata = pd.qcut(
            y_cv_series,
            q=min(N_BINS, max(2, len(y_cv_series) // 5)),
            labels=False,
            duplicates="drop",
        )
        n_splits_eff = int(min(5, strata.value_counts().min()))
        skf = StratifiedKFold(
            n_splits=max(2, n_splits_eff),
            shuffle=True,
            random_state=SEED + 202,
        )

        y_cv_pred_oof = np.empty_like(self._y_cv_log)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            for itr, iva in skf.split(X_cv_pre, strata.to_numpy()):
                m = clone(self.model).fit(X_cv_pre[itr], self._y_cv_log[itr])
                y_cv_pred_oof[iva] = m.predict(X_cv_pre[iva])

        self._abs_err_cv = np.abs(self._y_cv_log - y_cv_pred_oof)

        # --- kNN in StandardScaler-transformed feature space ---
        self._scaler = StandardScaler().fit(X_cv_pre)
        X_cv_scaled = self._scaler.transform(X_cv_pre)

        self._nn = NearestNeighbors(n_neighbors=K_NEIGHBORS, metric="euclidean")
        self._nn.fit(X_cv_scaled)

        # Training distances and local errors (used to compute thresholds)
        dist_train, idx_train = self._nn.kneighbors(X_cv_scaled)
        self._mean_dist_train = dist_train.mean(axis=1)
        self._local_err_train = self._abs_err_cv[idx_train].mean(axis=1)

        self._dist_threshold = float(np.quantile(self._mean_dist_train, Q_REF))
        self._err_threshold = float(np.quantile(self._local_err_train, AD_ERROR_QUANTILE))
        self._y_min = float(np.min(self._y_cv_log))
        self._y_max = float(np.max(self._y_cv_log))

        # --- SHAP explainer ---
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            self._shap_explainer = shap.TreeExplainer(self.model)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def validate_smiles(self, smiles: str):
        """Return (is_valid, canonical_smiles_or_error_message)."""
        from data_prep import canonical_main_organic_fragment
        canon = canonical_main_organic_fragment(str(smiles).strip())
        if canon is None:
            return False, "Could not parse SMILES"
        return True, canon

    def predict(
        self,
        cont_smiles: str,
        surf_smiles: str,
        cont_type: str = "Unknown",
        surf_type: str = "Unknown",
        cmc: Optional[float] = None,
        water_solubility: Optional[float] = None,
    ) -> PredictionResult:
        """
        Full prediction pipeline:
          1. Compute Mordred descriptors from SMILES
          2. Predict log10(MSR) via the trained pipeline
          3. Check applicability domain (3 criteria)
          4. Compute SHAP values for interpretation
        """
        input_row = self._build_input_row(
            cont_smiles, surf_smiles, cont_type, surf_type, cmc, water_solubility
        )

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            log10_msr = float(self.pipe.predict(input_row)[0])
            X_pre = np.asarray(self.preproc.transform(input_row), dtype=float)

        X_scaled = self._scaler.transform(X_pre)
        ad = self._compute_ad(X_scaled, log10_msr)

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            shap_vals = self._shap_explainer.shap_values(X_pre)

        # shap_vals shape: (1, n_features) for regression
        sv = shap_vals[0] if hasattr(shap_vals, "__len__") and shap_vals.ndim == 2 else shap_vals

        return PredictionResult(
            log10_msr=log10_msr,
            msr=10.0 ** log10_msr,
            ad=ad,
            shap_values=sv,
            shap_base=float(self._shap_explainer.expected_value),
            feature_names=self.feature_names,
            feature_values=X_pre[0],
        )

    def get_neighbors(self, ad: ADResult) -> pd.DataFrame:
        """Return DataFrame with k nearest training compounds."""
        rows = []
        for rank, (i, dist) in enumerate(zip(ad.neighbor_indices, ad.neighbor_dists)):
            row: dict = {
                "Rank": rank + 1,
                "Distance": round(float(dist), 3),
                "log₁₀(MSR)": round(float(self._y_cv_log[i]), 3),
                "MSR": round(float(10.0 ** self._y_cv_log[i]), 5),
            }
            if i < len(self._meta_cv):
                meta_row = self._meta_cv.iloc[i]
                for col in ["Contaminant", "Surfactant", "SMILES_contaminant", "SMILES_surfactant"]:
                    if col in meta_row.index:
                        row[col] = meta_row[col]
            rows.append(row)
        return pd.DataFrame(rows)

    def get_feature_stats(self) -> pd.DataFrame:
        """Return cached training feature statistics for model descriptors."""
        return self._feature_stats.copy()

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _build_input_row(
        self,
        cont_smiles: str,
        surf_smiles: str,
        cont_type: str,
        surf_type: str,
        cmc: Optional[float],
        water_solubility: Optional[float],
    ) -> pd.DataFrame:
        """Assemble a 1-row DataFrame with all columns expected by the pipeline."""
        from data_prep import canonical_main_organic_fragment, MORDRED_CALC
        from rdkit import Chem

        def _descs(smiles: str, prefix: str) -> dict:
            canon = canonical_main_organic_fragment(smiles.strip())
            if canon is None:
                raise ValueError(f"Invalid SMILES: {smiles!r}")
            mol = Chem.MolFromSmiles(canon)
            if mol is None:
                raise ValueError(f"RDKit cannot parse SMILES: {smiles!r}")
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                result = MORDRED_CALC(mol)
            out = {}
            for k, v in result.items():
                try:
                    val = float(v)
                    out[f"{prefix}{k}"] = val if np.isfinite(val) else np.nan
                except (TypeError, ValueError):
                    out[f"{prefix}{k}"] = np.nan
            return out

        base: dict = {
            "ContaminantType": cont_type,
            "SurfactantType": surf_type,
            "CMC": cmc if cmc is not None else np.nan,
            "WaterSolubility": water_solubility if water_solubility is not None else np.nan,
        }
        base.update(_descs(cont_smiles, "Cont_"))
        base.update(_descs(surf_smiles, "Surf_"))
        return pd.DataFrame([base])

    def _compute_ad(self, X_scaled: np.ndarray, y_pred: float) -> ADResult:
        dist, idx = self._nn.kneighbors(X_scaled)
        mean_dist = float(dist[0].mean())
        local_err = float(self._abs_err_cv[idx[0]].mean())

        inside_struct = mean_dist <= self._dist_threshold
        inside_y = (self._y_min <= y_pred <= self._y_max)
        inside_err = local_err <= self._err_threshold

        return ADResult(
            structural=ADCriterion(
                name="Structural Similarity",
                description=(
                    f"Mean Euclidean distance to {K_NEIGHBORS} nearest training neighbors "
                    f"in the scaled 30-descriptor space"
                ),
                value=mean_dist,
                threshold=self._dist_threshold,
                pass_=inside_struct,
                value_label=f"{mean_dist:.3f}",
                threshold_label=f"≤ {self._dist_threshold:.3f}  (q = {Q_REF:.0%} of training distances)",
            ),
            y_range=ADCriterion(
                name="Prediction Range",
                description=(
                    f"Predicted log₁₀(MSR) lies within the training Y-range "
                    f"[{self._y_min:.2f}, {self._y_max:.2f}]"
                ),
                value=y_pred,
                threshold=0.0,
                pass_=inside_y,
                value_label=f"{y_pred:.3f}",
                threshold_label=f"[{self._y_min:.2f},  {self._y_max:.2f}]",
            ),
            local_error=ADCriterion(
                name="Local Reliability",
                description=(
                    f"Mean CV out-of-fold error of the {K_NEIGHBORS} nearest training neighbors"
                ),
                value=local_err,
                threshold=self._err_threshold,
                pass_=inside_err,
                value_label=f"{local_err:.3f}",
                threshold_label=f"≤ {self._err_threshold:.3f}  (q = {AD_ERROR_QUANTILE:.0%} of training local errors)",
            ),
            overall=(inside_struct and inside_y and inside_err),
            neighbor_indices=idx[0],
            neighbor_dists=dist[0],
        )
