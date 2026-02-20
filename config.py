from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent

DATA_DIR = PROJECT_ROOT / "data"
ARTIFACTS_DIR = PROJECT_ROOT / "artifacts"

# Processed data
DATA_ARTIFACTS_DIR = ARTIFACTS_DIR / "data"
PROCESSED_FULL_DF_PATH = DATA_ARTIFACTS_DIR / "full_dataset_with_descriptors.pkl"
DATA_FIGURES_DIR = DATA_ARTIFACTS_DIR / "figures"

# Feature selection
FEAT_DIR = ARTIFACTS_DIR / "feature_selection"
FEAT_STABILITY_DIR = FEAT_DIR / "stability_rankings"
FEAT_K_CURVES_DIR = FEAT_DIR / "k_curves"
FEAT_SUMMARY_DIR = FEAT_DIR / "summaries"
FEAT_FIGURES_DIR = FEAT_DIR / "figures"

# Model selection (nested CV)
MODELSEL_DIR = ARTIFACTS_DIR / "model_selection"
MODELSEL_PARAMS_DIR = MODELSEL_DIR / "params"
MODELSEL_METRICS_DIR = MODELSEL_DIR / "metrics"
MODELSEL_SUMMARY_DIR = MODELSEL_DIR / "summary"

# Final model + holdout
FINAL_DIR = ARTIFACTS_DIR / "final_model"
FINAL_MODELS_DIR = FINAL_DIR / "models"
FINAL_PREDS_DIR = FINAL_DIR / "preds"
FINAL_METRICS_DIR = FINAL_DIR / "metrics"
FINAL_FIGURES_DIR = FINAL_DIR / "figures"

# Diagnostics
DIAG_DIR = ARTIFACTS_DIR / "diagnostics"
DIAG_YRAND_DIR = DIAG_DIR / "y_randomization"
DIAG_SHAP_DIR = DIAG_DIR / "shap"
DIAG_AD_DIR = DIAG_DIR / "applicability_domain"


def ensure_directories() -> None:
    """Create all required artifact directories if they do not exist."""
    dirs = [
        DATA_DIR,
        ARTIFACTS_DIR,
        DATA_ARTIFACTS_DIR,
        DATA_FIGURES_DIR,
        FEAT_DIR,
        FEAT_STABILITY_DIR,
        FEAT_K_CURVES_DIR,
        FEAT_SUMMARY_DIR,
        FEAT_FIGURES_DIR,
        MODELSEL_DIR,
        MODELSEL_PARAMS_DIR,
        MODELSEL_METRICS_DIR,
        MODELSEL_SUMMARY_DIR,
        FINAL_DIR,
        FINAL_MODELS_DIR,
        FINAL_PREDS_DIR,
        FINAL_METRICS_DIR,
        FINAL_FIGURES_DIR,
        DIAG_DIR,
        DIAG_YRAND_DIR,
        DIAG_SHAP_DIR,
        DIAG_AD_DIR
    ]
    for d in dirs:
        d.mkdir(parents=True, exist_ok=True)


# Global configuration
SEED = 42
TEST_SIZE = 0.10
N_BINS = 5  # target stratification

# CV settings
OUTER_N_SPLITS = 5
INNER_MAX_SPLITS = 3
INNER_REPEATS = 2

# Feature selection settings
B_REPEATS = 60
MAX_K_RANK = 80
RF_N_ESTIM = 300
K_GRID = [10, 20, 30, 40, 50, 60]
K_COVER = 0.98
FREQ_MIN = 0.60

# Y-randomization
N_PERM = 80

# AD
AD_ERROR_QUANTILE = 0.90

# Stage recompute controls (False = reuse cached artifacts when available)
FORCE_RECOMPUTE = {
    "data_prep": False,
    "feature_stability": False,
    "k_selection": False,
    "model_selection": False,
    "final_model": False,
    "y_randomization": False,
    "applicability_domain": False,
}
