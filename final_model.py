from typing import List, Dict, Any
import json

import numpy as np
import pandas as pd
import optuna
import joblib

from sklearn.base import clone
from sklearn.model_selection import RepeatedStratifiedKFold
from sklearn.metrics import r2_score

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
from plot_style import set_plot_style, get_discrete_colors


from config import (
    SEED,
    FINAL_MODELS_DIR,
    FINAL_PREDS_DIR,
    FINAL_METRICS_DIR,
    FINAL_FIGURES_DIR,
)

# Reuse utilities from model_selection
from model_selection import (
    _make_estimators,
    PARAM_SPACES,
    TRIALS_MAP,
    _make_study,
    _sample_from_space,
    _build_model_pipeline_for_fold,
    _rmse,
    _safe_name,
)


def _save_parity_plot_cv_holdout(
    y_cv_true: np.ndarray,
    y_cv_pred: np.ndarray,
    y_hold_true: np.ndarray,
    y_hold_pred: np.ndarray,
    out_paths: List,
    title: str = "",
) -> None:
    """Save a parity plot with CV and holdout points."""
    # Ensure parity figure uses the same global visual style as notebook/manuscript plots.
    set_plot_style()

    y_cv_true = np.asarray(y_cv_true, dtype=float)
    y_cv_pred = np.asarray(y_cv_pred, dtype=float)
    y_hold_true = np.asarray(y_hold_true, dtype=float)
    y_hold_pred = np.asarray(y_hold_pred, dtype=float)

    all_true = np.concatenate([y_cv_true, y_hold_true])
    all_pred = np.concatenate([y_cv_pred, y_hold_pred])

    vmin = float(min(all_true.min(), all_pred.min()))
    vmax = float(max(all_true.max(), all_pred.max()))
    span = vmax - vmin
    pad = 0.01
    if span <= 0:
        lo, hi = vmin - 1.0, vmax + 1.0
    else:
        lo, hi = vmin - pad * span, vmax + pad * span

    palette = get_discrete_colors(2)
    color_train = palette[0]
    color_holdout = palette[1]

    fig, ax = plt.subplots(figsize=(6.0, 6.0))
    sns.scatterplot(
        x=y_cv_true,
        y=y_cv_pred,
        ax=ax,
        s=24,
        alpha=0.4,
        color=color_train,
        edgecolor="none",
        label="Train (90%)",
    )
    sns.scatterplot(
        x=y_hold_true,
        y=y_hold_pred,
        ax=ax,
        s=45,
        alpha=1.0,
        marker="X",
        color=color_holdout,
        edgecolor=color_holdout,
        linewidth=0.7,
        label="Holdout (10%)",
    )
    ax.plot([lo, hi], [lo, hi], linestyle="--", linewidth=1.0, color="0.2")

    ax.set_xlim(lo, hi)
    ax.set_ylim(lo, hi)
    ax.set_axisbelow(True)
    ax.grid(True)
    ax.set_xlabel(r"True $\log_{10}(\mathrm{MSR})$")
    ax.set_ylabel(r"Predicted $\log_{10}(\mathrm{MSR})$")
    if title:
        ax.set_title(title)
    ax.legend(
        loc="lower center",
        bbox_to_anchor=(0.5, 0.99),
        ncol=2,
        frameon=False,
    )

    fig.tight_layout()
    for path in out_paths:
        fig.savefig(path, dpi=300)
    plt.close(fig)


def train_final_model(
    X_cv: pd.DataFrame,
    y_cv_log: pd.Series,
    bins_cv: pd.Series,
    X_holdout: pd.DataFrame,
    y_holdout_log: pd.Series,
    categorical_cols: List[str],
    log10_cols: List[str],
    final_features: List[str],
    best_model_name: str,
    force_recompute: bool = False,
) -> pd.DataFrame:
    """
    Tune the best model on the full CV set and train the final model.

    The tuned model is fitted on X_cv, evaluated on holdout, and artifacts
    (params, model, predictions, metrics, figures) are saved.
    """
    FINAL_MODELS_DIR.mkdir(parents=True, exist_ok=True)
    FINAL_PREDS_DIR.mkdir(parents=True, exist_ok=True)
    FINAL_METRICS_DIR.mkdir(parents=True, exist_ok=True)
    FINAL_FIGURES_DIR.mkdir(parents=True, exist_ok=True)

    estimators = _make_estimators()
    if best_model_name not in estimators:
        raise ValueError(f"Unknown best_model_name: {best_model_name}")

    base_est = estimators[best_model_name]
    mslug = _safe_name(best_model_name)

    params_path = FINAL_MODELS_DIR / f"{mslug}_final_params.json"
    model_path = FINAL_MODELS_DIR / f"{mslug}_final_model.joblib"
    preds_path = FINAL_PREDS_DIR 
    metrics_path = FINAL_METRICS_DIR / f"{mslug}_cv_holdout_metrics.csv"
    fig_path_png = FINAL_FIGURES_DIR / f"{mslug}_cv_holdout_parity.png"
    short_slug = "gb" if mslug == "gradient_boosting" else mslug
    fig_path_pdf = FINAL_FIGURES_DIR / f"parity_log10msr_{short_slug}.pdf"

    # Hyperparameter tuning on the full CV set
    if params_path.exists() and not force_recompute:
        with open(params_path, "r", encoding="utf-8") as f:
            best_params: Dict[str, Any] = json.load(f)
        print(f"[Final model] Loaded params from {params_path.name}")
    else:
        space = PARAM_SPACES[best_model_name]
        n_trials = TRIALS_MAP.get(best_model_name, 50)

        study = _make_study(n_trials=n_trials, seed=SEED + 999)

        strata = bins_cv
        min_per_class = pd.Series(strata).value_counts().min()
        n_splits_eff = max(2, min(5, int(min_per_class)))

        inner_cv = RepeatedStratifiedKFold(
            n_splits=n_splits_eff,
            n_repeats=3,
            random_state=SEED + 999,
        )
        inner_splits = list(inner_cv.split(X_cv, strata))

        model_pipe = _build_model_pipeline_for_fold(
            estimator=base_est,
            selected_features=final_features,
            categorical_cols=categorical_cols,
            log10_cols=log10_cols,
        )

        def objective(trial: optuna.Trial) -> float:
            params = _sample_from_space(trial, space)
            pipe = clone(model_pipe)
            pipe.set_params(**params)

            scores: List[float] = []
            for step, (itr, iva) in enumerate(inner_splits):
                X_tr = X_cv.iloc[itr]
                X_va = X_cv.iloc[iva]
                y_tr = y_cv_log.iloc[itr]
                y_va = y_cv_log.iloc[iva]

                m = clone(pipe).fit(X_tr, y_tr)
                pred = m.predict(X_va)
                rmse = _rmse(y_va.to_numpy(), pred)
                scores.append(-rmse)

                trial.report(float(np.median(scores)), step=step)
                if trial.should_prune():
                    raise optuna.TrialPruned()

            return float(np.median(scores))

        print(
            f"\n[Final model] Tuning {best_model_name} on full CV set "
            f"({n_trials} trials, {len(final_features)} features)..."
        )
        study.optimize(objective, n_trials=n_trials, show_progress_bar=True)
        best_params = study.best_trial.params

        with open(params_path, "w", encoding="utf-8") as f:
            json.dump(best_params, f, ensure_ascii=False, indent=2)
        print(
            f"[Final model] Best median(-RMSE)={study.best_value:.4f} "
            f"(trial {study.best_trial.number}); params saved → {params_path.name}"
        )

    # Fit final model on full CV data
    final_pipe = _build_model_pipeline_for_fold(
        estimator=base_est,
        selected_features=final_features,
        categorical_cols=categorical_cols,
        log10_cols=log10_cols,
    )
    final_pipe.set_params(**best_params)
    final_pipe.fit(X_cv, y_cv_log)

    joblib.dump(final_pipe, model_path)
    print(f"[Final model] Saved fitted model → {model_path.name}")

    # Predictions on CV and holdout
    y_cv_pred = final_pipe.predict(X_cv)
    y_hold_pred = final_pipe.predict(X_holdout)

    # Metrics
    r2_train_cv = float(r2_score(y_cv_log.to_numpy(), y_cv_pred))
    rmse_train_cv = _rmse(y_cv_log.to_numpy(), y_cv_pred)

    r2_holdout = float(r2_score(y_holdout_log.to_numpy(), y_hold_pred))
    rmse_holdout = _rmse(y_holdout_log.to_numpy(), y_hold_pred)

    metrics_df = pd.DataFrame(
        [
            {
                "Model": best_model_name,
                "n_features": len(final_features),
                "R2_train_cv": r2_train_cv,
                "RMSE_train_cv": rmse_train_cv,
                "R2_holdout": r2_holdout,
                "RMSE_holdout": rmse_holdout,
            }
        ]
    )
    metrics_df.to_csv(metrics_path, index=False)

    print(
        f"[CV train] R²={r2_train_cv:.3f}, RMSE={rmse_train_cv:.3f} | "
        f"[Holdout] R²={r2_holdout:.3f}, RMSE={rmse_holdout:.3f}"
    )

    # 5) Save predictions for CV and holdout
    preds_cv = pd.DataFrame(
        {
            "split": "cv",
            "y_true": y_cv_log.to_numpy(),
            "y_pred": y_cv_pred,
        }
    )
    preds_hold = pd.DataFrame(
        {
            "split": "holdout",
            "y_true": y_holdout_log.to_numpy(),
            "y_pred": y_hold_pred,
        }
    )
  
    preds_hold_path = preds_path / f"{mslug}_holdout_predictions.csv"
    preds_cv_path = preds_path / f"{mslug}_train_predictions.csv"

    preds_cv.to_csv(preds_cv_path, index=False)
    preds_hold.to_csv(preds_hold_path, index=False)

    # Parity plot CV + holdout
    _save_parity_plot_cv_holdout(
        y_cv_true=y_cv_log.to_numpy(),
        y_cv_pred=y_cv_pred,
        y_hold_true=y_holdout_log.to_numpy(),
        y_hold_pred=y_hold_pred,
        title="",
        out_paths=[fig_path_png, fig_path_pdf],
    )
    print(f"[Figures] Saved parity plots → {fig_path_png.name}, {fig_path_pdf.name}")

    return metrics_df
