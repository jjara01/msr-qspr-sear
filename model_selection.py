from typing import Dict, Any, List, Tuple

import json
import warnings

import numpy as np
import pandas as pd
import optuna
from optuna.distributions import IntDistribution, FloatDistribution, CategoricalDistribution

from sklearn.base import clone
from sklearn.model_selection import RepeatedStratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.svm import SVR
from sklearn.neighbors import KNeighborsRegressor

from optuna.exceptions import ExperimentalWarning

from config import (
    SEED,
    MODELSEL_PARAMS_DIR,
    MODELSEL_METRICS_DIR,
    MODELSEL_SUMMARY_DIR,
    FEAT_STABILITY_DIR,
)
from transformers import (
    DropMostlyNaN,
    DataFrameImputer,
    SafeLog10,
    make_encoder_df,
    ColumnSubset,
)


def _rmse(a: np.ndarray, b: np.ndarray) -> float:
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    return float(np.sqrt(np.mean((a - b) ** 2)))


def _safe_name(name: str) -> str:
    return name.replace(" ", "_").lower()


def _make_estimators() -> Dict[str, Any]:
    """Return base estimators for model comparison."""
    from xgboost import XGBRegressor

    return {
        "Random Forest":     RandomForestRegressor(random_state=SEED, n_jobs=-1),
        "Gradient Boosting": GradientBoostingRegressor(random_state=SEED, n_iter_no_change=None),
        "XGBoost":           XGBRegressor(random_state=SEED, n_estimators=200, n_jobs=-1, tree_method="hist"),
        "SVR":               SVR(),
        "KNN":               KNeighborsRegressor(),
    }


# Hyperparameter spaces
PARAM_SPACES: Dict[str, Dict[str, Any]] = {
    "Random Forest": {
        "model__n_estimators":      IntDistribution(100, 600, step=50),
        "model__max_depth":         CategoricalDistribution([None, 3, 4, 5, 6, 8, 10, 12]),
        "model__min_samples_split": IntDistribution(2, 20, step=1),
        "model__min_samples_leaf":  IntDistribution(1, 8, step=1),
        "model__max_features":      FloatDistribution(0.3, 0.9),
        "model__bootstrap":         CategoricalDistribution([True]),
    },
    "Gradient Boosting": {
        "model__n_estimators":      IntDistribution(100, 700, step=50),
        "model__learning_rate":     FloatDistribution(0.01, 0.15, log=True),
        "model__max_depth":         IntDistribution(1, 4, step=1),
        "model__min_samples_split": IntDistribution(2, 25, step=1),
        "model__min_samples_leaf":  IntDistribution(1, 10, step=1),
        "model__subsample":         FloatDistribution(0.6, 1.0),
        "model__max_features":      FloatDistribution(0.5, 1.0),
        "model__loss":              CategoricalDistribution(["huber"]),
        "model__alpha":             FloatDistribution(0.85, 0.95),
    },
    "XGBoost": {
        "model__n_estimators":      IntDistribution(100, 800, step=50),
        "model__learning_rate":     FloatDistribution(0.01, 0.2, log=True),
        "model__max_depth":         IntDistribution(2, 6, step=1),
        "model__min_child_weight":  IntDistribution(1, 20, step=1),
        "model__subsample":         FloatDistribution(0.6, 1.0),
        "model__colsample_bytree":  FloatDistribution(0.6, 1.0),
        "model__reg_alpha":         FloatDistribution(1e-3, 50.0, log=True),
        "model__reg_lambda":        FloatDistribution(1e-3, 50.0, log=True),
        "model__gamma":             FloatDistribution(0.0, 2.0),
    },
    "SVR": {
        "model__kernel":  CategoricalDistribution(["rbf"]),
        "model__C":       FloatDistribution(0.1, 200.0, log=True),
        "model__epsilon": FloatDistribution(1e-3, 0.5, log=True),
        "model__gamma":   FloatDistribution(1e-4, 1e-1, log=True),
    },
    "KNN": {
        "model__n_neighbors": IntDistribution(3, 25, step=2),
        "model__weights":     CategoricalDistribution(["uniform", "distance"]),
        "model__p":           CategoricalDistribution([1, 2]),
    },
}

TRIALS_MAP: Dict[str, int] = {
    "XGBoost":           100,
    "Gradient Boosting": 100,
    "Random Forest":      60,
    "SVR":                50,
    "KNN":                30,
}


def _make_study(n_trials: int, seed: int) -> optuna.Study:
    """Create Optuna study with TPE sampler and median pruner."""
    n_startup = int(max(20, min(90, round(0.15 * n_trials))))

    # Suppress Optuna ExperimentalWarning for multivariate/group options
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=ExperimentalWarning)
        sampler = optuna.samplers.TPESampler(
            seed=seed,
            n_startup_trials=n_startup,
            multivariate=True,
            group=True,
            consider_prior=True,
            prior_weight=1.0,
            consider_endpoints=True,
            consider_magic_clip=True,
        )

    pruner = optuna.pruners.MedianPruner(
        n_startup_trials=max(10, n_startup // 2),
        n_warmup_steps=2,
        interval_steps=1,
    )

    optuna.logging.set_verbosity(optuna.logging.WARNING)

    return optuna.create_study(direction="maximize", sampler=sampler, pruner=pruner)



def _sample_from_space(trial: optuna.Trial, space: Dict[str, Any]) -> Dict[str, Any]:
    """Sample a hyperparameter dict from an Optuna distribution space."""
    params: Dict[str, Any] = {}
    for key, dist in space.items():
        if isinstance(dist, IntDistribution):
            step = 1 if dist.step is None else dist.step
            if dist.log:
                params[key] = trial.suggest_int(key, dist.low, dist.high, log=True)
            else:
                params[key] = trial.suggest_int(key, dist.low, dist.high, step=step)
        elif isinstance(dist, FloatDistribution):
            params[key] = trial.suggest_float(
                key, dist.low, dist.high, log=bool(dist.log), step=dist.step
            )
        elif isinstance(dist, CategoricalDistribution):
            params[key] = trial.suggest_categorical(key, dist.choices)
        else:
            raise ValueError(f"Unsupported distribution type for {key}: {type(dist)}")
    return params


def _build_model_pipeline_for_fold(
    estimator,
    selected_features: List[str],
    categorical_cols: List[str],
    log10_cols: List[str],
) -> Pipeline:
    """
    Build the per-fold model pipeline.

    The feature set is fixed (top-k from stability ranking).
    Preprocessing steps are refit within each CV split.
    """
    steps = [
        ("drop_sparse", DropMostlyNaN(threshold=0.95)),
        ("imputer", DataFrameImputer()),
        ("log10", SafeLog10(columns=log10_cols)),
        ("ohe", make_encoder_df(categorical_cols)),
        ("subset", ColumnSubset(columns=selected_features)),
    ]

    name = estimator.__class__.__name__.lower()
    if any(s in name for s in ["svr", "kneighbors"]):
        steps.append(("scaler", StandardScaler()))

    steps.append(("model", clone(estimator)))
    return Pipeline(steps)


def run_model_selection(
    X_cv: pd.DataFrame,
    y_cv_log: pd.Series,
    bins_cv: pd.Series,
    outer_splits: List[Tuple[np.ndarray, np.ndarray]],
    categorical_cols: List[str],
    log10_cols: List[str],
    k_global: int,
    force_recompute: bool = False,
) -> Tuple[pd.DataFrame, pd.DataFrame, str]:
    """
    Run nested CV over candidate models using a fixed k_global per fold.

    For each outer fold:
      - load stability ranking
      - keep top-k_global features for that fold
      - tune each candidate model by inner CV (Optuna, median(-RMSE))
      - evaluate tuned model on outer-test

    Returns:
      outer_df       : per-fold metrics
      outer_summary  : aggregated metrics by model
      best_model_name: name of model with best mean outer R²
    """
    MODELSEL_PARAMS_DIR.mkdir(parents=True, exist_ok=True)
    MODELSEL_METRICS_DIR.mkdir(parents=True, exist_ok=True)
    MODELSEL_SUMMARY_DIR.mkdir(parents=True, exist_ok=True)

    estimators = _make_estimators()

    outer_records: List[Dict[str, Any]] = []

    for fold_id, (i_tr, i_te) in enumerate(outer_splits, start=1):
        print(f"\n[Nested] Outer fold {fold_id}/{len(outer_splits)}")

        # Outer split
        X_tr_outer = X_cv.iloc[i_tr]
        X_te_outer = X_cv.iloc[i_te]
        y_tr_outer_log = y_cv_log.iloc[i_tr]
        y_te_outer_log = y_cv_log.iloc[i_te]
        strata_tr = bins_cv.iloc[i_tr]

        # Per-fold stability ranking
        rank_path = FEAT_STABILITY_DIR / f"fold{fold_id}_freq_within_outer.csv"
        if not rank_path.exists():
            raise FileNotFoundError(f"Missing stability ranking for fold {fold_id}: {rank_path}")

        freq_df = (
            pd.read_csv(rank_path)
            .sort_values(["frequency", "count"], ascending=[False, False])
            .reset_index(drop=True)
        )
        ranked_feats = freq_df["feature"].astype(str).tolist()
        if len(ranked_feats) < k_global:
            raise ValueError(
                f"Fold {fold_id}: requested k_global={k_global} but only {len(ranked_feats)} features in ranking."
            )
        selected_feats = ranked_feats[:k_global]

        # Inner CV
        min_per_class = pd.Series(strata_tr).value_counts().min()
        n_splits_eff = max(2, min(3, int(min_per_class)))
        inner_cv = RepeatedStratifiedKFold(
            n_splits=n_splits_eff,
            n_repeats=2,
            random_state=SEED + fold_id,
        )
        inner_splits = list(inner_cv.split(X_tr_outer, strata_tr.to_numpy()))

        # Loop over candidate models
        for model_name, base_est in estimators.items():
            mslug = _safe_name(model_name)
            params_file = MODELSEL_PARAMS_DIR / f"fold{fold_id}_{mslug}_best_params.json"

            model_pipe = _build_model_pipeline_for_fold(
                estimator=base_est,
                selected_features=selected_feats,
                categorical_cols=categorical_cols,
                log10_cols=log10_cols,
            )

            if params_file.exists() and not force_recompute:
                with open(params_file, "r", encoding="utf-8") as f:
                    best_params = json.load(f)
                print(f"  [{model_name}] fold {fold_id}: loaded params from {params_file.name}")
            else:
                space = PARAM_SPACES[model_name]
                n_trials = TRIALS_MAP.get(model_name, 50)
                study = _make_study(n_trials=n_trials, seed=SEED + fold_id)

                print(f"  [{model_name}] fold {fold_id}: tuning ({n_trials} trials)...")

                def objective(trial: optuna.Trial) -> float:
                    params = _sample_from_space(trial, space)
                    pipe = clone(model_pipe)
                    pipe.set_params(**params)

                    scores: List[float] = []
                    for step, (itr, iva) in enumerate(inner_splits):
                        X_i_tr = X_tr_outer.iloc[itr]
                        X_i_va = X_tr_outer.iloc[iva]
                        y_i_tr_log = y_tr_outer_log.iloc[itr]
                        y_i_va_log = y_tr_outer_log.iloc[iva]

                        m = clone(pipe).fit(X_i_tr, y_i_tr_log)
                        pred_log = m.predict(X_i_va)
                        rmse_log = _rmse(y_i_va_log.to_numpy(), pred_log)
                        scores.append(-rmse_log)

                        trial.report(float(np.median(scores)), step=step)
                        if trial.should_prune():
                            raise optuna.TrialPruned()

                    return float(np.median(scores))

                study.optimize(objective, n_trials=n_trials, show_progress_bar=True)
                best_params = study.best_trial.params

                print(
                    f"    best median(-RMSE) = {study.best_value:.4f} "
                    f"(trial {study.best_trial.number})"
                )

                with open(params_file, "w", encoding="utf-8") as f:
                    json.dump(best_params, f, ensure_ascii=False, indent=2)
                print(f"  [{model_name}] fold {fold_id}: tuned and saved params → {params_file.name}")

            # Fit tuned model on full outer-train and evaluate on outer-test
            pipe_best = clone(model_pipe)
            pipe_best.set_params(**best_params)
            pipe_best.fit(X_tr_outer, y_tr_outer_log)

            pred_log_outer = pipe_best.predict(X_te_outer)
            r2_outer = float(r2_score(y_te_outer_log.to_numpy(), pred_log_outer))
            rmse_outer = float(_rmse(y_te_outer_log.to_numpy(), pred_log_outer))

            outer_records.append(
                {
                    "Fold": fold_id,
                    "Model": model_name,
                    "k_global": int(k_global),
                    "n_feats": int(len(selected_feats)),
                    "R2_outer": r2_outer,
                    "RMSE_outer": rmse_outer,
                }
            )

    outer_df = pd.DataFrame(outer_records)
    metrics_path = MODELSEL_METRICS_DIR / "nested_outer_fold_metrics.csv"
    outer_df.to_csv(metrics_path, index=False)

    outer_summary = (
        outer_df.groupby("Model", as_index=False)
        .agg(
            R2_outer_mean=("R2_outer", "mean"),
            R2_outer_std=("R2_outer", "std"),
            RMSE_outer_mean=("RMSE_outer", "mean"),
            RMSE_outer_std=("RMSE_outer", "std"),
        )
        .sort_values("R2_outer_mean", ascending=False)
    )

    summary_path = MODELSEL_SUMMARY_DIR / "nested_outer_summary.csv"
    outer_summary.to_csv(summary_path, index=False)

    best_model_name = outer_summary["Model"].iloc[0]

    print("\n=== Nested CV summary (outer folds) ===")
    print(outer_summary)
    print(f"\n[Best model] {best_model_name}")

    return outer_df, outer_summary, best_model_name
