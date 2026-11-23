"""Grouped nested search, outer predictions and recoverable model fitting."""

from msr_qspr import artifacts, config, progress
from typing import Dict, Any
import warnings
from pathlib import Path
import numpy as np
import pandas as pd
import optuna
from optuna.distributions import IntDistribution, FloatDistribution, CategoricalDistribution
from sklearn.base import clone
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import LinearRegression, Ridge
from optuna.exceptions import ExperimentalWarning
from msr_qspr.config import SEED
from msr_qspr.estimators import SolubilityAnchoredRegressor, standardized_ridge
from msr_qspr.features import inner_rankings, outer_ranking
from msr_qspr.preprocessing import build_preprocessor, descriptor_names, project_experiment_predictors
from msr_qspr.preprocessing import (_build_model_pipeline_for_fold,
                                    _build_trial_invariant_preprocessor,
                                    _build_post_preprocessing_pipeline)
from msr_qspr.config import _n_jobs

def _rmse(a: np.ndarray, b: np.ndarray) -> float:
    return artifacts.root_mean_squared_error(a, b)


def _make_estimators() -> Dict[str, Any]:
    """Return base estimators for model comparison."""
    from xgboost import XGBRegressor

    estimators = {
        "Random Forest":     RandomForestRegressor(random_state=SEED, n_jobs=_n_jobs()),
        "Gradient Boosting": GradientBoostingRegressor(random_state=SEED, n_iter_no_change=None),
        "XGBoost":           XGBRegressor(random_state=SEED, n_estimators=200, n_jobs=_n_jobs(), tree_method="hist"),
        "Linear Regression": LinearRegression(),
        "Ridge":             Ridge(),
    }
    return estimators


def base_estimator(experiment):
    """Unfitted estimator of one candidate; anchored candidates wrap their learner.

    An anchored Ridge standardizes inside the anchor so the offset reads raw inputs.
    """
    estimator = _make_estimators()[experiment.estimator]
    if experiment.target_anchor == 'none':
        return estimator
    if experiment.estimator == 'Ridge':
        estimator = standardized_ridge()
    return SolubilityAnchoredRegressor(estimator=estimator)


def estimator_parameters(experiment, params):
    """Map declared search parameters to pipeline parameter paths.

    Unanchored candidates use their declared names unchanged. Anchored candidates
    route learner parameters through the anchor; the optional initial Ridge penalty
    becomes a standardized Ridge initial estimator, and None keeps the constant one.
    """
    if experiment.target_anchor == 'none' and not experiment.linear_initialization:
        return dict(params)
    values = dict(params)
    learner = 'model__'
    result = {}
    if experiment.target_anchor != 'none':
        if 'model__anchor_coefficient' not in values:
            raise ValueError('Anchored candidates require a selected anchor coefficient.')
        result['model__anchor_coefficient'] = values.pop('model__anchor_coefficient')
        learner = 'model__estimator__' + ('ridge__' if experiment.estimator == 'Ridge' else '')
    if experiment.linear_initialization:
        if 'model__initial_ridge_alpha' not in values:
            raise ValueError('Ridge-initialized boosting requires a selected initial penalty.')
        alpha = values.pop('model__initial_ridge_alpha')
        result[learner + 'init'] = None if alpha is None else standardized_ridge(alpha)
    for key, value in values.items():
        if not key.startswith('model__'):
            raise ValueError(f'Unknown search parameter: {key}')
        result[learner + key.removeprefix('model__')] = value
    return result


# Hyperparameter spaces

def _make_study(n_trials: int, seed: int, *, search_space=None, pruning=True) -> optuna.Study:
    """Create Optuna study with TPE sampler and median pruner."""
    optuna.logging.set_verbosity(optuna.logging.WARNING)
    if search_space is not None:
        return optuna.create_study(direction='maximize',
                                  sampler=optuna.samplers.GridSampler(search_space, seed=seed),
                                  pruner=optuna.pruners.NopPruner())
    n_startup = min(n_trials, 20)

    # Suppress Optuna ExperimentalWarning for multivariate/group options
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=ExperimentalWarning)
        sampler = optuna.samplers.TPESampler(
            seed=seed,
            n_startup_trials=n_startup,
            multivariate=True,
            group=True,
        )

    pruner = optuna.pruners.MedianPruner(
        n_startup_trials=max(10, n_startup // 2),
        n_warmup_steps=2,
        interval_steps=1,
    )
    if not pruning:
        pruner = optuna.pruners.NopPruner()

    return optuna.create_study(direction="maximize", sampler=sampler, pruner=pruner)


def _sample_from_space(trial: optuna.Trial, space: Dict[str, Any]) -> Dict[str, Any]:
    """Sample a hyperparameter dict from an Optuna distribution space."""
    params: Dict[str, Any] = {}
    for key, dist in space.items():
        if key == "model__alpha" and "model__loss" in space and params.get("model__loss") != "huber":
            continue
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


def _search_history_valid(path, signature, expected, ranked):
    if not artifacts.cache_matches(path, signature):
        return False
    try:
        history = pd.read_csv(path)
        required = {'number', 'state', 'value'} | ({'params_k'} if ranked else set())
        return (required <= set(history) and len(history) == expected
                and history.number.tolist() == list(range(expected))
                and history.state.eq('COMPLETE').all()
                and np.isfinite(history.loc[history.state.eq('COMPLETE'), 'value']).all()
                and (not ranked or history.loc[history.state.eq('COMPLETE'), 'params_k'].notna().all()))
    except (OSError, ValueError, KeyError):
        return False


def tune_model_specification(X_train, y_train, *, experiment, base_est, space,
                             rankings, inner_splits, validation_axes, forced, categorical_cols,
                             log10_cols, n_trials, trial_seed, history_file,
                             signature, k_fold=None, resume=True, reuse_preprocessing=True):
    """Search one specification using only supplied training/inner-validation data.

    Ranked configurations jointly optimize integer molecular k and estimator
    parameters. Fixed configurations retain their predictor panel. The objective
    is worst relative pooled RMSE across two axes, with fold-training mean references.
    Trial histories reuse the same checksum-bound persistence as grouped CV.
    """
    if experiment is None:
        raise ValueError("An explicit experiment specification is required.")
    if n_trials != experiment.trial_budget:
        raise ValueError('Search budget differs from the declared candidate specification.')
    if rankings is not None and (len(rankings) != len(inner_splits) or
            any(len(names) < experiment.k_max or len(names) != len(set(names))
                or not set(names) <= set(X_train) for names in rankings)):
        raise ValueError('Requested molecular count range exceeds an eligible training ranking; no clipping.')
    axes = tuple(config.PRIMARY_SETTINGS)
    if (len(validation_axes) != len(inner_splits)
            or set(validation_axes) != set(axes)
            or any(validation_axes.count(axis) != config.GROUPED_INNER_FOLDS for axis in axes)):
        raise ValueError('Tuning requires three grouped folds on each primary chemical axis.')
    if (len(X_train) != len(y_train) or not X_train.index.equals(y_train.index)
            or not np.isfinite(np.asarray(y_train, dtype=float)).all()):
        raise ValueError('Tuning predictors and finite training responses must align.')
    for axis in axes:
        coverage = np.zeros(len(X_train), dtype=int)
        for (train, validation), label in zip(inner_splits, validation_axes):
            if label != axis:
                continue
            train, validation = np.asarray(train), np.asarray(validation)
            if (not len(train) or not len(validation)
                    or not np.issubdtype(train.dtype, np.integer)
                    or not np.issubdtype(validation.dtype, np.integer)
                    or len(set(train)) != len(train) or len(set(validation)) != len(validation)
                    or set(train) & set(validation)
                    or set(train) | set(validation) != set(range(len(X_train)))):
                raise ValueError('Tuning splits must partition the available training population.')
            coverage[validation] += 1
        if not np.all(coverage == 1):
            raise ValueError('Tuning must validate every training row once per chemical axis.')
    reference_errors = {}
    for axis in axes:
        residuals = [y_train.iloc[va].to_numpy() - float(y_train.iloc[tr].mean())
                     for (tr, va), label in zip(inner_splits, validation_axes) if label == axis]
        pooled = np.concatenate(residuals)
        reference_errors[axis] = max(_rmse(pooled, np.zeros_like(pooled)),
                                     config.TRAINING_POLICY['reference_floor'])
    grid = ({key: list(dist.choices) for key, dist in space.items()}
            if experiment.search_method == 'grid' else None)
    study = (None if experiment.search_method == 'none' else
             _make_study(n_trials=n_trials, seed=trial_seed, search_space=grid, pruning=False))
    checkpoint = Path(history_file).with_suffix('.study.joblib')
    search_contract = artifacts.fingerprint({
        'experiment': experiment.to_dict(), 'space': {k: str(v) for k, v in space.items()},
        'trial_seed': trial_seed,
        # XGBoost uses NaN as a meaningful missing-value parameter.
        'base_estimator': {key: repr(value) for key, value in base_est.get_params().items()},
        'inner_splits': [(tr.tolist(), va.tolist()) for tr, va in inner_splits],
        'validation_axes': validation_axes, 'rankings': rankings, 'forced': forced,
        'categorical_cols': categorical_cols, 'log10_cols': log10_cols,
        'training_policy': config.TRAINING_POLICY,
        'trial_invariant_preprocessing_reuse': bool(reuse_preprocessing)})
    if (resume and checkpoint.exists()
            and checkpoint.with_name(checkpoint.name + '.provenance.json').exists()
            and not artifacts.cache_matches(checkpoint, signature)):
        raise ValueError('Optuna checkpoint checksum or scientific identity differs; recovery is refused.')
    if resume and study is not None and artifacts.cache_matches(checkpoint, signature):
        import joblib
        saved = joblib.load(checkpoint)
        study = saved['study']
        if (saved.get('signature') != signature or saved.get('budget') != n_trials
                or saved.get('search_contract') != search_contract
                or len(study.trials) > n_trials
                or [t.number for t in study.trials] != list(range(len(study.trials)))
                or any(t.state.name != 'COMPLETE' or t.value is None or not np.isfinite(t.value)
                       or not np.isfinite(t.user_attrs.get('mean_relative_rmse', np.nan))
                       for t in study.trials)):
            raise ValueError('Incompatible or unfinished Optuna checkpoint; completion is refused.')

    recovered_trials = 0 if study is None else len(study.trials)

    prepared_inner = None
    if reuse_preprocessing and study is not None and recovered_trials < n_trials:
        prepared_inner = []
        invariant = _build_trial_invariant_preprocessor(categorical_cols, log10_cols)
        for itr, iva in inner_splits:
            X_i_tr = X_train.iloc[itr]
            X_i_va = X_train.iloc[iva]
            y_i_tr_log = y_train.iloc[itr]
            fitted = clone(invariant).fit(X_i_tr, y_i_tr_log)
            prepared_inner.append((fitted.transform(X_i_tr), fitted.transform(X_i_va)))

    def checkpoint_completed_trial(current, trial):
        # Store the sampler RNG with completed trials; an interrupted trial is replayed.
        if all(t.state.name == 'COMPLETE' for t in current.trials):
            import joblib
            with artifacts.atomic_artifact(checkpoint) as temporary:
                joblib.dump({'signature': signature, 'budget': n_trials,
                             'search_contract': search_contract, 'study': current}, temporary)
            artifacts.record_cache(checkpoint, signature)
            best = min(current.trials, key=lambda t: (-t.value, t.user_attrs['mean_relative_rmse'], t.number))
            progress.update(phase='tuning', completed_trials=len(current.trials),
                            best_objective=best.value, best_trial=best.number,
                            axis_rmse=best.user_attrs['axis_rmse'], current_trial=None)
            if len(current.trials) % 15 == 0 or len(current.trials) == n_trials:
                progress.log(f'Search: {len(current.trials)}/{n_trials}; best trial {best.number}: '
                             f'{best.value:+.4g}; axis RMSE {best.user_attrs["axis_rmse"]}')

    if study is None:
        progress.log('No hyperparameter tuning required.')
    else:
        progress.update(phase='tuning', completed_trials=len(study.trials),
                        recovered_trials=recovered_trials, checkpoint=str(checkpoint.resolve()),
                        search_signature=signature)
        if study.trials:
            recovered_best = min(study.trials, key=lambda t: (-t.value, t.user_attrs['mean_relative_rmse'], t.number))
            progress.update(best_objective=recovered_best.value, best_trial=recovered_best.number,
                            axis_rmse=recovered_best.user_attrs['axis_rmse'])
        progress.log(f'{experiment.search_method.upper()} search | {len(study.trials)}/{n_trials} '
                     f'trials complete | {len(inner_splits)} inner folds')

    def objective(trial: optuna.Trial) -> float:
        progress.update(phase='tuning', current_trial=trial.number)
        params = _sample_from_space(trial, space)
        k_trial = (trial.suggest_int('k', experiment.k_min, experiment.k_max, step=1)
                   if rankings is not None else k_fold)
        residuals = {axis: [] for axis in axes}
        numerical_warnings = []
        for step, (itr, iva) in enumerate(inner_splits):
            features = list(X_train) if rankings is None else rankings[step][:k_trial] + forced
            pipe = (_build_post_preprocessing_pipeline(base_est, features)
                    if prepared_inner is not None else
                    _build_model_pipeline_for_fold(base_est, features, categorical_cols, log10_cols))
            pipe.set_params(**estimator_parameters(experiment, params))
            if prepared_inner is None:
                X_i_tr = X_train.iloc[itr]
                X_i_va = X_train.iloc[iva]
            else:
                X_i_tr, X_i_va = prepared_inner[step]
            y_i_tr_log = y_train.iloc[itr]
            y_i_va_log = y_train.iloc[iva]

            m = clone(pipe).fit(X_i_tr, y_i_tr_log)
            pred_log = m.predict(X_i_va)
            if (np.shape(pred_log) != np.shape(y_i_va_log)
                    or not np.isfinite(pred_log).all()):
                raise ValueError('Inner prediction shape or finite-value contract failed.')
            axis = validation_axes[step]
            residuals[axis].append(y_i_va_log.to_numpy() - pred_log)
            fold_rmse = _rmse(y_i_va_log.to_numpy(), pred_log)
            trial.set_user_attr(f'inner_fold_{step + 1}_rmse', fold_rmse)
            # Operational flag only: retain the prediction, objective and completed trial.
            if fold_rmse / reference_errors[axis] > 1e6:
                message = (f'trial {trial.number}, {axis} inner fold {step % config.GROUPED_INNER_FOLDS + 1}: '
                           f'RMSE {fold_rmse:.6g} exceeds 1e6 times the pooled reference RMSE')
                numerical_warnings.append(message)
                progress.numerical_warning(message)


        errors = {axis: _rmse(np.concatenate(residuals[axis]),
                              np.zeros(sum(len(r) for r in residuals[axis]))) for axis in axes}
        relative = [errors[axis] / reference_errors[axis] - 1 for axis in axes]
        if not np.isfinite(relative).all():
            raise ValueError('Normalized inner RMSE exceeds the finite floating-point range.')
        trial.set_user_attr('axis_rmse', errors)
        trial.set_user_attr('mean_relative_rmse', sum(value / len(relative) for value in relative))
        trial.set_user_attr('numerical_warnings', numerical_warnings)
        trial.set_user_attr('warnings', progress.trial_warnings())
        return -max(relative)

    if study is None:
        best_params = {}
    else:
        try:
            study.optimize(objective, n_trials=n_trials - len(study.trials),
                           callbacks=[checkpoint_completed_trial], show_progress_bar=False)
        finally:
            with artifacts.atomic_artifact(history_file) as temporary:
                history = study.trials_dataframe()
                history['recovered_from_previous_execution'] = (history.number < recovered_trials
                    if 'number' in history else pd.Series(dtype=bool))
                history.to_csv(temporary, index=False)
        best_trial = min(study.trials, key=lambda t: (-t.value, t.user_attrs['mean_relative_rmse'], t.number))
        best_params = dict(best_trial.params)
        if rankings is not None:
            k_fold = best_params.pop('k')
    search_record = {
        'method': experiment.search_method,
        'sampler': None if study is None else type(study.sampler).__name__,
        'trial_seed': trial_seed, 'trial_budget': n_trials,
        'best_trial': None if study is None else best_trial.number,
        'k_bounds': None if rankings is None else [experiment.k_min, experiment.k_max],
        'k_step': None if rankings is None else 1,
        'objective': dict(config.TRAINING_POLICY), 'reference_errors': reference_errors,
        'validation_axes': list(validation_axes), 'space': {k: str(v) for k, v in space.items()},
        'n_startup_trials': min(n_trials, 20)
                             if experiment.search_method == 'tpe' else None,
        'pruner': 'NopPruner',
        'recovery': 'atomic_completed_trial_checkpoint_including_sampler_state',
    }
    if study is not None:
        if not _search_history_valid(history_file, signature, n_trials,
                                     rankings is not None):
            # The just-written file needs its receipt before validation.
            artifacts.record_cache(history_file, signature)
        if not _search_history_valid(history_file, signature, n_trials,
                                     rankings is not None):
            raise ValueError('Incomplete Optuna trial history.')
        artifacts.record_cache(history_file, signature)
    return best_params, k_fold, search_record


def _rank_outer_fold(X_train, y_train, experiment, inner_splits, fold_id,
                     ranking_directory, ranking_context, metadata):
    """Rank eligible molecular predictors using this outer training fold only."""
    registered = set(descriptor_names())
    ranking_X = X_train.loc[:, [c for c in X_train if c in registered]].copy()
    if not ranking_X.index.equals(metadata.index) or not metadata.index.is_unique:
        raise ValueError('Training structures must align with predictor rows.')
    ranking_X.attrs['training_structures'] = {
        'Cont_': metadata.contaminant_group.to_dict(),
        'Surf_': metadata.active_structure_group.to_dict()}
    preprocessor, _, _ = build_preprocessor(ranking_X)
    strategy = {'strategy': experiment.ranking_strategy}
    rankings = inner_rankings(
        ranking_X, y_train, preprocessor, inner_splits,
        ranking_directory / f'fold{fold_id}_inner_rankings.json', SEED + fold_id,
        scientific_context=ranking_context, **strategy,
    )
    ranking = outer_ranking(ranking_X, y_train, preprocessor, fold_id,
                            cache_dir=ranking_directory, scientific_context=ranking_context, **strategy)
    lists = [ranking['feature'].astype(str).tolist()] + rankings
    if any(len(names) != len(set(names)) or not set(names) <= set(ranking_X) for names in lists):
        raise ValueError('Molecular rankings contain duplicate or ineligible features.')
    available = dict(zip(['outer'] + [f'inner_{i+1}' for i in range(len(rankings))], map(len, lists)))
    if experiment.k_max > min(available.values()):
        raise ValueError(f'Requested molecular k range [{experiment.k_min}, {experiment.k_max}] '
                         f'exceeds training-only available descriptors {available}; bounds were not changed.')
    signature = artifacts.artifact_signature(ranking_X, y_train, {
        'rankings': lists, 'inner_splits': [(tr.tolist(), va.tolist()) for tr, va in inner_splits],
        'preprocessor': preprocessor.get_params(), **strategy,
    }, scientific_context=ranking_context)
    return rankings, ranking, signature


def _run_grouped_procedure(X, y, metadata, experiment, folds, directory, context):
    """Evaluate the same fitted procedure for training-bound diagnostics."""
    from msr_qspr.framework import TrainingBank, _write
    from msr_qspr.validation import grouped_inner_splits
    run = artifacts.RunPaths(directory, context)
    bank = TrainingBank(run.directory / 'shared' / 'training_bank', [experiment], context)
    rows = []
    for fold in folds:
        train, held = fold.train_pos, fold.validation_pos
        grouped_inner_splits(fold, train, held, metadata.row_id)  # fails on infeasible inner folds
        local_X, local_y, local_meta = (table.iloc[train].reset_index(drop=True)
                                       for table in (X, y, metadata))
        model = bank.fit(local_X, local_y, local_meta)[experiment.name]
        query = project_experiment_predictors(X.iloc[held], experiment)
        model.fit_metadata_ = dict(model.fit_metadata_,
            partition_membership_hashes=artifacts.nested_membership_hashes(fold),
            environment=artifacts.environment_signature(),
            experiment=experiment.to_dict(), scenario=fold.scenario,
            repetition=fold.repetition, outer_fold=fold.fold,
            outer_seed=fold.outer_seed, inner_seed=fold.inner_seed)
        prediction = model.predict(query)
        axis = config.VALIDATION_SETTINGS[fold.scenario]['group_column']
        rows.extend(dict(Scenario=fold.scenario, Repetition=fold.repetition, OuterFold=fold.fold,
            Model=experiment.estimator, Configuration=experiment.name, observation_id=obs,
            canonical_group=str(group), actual_log10_MSR=float(actual), predicted_log10_MSR=float(pred))
            for obs, group, actual, pred in zip(metadata.iloc[held].row_id,
                metadata.iloc[held][axis], y.iloc[held], prediction))
        model_path = run.fold(fold.fold) / 'model.joblib'
        signature = artifacts.artifact_signature(local_X, local_y, model.fit_metadata_, scientific_context=context)
        _write(model_path, model, signature)
    table = pd.DataFrame(rows)
    output = run.metrics / 'outer_oof_predictions.csv'
    signature = artifacts.artifact_signature(X, y, {'experiment': experiment.to_dict()}, scientific_context=context)
    _write(output, table, signature)
    return table, run.directory
