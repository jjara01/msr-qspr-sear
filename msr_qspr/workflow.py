"""Run MSR study experiments with namespaced artifacts."""

from __future__ import annotations

from msr_qspr import artifacts, progress
import json
import os
import hashlib
from copy import copy
from pathlib import Path
import numpy as np
import pandas as pd
from msr_qspr import config # noqa: E402
from msr_qspr import data as data_prep, preprocessing as split_preproc




def _load_dev_full(temp_policy: str):
    if temp_policy != 'temp_off':
        raise ValueError('The primary study excludes temperature predictors.')
    from msr_qspr.abraham import attach_descriptors
    df_full = data_prep.prepare_full_dataframe(
        use_cache=True,
        force_recompute=os.getenv("MSR_FORCE_RECOMPUTE", "0") == "1",
    )
    df_full = attach_descriptors(df_full, require_curated=True)
    X, y, meta = split_preproc.build_design_matrices(df_full, include_temperature=False)
    y_log = pd.Series(np.log10(y.to_numpy(dtype=float)), name="log10_MSR")
    return {
        "df_full": df_full,
        "X": X.reset_index(drop=True),
        "y_log": y_log.reset_index(drop=True),
        "meta": meta.reset_index(drop=True),
    }


from msr_qspr import models
from msr_qspr.framework import TrainingBank, _write
from msr_qspr.preprocessing import project_experiment_predictors
from sklearn.metrics import r2_score


def run_primary_experiment(scenario, temp_policy='temp_off', *, experiments=None,
                           diagnostic_k_grid=None, study=None, run_root=None,
                           fold_number=None, repetition=None, candidate=None):
    """Evaluate registered candidate-fold units and persist signed grouped artifacts."""
    from msr_qspr.study import check_run_metadata, COMPARISON, run_metadata
    if run_root is not None:
        Path(run_root)
    study = study or run_metadata()
    if temp_policy != 'temp_off' or diagnostic_k_grid is not None:
        raise ValueError('The comparison excludes temperature and conditional complexity curves.')
    if experiments is not None and tuple(experiments) != config.study_experiments():
        raise ValueError('Comparison requires the complete declared candidate registry.')
    if scenario not in config.VALIDATION_SETTINGS:
        raise ValueError('Unknown grouped validation setting.')
    from msr_qspr.validation import workflow_partitions, validation_memberships, _canonical_cv_metadata
    from msr_qspr.data import corpus_metadata
    identity = check_run_metadata(study, COMPARISON)
    registry = config.study_experiments()
    if candidate is not None and candidate not in {experiment.name for experiment in registry}:
        raise ValueError('Unknown registered candidate.')
    requested = tuple(experiment for experiment in registry
                      if candidate is None or experiment.name == candidate)
    root = Path(run_root or artifacts.artifact_directory('runs', study_id=identity))
    if root.exists():
        for path in artifacts.candidate_provenances(root):
            if json.loads(path.read_text()).get('study_sha256') != identity:
                raise ValueError('Execution root contains a different experiment; use an isolated directory.')
    prepared = _load_dev_full('temp_off')
    X, y = prepared['X'], prepared['y_log']
    metadata = _canonical_cv_metadata(prepared['df_full'], scenario)
    plan = workflow_partitions(corpus_metadata())
    folds, hashes = validation_memberships(prepared['df_full'], scenario, plan=plan)
    context = dict(scientific_identity=config.scientific_identity(),
        candidate_registry_sha256=artifacts.candidate_registry_fingerprint(),
        study_sha256=identity, scientific_sha256=artifacts.fingerprint(study['scientific']),
        partition_sha256=plan.sha256, scenario=scenario,
        holdout_sha256=study['scientific']['holdout_sha256'], development_ids=metadata.row_id.tolist(),
        development_population_sha256=artifacts.membership_fingerprint(metadata.row_id.tolist()),
        validation_plan_sha256=hashes, grouping_policy='active_structure',
        group_column=config.VALIDATION_SETTINGS[scenario]['group_column'],
        predictor_contract=X.attrs.get('predictor_contract'), temperature_policy='temp_off',
        counterion_policy='counterion_strip', environment=artifacts.environment_signature())
    bank = TrainingBank(root / 'shared' / 'training_bank', registry, context)
    for rep in range(1, len(config.VALIDATION_SETTINGS[scenario]['seeds']) + 1):
        if repetition is not None and rep != repetition:
            continue
        active = [f for f in folds if f.repetition == rep]
        runs = {}
        for experiment in requested:
            projected = project_experiment_predictors(X, experiment)
            digest = hashlib.sha256(pd.util.hash_pandas_object(projected, index=True).values.tobytes())
            digest.update(json.dumps({'columns': list(projected), 'dtypes': projected.dtypes.astype(str).tolist(),
                                     'attributes': projected.attrs}, sort_keys=True, default=str).encode())
            info = dict(context, repetition=rep, experiment=experiment.to_dict(),
                predictor_matrix_sha256=digest.hexdigest(), predictor_columns=list(projected),
                predictor_dtypes=projected.dtypes.astype(str).tolist())
            run_id = artifacts.artifact_signature(projected, y, {}, scientific_context=info)
            run = artifacts.RunPaths(artifacts.candidate_directory(
                root, scenario, rep, experiment.name, run_id), info)
            manifest = run.directory / 'run_provenance.json'
            with artifacts.exclusive_artifact(manifest):
                expected_manifest = json.loads(json.dumps(dict(info, identity_sha256=run_id), default=str))
                if manifest.exists():
                    if json.loads(manifest.read_text()) != expected_manifest:
                        raise ValueError('Existing candidate provenance differs from current execution inputs.')
                    if not artifacts.cache_matches(manifest, run_id):
                        if manifest.with_name(manifest.name + '.provenance.json').exists():
                            raise ValueError('Existing candidate provenance failed verification; preserve it and investigate.')
                        artifacts.record_cache(manifest, run_id)
                else:
                    _write(manifest, expected_manifest, run_id)
            runs[experiment.name] = (run, run_id)
        for fold in active:
            if fold_number is not None and fold.fold != fold_number:
                continue
            train, validation = fold.train_pos, fold.validation_pos
            local_X, local_y, local_meta = (table.iloc[train].reset_index(drop=True)
                                            for table in (X, y, metadata))
            for experiment in requested:
                run, run_id = runs[experiment.name]
                folder = run.fold(fold.fold)
                # Final writer barrier for this candidate-fold namespace. Different
                # candidates in the same outer fold remain fully concurrent.
                with artifacts.exclusive_artifact(folder / 'candidate_writer'), \
                        progress.fold(root, scenario, rep, fold.fold, candidate=experiment.name):
                    if not _fold_complete(run, run_id, experiment, fold, metadata):
                        fitted = list(bank.iter_fit(local_X, local_y, local_meta,
                                                    candidates={experiment.name}))
                        if len(fitted) != 1 or fitted[0][0] != experiment.name:
                            raise ValueError('Training bank did not return the requested candidate exactly once.')
                        _, fitted_model = fitted[0]
                        model = copy(fitted_model)
                        progress.update(phase='outer_prediction')
                        query = project_experiment_predictors(X.iloc[validation].copy(), experiment)
                        prediction = np.asarray(model.predict(query), dtype=float)
                        if prediction.shape != (len(validation),) or not np.isfinite(prediction).all():
                            raise ValueError('Invalid external predictions.')
                        progress.log(f'{experiment.name} | outer prediction complete')
                        fitted_spec = dict(model.fit_metadata_, experiment=experiment.to_dict(),
                            scientific_context=run.context, environment=artifacts.environment_signature(),
                            scenario=scenario, repetition=rep, outer_fold=fold.fold,
                            outer_seed=fold.outer_seed, inner_seed=fold.inner_seed,
                            partition_membership_hashes=artifacts.nested_membership_hashes(fold),
                            raw_predictors=run.context['predictor_columns'],
                            canonical_group_column=context['group_column'])
                        signature = artifacts.fingerprint({'run': run_id, 'fold': artifacts.nested_membership_hashes(fold)})
                        effective_count = config.effective_input_count(experiment, fitted_spec['selected_features'])
                        fitted_spec.update(signature=signature, effective_input_count=effective_count)
                        model.fit_metadata_ = fitted_spec
                        predictions = pd.DataFrame(dict(Scenario=scenario, Repetition=rep, OuterFold=fold.fold,
                            Model=experiment.estimator, Configuration=experiment.name,
                            observation_id=metadata.iloc[validation].row_id.tolist(),
                            canonical_group=metadata.iloc[validation][context['group_column']].tolist(),
                            actual_log10_MSR=np.asarray(y.iloc[validation]), predicted_log10_MSR=prediction))
                        metric = dict(Fold=fold.fold, Model=experiment.estimator, Configuration=experiment.name,
                            PredictorFamily=experiment.predictor_family, Scenario=scenario, Repetition=rep,
                            n_validation=len(validation), n_feats=effective_count, k_selected=fitted_spec['molecular_k'],
                            R2_outer=float(r2_score(y.iloc[validation], prediction)),
                            RMSE_outer=models._rmse(y.iloc[validation], prediction))
                        artifacts.validate_fold_metrics(predictions, metric)
                        contents = {
                            folder / 'model.joblib': model,
                            folder / 'selection.json': fitted_spec,
                            folder / 'parameters.json': dict(model_params=fitted_spec['model_params'],
                                k=fitted_spec['molecular_k'], search=fitted_spec['search'], experiment=experiment.to_dict()),
                            folder / 'predictions.csv': predictions,
                            folder / 'metrics.json': metric}
                        if experiment.search_method != 'none':
                            # Keep the genuine trial table; bind the copied file to the external unit.
                            source = bank.directory / fitted_spec['training_scope_id'] / experiment.name / 'search_trials.csv'
                            history = folder / 'trials.csv'
                            contents[history] = pd.read_csv(source)
                        progress.update(phase='publishing')
                        for path, value in contents.items():
                            _write(path, value, signature)
                        receipt = folder / 'complete.json'
                        artifacts.publish_completed_unit(receipt, signature, list(contents))
                        artifacts.record_cache(receipt, signature)
                        if not _fold_complete(run, run_id, experiment, fold, metadata):
                            raise ValueError('Published candidate fold did not verify.')
                        progress.update(phase='published', fold_directory=str(folder.resolve()),
                                        final_parameters=fitted_spec['model_params'])
                        progress.log(f'{experiment.name} | candidate-fold artifacts and receipt verified')
                # Completion may arrive in any fold order. The last verified fold
                # to acquire this lock publishes the repetition aggregate.
                with artifacts.exclusive_artifact(run.directory / 'repetition_writer'):
                    if all(_fold_complete(run, run_id, experiment, current, metadata)
                           for current in active):
                        _finish_repetition(run, run_id, experiment, active)


def _fold_complete(run, run_id, experiment, fold, metadata):
    folder = run.fold(fold.fold)
    receipt = folder / 'complete.json'
    if not receipt.exists():
        return False
    try:
        signature = json.loads(receipt.read_text())['identity']
    except (ValueError, KeyError, TypeError):
        raise ValueError('A completed fold receipt is malformed; preserve it and investigate.')
    required = [folder / name for name in
                ('model.joblib', 'selection.json', 'parameters.json', 'predictions.csv', 'metrics.json')]
    if experiment.search_method != 'none':
        required.append(folder / 'trials.csv')
    memberships = artifacts.nested_membership_hashes(fold)
    expected = artifacts.fingerprint({'run': run_id, 'fold': memberships})
    if (signature != expected or not artifacts.completed_unit_valid(
            receipt, expected, required, member_identity=expected, allow_missing_sidecar=True)):
        raise ValueError('A completed fold failed integrity checks; preserve it and investigate.')
    selected = json.loads((folder / 'selection.json').read_text())
    manifest_path = run.directory / 'run_provenance.json'
    manifest = json.loads(manifest_path.read_text())
    if (not artifacts.cache_matches(manifest_path, run_id)
            or manifest != json.loads(json.dumps(dict(run.context, identity_sha256=run_id), default=str))
            or selected.get('signature') != signature
            or artifacts.fingerprint(selected.get('experiment')) != artifacts.fingerprint(experiment.to_dict())
            or selected.get('scientific_context') != json.loads(json.dumps(run.context))
            or selected.get('outer_fold') != fold.fold
            or selected.get('partition_membership_hashes') != memberships
            or selected.get('outer_seed') != fold.outer_seed or selected.get('inner_seed') != fold.inner_seed
            or experiment.search_method != 'none' and not models._search_history_valid(
                required[-1], signature, experiment.trial_budget, experiment.selection_policy == 'ranked_molecular')):
        raise ValueError('Completed fold provenance differs from its declared training procedure.')
    predictions = pd.read_csv(folder / 'predictions.csv', dtype={'observation_id': str, 'canonical_group': str})
    if set(predictions.observation_id) != set(fold.validation_ids):
        raise ValueError('Completed predictions differ from the regenerated external fold.')
    from msr_qspr.framework import validate_training_provenance
    validate_training_provenance(selected, metadata.iloc[fold.train_pos].reset_index(drop=True))
    from msr_qspr.status import _fold_state
    state = _fold_state(folder, run.context['scenario'], run.context['repetition'], experiment,
                        fold.fold, run.context['development_ids'], allow_missing_sidecar=True)
    if state['state'] != 'complete':
        raise ValueError('Completed fold prediction verification failed: ' + state.get('reason', state['state']))
    artifacts.validate_fold_metrics(predictions, json.loads((folder / 'metrics.json').read_text()))
    if not artifacts.cache_matches(receipt, expected):
        artifacts.recover_completed_unit(receipt, expected, required, member_identity=expected)
    return True


def _finish_repetition(run, identity, experiment, folds):
    from msr_qspr.reporting import _primary_metrics
    receipt = run.summary / 'repetition_complete.json'
    if receipt.exists():
        from msr_qspr.status import _candidate_state
        state = _candidate_state(run.directory, run.context['scenario'], run.context['repetition'],
                                 experiment, allow_missing_sidecar=True)
        if state['state'] != 'complete' or state.get('identity') != identity:
            raise ValueError('Completed repetition failed verification; preserve it and investigate.')
        if not artifacts.cache_matches(receipt, identity):
            required = [run.directory / 'run_provenance.json',
                        run.metrics / 'outer_oof_predictions.csv', run.metrics / 'nested_outer_fold_metrics.csv',
                        run.summary / 'nested_outer_summary.csv',
                        *(run.fold(f.fold) / 'complete.json' for f in folds)]
            artifacts.recover_completed_unit(receipt, identity, required)
        return
    predictions = pd.concat([pd.read_csv(run.fold(f.fold) / 'predictions.csv',
        dtype={'observation_id': str, 'canonical_group': str}) for f in folds])
    metrics = pd.DataFrame([json.loads((run.fold(f.fold) / 'metrics.json').read_text()) for f in folds])
    summary = pd.DataFrame([dict(Model=experiment.estimator, **_primary_metrics(predictions))])
    contents = {run.metrics / 'outer_oof_predictions.csv': predictions,
                run.metrics / 'nested_outer_fold_metrics.csv': metrics,
                run.summary / 'nested_outer_summary.csv': summary}
    for path, value in contents.items():
        _write(path, value, identity)
    units = [run.fold(f.fold) / 'complete.json' for f in folds]
    receipt = run.summary / 'repetition_complete.json'
    artifacts.publish_completed_unit(receipt, identity, [run.directory / 'run_provenance.json', *contents, *units])
    artifacts.record_cache(receipt, identity)
