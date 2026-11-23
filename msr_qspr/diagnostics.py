"""Training-bound reliability, complete-procedure randomization and fitted-model interpretation.

All errors and response ranges use log10(MSR), with dimensionless molar MSR.
"""

from msr_qspr import artifacts
from msr_qspr import reporting
from pathlib import Path
import hashlib
import json
import joblib
import numpy as np
import pandas as pd
from msr_qspr import config
from msr_qspr.artifacts import fingerprint, membership_fingerprint
from msr_qspr.preprocessing import validate_predictor_matrix
from msr_qspr.reliability import INTERPRETATION_LIMITATIONS, fitted_input_space, run_applicability_domain, coverage_error_report, descriptor_definitions, fitted_shap
from msr_qspr.models import _run_grouped_procedure


def computational_budget(experiment=None, *, n_permutations=None):
    """Count the authoritative training procedure without fitting or loading targets."""
    from msr_qspr.framework import computational_budget as training_budget
    result = training_budget()
    result['comparison_oof_records'] = result['oof_prediction_records']
    if experiment is not None:
        units, inner_folds = result['outer_evaluations'], result['tuning_folds_per_scope']
        per_outer = inner_folds * experiment.trial_budget + 1
        ranking = int(experiment.selection_policy == 'ranked_molecular' and experiment.ranking_strategy == 'rf')
        linear_ranking = int(experiment.selection_policy == 'ranked_molecular' and experiment.ranking_strategy == 'linear')
        result.update(per_permutation_estimator_fits=units * per_outer,
            per_permutation_ranking_rf_fits=units * ranking * (inner_folds + 1) * config.B_REPEATS,
            per_permutation_ranking_linear_fits=units * linear_ranking * (inner_folds + 1),
            final_estimator_fits=4 * per_outer,
            final_ranking_rf_fits=4 * ranking * (inner_folds + 1) * config.B_REPEATS,
            final_ranking_linear_fits=4 * linear_ranking * (inner_folds + 1))
        if n_permutations is not None:
            if type(n_permutations) is not int or n_permutations < 1:
                raise ValueError('Permutation count must be a positive integer.')
            result.update(n_permutations=n_permutations,
                permutation_estimator_fits=result['per_permutation_estimator_fits'] * n_permutations,
                permutation_ranking_rf_fits=result['per_permutation_ranking_rf_fits'] * n_permutations,
                permutation_ranking_linear_fits=result['per_permutation_ranking_linear_fits'] * n_permutations)
    return result


def run_y_randomization(*, X, y_log, development, comparison_directory, selection,
                        study, output_directory=None, workspace=None):
    """One selected family, fresh ranking/k/tuning per fold, one whole-population shuffle.

    Global target shuffling is a descriptive negative control. Chemical clustering
    and observational exchangeability limit significance interpretation; no exact
    permutation-test p-value is computed. Failures never form partial OOF scores.
    """
    plan = prepare_y_randomization(X=X, y_log=y_log, development=development,
        comparison_directory=comparison_directory, selection=selection, study=study,
        output_directory=output_directory, workspace=workspace)
    for unit in randomization_units(plan):
        run_y_randomization_unit(plan, *unit)
    return aggregate_y_randomization(plan)


def prepare_y_randomization(*, X, y_log, development, comparison_directory, selection,
                            study, output_directory=None, workspace=None):
    """Verify the stage once and fix its identity, permutations and validation plans.

    Complete, single-unit and aggregation executions share this plan, so every
    (permutation, scenario, repetition) unit runs the same procedure in any process.
    Units are written to the workspace; the verified aggregate is published in
    output_directory (default: artifacts/y_randomization).
    """
    validate_predictor_matrix(X)
    from msr_qspr.study import (check_run_metadata, experiment_specification)
    from msr_qspr.validation import validation_memberships, _canonical_cv_metadata, VALIDATION_SETTINGS
    identity = check_run_metadata(study, 'y_randomization')
    reporting.validate_single_selection(comparison_directory, selection, study)
    spec = study['stages']['y_randomization']
    count, seed = spec['n_permutations'], spec['seed']
    if type(count) is not int or count < 1 or type(seed) is not int:
        raise ValueError('Permutation count and seed require explicit scientific evidence.')
    if spec['settings'] != list(VALIDATION_SETTINGS):
        raise ValueError('Randomization must preserve the three established validation settings.')
    if (membership_fingerprint(development.row_id) != study['scientific']['development_sha256']
            or not X.index.equals(development.index) or not y_log.index.equals(X.index)
            or len(X) != len(y_log) or not np.isfinite(y_log).all()):
        raise ValueError('Randomization must use the common development population exclusively.')
    experiment = experiment_specification(study['stages']['final_model']['tuning_policy']['experiment'])
    if experiment.name != selection['configuration']:
        raise ValueError('Randomization configuration differs from the single selected winner.')
    root = artifacts.artifact_directory('y_randomization_units', study_id=identity, workspace=workspace)
    output = Path(output_directory or artifacts.artifact_directory('y_randomization'))
    from msr_qspr.validation import workflow_partitions
    from msr_qspr.data import corpus_metadata
    partitions = workflow_partitions(corpus_metadata())
    plans = {setting: validation_memberships(development, setting, plan=partitions)[0]
             for setting in spec['settings']}
    return {'identity': identity, 'root': root, 'output': output, 'count': count, 'seed': seed,
            'settings': list(spec['settings']), 'experiment': experiment, 'plans': plans,
            'metadata': {setting: _canonical_cv_metadata(development, setting) for setting in spec['settings']},
            'scientific_sha256': fingerprint(study['scientific']),
            'X': X, 'y_log': y_log, 'development': development}


def randomization_units(plan):
    """Every (permutation, scenario, repetition) unit, in canonical order."""
    return [(number, setting, repetition) for number in range(plan['count'])
            for setting in plan['settings']
            for repetition in sorted({f.repetition for f in plan['plans'][setting]})]


def _shuffled_response(plan, number):
    y_log, development = plan['y_log'], plan['development']
    indices = np.random.default_rng(np.random.SeedSequence([plan['seed'], number])).permutation(len(y_log))
    target = pd.Series(y_log.to_numpy()[indices], index=y_log.index, name=y_log.name)
    mapping = pd.DataFrame({'observation_id': development.row_id,
                            'target_source_id': development.row_id.to_numpy()[indices],
                            'permuted_log10_MSR': target})
    permutation_identity = fingerprint({'stage': plan['identity'], 'seed': plan['seed'], 'number': number,
                                        'mapping': mapping.to_dict('records')})
    directory = plan['root'] / f'permutation_{number:05d}_{permutation_identity[:12]}'
    return target, mapping, permutation_identity, directory


def _unit_paths(plan, permutation_identity, directory, setting, repetition):
    unit = directory / f'{setting}_rep{repetition}'
    identity = fingerprint({'stage': plan['identity'], 'permutation_sha256': permutation_identity,
                            'scenario': setting, 'repetition': repetition})
    return unit, identity


def _permutation_scores(prediction, target, development):
    """Require complete aligned OOF predictions of the permuted response."""
    if (not prediction.observation_id.is_unique or
            set(prediction.observation_id) != set(development.row_id)):
        raise ValueError('Permutation OOF membership is incomplete.')
    matched = prediction.set_index('observation_id').loc[development.row_id]
    if (not np.isfinite(matched.predicted_log10_MSR).all() or
            not np.allclose(matched.actual_log10_MSR, target, rtol=1e-12, atol=1e-12)):
        raise ValueError('Permutation response or prediction provenance differs.')
    error = matched.predicted_log10_MSR.to_numpy() - target.to_numpy()
    return dict(status='complete', RMSE_OOF=artifacts.root_mean_squared_error(
                    target, matched.predicted_log10_MSR),
                MAE_OOF=float(np.mean(np.abs(error))), n_observations=len(error))


def run_y_randomization_unit(plan, number, setting, repetition):
    """Run one permutation's grouped outer validation for one scenario repetition.

    Units are independent and may run concurrently in separate processes. The
    shared permutation mapping is written once under a lock; the unit publishes
    its record and OOF predictions under its own checksummed receipt.
    """
    if (number, setting, repetition) not in randomization_units(plan):
        raise ValueError('Unknown Y-randomization unit for this stage identity.')
    target, mapping, permutation_identity, directory = _shuffled_response(plan, number)
    path = directory / 'response_permutation.csv'
    with artifacts.exclusive_artifact(path):
        if not artifacts.cache_matches(path, permutation_identity):
            with artifacts.atomic_artifact(path) as temporary:
                mapping.to_csv(temporary, index=False)
            artifacts.record_cache(path, permutation_identity)
    experiment = plan['experiment']
    folds = [f for f in plan['plans'][setting] if f.repetition == repetition]
    context = {'target_kind': 'shuffled', 'permutation_sha256': permutation_identity,
        'permutation_number': number, 'seed': plan['seed'], 'study_sha256': plan['identity'],
        'scientific_sha256': plan['scientific_sha256'],
        'scenario': setting, 'repetition': repetition, 'experiment': experiment.to_dict()}
    record = dict(Permutation=number, PermutationIdentity=permutation_identity,
                  Scenario=setting, Repetition=repetition)
    unit, identity = _unit_paths(plan, permutation_identity, directory, setting, repetition)
    try:
        prediction, _ = _run_grouped_procedure(plan['X'], target, plan['metadata'][setting], experiment,
                                               folds, unit, context)
        record.update(_permutation_scores(prediction, target, plan['development']))
    except Exception as exc:
        record.update(status='failed', failure=f'{type(exc).__name__}: {exc}')
    path = unit / 'randomization_record.json'
    with artifacts.atomic_artifact(path) as temporary:
        temporary.write_text(json.dumps(record, indent=2) + '\n')
    artifacts.record_cache(path, identity)
    members = [path] + ([unit / 'outer_oof_predictions.csv'] if record['status'] == 'complete' else [])
    receipt = unit / 'randomization_complete.json'
    artifacts.publish_completed_unit(receipt, identity, members)
    artifacts.record_cache(receipt, identity)
    return record


def aggregate_y_randomization(plan):
    """Publish canonical metrics only from every verified unit of the stage identity."""
    rows = []
    for number, setting, repetition in randomization_units(plan):
        target, _, permutation_identity, directory = _shuffled_response(plan, number)
        if not artifacts.cache_matches(directory / 'response_permutation.csv', permutation_identity):
            raise ValueError(f'Permutation {number} mapping is missing or unverified.')
        unit, identity = _unit_paths(plan, permutation_identity, directory, setting, repetition)
        path = unit / 'randomization_record.json'
        try:
            record = json.loads(path.read_text())
        except (OSError, ValueError):
            record = {}
        prediction = unit / 'outer_oof_predictions.csv'
        members = [path] + ([prediction] if record.get('status') == 'complete' else [])
        if (not artifacts.completed_unit_valid(unit / 'randomization_complete.json', identity, members)
                or {k: record.get(k) for k in ('Permutation', 'PermutationIdentity', 'Scenario', 'Repetition')}
                != dict(Permutation=number, PermutationIdentity=permutation_identity,
                        Scenario=setting, Repetition=repetition)):
            raise ValueError(f'Incomplete or unverified Y-randomization unit: {(number, setting, repetition)}.')
        if record['status'] == 'complete':
            scores = _permutation_scores(pd.read_csv(prediction), target, plan['development'])
            if any(record.get(k) != v for k, v in scores.items()):
                raise ValueError(f'Recorded scores differ from verified OOF predictions: {(number, setting, repetition)}.')
        rows.append(record)
    results = pd.DataFrame(rows)
    aggregates = []
    for (number, setting), frame in results.groupby(['Permutation', 'Scenario'], sort=True):
        complete = frame.status.eq('complete').all()
        aggregates.append({'Permutation': number, 'Scenario': setting,
            'status': 'complete' if complete else 'failed', 'n_repetitions': len(frame),
            'mean_repetition_RMSE': float(frame.RMSE_OOF.mean()) if complete else None,
            'limitations': 'Dependent repetitions; descriptive global target shuffling, not an exact significance test.'})
    output, identity = plan['output'], plan['identity']
    tables = [('repetition_metrics.csv', results), ('permutation_metrics.csv', pd.DataFrame(aggregates))]
    artifacts.protect_stage_destination(output, identity, [name for name, _ in tables])
    for name, table in tables:
        path = output / name
        with artifacts.atomic_artifact(path) as temporary:
            table.to_csv(temporary, index=False)
        artifacts.record_cache(path, identity)
    return {'repetitions': results, 'permutations': pd.DataFrame(aggregates),
            'budget': computational_budget(plan['experiment'], n_permutations=plan['count'])}


def run_shap(*, bundle_directory, X, y_log,
             output_directory, study, selection, comparison_directory,
             n_repeats=config.INTERPRETATION_POLICY['n_repeats'],
             seed=config.INTERPRETATION_POLICY['seed']):
    """Final-model interpretation: no training; importance is descriptive on development rows."""
    validate_predictor_matrix(X)
    from msr_qspr.study import check_run_metadata
    stage_identity = check_run_metadata(study, 'interpretation')
    expected_policy = {'n_repeats': n_repeats, 'seed': seed, 'role': 'descriptive_development'}
    if study['stages']['interpretation']['policy'] != expected_policy:
        raise ValueError('Interpretation settings differ from the recorded scientific specification.')
    reporting.validate_single_selection(comparison_directory, selection, study)
    from msr_qspr.final import load_inference_bundle
    from sklearn.inspection import permutation_importance
    pipe, bundle = load_inference_bundle(bundle_directory)
    if bundle.get('analysis_support', {}).get('interpretation') != 'available':
        raise ValueError('This prediction bundle has no validated interpretation reference.')
    if (bundle['experiment']['name'] != selection['configuration']
            or bundle['scientific_sha256'] != fingerprint(study['scientific'])
            or len(X) != len(bundle['development_ids'])):
        raise ValueError('Interpretation must describe the one selected development model.')
    if (len(X) != len(y_log) or not X.index.equals(y_log.index) or not np.isfinite(y_log).all()
            or type(n_repeats) is not int or n_repeats < 1):
        raise ValueError('Interpretation requires aligned finite responses and positive repeats.')
    values = fitted_input_space(pipe, X)
    if (not np.allclose(values, pipe.interpretation_background_, rtol=1e-12, atol=1e-12)
            or not np.allclose(y_log, pipe.reliability_.y_log, rtol=1e-12, atol=1e-12)):
        raise ValueError('Interpretation population differs from the serialized development reference.')
    features = list(values)
    importance = permutation_importance(pipe.steps[-1][1], values.to_numpy(), y_log.to_numpy(),
        scoring='neg_root_mean_squared_error', n_repeats=n_repeats, random_state=seed)
    table = descriptor_definitions(features)
    table['permutation_RMSE_increase'] = importance.importances_mean
    table['permutation_std'] = importance.importances_std
    fold_selections = reporting.load_verified_comparison(comparison_directory, study)['fold_selections']
    frames = fold_selections.loc[fold_selections.Configuration.eq(bundle['experiment']['name'])]
    if frames.empty or frames.duplicated(['Scenario', 'Repetition', 'OuterFold']).any():
        raise ValueError('Feature stability requires unique external-fold selections for the selected model.')
    stability = []
    for setting, group in frames.groupby('Scenario'):
        selections = [json.loads(s) if isinstance(s, str) else s for s in group.selected_features]
        for feature in sorted(set(features) | {f for names in selections for f in names}):
            stability.append({'Scenario': setting, 'feature': feature, 'n_outer_folds': len(group),
                              'selection_frequency': sum(feature in names for names in selections) / len(group),
                              'selected_final': feature in features})
    shap_result = fitted_shap(pipe, X, None)
    offset_summary = None
    if shap_result['status'] == 'available':
        table['mean_absolute_SHAP'] = np.mean(np.abs(shap_result['values']), axis=0)
        components = shap_result.get('components')
        if components is not None:
            table['mean_absolute_linear_contribution'] = np.mean(np.abs(components['linear']), axis=0)
            table['mean_absolute_nonlinear_contribution'] = np.mean(np.abs(components['nonlinear']), axis=0)
            offset = np.asarray(components['imposed_offset'], dtype=float)
            offset_summary = {'definition': 'anchor_coefficient * log10(WaterSolubility_g_L / Cont_MW_g_mol)',
                              'anchor_coefficient': components['anchor_coefficient'],
                              'mean': float(offset.mean()), 'mean_absolute': float(np.abs(offset).mean()),
                              'role': 'imposed physical offset; excluded from learned attributions'}
    provenance = {'model_sha256': bundle['model_sha256'], 'study_sha256': stage_identity,
        'target': 'log10_MSR',
        'n_repeats': n_repeats, 'seed': seed, 'shap_status': shap_result['status'],
        'shap_reason': shap_result.get('reason'), 'imposed_offset': offset_summary,
        'interpretation_population': 'descriptive_development', 'limitations': INTERPRETATION_LIMITATIONS,
        'effective_features': features, 'environment': artifacts.environment_signature()}
    identity = artifacts.artifact_signature(X, y_log, provenance, stage='interpretation')
    output = Path(output_directory or artifacts.artifact_directory('interpretation'))
    tables = [('interpretation.csv', table), ('selection_stability.csv', pd.DataFrame(stability))]
    if shap_result['status'] == 'available':
        tables.append(('shap_values.csv', _observation_attributions(shap_result, bundle['development_ids'])))
    artifacts.protect_stage_destination(output, identity, [name for name, _ in tables] + ['interpretation.json'])
    for name, data in tables:
        path = output / name
        with artifacts.atomic_artifact(path) as temporary:
            data.to_csv(temporary, index=False)
        artifacts.record_cache(path, identity)
    with artifacts.atomic_artifact(output / 'interpretation.json') as temporary:
        temporary.write_text(json.dumps(provenance, indent=2) + '\n')
    artifacts.record_cache(output / 'interpretation.json', identity)
    return {'importance': table, 'stability': pd.DataFrame(stability), 'shap': shap_result,
            'provenance': provenance}


def _observation_attributions(shap_result, observation_ids):
    """Per-observation learned attributions with fitted-space feature values.

    Columns: observation_id, base_value (includes any imposed solubility offset),
    imposed_offset (anchored estimators only), shap__<feature> and value__<feature>.
    """
    features = shap_result['features']
    frame = pd.DataFrame({'observation_id': list(observation_ids),
                          'base_value': np.asarray(shap_result['base_values'], dtype=float)})
    components = shap_result.get('components')
    if components is not None:
        frame['imposed_offset'] = np.asarray(components['imposed_offset'], dtype=float)
    attributions = pd.DataFrame(shap_result['values'], columns=['shap__' + f for f in features])
    values = pd.DataFrame(shap_result['feature_values'], columns=['value__' + f for f in features])
    return pd.concat([frame, attributions, values], axis=1)


def run_reliability(*, X, y_log, development, comparison_directory,
                    selection, study, output_directory=None, workspace=None, run_root=None):
    """Assess the selected candidate's outer models with nested grouped cross-fitting.

    Comparison models and OOF artifacts are read-only. Independent residual
    analyses are confined to the fixed selected configuration. Residual
    calibration references are written to the workspace; observation statuses
    and coverage-error tables are published in output_directory
    (default: artifacts/reliability).
    """
    from msr_qspr.study import (check_run_metadata, experiment_specification, COMPARISON, run_fingerprint)
    from msr_qspr.validation import _canonical_cv_metadata
    from msr_qspr.preprocessing import project_experiment_predictors
    from msr_qspr.residuals import nested_calibration_plan, generate_nested_residual_atlas
    identity = check_run_metadata(study, 'reliability')
    reporting.validate_single_selection(comparison_directory, selection, study)
    evidence = reporting.load_verified_comparison(comparison_directory, study)
    if (membership_fingerprint(development.row_id) != study['scientific']['development_sha256']
            or not X.index.equals(development.index) or not y_log.index.equals(X.index)
            or not np.isfinite(y_log).all()):
        raise ValueError('Reliability analysis requires the exact aligned development population.')
    experiment = experiment_specification(study['stages']['final_model']['tuning_policy']['experiment'])
    if experiment.name != selection['configuration']:
        raise ValueError('Reliability configuration differs from the single selected winner.')
    plan = {'nested_plan': json.loads(json.dumps(nested_calibration_plan(experiment)))}
    from msr_qspr.validation import workflow_partitions
    from msr_qspr.data import corpus_metadata
    partitions = workflow_partitions(corpus_metadata())
    expected = evidence['oof_predictions']
    expected = expected.loc[expected.Configuration.eq(experiment.name)]
    if expected.empty:
        raise ValueError('Selected configuration has no verified OOF predictions.')
    runs = {}
    comparison_identity = run_fingerprint(study, COMPARISON)
    if run_root is None:
        sources = json.loads((Path(comparison_directory) / 'aggregation_sources.json').read_text())
        run_root = Path(comparison_directory) / sources['runs_directory']
    run_root = Path(run_root)
    for path in artifacts.candidate_provenances(run_root):
        info = json.loads(path.read_text())
        if info.get('study_sha256') != comparison_identity or not info.get('experiment'):
            continue
        key = (info['scenario'], info['experiment']['name'], info['repetition'])
        if key in runs:
            raise ValueError('Ambiguous completed run identity in reliability analysis.')
        runs[key] = (path.parent, info)
    all_rows, specifications = [], []
    calibration = artifacts.artifact_directory('reliability_calibration', study_id=identity, workspace=workspace)
    output = Path(output_directory or artifacts.artifact_directory('reliability'))
    names = ['observation_status.csv', 'coverage_error.csv', 'reference_specifications.json']
    artifacts.protect_stage_destination(output, identity, names)
    for key, queries in expected.groupby(['Scenario', 'Configuration', 'Repetition'], sort=True):
        directory, info = runs[key]
        spec = experiment_specification(info['experiment'])
        predictors = project_experiment_predictors(X, spec)
        digest = hashlib.sha256(pd.util.hash_pandas_object(predictors, index=True).values.tobytes())
        digest.update(json.dumps({'columns': list(predictors), 'dtypes': predictors.dtypes.astype(str).tolist(),
                                  'attributes': predictors.attrs}, sort_keys=True, default=str).encode())
        if digest.hexdigest() != info['predictor_matrix_sha256']:
            raise ValueError('Reliability predictors differ from the completed comparison matrix.')
        metadata = _canonical_cv_metadata(development, key[0])
        grouped = {f.fold: f for f in partitions.folds(development, key[0]) if f.repetition == key[2]}
        for fold, query in queries.groupby('OuterFold', sort=True):
            path = directory / f'fold-{fold:02d}' / 'model.joblib'
            pipe = joblib.load(path)
            fit = pipe.fit_metadata_
            regenerated = grouped[fold]
            if fit.get('partition_membership_hashes') != artifacts.nested_membership_hashes(regenerated):
                raise ValueError('Saved model partition lineage differs from regenerated MILP folds.')
            train_pos, query_pos = regenerated.train_pos, regenerated.validation_pos
            training = metadata.iloc[train_pos].reset_index(drop=True)
            assessed = metadata.iloc[query_pos].reset_index(drop=True)
            X_train = predictors.iloc[train_pos].reset_index(drop=True)
            X_query = predictors.iloc[query_pos].reset_index(drop=True)
            target = y_log.iloc[train_pos].reset_index(drop=True)
            if not np.allclose(query.set_index('observation_id').loc[assessed.row_id].actual_log10_MSR,
                               y_log.iloc[query_pos], rtol=0, atol=1e-10):
                raise ValueError('Reliability response identity differs from completed OOF evidence.')
            provenance = {'model_identity': fingerprint(spec.to_dict()),
                'representation_identity': fingerprint({'features': fit['selected_features'],
                                                        'ranking': fit['ranking_identity']}),
                'scientific_sha256': fingerprint(study['scientific']), 'study_sha256': identity}
            unit = f'{key[0]}__{key[1]}__rep{key[2]}__fold{fold}'
            threshold = {'fit_ids': training.row_id.tolist(),
                         'quantile': config.AD_POLICY['local_error_quantile']}
            threshold['policy_sha256'] = fingerprint(threshold)
            atlas = generate_nested_residual_atlas(X_train=X_train, y_train=target,
                training_metadata=training, query_metadata=assessed, scenario=key[0],
                experiment=spec, provenance=provenance, threshold_policy=threshold,
                output_directory=calibration / unit,
                reference_folds=plan['nested_plan']['reference_folds'], seed=plan['nested_plan']['seed'])
            reference, statuses = run_applicability_domain(pipe=pipe, X_train=X_train,
                y_train_log=target, training_metadata=training, X_query=X_query,
                query_metadata=assessed, scenario=key[0], provenance=provenance,
                residual_atlas=atlas, threshold_policy=threshold)
            matched = query.set_index('observation_id').loc[statuses.observation_id]
            if not np.allclose(statuses.predicted_log10_MSR, matched.predicted_log10_MSR, rtol=0, atol=1e-10):
                raise ValueError('Reliability predictions differ from signed comparison OOF predictions.')
            statuses = statuses.assign(Scenario=key[0], Configuration=key[1], Repetition=key[2], OuterFold=fold,
                actual_log10_MSR=matched.actual_log10_MSR.to_numpy(),
                canonical_group=matched.canonical_group.to_numpy(),
                SurfactantType=assessed.SurfactantType.to_numpy(),
                ContaminantType=assessed.ContaminantType.to_numpy())
            all_rows.append(statuses)
            specifications.append(dict(unit=unit, model_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
                                       reference=reference.specification()))
    status = pd.concat(all_rows, ignore_index=True)
    reports = []
    for key, frame in status.groupby(['Scenario', 'Configuration', 'Repetition'], sort=True):
        report = coverage_error_report(frame, frame.actual_log10_MSR, frame.canonical_group,
                                       frame[['SurfactantType', 'ContaminantType']])
        reports.append(report.assign(Scenario=key[0], Configuration=key[1], Repetition=key[2]))
    coverage = pd.concat(reports, ignore_index=True)
    for name, table in [('observation_status.csv', status), ('coverage_error.csv', coverage)]:
        serial = table.copy()
        for col in ('neighbor_ids', 'neighbor_groups', 'neighbor_distances'):
            if col in serial:
                serial[col] = serial[col].map(json.dumps)
        path = output / name
        with artifacts.atomic_artifact(path) as temporary:
            serial.to_csv(temporary, index=False)
        artifacts.record_cache(path, identity)
    path = output / 'reference_specifications.json'
    with artifacts.atomic_artifact(path) as temporary:
        temporary.write_text(json.dumps(specifications, indent=2) + '\n')
    artifacts.record_cache(path, identity)
    return {'observation_status': status, 'coverage_error': coverage}
