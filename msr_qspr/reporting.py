"""Aggregate MSR experiment artifacts into summary tables."""

from __future__ import annotations
from msr_qspr import selection as selection_methods

from msr_qspr import artifacts
import json
import hashlib
import itertools
import os
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.metrics import r2_score
from msr_qspr import config # noqa: E402
from msr_qspr.validation import (ACTIVE_HOLDOUT, VALIDATION_SETTINGS,
                               _canonical_cv_metadata, novelty_from_canonical_metadata)  # noqa: E402
from msr_qspr import study as _study
from msr_qspr.artifacts import fingerprint, membership_fingerprint


COMPARISON_TABLES = (
    'primary_oof_predictions.csv', 'repetition_metrics.csv', 'repetition_summary.csv',
    'group_errors.csv', 'cross_entity_novelty.csv', 'chemical_class_errors.csv',
    'selection_counts.csv', 'fold_descriptor_selections.csv', 'descriptor_frequencies.csv',
    'paired_model_differences.csv',
)


def _read_json(path: Path) -> dict:
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def _protect_aggregation_destination(directory, identity, names):
    """Never replace completed evidence or signed files from another execution."""
    receipt = directory / 'aggregation_complete.json'
    if receipt.exists() and (not artifacts.cache_matches(receipt, identity)
                             or not artifacts.completed_unit_valid(receipt, identity)):
        raise ValueError('Existing aggregation belongs to another execution or failed integrity checks; preserve it.')
    for name in names:
        path = directory / name
        provenance = path.with_name(path.name + '.provenance.json')
        if provenance.exists() and _read_json(provenance).get('signature') != identity:
            raise ValueError('Existing signed aggregation artifacts belong to another execution; use an isolated destination.')


def _registered_experiments(study):
    """Require the complete, exact authoritative candidate registry."""
    declared = study['stages'][_study.COMPARISON]['execution']['experiments']
    if not isinstance(declared, list) or not declared or any(not isinstance(e, dict) for e in declared):
        raise ValueError('Comparison requires a nonempty candidate registry.')
    experiments = config.study_experiments()
    if fingerprint(declared) != fingerprint([e.to_dict() for e in experiments]):
        raise ValueError('Comparison requires the complete registered candidate specifications.')
    return experiments


def _primary_metrics(frame):
    actual = frame.actual_log10_MSR.to_numpy(dtype=float)
    prediction = frame.predicted_log10_MSR.to_numpy(dtype=float)
    error = prediction - actual
    return {'n_observations': len(frame), 'R2_OOF': float(r2_score(actual, prediction)),
            'RMSE_OOF': artifacts.root_mean_squared_error(actual, prediction),
            'MAE_OOF': float(np.mean(np.abs(error)))}


def _outer_memberships(development, scenario, holdout_sha256, *, plan=None):
    """Reproduce authoritative outer and inner identities directly from Python."""
    from msr_qspr.validation import validation_memberships, grouped_cv_membership
    folds, hashes = validation_memberships(development, scenario, plan=plan)
    outer, inner = grouped_cv_membership(development, folds)
    return {'holdout_sha256': holdout_sha256,
            'outputs_sha256': hashes}, outer, inner


def _oof_reporting_tables(oof, selections, scenarios, configurations, n_development):
    """Build descriptive and paired-comparison tables from verified OOF evidence."""
    oof['absolute_error'] = (oof.predicted_log10_MSR - oof.actual_log10_MSR).abs()
    repeat_rows, group_rows, class_rows, novelty_rows = [], [], [], []
    for (scenario, configuration, repetition), frame in oof.groupby(
            ['Scenario', 'Configuration', 'Repetition'], sort=True):
        metrics = _primary_metrics(frame)
        group_error = frame.groupby('canonical_group', sort=True).absolute_error.agg(['size', 'mean'])
        macro = float(group_error['mean'].mean())
        repeat_rows.append({'Scenario': scenario, 'Configuration': configuration,
                            'Repetition': repetition, 'role': VALIDATION_SETTINGS[scenario]['role'],
                            'Estimator': frame.Model.iloc[0],
                            'PredictorFamily': next(e['PredictorFamily'] for e in selections
                                                    if e['Configuration'] == configuration),
                            **metrics, 'MacroGroupMAE': macro,
                            'n_groups': len(group_error)})
        for group, row in group_error.iterrows():
            group_rows.append({'Scenario': scenario, 'Configuration': configuration,
                               'Repetition': repetition, 'canonical_group': group,
                               'n_observations': int(row['size']), 'GroupMAE': float(row['mean'])})
        for field in ('SurfactantType', 'ContaminantType'):
            for label, subset in frame.groupby(field, sort=True):
                class_rows.append({'Scenario': scenario, 'Configuration': configuration,
                                   'Repetition': repetition, 'class_field': field,
                                   'class_value': label, 'n_observations': len(subset),
                                   'ClassMAE': float(subset.absolute_error.mean()),
                                   'ClassRMSE': artifacts.root_mean_squared_error(
                                       subset.actual_log10_MSR, subset.predicted_log10_MSR)})
        strata = ({'both_unseen', 'surfactant_unseen_only'} if scenario == 'unseen_surfactant' else
                  {'both_unseen', 'contaminant_unseen_only'} if scenario == 'unseen_contaminant' else
                  set(frame.novelty_stratum))
        for stratum in sorted(strata):
            subset = frame[frame.novelty_stratum.eq(stratum)]
            error = subset.predicted_log10_MSR - subset.actual_log10_MSR
            novelty_rows.append({'Scenario': scenario, 'Configuration': configuration,
                'Repetition': repetition, 'novelty_stratum': stratum,
                'n_observations': len(subset), 'denominator': len(frame),
                'n_surfactants': subset.active_structure_group.nunique(),
                'n_contaminants': subset.contaminant_group.nunique(),
                'n_publications': subset.publication_group.nunique(),
                'RMSE': artifacts.root_mean_squared_error(subset.actual_log10_MSR,
                            subset.predicted_log10_MSR) if len(subset) else np.nan,
                'MAE': float(np.abs(error).mean()) if len(subset) else np.nan,
                'surfactant_class_counts': json.dumps(subset.SurfactantType.value_counts().to_dict()),
                'contaminant_class_counts': json.dumps(subset.ContaminantType.value_counts().to_dict())})
    per_repeat = pd.DataFrame(repeat_rows)
    stats = per_repeat.groupby(['Scenario', 'Configuration'], sort=True)[
        ['R2_OOF', 'RMSE_OOF', 'MAE_OOF', 'MacroGroupMAE']].agg(['mean', 'min', 'max'])
    stats.columns = [f'{name}_{aggregation}' for name, aggregation in stats.columns]
    repeat_summary = stats.reset_index()
    repeat_summary = repeat_summary.merge(per_repeat.groupby(['Scenario', 'Configuration']).size()
        .rename('n_repetitions').reset_index(), on=['Scenario', 'Configuration'])

    selection_df = pd.DataFrame(selections)
    fold_selections = selection_df.assign(
        descriptors=selection_df.descriptors.map(json.dumps))
    count_rows, frequency_rows = [], []
    for (scenario, configuration), frame in selection_df.groupby(['Scenario', 'Configuration']):
        counts = frame.molecular_k.dropna().astype(int)
        overlap = [len(set(a) & set(b)) / len(set(a) | set(b)) if set(a) | set(b) else np.nan
                   for a, b in itertools.combinations(frame.descriptors, 2)]
        count_rows.append({'Scenario': scenario, 'Configuration': configuration,
                           'Estimator': frame.Estimator.iloc[0],
                           'PredictorFamily': frame.PredictorFamily.iloc[0],
                           'n_outer_evaluations': len(frame),
                           'k_min': int(counts.min()) if len(counts) else np.nan,
                           'k_median': float(counts.median()) if len(counts) else np.nan,
                           'k_max': int(counts.max()) if len(counts) else np.nan,
                           'mean_pairwise_jaccard': float(np.nanmean(overlap))
                           if overlap and np.isfinite(overlap).any() else np.nan})
        frequency = {}
        for names in frame.descriptors:
            for name in set(names):
                frequency[name] = frequency.get(name, 0) + 1
        for descriptor, n in sorted(frequency.items()):
            frequency_rows.append({'Scenario': scenario, 'Configuration': configuration,
                                   'descriptor': descriptor, 'n_selected': n,
                                   'frequency': n / len(frame)})

    paired_rows = []
    for scenario in scenarios:
        for first, second in itertools.combinations(configurations, 2):
            left = oof[(oof.Scenario == scenario) & (oof.Configuration == first)]
            right = oof[(oof.Scenario == scenario) & (oof.Configuration == second)]
            matched = left.merge(right, on=['Repetition', 'observation_id'],
                                 suffixes=('_a', '_b'), how='outer', validate='one_to_one', indicator=True)
            if (len(matched) != len(VALIDATION_SETTINGS[scenario]['seeds']) * n_development or not matched._merge.eq('both').all()
                    or not np.allclose(matched.actual_log10_MSR_a,
                                       matched.actual_log10_MSR_b, rtol=0, atol=1e-10)
                    or not matched.OuterFold_a.eq(matched.OuterFold_b).all()
                    or not matched.canonical_group_a.eq(matched.canonical_group_b).all()):
                raise ValueError(f'Unmatched paired model observations: {scenario}, {first}, {second}')
            for repetition, frame in matched.groupby('Repetition'):
                paired_rows.append({'Scenario': scenario, 'ConfigurationA': first,
                                    'ConfigurationB': second, 'Repetition': repetition,
                                    'n_matched': len(frame),
                                    'MAE_A_minus_B': float((frame.absolute_error_a -
                                                            frame.absolute_error_b).mean()),
                                    'RMSE_A_minus_B': artifacts.root_mean_squared_error(
                                        frame.actual_log10_MSR_a, frame.predicted_log10_MSR_a) -
                                        artifacts.root_mean_squared_error(
                                        frame.actual_log10_MSR_b, frame.predicted_log10_MSR_b)})

    outputs = {'primary_oof_predictions.csv': oof.drop(columns=['absolute_error']),
               'repetition_metrics.csv': per_repeat,
               'repetition_summary.csv': repeat_summary,
               'group_errors.csv': pd.DataFrame(group_rows),
               'cross_entity_novelty.csv': pd.DataFrame(novelty_rows),
               'chemical_class_errors.csv': pd.DataFrame(class_rows),
               'selection_counts.csv': pd.DataFrame(count_rows),
               'fold_descriptor_selections.csv': fold_selections,
               'descriptor_frequencies.csv': pd.DataFrame(frequency_rows),
               'paired_model_differences.csv': pd.DataFrame(paired_rows)}
    return outputs


def aggregate_primary_results(*, run_root=None, output_dir=None, synthetic_memberships=None,
                              scenarios=tuple(VALIDATION_SETTINGS),
                              configurations=None, n_development=390, chemical_metadata=None, study=None):
    """Strictly reassemble signed, complete repeated grouped-CV development predictions.

    Repetitions share observations and folds; reported ranges are descriptive,
    not confidence intervals or independent experimental replications.
    """
    if run_root is None:
        from msr_qspr.study import run_metadata, run_fingerprint, COMPARISON
        identity = run_fingerprint(study or run_metadata(), COMPARISON)
        run_root = artifacts.artifact_directory('runs', study_id=identity)
    run_root = Path(run_root)
    if output_dir is None:
        from msr_qspr.study import run_metadata, run_fingerprint, COMPARISON
        identity = run_fingerprint(study or run_metadata(), COMPARISON)
        output_dir = artifacts.artifact_directory('comparison')
    output_dir = Path(output_dir)
    experiments = _registered_experiments(study) if study is not None else config.study_experiments()
    configurations = tuple(configurations or [e.name for e in experiments])
    if len(set(configurations)) != len(configurations) or not configurations or not scenarios:
        raise ValueError('Expected primary configurations and scenarios must be unique and nonempty.')

    if chemical_metadata is None:
        from msr_qspr.data import corpus_metadata
        chemical_metadata = corpus_metadata()
    if 'SMILES_contaminant' in chemical_metadata:
        chemical_metadata = _canonical_cv_metadata(chemical_metadata, ACTIVE_HOLDOUT)
    chemistry = chemical_metadata.rename(columns={'row_id': 'observation_id'}).copy()
    if (not {'observation_id', 'SurfactantType', 'ContaminantType', 'publication_group',
                 'active_structure_group', 'contaminant_group'} <= set(chemistry)
            or chemistry.observation_id.duplicated().any()
            or chemistry[['SurfactantType', 'ContaminantType']].isna().any().any()
            or chemistry[['SurfactantType', 'ContaminantType']].eq('').any().any()):
        raise ValueError('Chemical-class diagnostic metadata has invalid keys or labels.')

    if n_development == 390:
        from msr_qspr.validation import workflow_partitions
        plan = workflow_partitions(chemical_metadata)
        dev, held = plan.holdout_positions(chemical_metadata)
        development = chemical_metadata.iloc[dev].reset_index(drop=True)
        if synthetic_memberships is not None:
            raise ValueError('Synthetic memberships cannot replace scientific partitions.')
        holdout_sha256 = membership_fingerprint(chemical_metadata.iloc[held].row_id)
        plans = {scenario: _outer_memberships(development, scenario, holdout_sha256, plan=plan)
                 for scenario in scenarios}
        verified = study or _study.run_metadata()
        verified_identity = _study.check_run_metadata(verified, _study.COMPARISON)
    else:
        if (not synthetic_memberships or not chemistry.observation_id.str.startswith('synthetic_').all()):
            raise ValueError('Only explicitly synthetic populations can supply test memberships.')
        plans = synthetic_memberships
        for summary, outer, inner in plans.values():
            for name, table in [('outer_validation_membership.csv', outer), ('inner_validation_membership.csv', inner)]:
                if hashlib.sha256(table.to_csv(index=False).encode()).hexdigest() != summary['outputs_sha256'][name]:
                    raise ValueError('Altered synthetic fold membership.')
        verified = None

    runs = {}
    for provenance_file in artifacts.candidate_provenances(run_root):
        directory = provenance_file.parent
        provenance = _read_json(provenance_file)
        if not provenance.get('experiment'):
            continue  # Shared ranking namespaces have no estimator-specific result.
        key = (provenance.get('scenario'), provenance.get('experiment', {}).get('name'),
               provenance.get('repetition'))
        if key[0] not in scenarios or key[1] not in configurations:
            continue
        identity = provenance.get('identity_sha256')
        if (not isinstance(identity, str) or len(identity) != 64 or
                directory != artifacts.candidate_directory(run_root, key[0], key[2], key[1], identity)
                or key in runs):
            raise ValueError(f'Duplicate or incompatible primary run namespace: {directory}')
        runs[key] = (directory, provenance)

    predictions, selections = [], []
    shared_science = {}
    common_science = None
    predictor_families = {}
    for scenario in scenarios:
        plan_summary, plan, nested_membership = plans[scenario]
        reference = plan[['repetition', 'observation_id', 'outer_fold', 'canonical_group']]
        for configuration in configurations:
            stable_contract = None
            for repetition in range(1, len(VALIDATION_SETTINGS[scenario]['seeds']) + 1):
                key = (scenario, configuration, repetition)
                if key not in runs:
                    raise ValueError(f'Missing complete primary run: {key}')
                directory, info = runs[key]
                receipt = directory / 'repetition_complete.json'
                summary_file = directory / 'nested_outer_summary.csv'
                oof_file = directory / 'outer_oof_predictions.csv'
                metrics_file = directory / 'nested_outer_fold_metrics.csv'
                estimator = info['experiment']['estimator']
                unit_files = [directory / f'fold-{fold:02d}/complete.json'
                              for fold in range(1, 6)]
                if not artifacts.completed_unit_valid(receipt, info['identity_sha256'],
                                                  [directory / 'run_provenance.json',
                                                   summary_file, oof_file, metrics_file, *unit_files]):
                    raise ValueError(f'Incomplete or corrupted repetition: {key}')
                if (info.get('holdout_sha256') != plan_summary['holdout_sha256']
                        or info.get('validation_plan_sha256') != plan_summary['outputs_sha256']
                        or info.get('development_population_sha256') is None
                        or not info.get('study_sha256') or not info.get('scientific_sha256')
                        or not info.get('predictor_matrix_sha256')
                        or info.get('group_column') != VALIDATION_SETTINGS[scenario]['group_column']
                        or not isinstance(info.get('environment'), dict)
                        or info.get('grouping_policy') != 'active_structure'
                        or not info.get('predictor_contract')
                        or not info.get('predictor_columns') or not info.get('predictor_dtypes')
                        or len(info['predictor_columns']) != len(info['predictor_dtypes'])
                        or len(set(info['predictor_columns'])) != len(info['predictor_columns'])
                        or info.get('counterion_policy') != info['experiment']['counterion_policy']
                        or info.get('temperature_policy') != info['experiment']['temperature_policy']
                        or set(info.get('development_ids', [])) != set(
                            reference.observation_id.astype(str))
                        or len(info.get('development_ids', [])) != n_development):
                    raise ValueError(f'Scientific or predictor provenance mismatch: {key}')
                if info['development_population_sha256'] != membership_fingerprint(info['development_ids']):
                    raise ValueError('Training-population fingerprint does not match membership.')
                if verified is not None:
                    specs = {e['name']: e for e in verified['stages'][_study.COMPARISON]['execution']['experiments']}
                    if (info.get('scientific_identity') != config.scientific_identity()
                            or info.get('candidate_registry_sha256') != artifacts.candidate_registry_fingerprint()
                            or info.get('predictor_contract') != config.PREDICTOR_CONTRACT
                            or info.get('partition_sha256') != verified['scientific']['partition_sha256']
                            or info['study_sha256'] != verified_identity
                            or info['scientific_sha256'] != fingerprint(verified['scientific'])
                            or info['development_population_sha256'] != verified['scientific']['development_sha256']
                            or info['experiment'] != specs.get(configuration)
                            or info['environment'] != verified['software']['environment']):
                        raise ValueError('Run differs from verified study or software fingerprints.')
                contract_keys = ('experiment', 'development_population_sha256',
                                 'validation_plan_sha256', 'holdout_sha256', 'environment',
                                 'grouping_policy', 'group_column', 'predictor_contract',
                                 'predictor_columns', 'predictor_dtypes', 'temperature_policy',
                                 'counterion_policy', 'development_ids', 'study_sha256', 'scientific_sha256',
                                 'predictor_matrix_sha256')
                contract = {name: info[name] for name in contract_keys}
                if stable_contract is None:
                    stable_contract = contract
                elif contract != stable_contract:
                    raise ValueError(f'Incompatible repeated run fingerprints: {key}')
                science = {name: info[name] for name in ('holdout_sha256',
                          'development_population_sha256', 'validation_plan_sha256',
                          'environment', 'temperature_policy', 'counterion_policy', 'study_sha256', 'scientific_sha256',
                          'predictor_contract')}
                if scenario in shared_science and shared_science[scenario] != science:
                    raise ValueError(f'Incompatible scenario-wide scientific fingerprints: {key}')
                shared_science[scenario] = science

                common = {name: science[name] for name in science if name != 'validation_plan_sha256'}
                if common_science is not None and common != common_science:
                    raise ValueError(f'Incompatible common population or study fingerprints: {key}')
                common_science = common
                family = info['experiment']['predictor_family']
                predictors = {k: info[k] for k in ('predictor_columns', 'predictor_dtypes', 'predictor_matrix_sha256')}
                if family in predictor_families and predictor_families[family] != predictors:
                    raise ValueError('Incompatible predictors within a representation family.')
                predictor_families[family] = predictors

                frame = pd.read_csv(oof_file, dtype={'observation_id': str, 'canonical_group': str})
                mandatory = {'Scenario', 'Configuration', 'Repetition', 'OuterFold',
                             'Model', 'observation_id', 'canonical_group',
                             'actual_log10_MSR', 'predicted_log10_MSR'}
                if (not mandatory <= set(frame) or len(frame) != n_development or
                        frame.observation_id.duplicated().any() or
                        set(frame.Scenario) != {scenario} or
                        set(frame.Configuration) != {configuration} or
                        set(frame.Repetition) != {repetition} or
                        set(frame.Model) != {estimator} or
                        not np.isfinite(frame[['actual_log10_MSR',
                                               'predicted_log10_MSR']].to_numpy(dtype=float)).all()):
                    raise ValueError(f'Incomplete, duplicate or nonfinite OOF predictions: {key}')
                expected = reference[reference.repetition == repetition]
                actual_membership = frame.rename(columns={'Repetition': 'repetition',
                                                           'OuterFold': 'outer_fold'})
                merged = actual_membership.merge(expected, on=['repetition', 'observation_id'],
                                                 how='outer', validate='one_to_one',
                                                 indicator=True, suffixes=('', '_verified'))
                if (not merged._merge.eq('both').all() or
                        not merged.outer_fold.eq(merged.outer_fold_verified).all() or
                        not merged.canonical_group.eq(merged.canonical_group_verified).all()):
                    raise ValueError(f'OOF group or verified fold mismatch: {key}')
                stored = pd.read_csv(summary_file)
                expected_metrics = _primary_metrics(frame)
                if (len(stored) != 1 or stored.Model.iloc[0] != estimator or
                        any(not np.isclose(stored[field].iloc[0], expected_metrics[field], rtol=0, atol=1e-10)
                            for field in ('R2_OOF', 'RMSE_OOF', 'MAE_OOF'))):
                    raise ValueError(f'OOF summary differs from observation-level predictions: {key}')
                fold_table = pd.read_csv(metrics_file)
                if ('n_feats' not in fold_table or len(fold_table) != 5 or set(fold_table.Fold) != set(range(1, 6))
                        or set(fold_table.Model) != {estimator}):
                    raise ValueError(f'Missing outer-fold metric records: {key}')

                for fold_number, unit in enumerate(unit_files, 1):
                    statement = _read_json(unit)
                    identity = statement.get('identity')
                    base = directory / f'fold-{fold_number:02d}'
                    fold_artifacts = [base / name for name in
                                      ('parameters.json', 'model.joblib', 'selection.json',
                                       'predictions.csv', 'metrics.json')]
                    if info['experiment']['search_method'] != 'none':
                        fold_artifacts.append(base / 'trials.csv')
                    if not artifacts.completed_unit_valid(unit, identity, fold_artifacts, member_identity=identity):
                        raise ValueError(f'Incomplete outer-fold work unit: {key}, fold {fold_number}')
                    selected = _read_json(base / 'selection.json')
                    tuned = _read_json(base / 'parameters.json')
                    fold_frame = pd.read_csv(base / 'predictions.csv', dtype={'observation_id': str})
                    fold_record = _read_json(base / 'metrics.json')
                    artifacts.validate_fold_metrics(fold_frame, fold_record)
                    artifacts.validate_fold_metrics(fold_frame,
                        fold_table.loc[fold_table.Fold.eq(fold_number)].iloc[0].to_dict())
                    expected_frame = frame[frame.OuterFold.eq(fold_number)]
                    if (len(fold_frame) != len(expected_frame)
                            or not {'observation_id', 'actual_log10_MSR', 'predicted_log10_MSR'} <= set(fold_frame)
                            or not fold_frame.observation_id.is_unique
                            or not fold_frame.observation_id.isin(expected_frame.observation_id).all()):
                        raise ValueError(f'Fold predictions differ from repetition OOF: {key}, fold {fold_number}')
                    aligned = fold_frame.set_index('observation_id').loc[
                        expected_frame.observation_id]
                    if not np.allclose(aligned[['actual_log10_MSR', 'predicted_log10_MSR']],
                                       expected_frame[['actual_log10_MSR', 'predicted_log10_MSR']],
                                       rtol=0, atol=1e-10):
                        raise ValueError(f'Fold predictions differ from repetition OOF: {key}, fold {fold_number}')
                    family = info['experiment']['predictor_family']
                    descriptors = selected.get('molecular_descriptors', [])
                    required = list(info['experiment']['required_reference_properties'])
                    expected_descriptors = list(config.PREDICTOR_FAMILIES[family] or [])
                    expected_features = (descriptors + required
                                         if info['experiment']['selection_policy'] == 'ranked_molecular'
                                         else info['predictor_columns'])
                    experiment = _study.experiment_specification(info['experiment'])
                    effective_count = config.effective_input_count(experiment, expected_features)
                    if (selected.get('signature') != identity or
                            selected.get('experiment') != info['experiment'] or
                            selected.get('outer_fold') != fold_number or
                            selected.get('scenario') != scenario or
                            selected.get('repetition') != repetition or
                            selected.get('environment') != info['environment'] or
                            selected.get('raw_predictors') != info['predictor_columns'] or
                            selected.get('mandatory_reference_properties') != required or
                            selected.get('selected_features') != expected_features or
                            selected.get('effective_input_count') != effective_count or
                            tuned.get('experiment') != info['experiment'] or
                            selected.get('model_params') != tuned.get('model_params') or
                            selected.get('search') != tuned.get('search') or
                            selected.get('molecular_k') != tuned.get('k') or
                            not selected.get('ranking_identity') or
                            (info['experiment']['selection_policy'] == 'ranked_molecular' and
                             (type(selected.get('molecular_k')) is not int or
                              not info['experiment']['k_min'] <= selected['molecular_k'] <=
                              info['experiment']['k_max'])) or
                            (info['experiment']['selection_policy'] != 'ranked_molecular' and
                             (selected.get('molecular_k') is not None or
                              descriptors != expected_descriptors)) or
                            (selected.get('molecular_k') is not None and
                             len(descriptors) != selected['molecular_k']) or
                            len(descriptors) != len(set(descriptors)) or
                            len(expected_features) != len(set(expected_features)) or
                            not set(expected_features) <= set(info['predictor_columns']) or
                            fold_table.loc[fold_table.Fold.eq(fold_number), 'n_feats'].iloc[0] != effective_count
                            or fold_record['n_feats'] != effective_count
                            or any(not np.isclose(fold_table.loc[fold_table.Fold.eq(fold_number), field].iloc[0],
                                                  fold_record[field], rtol=0, atol=1e-10)
                                   for field in ('RMSE_outer', 'R2_outer'))):
                        raise ValueError(f'Invalid selection provenance: {key}, fold {fold_number}')
                    membership = expected[expected.outer_fold == fold_number]
                    inner_plan = nested_membership[nested_membership.repetition.eq(repetition) &
                                                   nested_membership.outer_fold.eq(fold_number)]
                    training = expected[expected.outer_fold.ne(fold_number)]
                    hashes = {'outer': artifacts.split_membership_hashes(
                        training.observation_id, membership.observation_id,
                        training.canonical_group.unique(), membership.canonical_group.unique()), 'inner': []}
                    for child in (1, 2, 3):
                        held = inner_plan[inner_plan.inner_fold.eq(child)]
                        trained = inner_plan[inner_plan.inner_fold.ne(child)]
                        hashes['inner'].append(dict(fold=child, **artifacts.split_membership_hashes(
                            trained.observation_id, held.observation_id,
                            trained.canonical_group.unique(), held.canonical_group.unique())))
                    context = {name: value for name, value in info.items() if name != 'identity_sha256'}
                    seed = VALIDATION_SETTINGS[scenario]['seeds'][repetition - 1]
                    if (selected.get('partition_membership_hashes') != hashes
                            or identity != fingerprint({'run': info['identity_sha256'], 'fold': hashes})
                            or selected.get('scientific_context') != context
                            or selected.get('outer_seed') != seed
                            or selected.get('inner_seed') != seed * 1000 + fold_number):
                        raise ValueError('Selected model populations differ from regenerated grouped partitions.')
                    if verified is not None:
                        from msr_qspr.framework import validate_training_provenance
                        local_metadata = development.set_index('row_id').loc[
                            selected.get('training_ids', [])].reset_index()
                        if set(local_metadata.row_id) != set(training.observation_id):
                            raise ValueError('Actual model training population differs from the external training fold.')
                        validate_training_provenance(selected, local_metadata)
                    selection_row = {'Scenario': scenario, 'Configuration': configuration,
                                       'Repetition': repetition, 'OuterFold': fold_number,
                                       'Estimator': estimator, 'PredictorFamily': info['experiment']['predictor_family'],
                                       'molecular_k': selected['molecular_k'],
                                       'n_molecular_descriptors': len(descriptors),
                                       'selected_features': json.dumps(selected['selected_features']),
                                       'n_effective_predictors': effective_count,
                                       'mandatory_reference_properties': json.dumps(
                                           selected['mandatory_reference_properties']),
                                       'descriptors': descriptors}
                    selections.append(selection_row)
                novel = []
                for fold_number, membership in expected.groupby('outer_fold'):
                    held_ids = membership.observation_id.tolist()
                    train_ids = sorted(set(expected.observation_id) - set(held_ids))
                    novelty = novelty_from_canonical_metadata(
                        chemistry.rename(columns={'observation_id': 'row_id'}), train_ids, held_ids, scenario)
                    axis = VALIDATION_SETTINGS[scenario]['group_column']
                    if not novelty.set_index('observation_id')[axis].reindex(membership.observation_id).eq(
                            membership.set_index('observation_id').canonical_group).all():
                        raise ValueError('Canonical chemical metadata disagrees with validation membership.')
                    novel.append(novelty.drop(columns=['SurfactantType', 'ContaminantType']))
                frame = frame.merge(pd.concat(novel), on='observation_id', validate='one_to_one')
                predictions.append(frame)

    oof = pd.concat(predictions, ignore_index=True)
    keys = ['Scenario', 'Configuration', 'Repetition', 'observation_id']
    if oof.duplicated(keys).any() or len(oof) != sum(len(VALIDATION_SETTINGS[s]['seeds']) for s in scenarios) * len(configurations) * n_development:
        raise ValueError('Incomplete cross-run OOF reassembly.')
    for _, matched in oof.groupby('observation_id'):
        if not np.allclose(matched.actual_log10_MSR, matched.actual_log10_MSR.iloc[0],
                           rtol=0, atol=1e-10):
            raise ValueError('Original log target changes across matched model/repetition keys.')

    oof = oof.merge(chemistry[['observation_id', 'SurfactantType', 'ContaminantType']], on='observation_id', how='left', validate='many_to_one')
    if oof[['SurfactantType', 'ContaminantType']].isna().any().any():
        raise ValueError('Missing chemical class for development OOF metadata join.')
    outputs = _oof_reporting_tables(oof, selections, scenarios, configurations, n_development)
    per_repeat = outputs['repetition_metrics.csv']
    count_rows = outputs['selection_counts.csv']
    aggregation_identity = hashlib.sha256(json.dumps({str(k): v[1]['identity_sha256']
                                              for k, v in sorted(runs.items())}, sort_keys=True).encode()).hexdigest()
    with artifacts.exclusive_artifact(output_dir / 'primary_aggregation_writer'):
        _protect_aggregation_destination(output_dir, aggregation_identity,
                                        [*outputs, 'aggregation_sources.json'])
        artifact_paths = []
        for name, table in outputs.items():
            path = output_dir / name
            with artifacts.atomic_artifact(path) as temporary:
                table.to_csv(temporary, index=False)
            artifacts.record_cache(path, aggregation_identity)
            artifact_paths.append(path)
        sources = output_dir / 'aggregation_sources.json'
        with artifacts.atomic_artifact(sources) as temporary:
            temporary.write_text(json.dumps({
                'format_version': 1,
                'runs_directory': os.path.relpath(run_root, output_dir),
            }, indent=2) + '\n')
        artifacts.record_cache(sources, aggregation_identity)
        artifact_paths.append(sources)
        receipt = output_dir / 'aggregation_complete.json'
        artifacts.publish_completed_unit(receipt, aggregation_identity, artifact_paths)
        artifacts.record_cache(receipt, aggregation_identity)
    return {'n_predictions': len(oof), 'n_repetitions': len(per_repeat),
            'n_completed_configurations': len(count_rows), 'output_dir': str(output_dir),
            'aggregation_identity': aggregation_identity}


def load_verified_comparison(directory, study):
    """Rebuild comparison evidence from authoritative completed runs without fitting.

    A hand-built DataFrame is never a production source. Stored aggregation
    receipts and their table identities must match a fresh strict aggregation of
    the verified runs, with runtime scientific/software/plan identities checked.
    Temporary reports are discarded; accepted artifacts are read-only.
    """
    import tempfile
    import pandas as pd
    if not isinstance(directory, (str, Path)):
        raise ValueError('Production selection requires a verified aggregation directory.')
    directory = Path(directory)
    experiments = _registered_experiments(study)
    _study.check_run_metadata(study, _study.COMPARISON)
    receipt = directory / 'aggregation_complete.json'
    record = json.loads(receipt.read_text())
    identity = record.get('identity')
    if not identity or not artifacts.cache_matches(receipt, identity) or not artifacts.completed_unit_valid(receipt, identity):
        raise ValueError('Incomplete or changed aggregation receipt/table fingerprint.')
    sources_path = directory / 'aggregation_sources.json'
    if ('aggregation_sources.json' not in record['artifacts']
            or not artifacts.cache_matches(sources_path, identity)):
        raise ValueError('Aggregation source directories lack matching provenance.')
    sources = json.loads(sources_path.read_text())
    if (sources.get('format_version') != 1 or any(
            not isinstance(sources.get(key), str) or not sources[key] or Path(sources[key]).is_absolute()
            for key in ('runs_directory',))):
        raise ValueError('Aggregation source directories must be portable relative references.')
    table = pd.read_csv(directory / 'repetition_metrics.csv')
    if 'Scenario' not in table or not set(table.Scenario) <= set(VALIDATION_SETTINGS):
        raise ValueError('Unknown or missing comparison settings.')
    scenarios = tuple(s for s in VALIDATION_SETTINGS if s in set(table.Scenario))
    if not set(config.PRIMARY_SETTINGS) <= set(scenarios):
        raise ValueError('Both primary settings are required.')
    with tempfile.TemporaryDirectory(prefix='msr_selection_verification_') as temporary:
        rebuilt = aggregate_primary_results(output_dir=temporary, scenarios=scenarios,
            configurations=[e.name for e in experiments], study=study,
            run_root=(directory / sources['runs_directory']).resolve())
        if rebuilt['aggregation_identity'] != identity:
            raise ValueError('Comparison aggregation differs from the verified run identities.')
        frames = {}
        for name in ('repetition_metrics.csv', 'primary_oof_predictions.csv', 'fold_descriptor_selections.csv'):
            stored = pd.read_csv(directory / name)
            verified = pd.read_csv(Path(temporary) / name)
            if selection_methods.comparison_table_fingerprint(stored) != selection_methods.comparison_table_fingerprint(verified):
                raise ValueError('Comparison table differs from authoritative aggregation identity.')
            frames[name] = verified
    return dict(repetition_metrics=frames['repetition_metrics.csv'],
                oof_predictions=frames['primary_oof_predictions.csv'],
                fold_selections=frames['fold_descriptor_selections.csv'], experiments=experiments,
                development_ids=sorted(frames['primary_oof_predictions.csv'].observation_id.unique()))


def validate_single_selection(table, selection, study, *, synthetic_evidence=None):
    """Verify that the verified declaration identifies the objective common winner.

    Production takes an aggregation directory, not manually supplied summary
    values. The explicit synthetic interface accepts only synthetic-prefixed
    populations and studys without scientific freeze files or plan bindings.
    """
    identity = _study.run_fingerprint(study, 'final_model')
    if study['stages'][_study.COMPARISON].get('common_selection_rule') != config.COMMON_SELECTION_RULE:
        raise ValueError('Selection requires the exact normalized minimax rule.')
    final = study['stages']['final_model']
    rule = study['stages'][_study.COMPARISON]['common_selection_rule']
    if synthetic_evidence is None:
        evidence = load_verified_comparison(table, study)
        table = evidence['repetition_metrics']
    else:
        evidence = synthetic_evidence
        ids = evidence['development_ids']
        if (study['scientific'].get('files_sha256') or study['scientific'].get('fold_memberships') or
                not ids or any(not i.startswith('synthetic_') for i in ids)):
            raise ValueError('Synthetic evidence cannot authorize a scientific population.')
        if selection_methods.comparison_table_fingerprint(table) != selection_methods.comparison_table_fingerprint(evidence['repetition_metrics']):
            raise ValueError('Synthetic comparison table fingerprint mismatch.')
    if membership_fingerprint(evidence['development_ids']) != study['scientific']['development_sha256']:
        raise ValueError('Comparison development identity differs from verified science.')
    declared = study['stages'][_study.COMPARISON]['execution'].get('experiments')
    if declared != [json.loads(json.dumps(e.to_dict())) for e in evidence['experiments']]:
        raise ValueError('Comparison predictor specifications differ from verified execution.')
    expected = {'table_sha256': selection_methods.comparison_table_fingerprint(table),
                'scientific_sha256': fingerprint(study['scientific']),
                'comparison_study_sha256': _study.run_fingerprint(study, _study.COMPARISON),
                'rule_sha256': fingerprint(rule),
                'tuning_policy_sha256': fingerprint(final['tuning_policy'])}
    if (selection != final['selection_record'] or
            any(selection.get(k) != v for k, v in expected.items())):
        raise ValueError('Single-model decision does not match verified comparison and tuning evidence.')
    result = selection_methods.select_common_configuration(**evidence)
    experiment = next(e for e in evidence['experiments'] if e.name == result['configuration'])
    if (selection.get('configuration') != result['configuration'] or
            selection.get('estimator') != experiment.estimator or
            selection.get('predictor_family') != experiment.predictor_family):
        raise ValueError('Declared configuration is not the objective common-selection winner.')
    if final['tuning_policy'] != _study.final_tuning_policy(experiment):
        raise ValueError('Final tuning must use the selected specification and the prescribed grouped policy.')
    return {'configuration': selection['configuration'], 'study_sha256': identity,
            'development_sha256': study['scientific']['development_sha256'],
            'tuning_policy': final['tuning_policy'], 'objective_selection': result}


def select_final_specification(directory, study):
    """Persist the deterministic common winner and its full-development tuning contract."""
    from copy import deepcopy
    directory = Path(directory)
    value = deepcopy(study)
    evidence = load_verified_comparison(directory, value)
    result = selection_methods.select_common_configuration(**evidence)
    experiment = next(e for e in evidence['experiments'] if e.name == result['configuration'])
    policy = _study.final_tuning_policy(experiment)
    selection = {
        'configuration': experiment.name, 'estimator': experiment.estimator,
        'predictor_family': experiment.predictor_family,
        'table_sha256': selection_methods.comparison_table_fingerprint(evidence['repetition_metrics']),
        'scientific_sha256': fingerprint(value['scientific']),
        'comparison_study_sha256': _study.run_fingerprint(value, _study.COMPARISON),
        'rule_sha256': fingerprint(config.COMMON_SELECTION_RULE),
        'tuning_policy_sha256': fingerprint(policy),
    }
    value['stages']['final_model'] = {'tuning_policy': policy, 'selection_record': selection}
    path = Path(directory) / 'selected_model.json'
    with artifacts.exclusive_artifact(path):
        with artifacts.atomic_artifact(path) as temporary:
            temporary.write_text(json.dumps(value, indent=2) + '\n')
        artifacts.record_cache(path, _study.run_fingerprint(value, 'final_model'))
    return value


# ---------------------------------------------------------------------------
# Verified stage outputs
# ---------------------------------------------------------------------------
# Each stage directory holds signed files: <name>.provenance.json records the SHA-256 and
# the identity of the producing execution. Downstream identities derive from the
# selection record comparison/selected_model.json, so a stage is accepted only when its
# files are intact and bound to that record and, where applicable, to the fitted bundle.

from copy import deepcopy as _deepcopy
from dataclasses import dataclass as _dataclass, field as _field
from functools import cached_property as _cached_property, lru_cache as _lru_cache
from msr_qspr.artifacts import InvalidArtifact, MissingArtifact, verify_signed_file

SCENARIOS = tuple(VALIDATION_SETTINGS)
PRIMARY = tuple(config.PRIMARY_SETTINGS)
BOOTSTRAP_RESAMPLES = 4000
# Contrasts that isolate one representation or estimator component at a time.
DESIGNED_CONTRASTS = (
    ('physicochemical_ridge', 'property_linear'),          # compact molecular panel over reference properties
    ('enriched_gb', 'molecular_gb'),                       # reference properties added to ranked descriptors
    ('physicochemical_gb', 'enriched_gb'),                 # compact panel versus ranked descriptors
    ('solubility_anchored_gb', 'physicochemical_gb'),      # solubility anchor
    ('abraham_anchored_gb', 'solubility_anchored_gb'),     # Abraham descriptors with the anchor
    ('abraham_anchored_ridge', 'physicochemical_ridge'),   # Abraham descriptors and anchor, linear
    ('abraham_hybrid_gb', 'abraham_anchored_gb'),          # Ridge initialization of boosting
)


@_dataclass
class StageOutputs:
    """Verified files of one stage, keyed by file name."""
    stage: str
    directory: Path
    files: dict = _field(default_factory=dict)

    def table(self, name, **kwargs):
        return pd.read_csv(self.files[name], **kwargs)

    def json(self, name):
        return json.loads(self.files[name].read_text())


class VerifiedOutputs:
    """Stage outputs under one artifacts directory, verified on first access."""

    def __init__(self, root=None):
        self.root = Path(root or config.ARTIFACTS_DIR)

    def directory(self, stage):
        return artifacts.artifact_directory(stage, root=self.root)

    @_cached_property
    def comparison(self):
        directory = self.directory('comparison')
        receipt = directory / 'aggregation_complete.json'
        if not receipt.is_file():
            raise MissingArtifact(f'No completed comparison in {directory}; run aggregate.')
        record = json.loads(receipt.read_text())
        if not record.get('identity') or not artifacts.completed_unit_valid(receipt, record['identity']):
            raise InvalidArtifact(f'The comparison receipt in {directory} failed verification.')
        return StageOutputs('comparison', directory, {name: directory / name for name in record['artifacts']})

    @_cached_property
    def selection(self):
        """Selection record, checked against its checksum and the published comparison tables."""
        comparison = self.comparison
        path = comparison.directory / 'selected_model.json'
        if not path.is_file():
            raise MissingArtifact(f'No selection record in {comparison.directory}; run select.')
        record = json.loads(path.read_text())
        verify_signed_file(path, _study.run_fingerprint(record, 'final_model'))
        selection = record['stages']['final_model']['selection_record']
        table = comparison.table('repetition_metrics.csv')
        if selection_methods.comparison_table_fingerprint(table) != selection['table_sha256']:
            raise InvalidArtifact('The selection record does not describe the published comparison tables.')
        experiments = config.study_experiments()
        if record['stages'][_study.COMPARISON]['execution']['experiments'] != [
                json.loads(json.dumps(e.to_dict())) for e in experiments]:
            raise InvalidArtifact('The selection record was made for a different candidate registry.')
        oof = comparison.table('primary_oof_predictions.csv', dtype={'observation_id': str})
        result = selection_methods.select_common_configuration(
            table, oof, comparison.table('fold_descriptor_selections.csv'), experiments=experiments,
            development_ids=sorted(oof.observation_id.unique()))
        if result['configuration'] != selection['configuration']:
            raise InvalidArtifact('The recorded selection is not the winner of the published comparison.')
        files = dict(comparison.files, **{'selected_model.json': path})
        return {'record': record, 'selection': selection, 'scores': result['scores'], 'files': files}

    @property
    def selected_configuration(self):
        return self.selection['selection']['configuration']

    def stage_identity(self, stage, *, n_permutations=None):
        record = self.selection['record']
        if stage == 'y_randomization':
            record = _deepcopy(record)
            record['stages']['y_randomization']['n_permutations'] = n_permutations
        return _study.run_fingerprint(record, stage)

    @_cached_property
    def bundle(self):
        """Fitted bundle bound to the selection record; files are checked, not loaded."""
        directory = artifacts.bundle_directory(self.root)
        manifest = directory / 'prediction_ad_bundle.json'
        if not manifest.is_file():
            raise MissingArtifact(f'No fitted bundle in {self.directory("final_model")}; run fit.')
        bundle = json.loads(manifest.read_text())
        if not artifacts.completed_unit_valid(directory / 'final_fit_complete.json', bundle.get('identity_sha256')):
            raise InvalidArtifact(f'The bundle in {directory} is incomplete or failed verification.')
        for name, checksum in bundle.get('artifacts_sha256', {}).items():
            if artifacts.file_sha256(directory / name) != checksum:
                raise InvalidArtifact(f'Bundle file {name} differs from its recorded checksum.')
        if (bundle.get('study_sha256') != self.stage_identity('final_model')
                or bundle['experiment']['name'] != self.selected_configuration):
            raise InvalidArtifact('The fitted bundle does not belong to the selected model.')
        files = {name: directory / name for name in bundle['artifacts_sha256']}
        files['prediction_ad_bundle.json'] = manifest
        outputs = StageOutputs('final_model', directory, files)
        outputs.manifest = bundle
        return outputs

    def _signed_stage(self, stage, names, identity, optional=()):
        directory = self.directory(stage)
        files = {}
        for name in (*names, *optional):
            if name in optional and not (directory / name).exists():
                continue
            verify_signed_file(directory / name, identity)
            files[name] = directory / name
        return StageOutputs(stage, directory, files)

    @_cached_property
    def reliability(self):
        outputs = self._signed_stage('reliability', ('observation_status.csv', 'coverage_error.csv',
                                                     'reference_specifications.json'),
                                     self.stage_identity('reliability'))
        if set(outputs.table('observation_status.csv', usecols=['Configuration']).Configuration) != {
                self.selected_configuration}:
            raise InvalidArtifact('Reliability outputs describe a different configuration.')
        return outputs

    @_cached_property
    def interpretation(self):
        directory = self.directory('interpretation')
        if not (directory / 'interpretation.json').is_file():
            raise MissingArtifact(f'No interpretation in {directory}; run interpret.')
        identity = verify_signed_file(directory / 'interpretation.json')
        outputs = self._signed_stage('interpretation', ('interpretation.json', 'interpretation.csv',
                                                        'selection_stability.csv'), identity,
                                     optional=('shap_values.csv',))
        record = outputs.json('interpretation.json')
        if (record.get('study_sha256') != self.stage_identity('interpretation')
                or record.get('model_sha256') != self.bundle.manifest['model_sha256']):
            raise InvalidArtifact('Interpretation outputs do not describe the selected fitted bundle.')
        return outputs

    @_cached_property
    def y_randomization(self):
        directory = self.directory('y_randomization')
        path = directory / 'permutation_metrics.csv'
        if not path.is_file():
            raise MissingArtifact(f'No aggregated target randomization in {directory}; run randomize --aggregate.')
        numbers = sorted(pd.read_csv(path).Permutation.unique())
        if not numbers or numbers != list(range(len(numbers))):
            raise InvalidArtifact('Target randomization is incomplete.')
        record = self.selection['record']
        record_path = directory / 'selected_model.json'
        if record_path.is_file():
            alternate = json.loads(record_path.read_text())
            verify_signed_file(record_path, _study.run_fingerprint(alternate, 'final_model'))
            compared = [_deepcopy(record), _deepcopy(alternate)]
            for value in compared:
                for stage in ('final_model', 'interpretation', 'reliability', 'y_randomization', 'holdout'):
                    hashes = value.get('software', {}).get('stages', {}).get(stage, {}).get('code_sha256', {})
                    for name in ('msr_qspr/diagnostics.py', 'msr_qspr/reliability.py'):
                        hashes.pop(name, None)
            if compared[0] != compared[1]:
                raise InvalidArtifact('Target randomization selection or scientific settings differ from the selected model.')
            record = alternate
        record = _deepcopy(record)
        record['stages']['y_randomization']['n_permutations'] = len(numbers)
        outputs = self._signed_stage('y_randomization', ('permutation_metrics.csv', 'repetition_metrics.csv'),
                                     _study.run_fingerprint(record, 'y_randomization'))
        if record_path.is_file():
            outputs.files['selected_model.json'] = record_path
        if not outputs.table('permutation_metrics.csv').status.eq('complete').all():
            raise InvalidArtifact('Target randomization contains failed permutations.')
        return outputs

    @_cached_property
    def holdout(self):
        directory = self.directory('holdout')
        receipt = directory / 'holdout_complete.json'
        if not receipt.is_file():
            raise MissingArtifact(f'No holdout evaluation in {directory}.')
        identity = self.stage_identity('holdout')
        if not artifacts.completed_unit_valid(receipt, identity, allow_missing_sidecar=True):
            raise InvalidArtifact('The holdout evaluation does not belong to the selected model or failed verification.')
        outputs = self._signed_stage('holdout', ('predictions.csv', 'metrics.json', 'provenance.json',
                                                 'ad_coverage_errors.csv'), identity)
        provenance = outputs.json('provenance.json')
        if (provenance.get('model_sha256') != self.bundle.manifest['model_sha256']
                or provenance.get('study_sha256') != self.stage_identity('final_model')
                or provenance.get('confirmed') is not True):
            raise InvalidArtifact('The holdout evaluation does not describe the selected fitted bundle.')
        outputs.files['holdout_complete.json'] = receipt
        return outputs

    @_cached_property
    def characterization(self):
        directory = self.directory('characterization')
        if not (directory / 'provenance.json').is_file():
            raise MissingArtifact(f'No characterization in {directory}; run characterize.')
        provenance = json.loads((directory / 'provenance.json').read_text())
        if not provenance.get('identity'):
            raise InvalidArtifact(f'The characterization in {directory} is unsigned; run characterize again.')
        files = {}
        for name in provenance.get('generated_files', []):
            verify_signed_file(directory / name, provenance['identity'])
            files[name] = directory / name
        from msr_qspr.data import verify_scientific_inputs
        if provenance.get('freeze_sha256_before_and_after') != verify_scientific_inputs():
            raise InvalidArtifact('The characterization was produced from a different workbook.')
        return StageOutputs('characterization', directory, files)


@_lru_cache(maxsize=4)
def _verified_outputs(root):
    return VerifiedOutputs(root)


def selected_report_table(root, name):
    """Verified stage table restricted to the selected configuration, or None if unavailable."""
    outputs = _verified_outputs(str(Path(root).resolve()))
    sources = {'repetition_metrics.csv': ('comparison', name), 'primary_oof_predictions.csv': ('comparison', name),
               'coverage_error.csv': ('reliability', name), 'permutation_metrics.csv': ('y_randomization', name),
               'interpretation.csv': ('interpretation', name), 'selection_stability.csv': ('interpretation', name),
               'holdout_predictions.csv': ('holdout', 'predictions.csv')}
    try:
        stage, filename = sources[name]
        table = getattr(outputs, stage).table(filename)
        configuration = outputs.selected_configuration
        outputs.bundle
    except (MissingArtifact, InvalidArtifact, KeyError):
        return None
    if 'Configuration' in table:
        table = table.loc[table.Configuration.eq(configuration)].copy()
    return table if not table.empty else None


# ---------------------------------------------------------------------------
# Report tables (no fitting; development OOF unless stated otherwise)
# ---------------------------------------------------------------------------
# Repeated grouped cross-validation reuses the same observations, so repetition ranges are
# descriptive, not confidence intervals. Paired uncertainty uses a chemical-group bootstrap
# that resamples held-out groups with all their repetitions together (BOOTSTRAP_RESAMPLES
# resamples, percentile 95% intervals, seeded by config.SEED). Errors are on the log10(MSR)
# scale with MSR in mol/mol.

def candidate_label(name):
    from msr_qspr.plotting import candidate_label as label
    return label(name)


def feature_label_plain(name):
    from msr_qspr.plotting import feature_label
    return {'CMC': 'log10 CMC', 'WaterSolubility': 'log10 Sw'}.get(name) or feature_label(name)


class ReportData:
    """Verified stage outputs plus derived frames shared by several report tables."""

    def __init__(self, outputs):
        self.outputs = outputs
        self._cache = {}

    def _memo(self, key, function):
        if key not in self._cache:
            self._cache[key] = function()
        return self._cache[key]

    @property
    def selected(self):
        return self.outputs.selected_configuration

    @property
    def scores(self):
        return self.outputs.selection['scores']

    @property
    def order(self):
        """Candidates in selection-rule order (the selected model first)."""
        def key(r):
            return (r['worst_relative_deterioration'], r['mean_relative_deterioration'],
                    r['mandatory_external_reference_count'], r['median_effective_selected_input_count'],
                    r['configuration'])
        return [r['configuration'] for r in sorted(self.scores, key=key)]

    @property
    def oof(self):
        def load():
            frame = self.outputs.comparison.table('primary_oof_predictions.csv',
                                                   dtype={'observation_id': str, 'canonical_group': str})
            frame['error'] = frame.predicted_log10_MSR - frame.actual_log10_MSR
            frame['squared_error'] = frame.error ** 2
            frame['absolute_error'] = frame.error.abs()
            return frame
        return self._memo('oof', load)

    @property
    def repetitions(self):
        return self._memo('repetitions', lambda: self.outputs.comparison.table('repetition_metrics.csv'))

    @property
    def reliability_rows(self):
        """Selected-model OOF applicability statuses joined with the comparison group labels."""
        def load():
            status = self.outputs.reliability.table('observation_status.csv', dtype={'observation_id': str})
            oof = self.oof[self.oof.Configuration.eq(self.selected)]
            labels = oof[['Scenario', 'Repetition', 'observation_id', 'contaminant_group',
                          'active_structure_group', 'publication_group']]
            frame = status.merge(labels, on=['Scenario', 'Repetition', 'observation_id'], validate='one_to_one')
            if len(frame) != len(status) or len(frame) != len(oof):
                raise InvalidArtifact('Reliability statuses do not match the selected-model OOF predictions.')
            frame['error'] = frame.predicted_log10_MSR - frame.actual_log10_MSR
            frame['absolute_error'] = frame.error.abs()
            frame['abraham_source'] = frame.contaminant_group.map(abraham_sources())
            return frame
        return self._memo('reliability_rows', load)


def abraham_sources():
    """Contaminant canonical SMILES -> Abraham descriptor source, when the curated files are installed."""
    from msr_qspr import abraham
    try:
        abraham.verify_files()
    except (FileNotFoundError, ValueError):
        return {}
    directory = abraham.descriptor_directory()
    descriptors = pd.read_csv(directory / abraham.DESCRIPTOR_FILE)
    provenance = pd.read_csv(directory / abraham.PROVENANCE_FILE)
    merged = descriptors.merge(provenance[['inchikey', 'source_type']], on='inchikey', validate='one_to_one')
    return dict(zip(merged.canonical_smiles, merged.source_type))


def rmse(errors):
    errors = np.asarray(errors, dtype=float)
    return float(np.sqrt(np.mean(errors ** 2))) if errors.size else float('nan')


def r_squared(actual, predicted):
    actual, predicted = np.asarray(actual, float), np.asarray(predicted, float)
    total = np.sum((actual - actual.mean()) ** 2)
    return float(1 - np.sum((predicted - actual) ** 2) / total) if total > 0 else float('nan')


def pooled_errors(frame):
    return pd.Series({'n_observations_per_repetition': frame.observation_id.nunique(),
                      'RMSE': rmse(frame.error), 'MAE': float(frame.absolute_error.mean()),
                      'bias': float(frame.error.mean())})


def _contrasts(data):
    pairs = [(data.selected, other) for other in data.order if other != data.selected]
    return list(dict.fromkeys(pairs + [p for p in DESIGNED_CONTRASTS if p[0] != p[1]]))


# ---------------------------------------------------------------------------
# Comparison tables
# ---------------------------------------------------------------------------

def table_candidate_performance(data):
    long = table_candidate_performance_long(data)
    scores = {r['configuration']: r for r in data.scores}
    rows = []
    for rank, name in enumerate(data.order, start=1):
        experiment = next(e for e in config.study_experiments() if e.name == name)
        score = scores[name]
        row = {'rank': rank, 'configuration': name, 'label': candidate_label(name),
               'predictor_family': experiment.predictor_family, 'estimator': experiment.estimator,
               'selected': name == data.selected,
               'worst_relative_deterioration_percent': 100 * score['worst_relative_deterioration'],
               'mean_relative_deterioration_percent': 100 * score['mean_relative_deterioration'],
               'external_references': ' + '.join(experiment.external_references) or 'none',
               'median_inputs': score['median_effective_selected_input_count']}
        for scenario in SCENARIOS:
            values = long[(long.Scenario == scenario) & (long.Configuration == name)].iloc[0]
            for metric in ('RMSE', 'MAE', 'R2'):
                row[f'{scenario}_{metric}'] = values[f'{metric}_mean']
            if scenario in PRIMARY:
                row[f'{scenario}_RMSE_min'] = values['RMSE_min']
                row[f'{scenario}_RMSE_max'] = values['RMSE_max']
        rows.append(row)
    return pd.DataFrame(rows)


def table_candidate_performance_long(data):
    repetitions, oof = data.repetitions, data.oof
    rows = []
    for (scenario, name), frame in repetitions.groupby(['Scenario', 'Configuration']):
        pooled = oof[(oof.Scenario == scenario) & (oof.Configuration == name)]
        row = {'Scenario': scenario, 'Configuration': name, 'label': candidate_label(name),
               'n_repetitions': len(frame), 'n_groups': int(frame.n_groups.iloc[0])}
        for metric, column in (('RMSE', 'RMSE_OOF'), ('MAE', 'MAE_OOF'), ('R2', 'R2_OOF'),
                               ('MacroGroupMAE', 'MacroGroupMAE')):
            values = frame[column]
            row.update({f'{metric}_mean': values.mean(), f'{metric}_sd': values.std(ddof=1) if len(values) > 1 else np.nan,
                        f'{metric}_min': values.min(), f'{metric}_max': values.max()})
        row['bias'] = float(pooled.error.mean())
        rows.append(row)
    table = pd.DataFrame(rows)
    table['rank'] = table.Configuration.map({c: i + 1 for i, c in enumerate(data.order)})
    table['Scenario'] = pd.Categorical(table.Scenario, SCENARIOS, ordered=True)
    return table.sort_values(['rank', 'Scenario']).reset_index(drop=True)


def _group_totals(oof, scenario, name):
    frame = oof[(oof.Scenario == scenario) & (oof.Configuration == name)]
    return frame.groupby('canonical_group').agg(sse=('squared_error', 'sum'), sae=('absolute_error', 'sum'),
                                                n=('squared_error', 'size')).sort_index()


def _paired(data, scenario, a, b, rng):
    ga, gb = _group_totals(data.oof, scenario, a), _group_totals(data.oof, scenario, b)
    if not ga.index.equals(gb.index) or not ga.n.equals(gb.n):
        raise InvalidArtifact('Paired candidates do not share held-out groups.')
    n = ga.n.to_numpy(float)
    weights = rng.multinomial(len(n), np.full(len(n), 1 / len(n)), size=BOOTSTRAP_RESAMPLES).astype(float)
    total = weights @ n
    boot_rmse = np.sqrt(weights @ ga.sse.to_numpy() / total) - np.sqrt(weights @ gb.sse.to_numpy() / total)
    boot_mae = (weights @ ga.sae.to_numpy() - weights @ gb.sae.to_numpy()) / total
    reps = data.repetitions[data.repetitions.Scenario.eq(scenario)]
    per_repetition = (reps[reps.Configuration.eq(a)].sort_values('Repetition').RMSE_OOF.to_numpy()
                      - reps[reps.Configuration.eq(b)].sort_values('Repetition').RMSE_OOF.to_numpy())
    difference = ga.sse - gb.sse
    return {'Scenario': scenario, 'A': a, 'B': b, 'n_groups': len(n),
            'dRMSE': np.sqrt(ga.sse.sum() / n.sum()) - np.sqrt(gb.sse.sum() / n.sum()),
            'dRMSE_low': np.percentile(boot_rmse, 2.5), 'dRMSE_high': np.percentile(boot_rmse, 97.5),
            'dMAE': (ga.sae.sum() - gb.sae.sum()) / n.sum(),
            'dMAE_low': np.percentile(boot_mae, 2.5), 'dMAE_high': np.percentile(boot_mae, 97.5),
            'dRMSE_per_repetition': json.dumps([round(float(v), 4) for v in per_repetition]),
            'repetitions_A_better': int((per_repetition < 0).sum()), 'n_repetitions': len(per_repetition),
            'groups_A_lower_SSE': int((difference < 0).sum()), 'groups_A_higher_SSE': int((difference > 0).sum())}


def table_paired_differences(data):
    rows = []
    for i, (a, b) in enumerate(_contrasts(data)):
        for j, scenario in enumerate(SCENARIOS):
            # One fixed random stream per contrast and axis keeps every interval reproducible on its own.
            rows.append(_paired(data, scenario, a, b, np.random.default_rng([config.SEED, i, j])))
    table = pd.DataFrame(rows)
    table.insert(3, 'label_A', table.A.map(candidate_label))
    table.insert(4, 'label_B', table.B.map(candidate_label))
    return table


def table_class_errors(data):
    frames = []
    for field in ('ContaminantType', 'SurfactantType'):
        table = data.oof.groupby(['Scenario', 'Configuration', field]).apply(
            pooled_errors, include_groups=False).reset_index()
        frames.append(table.rename(columns={field: 'class_value'}).assign(class_field=field))
    return pd.concat(frames, ignore_index=True)


def table_response_quintiles(data):
    oof = data.oof.copy()
    observations = oof.drop_duplicates('observation_id').actual_log10_MSR
    edges = observations.quantile(np.linspace(0, 1, 6)).to_numpy().copy()
    edges[0], edges[-1] = -np.inf, np.inf
    oof['response_quintile'] = pd.cut(oof.actual_log10_MSR, edges, labels=['Q1 (lowest)', 'Q2', 'Q3', 'Q4', 'Q5 (highest)'])
    return oof.groupby(['Scenario', 'Configuration', 'response_quintile'], observed=True).apply(
        pooled_errors, include_groups=False).reset_index()


def table_group_contributions(data):
    rows = []
    for a, b in _contrasts(data):
        for scenario in PRIMARY:
            fa = data.oof[(data.oof.Scenario == scenario) & (data.oof.Configuration == a)]
            fb = data.oof[(data.oof.Scenario == scenario) & (data.oof.Configuration == b)]
            for key in ('contaminant_group', 'active_structure_group'):
                ga = fa.groupby(key).agg(SSE_A=('squared_error', 'sum'), n=('squared_error', 'size'),
                                         bias_A=('error', 'mean'), ContaminantType=('ContaminantType', 'first'),
                                         SurfactantType=('SurfactantType', 'first'))
                gb = fb.groupby(key).agg(SSE_B=('squared_error', 'sum'), bias_B=('error', 'mean'))
                groups = ga.join(gb)
                groups['dSSE'] = groups.SSE_A - groups.SSE_B
                groups['share_of_net_dSSE'] = groups.dSSE / groups.dSSE.sum()
                rows.append(groups.reset_index().rename(columns={key: 'group'}).assign(
                    A=a, B=b, Scenario=scenario, group_type=key))
    return pd.concat(rows, ignore_index=True)


def table_search_spaces(data):
    spaces = data.outputs.selection['record']['stages']['development_comparison']['execution']['search_spaces']
    rows = []
    for name in data.order:
        experiment = next(e for e in config.study_experiments() if e.name == name)
        entries = spaces.get(name) or {'(none)': 'no tuned parameters'}
        for parameter, distribution in entries.items():
            rows.append({'configuration': name, 'label': candidate_label(name), 'search_method': experiment.search_method,
                         'trial_budget': experiment.trial_budget,
                         'parameter': parameter.removeprefix('model__'), 'distribution': distribution})
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Final model and interpretation tables
# ---------------------------------------------------------------------------

def table_final_model(data):
    stage = data.outputs.bundle
    bundle = stage.manifest
    specification = json.loads((stage.directory / 'selected_specification.json').read_text())
    rows = [('configuration', bundle['experiment']['name']), ('label', candidate_label(bundle['experiment']['name'])),
            ('estimator', bundle['experiment']['estimator']),
            ('predictor_family', bundle['experiment']['predictor_family']),
            ('development_observations', len(bundle['development_ids'])),
            ('inputs_in_fitted_order', ', '.join(specification['selected_features'])),
            ('effective_input_count', specification['effective_input_count']),
            ('molecular_k', specification['molecular_k']),
            ('search_method', specification['search']['method']),
            ('trial_budget', specification['search']['trial_budget']),
            ('best_trial', specification['search']['best_trial']),
            ('model_sha256', bundle['model_sha256'])]
    rows += [(f'parameter: {k.removeprefix("model__")}', v) for k, v in specification['model_params'].items()]
    return pd.DataFrame(rows, columns=['item', 'value'])


def table_feature_attributions(data):
    interpretation = data.outputs.interpretation
    table = interpretation.table('interpretation.csv')
    stability = interpretation.table('selection_stability.csv')
    table.insert(1, 'label', table.feature.map(feature_label_plain))
    for scenario, frame in stability.groupby('Scenario'):
        table[f'selection_frequency_{scenario}'] = table.feature.map(dict(zip(frame.feature, frame.selection_frequency)))
    offset = interpretation.json('interpretation.json').get('imposed_offset')
    if offset:
        table = pd.concat([table, pd.DataFrame([{
            'feature': 'imposed_solubility_offset', 'label': f'Imposed offset ({offset["definition"]})',
            'entity': 'contaminant', 'mean_absolute_SHAP': np.nan,
            'limitation': f'Fixed physical term with anchor coefficient {offset["anchor_coefficient"]}; '
                          f'mean absolute value {offset["mean_absolute"]:.3f}; not a learned attribution.'}])],
            ignore_index=True)
    return table


# ---------------------------------------------------------------------------
# Reliability and screening tables (selected model, development OOF)
# ---------------------------------------------------------------------------

def _repetition_mean(frame, function):
    values = [function(f) for _, f in frame.groupby('Repetition')]
    return float(np.mean(values)), float(np.min(values)), float(np.max(values))


def _quantile_interval(frame, q, seed):
    """Percentile interval for an absolute-error quantile, resampling held-out groups with repetitions."""
    frame = frame.reset_index(drop=True)
    groups = frame.canonical_group.unique()
    index = {g: frame.index[frame.canonical_group.eq(g)].to_numpy() for g in groups}
    errors = frame.absolute_error.to_numpy()
    rng = np.random.default_rng(seed)
    stats = [np.quantile(errors[np.concatenate([index[g] for g in rng.choice(groups, len(groups))])], q)
             for _ in range(BOOTSTRAP_RESAMPLES)]
    return float(np.percentile(stats, 2.5)), float(np.percentile(stats, 97.5))


def table_error_summary(data):
    rows = []
    for j, scenario in enumerate(SCENARIOS):
        frame = data.reliability_rows[data.reliability_rows.Scenario.eq(scenario)]
        mean, low, high = _repetition_mean(frame, lambda f: rmse(f.error))
        row = {'Scenario': scenario, 'n_repetitions': frame.Repetition.nunique(),
               'n_groups': frame.canonical_group.nunique(), 'RMSE_mean': mean, 'RMSE_min': low, 'RMSE_max': high,
               'MAE_mean': _repetition_mean(frame, lambda f: f.absolute_error.mean())[0],
               'bias': float(frame.error.mean())}
        for q in (0.5, 0.8, 0.9, 0.95):
            value = float(np.quantile(frame.absolute_error, q))
            row[f'absolute_error_q{int(q * 100)}'] = value
            row[f'absolute_error_q{int(q * 100)}_fold_factor'] = 10 ** value
        row['absolute_error_q90_low'], row['absolute_error_q90_high'] = _quantile_interval(
            frame, 0.9, [config.SEED, j])
        rows.append(row)
    return pd.DataFrame(rows)


def table_applicability_errors(data):
    from scipy.stats import spearmanr
    rows = []
    for scenario in SCENARIOS:
        frame = data.reliability_rows[data.reliability_rows.Scenario.eq(scenario)]
        for component in ('chemical_support', 'response_range', 'local_error', 'ad_status'):
            for status, group in frame.groupby(component):
                rows.append({'Scenario': scenario, 'component': component, 'status': status,
                             'share_of_rows': len(group) / len(frame), 'n_groups': group.canonical_group.nunique(),
                             'RMSE': rmse(group.error), 'MAE': float(group.absolute_error.mean()),
                             'bias': float(group.error.mean())})
        valid = frame[['local_error_value', 'absolute_error']].dropna()
        rows.append({'Scenario': scenario, 'component': 'local_error_value', 'status': 'spearman_with_absolute_error',
                     'share_of_rows': len(valid) / len(frame),
                     'spearman': float(spearmanr(valid.local_error_value, valid.absolute_error).statistic)})
    return pd.DataFrame(rows)


def table_prediction_range_errors(data):
    rows = []
    for scenario in SCENARIOS:
        frame = data.reliability_rows[data.reliability_rows.Scenario.eq(scenario)].copy()
        frame['bin'] = pd.qcut(frame.predicted_log10_MSR, 5, labels=['P1 (lowest)', 'P2', 'P3', 'P4', 'P5 (highest)'])
        for label, group in frame.groupby('bin', observed=True):
            rows.append({'Scenario': scenario, 'subset': f'predicted {label}',
                         'predicted_min': group.predicted_log10_MSR.min(), 'predicted_max': group.predicted_log10_MSR.max(),
                         'RMSE': rmse(group.error), 'bias': float(group.error.mean()), 'n_rows': len(group)})
        low = frame[frame.actual_log10_MSR < -3]
        rows.append({'Scenario': scenario, 'subset': 'observed log10(MSR) < -3', 'RMSE': rmse(low.error),
                     'bias': float(low.error.mean()) if len(low) else np.nan, 'n_rows': len(low)})
    return pd.DataFrame(rows)


def table_reliability_class_errors(data):
    rows = []
    fields = ['ContaminantType', 'SurfactantType']
    if data.reliability_rows.abraham_source.notna().any():
        fields.append('abraham_source')
    for scenario in SCENARIOS:
        frame = data.reliability_rows[data.reliability_rows.Scenario.eq(scenario)]
        for field in fields:
            for value, group in frame.groupby(field):
                rows.append({'Scenario': scenario, 'field': field, 'value': value,
                             'n_observations_per_repetition': group.observation_id.nunique(),
                             'n_contaminants': group.contaminant_group.nunique(),
                             'RMSE': rmse(group.error), 'bias': float(group.error.mean()),
                             'share_outside_or_incomplete': float(group.ad_status.ne('inside').mean())})
    return pd.DataFrame(rows)


def table_support_distance_errors(data):
    rows = []
    for scenario in SCENARIOS:
        frame = data.reliability_rows[data.reliability_rows.Scenario.eq(scenario)].copy()
        frame['relative_distance'] = frame.distance / frame.distance_threshold
        frame['bin'] = pd.qcut(frame.relative_distance, 4, labels=['D1 (nearest)', 'D2', 'D3', 'D4 (farthest)'])
        for label, group in frame.groupby('bin', observed=True):
            rows.append({'Scenario': scenario, 'distance_quartile': label,
                         'relative_distance_median': float(group.relative_distance.median()),
                         'relative_distance_max': float(group.relative_distance.max()),
                         'RMSE': rmse(group.error), 'bias': float(group.error.mean()), 'n_rows': len(group)})
    return pd.DataFrame(rows)


# Compact unanchored baseline compared with the selected model for within-contaminant ranking.
RANKING_COMPARATOR = 'physicochemical_ridge'
RANKING_MIN_LIST = 5        # surfactants per contaminant list
RANKING_MIN_DISTINCT = 3    # distinct observed responses per list


def _ranked_list(group):
    """Spearman correlation and top-1 hit of one list, with explicit tie conventions.

    Spearman uses average ranks. The observed top three are all observations whose response is
    at least the third-largest value, so ties at the boundary are included. When several
    observations share the highest prediction, the hit is the fraction of them in the observed top
    three, which is also the hit expected from a random choice among them. The chance rate is the
    size of the observed top-three set divided by the list size.
    """
    from scipy.stats import spearmanr
    actual = group.actual_log10_MSR.to_numpy(dtype=float)
    predicted = group.predicted_log10_MSR.to_numpy(dtype=float)
    top = actual >= np.sort(actual)[-3]
    best = predicted == predicted.max()
    error = predicted - actual
    return {'n': len(group), 'spearman': float(spearmanr(predicted, actual).statistic),
            'top1_in_observed_top3': float(top[best].mean()), 'chance_top1_in_top3': float(top.mean()),
            'observed_top3_size': int(top.sum()), 'contaminant_offset': float(error.mean()),
            'within_contaminant_RMSE': rmse(error - error.mean())}


def table_within_contaminant_ranking(data):
    """Within-contaminant surfactant ranking on unseen contaminants: selected model and comparator.

    Lists are the (repetition, contaminant) groups of the unseen-contaminant OOF predictions with at
    least RANKING_MIN_LIST observations and RANKING_MIN_DISTINCT distinct responses; both models are
    scored on the same lists.
    """
    oof = data.oof[data.oof.Scenario.eq('unseen_contaminant')]
    rows = []
    for model in (data.selected, RANKING_COMPARATOR):
        frame = oof[oof.Configuration.eq(model)]
        for (repetition, contaminant), group in frame.groupby(['Repetition', 'contaminant_group']):
            if len(group) < RANKING_MIN_LIST or group.actual_log10_MSR.nunique() < RANKING_MIN_DISTINCT:
                continue
            rows.append({'Configuration': model, 'label': candidate_label(model), 'Repetition': repetition,
                         'contaminant_group': contaminant, 'ContaminantType': group.ContaminantType.iloc[0],
                         **_ranked_list(group)})
    table = pd.DataFrame(rows)
    keys = [set(map(tuple, table.loc[table.Configuration.eq(m), ['Repetition', 'contaminant_group']].values))
            for m in (data.selected, RANKING_COMPARATOR)]
    if keys[0] != keys[1]:
        raise InvalidArtifact('Ranking lists differ between the compared models.')
    return table


def _ranking_statistics(frame):
    return {'weighted_spearman': float(np.average(frame.spearman, weights=frame.n)),
            'mean_spearman': float(frame.spearman.mean()),
            'contaminant_mean_spearman': float(frame.groupby('contaminant_group').spearman.mean().mean()),
            'median_spearman': float(frame.spearman.median()),
            'share_negative_spearman': float((frame.spearman < 0).mean()),
            'top1_in_observed_top3': float(frame.top1_in_observed_top3.mean()),
            'chance_top1_in_top3': float(frame.chance_top1_in_top3.mean())}


def table_ranking_summary(data):
    """Ranking statistics per model and the paired difference with a contaminant bootstrap interval.

    weighted_spearman weights lists by their size; mean_spearman weights lists equally;
    contaminant_mean_spearman averages repetitions within each contaminant first. Top-1 hit and
    chance rates weight lists equally. The interval resamples contaminants with all their
    repetitions (BOOTSTRAP_RESAMPLES resamples, percentile 95%).
    """
    ranking = table_within_contaminant_ranking(data)
    oof = data.oof[data.oof.Scenario.eq('unseen_contaminant')]
    rows = []
    for model in (data.selected, RANKING_COMPARATOR):
        lists = ranking[ranking.Configuration.eq(model)]
        frame = oof[oof.Configuration.eq(model)]
        offsets = frame.error.groupby([frame.Repetition, frame.contaminant_group]).transform('mean')
        base = {'Configuration': model, 'label': candidate_label(model), 'n_lists': len(lists),
                'n_contaminants': lists.contaminant_group.nunique(), 'n_repetitions': lists.Repetition.nunique(),
                'list_size_min': int(lists.n.min()), 'list_size_median': float(lists.n.median()),
                'list_size_max': int(lists.n.max())}
        rows.append({**base, 'subset': 'all', **_ranking_statistics(lists),
                     'RMSE_total': rmse(frame.error), 'RMSE_contaminant_offset': rmse(offsets),
                     'RMSE_within_contaminant': rmse(frame.error - offsets),
                     'offset_share_of_SSE': float((offsets ** 2).sum() / (frame.error ** 2).sum())})
        for value, group in lists.groupby('ContaminantType'):
            rows.append({**base, 'subset': value, 'n_lists': len(group),
                         'n_contaminants': group.contaminant_group.nunique(), **_ranking_statistics(group)})
    pivot = {m: ranking[ranking.Configuration.eq(m)].set_index(['Repetition', 'contaminant_group']).sort_index()
             for m in (data.selected, RANKING_COMPARATOR)}
    first, second = pivot[data.selected], pivot[RANKING_COMPARATOR]
    contaminants = first.index.get_level_values('contaminant_group')
    unique = np.array(sorted(set(contaminants)))
    position = {c: np.flatnonzero(contaminants == c) for c in unique}
    rng = np.random.default_rng(config.SEED)
    samples = {'weighted_spearman': [], 'top1_in_observed_top3': []}
    for _ in range(BOOTSTRAP_RESAMPLES):
        rows_ = np.concatenate([position[c] for c in rng.choice(unique, len(unique))])
        a, b = first.iloc[rows_], second.iloc[rows_]
        samples['weighted_spearman'].append(np.average(a.spearman, weights=a.n) - np.average(b.spearman, weights=b.n))
        samples['top1_in_observed_top3'].append(a.top1_in_observed_top3.mean() - b.top1_in_observed_top3.mean())
    difference = {'Configuration': f'{data.selected} - {RANKING_COMPARATOR}', 'label': 'Difference',
                  'subset': 'all', 'n_lists': len(first), 'n_contaminants': len(unique),
                  'n_repetitions': first.index.get_level_values('Repetition').nunique(),
                  'lists_selected_higher_spearman': int((first.spearman > second.spearman).sum()),
                  'lists_equal_spearman': int((first.spearman == second.spearman).sum())}
    for key, values in samples.items():
        observed = (_ranking_statistics(first.reset_index())[key] - _ranking_statistics(second.reset_index())[key])
        difference.update({key: observed, f'{key}_low': float(np.percentile(values, 2.5)),
                           f'{key}_high': float(np.percentile(values, 97.5))})
    rows.append(difference)
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Target randomization and holdout tables
# ---------------------------------------------------------------------------

def table_y_randomization(data):
    permutations = data.outputs.y_randomization.table('permutation_metrics.csv')
    selected = data.repetitions[data.repetitions.Configuration.eq(data.selected)]
    rows = []
    for scenario in SCENARIOS:
        shuffled = permutations[permutations.Scenario.eq(scenario)].mean_repetition_RMSE
        observed = float(selected[selected.Scenario.eq(scenario)].RMSE_OOF.mean())
        rows.append({'Scenario': scenario, 'n_permutations': len(shuffled),
                     'shuffled_RMSE_mean': shuffled.mean(), 'shuffled_RMSE_sd': shuffled.std(ddof=1),
                     'shuffled_RMSE_min': shuffled.min(), 'shuffled_RMSE_max': shuffled.max(),
                     'selected_RMSE': observed, 'permutations_at_or_below_selected': int((shuffled <= observed).sum())})
    return pd.DataFrame(rows)


def _holdout_frame(data):
    def load():
        predictions = data.outputs.holdout.table('predictions.csv', dtype={'observation_id': str})
        metadata = holdout_metadata(predictions.observation_id)
        frame = predictions.merge(metadata, on='observation_id', how='left', validate='one_to_one')
        return frame
    return data._memo('holdout', load)


def holdout_metadata(observation_ids):
    """Chemical classes and groups of the reserved observations, from structure metadata only."""
    from msr_qspr.data import corpus_metadata
    from msr_qspr.validation import _canonical_cv_metadata
    metadata = _canonical_cv_metadata(corpus_metadata(), config.ACTIVE_HOLDOUT)
    columns = ['row_id', 'ContaminantType', 'SurfactantType', 'publication_group', 'active_structure_group',
               'contaminant_group']
    metadata = metadata[columns].rename(columns={'row_id': 'observation_id'})
    return metadata[metadata.observation_id.isin(set(observation_ids))]


def table_holdout_summary(data):
    metrics = data.outputs.holdout.json('metrics.json')
    rows = [{'subset': 'all', 'value': 'all', **{k: metrics[k] for k in ('n_observations', 'RMSE', 'MAE', 'R2')}}]
    rows.extend({'subset': 'abraham_descriptor_status', 'value': status, **values}
                for status, values in metrics.get('by_abraham_descriptor_status', {}).items())
    return pd.DataFrame(rows)


def table_holdout_composition(data):
    holdout = _holdout_frame(data)
    oof = data.oof[(data.oof.Configuration == data.selected) & (data.oof.Scenario == 'unseen_surfactant')]
    development = oof.drop_duplicates('observation_id')
    rows = []
    for population, frame, surfactants in (('development', development, development.active_structure_group),
                                           ('holdout', holdout, holdout.active_structure_group)):
        response = frame.actual_log10_MSR
        rows.append({'population': population, 'n_observations': len(frame),
                     'n_surfactant_groups': surfactants.nunique(),
                     'n_contaminants': frame.contaminant_group.nunique(),
                     'n_publications': frame.publication_group.nunique(),
                     'response_min': response.min(), 'response_max': response.max(), 'response_sd': response.std(ddof=1),
                     **{f'contaminant_class: {k}': v for k, v in frame.ContaminantType.value_counts().items()},
                     **{f'surfactant_class: {k}': v for k, v in frame.SurfactantType.value_counts().items()}})
    return pd.DataFrame(rows)




# ---------------------------------------------------------------------------
# Report outputs
# ---------------------------------------------------------------------------

def _plot(name, *args, **kwargs):
    from msr_qspr import plotting
    return getattr(plotting, name)(*args, **kwargs)


def figure_candidate_rmse(data):
    return _plot('plot_candidate_rmse', table_candidate_performance_long(data), data.order)


def figure_paired_differences(data):
    return _plot('plot_paired_differences', table_paired_differences(data), data.order, data.selected)


def figure_axis_balance(data):
    return _plot('plot_axis_balance', table_candidate_performance_long(data), data.order)


def figure_class_errors(data):
    return _plot('plot_class_errors', table_class_errors(data), data.order)


def figure_interpretation(data):
    interpretation = data.outputs.interpretation
    if 'shap_values.csv' not in interpretation.files:
        raise MissingArtifact('interpretation/shap_values.csv is not available; run interpret to produce it.')
    return _plot('plot_interpretation', interpretation.table('interpretation.csv'),
                 interpretation.table('shap_values.csv'),
                 anchored=bool(interpretation.json('interpretation.json').get('imposed_offset')))


def figure_applicability(data):
    return _plot('plot_applicability', table_applicability_errors(data), table_support_distance_errors(data))


def figure_ranking(data):
    return _plot('plot_within_contaminant_ranking', table_within_contaminant_ranking(data),
                 (data.selected, RANKING_COMPARATOR), seed=config.SEED)


def figure_screening_cases(data):
    """One figure per saved screening case (screen --save NAME)."""
    cases = sorted((data.outputs.directory('screening') / 'cases').glob('*.csv'))
    if not cases:
        raise MissingArtifact('No screening cases in screening/cases/; save one with screen --save NAME.')
    model_sha256 = data.outputs.bundle.manifest['model_sha256']
    figures = {}
    for path in cases:
        metadata = json.loads(path.with_suffix('.json').read_text()) if path.with_suffix('.json').is_file() else {}
        if metadata.get('model_sha256') != model_sha256:
            raise InvalidArtifact(f'Screening case {path.name} was not produced with the selected fitted bundle.')
        if metadata.get('ranking_sha256') != artifacts.file_sha256(path):
            raise InvalidArtifact(f'Screening case {path.name} failed its checksum.')
        title = metadata.get('contaminant_name') or metadata.get('contaminant_smiles')
        figures[f'case_{path.stem}'] = _plot('plot_screening_case', pd.read_csv(path), title)
    return figures


def figure_y_randomization(data):
    return _plot('plot_y_randomization', data.outputs.y_randomization.table('permutation_metrics.csv'),
                 table_y_randomization(data))


def figure_holdout_parity(data):
    """Unseen-surfactant OOF averaged over repetitions per observation, and the recorded holdout."""
    oof = data.oof[(data.oof.Configuration == data.selected) & data.oof.Scenario.eq('unseen_surfactant')]
    development = oof.groupby('observation_id').agg(actual=('actual_log10_MSR', 'first'),
                                                    predicted=('predicted_log10_MSR', 'mean'))
    return _plot('plot_holdout_parity', development, _holdout_frame(data), data.outputs.holdout.json('metrics.json'))


@_dataclass(frozen=True)
class ReportOutput:
    """One table or figure: its stage, intended use, required verified outputs and builder.

    Outputs without a builder are files that the stage command writes itself; ``name`` selects
    them ('*' for every table of the stage).
    """
    key: str
    stage: str
    kind: str
    name: str
    description: str
    requires: tuple
    build: object = None
    use: str = 'supplement'
    command: str = 'msr-qspr report'


def _outputs():
    T, F, MAIN = 'table', 'figure', 'main'
    sel, rel, interp = ('selection',), ('selection', 'reliability'), ('selection', 'bundle', 'interpretation')
    held, char = ('selection', 'bundle', 'holdout'), ('characterization',)
    characterize = 'msr-qspr characterize'
    return (
        ReportOutput('chemical_classes', 'characterization', F, 'chemical_classes',
                     'log10(MSR) by contaminant and surfactant class, with observation counts.', char, use=MAIN,
                     command=characterize),
        ReportOutput('class_coverage', 'characterization', F, 'class_coverage',
                     'Observations per contaminant and surfactant class.', char, command=characterize),
        ReportOutput('partition_ecdf', 'characterization', F, 'partition_ecdf',
                     'log10(MSR) cumulative distributions of the development and holdout populations.', char,
                     command=characterize),
        ReportOutput('target_distribution', 'characterization', F, 'target_distribution',
                     'log10(MSR) histogram and cumulative distribution of all observations.', char, command=characterize),
        ReportOutput('target_diagnostics', 'characterization', F, 'target_diagnostics',
                     'MSR histogram and normal Q-Q plots on the raw and log10 scales.', char, command=characterize),
        ReportOutput('physicochemical_properties', 'characterization', F, 'physicochemical_properties',
                     'CMC and aqueous solubility by class, and Spearman correlations with log10(MSR).', char,
                     command=characterize),
        ReportOutput('group_heterogeneity', 'characterization', F, 'group_heterogeneity',
                     'log10(MSR) range and median of the largest publication and surfactant groups.', char,
                     command=characterize),
        ReportOutput('condition_coverage', 'characterization', F, 'condition_coverage',
                     'Assay-temperature reporting and its availability by contaminant class.', char,
                     command=characterize),
        ReportOutput('characterization_tables', 'characterization', T, '*',
                     'Descriptive tables of all accepted observations and the characterization report.', char,
                     command=characterize),
        ReportOutput('candidate_performance', 'comparison', T, 'candidate_performance',
                     'Candidates in selection order: criterion and grouped OOF RMSE, MAE and R2 on each axis '
                     '(means over repetitions).', sel, table_candidate_performance, MAIN),
        ReportOutput('candidate_performance_by_axis', 'comparison', T, 'candidate_performance_by_axis',
                     'Mean, SD and range over repetitions of every metric, per candidate and axis.',
                     sel, table_candidate_performance_long),
        ReportOutput('candidate_rmse_figure', 'comparison', F, 'candidate_rmse_by_axis',
                     'Grouped OOF RMSE of every candidate on the three validation axes: mean and range over '
                     'repetitions; values beyond the axis are marked with their value.', sel, figure_candidate_rmse, MAIN),
        ReportOutput('paired_differences', 'comparison', T, 'paired_differences',
                     'Paired RMSE and MAE differences with chemical-group bootstrap intervals.', sel, table_paired_differences),
        ReportOutput('paired_differences_figure', 'comparison', F, 'paired_differences',
                     'RMSE of the selected model minus each alternative, with bootstrap 95% intervals.', sel,
                     figure_paired_differences),
        ReportOutput('axis_balance_figure', 'comparison', F, 'axis_balance',
                     'Unseen-surfactant versus unseen-contaminant RMSE of all candidates: mean and range over '
                     'repetitions; clipped ranges are marked with their value.', sel, figure_axis_balance),
        ReportOutput('class_errors', 'comparison', T, 'class_errors',
                     'Pooled OOF errors by contaminant and surfactant class.', sel, table_class_errors),
        ReportOutput('class_errors_figure', 'comparison', F, 'class_errors',
                     'Unseen-contaminant RMSE by contaminant class for every candidate.', sel, figure_class_errors),
        ReportOutput('response_quintile_errors', 'comparison', T, 'response_quintile_errors',
                     'Pooled OOF errors by observed-response quintile.', sel, table_response_quintiles),
        ReportOutput('group_contributions', 'comparison', T, 'group_contributions',
                     'Held-out group contributions to paired squared-error differences.', sel, table_group_contributions),
        ReportOutput('search_spaces', 'comparison', T, 'search_spaces',
                     'Hyperparameter search space of every candidate.', sel, table_search_spaces),
        ReportOutput('final_model_specification', 'final_model', T, 'final_model_specification',
                     'Selected procedure refitted on all development observations: inputs and tuned parameters.',
                     ('selection', 'bundle'), table_final_model),
        ReportOutput('feature_attributions', 'interpretation', T, 'feature_attributions',
                     'Learned attributions, linear and nonlinear components, permutation importance, selection stability.',
                     interp, table_feature_attributions),
        ReportOutput('interpretation_figure', 'interpretation', F, 'interpretation',
                     'Learned attributions of the final model: mean absolute SHAP and SHAP per development '
                     'observation, in a shared feature order; the imposed solubility offset is not attributed.',
                     interp, figure_interpretation, MAIN),
        ReportOutput('error_summary', 'reliability', T, 'error_summary_by_axis',
                     'Selected-model OOF error and empirical absolute-error quantiles per axis.', rel, table_error_summary),
        ReportOutput('applicability_errors', 'reliability', T, 'applicability_component_errors',
                     'Share of rows and OOF error inside and outside each applicability component.', rel,
                     table_applicability_errors),
        ReportOutput('applicability_figure', 'reliability', F, 'applicability_errors',
                     'OOF RMSE by applicability status and by distance to the training support.', rel,
                     figure_applicability),
        ReportOutput('prediction_range_errors', 'reliability', T, 'prediction_range_errors',
                     'OOF error by predicted-response quintile and for the lowest observed responses.', rel,
                     table_prediction_range_errors),
        ReportOutput('reliability_class_errors', 'reliability', T, 'class_errors',
                     'Selected-model OOF error and applicability by chemical class and descriptor provenance.', rel,
                     table_reliability_class_errors),
        ReportOutput('support_distance_errors', 'reliability', T, 'support_distance_errors',
                     'OOF error by distance to the training support relative to the threshold.', rel,
                     table_support_distance_errors),
        ReportOutput('y_randomization_summary', 'y_randomization', T, 'y_randomization_summary',
                     'Shuffled-target RMSE across permutations compared with the selected model.',
                     ('selection', 'y_randomization'), table_y_randomization),
        ReportOutput('y_randomization_figure', 'y_randomization', F, 'y_randomization',
                     'Shuffled-target RMSE distribution and the selected-model RMSE per axis.',
                     ('selection', 'y_randomization'), figure_y_randomization),
        ReportOutput('holdout_summary', 'holdout', T, 'holdout_summary',
                     'Recorded holdout metrics overall and by descriptor provenance.', held, table_holdout_summary, MAIN),
        ReportOutput('holdout_composition', 'holdout', T, 'holdout_composition',
                     'Composition and response range of the development and holdout populations.', held,
                     table_holdout_composition),
        ReportOutput('holdout_parity_figure', 'holdout', F, 'holdout_parity',
                     'Observed versus predicted log10(MSR): unseen-surfactant OOF averaged over the three '
                     'repetitions of each development observation, and the 69 recorded holdout predictions; '
                     'the annotation gives the recorded holdout metrics.', held, figure_holdout_parity, MAIN),
        ReportOutput('within_contaminant_ranking', 'screening', T, 'within_contaminant_ranking',
                     'Within-contaminant ranking per unseen-contaminant list for the selected model and the '
                     'compact Ridge comparator, with chance rates.', sel, table_within_contaminant_ranking),
        ReportOutput('ranking_summary', 'screening', T, 'ranking_summary',
                     'Ranking statistics per model (size-weighted, list and contaminant means), chance rate, '
                     'error decomposition and the paired difference with a contaminant bootstrap interval.',
                     sel, table_ranking_summary),
        ReportOutput('ranking_figure', 'screening', F, 'within_contaminant_ranking',
                     'Within-contaminant Spearman correlation by contaminant class for both models.', sel,
                     figure_ranking),
        ReportOutput('screening_cases', 'screening', F, 'case_*',
                     'Predicted MSR and WSR with applicability for each saved screening case; WSR uses the '
                     'molecular weights of the full supplied structures.', ('selection', 'bundle'),
                     figure_screening_cases, MAIN),
    )


REPORT_OUTPUTS = _outputs()
REPORT_STAGES = tuple(dict.fromkeys(o.stage for o in REPORT_OUTPUTS))


def _source_files(outputs, names):
    files = {}
    for name in names:
        stage = getattr(outputs, name)
        for path in (stage['files'] if isinstance(stage, dict) else stage.files).values():
            files[str(path)] = path
    return files


def build_report(root=None, *, stages=None, log=None):
    """Build every output whose required stage outputs verify, and update INDEX.md / index.json."""
    from msr_qspr.plotting import save_figure
    outputs = VerifiedOutputs(root)
    data, root = ReportData(outputs), outputs.root
    counts = {'built': 0, 'missing': 0, 'invalid': 0, 'error': 0}
    entries = []
    for output in REPORT_OUTPUTS:
        if stages and output.stage not in stages:
            continue
        entry = {'key': output.key, 'stage': output.stage, 'kind': output.kind, 'use': output.use,
                 'description': output.description, 'command': output.command}
        try:
            sources = _source_files(outputs, output.requires)
            if output.build is None:
                stage = getattr(outputs, output.stage)
                if output.kind == 'table':
                    paths = [p for name, p in stage.files.items() if name.endswith(('.csv', '.txt'))]
                else:
                    paths = [stage.files[f'figures/{output.name}.{suffix}'] for suffix in ('pdf', 'png')]
            else:
                result = output.build(data)
                folder = outputs.directory(output.stage) / ('tables' if output.kind == 'table' else 'figures')
                if output.kind == 'table':
                    folder.mkdir(parents=True, exist_ok=True)
                    paths = [folder / f'{output.name}.csv']
                    result.to_csv(paths[0], index=False)
                else:
                    figures = result if isinstance(result, dict) else {output.name: result}
                    paths = [p for name, figure in figures.items() for p in save_figure(figure, folder, name)]
            entry.update(status='built', files=[str(p.relative_to(root)) for p in paths],
                         sources={str(p.relative_to(root)): artifacts.file_sha256(p)
                                  for p in sorted(map(Path, sources.values())) if p.is_relative_to(root)})
            if 'y_randomization' in output.requires and (outputs.directory('y_randomization') / 'selected_model.json').is_file():
                entry['note'] = 'Randomization has a separately verified selection record; software provenance may differ.'
        except MissingArtifact as reason:
            entry.update(status='missing', reason=str(reason))
        except InvalidArtifact as reason:
            entry.update(status='invalid', reason=str(reason))
        except Exception as reason:  # one failing output must not hide the others
            entry.update(status='error', reason=f'{type(reason).__name__}: {reason}')
        counts[entry['status']] += 1
        if log:
            log(f"{output.stage:>16}  {output.key:<30} {entry['status']}"
                + (f" - {entry['reason']}" if 'reason' in entry else ''))
        entries.append(entry)
    _write_index(root, entries)
    try:
        configuration = outputs.selected_configuration
    except (MissingArtifact, InvalidArtifact):
        configuration = None
    return {'artifacts': str(root), 'selected_configuration': configuration, **counts,
            'index': str(root / 'INDEX.md'), 'outputs': entries}


def _write_index(root, entries):
    path = root / 'index.json'
    known = {e['key']: e for e in json.loads(path.read_text()).get('outputs', [])} if path.is_file() else {}
    known.update({e['key']: e for e in entries})
    registered = {o.key for o in REPORT_OUTPUTS}
    ordered = [known[o.key] for o in REPORT_OUTPUTS if o.key in known]
    path.write_text(json.dumps({'format_version': 1, 'outputs': ordered}, indent=2) + '\n')
    lines = ['# Artifact index', '',
             'Written by `msr-qspr report`. Each output lists the stage files it was built from; '
             'their SHA-256 checksums are in `index.json`.']
    for use, heading in (('main', 'Main outputs'), ('supplement', 'Supplementary outputs')):
        lines += ['', f'## {heading}', '', '| Output | Stage | Status | Files | Built from | Command |',
                  '|---|---|---|---|---|---|']
        for entry in (e for e in ordered if e.get('use', 'supplement') == use and e['key'] in registered):
            files = entry.get('files', [])
            shown = '<br>'.join(f'`{p}`' for p in files[:6]) + (f'<br>… ({len(files)} files)' if len(files) > 6 else '')
            sources = sorted({str(Path(p).parent) + '/' for p in entry.get('sources', {})})
            built = '<br>'.join(f'`{s}`' for s in sources) or entry.get('reason', '')
            if entry.get('note'):
                built += '<br>' + entry['note']
            lines.append(f"| {entry['description']} | {entry['stage']} | {entry['status']} | {shown or '—'} "
                         f"| {built or '—'} | `{entry['command']}` |")
    (root / 'INDEX.md').write_text('\n'.join(lines) + '\n')
