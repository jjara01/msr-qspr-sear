"""Eligibility of local-error references for three-component applicability domains.

Residuals and predictions are in log10(MSR). Independent assessment requires
nested grouped cross-fitting wholly inside the assessed outer-training set,
including representation fitting, ranking, tuning and threshold calibration.
Development OOF residuals after model selection support descriptive use only.
This module checks provenance; it does not fit or evaluate an AD geometry.
"""

from msr_qspr import artifacts
from msr_qspr import config
import numpy as np
import pandas as pd
from msr_qspr.artifacts import fingerprint

SUPERVISED_STEPS = ('source_training_ids', 'preprocessing_fit_ids', 'ranking_fit_ids',
                    'tuning_fit_ids', 'model_selection_ids')
REFERENCE_FIELDS = {'observation_id', 'scenario', 'model_identity', 'publication_group',
                    'active_structure_group', 'contaminant_group', 'prediction', 'residual',
                    'source_validation_ids', 'representation_identity', 'preprocessing_identity',
                    'calibration_eligibility', 'generation_method', 'scientific_sha256',
                    'study_sha256', 'threshold_policy_sha256', *SUPERVISED_STEPS}
CROSSFIT_REQUIREMENT = ('Additional grouped nested cross-fitting inside each outer-training '
                       'population must exclude reference validation observations from '
                       'preprocessing, ranking, tuning and model selection; threshold fitting '
                       'must also exclude the assessed outer validation population.')


def validate_residual_references(atlas, *, query_ids, outer_training_ids, outer_validation_ids,
                                 scenario, model_identity, representation_identity,
                                 scientific_sha256, study_sha256, threshold_policy, canonical_metadata):
    """Return eligible independent references or a reason for unavailable calibration.

    Never return a query observation's own residual. All lineage lists must
    identify real training subsets; labels alone do not establish independence.
    Threshold policy identity and its training membership are checked separately.
    """
    def unavailable(reason):
        return {'status': 'uncalibrated', 'reason': reason,
                'required_work': CROSSFIT_REQUIREMENT, 'references': pd.DataFrame()}

    if atlas is None or len(atlas) == 0:
        return unavailable('No independent local-error residual atlas is available.')
    if not REFERENCE_FIELDS <= set(atlas) or atlas.observation_id.duplicated().any():
        raise ValueError('Residual-reference contract is incomplete or duplicated.')
    train, validation, query = map(set, (outer_training_ids, outer_validation_ids, query_ids))
    if not train or not query or train & validation or not query <= validation:
        raise ValueError('Outer populations and assessed queries are incompatible.')
    if (not np.isfinite(atlas[['prediction', 'residual']].to_numpy(dtype=float)).all()
            or atlas[list(REFERENCE_FIELDS - set(SUPERVISED_STEPS) - {'source_validation_ids'})].isna().any().any()):
        raise ValueError('Residual references contain nonfinite values or missing identities.')
    if set(atlas.observation_id) & query:
        raise ValueError('An observation cannot use its own observed residual.')
    if not set(atlas.observation_id) <= train:
        raise ValueError('Residual reference is outside the assessed outer-training population.')
    if (set(atlas.calibration_eligibility) != {'independent'} or
            set(atlas.generation_method) != {'nested_grouped_crossfit'}):
        return unavailable('Trial-selected, globally preprocessed or model-selected OOF residuals are ineligible.')
    from msr_qspr.validation import VALIDATION_SETTINGS
    if scenario not in VALIDATION_SETTINGS:
        raise ValueError('Unknown residual-reference grouping axis.')
    columns = ['publication_group', 'active_structure_group', 'contaminant_group']
    if (not {'row_id', *columns} <= set(canonical_metadata)
            or canonical_metadata.row_id.duplicated().any()
            or canonical_metadata[['row_id'] + columns].isna().any().any()):
        raise ValueError('Residual calibration requires complete canonical population metadata.')
    metadata = canonical_metadata.set_index('row_id')
    if not (train | validation) <= set(metadata.index):
        raise ValueError('Residual lineage is absent from canonical population metadata.')
    axis = VALIDATION_SETTINGS[scenario]['group_column']
    if set(metadata.loc[list(train), axis]) & set(metadata.loc[list(validation), axis]):
        raise ValueError('Assessed outer-training canonical groups leak validation.')
    expected = {'scenario': scenario, 'model_identity': model_identity,
                'representation_identity': representation_identity,
                'scientific_sha256': scientific_sha256, 'study_sha256': study_sha256,
                'threshold_policy_sha256': fingerprint(threshold_policy)}
    if any(not atlas[k].eq(v).all() for k, v in expected.items()):
        raise ValueError('Residual atlas scientific, representation or threshold identity mismatch.')
    _check_grouped_lineage(atlas, metadata, train, axis)
    policy = dict(threshold_policy)
    policy_digest = policy.pop('policy_sha256', None)
    fit_ids = policy.get('fit_ids', [])
    if (policy_digest != fingerprint(policy)
            or not fit_ids or len(fit_ids) != len(set(fit_ids))
            or not set(fit_ids) <= train or set(fit_ids) & query):
        return unavailable('Local-error threshold policy is unbound or uses ineligible observations.')
    return {'status': 'eligible', 'calibration_status': 'not_evaluated',
            'reason': 'Eligible nested grouped references; numerical calibration is not evaluated.',
            'references': atlas.copy(), 'required_work': None}


def _check_grouped_lineage(atlas, metadata, train, axis):
    columns = ['publication_group', 'active_structure_group', 'contaminant_group']
    for row in atlas.to_dict('records'):
        if any(row[c] != metadata.loc[row['observation_id'], c] for c in columns):
            raise ValueError('Residual canonical group identity mismatch.')
        held = row['source_validation_ids']
        if (not isinstance(held, list) or not held or len(held) != len(set(held))
                or row['observation_id'] not in held or not set(held) <= train):
            raise ValueError('Residual validation provenance is incomplete or leaks outer validation.')
        for step in SUPERVISED_STEPS:
            ids = row[step]
            if (not isinstance(ids, list) or len(ids) != len(set(ids)) or not ids
                    or not set(ids) <= train or set(ids) & set(held)):
                raise ValueError(f'Residual supervised provenance leaks validation: {step}')
            if set(metadata.loc[ids, axis]) & set(metadata.loc[held, axis]):
                raise ValueError(f'Residual supervised canonical groups leak validation: {step}')
        if not row['preprocessing_identity']:
            raise ValueError('Residual references lack a fitted preprocessing identity.')


def validate_final_residual_references(atlas, *, training_metadata, y_log, scenario,
                                      model_identity, representation_identity, scientific_sha256,
                                      study_sha256, threshold_policy):
    """Require a complete final-development, surfactant-grouped OOT residual atlas."""
    required = REFERENCE_FIELDS | {'observed_log10_MSR', 'absolute_residual',
        'reference_fold', 'fold_selected_features', 'fold_ranking_identity',
        'outer_seed', 'inner_seed', 'software_identity'}
    ids = training_metadata.row_id.astype(str).tolist()
    if (scenario != 'unseen_surfactant' or len(ids) != len(set(ids)) or
            atlas is None or not required <= set(atlas) or len(atlas) != len(ids)
            or atlas.observation_id.duplicated().any() or set(atlas.observation_id) != set(ids)):
        raise ValueError('Final residual atlas must cover the complete unique surfactant-grouped development population.')
    if (not set(threshold_policy.get('fit_ids', [])) == set(ids)
            or len(threshold_policy['fit_ids']) != len(ids)
            or threshold_policy.get('quantile') != config.AD_POLICY['local_error_quantile']
            or threshold_policy.get('policy_sha256') != fingerprint({k:v for k,v in threshold_policy.items()
                                                                      if k != 'policy_sha256'})):
        raise ValueError('Final residual threshold calibration population or quantile mismatch.')
    expected = dict(scenario=scenario, model_identity=model_identity,
        representation_identity=representation_identity, scientific_sha256=scientific_sha256,
        study_sha256=study_sha256, threshold_policy_sha256=fingerprint(threshold_policy),
        calibration_eligibility='independent', generation_method='nested_grouped_crossfit')
    if any(not atlas[k].eq(v).all() for k, v in expected.items()):
        raise ValueError('Final atlas model, representation, cross-fitting or population identity mismatch.')
    numeric = atlas[['prediction', 'residual', 'observed_log10_MSR', 'absolute_residual']].to_numpy(dtype=float)
    observed = pd.Series(np.asarray(y_log, dtype=float), index=ids)
    if (len(observed) != len(ids) or not np.isfinite(numeric).all() or not np.isfinite(observed).all()
            or not np.allclose(atlas.observed_log10_MSR, observed.loc[atlas.observation_id], rtol=0, atol=1e-10)
            or not np.allclose(atlas.residual, atlas.observed_log10_MSR - atlas.prediction, rtol=0, atol=1e-10)
            or not np.allclose(atlas.absolute_residual, atlas.residual.abs(), rtol=0, atol=1e-10)):
        raise ValueError('Final atlas residual, actual response or finite numerical evidence is corrupted.')
    columns = ['row_id', 'publication_group', 'active_structure_group', 'contaminant_group']
    if (not set(columns) <= set(training_metadata) or training_metadata[columns].isna().any().any()):
        raise ValueError('Final residual reference lacks canonical identities.')
    metadata = training_metadata.set_index('row_id')
    _check_grouped_lineage(atlas, metadata, set(ids), 'active_structure_group')
    for row in atlas.to_dict('records'):
        held = set(row['source_validation_ids'])
        training = set(ids) - held
        if (any(set(row[step]) != training for step in SUPERVISED_STEPS)
                or not isinstance(row['reference_fold'], (int, np.integer))
                or row['reference_fold'] not in (1, 2, 3)
                or not isinstance(row['fold_selected_features'], list)
                or not row['fold_selected_features'] or not row['fold_ranking_identity']
                or not row['software_identity'] or row['outer_seed'] != 42
                or row['inner_seed'] != 42 + row['reference_fold']):
            raise ValueError('Final atlas fold identity, train-only configuration or seed is invalid.')
    for fold, rows in atlas.groupby('reference_fold'):
        held = set(rows.observation_id)
        if any(set(v) != held for v in rows.source_validation_ids):
            raise ValueError('Final atlas reference-fold validation membership is inconsistent.')
    if set(atlas.reference_fold) != {1, 2, 3}:
        raise ValueError('Final atlas requires exactly three complete reference folds.')
    return {'status': 'eligible', 'references': atlas.copy(),
            'reason': 'Selection-conditional grouped cross-fitted residual reference; no independent validation claim.'}


def nested_calibration_plan(experiment, *, outer_units=35, reference_folds=3, seed=42):
    """Exact additional nominal cost for a prespecified training-residual quantile policy.

    The cutoff is a fixed quantile of group-mean leave-group-out local scores,
    not a coverage/error-optimized threshold. No extra estimator fits
    are needed to compute that quantile. The plan records computation without executing it.
    """
    if (type(outer_units) is not int or outer_units < 1 or
            type(reference_folds) is not int or reference_folds < 3 or type(seed) is not int):
        raise ValueError('Calibration needs explicit positive outer units, at least three folds and a seed.')
    inner_folds = len(config.PRIMARY_SETTINGS) * config.GROUPED_INNER_FOLDS
    ranked = experiment.selection_policy == 'ranked_molecular'
    ranking_scopes = outer_units * reference_folds * (inner_folds + 1)
    return {'experiment': experiment.to_dict(),
        'outer_units': outer_units, 'reference_folds': reference_folds, 'inner_folds': inner_folds,
        'seed': seed, 'threshold_policy': 'prespecified_quantile_of_group_mean_leave_group_out_local_scores',
        'local_error_quantile': config.AD_POLICY['local_error_quantile'],
        'estimator_fits': outer_units * reference_folds * (inner_folds * experiment.trial_budget + 1),
        'ranking_rf_fits': ranking_scopes * config.B_REPEATS if ranked and experiment.ranking_strategy == 'rf' else 0,
        'ranking_linear_fits': ranking_scopes if ranked and experiment.ranking_strategy == 'linear' else 0,
        'threshold_estimator_fits': 0,
        'scope': 'Conditional on a fixed candidate configuration; no claim to validate global model selection.',
        'lineage': 'Population-bound reference folds and training threshold policy.'}


def generate_nested_residual_atlas(*, X_train, y_train, training_metadata, query_metadata,
                                   scenario, experiment, provenance, threshold_policy,
                                   output_directory, reference_folds=3, seed=42,
                                   final_reference=False):
    """Execute the nested calibration primitive after its study-run gate has passed.

    Only outer-training targets are accepted. Every reference fold repeats ranking,
    k selection and six dual-axis grouped inner tuning folds. No globally selected features
    or hyperparameters are accepted. The returned atlas remains training calibration.
    """
    from pathlib import Path
    from msr_qspr.validation import _allocate_chemical_folds, NestedGroupedFold, VALIDATION_SETTINGS
    from msr_qspr.models import _run_grouped_procedure
    if (not X_train.index.equals(y_train.index) or not X_train.index.equals(training_metadata.index)
            or not np.isfinite(y_train).all() or not training_metadata.row_id.is_unique
            or (not final_reference and
                (query_metadata is None or set(training_metadata.row_id) & set(query_metadata.row_id)))):
        raise ValueError('Calibration requires aligned disjoint outer populations and finite training targets.')
    if final_reference and (query_metadata is not None or scenario != 'unseen_surfactant'
                            or reference_folds != 3 or seed != 42):
        raise ValueError('Final reference requires three seed-42 surfactant-grouped folds and no external queries.')
    policy = dict(threshold_policy)
    policy_digest = policy.pop('policy_sha256', None)
    if (policy_digest != fingerprint(policy)
            or set(policy.get('fit_ids', [])) != set(training_metadata.row_id)
            or policy.get('quantile') != config.AD_POLICY['local_error_quantile']):
        raise ValueError('Calibration requires a separately verified, training-bound 0.90 quantile policy.')
    if type(reference_folds) is not int or reference_folds < 3:
        raise ValueError('Independent local-error calibration requires at least three reference folds.')
    axis = VALIDATION_SETTINGS[scenario]['group_column']
    if (not final_reference and (query_metadata.empty or not query_metadata.row_id.is_unique
            or set(training_metadata[axis]) & set(query_metadata[axis]))):
        raise ValueError('Calibration outer populations must exclude all validation canonical groups.')
    parents = _allocate_chemical_folds(training_metadata, axis, reference_folds, seed)
    folds = []
    for parent in parents:
        local = training_metadata.iloc[parent.train_pos].reset_index(drop=True)
        inner = _allocate_chemical_folds(local, axis, 3, seed + parent.fold)
        folds.append(NestedGroupedFold(**parent.__dict__, scenario=scenario, repetition=1,
                     outer_seed=seed, inner_seed=seed + parent.fold, inner_folds=inner))
    context = dict(provenance, target_kind='true_training_crossfit', scenario=scenario,
                   threshold_policy_sha256=fingerprint(threshold_policy), experiment=experiment.to_dict(),
                   calibration_folds=reference_folds, seed=seed)
    prediction, directory = _run_grouped_procedure(X_train, y_train, training_metadata, experiment,
                                                  folds, output_directory, context)
    indexed = prediction.set_index('observation_id')
    if (not prediction.observation_id.is_unique or set(indexed.index) != set(training_metadata.row_id)
            or not np.allclose(indexed.loc[training_metadata.row_id].actual_log10_MSR, y_train)):
        raise ValueError('Nested calibration OOF coverage or response provenance is incomplete.')
    rows = []
    for fold in folds:
        import joblib
        model_path = directory / f'fold-{fold.fold:02d}' / 'model.joblib'
        preprocessing_identity = artifacts.file_sha256(model_path)
        fitted = joblib.load(model_path).fit_metadata_
        if (fitted.get('partition_membership_hashes') != artifacts.nested_membership_hashes(fold)
                or fitted['experiment'] != experiment.to_dict()):
            raise ValueError('Cross-fitted residual model lineage or candidate specification mismatch.')
        for obs in fold.validation_ids:
            meta = training_metadata.set_index('row_id').loc[obs]
            pred = float(indexed.loc[obs, 'predicted_log10_MSR'])
            rows.append(dict(observation_id=obs, scenario=scenario,
                **{c: meta[c] for c in ['publication_group', 'active_structure_group', 'contaminant_group']},
                **{step: list(fold.train_ids) for step in SUPERVISED_STEPS},
                source_validation_ids=list(fold.validation_ids), prediction=pred,
                residual=float(indexed.loc[obs, 'actual_log10_MSR'] - pred),
                observed_log10_MSR=float(indexed.loc[obs, 'actual_log10_MSR']),
                absolute_residual=abs(float(indexed.loc[obs, 'actual_log10_MSR'] - pred)),
                reference_fold=fold.fold, outer_seed=fold.outer_seed, inner_seed=fold.inner_seed,
                fold_selected_features=list(fitted['selected_features']),
                fold_ranking_identity=fitted['ranking_identity'],
                software_identity=fitted['environment'],
                preprocessing_identity=preprocessing_identity, calibration_eligibility='independent',
                generation_method='nested_grouped_crossfit',
                threshold_policy_sha256=fingerprint(threshold_policy), **provenance))
    atlas = pd.DataFrame(rows)
    if final_reference:
        validate_final_residual_references(atlas, training_metadata=training_metadata, y_log=y_train,
            scenario=scenario, threshold_policy=threshold_policy, **provenance)
    else:
        validate_residual_references(atlas, query_ids=query_metadata.row_id.tolist(),
            outer_training_ids=training_metadata.row_id.tolist(), outer_validation_ids=query_metadata.row_id.tolist(),
            scenario=scenario, threshold_policy=threshold_policy,
            canonical_metadata=pd.concat([training_metadata, query_metadata], ignore_index=True), **provenance)
    target = Path(output_directory) / 'independent_residual_atlas.json'
    with artifacts.atomic_artifact(target) as temporary:
        temporary.write_text(atlas.to_json(orient='records', indent=2) + '\n')
    artifacts.record_cache(target, fingerprint(context))
    return atlas


def generate_final_development_residual_atlas(*, X_train, y_train, training_metadata,
                                              experiment, provenance, threshold_policy,
                                              output_directory):
    """Calibrate the chosen F3 candidate on development alone, without future queries."""
    if len(training_metadata) != 390:
        raise ValueError('Final-development cross-fitting requires exactly 390 observations.')
    return generate_nested_residual_atlas(X_train=X_train, y_train=y_train,
        training_metadata=training_metadata, query_metadata=None,
        scenario='unseen_surfactant', experiment=experiment, provenance=provenance,
        threshold_policy=threshold_policy, output_directory=output_directory,
        reference_folds=3, seed=42, final_reference=True)
