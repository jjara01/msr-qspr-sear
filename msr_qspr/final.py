"""Verified procedure fitting, bundle inference and explicitly guarded holdout scoring."""

from msr_qspr import artifacts, config
from msr_qspr.reporting import validate_single_selection
import json
import numpy as np
import pandas as pd
import joblib
from sklearn.metrics import r2_score

def prepare_final_model_training(development, comparison_table, selection, study):
    """Bind one eventual final fit to all development IDs and a verified tuning policy.

    Returns a reproducible training request without fitting. The fixed policy
    must state grouping, allocation seed, fold count, joint descriptor-count and
    parameter search, and an explicit budget. No numerical policy is inferred.
    """
    from msr_qspr.study import membership_fingerprint
    from msr_qspr.validation import _canonical_cv_metadata
    request = validate_single_selection(comparison_table, selection, study)
    metadata = _canonical_cv_metadata(development, 'unseen_surfactant')
    if len(metadata) != 390 or membership_fingerprint(metadata.row_id) != request['development_sha256']:
        raise ValueError('Final fitting requires the exact full 390-observation development population.')
    policy = request['tuning_policy']
    if (policy.get('settings') != list(config.PRIMARY_SETTINGS) or type(policy.get('seed')) is not int
            or policy.get('folds_per_axis') != 3 or policy.get('training_policy') != dict(config.TRAINING_POLICY)
            or policy.get('search') not in {'joint_molecular_k_and_parameters', 'fixed_predictors'}
            or not policy.get('experiment') or type(policy.get('trial_budget')) is not int
            or policy['trial_budget'] < 0):
        raise ValueError('Final-model tuning policy is incomplete or unsupported.')
    request.update(observation_ids=metadata.row_id.tolist(), n_estimators_to_fit=1,
                   operations=['training_only_inner_rankings', 'joint_search',
                               'full_development_ranking', 'final_fit', 'prediction_bundle'],
                   execution_status='verified_training_request')
    return request


def evaluate_holdout(*, study, bundle_root, confirmed=False, output_directory=None, load_holdout=None):
    """Score a completed fitted bundle once after explicit confirmation, without refitting."""
    from pathlib import Path
    from msr_qspr.artifacts import fingerprint, membership_fingerprint
    from msr_qspr.study import check_run_metadata, run_fingerprint, experiment_specification
    from msr_qspr import config
    if confirmed is not True:
        raise ValueError('Explicit confirmation is required before reserved response access.')
    identity = check_run_metadata(study, 'holdout')
    if study['stages']['holdout'] != {'ad_policy': dict(config.AD_POLICY),
                                    'metrics_reporting': dict(config.HOLDOUT_REPORTING)}:
        raise ValueError('Holdout evaluation requires the fixed AD and reporting policies.')
    model, bundle = load_inference_bundle(bundle_root)
    if bundle.get('analysis_support', {}).get('holdout') != 'confirmation_required':
        raise ValueError('Reserved scoring is blocked until reliability for this predictor procedure is validated.')
    expected = {'study_sha256': run_fingerprint(study, 'final_model'),
                'scientific_sha256': fingerprint(study['scientific']),
                'development_sha256': study['scientific']['development_sha256'],
                'holdout_sha256': study['scientific']['holdout_sha256']}
    if any(bundle.get(k) != v for k, v in expected.items()):
        raise ValueError('Final bundle differs from the selected scientific model or population.')
    directory = Path(output_directory or artifacts.artifact_directory('holdout'))
    # Claim the destination before responses load; even an interrupted evaluation is preserved.
    directory.parent.mkdir(parents=True, exist_ok=True)
    directory.mkdir(exist_ok=False)
    if load_holdout is None:
        from msr_qspr.data import load_reserved_predictors
        experiment = experiment_specification(study['stages']['final_model']['tuning_policy']['experiment'])
        load_holdout = lambda: load_reserved_predictors(experiment)
    X, y_log, metadata = load_holdout()
    observation_ids = metadata.row_id.tolist()
    if membership_fingerprint(observation_ids) != bundle['holdout_sha256']:
        raise ValueError('Holdout loader returned incompatible observation membership.')
    actual = np.asarray(y_log, dtype=float)
    predicted = np.asarray(model.predict(X), dtype=float)
    if (len(actual) != len(observation_ids) or actual.shape != predicted.shape
            or not np.isfinite(actual).all() or not np.isfinite(predicted).all()):
        raise ValueError('Holdout predictions and responses must be aligned and finite.')
    error = predicted - actual
    ad = model.reliability_.assess(model, X, metadata)
    predictions = pd.DataFrame({'observation_id': observation_ids,
        'actual_log10_MSR': actual, 'predicted_log10_MSR': predicted})
    predictions = predictions.merge(ad.drop(columns='predicted_log10_MSR'), on='observation_id', validate='one_to_one')
    metrics = {'R2': float(r2_score(actual, predicted)), 'RMSE': artifacts.root_mean_squared_error(actual, predicted),
               'MAE': float(np.abs(error).mean()), 'n_observations': len(actual),
               'ad_components': model.reliability_.specification()['components']}
    from msr_qspr.abraham import STATUS_COLUMN
    if STATUS_COLUMN in metadata and experiment_uses_abraham(bundle):
        # Prespecified stratification by curated, estimated or imputed descriptor provenance.
        predictions[STATUS_COLUMN] = metadata[STATUS_COLUMN].to_numpy()
        metrics['by_abraham_descriptor_status'] = {
            status: {'n_observations': int(mask.sum()),
                     'RMSE': artifacts.root_mean_squared_error(actual[mask], predicted[mask]),
                     'MAE': float(np.abs(error[mask]).mean())}
            for status in sorted(set(metadata[STATUS_COLUMN]))
            for mask in [metadata[STATUS_COLUMN].to_numpy() == status]}
    provenance = dict(expected, model_sha256=bundle['model_sha256'], confirmed=True,
                      ad_policy=dict(config.AD_POLICY), reporting=dict(config.HOLDOUT_REPORTING))
    from msr_qspr.reliability import coverage_error_report
    classes = (metadata[['SurfactantType', 'ContaminantType']]
               if {'SurfactantType', 'ContaminantType'} <= set(metadata) else None)
    coverage = coverage_error_report(ad, actual, ad.support_group, classes)
    paths = [directory / name for name in ('predictions.csv', 'metrics.json', 'provenance.json', 'ad_coverage_errors.csv')]
    for path, content in zip(paths, (predictions, metrics, provenance, coverage)):
        with artifacts.atomic_artifact(path) as temporary:
            if isinstance(content, pd.DataFrame):
                content.to_csv(temporary, index=False)
            else:
                temporary.write_text(json.dumps(content, indent=2) + '\n')
        artifacts.record_cache(path, identity)
    artifacts.publish_completed_unit(directory / 'holdout_complete.json', identity, paths)
    return metrics


def experiment_uses_abraham(bundle):
    return bundle['experiment']['predictor_family'] == 'abraham_physicochemical'


def fit_selected_final_model(*, development, X, y_log, comparison_table, selection,
                             study, output_root=None):
    """Reproduce the verified selected procedure on the full development population."""
    from msr_qspr.preprocessing import validate_predictor_matrix
    from msr_qspr.study import experiment_specification
    from msr_qspr.framework import fit_final_predictor
    validate_predictor_matrix(X)
    request = prepare_final_model_training(development, comparison_table, selection, study)
    experiment = experiment_specification(request['tuning_policy']['experiment'])
    if (len(X) != len(development) or len(y_log) != len(X)
            or not X.index.equals(development.index) or not y_log.index.equals(X.index)
            or not np.isfinite(np.asarray(y_log, dtype=float)).all()):
        raise ValueError('Final predictors and log responses must align with development membership.')
    return fit_final_predictor(X=X, y_log=y_log, development=development,
        experiment=experiment, request=request, study=study, output_root=output_root)


def load_inference_bundle(directory):
    """Verify checksummed local artifacts and load the fitted prediction procedure."""
    from pathlib import Path
    root = Path(directory).resolve()
    bundle = json.loads((root / 'prediction_ad_bundle.json').read_text())
    if (bundle.get('format_version') != 3
            or bundle.get('scientific_identity') != config.scientific_identity()
            or bundle.get('candidate_registry_sha256') != artifacts.candidate_registry_fingerprint()
            or not isinstance(bundle.get('experiment'), dict)
            or not config.is_study_candidate(bundle.get('experiment', {}).get('name'))):
        raise ValueError('Incompatible prediction bundle schema; use its producing source revision.')
    return _load_prediction_bundle(root, bundle)


def _attach_final_reliability(model, X, y_log, metadata, experiment, directory, context, signature):
    """Retain the existing grouped residual calibration for compatible fitted estimators."""
    from msr_qspr.preprocessing import project_experiment_predictors
    from msr_qspr.residuals import generate_final_development_residual_atlas
    from msr_qspr.reliability import ReliabilityReference, fitted_input_space, descriptor_definitions
    from msr_qspr.artifacts import fingerprint
    predictors = project_experiment_predictors(X, experiment)
    provenance = dict(model_identity=fingerprint(experiment.to_dict()),
        representation_identity=fingerprint({'candidate': experiment.to_dict(),
            'input_features': list(predictors), 'input_attributes': predictors.attrs}),
        scientific_sha256=context['scientific_sha256'], study_sha256=context['study_sha256'])
    threshold = dict(fit_ids=metadata.row_id.tolist(), quantile=config.AD_POLICY['local_error_quantile'])
    threshold['policy_sha256'] = fingerprint(threshold)
    atlas = generate_final_development_residual_atlas(X_train=predictors, y_train=y_log,
        training_metadata=metadata, experiment=experiment, provenance=provenance,
        threshold_policy=threshold, output_directory=directory / 'crossfit')
    model.reliability_ = ReliabilityReference(model, predictors, y_log, metadata,
        scenario='unseen_surfactant', provenance=provenance)
    model.reliability_.attach_final_local_error(atlas, threshold_policy=threshold)
    if model.reliability_.distance_threshold is None or model.reliability_.local_threshold is None:
        raise ValueError('Final AD requires at least seven eligible canonical surfactant groups.')
    model.fit_metadata_['local_error_status'] = 'training_calibrated'
    model.interpretation_background_ = fitted_input_space(model, predictors).to_numpy()
    path = directory / 'final_residual_atlas.json'
    with artifacts.atomic_artifact(path) as temporary:
        temporary.write_text(atlas.to_json(orient='records', indent=2) + '\n')
    artifacts.record_cache(path, signature)
    return dict(effective_features=model.reliability_.features,
        descriptor_definitions=descriptor_definitions(model.reliability_.features).to_dict('records'),
        reliability=model.reliability_.specification(), ad_components=model.reliability_.specification()['components'],
        residual_reference=dict(filename=path.name, n_observations=len(atlas), reference_folds=3, seed=42,
            method='Selection-conditional grouped cross-fitted residual reference', independently_validated=False))


def _load_prediction_bundle(root, bundle):
    """Load prediction-only procedures without inventing unsupported AD analyses."""
    from msr_qspr.preprocessing import require_predictor_schema, PredictorSchema
    from msr_qspr.study import experiment_specification
    experiment = experiment_specification(bundle['experiment'])
    support = bundle.get('analysis_support', {})
    compatible = support == {'reliability': 'available', 'interpretation': 'available',
                            'y_randomization': 'complete_procedure', 'holdout': 'confirmation_required'}
    if (bundle.get('status') != 'frozen_validated'
            or bundle.get('target') != {'name': 'log10_MSR', 'MSR_units': 'mol/mol'}
            or not compatible
            or bundle.get('inference', {}).get('counterion_policy') != 'counterion_strip'
            or bundle.get('inference', {}).get('temperature_policy') != 'temp_off'
            or not bundle.get('identity_sha256')
            or not artifacts.completed_unit_valid(root / 'final_fit_complete.json', bundle['identity_sha256'])):
        raise ValueError('Incomplete or incompatible prediction bundle.')
    files = bundle.get('artifacts_sha256', {})
    name = bundle.get('model_file')
    if not name or files.get(name) != bundle.get('model_sha256'):
        raise ValueError('Model identity is not bound to its prediction manifest.')
    for filename, checksum in files.items():
        path = (root / filename).resolve()
        if (root not in path.parents or not path.is_file()
                or artifacts.file_sha256(path) != checksum):
            raise ValueError('Prediction bundle checksum mismatch.')
    model = joblib.load(root / name)
    schema = require_predictor_schema(model)
    if (not isinstance(schema, PredictorSchema) or getattr(schema, 'contract_', None) != config.PREDICTOR_CONTRACT
            or list(schema.columns_) != bundle['inference']['raw_predictors']
            or model.fit_metadata_.get('development_ids') != bundle['development_ids']):
        raise ValueError('Prediction schema or development metadata mismatch.')
    selected = model.fit_metadata_.get('selected_features', [])
    effective_count = config.effective_input_count(experiment, selected)
    if (artifacts.fingerprint(model.fit_metadata_.get('experiment')) != artifacts.fingerprint(bundle['experiment'])
            or model.fit_metadata_.get('effective_input_count') != effective_count
            or model.steps[-1][1].n_features_in_ != effective_count):
        raise ValueError('Stored predictor specification or effective input count is inconsistent.')
    reference = getattr(model, 'reliability_', None)
    if (reference is None
            or reference.specification() != bundle.get('reliability')
            or reference.features != bundle.get('effective_features')
            or reference.ids != bundle['development_ids'] or reference.local_atlas is None
            or len(reference.local_atlas) != len(reference.ids) or reference.local_threshold is None
            or reference.distance_threshold is None or not hasattr(model, 'interpretation_background_')
            or bundle.get('residual_reference', {}).get('filename') not in files):
        raise ValueError('Stored predictor and calibrated reliability reference are incompatible.')
    return model, bundle
