"""Verified training-scoped reuse for individual grouped modeling procedures."""

from pathlib import Path
import json

import joblib
import numpy as np
import pandas as pd

from msr_qspr import artifacts, config, models, progress
from msr_qspr.preprocessing import (project_experiment_predictors, infer_feature_roles,
                                   _build_model_pipeline_for_fold)
from msr_qspr.validation import _allocate_chemical_folds


def _write(path, value, signature):
    with artifacts.atomic_artifact(path) as temporary:
        if isinstance(value, pd.DataFrame):
            value.to_csv(temporary, index=False)
        elif path.suffix == '.joblib':
            joblib.dump(value, temporary)
        else:
            temporary.write_text(json.dumps(value, indent=2, default=str) + '\n')
    artifacts.record_cache(path, signature)


def _aligned(X, y, metadata):
    if (len(X) != len(y) or len(X) != len(metadata) or not X.index.equals(y.index)
            or not X.index.equals(metadata.index) or not metadata.row_id.is_unique
            or not np.isfinite(np.asarray(y, dtype=float)).all()):
        raise ValueError('Training predictors, targets and observation IDs must align exactly.')
    artifacts.membership_fingerprint(metadata.row_id.tolist())


def validate_training_provenance(specification, metadata):
    """Check the actual dual-axis tuning scopes separately from external-plan lineage."""
    ids = metadata.row_id.tolist()
    if (not metadata.row_id.is_unique or specification.get('training_ids') != ids
            or specification.get('training_policy') != config.TRAINING_POLICY):
        raise ValueError('Fitted procedure has incompatible training population or policy.')
    if 'component_fit_metadata' in specification:
        raise ValueError('Individual provenance cannot bypass tuning memberships with component metadata.')
    memberships = specification.get('inner_memberships', [])
    if len(memberships) != 2 * config.GROUPED_INNER_FOLDS:
        raise ValueError('Fitted procedure requires six actual dual-axis tuning memberships.')
    population = set(ids)
    indexed = metadata.set_index('row_id')
    for axis in config.PRIMARY_SETTINGS:
        coverage = dict.fromkeys(ids, 0)
        folds = [m for m in memberships if m.get('axis') == axis]
        if len(folds) != config.GROUPED_INNER_FOLDS:
            raise ValueError('Fitted procedure requires three tuning folds per chemical axis.')
        column = config.VALIDATION_SETTINGS[axis]['group_column']
        for fold in folds:
            train, validation = fold['train_ids'], fold['validation_ids']
            if (not train or not validation or len(set(train)) != len(train)
                    or len(set(validation)) != len(validation) or set(train) & set(validation)
                    or set(train) | set(validation) != population
                    or set(indexed.loc[train, column]) & set(indexed.loc[validation, column])):
                raise ValueError('Actual tuning memberships leak or differ from their training scope.')
            for identifier in validation:
                coverage[identifier] += 1
        if set(coverage.values()) != {1}:
            raise ValueError('Actual tuning folds must validate each training observation exactly once per axis.')


class TrainingBank:
    """Reuse only identically scoped tuning, rankings and fitted base estimators."""

    def __init__(self, directory, experiments, context, *, splitter=None):
        self.directory = Path(directory)
        self.experiments = tuple(experiments)
        if (not self.experiments
                or len({e.name for e in self.experiments}) != len(self.experiments)):
            raise ValueError('A model bank requires distinct individual specifications.')
        self.context = dict(context)
        self.splitter = splitter or _allocate_chemical_folds
        self._partitions = {}

    def splits(self, metadata, scenario):
        axis = config.VALIDATION_SETTINGS[scenario]['group_column']
        values = metadata[['row_id', axis, 'SurfactantType', 'ContaminantType']]
        identity = artifacts.fingerprint(values.to_dict('records'))
        key = (identity, scenario)
        if key not in self._partitions:
            seed = int(artifacts.fingerprint([identity, scenario, config.SEED])[:8], 16)
            folds = self.splitter(metadata, axis, config.GROUPED_INNER_FOLDS, seed)
            self._partitions[key] = [(f.train_pos, f.validation_pos) for f in folds]
        result = self._partitions[key]
        self.check_splits(metadata, scenario, result)
        return result

    @staticmethod
    def check_splits(metadata, scenario, splits):
        axis = config.VALIDATION_SETTINGS[scenario]['group_column']
        coverage = np.zeros(len(metadata), dtype=int)
        if len(splits) != config.GROUPED_INNER_FOLDS:
            raise ValueError('Training-only validation requires three grouped folds.')
        for train, validation in splits:
            train, validation = np.asarray(train), np.asarray(validation)
            if (not len(train) or not len(validation)
                    or not np.issubdtype(train.dtype, np.integer)
                    or not np.issubdtype(validation.dtype, np.integer)
                    or len(set(train)) != len(train) or len(set(validation)) != len(validation)
                    or set(train) & set(validation)
                    or set(train) | set(validation) != set(range(len(metadata)))
                    or set(metadata.iloc[train][axis]) & set(metadata.iloc[validation][axis])):
                raise ValueError('Invalid or leaking training-only chemical partition.')
            coverage[validation] += 1
        if not np.all(coverage == 1):
            raise ValueError('Cross-fitting must predict every observation exactly once.')

    def fit(self, X, y, metadata):
        return dict(self.iter_fit(X, y, metadata))

    def iter_fit(self, X, y, metadata, *, candidates=None):
        """Yield verified training-bank models as each candidate becomes available."""
        if candidates is not None:
            candidates = set(candidates)
            if not candidates <= {e.name for e in self.experiments}:
                raise ValueError('Unknown candidate requested from training bank.')
        _aligned(X, y, metadata)
        partitions = {axis: self.splits(metadata, axis) for axis in config.PRIMARY_SETTINGS}
        splits = [split for axis in config.PRIMARY_SETTINGS for split in partitions[axis]]
        labels = [axis for axis in config.PRIMARY_SETTINGS for _ in partitions[axis]]
        scope = dict(self.context, scientific_protocol=config.SCIENTIFIC_PROTOCOL,
                     chemical_metadata=metadata[['row_id', 'active_structure_group',
                         'contaminant_group', 'SurfactantType', 'ContaminantType']].to_dict('records'),
                     training_ids=metadata.row_id.tolist(),
                     training_policy=config.TRAINING_POLICY,
                     inner_splits_by_axis={axis: [(tr.tolist(), va.tolist()) for tr, va in folds]
                                           for axis, folds in partitions.items()})
        scope_id = artifacts.artifact_signature(X, y, {}, scientific_context=scope)
        directory = self.directory / scope_id
        for experiment in self.experiments:
            if candidates is not None and experiment.name not in candidates:
                continue
            progress.update(candidate=experiment.name, estimator=experiment.estimator,
                            predictor_family=experiment.predictor_family,
                            search_method=experiment.search_method, trial_budget=experiment.trial_budget,
                            completed_trials=0, best_objective=None, best_trial=None,
                            axis_rmse=None, recovered_trials=0, checkpoint=None,
                            current_trial=None, phase='preparing')
            progress.log(f'[{self.experiments.index(experiment) + 1:02d}/{len(self.experiments):02d}] '
                         f'{experiment.name} | {experiment.estimator} | preparing')
            projected = project_experiment_predictors(X, experiment)
            identity = artifacts.artifact_signature(projected, y, {
                'experiment': experiment.to_dict(), 'parameter_space': config.parameter_spaces(experiment)},
                scientific_context=scope)
            target = directory / experiment.name / 'model.joblib'
            metadata_path = target.with_name('specification.json')
            complete = target.with_name('complete.json')
            required = [target, metadata_path]
            if experiment.search_method != 'none':
                required.append(target.with_name('search_trials.csv'))
            with artifacts.exclusive_artifact(target):
                if complete.exists():
                    if not artifacts.completed_unit_valid(complete, identity, required,
                            member_identity=identity, allow_missing_sidecar=True):
                        raise ValueError('Completed model bank failed integrity checks; preserve it and investigate.')
                    if experiment.search_method != 'none' and not models._search_history_valid(
                            required[-1], identity, experiment.trial_budget,
                            experiment.selection_policy == 'ranked_molecular'):
                        raise ValueError('Cached model bank has an invalid completed search history.')
                    model = joblib.load(target)
                    specification = json.loads(metadata_path.read_text())
                    expected_memberships = [dict(axis=axis,
                        train_ids=metadata.iloc[tr].row_id.tolist(),
                        validation_ids=metadata.iloc[va].row_id.tolist())
                        for (tr, va), axis in zip(splits, labels)]
                    if (model.fit_metadata_ != specification
                            or artifacts.fingerprint(specification.get('experiment')) != artifacts.fingerprint(experiment.to_dict())
                            or specification.get('signature') != identity
                            or specification.get('training_scope_id') != scope_id
                            or specification.get('raw_predictors') != list(projected)
                            or specification.get('inner_memberships') != expected_memberships
                            or specification.get('effective_input_count') != config.effective_input_count(
                                experiment, specification.get('selected_features', []))):
                        raise ValueError('Cached estimator and specification disagree.')
                    validate_training_provenance(specification, metadata)
                    if not artifacts.cache_matches(complete, identity):
                        artifacts.recover_completed_unit(complete, identity, required, member_identity=identity)
                    progress.update(phase='bank_recovered', completed_trials=experiment.trial_budget,
                                    recovered_trials=experiment.trial_budget,
                                    checkpoint=str(target.with_name('search_trials.study.joblib').resolve())
                                    if experiment.search_method != 'none' else None,
                                    final_parameters=specification['model_params'],
                                    best_trial=specification['search'].get('best_trial'))
                    progress.log(f'{experiment.name} | verified training-bank model recovered')
                else:
                    # Shared rankings bind the same training values, folds and strategy.
                    ranking_context = dict(self.context, training_ids=metadata.row_id.tolist(),
                                           training_policy=config.TRAINING_POLICY)
                    ranking = rankings = None
                    if experiment.selection_policy == 'ranked_molecular':
                        progress.update(phase='ranking')
                        rankings, ranking, rank_id = models._rank_outer_fold(
                            projected, y, experiment, splits, 1, directory / 'rankings', ranking_context, metadata)
                    else:
                        rank_id = artifacts.fingerprint({'fixed': list(projected)})
                    estimator = models.base_estimator(experiment)
                    categories, logs = infer_feature_roles(projected)
                    forced = list(experiment.required_reference_properties)
                    parameters, k, search = models.tune_model_specification(
                        projected, y, experiment=experiment, base_est=estimator,
                        space=config.parameter_spaces(experiment), rankings=rankings,
                        inner_splits=splits, validation_axes=labels, forced=forced, categorical_cols=categories,
                        log10_cols=logs, n_trials=experiment.trial_budget,
                        trial_seed=config.SEED, history_file=target.with_name('search_trials.csv'),
                        signature=identity)
                    if experiment.search_method != 'none':
                        progress.log(f'{experiment.name} | search complete')
                    chosen = list(projected) if ranking is None else ranking.feature.tolist()[:k] + forced
                    support = []
                    if ranking is not None:
                        support_table = ranking.loc[ranking.feature.isin(chosen),
                            ['feature', 'training_finite_structures', 'training_unique_structures',
                             'training_finite_rows', 'training_rows']]
                        support = support_table.astype(object).where(support_table.notna(), None).to_dict('records')
                    progress.update(phase='refitting')
                    model = _build_model_pipeline_for_fold(estimator, chosen, categories, logs)
                    model.set_params(**models.estimator_parameters(experiment, parameters))
                    model.fit(projected, y)
                    progress.update(phase='refit_complete', final_parameters=parameters)
                    progress.log(f'{experiment.name} | final refit complete')
                    specification = dict(experiment=experiment.to_dict(), signature=identity,
                        training_ids=metadata.row_id.tolist(), training_scope_id=scope_id,
                        training_policy=config.TRAINING_POLICY, selected_features=chosen, molecular_k=k,
                        effective_input_count=config.effective_input_count(experiment, chosen),
                        molecular_descriptors=([c for c in chosen if c not in forced]),
                        mandatory_reference_properties=forced, ranking_identity=rank_id,
                        descriptor_training_support=support,
                        raw_predictors=list(projected), model_params=parameters, search=search,
                        inner_memberships=[{'axis': axis, 'train_ids': metadata.iloc[tr].row_id.tolist(),
                                            'validation_ids': metadata.iloc[va].row_id.tolist()}
                                           for (tr, va), axis in zip(splits, labels)])
                    model.fit_metadata_ = json.loads(json.dumps(specification, default=str))
                    validate_training_provenance(model.fit_metadata_, metadata)
                    _write(target, model, identity)
                    _write(metadata_path, model.fit_metadata_, identity)
                    # Search histories have their own trial-level recovery receipts.
                    artifacts.publish_completed_unit(complete, identity, required)
                    artifacts.record_cache(complete, identity)
            yield experiment.name, model


def computational_budget():
    """Count estimator work separately from unmeasured orchestration overhead."""
    experiments = config.study_experiments()
    inner = len(config.PRIMARY_SETTINGS) * config.GROUPED_INNER_FOLDS
    outer = sum(config.GROUPED_OUTER_FOLDS * len(v['seeds'])
                for v in config.VALIDATION_SETTINGS.values())
    tuning = outer * inner * sum(e.trial_budget for e in experiments)
    refits = outer * len(experiments)
    rankings = outer * (inner + 1)
    rf = rankings * config.B_REPEATS
    return dict(base_candidates=len(experiments), outer_evaluations=outer,
        tuning_folds_per_scope=inner, tuning_estimator_fits=tuning,
        outer_training_refits=refits, comparison_base_estimator_fits=tuning + refits,
        individual_evaluation=dict(tuning=tuning, refits=refits, total=tuning + refits),
        base_estimator_fits_per_outer_fold=(tuning + refits) // outer,
        ranking_rf_fits_upper_bound=rf, ranking_linear_fits_upper_bound=rankings,
        ranking_rf_trees=rf * config.RF_N_ESTIM,
        oof_prediction_records=len(experiments) * 390 * sum(
            len(v['seeds']) for v in config.VALIDATION_SETTINGS.values()),
        algorithm_base_fits={e.name: outer * (inner * e.trial_budget + 1) for e in experiments},
        final_model_estimator_fits={e.name: inner * e.trial_budget + 1 for e in experiments},
        optional_complexity_curve_fits_per_candidate_per_point=outer * inner,
        auxiliary_work={
            'milp_allocation': 'target-blind accepted outer plan plus two training-local allocations per unique scope',
            'ranking_preprocessing': rankings * 2,
            'descriptor_calculations': 'structure-cache dependent; measure separately',
            'cache_and_checkpoints': 'checksum, locks and atomic writes; measure separately',
            'artifact_serialization': 'models, histories, predictions and receipts; measure separately',
            'reliability_calibration': 'three grouped residual reference fits, each with six-fold retuning, per training scope',
            'y_randomization': 'complete selected-procedure refitting per permutation and evaluation; separate authorization',
        },
        conditional_complexity_curves=False, reserved_responses_accessed=False,
        note='Full-budget individual comparison; no wall-time estimate. Ranking counts assume compatible shared rankings; final fitting and diagnostics excluded.')


def fit_final_predictor(*, X, y_log, development, experiment, request, study, output_root=None):
    """Reproduce the selected procedure on development, with explicit analysis gates."""
    from msr_qspr.validation import _canonical_cv_metadata
    from msr_qspr.study import check_run_metadata
    identity = check_run_metadata(study, 'final_model')
    metadata = _canonical_cv_metadata(development, 'unseen_surfactant')
    _aligned(X, y_log, metadata)
    if artifacts.membership_fingerprint(metadata.row_id) != study['scientific']['development_sha256']:
        raise ValueError('Final fitting requires the declared development observation membership.')
    context = dict(scientific_identity=config.scientific_identity(),
                   candidate_registry_sha256=artifacts.candidate_registry_fingerprint(),
                   study_sha256=identity, scientific_sha256=artifacts.fingerprint(study['scientific']),
                   development_sha256=study['scientific']['development_sha256'],
                   holdout_sha256=study['scientific']['holdout_sha256'])
    signature = artifacts.artifact_signature(X, y_log, {'request': request}, scientific_context=context,
                                           stage='final_model')
    directory = Path(output_root or artifacts.artifact_directory('final_model')) / ('selected_predictor__' + signature[:12])
    path = directory / 'model.joblib'
    manifest = directory / 'prediction_ad_bundle.json'
    specification = directory / 'selected_specification.json'
    complete = directory / 'final_fit_complete.json'
    required = [path, manifest, specification, directory / 'final_residual_atlas.json']
    with artifacts.exclusive_artifact(directory / 'final_model_writer'):
        if artifacts.completed_unit_valid(complete, signature, required):
            return json.loads(manifest.read_text())
        if complete.exists():
            raise ValueError('A completed final predictor failed verification; preserve it and investigate.')
        bank = TrainingBank(directory / 'shared' / 'training_bank', [experiment], context)
        model = bank.fit(X, y_log, metadata)[experiment.name]
        schema = model.named_steps['schema']
        fitted_spec = dict(model.fit_metadata_, **context, experiment=experiment.to_dict(),
                           development_ids=metadata.row_id.tolist(), training_seed=config.SEED,
                           identity_sha256=signature)
        model.fit_metadata_ = fitted_spec
        from msr_qspr.final import _attach_final_reliability
        extra = _attach_final_reliability(model, X, y_log, metadata, experiment,
                                           directory, context, signature)
        analysis_support = dict(reliability='available', interpretation='available',
                                y_randomization='complete_procedure', holdout='confirmation_required')
        external = {}
        if experiment.predictor_family == 'abraham_physicochemical':
            from msr_qspr.abraham import verify_files
            external = {'abraham_descriptors': {'files_sha256': verify_files(),
                'precedence': ['curated', 'optional_provider_estimate', 'training_median_imputation']}}
        _write(path, model, signature)
        _write(specification, fitted_spec, signature)
        bundle = dict(context, format_version=3, status='frozen_validated', identity_sha256=signature,
            experiment=experiment.to_dict(),
            development_ids=metadata.row_id.tolist(), model_file=path.name,
            model_sha256=artifacts.file_sha256(path),
            target={'name': 'log10_MSR', 'MSR_units': 'mol/mol'},
            inference={'fit_on_load': False, 'counterion_policy': experiment.counterion_policy,
                       'temperature_policy': experiment.temperature_policy,
                       'raw_predictors': list(schema.columns_),
                       'required_reference_properties': [c for c in config.LOG10 if c in schema.columns_],
                       'external_descriptors': external},
            analysis_support=analysis_support, **extra,
            artifacts_sha256={p.name: artifacts.file_sha256(p)
                              for p in required if p != manifest})
        _write(manifest, bundle, signature)
        artifacts.publish_completed_unit(complete, signature, required)
        artifacts.record_cache(complete, signature)
        return json.loads(manifest.read_text())
