"""Automatic computational metadata for the fixed Python study specification."""

import json
from msr_qspr import artifacts, config
from msr_qspr.artifacts import fingerprint, membership_fingerprint
from msr_qspr.config import VALIDATION_SETTINGS, COMMON_SELECTION_RULE

COMPARISON = 'development_comparison'

def work_units():
    """Enumerate independent candidate x outer-fold units without reading scientific data.

    Each unit maps to: python -m msr_qspr compare --run-root RUN_ROOT --scenario S
    --repetition R --outer-fold F --candidate C. Any scheduler may execute units in any
    order; repetition summaries are published when their five folds verify.
    """
    units = [{'scenario': scenario, 'repetition': repetition, 'outer_fold': fold, 'candidate': experiment.name}
             for scenario, setting in VALIDATION_SETTINGS.items()
             for repetition in range(1, len(setting['seeds']) + 1)
             for fold in range(1, config.GROUPED_OUTER_FOLDS + 1)
             for experiment in config.study_experiments()]
    return {'candidate_registry_sha256': artifacts.candidate_registry_fingerprint(),
            'candidates': [e.name for e in config.study_experiments()],
            'n_units': len(units), 'units': units}


def experiment_specification(raw):
    """Reconstruct a declared experiment and reject inconsistent derived metadata."""
    from dataclasses import fields
    keys = {f.name for f in fields(config.ExperimentConfig)}
    values = {k: raw[k] for k in keys if k in raw}
    if 'ridge_alphas' in values:
        values['ridge_alphas'] = tuple(values['ridge_alphas'])
    result = config.ExperimentConfig(**values)
    if json.loads(json.dumps(result.to_dict())) != raw:
        raise ValueError('Verified estimator specification is internally inconsistent.')
    return result


def final_tuning_policy(experiment=None):
    """Describe the prescribed full-development search without fitting a model."""
    return {'settings': list(config.PRIMARY_SETTINGS), 'seed': config.SEED,
            'folds_per_axis': config.GROUPED_INNER_FOLDS, 'training_policy': dict(config.TRAINING_POLICY),
            'training_population': 'all_development', 'training_rows': 390,
            'search': ('joint_molecular_k_and_parameters' if experiment is not None and
                       experiment.selection_policy == 'ranked_molecular' else
                       'fixed_predictors' if experiment is not None else None),
            'experiment': json.loads(json.dumps(experiment.to_dict())) if experiment is not None else None,
            'trial_budget': experiment.trial_budget if experiment is not None else None,
            'execution': execution_specification([experiment]) if experiment is not None else None}


def execution_specification(experiments, diagnostic_k_grid=None, *, search_seed=config.SEED):
    """Describe the reused training code, transformations and estimator search."""
    from msr_qspr import preprocessing as sp
    from msr_qspr.transformers import DropMostlyNaN, DataFrameVarianceThreshold, CorrelationFilter
    experiments = tuple(experiments)
    if not experiments:
        raise ValueError('Execution requires declared candidate specifications.')
    ranking = {'method': 'training_only_estimator_appropriate',
               'strategies': {e.name: e.ranking_strategy for e in experiments
                              if e.selection_policy == 'ranked_molecular'},
               'rf': {'method': 'training_only_bootstrap_rf', 'bootstrap_repeats': config.B_REPEATS,
                      'n_estimators': config.RF_N_ESTIM, 'max_rank': config.MAX_K_RANK,
                      'row_sampling': 'equal_mixture_inverse_surfactant_and_contaminant_frequency',
                      'tie_break': ['frequency', 'mean_importance', 'descriptor_name']},
               'linear': {'method': 'absolute_standardized_ridge_coefficients', 'alpha': 1.,
                          'tie_break': ['absolute_coefficient', 'descriptor_name'],
                          'scaler': 'training_fitted_standard_scaler'}}
    return json.loads(json.dumps({
        'experiments': [e.to_dict() for e in experiments],
        'search_spaces': {e.name: config.parameter_spaces(e) for e in experiments},
        'predictor_eligibility': sp.PREDICTOR_CONTRACT,
        'transformations': {'log10': sp.LOG10, 'categorical': sp.CATEGORICAL,
                            'missingness': DropMostlyNaN(threshold=config.MISSINGNESS_THRESHOLD).get_params(),
                            'variance': DataFrameVarianceThreshold(threshold=config.VARIANCE_THRESHOLD).get_params(),
                            'correlation': CorrelationFilter(threshold=config.CORRELATION_THRESHOLD).get_params()},
        'ranking': ranking,
        'molecular_k_feasibility': 'refuse_without_clipping',
        'descriptor_policy': dict(config.DESCRIPTOR_POLICY),
        'inner_validation': 'three_grouped_folds_per_primary_axis',
        'objective': dict(config.TRAINING_POLICY), 'search_seed': search_seed,
        'diagnostic_k_grid': diagnostic_k_grid,
        'scientific_protocol': config.SCIENTIFIC_PROTOCOL,
        'adaptive_development': True, 'independent_prospective_validation': False,
    }, default=str))


def run_metadata(*, diagnostic_k_grid=None):
    """Record accepted inputs, populations, software and actual fixed execution settings."""
    from msr_qspr.data import verify_scientific_inputs, corpus_metadata
    from msr_qspr.validation import workflow_partitions
    experiments = config.study_experiments()
    if (diagnostic_k_grid is not None
            and (not diagnostic_k_grid or any(type(k) is not int or not 5 <= k <= 30
                                             for k in diagnostic_k_grid)
                 or list(diagnostic_k_grid) != sorted(set(diagnostic_k_grid)))):
        raise ValueError('Complexity diagnostics require ordered unique integer k values within [5, 30].')
    from msr_qspr.abraham import verify_files
    hashes = verify_scientific_inputs()
    external = verify_files()
    meta = corpus_metadata()
    plan = workflow_partitions(meta)
    dev, held = plan.holdout_positions(meta)
    return {'schema_version': 1,
        'scientific': {'files_sha256': hashes, 'external_inputs_sha256': {'abraham': external},
            'partition_sha256': plan.sha256,
            'partition_configuration': plan.configuration,
            'development_sha256': membership_fingerprint(meta.iloc[dev].row_id),
            'holdout_sha256': membership_fingerprint(meta.iloc[held].row_id)},
        'software': dict(artifacts.software_signature(COMPARISON),
            stages={stage: artifacts.software_signature(stage)
                    for stage in config.STAGE_CODE if stage != COMPARISON}),
        'stages': {
            COMPARISON: {'execution': execution_specification(experiments, diagnostic_k_grid),
                         'common_selection_rule': dict(COMMON_SELECTION_RULE)},
            'final_model': {'tuning_policy': final_tuning_policy(), 'selection_record': None},
            'reliability': {'ad_policy': dict(config.AD_POLICY)},
            'y_randomization': {'n_permutations': None, 'seed': config.SEED, 'settings': list(VALIDATION_SETTINGS)},
            'interpretation': {'policy': dict(config.INTERPRETATION_POLICY)},
            'holdout': {'ad_policy': dict(config.AD_POLICY), 'metrics_reporting': dict(config.HOLDOUT_REPORTING)},
        }}


def stage_software(study, stage):
    """Comparison excludes downstream-only modules; every other stage binds its own code."""
    software = study['software']
    if stage == COMPARISON:
        return {k: v for k, v in software.items() if k != 'stages'}
    return software.get('stages', {}).get(stage, software)


def run_fingerprint(study, stage):
    if stage not in config.STAGE_CODE:
        raise ValueError('Unknown scientific execution stage.')
    stages = [COMPARISON] + (['final_model'] if stage in {'final_model', 'holdout'} else [])
    if stage == 'holdout':
        stages.append('holdout')
    if stage in {'reliability', 'y_randomization', 'interpretation'}:
        stages += ['final_model', stage]
    return fingerprint({k: study[k] for k in ('schema_version', 'scientific')} |
                       {'software': stage_software(study, stage)} |
                       {'stages': {s: study['stages'][s] for s in stages}})


def check_run_metadata(study, stage):
    """Reject incompatible inputs, numerical settings or code at execution boundaries."""
    comparison = study.get('stages', {}).get(COMPARISON, {})
    if not comparison.get('execution') or comparison.get('common_selection_rule') != COMMON_SELECTION_RULE:
        raise ValueError('Run metadata differs from the fixed comparison specification.')
    actual = run_metadata(diagnostic_k_grid=comparison['execution'].get('diagnostic_k_grid'))
    if (study.get('schema_version') != 1 or study.get('scientific') != actual['scientific']
            or comparison != actual['stages'][COMPARISON]
            or stage_software(study, stage) != stage_software(actual, stage)):
        raise ValueError('Run scientific settings or software differ from runtime.')
    if stage != COMPARISON and not study['stages'].get('final_model', {}).get('selection_record'):
        raise ValueError('A completed common-model selection is required.')
    return run_fingerprint(study, stage)
