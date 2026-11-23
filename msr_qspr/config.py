"""Fixed scientific settings, candidate configurations and parameter spaces."""

from pathlib import Path
from dataclasses import asdict, dataclass
import math
import os
import re
from typing import Dict, Any
from optuna.distributions import IntDistribution, FloatDistribution, CategoricalDistribution

PROJECT_ROOT = Path(__file__).resolve().parents[1]

DATA_DIR = Path.cwd() / "data"
# Stage outputs live under ARTIFACTS_DIR; restartable worker state lives under WORKSPACE_DIR
# (default ARTIFACTS_DIR / 'workspace'). The command-line interface may override both.
ARTIFACTS_DIR = Path(os.getenv("MSR_ARTIFACTS_DIR") or Path.cwd() / "artifacts")
WORKSPACE_DIR = Path(os.environ["MSR_WORKSPACE_DIR"]) if os.getenv("MSR_WORKSPACE_DIR") else None


ANCHOR_COEFFICIENTS = (0.0, 0.5, 1.0)


@dataclass(frozen=True)
class ExperimentConfig:
    """Predictor and search contract; partition identity is supplied by the grouped runner.

    ``target_anchor='molar_solubility'`` fits log10(MSR) - beta * log10(Sw / MW_contaminant)
    and restores the offset at prediction, with beta selected from ANCHOR_COEFFICIENTS.
    ``linear_initialization`` adds an optional standardized Ridge initial estimator to the
    gradient boosting search; the default constant initialization remains a search option.
    """
    name: str
    predictor_family: str
    estimator: str
    search_method: str
    n_trials: int = 0
    k_min: int = 5
    k_max: int = 60
    ridge_alphas: tuple = (0.001, 0.01, 0.1, 1., 10., 100., 1000.)
    temperature_policy: str = 'temp_off'
    counterion_policy: str = 'counterion_strip'
    target_anchor: str = 'none'
    linear_initialization: bool = False

    def __post_init__(self):
        if not re.fullmatch(r'[a-z][a-z0-9_]*', self.name):
            raise ValueError('Experiment names must be descriptive lowercase identifiers.')
        if self.predictor_family not in PREDICTOR_FAMILIES:
            raise ValueError('Unknown predictor family.')
        if self.search_method not in {'none', 'grid', 'tpe'}:
            raise ValueError('Unknown search method.')
        if (type(self.k_min) is not int or type(self.k_max) is not int
                or not 1 <= self.k_min <= self.k_max):
            raise ValueError('Molecular count bounds must be positive ordered integers.')
        if (not isinstance(self.ridge_alphas, tuple) or not self.ridge_alphas
                or len(set(self.ridge_alphas)) != len(self.ridge_alphas)
                or any(not math.isfinite(a) or a <= 0 for a in self.ridge_alphas)):
            raise ValueError('Ridge candidates must be unique positive finite values.')
        if self.temperature_policy != 'temp_off':
            raise ValueError('Unknown temperature policy.')
        if self.counterion_policy != 'counterion_strip':
            raise ValueError('Unknown representation policy.')
        if self.estimator not in {'Linear Regression', 'Ridge', 'Random Forest',
                                  'Gradient Boosting', 'XGBoost'}:
            raise ValueError('Unknown estimator.')
        if self.target_anchor not in {'none', 'molar_solubility'} or type(self.linear_initialization) is not bool:
            raise ValueError('Unknown target anchor or initialization policy.')
        compact = self.predictor_family in {'compact_physicochemical', 'abraham_physicochemical'}
        if self.estimator == 'Linear Regression':
            valid = self.predictor_family == 'physicochemical' and self.search_method == 'none'
        elif self.estimator == 'Ridge':
            valid = self.search_method == ('grid' if compact else 'tpe') and self.predictor_family != 'physicochemical'
        elif self.estimator == 'Gradient Boosting':
            valid = self.search_method == 'tpe' and self.predictor_family != 'physicochemical'
        else:
            valid = self.search_method == 'tpe' and self.predictor_family in {'molecular', 'enriched'}
        valid = (valid and (self.target_anchor == 'none' or compact)
                 and (not self.linear_initialization or self.estimator == 'Gradient Boosting'))
        if (not valid or type(self.n_trials) is not int
                or self.search_method == 'tpe' and self.n_trials <= 0
                or self.search_method != 'tpe' and self.n_trials != 0):
            raise ValueError('Incompatible estimator, selection or search specification.')

    @property
    def selection_policy(self):
        return 'ranked_molecular' if self.predictor_family in {'molecular', 'enriched'} else 'fixed'

    @property
    def ranking_strategy(self):
        return 'linear' if self.estimator == 'Ridge' else 'rf'

    @property
    def required_reference_properties(self):
        return () if self.predictor_family == 'molecular' else ('CMC', 'WaterSolubility')

    @property
    def descriptor_panel(self):
        """Fixed descriptors preceding the reference properties; None for ranked families."""
        if self.selection_policy == 'ranked_molecular':
            return None
        return PREDICTOR_FAMILIES[self.predictor_family]

    @property
    def external_references(self):
        """External inputs a user must supply beyond the two chemical structures."""
        extra = ('Abraham descriptors',) if self.predictor_family == 'abraham_physicochemical' else ()
        return self.required_reference_properties + extra

    @property
    def trial_budget(self):
        if self.search_method == 'grid':
            return len(self.ridge_alphas) * (len(ANCHOR_COEFFICIENTS) if self.target_anchor != 'none' else 1)
        return self.n_trials

    def to_dict(self):
        return dict(asdict(self), selection_policy=self.selection_policy,
                    ranking_strategy=self.ranking_strategy,
                    required_reference_properties=self.required_reference_properties,
                    external_references=self.external_references,
                    trial_budget=self.trial_budget, k_step=1)


FIXED_MOLECULAR_PANEL = ('Cont_MW', 'Cont_SLogP', 'Cont_TopoPSA',
                         'Surf_MW', 'Surf_SLogP', 'Surf_TopoPSA')
ABRAHAM_DESCRIPTORS = ('Abraham_E', 'Abraham_S', 'Abraham_A', 'Abraham_B', 'Abraham_V')
# Fixed descriptor panels of each family; ranked families select from all descriptors.
PREDICTOR_FAMILIES = {
    'physicochemical': (),
    'molecular': None,
    'enriched': None,
    'compact_physicochemical': FIXED_MOLECULAR_PANEL,
    'abraham_physicochemical': FIXED_MOLECULAR_PANEL + ABRAHAM_DESCRIPTORS,
}


def study_experiments():
    """Return the candidate procedures, from reference properties to physically informed models."""
    tpe = dict(search_method='tpe', n_trials=60)
    return (
        ExperimentConfig('property_linear', 'physicochemical', 'Linear Regression', 'none'),
        ExperimentConfig('molecular_ridge', 'molecular', 'Ridge', **tpe),
        ExperimentConfig('molecular_gb', 'molecular', 'Gradient Boosting', **tpe),
        ExperimentConfig('enriched_rf', 'enriched', 'Random Forest', **tpe),
        ExperimentConfig('enriched_gb', 'enriched', 'Gradient Boosting', **tpe),
        ExperimentConfig('enriched_xgb', 'enriched', 'XGBoost', **tpe),
        ExperimentConfig('physicochemical_ridge', 'compact_physicochemical', 'Ridge', 'grid'),
        ExperimentConfig('physicochemical_gb', 'compact_physicochemical', 'Gradient Boosting', **tpe),
        ExperimentConfig('solubility_anchored_gb', 'compact_physicochemical', 'Gradient Boosting',
                         target_anchor='molar_solubility', **tpe),
        ExperimentConfig('abraham_anchored_ridge', 'abraham_physicochemical', 'Ridge', 'grid',
                         target_anchor='molar_solubility'),
        ExperimentConfig('abraham_anchored_gb', 'abraham_physicochemical', 'Gradient Boosting',
                         target_anchor='molar_solubility', **tpe),
        ExperimentConfig('abraham_hybrid_gb', 'abraham_physicochemical', 'Gradient Boosting',
                         target_anchor='molar_solubility', linear_initialization=True, **tpe),
    )


def is_study_candidate(name):
    return any(experiment.name == name for experiment in study_experiments())


def effective_input_count(experiment, selected_features):
    """Count estimator input coordinates."""
    return len(selected_features)


COMPARISON_CODE = ('msr_qspr/config.py', 'msr_qspr/abraham.py', 'msr_qspr/estimators.py', 'msr_qspr/artifacts.py', 'msr_qspr/chemistry.py', 'msr_qspr/validation.py', 'msr_qspr/study.py', 'msr_qspr/selection.py',
                  'msr_qspr/workflow.py', 'msr_qspr/models.py', 'msr_qspr/features.py',
                  'msr_qspr/transformers.py', 'msr_qspr/preprocessing.py', 'msr_qspr/data.py',
                  'msr_qspr/framework.py', 'msr_qspr/reporting.py', 'msr_qspr/status.py')
STAGE_CODE = {
    'development_comparison': COMPARISON_CODE,
    'final_model': COMPARISON_CODE + ('msr_qspr/reporting.py', 'msr_qspr/final.py',
                                     'msr_qspr/diagnostics.py', 'msr_qspr/reliability.py', 'msr_qspr/residuals.py'),
    'reliability': COMPARISON_CODE + ('msr_qspr/reporting.py', 'msr_qspr/diagnostics.py', 'msr_qspr/reliability.py',
                                     'msr_qspr/residuals.py'),
    'y_randomization': COMPARISON_CODE + ('msr_qspr/reporting.py', 'msr_qspr/diagnostics.py', 'msr_qspr/reliability.py', 'msr_qspr/residuals.py'),
    'interpretation': COMPARISON_CODE + ('msr_qspr/reporting.py', 'msr_qspr/diagnostics.py',
                                        'msr_qspr/final.py', 'msr_qspr/reliability.py', 'msr_qspr/residuals.py', 'msr_qspr/inference.py'),
    'holdout': COMPARISON_CODE + ('msr_qspr/reporting.py', 'msr_qspr/final.py',
                                 'msr_qspr/diagnostics.py', 'msr_qspr/reliability.py', 'msr_qspr/residuals.py', 'msr_qspr/inference.py'),
}


# Deterministic numerical settings.
SEED = 42
SCIENTIFIC_PROTOCOL = 'grouped-individual-qspr'
DESCRIPTOR_POLICY = {'version': 'adjacency-vr-exclusion-structural-and-pair-row-support',
                     'excluded_names': ['VR1_A', 'VR2_A', 'VR3_A'],
                     'minimum_unique_structure_finite_fraction': 0.8,
                     'minimum_pair_training_row_finite_fraction': 0.8}
TRAINING_POLICY = {
    'axes': ['unseen_surfactant', 'unseen_contaminant'], 'folds_per_axis': 3,
    'objective': 'worst_relative_rmse_to_fold_training_mean',
    'reference_floor': 1e-12, 'tie_breaks': ['mean_relative_rmse', 'trial_number'],
    'partition_seed': 'sha256_training_metadata_axis_seed42',
}

# Feature selection settings
B_REPEATS = 30
MAX_K_RANK = 80
RF_N_ESTIM = 200

CATEGORICAL = ["ContaminantType", "SurfactantType"]
LOG10 = ["CMC", "WaterSolubility"]
PREDICTOR_CONTRACT = "molecular-descriptors-and-reference-properties"
STRUCTURE_COLUMNS = ("SMILES_contaminant", "SMILES_surfactant")


def scientific_identity():
    return {'protocol': SCIENTIFIC_PROTOCOL, 'predictor_contract': PREDICTOR_CONTRACT,
            'descriptor_policy': DESCRIPTOR_POLICY['version']}

def _n_jobs() -> int:
    value = os.getenv("MSR_N_JOBS", "").strip()
    if value:
        return max(1, int(value))
    return -1


PARAM_SPACES: Dict[str, Dict[str, Any]] = {
    "Linear Regression": {},
    "Ridge": {"model__alpha": FloatDistribution(1e-3, 1e5, log=True)},
    "Random Forest": {
        "model__n_estimators":      IntDistribution(200, 500, step=100),
        "model__max_depth":         CategoricalDistribution([2, 3, 4, 5, 6, 8, 10]),
        "model__min_samples_split": IntDistribution(4, 40),
        "model__min_samples_leaf":  IntDistribution(2, 24),
        "model__max_features":      FloatDistribution(.2, 1.),
        "model__bootstrap":         CategoricalDistribution([True]),
    },
    "Gradient Boosting": {
        "model__n_estimators":      IntDistribution(100, 800, step=100),
        "model__learning_rate":     FloatDistribution(.005, .1, log=True),
        "model__max_depth":         IntDistribution(1, 4),
        "model__min_samples_split": IntDistribution(8, 40),
        "model__min_samples_leaf":  IntDistribution(3, 24),
        "model__subsample":         FloatDistribution(.5, 1.),
        "model__max_features":      FloatDistribution(.4, 1.),
        "model__loss":              CategoricalDistribution(["squared_error", "huber"]),
        "model__alpha":             FloatDistribution(0.85, 0.95),
    },
    "XGBoost": {
        "model__n_estimators":      IntDistribution(100, 600, step=100),
        "model__learning_rate":     FloatDistribution(.01, .1, log=True),
        "model__max_depth":         IntDistribution(1, 5),
        "model__min_child_weight":  IntDistribution(2, 40),
        "model__subsample":         FloatDistribution(.5, 1.),
        "model__colsample_bytree":  FloatDistribution(.4, 1.),
        "model__reg_alpha":         FloatDistribution(1e-3, 100., log=True),
        "model__reg_lambda":        FloatDistribution(1., 100., log=True),
        "model__gamma":             FloatDistribution(0., 5.),
    },
}


def parameter_spaces(experiment):
    """Resolve the declared estimator space, including fixed-panel Ridge grids.

    Anchored candidates add the solubility-anchor coefficient; Ridge-initialized
    boosting adds the initial Ridge penalty, where None keeps the constant initialization.
    """
    if experiment.search_method == 'grid':
        space = {'model__alpha': CategoricalDistribution(experiment.ridge_alphas)}
    else:
        space = dict(PARAM_SPACES[experiment.estimator])
    if experiment.target_anchor != 'none':
        space['model__anchor_coefficient'] = CategoricalDistribution(ANCHOR_COEFFICIENTS)
    if experiment.linear_initialization:
        space['model__initial_ridge_alpha'] = CategoricalDistribution((None,) + experiment.ridge_alphas)
    return space


GROUPED_OUTER_FOLDS = 5
GROUPED_INNER_FOLDS = 3

VALIDATION_SETTINGS = {
    'unseen_surfactant': {'group_column': 'active_structure_group', 'seeds': (42, 43, 44), 'role': 'primary'},
    'unseen_contaminant': {'group_column': 'contaminant_group', 'seeds': (42, 43, 44), 'role': 'primary'},
    'unseen_study': {'group_column': 'publication_group', 'seeds': (42,), 'role': 'complementary'},
}
ACTIVE_HOLDOUT = 'unseen_surfactant'
HOLDOUT_FRACTION = 0.15
HOLDOUT_SEED = 42

PRIMARY_SETTINGS = ('unseen_surfactant', 'unseen_contaminant')
COMMON_SELECTION_RULE = {
    'primary_metric': 'RMSE_OOF', 'settings': list(PRIMARY_SETTINGS),
    'definition': 'minimize_worst_relative_deterioration',
    'repetitions': 3,
    'setting_error': 'arithmetic_mean_of_repetition_pooled_observation_rmse',
    'reference_error': 'minimum_setting_error_across_all_declared_candidates',
    'relative_deterioration': 'setting_error / reference_error - 1',
    'tie_breaks': ['mean_relative_deterioration', 'mandatory_external_reference_count',
                   'median_effective_selected_input_count', 'configuration_lexical_order'],
    'floating_point_policy': 'binary64_exact_comparison_no_equivalence_margin',
}


AD_POLICY = {'neighbors': 7, 'distance_quantile': .95,
             'blocks': 'equal_weight_surfactant_and_contaminant_effective_fitted_coordinates',
             'reference_groups': 'withheld_chemical_axis_or_canonical_pair_for_publications',
             'group_distance': 'minimum_observation_distance',
             'threshold_population': 'mean_leave_chemical_group_out_distance_per_training_group',
             'response_support': 'closed_training_log10_MSR_range',
             'local_error_quantile': .90}

INTERPRETATION_POLICY = {'n_repeats': 10, 'seed': 42, 'role': 'descriptive_development'}
HOLDOUT_REPORTING = {'target': 'log10_MSR', 'metrics': ['R2', 'RMSE', 'MAE'], 'population': 'all_reserved'}

MISSINGNESS_THRESHOLD = 0.95
CORRELATION_THRESHOLD = 0.95
VARIANCE_THRESHOLD = 0.0

# Accepted scientific input identities; source-code history is not a runtime input.
SCIENTIFIC_INPUTS = {'data/msr_dataset.xlsx': '4e8a1f67053005c367040e9aed9e235def2c1efc8a2f01536813f0da08acb534'}

