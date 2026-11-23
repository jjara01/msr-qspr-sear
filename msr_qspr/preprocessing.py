"""Predictor eligibility, fitted schemas and training-only pipelines."""

from msr_qspr import config
from typing import List, Tuple
import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.pipeline import Pipeline
from msr_qspr.transformers import (
    DropMostlyNaN,
    DataFrameImputer,
    SafeLog10,
    DataFrameVarianceThreshold,
    CorrelationFilter,
    make_encoder_df,
)
from msr_qspr.config import CATEGORICAL, LOG10, PREDICTOR_CONTRACT, STRUCTURE_COLUMNS
from msr_qspr.chemistry import (descriptor_names, _descriptor_fingerprint, descriptor_generator_signature)
from sklearn.base import clone
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Ridge
from sklearn.utils.validation import check_is_fitted
from msr_qspr.transformers import ColumnSubset

TARGET_COL = "MSR"

REFERENCE_PREDICTORS = tuple(CATEGORICAL + LOG10 + ["temperature_C"])
EXTERNAL_DESCRIPTORS = config.ABRAHAM_DESCRIPTORS
DISQUALIFIED_DESCRIPTORS = frozenset(
    prefix + name for prefix in ('Cont_', 'Surf_')
    for name in config.DESCRIPTOR_POLICY['excluded_names'])


def build_predictor_matrix(df: pd.DataFrame, *, include_temperature=False) -> pd.DataFrame:
    """Project admissible inputs; every other field stays outside X.

    CMC is a shared reference in mol/L; WaterSolubility is in g/L.
    Temperature is optional and requires assay provenance; conversion MWs are metadata.
    Descriptor provenance binds exact names and values to represented structures.
    """
    if not df.columns.is_unique or not all(isinstance(c, str) for c in df.columns):
        raise ValueError("Predictor inputs require unique string columns.")
    desc = sorted(set(df.columns) & set(descriptor_names()))
    provenance = df.attrs.get("descriptor_provenance", {})
    if desc and (provenance.get("contract") != PREDICTOR_CONTRACT
                 or provenance.get("generator_sha256") != descriptor_generator_signature()
                 or provenance.get("columns") != desc
                 or not set(STRUCTURE_COLUMNS) <= set(df.columns)
                 or provenance.get("fingerprint") != _descriptor_fingerprint(df, desc)):
        raise ValueError("Mordred columns require verified structure-only descriptor provenance.")
    external = [c for c in EXTERNAL_DESCRIPTORS if c in df]
    if external and external != list(EXTERNAL_DESCRIPTORS):
        raise ValueError("Abraham descriptors must be attached as one complete set.")
    columns = ([c for c in REFERENCE_PREDICTORS if c in df
                and (c != "temperature_C" or include_temperature)]
               + [c for c in desc if c not in DISQUALIFIED_DESCRIPTORS] + external)
    X = df.loc[:, columns].copy()
    X.attrs = {"predictor_contract": PREDICTOR_CONTRACT,
               "counterion_policy": provenance.get("counterion_policy")}
    for col in LOG10:
        if col not in X:
            continue
        values = pd.to_numeric(X[col], errors="raise")
        if ((values.notna() & ~np.isfinite(values)) | (values <= 0)).any():
            raise ValueError(f"{col} must be positive or missing in its reference units.")
        if col == "CMC" and "CMC_predefined_M" in df:
            reference = pd.to_numeric(df["CMC_predefined_M"], errors="raise")
            if not np.isclose(values, reference, rtol=1e-12, atol=0, equal_nan=True).all():
                raise ValueError("CMC differs from its shared reference; assay provenance must match the scientific inputs.")
        X[col] = values
    if include_temperature:
        _validate_assay_temperature(df)
        X["temperature_C"] = pd.to_numeric(df["temperature_C"], errors="raise")
    if len(X.columns):
        validate_predictor_matrix(X)
    return X


def _validate_assay_temperature(df):
    required = {"temperature_C", "temperature_mode", "temperature_value_1", "temperature_value_2"}
    if not required <= set(df):
        raise ValueError("Temperature requires Celsius values and explicit MSR assay provenance.")
    mode = df["temperature_mode"]
    if not mode.isin(["reported_C", "reported_K", "range_C", "room_temperature", "not_reported"]).all():
        raise ValueError("Unknown MSR assay temperature provenance.")
    value = pd.to_numeric(df["temperature_C"], errors="raise")
    first = pd.to_numeric(df["temperature_value_1"], errors="raise")
    second = pd.to_numeric(df["temperature_value_2"], errors="raise")
    expected = first.copy()
    expected.loc[mode == "reported_K"] -= 273.15
    expected.loc[mode == "range_C"] = (first + second).loc[mode == "range_C"] / 2
    qualitative = mode.isin(["room_temperature", "not_reported"])
    expected.loc[qualitative] = np.nan
    if (not np.isclose(value, expected, rtol=0, atol=1e-8, equal_nan=True).all()
            or (value.dropna() <= -273.15).any()
            or (~np.isfinite(value.dropna())).any()
            or value.loc[~qualitative].isna().any()
            or first.loc[~qualitative].isna().any()
            or (mode.eq('range_C') & (second.isna() | second.lt(first))).any()
            or first.loc[qualitative].notna().any()
            or second.loc[~mode.eq('range_C')].notna().any()):
        raise ValueError("Temperature disagrees with its MSR assay provenance.")


def validate_predictor_matrix(X):
    """Reject bypasses of assembly before preprocessing, caching or fitting."""
    if not isinstance(X, pd.DataFrame) or not X.columns.is_unique:
        raise ValueError("Predictors require a DataFrame with unique columns.")
    unknown = ((set(X.columns) - set(REFERENCE_PREDICTORS) - set(descriptor_names()) - set(EXTERNAL_DESCRIPTORS))
               | (set(X.columns) & DISQUALIFIED_DESCRIPTORS))
    if unknown or X.empty:
        raise ValueError(f"Ineligible predictor columns or empty matrix: {sorted(unknown)}")
    if ((set(X.columns) - set(REFERENCE_PREDICTORS) or "temperature_C" in X)
            and X.attrs.get("predictor_contract") != PREDICTOR_CONTRACT):
        raise ValueError("Descriptor matrix must come from verified predictor assembly.")
    for col in X:
        if col in CATEGORICAL:
            if not X[col].map(lambda value: isinstance(value, str)).all():
                raise ValueError(f"Chemical class labels must be strings: {col}")
            continue
        if not pd.api.types.is_numeric_dtype(X[col]):
            raise ValueError(f"Predictor must be numeric: {col}")
        if np.isinf(X[col]).any() or (col in LOG10 and (X[col] <= 0).any()):
            raise ValueError(f"Invalid predictor values or units: {col}")


def validate_selected_features(columns):
    allowed = (set(REFERENCE_PREDICTORS) | set(EXTERNAL_DESCRIPTORS)
               | (set(descriptor_names()) - DISQUALIFIED_DESCRIPTORS))
    invalid = [c for c in columns if c not in allowed
               and not any(c.startswith(cat + "_") for cat in CATEGORICAL)]
    if invalid:
        raise ValueError(f"Ineligible selected predictors: {invalid}")


class PredictorSchema(BaseEstimator, TransformerMixin):
    """Persist the raw training schema and enforce it before learned transforms."""
    def fit(self, X, y=None):
        validate_predictor_matrix(X)
        self.columns_ = tuple(X.columns)
        self.counterion_policy_ = X.attrs.get("counterion_policy")
        self.contract_ = PREDICTOR_CONTRACT
        return self

    def transform(self, X):
        validate_predictor_matrix(X)
        if self.contract_ != PREDICTOR_CONTRACT:
            raise ValueError("Stored predictor contract is incompatible.")
        if (set(X.columns) != set(self.columns_)
                or X.attrs.get("counterion_policy") != self.counterion_policy_):
            raise ValueError("Prediction inputs differ from the fitted predictor schema or structure policy.")
        return X.loc[:, list(self.columns_)].copy()

    def get_feature_names_out(self, input_features=None):
        return np.asarray(self.columns_, dtype=object)


def require_predictor_schema(pipe):
    """Reject unversioned stored models before inference or reference computation."""
    schema = pipe.named_steps.get("schema")
    if (not isinstance(schema, PredictorSchema)
            or getattr(schema, "contract_", None) != PREDICTOR_CONTRACT
            or not getattr(schema, "columns_", ())
            or pipe.steps[0][0] != "schema"):
        raise ValueError("Model artifact has no compatible predictor contract. A schema-validated model and matching training references are required.")
    return schema


def build_design_matrices(
    df_full: pd.DataFrame,
    *, include_temperature=False,
) -> Tuple[pd.DataFrame, pd.Series, pd.DataFrame]:
    """Split full dataframe into X (features), y (target), and metadata."""
    if TARGET_COL not in df_full.columns:
        raise KeyError(f"Target column '{TARGET_COL}' not found in dataframe.")

    X = build_predictor_matrix(df_full, include_temperature=include_temperature)
    y = pd.to_numeric(df_full[TARGET_COL], errors="raise").copy()
    if y.isna().any() or not np.isfinite(y).all() or not y.gt(0).all():
        raise ValueError("MSR must be finite and positive for the log10(MSR) target.")
    meta = df_full.drop(columns=list(X.columns) + [TARGET_COL]).copy()
    meta.attrs = {}

    return X, y, meta


def infer_feature_roles(X: pd.DataFrame) -> Tuple[List[str], List[str]]:
    """Infer categorical and to log10 columns from X."""
    categorical_cols = [c for c in CATEGORICAL if c in X.columns]
    log10_cols = [c for c in LOG10 if c in X.columns]
    return categorical_cols, log10_cols


def project_experiment_predictors(X: pd.DataFrame, experiment) -> pd.DataFrame:
    """Project a verified matrix to the declared inputs without learning feature filters."""
    validate_predictor_matrix(X)
    registered = set(descriptor_names())
    molecular = [c for c in X if c in registered]
    if experiment.descriptor_panel is not None:
        columns = list(experiment.descriptor_panel) + list(experiment.required_reference_properties)
    else:
        if not molecular:
            raise ValueError('The experiment requires verified molecular descriptors.')
        columns = molecular + list(experiment.required_reference_properties)
    missing = set(columns) - set(X)
    if missing:
        hint = (' Curated Abraham descriptors must be attached before predictor assembly.'
                if missing & set(config.ABRAHAM_DESCRIPTORS) else '')
        raise ValueError(f'Missing required experiment predictors: {sorted(missing)}.{hint}')
    result = X.loc[:, columns].copy()
    if experiment.predictor_family == 'physicochemical':
        # A property-only model must not acquire a hidden structural input requirement.
        result.attrs['counterion_policy'] = None
    elif result.attrs.get('counterion_policy') != experiment.counterion_policy:
        raise ValueError('Experiment representation differs from verified descriptor provenance.')
    validate_predictor_matrix(result)
    return result


def build_preprocessor(X: pd.DataFrame) -> Tuple[Pipeline, List[str], List[str]]:
    """
    Build the preprocessing pipeline for features.

    Returns the fitted roles (categorical/log10) together with the pipeline.
    The pipeline:
    - drops columns with too many NaNs
    - imputes numeric-like columns
    - applies log10 to selected columns
    - one-hot encodes categoricals and passes the rest
    - removes zero-variance features
    - filters highly correlated features
    """
    validate_predictor_matrix(X)
    categorical_cols, log10_cols = infer_feature_roles(X)

    encoder = make_encoder_df(categorical_cols)

    preprocessor = Pipeline(
        steps=[
            ("schema", PredictorSchema()),
            ("structure_support", StructureSupport()),
            ("drop_sparse", DropMostlyNaN(threshold=config.MISSINGNESS_THRESHOLD)),
            ("imputer", DataFrameImputer()),
            ("log10", SafeLog10(columns=log10_cols)),
            ("ohe", encoder),
            ("var", DataFrameVarianceThreshold(threshold=config.VARIANCE_THRESHOLD)),
            ("corr", CorrelationFilter(threshold=config.CORRELATION_THRESHOLD)),
        ]
    )

    return preprocessor, categorical_cols, log10_cols


class StructureSupport(BaseEstimator, TransformerMixin):
    """Fit descriptor support on distinct training structures for each role."""
    def fit(self, X, y=None):
        groups = X.attrs.get('training_structures')
        molecular = [c for c in X if c.startswith(('Cont_', 'Surf_'))]
        if molecular and (not X.index.is_unique or not isinstance(groups, dict)):
            raise ValueError('Molecular ranking requires aligned training structure identities.')
        keep = [c for c in X if c not in molecular and not c.startswith('Pair_')]
        self.support_ = {}
        for prefix in ('Cont_', 'Surf_'):
            columns = [c for c in molecular if c.startswith(prefix)]
            if not columns:
                continue
            mapping = groups.get(prefix, {})
            if any(index not in mapping or not isinstance(mapping[index], str) or not mapping[index]
                   for index in X.index):
                raise ValueError('Missing training structure identity for molecular ranking.')
            identities = pd.Series([mapping[index] for index in X.index], index=X.index)
            unique = X.loc[~identities.duplicated(), columns]
            finite = np.isfinite(unique.to_numpy(dtype=float)).sum(axis=0)
            for column, count in zip(columns, finite):
                self.support_[column] = {'finite': int(count), 'structures': len(unique)}
                if 5 * int(count) >= 4 * len(unique):
                    keep.append(column)
        for column in (c for c in X if c.startswith('Pair_')):
            count = int(np.isfinite(X[column].to_numpy(dtype=float)).sum())
            self.support_[column] = {'finite_rows': count, 'rows': len(X)}
            if 5 * count >= 4 * len(X):
                keep.append(column)
        keep = set(keep)
        self.keep_cols_ = tuple(c for c in X if c in keep)
        if not self.keep_cols_:
            raise ValueError('No descriptors meet training structure support.')
        return self

    def transform(self, X):
        check_is_fitted(self, 'keep_cols_')
        return X.loc[:, list(self.keep_cols_)].copy()


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
    preprocessor = _build_trial_invariant_preprocessor(categorical_cols, log10_cols)
    fitted = _build_post_preprocessing_pipeline(estimator, selected_features)
    return Pipeline(preprocessor.steps + fitted.steps)


def _build_trial_invariant_preprocessor(categorical_cols: List[str], log10_cols: List[str]) -> Pipeline:
    """Build transformations whose fitted state is independent of trial parameters and k."""
    if not set(categorical_cols) <= set(CATEGORICAL) or not set(log10_cols) <= set(LOG10):
        raise ValueError("Preprocessing roles differ from the predictor contract.")
    return Pipeline([
        ("schema", PredictorSchema()),
        ("drop_sparse", DropMostlyNaN(threshold=config.MISSINGNESS_THRESHOLD)),
        ("imputer", DataFrameImputer()),
        ("log10", SafeLog10(columns=log10_cols)),
        ("ohe", make_encoder_df(categorical_cols)),
    ])


def _build_post_preprocessing_pipeline(estimator, selected_features: List[str]) -> Pipeline:
    """Fit trial-specific feature selection, optional scaling and estimator."""
    validate_selected_features(selected_features)
    steps = [("subset", ColumnSubset(columns=selected_features))]
    if isinstance(estimator, Ridge):
        steps.append(("scaler", StandardScaler()))
    steps.append(("model", clone(estimator)))
    return Pipeline(steps)
