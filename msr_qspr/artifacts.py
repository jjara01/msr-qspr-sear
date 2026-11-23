"""Artifact layout, atomic publication, identity checks and deterministic recovery."""

from pathlib import Path
from functools import lru_cache
from contextlib import contextmanager
import hashlib
import importlib.metadata
import json
import os
import platform
import re
import tempfile
from filelock import FileLock
import pandas as pd
from msr_qspr import config


# One output directory per scientific stage; restartable worker state is kept apart.
STAGE_DIRECTORIES = {
    'characterization': 'characterization', 'comparison': 'comparison', 'final_model': 'final_model',
    'interpretation': 'interpretation', 'reliability': 'reliability',
    'y_randomization': 'y_randomization', 'holdout': 'holdout', 'screening': 'screening',
}
WORKSPACE_DIRECTORIES = {
    'runs': 'comparison', 'descriptor_cache': 'descriptor_cache',
    'reliability_calibration': 'reliability', 'y_randomization_units': 'y_randomization',
}


def workspace_directory(workspace=None, *, root=None):
    """Worker outputs, checkpoints and caches; not part of the published stage outputs."""
    if workspace is not None:
        return Path(workspace)
    if root is None and config.WORKSPACE_DIR is not None:
        return Path(config.WORKSPACE_DIR)
    return Path(config.ARTIFACTS_DIR if root is None else root) / 'workspace'


def artifact_directory(stage, *, study_id=None, root=None, workspace=None):
    """Resolve the output directory of a stage, or of identity-scoped worker state."""
    if stage in WORKSPACE_DIRECTORIES:
        base = workspace_directory(workspace, root=root) / WORKSPACE_DIRECTORIES[stage]
        if stage == 'descriptor_cache':
            return base
        if not study_id or not re.fullmatch(r'[a-zA-Z0-9_-]+', study_id):
            raise ValueError('Workspace state requires a safe study identity.')
        return base / study_id
    if stage not in STAGE_DIRECTORIES:
        raise ValueError(f'Unknown artifact stage: {stage}')
    return Path(config.ARTIFACTS_DIR if root is None else root) / STAGE_DIRECTORIES[stage]


def protect_stage_destination(directory, identity, names):
    """Refuse to mix signed outputs of different executions in one stage directory."""
    directory = Path(directory)
    for name in names:
        path = directory / name
        provenance = path.with_name(path.name + '.provenance.json')
        if (path.exists() or provenance.exists()) and not cache_matches(path, identity):
            raise ValueError(f'{directory} contains unverified or incompatible {name}; '
                             'preserve it and use a separate artifacts directory.')


def bundle_directory(artifacts=None):
    """Resolve the one fitted bundle under final_model/, or the MSR_BUNDLE_DIR override."""
    override = os.getenv('MSR_BUNDLE_DIR')
    if override:
        return Path(override)
    root = artifact_directory('final_model', root=artifacts)
    candidates = sorted(p.parent for p in root.glob('*/prediction_ad_bundle.json'))
    if len(candidates) > 1:
        raise ValueError(f'Several fitted bundles exist in {root}; select one with --bundle or MSR_BUNDLE_DIR.')
    return candidates[0] if candidates else root


def software_signature(stage):
    """Explicit transitive scientific dependencies for one execution stage."""
    optional = {}
    if stage in {'interpretation', 'holdout'}:
        for name in ('shap', 'numba', 'llvmlite'):
            try:
                optional[name] = importlib.metadata.version(name)
            except importlib.metadata.PackageNotFoundError:
                optional[name] = None
    return {'environment': environment_signature(), 'stage_packages': optional,
            'runtime_threads': {name: os.getenv(name) for name in
                ('MSR_N_JOBS', 'OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS',
                 'NUMEXPR_NUM_THREADS', 'BLIS_NUM_THREADS', 'VECLIB_MAXIMUM_THREADS')},
            'code_sha256': {name: hashlib.sha256((config.PROJECT_ROOT / name).read_bytes()).hexdigest()
                           for name in config.STAGE_CODE[stage]}}


@lru_cache(maxsize=1)
def environment_signature():
    """Installed numerical environment identity, without machine-specific paths."""
    from rdkit import rdBase
    packages = {}
    for name in ('numpy', 'pandas', 'scipy', 'scikit-learn', 'optuna', 'xgboost',
                 'mordred', 'mordredcommunity', 'rdkit', 'rdkit-pypi', 'networkx', 'six', 'joblib', 'filelock'):
        try:
            packages[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            packages[name] = None
    return dict(python=platform.python_version(), platform=platform.system(),
                machine=platform.machine(), rdkit=rdBase.rdkitVersion, packages=packages)


def artifact_signature(X, y, settings, *, scientific_context=None,
                       stage='development_comparison') -> str:
    """Identify input values, ordering, settings and the code producing an artifact."""
    digest = hashlib.sha256()
    context = scientific_context or {}
    digest.update(json.dumps(context, sort_keys=True).encode())
    digest.update(json.dumps(software_signature(stage), sort_keys=True).encode())
    digest.update(json.dumps(X.attrs, sort_keys=True, default=str).encode())
    for table in (X, y):
        digest.update(pd.util.hash_pandas_object(table, index=True).values.tobytes())
        schema = ([table.name, str(table.dtype)] if isinstance(table, pd.Series)
                  else [table.columns.tolist(), table.dtypes.astype(str).tolist()])
        digest.update(json.dumps(schema, default=str).encode())
    digest.update(json.dumps(settings, sort_keys=True, default=str).encode())
    return digest.hexdigest()


def file_sha256(path):
    """Hash large serialized models without loading them into memory."""
    digest = hashlib.sha256()
    with Path(path).open('rb') as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def cache_matches(path: Path, signature: str) -> bool:
    """Require the artifact's own checksum-bound receipt, including for shared rankings."""
    try:
        path = Path(path)
    except ValueError:
        return False
    if not path.exists():
        return False
    try:
        entry = json.loads(path.with_name(path.name + '.provenance.json').read_text())
    except (OSError, ValueError):
        return False
    try:
        return (isinstance(entry, dict) and entry.get('format_version') == 1
                and entry.get('signature') == signature
                and entry.get('sha256') == file_sha256(path))
    except OSError:
        return False


def record_cache(path: Path, signature: str) -> None:
    """Publish one receipt per artifact; distinct writers never merge a shared manifest."""
    path = Path(path)
    provenance = {
        'format_version': 1,
        'signature': signature, 'sha256': file_sha256(path),
    }
    target = path.with_name(path.name + '.provenance.json')
    target.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(mode='w', dir=target.parent, delete=False) as handle:
        temporary = Path(handle.name)
        handle.write(json.dumps(provenance, indent=2) + '\n')
    try:
        temporary.replace(target)
    finally:
        temporary.unlink(missing_ok=True)


@contextmanager
def atomic_artifact(path: Path):
    """Write an artifact completely before exposing its final filename."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(dir=path.parent, suffix=path.suffix, delete=False) as handle:
        temporary = Path(handle.name)
    try:
        yield temporary
        temporary.replace(path)
    finally:
        temporary.unlink(missing_ok=True)


@contextmanager
def exclusive_artifact(path: Path):
    """Serialize writers to one namespace on Linux, Windows and macOS."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with FileLock(path.with_name(path.name + '.lock'), timeout=-1):
        yield


def completed_unit_valid(path: Path, identity: str, required=None, *, member_identity=None,
                         allow_missing_sidecar=False) -> bool:
    """Validate a completed work unit and every checksummed constituent artifact."""
    try:
        path = Path(path)
        receipt = json.loads(path.read_text())
        if (receipt.get('format_version') != 1 or receipt.get('identity') != identity
                or not isinstance(receipt.get('artifacts'), dict) or not receipt['artifacts']):
            return False
        sidecar = path.with_name(path.name + '.provenance.json')
        if (sidecar.exists() or not allow_missing_sidecar) and not cache_matches(path, identity):
            return False
        if required is not None and set(receipt['artifacts']) != {
                os.path.relpath(item, Path(path).parent) for item in required}:
            return False
        unit_root = Path(path).parent.resolve()
        for name, checksum in receipt['artifacts'].items():
            artifact = Path(path).parent / name
            if not artifact.resolve().is_relative_to(unit_root):
                return False
            if not artifact.is_file() or file_sha256(artifact) != checksum:
                return False
            provenance = json.loads(artifact.with_name(artifact.name + '.provenance.json').read_text())
            if (provenance.get('format_version') != 1
                    or not isinstance(provenance.get('signature'), str) or not provenance['signature']
                    or (member_identity is not None and provenance['signature'] != member_identity)
                    or provenance.get('sha256') != checksum):
                return False
        return True
    except (OSError, KeyError, ValueError, TypeError, AttributeError):
        return False


def recover_completed_unit(path: Path, identity: str, required, *, member_identity=None) -> None:
    """Finish a missing receipt sidecar only after writer-side semantic validation under lock."""
    path = Path(path)
    if (path.with_name(path.name + '.provenance.json').exists()
            or not completed_unit_valid(path, identity, required,
                                        member_identity=member_identity, allow_missing_sidecar=True)):
        raise ValueError('Cannot recover a missing or invalid completed-unit sidecar.')
    record_cache(path, identity)
    if not completed_unit_valid(path, identity, required, member_identity=member_identity):
        raise ValueError('Recovered completed-unit sidecar failed verification.')


def publish_completed_unit(path: Path, identity: str, artifacts) -> None:
    """Publish completion last, after each independently signed artifact exists."""
    path = Path(path)
    unit_root = path.parent.resolve()
    members = {}
    for artifact in artifacts:
        artifact = Path(artifact)
        name = os.path.relpath(artifact, path.parent)
        if not artifact.resolve().is_relative_to(unit_root):
            raise ValueError(f'Artifact outside work-unit namespace: {artifact}')
        provenance = json.loads(artifact.with_name(artifact.name + '.provenance.json').read_text())
        if not cache_matches(artifact, provenance['signature']):
            raise ValueError(f'Cannot complete unit with unverified artifact: {name}')
        members[name] = file_sha256(artifact)
    with atomic_artifact(path) as temporary:
        temporary.write_text(json.dumps({'format_version': 1, 'identity': identity,
                                         'artifacts': dict(sorted(members.items()))}, indent=2) + '\n')


class RunPaths:
    """Paths and scientific identity owned by one run, safe to pass between workers."""

    def __init__(self, directory, scientific_context=None):
        self.directory = Path(directory)
        self.context = dict(scientific_context or {})
        self.stability = self.directory / 'rankings'
        self.curves = self.directory / 'feature_selection/k_curves'
        self.feature_summary = self.directory / 'feature_selection/summaries'
        self.figures = self.directory / 'feature_selection/figures'
        self.metrics = self.directory
        self.summary = self.directory

    def fold(self, number):
        """One namespace per independently recoverable outer fold."""
        if type(number) is not int or number < 1:
            raise ValueError('Outer fold must be a positive integer.')
        return self.directory / f'fold-{number:02d}'


def candidate_directory(run_root, scenario, repetition, name, identity):
    """Canonical path for a candidate repetition, independent of estimator type."""
    if (not all(re.fullmatch(r'[a-zA-Z0-9_-]+', value or '') for value in (scenario, name))
            or type(repetition) is not int or repetition < 1
            or not re.fullmatch(r'[0-9a-f]{64}', identity or '')):
        raise ValueError('Invalid candidate artifact identity.')
    return Path(run_root) / scenario / f'rep-{repetition:02d}' / f'{name}__{identity[:12]}'


def candidate_provenances(run_root):
    """Find candidate namespaces; shared training-bank artifacts live elsewhere."""
    return sorted(Path(run_root).glob('*/rep-*/*/run_provenance.json'))


def root_mean_squared_error(observed, predicted):
    """Calculate finite RMSE without overflow from squaring large residuals."""
    import numpy as np
    observed, predicted = np.asarray(observed, dtype=float), np.asarray(predicted, dtype=float)
    if (observed.shape != predicted.shape or not observed.size
            or not np.isfinite(observed).all() or not np.isfinite(predicted).all()):
        raise ValueError('RMSE requires matching nonempty finite arrays.')
    with np.errstate(over='ignore', invalid='ignore'):
        residual = observed - predicted
    if not np.isfinite(residual).all():
        raise ValueError('RMSE residuals exceed the finite floating-point range.')
    scale = float(np.max(np.abs(residual)))
    return 0.0 if scale == 0 else float(scale * np.sqrt(np.mean((residual / scale) ** 2)))


def validate_fold_metrics(predictions, metrics):
    """Require persisted fold statistics to agree with its signed predictions."""
    import numpy as np
    from sklearn.metrics import r2_score
    required = {'actual_log10_MSR', 'predicted_log10_MSR', 'OuterFold', 'Model'}
    if not required <= set(predictions) or predictions.empty:
        raise ValueError('Outer-fold predictions are missing required fields.')
    values = predictions[['actual_log10_MSR', 'predicted_log10_MSR']].to_numpy(dtype=float)
    if not np.isfinite(values).all():
        raise ValueError('Outer-fold predictions contain nonfinite values.')
    observed, predicted = values.T
    actual_metrics = {'RMSE_outer': root_mean_squared_error(observed, predicted),
                      'R2_outer': float(r2_score(observed, predicted))}
    if not all(np.isfinite(value) for value in actual_metrics.values()):
        raise ValueError('Outer-fold metrics exceed the finite floating-point range.')
    if (len(predictions.OuterFold.unique()) != 1 or len(predictions.Model.unique()) != 1
            or metrics.get('Fold') != predictions.OuterFold.iloc[0]
            or metrics.get('Model') != predictions.Model.iloc[0]
            or metrics.get('n_validation') != len(predictions)
            or any(not np.isclose(metrics.get(key, float('nan')), value, rtol=0, atol=1e-10)
                   for key, value in actual_metrics.items())):
        raise ValueError('Fold metrics disagree with persisted outer predictions.')
    return actual_metrics


def fingerprint(value):
    """SHA-256 of canonical JSON; reject nonfinite numerical specifications."""
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':'),
                                     allow_nan=False, default=str).encode()).hexdigest()


def candidate_registry_fingerprint():
    return fingerprint([experiment.to_dict() for experiment in config.study_experiments()])


def membership_fingerprint(ids):
    ids = list(ids)
    if not ids or len(ids) != len(set(ids)) or any(not isinstance(i, str) or not i for i in ids):
        raise ValueError('Population identities must be unique nonempty strings.')
    return hashlib.sha256(('\n'.join(sorted(ids)) + '\n').encode()).hexdigest()


def split_membership_hashes(train_ids, validation_ids, train_groups, validation_groups):
    """Bind result provenance to populations without saving partition assignments."""
    return {name + '_sha256': membership_fingerprint(values) for name, values in (
        ('train_ids', train_ids), ('validation_ids', validation_ids),
        ('train_groups', train_groups), ('validation_groups', validation_groups))}


def nested_membership_hashes(fold):
    return {'outer': split_membership_hashes(fold.train_ids, fold.validation_ids,
                                            fold.train_groups, fold.validation_groups),
            'inner': [dict(fold=child.fold, **split_membership_hashes(
                child.train_ids, child.validation_ids, child.train_groups, child.validation_groups))
                for child in fold.inner_folds]}


class MissingArtifact(LookupError):
    """A required stage output does not exist yet."""


class InvalidArtifact(ValueError):
    """A stage output exists but fails integrity or binding checks."""


def verify_signed_file(path, identity=None):
    """Check a file against its provenance sidecar; return the recorded execution identity."""
    path = Path(path)
    sidecar = path.with_name(path.name + '.provenance.json')
    if not path.is_file():
        raise MissingArtifact(f'Missing {path}.')
    if not sidecar.is_file():
        raise InvalidArtifact(f'{path} has no provenance sidecar.')
    try:
        record = json.loads(sidecar.read_text())
    except ValueError as error:
        raise InvalidArtifact(f'Unreadable provenance sidecar for {path}.') from error
    if record.get('format_version') != 1 or record.get('sha256') != file_sha256(path):
        raise InvalidArtifact(f'{path} differs from its recorded checksum.')
    if identity is not None and record.get('signature') != identity:
        raise InvalidArtifact(f'{path} was produced by a different execution than the selected model.')
    return record['signature']
