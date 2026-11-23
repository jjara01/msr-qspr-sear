"""Validation split strategies for the MSR development set."""

from __future__ import annotations

from msr_qspr.config import GROUPED_OUTER_FOLDS, GROUPED_INNER_FOLDS, VALIDATION_SETTINGS, ACTIVE_HOLDOUT
import hashlib
import json
from math import ceil
import re
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from threading import RLock
from typing import List, Optional, Sequence, Tuple
import numpy as np
import pandas as pd
from rdkit import Chem
from scipy.optimize import Bounds, LinearConstraint, milp
from scipy.sparse import csr_matrix
from msr_qspr import artifacts, config

IndexSplits = List[Tuple[np.ndarray, np.ndarray]]

# Target-free identity and class tables for the common development population.
_DEVELOPMENT_SIGNATURES = (
    '3fe05e9c2f484e139cc5168555b6b4e730aa65391ae13963cd2ceec69a0604bd',
    '2fd6eb6de2e33296800bcc104bf4ae0d252eba5f017167c04ee62270e744250e',
    '097e105102837de94fcc7658ecd95742307cabe3f3dd4d3f0855c3c7af244782',
)

PARTITION_ALGORITHM = 'chemical-lexicographic-milp-v1'
_PARTITION_LOCK = RLock()
_IDENTITY_COLUMNS = ('row_id', 'intake_row', 'source_doi', 'source_ref',
                     'SMILES_surfactant', 'SMILES_contaminant',
                     'SurfactantType', 'ContaminantType')
_CANONICAL_COLUMNS = ['row_id', 'intake_row', 'publication_group', 'active_structure_group',
                      'contaminant_group', 'SurfactantType', 'ContaminantType']
_CORPUS_METADATA_SHA256 = '33e39592fd362840ecef8f9636e82b4a3cc5d16f1db8c9273b54cd9d5e13a705'


@dataclass(frozen=True)
class GroupedFold:
    """Indices are positional within the supplied population (outer train for inner folds)."""
    fold: int
    train_pos: np.ndarray
    validation_pos: np.ndarray
    train_ids: Tuple[str, ...]
    validation_ids: Tuple[str, ...]
    train_groups: Tuple[str, ...]
    validation_groups: Tuple[str, ...]


@dataclass(frozen=True)
class NestedGroupedFold(GroupedFold):
    scenario: str
    repetition: int
    outer_seed: int
    inner_seed: int
    inner_folds: Tuple[GroupedFold, ...]


def _canonical_cv_metadata(meta: pd.DataFrame, scenario: str) -> pd.DataFrame:
    """Rebuild scientific group identities from source and molecular structure."""
    if scenario not in VALIDATION_SETTINGS:
        raise ValueError(f"Unknown grouped validation scenario: {scenario}")
    required = ("row_id", "intake_row", "source_doi", "source_ref",
                "SMILES_surfactant", "SMILES_contaminant", "SurfactantType", "ContaminantType")
    if not meta.columns.is_unique or not set(required) <= set(meta):
        raise ValueError("Grouped validation requires complete, unique identity and chemical metadata.")
    frame = meta.loc[:, required].reset_index(drop=True).copy()
    if (frame.row_id.isna().any() or not frame.row_id.is_unique
            or frame.row_id.astype(str).str.strip().eq("").any()
            or not pd.api.types.is_integer_dtype(frame.intake_row)
            or frame.intake_row.isna().any() or not frame.intake_row.is_unique
            or not frame.row_id.eq(frame.intake_row.map(lambda i: f"msr_intake_{i:04d}")).all()):
        raise ValueError("Grouped validation requires unique observation IDs and intake rows.")
    for column in ("SurfactantType", "ContaminantType"):
        if frame[column].isna().any() or frame[column].astype(str).str.strip().eq("").any():
            raise ValueError(f"Grouped validation has invalid {column} metadata.")
    frame["publication_group"] = [
        _publication_group(doi, ref) for doi, ref in zip(frame.source_doi, frame.source_ref)
    ]
    structures = {smiles: _active_structure(smiles)
                  for smiles in frame.SMILES_surfactant.unique()}
    frame["active_structure_group"] = frame.SMILES_surfactant.map(structures)
    from msr_qspr.chemistry import canonicalize
    frame['contaminant_group'] = frame.SMILES_contaminant.map(lambda s: canonicalize(s, True))
    return frame


def _allocate_chemical_folds(frame: pd.DataFrame, group_column: str,
                             n_folds: int, seed: int) -> Tuple[GroupedFold, ...]:
    """Solve ordered, target-blind objectives without weighted science scores.

    In order: minimize avoidable class loss from training, minimize maximum
    absolute fold-size deviation, maximize validation class support, maximize
    occupied class-pair support, minimize total size deviation, then use a
    seed-dependent stable hash to break remaining ties. All earlier optima are
    fixed exactly before the next objective is solved. Singleton-group classes
    have unavoidable training losses and are not penalized.
    """
    if not isinstance(n_folds, int) or n_folds < 2:
        raise ValueError("Grouped validation requires at least two folds.")
    groups = sorted(frame[group_column].unique())
    if len(groups) < n_folds:
        raise ValueError(f"Grouped validation needs at least {n_folds} distinct groups.")
    codes = pd.Categorical(frame[group_column], categories=groups).codes
    sizes = np.bincount(codes, minlength=len(groups)).astype(int)
    chemical = [(column, value) for column in ("SurfactantType", "ContaminantType")
                for value in sorted(frame[column].unique())]
    cells = sorted(set(zip(frame.SurfactantType, frame.ContaminantType)))
    features = chemical + [("cell", cell) for cell in cells]
    incidence = []
    for column, value in features:
        present = (frame.SurfactantType.eq(value[0]) & frame.ContaminantType.eq(value[1])
                   if column == "cell" else frame[column].eq(value))
        incidence.append(np.bincount(codes, weights=present.astype(int),
                                     minlength=len(groups)) > 0)
    n_groups = len(groups)
    n_class = len(chemical)
    avoidable = [i for i in range(n_class) if incidence[i].sum() >= 2]
    nx = n_groups * n_folds
    nm = len(avoidable) * n_folds
    nc = len(features) * n_folds
    missing_start = nx
    cover_start = missing_start + nm
    deviation_start = cover_start + nc
    maximum = deviation_start + n_folds
    n_variables = maximum + 1
    lower, upper = np.zeros(n_variables), np.full(n_variables, np.inf)
    upper[:deviation_start] = 1
    upper[deviation_start:] = n_folds * len(frame)
    integrality = np.ones(n_variables)
    constraints, low, high = [], [], []

    def add(coefs, lo=-np.inf, hi=np.inf):
        row = np.zeros(n_variables)
        for idx, value in coefs.items():
            row[idx] = value
        constraints.append(row)
        low.append(lo)
        high.append(hi)

    def x(g, f):
        return g * n_folds + f

    for g in range(n_groups):
        add({x(g, f): 1 for f in range(n_folds)}, 1, 1)
    for f in range(n_folds):
        add({x(g, f): 1 for g in range(n_groups)}, lo=1)
        counts = {x(g, f): n_folds * int(sizes[g]) for g in range(n_groups)}
        add({**counts, deviation_start + f: -1}, hi=len(frame))
        add({**{i: -v for i, v in counts.items()}, deviation_start + f: -1},
            hi=-len(frame))
        add({deviation_start + f: 1, maximum: -1}, hi=0)

    for i, feature_id in enumerate(avoidable):
        supported = np.flatnonzero(incidence[feature_id])
        for f in range(n_folds):
            # A fold holding all supporting groups would erase this class from training.
            add({**{x(g, f): 1 for g in supported},
                 missing_start + i * n_folds + f: -len(supported)},
                hi=len(supported) - 1)
    for i, present in enumerate(incidence):
        supported = np.flatnonzero(present)
        for f in range(n_folds):
            add({**{x(g, f): 1 for g in supported},
                 cover_start + i * n_folds + f: -1}, lo=0)

    objectives = []
    loss = np.zeros(n_variables)
    loss[missing_start:cover_start] = 1
    objectives.append(loss)
    worst = np.zeros(n_variables)
    worst[maximum] = 1
    objectives.append(worst)
    for first, last in ((0, n_class), (n_class, len(features))):
        coverage = np.zeros(n_variables)
        coverage[cover_start + first * n_folds:cover_start + last * n_folds] = -1
        objectives.append(coverage)
    total = np.zeros(n_variables)
    total[deviation_start:maximum] = 1
    objectives.append(total)
    tie = np.zeros(n_variables)
    for g, key in enumerate(groups):
        for f in range(n_folds):
            digest = hashlib.sha256(f"{seed}|{key}|{f}".encode("utf-8")).digest()
            tie[x(g, f)] = int.from_bytes(digest[:4], "big") % 1000003
    objectives.append(tie)

    for objective in objectives:
        result = milp(objective, integrality=integrality, bounds=Bounds(lower, upper),
                      constraints=LinearConstraint(csr_matrix(np.array(constraints)), low, high),
                      options={"mip_rel_gap": 0.0, "presolve": True})
        if not result.success or result.x is None:
            raise ValueError(f"Grouped fold allocation failed: {result.message}")
        if (not np.isfinite(result.x).all()
                or not np.isclose(result.x, np.rint(result.x), atol=1e-6, rtol=0).all()):
            raise ValueError("Grouped fold allocation returned nonintegral decisions.")
        if objective is not tie:
            optimum = int(round(result.fun))
            if abs(result.fun - optimum) > 1e-4:
                raise ValueError("Grouped fold objective did not converge to an integer optimum.")
            add({i: v for i, v in enumerate(objective) if v}, optimum, optimum)

    decisions = np.rint(result.x[:nx]).reshape(n_groups, n_folds)
    if (not np.isin(decisions, [0, 1]).all() or not (decisions.sum(axis=1) == 1).all()):
        raise ValueError('Grouped allocation must assign every group to exactly one fold.')
    allocation = decisions.argmax(axis=1)
    group_values = frame[group_column].to_numpy()
    observation_ids = frame.row_id.astype(str).to_numpy()
    answer = []
    for f in range(n_folds):
        held_groups = set(groups[g] for g in range(n_groups) if allocation[g] == f)
        test = np.flatnonzero(np.isin(group_values, list(held_groups)))
        train = np.flatnonzero(~np.isin(group_values, list(held_groups)))
        answer.append(GroupedFold(f + 1, train, test, tuple(observation_ids[train]),
                                  tuple(observation_ids[test]),
                                  tuple(sorted(set(group_values[train]))),
                                  tuple(sorted(held_groups))))
    return tuple(answer)


def make_repeated_nested_grouped_cv(
    development: pd.DataFrame, scenario: str, *,
    outer_seeds: Optional[Sequence[int]] = None,
    n_outer: int = GROUPED_OUTER_FOLDS,
    n_inner: int = GROUPED_INNER_FOLDS,
) -> Tuple[NestedGroupedFold, ...]:
    """Return repeated outer folds with inner positions relative to each outer train.

    Input must already be the separately reserved development population. Only
    target-blind source, structure, chemical class and observation IDs are read.
    """
    frame = _canonical_cv_metadata(development, scenario)
    if (len(frame) != 390 or n_outer != GROUPED_OUTER_FOLDS
            or n_inner != GROUPED_INNER_FOLDS):
        raise ValueError("Primary grouped CV requires 390 development rows and 5 outer/3 inner folds.")
    _verify_development_metadata(frame)
    expected_seeds = VALIDATION_SETTINGS[scenario]['seeds']
    outer_seeds = expected_seeds if outer_seeds is None else tuple(outer_seeds)
    if outer_seeds != expected_seeds:
        raise ValueError(f'Grouped CV requires the declared repetition seeds {expected_seeds}.')
    group_column = VALIDATION_SETTINGS[scenario]['group_column']
    observed_partitions = set()
    result = []
    for repetition, seed in enumerate(outer_seeds, start=1):
        outer = _allocate_chemical_folds(frame, group_column, n_outer, seed)
        partition = frozenset(frozenset(split.validation_groups) for split in outer)
        if partition in observed_partitions:
            raise ValueError("Outer repetitions must differ beyond fold-label permutations.")
        observed_partitions.add(partition)
        for split in outer:
            inner_seed = seed * 1000 + split.fold
            local = frame.iloc[split.train_pos].reset_index(drop=True)
            inner = _allocate_chemical_folds(local, group_column, n_inner, inner_seed)
            result.append(NestedGroupedFold(
                split.fold, split.train_pos, split.validation_pos, split.train_ids,
                split.validation_ids, split.train_groups, split.validation_groups,
                scenario, repetition, seed, inner_seed, inner,
            ))
    return tuple(result)


def _verify_development_metadata(frame):
    """Retain the accepted target-free population and chemical identity checks."""
    if len(frame) != 390:
        raise ValueError("Primary grouped CV requires 390 development rows.")
    for columns, expected in zip(
        (("row_id", "intake_row", "publication_group", "active_structure_group"),
         ("row_id", "intake_row", "SurfactantType", "ContaminantType"),
         ("row_id", "intake_row", "contaminant_group")),
        _DEVELOPMENT_SIGNATURES,
    ):
        content = frame.sort_values("intake_row")[list(columns)].to_csv(
            sep="\t", index=False, lineterminator="\n"
        )
        if hashlib.sha256(content.encode("utf-8")).hexdigest() != expected:
            raise ValueError("Primary development membership or scientific metadata fingerprint mismatch.")


def partition_configuration():
    """Identify the target-blind algorithm, declared constraints and pinned solver."""
    import platform
    import scipy
    from scipy.optimize._highspy import _core as _highs_core
    from rdkit import rdBase
    return json.loads(json.dumps({
        'algorithm': PARTITION_ALGORITHM,
        'code_sha256': {name: hashlib.sha256(Path(__file__).with_name(name).read_bytes()).hexdigest()
                        for name in ('validation.py', 'chemistry.py')},
        'grouping_policy': 'canonical-active-structure-approved-counterions-v1',
        'holdout_seed': config.HOLDOUT_SEED, 'holdout_fraction': config.HOLDOUT_FRACTION,
        'settings': config.VALIDATION_SETTINGS,
        'outer_folds': GROUPED_OUTER_FOLDS, 'inner_folds': GROUPED_INNER_FOLDS,
        'inner_seed_rule': 'outer_seed * 1000 + outer_fold',
        'objectives': ['avoidable_training_class_loss', 'maximum_size_deviation',
                       'validation_class_coverage', 'validation_class_pair_coverage',
                       'total_size_deviation', 'seeded_hash_tie_break'],
        'solver': {'interface': 'scipy.optimize.milp', 'backend': 'bundled HiGHS',
                   'mip_rel_gap': 0.0, 'presolve': True,
                   'binary_sha256': hashlib.sha256(Path(_highs_core.__file__).read_bytes()).hexdigest()},
        'environment': {'python': platform.python_version(), 'numpy': np.__version__,
                        'scipy': scipy.__version__, 'rdkit': rdBase.rdkitVersion,
                        'platform': platform.system(), 'machine': platform.machine()},
    }))


def _bind_validation_ids(frame, group_column, validation_ids, number):
    ids = frame.row_id.to_numpy()
    mask = frame.row_id.isin(validation_ids).to_numpy()
    train, valid = np.flatnonzero(~mask), np.flatnonzero(mask)
    training_groups = tuple(sorted(frame.iloc[train][group_column].unique()))
    validation_groups = tuple(sorted(frame.iloc[valid][group_column].unique()))
    if (not len(train) or not len(valid) or len(valid) != len(validation_ids)
            or set(training_groups) & set(validation_groups)):
        raise ValueError('Grouped partition has incomplete identities or canonical group leakage.')
    return GroupedFold(number, train, valid, tuple(ids[train]), tuple(ids[valid]),
                       training_groups, validation_groups)


@dataclass(frozen=True)
class GroupedPartitionPlan:
    """Execution-local Python objects; memberships are never serialized or loaded."""
    development_ids: Tuple[str, ...]
    holdout_ids: Tuple[str, ...]
    nested: dict
    configuration: dict
    sha256: str

    def holdout_positions(self, corpus):
        ids = corpus.row_id
        if not ids.is_unique or set(ids) != set(self.development_ids + self.holdout_ids):
            raise ValueError('Grouped holdout observation universe mismatch.')
        return (np.flatnonzero(ids.isin(self.development_ids)),
                np.flatnonzero(ids.isin(self.holdout_ids)))

    def folds(self, development, scenario):
        """Rebind indices, including inner positions within the current outer train."""
        frame = _canonical_cv_metadata(development, scenario)
        _verify_development_metadata(frame)
        axis = VALIDATION_SETTINGS[scenario]['group_column']
        answer = []
        for original in self.nested[scenario]:
            fold = _bind_validation_ids(frame, axis, original.validation_ids, original.fold)
            local = frame.iloc[fold.train_pos].reset_index(drop=True)
            inner = tuple(_bind_validation_ids(local, axis, child.validation_ids, child.fold)
                          for child in original.inner_folds)
            answer.append(NestedGroupedFold(fold.fold, fold.train_pos, fold.validation_pos,
                fold.train_ids, fold.validation_ids, fold.train_groups, fold.validation_groups,
                scenario, original.repetition, original.outer_seed, original.inner_seed, inner))
        return tuple(answer)


def workflow_partitions(corpus, *, fresh=False):
    """Generate once per execution, sharing a target-free process-local plan.

    Call with fresh=True at a new library workflow boundary. CLI executions clear
    this memory at entry; a new Python process always performs fresh MILP solves.
    The caller must verify workbook bytes before supplying its metadata.
    """
    _holdout_metadata(corpus)
    frame = _canonical_cv_metadata(corpus, ACTIVE_HOLDOUT).sort_values('row_id').reset_index(drop=True)
    if artifacts.fingerprint(frame[_CANONICAL_COLUMNS].to_dict('records')) != _CORPUS_METADATA_SHA256:
        raise ValueError('Frozen corpus chemical metadata fingerprint mismatch.')
    rows = tuple(frame[list(_IDENTITY_COLUMNS)].itertuples(index=False, name=None))
    configuration = partition_configuration()
    with _PARTITION_LOCK:
        if fresh:
            _cached_partition_plan.cache_clear()
        return _cached_partition_plan(rows, artifacts.fingerprint(configuration))


@lru_cache(maxsize=1)
def _cached_partition_plan(rows, configuration_sha256):
    corpus = pd.DataFrame.from_records(rows, columns=_IDENTITY_COLUMNS)
    development, held = reserved_grouped_holdout(corpus)
    frame = corpus.iloc[development].reset_index(drop=True)
    nested = {scenario: make_repeated_nested_grouped_cv(frame, scenario)
              for scenario in VALIDATION_SETTINGS}
    hashes = {}
    for scenario, folds in nested.items():
        _validate_nested_plan(frame, scenario, folds)
        hashes[scenario] = membership_hashes(frame, folds)
    configuration = partition_configuration()
    if artifacts.fingerprint(configuration) != configuration_sha256:
        raise ValueError('Partition configuration changed during allocation.')
    dev_ids, held_ids = tuple(sorted(corpus.iloc[development].row_id)), tuple(sorted(corpus.iloc[held].row_id))
    identity = artifacts.fingerprint({'configuration': configuration, 'development': dev_ids,
                                      'holdout': held_ids, 'memberships': hashes})
    return GroupedPartitionPlan(dev_ids, held_ids, nested, configuration, identity)


def _validate_nested_plan(frame, scenario, folds):
    """Require exhaustive, group-isolated outer and parent-bound inner partitions."""
    frame = _canonical_cv_metadata(frame, scenario)
    seeds = VALIDATION_SETTINGS[scenario]['seeds']
    if len(folds) != len(seeds) * GROUPED_OUTER_FOLDS:
        raise ValueError('Incomplete grouped outer repetitions.')
    universe = set(frame.row_id)
    axis = VALIDATION_SETTINGS[scenario]['group_column']
    for repetition, seed in enumerate(seeds, 1):
        current = [f for f in folds if f.repetition == repetition]
        if (sorted(f.fold for f in current) != list(range(1, GROUPED_OUTER_FOLDS + 1))
                or set().union(*(set(f.validation_ids) for f in current)) != universe
                or sum(len(f.validation_ids) for f in current) != len(frame)):
            raise ValueError('Grouped outer folds do not cover development exactly once.')
        for fold in current:
            if (not len(fold.train_ids) or not len(fold.validation_ids)
                    or fold.scenario != scenario or fold.outer_seed != seed
                    or fold.inner_seed != seed * 1000 + fold.fold
                    or set(fold.train_ids + fold.validation_ids) != universe
                    or tuple(sorted(frame.iloc[fold.train_pos][axis].unique())) != fold.train_groups
                    or tuple(sorted(frame.iloc[fold.validation_pos][axis].unique())) != fold.validation_groups
                    or set(fold.train_groups) & set(fold.validation_groups)):
                raise ValueError('Grouped outer fold identity, seed or isolation mismatch.')
            grouped_inner_splits(fold, fold.train_pos, fold.validation_pos, frame.row_id)
            local = frame.iloc[fold.train_pos].reset_index(drop=True)
            if [f.fold for f in fold.inner_folds] != list(range(1, GROUPED_INNER_FOLDS + 1)):
                raise ValueError('Incomplete grouped inner fold identities.')
            for child in fold.inner_folds:
                if (tuple(local.iloc[child.train_pos].row_id) != child.train_ids
                        or tuple(local.iloc[child.validation_pos].row_id) != child.validation_ids
                        or set(local.iloc[child.train_pos][axis]) & set(local.iloc[child.validation_pos][axis])):
                    raise ValueError('Grouped inner positions or canonical group isolation mismatch.')


def grouped_cv_membership(development: pd.DataFrame,
                          folds: Sequence[NestedGroupedFold]) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Auditable validation membership. Training is its complement within each parent fold.

    Outer membership has one row per development observation per repetition;
    inner membership has one row per outer-training observation per outer fold.
    Both use observation-weighted units and retain canonical group IDs.
    """
    if not folds or len({fold.scenario for fold in folds}) != 1:
        raise ValueError("CV membership requires folds from one scenario.")
    scenario = folds[0].scenario
    meta = _canonical_cv_metadata(development, scenario)
    group = dict(zip(meta.row_id, meta[VALIDATION_SETTINGS[scenario]["group_column"]]))
    outer_rows, inner_rows = [], []
    for fold in folds:
        common = {"scenario": fold.scenario, "repetition": fold.repetition,
                  "outer_seed": fold.outer_seed, "outer_fold": fold.fold}
        stats = {"n_train": len(fold.train_ids), "n_validation": len(fold.validation_ids),
                 "n_train_groups": len(fold.train_groups),
                 "n_validation_groups": len(fold.validation_groups)}
        for obs in fold.validation_ids:
            outer_rows.append({**common, **stats, "observation_id": obs,
                               "canonical_group": group[obs]})
        for inner in fold.inner_folds:
            inner_common = {**common, "inner_seed": fold.inner_seed, "inner_fold": inner.fold}
            inner_stats = {"n_train": len(inner.train_ids), "n_validation": len(inner.validation_ids),
                           "n_train_groups": len(inner.train_groups),
                           "n_validation_groups": len(inner.validation_groups)}
            for obs in inner.validation_ids:
                inner_rows.append({**inner_common, **inner_stats, "observation_id": obs,
                                   "canonical_group": group[obs]})
    outer = pd.DataFrame(outer_rows).sort_values(
        ["repetition", "outer_fold", "observation_id"]).reset_index(drop=True)
    inner = pd.DataFrame(inner_rows).sort_values(
        ["repetition", "outer_fold", "inner_fold", "observation_id"]).reset_index(drop=True)
    return outer, inner


def grouped_inner_splits(fold: NestedGroupedFold, outer_train: np.ndarray,
                         outer_validation: np.ndarray,
                         observation_ids: Sequence[str]) -> IndexSplits:
    """Bind modeling consumers to the authoritative nested partition and row order."""
    ids = np.asarray(observation_ids, dtype=str)
    if (not np.array_equal(fold.train_pos, outer_train)
            or not np.array_equal(fold.validation_pos, outer_validation)
            or tuple(ids[outer_train]) != fold.train_ids
            or tuple(ids[outer_validation]) != fold.validation_ids
            or len(fold.inner_folds) != GROUPED_INNER_FOLDS):
        raise ValueError("Grouped fold indices or observation identities differ from the validation plan.")
    inner = [(f.train_pos, f.validation_pos) for f in fold.inner_folds]
    n = len(fold.train_ids)
    if (set(np.concatenate([v for _, v in inner])) != set(range(n))
            or sum(len(v) for _, v in inner) != n
            or any(len(tr) == 0 or len(va) == 0 or np.intersect1d(tr, va).size
                   or set(tr) | set(va) != set(range(n))
                   for tr, va in inner)):
        raise ValueError("Precomputed grouped inner folds are incomplete or overlapping.")
    return inner


# SHA-256 of the target-free identity table, sorted by intake_row, UTF-8 TSV/LF.
# Columns: row_id, intake_row, publication_group, active_structure_group (with header).
_OBSERVATION_GROUPS_SHA256 = "9d403d8d9fc205ec27a93e8f03311f95f524859e81040514ae9673677189e25b"


def _publication_group(doi, source_ref) -> str:
    doi = re.sub(r"^(?:https?://(?:dx\.)?doi\.org/|doi:\s*)", "", str(doi).strip().lower())
    if re.fullmatch(r"10\.\d{4,9}/\S+", doi):
        return "doi:" + doi
    if pd.isna(source_ref) or not str(source_ref).strip():
        raise ValueError("Publication grouping requires a DOI or source_ref.")
    return "ref:" + str(source_ref).strip()


def _active_structure(smiles: str) -> str:
    """Extract one carbon-containing fragment, preserving charge and stereochemistry."""
    mol = Chem.MolFromSmiles(str(smiles))
    if mol is None:
        raise ValueError("Invalid surfactant SMILES.")
    organic, counterions = [], []
    for fragment in Chem.GetMolFrags(mol, asMols=True):
        if any(atom.GetAtomicNum() == 6 for atom in fragment.GetAtoms()):
            organic.append(fragment)
        else:
            counterions.append(Chem.MolToSmiles(fragment, canonical=True, isomericSmiles=True))
    if (len(organic) != 1 or not set(counterions) <= {"[Cl-]", "[Br-]", "[Na+]", "[NH4+]"}
            or (counterions and sum(atom.GetFormalCharge() for atom in mol.GetAtoms()) != 0)):
        raise ValueError("Surfactant fragments violate the active-structure grouping policy.")
    return Chem.MolToSmiles(organic[0], canonical=True, isomericSmiles=True)


def _holdout_metadata(meta: pd.DataFrame) -> pd.DataFrame:
    columns = ["row_id", "intake_row", "source_doi", "source_ref", "SMILES_surfactant"]
    if not meta.columns.is_unique or not set(columns) <= set(meta.columns):
        raise ValueError("Reserved holdouts require unique columns and complete identity metadata.")
    identity = meta.loc[:, columns].copy()
    intake = identity["intake_row"]
    if (len(identity) != 459 or not pd.api.types.is_integer_dtype(intake)
            or intake.isna().any() or not intake.is_unique or not identity["row_id"].is_unique
            or not identity["row_id"].eq(intake.map(lambda i: f"msr_intake_{i:04d}")).all()):
        raise ValueError("Reserved holdouts require exactly 459 unique, compatible observation IDs.")
    identity["publication_group"] = [
        _publication_group(doi, ref)
        for doi, ref in zip(identity["source_doi"], identity["source_ref"])
    ]
    structures = {s: _active_structure(s) for s in identity["SMILES_surfactant"].unique()}
    identity["active_structure_group"] = identity["SMILES_surfactant"].map(structures)
    table = identity.sort_values("intake_row")[
        ["row_id", "intake_row", "publication_group", "active_structure_group"]
    ].to_csv(sep="\t", index=False, lineterminator="\n")
    if hashlib.sha256(table.encode("utf-8")).hexdigest() != _OBSERVATION_GROUPS_SHA256:
        raise ValueError("Observation universe or grouping identity fingerprint mismatch.")
    return identity


def reserved_grouped_holdout(meta: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray]:
    """Select ceil(15% of observations), withholding whole active-structure groups.

    Only frozen observation identities and target-free molecular metadata are read.
    Sorted groups and the fixed random seed make membership row-order invariant.
    """
    from msr_qspr.config import HOLDOUT_FRACTION, HOLDOUT_SEED

    identity = _holdout_metadata(meta)
    groups = sorted(identity.active_structure_group.unique())
    group_codes = pd.Categorical(identity.active_structure_group, categories=groups).codes
    sizes = np.bincount(group_codes, minlength=len(groups))
    n_test = ceil(len(identity) * HOLDOUT_FRACTION)
    objective = np.random.default_rng(HOLDOUT_SEED).random(len(groups))
    result = milp(objective, integrality=np.ones(len(groups)),
                  bounds=Bounds(0, 1),
                  constraints=LinearConstraint(sizes[np.newaxis, :], n_test, n_test),
                  options={'mip_rel_gap': 0.0})
    if not result.success or result.x is None:
        raise ValueError(f"Exact group-disjoint holdout allocation failed: {result.message}")
    if (result.x.shape != (len(groups),) or not np.isfinite(result.x).all()
            or not np.isclose(result.x, np.rint(result.x), atol=1e-6, rtol=0).all()
            or (result.x < -1e-6).any() or (result.x > 1 + 1e-6).any()):
        raise ValueError('Holdout allocation did not return binary group decisions.')
    selected = np.rint(result.x).astype(bool)
    test_mask = selected[group_codes]
    if (int(sizes @ selected.astype(int)) != n_test or test_mask.sum() != n_test
            or set(identity.active_structure_group[test_mask]) &
            set(identity.active_structure_group[~test_mask])):
        raise ValueError('Holdout allocation violates the exact size or group isolation constraint.')
    return np.flatnonzero(~test_mask), np.flatnonzero(test_mask)


def cross_entity_novelty(metadata, training_ids, validation_ids, scenario):
    """Target-free exact-identity novelty relative to actual outer training.

    Both-unseen is a descriptive stratum of a chemical validation setting, not
    another experiment. Publication transfer retains both chemical indicators.
    """
    return novelty_from_canonical_metadata(_canonical_cv_metadata(metadata, scenario),
                                          training_ids, validation_ids, scenario)


def novelty_from_canonical_metadata(metadata, training_ids, validation_ids, scenario):
    """Classify novelty from already verified canonical metadata, without responses."""
    frame = metadata.set_index('row_id')
    if not frame.index.is_unique or frame.index.isna().any():
        raise ValueError('Canonical novelty identities must be unique and complete.')
    train, valid = list(training_ids), list(validation_ids)
    if (len(set(train)) != len(train) or len(set(valid)) != len(valid) or not train or not valid
            or set(train) & set(valid) or not set(train + valid) <= set(frame.index)):
        raise ValueError('Novelty requires disjoint, complete canonical training and validation identities.')
    out = frame.loc[valid, ['publication_group', 'active_structure_group', 'contaminant_group',
                            'SurfactantType', 'ContaminantType']].copy()
    for name, column in (('surfactant', 'active_structure_group'), ('contaminant', 'contaminant_group')):
        out[name + '_unseen'] = ~out[column].isin(frame.loc[train, column])
    required = {'unseen_surfactant': 'surfactant_unseen', 'unseen_contaminant': 'contaminant_unseen'}
    if scenario in required and not out[required[scenario]].all():
        raise ValueError('Novelty metadata violates the required chemical group exclusion.')
    out['both_unseen'] = out.surfactant_unseen & out.contaminant_unseen
    out['novelty_stratum'] = np.where(out.both_unseen, 'both_unseen',
        np.where(out.surfactant_unseen, 'surfactant_unseen_only',
                 np.where(out.contaminant_unseen, 'contaminant_unseen_only', 'both_represented')))
    return out.rename_axis('observation_id').reset_index()


def grouped_feasibility(development, scenario, folds):
    """Report target-blind group support, sizes and actual outer/inner class absences."""
    frame = _canonical_cv_metadata(development, scenario)
    axes = list(dict.fromkeys(v['group_column'] for v in VALIDATION_SETTINGS.values()))
    support = {}
    relationships = []
    for axis in axes:
        sizes = frame.groupby(axis).size().sort_values(ascending=False)
        support[axis] = {'n_groups': len(sizes), 'size_distribution': sizes.tolist(),
                         'dominant_groups': sizes.head(5).to_dict(), 'classes': {}}
        for column in ('SurfactantType', 'ContaminantType'):
            support[axis]['classes'][column] = {
                str(label): {'observations': len(subset), 'groups': subset[axis].nunique(),
                             'validation_absence_unavoidable': subset[axis].nunique() < 5,
                             'training_absence_unavoidable': subset[axis].nunique() == 1}
                for label, subset in frame.groupby(column)}
        support[axis]['chemical_cells'] = {
            ' | '.join(labels): {'observations': len(part), 'groups': part[axis].nunique()}
            for labels, part in frame.groupby(['SurfactantType', 'ContaminantType'])}
        for key, group in frame.groupby(axis):
            relationships.append({'axis': axis, 'group': key, 'n_observations': len(group),
                                  **{other: group[other].nunique() for other in axes}})
    rows = []
    for fold in folds:
        partitions = [('outer', fold.train_ids, fold.validation_ids, fold.train_groups, fold.validation_groups)]
        partitions += [(f'inner_{f.fold}', f.train_ids, f.validation_ids, f.train_groups, f.validation_groups)
                       for f in fold.inner_folds]
        for name, train_ids, val_ids, train_groups, val_groups in partitions:
            parent = frame if name == 'outer' else frame[frame.row_id.isin(fold.train_ids)]
            axis = VALIDATION_SETTINGS[scenario]['group_column']
            n_folds = GROUPED_OUTER_FOLDS if name == 'outer' else GROUPED_INNER_FOLDS
            if set(train_ids) & set(val_ids) or set(train_groups) & set(val_groups):
                raise ValueError('Feasibility preflight detected group or observation leakage.')
            for role, ids, groups in (('training', train_ids, train_groups), ('validation', val_ids, val_groups)):
                part = frame[frame.row_id.isin(ids)]
                rows.append({'repetition': fold.repetition, 'outer_fold': fold.fold,
                             'partition': name, 'role': role, 'n_observations': len(ids),
                             'n_groups': len(groups), 'class_counts': {
                                 c: part[c].value_counts().to_dict() for c in ('SurfactantType', 'ContaminantType')},
                             'absent_classes': {c: sorted(set(parent[c]) - set(part[c]))
                                                for c in ('SurfactantType', 'ContaminantType')},
                             'unavoidable_absent_classes': {
                                 c: sorted(label for label in set(parent[c]) - set(part[c])
                                     if parent.loc[parent[c].eq(label), axis].nunique()
                                     < (2 if role == 'training' else n_folds))
                                 for c in ('SurfactantType', 'ContaminantType')}})
    return {'status': 'verified', 'scenario': scenario, 'development_rows': len(frame),
            'allocation_converged': True, 'group_support': support,
            'group_relationships': relationships, 'partitions': rows,
            'limitations': 'Group/class counts describe support, not predictive performance or independent experiments.'}


def membership_hashes(frame, folds):
    outer, inner = grouped_cv_membership(frame, folds)
    return {name: hashlib.sha256(table.to_csv(index=False).encode('utf-8')).hexdigest()
              for name, table in [('outer_validation_membership.csv', outer),
                                  ('inner_validation_membership.csv', inner)]}


def validation_memberships(frame, scenario, *, plan=None):
    """Bind a shared execution plan, or generate a standalone scenario through MILP."""
    folds = (plan.folds(frame, scenario) if plan is not None
             else make_repeated_nested_grouped_cv(frame, scenario))
    return folds, membership_hashes(frame, folds)
