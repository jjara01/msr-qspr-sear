"""Frozen Excel inputs, response-isolated loading and descriptor caching."""

from pathlib import Path
import hashlib
import pickle
import re
import warnings
from xml.etree import ElementTree as ET
from zipfile import ZipFile
import numpy as np
import pandas as pd
from msr_qspr import artifacts, config
from msr_qspr.config import DATA_DIR
from msr_qspr.chemistry import add_descriptors, descriptor_names, prepare_descriptor_inputs

WORKBOOK_PATH = DATA_DIR / 'msr_dataset.xlsx'
WORKBOOK_COLUMNS = {
    'Row ID': 'row_id', 'Contaminant': 'Contaminant',
    'Contaminant SMILES': 'SMILES_contaminant', 'Contaminant Type': 'ContaminantType',
    'Water solubility (g/L)': 'WaterSolubility', 'Surfactant': 'Surfactant',
    'Surfactant SMILES': 'SMILES_surfactant', 'Surfactant Type': 'SurfactantType',
    'CMC (mol/L)': 'CMC', 'Temperature (°C)': 'temperature_C',
    'Temperature basis': 'temperature_basis', 'MSR': 'MSR',
    'Source location': 'source_location', 'Ref.': 'reference_number',
}
# Preserve the frozen publication keys for the two bibliography entries without DOI.
NO_DOI_SOURCE_REFS = {21: 'leeOptimizingSurfactantEnhancedSolubilzation1999',
                     63: 'yuweiqianEffectCationicSurfactant1998'}
_XML = '{http://schemas.openxmlformats.org/spreadsheetml/2006/main}'


def verify_scientific_inputs(root=None):
    """Reject absent or modified authoritative input before decoding scientific cells."""
    from msr_qspr.config import SCIENTIFIC_INPUTS
    root = WORKBOOK_PATH.parent.parent if root is None else Path(root)
    for name, expected in SCIENTIFIC_INPUTS.items():
        path = root / name
        if not path.is_file():
            raise FileNotFoundError(f'Missing frozen scientific workbook: {path}')
        if hashlib.sha256(path.read_bytes()).hexdigest() != expected:
            raise ValueError(f'Scientific freeze integrity failure (checksum mismatch): {name}')
    return dict(SCIENTIFIC_INPUTS)


def _cell_value(cell, shared):
    """Decode only a requested cell; reserved response cells never reach this function."""
    if cell is None:
        return None
    if cell.find(_XML + 'f') is not None:
        raise ValueError('Scientific dataset must contain saved values, not formulas.')
    kind = cell.get('t')
    if kind == 'inlineStr':
        return ''.join(node.text or '' for node in cell.iter(_XML + 't'))
    value = cell.findtext(_XML + 'v')
    if value is None:
        return None
    if kind == 's':
        return shared[int(value)]
    if kind == 'str':
        return value
    if kind in ('e', 'b'):
        raise ValueError('Unexpected scientific cell type.')
    return float(value)


def _read_workbook(columns, observation_ids=None):
    """Project cells before decoding: metadata first, then only selected responses.

    Excel row readers decode every cell even with usecols/skiprows. Reading the
    frozen worksheet XML lets us exclude reserved MSR values before conversion.
    """
    verify_scientific_inputs()
    if not set(columns) <= set(WORKBOOK_COLUMNS.values()):
        raise ValueError('Unknown scientific dataset columns.')
    selected = None if observation_ids is None else set(observation_ids)
    rows, references = [], {}
    with ZipFile(WORKBOOK_PATH) as book:
        sheets = ET.fromstring(book.read('xl/workbook.xml')).find(_XML + 'sheets')
        if [sheet.get('name') for sheet in sheets] != ['full_dataset']:
            raise ValueError('Expected only the full_dataset sheet.')
        shared = []
        if 'xl/sharedStrings.xml' in book.namelist():
            shared = [''.join(t.text or '' for t in item.iter(_XML + 't'))
                      for item in ET.fromstring(book.read('xl/sharedStrings.xml'))]
        header = None
        with book.open('xl/worksheets/sheet1.xml') as sheet:
            for _, row in ET.iterparse(sheet, events=('end',)):
                if row.tag != _XML + 'row':
                    continue
                cells = {re.sub(r'\d+', '', c.get('r')): c for c in row}
                first = _cell_value(cells.get('A'), shared)
                if first == 'Row ID':
                    if header is not None:
                        raise ValueError('Duplicate dataset header.')
                    titles = [_cell_value(c, shared) for c in row]
                    if titles != list(WORKBOOK_COLUMNS):
                        raise ValueError('Expected the 14-column scientific dataset schema.')
                    header = dict(zip(cells, WORKBOOK_COLUMNS.values()))
                elif isinstance(first, str) and re.fullmatch(r'msr_intake_\d{4}', first):
                    if header is None:
                        raise ValueError('Observation precedes dataset header.')
                    if selected is None or first in selected:
                        rows.append({name: _cell_value(cells.get(letter), shared)
                                     for letter, name in header.items() if name in columns})
                elif isinstance(first, str) and re.match(r'^\[\d+\] ', first):
                    match = re.fullmatch(r'\[(\d+)\] (.+)', first)
                    number = int(match[1])
                    if number in references:
                        raise ValueError('Duplicate bibliography reference.')
                    references[number] = match[2]
                row.clear()
    frame = pd.DataFrame(rows, columns=columns)
    if header is None or frame.empty:
        raise ValueError('Missing scientific observations or header.')
    return frame, references


def corpus_metadata():
    """Read all 459 identities, predictors and conditions without decoding any MSR."""
    frame, references = _read_workbook([c for c in WORKBOOK_COLUMNS.values() if c != 'MSR'])
    if (len(frame) != 459 or not frame.row_id.is_unique
            or not frame.row_id.str.fullmatch(r'msr_intake_\d{4}').all()):
        raise ValueError('Expected 459 unique stable Row IDs.')
    # Intake numbers are compatibility metadata derived from Row ID, never Excel positions.
    frame['intake_row'] = frame.row_id.str.removeprefix('msr_intake_').astype(int)
    required = ['Contaminant', 'SMILES_contaminant', 'ContaminantType',
                'Surfactant', 'SMILES_surfactant', 'SurfactantType']
    if frame[required].isna().any().any() or frame[required].eq('').any().any():
        raise ValueError('Incomplete chemical identities or classes.')
    for column in ('CMC', 'WaterSolubility'):
        frame[column] = pd.to_numeric(frame[column], errors='raise')
        if not np.isfinite(frame[column]).all() or not frame[column].gt(0).all():
            raise ValueError(f'{column} must be finite and positive in reference units.')
    numbers = frame.reference_number
    if (numbers.isna().any() or not numbers.mod(1).eq(0).all()
            or not numbers.isin(references).all()):
        raise ValueError('Invalid or unresolved bibliography reference.')
    frame['reference_number'] = numbers.astype(int)
    frame['source_ref'] = frame.reference_number.map(references)
    frame['source_doi'] = frame.source_ref.str.extract(r'https://doi\.org/(\S+)', expand=False).fillna('')
    for number in frame.loc[frame.source_doi.eq(''), 'reference_number'].unique():
        if number not in NO_DOI_SOURCE_REFS:
            raise ValueError(f'Unrecognized DOI-free publication: {number}')
        frame.loc[frame.reference_number.eq(number), 'source_ref'] = NO_DOI_SOURCE_REFS[number]
    frame['temperature_mode'] = frame.temperature_basis.replace({r'^range_C \(.+\)$': 'range_C'}, regex=True)
    frame['temperature_C'] = pd.to_numeric(frame.temperature_C, errors='raise')
    frame['temperature_value_1'] = frame.temperature_C.where(frame.temperature_mode.eq('reported_C'))
    kelvin = frame.temperature_mode.eq('reported_K')
    frame.loc[kelvin, 'temperature_value_1'] = frame.loc[kelvin, 'temperature_C'] + 273.15
    bounds = frame.temperature_basis.str.extract(r'^range_C \((-?\d+(?:\.\d+)?)–(-?\d+(?:\.\d+)?)\)$').astype(float)
    ranges = frame.temperature_mode.eq('range_C')
    frame.loc[ranges, 'temperature_value_1'] = bounds.loc[ranges, 0]
    frame['temperature_value_2'] = bounds[1]
    from msr_qspr.preprocessing import _validate_assay_temperature
    _validate_assay_temperature(frame)
    return frame.sort_values('row_id').reset_index(drop=True)


def _load_observations(metadata, positions):
    """Attach responses only for the already verified observation membership."""
    selected = metadata.iloc[positions].copy()
    responses, _ = _read_workbook(['row_id', 'MSR'], selected.row_id)
    if not responses.row_id.is_unique or set(responses.row_id) != set(selected.row_id):
        raise ValueError('Response loading differs from verified membership.')
    selected['MSR'] = selected.row_id.map(responses.set_index('row_id').MSR)
    if not np.isfinite(selected.MSR).all() or not selected.MSR.gt(0).all():
        raise ValueError('MSR must be finite and positive for log10 transformation.')
    return selected.reset_index(drop=True)


def prepare_base_dataframe() -> pd.DataFrame:
    """Read development observations without decoding reserved MSR cells."""
    from msr_qspr.validation import workflow_partitions
    metadata = corpus_metadata()
    development, _ = workflow_partitions(metadata).holdout_positions(metadata)
    return _load_observations(metadata, development)


def prepare_full_dataframe(
    use_cache: bool = True,
    force_recompute: bool = False,
) -> pd.DataFrame:
    """
    Return the full modeling DataFrame with descriptors included.

    Reuse only a complete cache matching development data and its environment.
    Otherwise recompute descriptors, save, and return.
    """
    from msr_qspr.preprocessing import build_predictor_matrix, descriptor_generator_signature
    df_base = prepare_base_dataframe()
    # A changed holdout must never load cached responses from its old complement.
    population = artifacts.membership_fingerprint(df_base.row_id)
    identity = artifacts.fingerprint(config.scientific_identity())
    shared_cache_path = artifacts.artifact_directory('descriptor_cache') / f'development_counterion_strip__{population}__{identity}.pkl'
    with artifacts.exclusive_artifact(shared_cache_path):
        if use_cache and not force_recompute and shared_cache_path.exists():
            represented = prepare_descriptor_inputs(df_base)
            try:
                df_full = pd.read_pickle(shared_cache_path)
                if isinstance(df_full, pd.DataFrame):
                    df_full = df_full.reset_index(drop=True)
                    provenance = df_full.attrs.get("descriptor_provenance", {})
                    if (isinstance(provenance, dict)
                            and set(represented) | set(descriptor_names()) <= set(df_full)
                            and df_full[list(represented)].equals(represented)
                            and provenance.get("generator_sha256") == descriptor_generator_signature()
                            and provenance.get("counterion_policy") == "counterion_strip"
                            and provenance.get("environment") == artifacts.environment_signature()):
                        build_predictor_matrix(df_full)
                        return df_full
            except (OSError, EOFError, pickle.UnpicklingError, ImportError,
                    ValueError, TypeError, AttributeError) as error:
                warnings.warn(f'Rebuilding invalid descriptor cache {shared_cache_path}: {error}', RuntimeWarning)

        df_full = add_descriptors(df_base)
        df_full = df_full.reset_index(drop=True)
        build_predictor_matrix(df_full)
        if not set(descriptor_names()) <= set(df_full):
            raise ValueError('Descriptor generation returned an incomplete schema.')
        df_full.attrs["descriptor_provenance"]["environment"] = artifacts.environment_signature()

        with artifacts.atomic_artifact(shared_cache_path) as temporary:
            df_full.to_pickle(temporary)

    return df_full


def load_reserved_predictors(experiment):
    """Reserved responses are loaded only by the separately confirmed final scorer."""
    from msr_qspr.validation import workflow_partitions, ACTIVE_HOLDOUT, _canonical_cv_metadata
    from msr_qspr.preprocessing import build_design_matrices, project_experiment_predictors
    metadata = corpus_metadata()
    _, held = workflow_partitions(metadata).holdout_positions(metadata)
    from msr_qspr.abraham import attach_descriptors, STATUS_COLUMN
    frame = _load_observations(metadata, held)
    described = add_descriptors(frame, counterion_policy=experiment.counterion_policy)
    if experiment.predictor_family == 'abraham_physicochemical':
        described = attach_descriptors(described)
    X, y, _ = build_design_matrices(described, include_temperature=False)
    identities = _canonical_cv_metadata(frame, ACTIVE_HOLDOUT)
    if STATUS_COLUMN in described:
        identities[STATUS_COLUMN] = described[STATUS_COLUMN].to_numpy()
    return project_experiment_predictors(X, experiment), np.log10(y.to_numpy()), identities


def verify():
    """Verify frozen workbook and generate/share target-free in-memory MILP plans."""
    from msr_qspr import config, validation
    hashes = verify_scientific_inputs()
    corpus = corpus_metadata()
    plan = validation.workflow_partitions(corpus)
    development, held = plan.holdout_positions(corpus)
    frame = corpus.iloc[development].reset_index(drop=True)
    plans = {setting: validation.validation_memberships(frame, setting, plan=plan)[1]
             for setting in config.VALIDATION_SETTINGS}
    if hashes != verify_scientific_inputs():
        raise ValueError('Scientific inputs changed during verification.')
    return {'observations': len(corpus), 'development': len(development), 'holdout': len(held),
            'frozen_sha256': hashes, 'validation_sha256': plans,
            'partition_sha256': plan.sha256, 'partition_configuration': plan.configuration}
