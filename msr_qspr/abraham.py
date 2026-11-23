"""Curated Abraham solute descriptors for contaminants, with identity and provenance checks.

The curated descriptor table is a locally supplied scientific input. It is not
redistributed with the repository; its files are verified by checksum before
use. Curated values always take precedence. Contaminants without a curated
entry, or whose stereochemistry cannot be resolved, receive missing values that
the fitted pipeline imputes from training data, together with an explicit status.
"""

from dataclasses import dataclass
import hashlib
import os
from pathlib import Path
from typing import Callable, Optional

import numpy as np
import pandas as pd

from msr_qspr.config import ABRAHAM_DESCRIPTORS

DIRECTORY_VARIABLE = 'MSR_ABRAHAM_DIRECTORY'
DEFAULT_DIRECTORY = Path(__file__).resolve().parents[1] / 'data' / 'abraham'
DESCRIPTOR_FILE = 'descriptors.csv'
PROVENANCE_FILE = 'provenance.csv'
FILES_SHA256 = {
    DESCRIPTOR_FILE: '3135a3fdaa1a125f3815c64f9c5c3145ae2a0b0ee34d6a1603fdc88727b37864',
    PROVENANCE_FILE: '4a4dd65313ef3b9c9da5b9bb3a18c84d6b7e07fc0e6b705765a15045c4365ed8',
}
SOURCE_STATUS = {
    'COMPTOX_2017_BIBLIO': 'curated_literature',
    'UFZ_QSPR_FALLBACK': 'curated_qspr_estimate',
    'TULP_2008_E_ACD_ESTIMATE': 'curated_estimate',
}
CURATED_STATUSES = frozenset(SOURCE_STATUS.values())
ESTIMATED_STATUS = 'provider_estimate'
STATUS_COLUMN = 'abraham_descriptor_status'


def descriptor_directory(directory=None) -> Path:
    """Resolve the local input directory: argument, environment variable, then data/abraham."""
    if directory is not None:
        return Path(directory)
    configured = os.getenv(DIRECTORY_VARIABLE, '').strip()
    return Path(configured) if configured else DEFAULT_DIRECTORY


def verify_files(directory=None):
    """Return verified checksums, or fail before any value is decoded."""
    root = descriptor_directory(directory)
    for name, expected in FILES_SHA256.items():
        path = root / name
        if not path.is_file():
            raise FileNotFoundError(
                f'Missing curated Abraham descriptor input {path}. Supply the authorized '
                f'{DESCRIPTOR_FILE} and {PROVENANCE_FILE} in {root} or set {DIRECTORY_VARIABLE}.')
        if hashlib.sha256(path.read_bytes()).hexdigest() != expected:
            raise ValueError(f'Curated Abraham descriptor input checksum mismatch: {path}')
    return dict(FILES_SHA256)


def _inchikey(mol):
    from rdkit.Chem import inchi
    return inchi.MolToInchiKey(mol)


def _unspecified_stereo_elements(mol):
    from rdkit import Chem
    return sum(element.specified == Chem.StereoSpecified.Unspecified
               for element in Chem.FindPotentialStereo(mol))


@dataclass(frozen=True)
class AbrahamPanel:
    """Verified curated descriptors indexed by the InChIKey of their recorded structure."""
    table: pd.DataFrame

    def resolve(self, smiles, *, name=None, cas=None):
        """Return (descriptor values or None, status, curated InChIKey or None).

        A recorded structure with more than one unspecified stereo element denotes a
        specific stereoisomer that its graph cannot identify. Such an entry resolves
        only through its curated name or CAS number, never from the structure alone.
        A different stereospecific structure cannot be verified against that graph.
        """
        from rdkit import Chem
        from msr_qspr.chemistry import canonical_main_organic_fragment
        canonical = canonical_main_organic_fragment(smiles)
        mol = Chem.MolFromSmiles(canonical) if canonical else None
        if mol is None:
            return None, 'invalid_structure', None
        key = _inchikey(mol)
        confirmed = lambda entry: ((cas is not None and str(cas).strip() == entry.cas)
                                   or (name is not None and str(name).strip().casefold() == entry.curated_name.casefold()))
        if key in self.table.index:
            entry = self.table.loc[key]
            if entry.stereo_ambiguous and not confirmed(entry):
                return None, 'stereochemistry_unresolved', None
            return entry[list(ABRAHAM_DESCRIPTORS)].to_numpy(dtype=float), entry.status, key
        skeleton = self.table.loc[self.table.index.str[:14] == key[:14]]
        if len(skeleton) == 1 and cas is not None and str(cas).strip() == skeleton.iloc[0].cas:
            entry = skeleton.iloc[0]
            if entry.stereo_ambiguous:
                return None, 'stereochemistry_unresolved', None
            return entry[list(ABRAHAM_DESCRIPTORS)].to_numpy(dtype=float), entry.status, entry.name
        if len(skeleton):
            return None, 'stereochemistry_unresolved', None
        return None, 'not_curated', None


def load_panel(directory=None) -> AbrahamPanel:
    """Load and validate the curated descriptor and provenance tables."""
    from rdkit import Chem
    verify_files(directory)
    root = descriptor_directory(directory)
    descriptors = pd.read_csv(root / DESCRIPTOR_FILE)
    provenance = pd.read_csv(root / PROVENANCE_FILE, dtype={'ufz_CAS': str})
    expected = ['contaminant_group', 'canonical_smiles', 'inchikey', *ABRAHAM_DESCRIPTORS]
    if list(descriptors.columns) != expected:
        raise ValueError('Curated Abraham descriptor table has an unexpected schema.')
    if not {'contaminant_group', 'inchikey', 'source_type', 'ufz_CAS'} <= set(provenance):
        raise ValueError('Curated Abraham provenance table has an unexpected schema.')
    for frame in (descriptors, provenance):
        if (frame[['contaminant_group', 'inchikey']].isna().any().any()
                or frame.contaminant_group.duplicated().any() or frame.inchikey.duplicated().any()):
            raise ValueError('Curated Abraham entries require unique identities.')
    table = descriptors.merge(provenance[['contaminant_group', 'inchikey', 'source_type', 'ufz_CAS']],
                              on=['contaminant_group', 'inchikey'], validate='one_to_one')
    values = table[list(ABRAHAM_DESCRIPTORS)].to_numpy(dtype=float)
    if (len(table) != len(descriptors) or len(table) != len(provenance)
            or not np.isfinite(values).all() or not table.source_type.isin(SOURCE_STATUS).all()):
        raise ValueError('Curated Abraham descriptors require finite values and recognized provenance.')
    ambiguous = []
    for smiles, key in zip(table.canonical_smiles, table.inchikey):
        mol = Chem.MolFromSmiles(smiles)
        if mol is None or _inchikey(mol) != key:
            raise ValueError(f'Curated Abraham structure does not match its InChIKey: {key}')
        ambiguous.append(_unspecified_stereo_elements(mol) > 1)
    table = table.assign(curated_name=table.contaminant_group.str.strip(),
                         cas=table.ufz_CAS.fillna('').str.strip(),
                         status=table.source_type.map(SOURCE_STATUS),
                         stereo_ambiguous=ambiguous).set_index('inchikey')
    if (table.stereo_ambiguous & table.cas.eq('')).any():
        raise ValueError('Stereochemically ambiguous curated entries require a CAS identity.')
    return AbrahamPanel(table)


# A provider estimates descriptors for one contaminant SMILES when no curated
# entry exists. It returns None or a mapping with the five descriptor values and
# a 'source' string; estimates are labelled separately from curated values.
DescriptorProvider = Callable[[str], Optional[dict]]


def attach_descriptors(frame, *, panel=None, provider: Optional[DescriptorProvider] = None,
                       require_curated=False):
    """Return a copy of frame with Abraham descriptors and their resolution status.

    Development data must resolve entirely from curated entries. Other queries keep
    missing values for unresolved identities so the fitted pipeline imputes them.
    """
    if not {'SMILES_contaminant'} <= set(frame) or set(ABRAHAM_DESCRIPTORS) & set(frame):
        raise ValueError('Abraham attachment requires contaminant structures and no existing descriptors.')
    panel = load_panel() if panel is None else panel
    names = frame['Contaminant'] if 'Contaminant' in frame else pd.Series(None, index=frame.index)
    numbers = frame['contaminant_cas'] if 'contaminant_cas' in frame else pd.Series(None, index=frame.index)
    rows, statuses = [], []
    for smiles, name, cas in zip(frame.SMILES_contaminant, names, numbers):
        values, status, _ = panel.resolve(smiles, name=None if pd.isna(name) else name,
                                          cas=None if pd.isna(cas) else cas)
        if values is None and provider is not None and status == 'not_curated':
            estimate = provider(smiles)
            if estimate is not None:
                values = np.array([float(estimate[d]) for d in ABRAHAM_DESCRIPTORS])
                if not np.isfinite(values).all():
                    raise ValueError('Descriptor provider returned non-finite values.')
                status = ESTIMATED_STATUS
        rows.append(np.full(len(ABRAHAM_DESCRIPTORS), np.nan) if values is None else values)
        statuses.append(status)
    if require_curated and any(s not in CURATED_STATUSES for s in statuses):
        unresolved = sorted({s for s in statuses if s not in CURATED_STATUSES})
        raise ValueError(f'Development contaminants must resolve to curated Abraham descriptors: {unresolved}')
    result = frame.copy()
    result[list(ABRAHAM_DESCRIPTORS)] = np.vstack(rows) if rows else np.empty((0, len(ABRAHAM_DESCRIPTORS)))
    result[STATUS_COLUMN] = statuses
    return result
