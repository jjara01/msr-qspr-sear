"""Structure-only representations, Mordred generation and descriptor provenance."""

from pathlib import Path
from typing import Any, Optional, Tuple, cast
import sys
import hashlib
import os
import time
import warnings
import numpy as np
import pandas as pd
from rdkit import Chem
warnings.filterwarnings('ignore', category=SyntaxWarning, message='"is" with a literal.*', module=r'mordred\..*')
warnings.filterwarnings('ignore', category=RuntimeWarning, message='overflow encountered in reduce', module='numpy.core.fromnumeric')
warnings.filterwarnings('ignore', category=RuntimeWarning, message='divide by zero encountered in double_scalars', module=r'mordred\..*')
from mordred import Calculator, descriptors
from msr_qspr.config import PREDICTOR_CONTRACT, STRUCTURE_COLUMNS

MORDRED_CALC = Calculator(descriptors, ignore_3D=True)
PAIR_NAMES = ('Pair_LogMWRatio', 'Pair_DeltaSLogP', 'Pair_DeltaPolarDensity',
              'Pair_SLogPProduct', 'Pair_ContLogP_SurfPolar', 'Pair_SurfLogP_ContPolar')

def canonicalize(smiles: str, strip_counterions: bool) -> str:
    mol = Chem.MolFromSmiles(str(smiles))
    if mol is None:
        raise ValueError(f"Invalid SMILES: {smiles}")
    if strip_counterions:
        fragments = Chem.GetMolFrags(mol, asMols=True)
        organic = [frag for frag in fragments if any(atom.GetAtomicNum() == 6 for atom in frag.GetAtoms())]
        chosen = organic if organic else list(fragments)
        mol = max(chosen, key=lambda frag: frag.GetNumHeavyAtoms())
    return Chem.MolToSmiles(mol, canonical=True)

def descriptor_names() -> Tuple[str, ...]:
    """Exact installed 2D Mordred names, rather than a permissive prefix rule."""
    return tuple(prefix + str(d) for prefix in ("Cont_", "Surf_")
                 for d in MORDRED_CALC.descriptors) + PAIR_NAMES


def pair_descriptors(df):
    """Derive pair properties from the same represented molecular descriptors."""
    values = df.loc[:, ['Cont_MW', 'Surf_MW', 'Cont_SLogP', 'Surf_SLogP',
                        'Cont_TopoPSA', 'Surf_TopoPSA']].astype(float)
    if np.isinf(values.to_numpy()).any():
        raise ValueError('Pair source properties must be finite or missing.')
    cm, sm, cl, sl, cp, sp = (values[c] for c in values)
    if (cm.le(0) | sm.le(0)).any():
        raise ValueError('Pair molecular weights must be strictly positive or missing.')
    with np.errstate(over='ignore', divide='ignore', invalid='ignore'):
        result = pd.DataFrame(dict(zip(PAIR_NAMES, [np.log10(cm / sm), cl - sl,
            cp / cm - sp / sm, cl * sl, cl * (sp / sm), sl * (cp / cm)])), index=df.index)
    if np.isinf(result.to_numpy()).any():
        raise ValueError('Pair descriptor calculation produced a nonfinite value.')
    return result


def _descriptor_fingerprint(df, columns):
    table = df.loc[:, list(STRUCTURE_COLUMNS) + sorted(columns)]
    digest = hashlib.sha256(pd.util.hash_pandas_object(table, index=False).values.tobytes())
    digest.update("\n".join(table.columns).encode())
    return digest.hexdigest()


def descriptor_generator_signature():
    generator_path = __file__
    return hashlib.sha256(Path(generator_path).read_bytes()).hexdigest()


def record_descriptor_provenance(df, columns, counterion_policy=None):
    """Called only by the shared structure-only descriptor generator."""
    if not set(columns) <= set(descriptor_names()):
        raise ValueError("Descriptor generator returned unregistered Mordred names.")
    df.attrs["descriptor_provenance"] = {
        "contract": PREDICTOR_CONTRACT,
        "columns": sorted(columns),
        "counterion_policy": counterion_policy or 'counterion_strip',
        "generator_sha256": descriptor_generator_signature(),
        "fingerprint": _descriptor_fingerprint(df, columns),
    }


def canonical_main_organic_fragment(smiles: str) -> Optional[str]:
    """Return the canonical largest carbon-containing fragment, or None if invalid."""
    if pd.isna(smiles):
        return None
    try:
        return canonicalize(smiles, True)
    except (ValueError, TypeError):
        return None


def canonical_surfactant_for_policy(smiles: str, policy=None) -> Optional[str]:
    """Canonicalize surfactant SMILES according to the selected counterion policy."""
    policy = policy or 'counterion_strip'
    if policy != "counterion_strip":
        raise ValueError("Unknown descriptor counterion policy.")
    return canonical_main_organic_fragment(smiles)


def compute_mordred_descriptors(molecules: pd.Series, prefix: str) -> pd.DataFrame:
    """
    Compute Mordred descriptors for a Series of RDKit Mol objects.

    Returns a numeric DataFrame with the given prefix applied to column names.
    """
    nproc_env = os.getenv("MORDRED_NPROC")
    nproc = 1
    if nproc_env is not None:
        try:
            parsed = int(nproc_env)
            if parsed > 0:
                nproc = parsed
        except ValueError:
            pass

    dfx = MORDRED_CALC.pandas(molecules, nproc=nproc, quiet=True)
    dfx = dfx.add_prefix(prefix)
    dfx = dfx.apply(pd.to_numeric, errors="coerce")
    dfx = dfx.replace([np.inf, -np.inf], np.nan)

    return dfx


def compute_mordred_descriptors_from_smiles(
    smiles: pd.Series,
    prefix: str,
    label: str,
) -> pd.DataFrame:
    """
    Compute descriptors once per unique canonical SMILES and map back to rows.

    This avoids recomputing identical molecules many times.
    """
    chem = cast(Any, Chem)

    unique_smiles = pd.Index(pd.unique(smiles.dropna()))
    print(f"[DataPrep] Mordred ({label}): {len(unique_smiles)} unique molecules across {len(smiles)} rows.",
          file=sys.stderr)

    unique_mols = pd.Series(
        [chem.MolFromSmiles(str(s)) for s in unique_smiles],
        index=unique_smiles,
        dtype=object,
    )

    valid_mask = unique_mols.notna()
    n_invalid_unique = int((~valid_mask).sum())
    if n_invalid_unique:
        print(f"[DataPrep] {label}: {n_invalid_unique} unique canonical SMILES failed RDKit parsing.", file=sys.stderr)

    t0 = time.perf_counter()
    desc_valid = compute_mordred_descriptors(unique_mols.loc[valid_mask].reset_index(drop=True), prefix)
    elapsed = time.perf_counter() - t0
    print(f"[DataPrep] Mordred ({label}) finished in {elapsed:.1f}s", file=sys.stderr)

    desc_valid.index = unique_smiles[valid_mask]
    desc_unique = desc_valid.reindex(unique_smiles)
    desc_all = desc_unique.reindex(smiles.to_numpy()).reset_index(drop=True)
    return desc_all


def prepare_descriptor_inputs(df_base: pd.DataFrame, *, counterion_policy=None) -> pd.DataFrame:
    """Canonicalize structures and grouping IDs without computing descriptors."""
    df = df_base.copy()

    # Canonical SMILES
    df["SMILES_contaminant"] = df["SMILES_contaminant"].apply(
        canonical_main_organic_fragment
    )
    df["SMILES_surfactant"] = df["SMILES_surfactant"].apply(
        lambda s: canonical_surfactant_for_policy(s, counterion_policy))

    # Invalid structures must not silently change a validated development partition.
    n_invalid_cont = df["SMILES_contaminant"].isna().sum()
    n_invalid_surf = df["SMILES_surfactant"].isna().sum()
    if n_invalid_cont or n_invalid_surf:
        raise ValueError(
            f"Invalid SMILES: "
            f"{n_invalid_cont} contaminants, {n_invalid_surf} surfactants."
        )

    df = df.dropna(subset=["SMILES_contaminant", "SMILES_surfactant"]).reset_index(
        drop=True
    )

    df["contaminant_id"] = df["SMILES_contaminant"]
    df["surfactant_id"] = df["SMILES_surfactant"]
    df["pair_id"] = df["surfactant_id"].astype(str) + "__" + df["contaminant_id"].astype(str)
    return df


def add_descriptors(df_base: pd.DataFrame, *, counterion_policy=None) -> pd.DataFrame:
    """Add Mordred descriptors using the same structures as validation groups."""
    if not df_base.columns.is_unique or set(df_base.columns) & set(descriptor_names()):
        raise ValueError("Descriptor inputs require unique columns without precomputed descriptors.")
    if not set(["SMILES_contaminant", "SMILES_surfactant"]) <= set(df_base):
        raise ValueError("Both molecular structural inputs are required.")
    df = prepare_descriptor_inputs(df_base, counterion_policy=counterion_policy)

    # Compute descriptors from unique canonical SMILES and map back to rows.
    desc_contaminant = compute_mordred_descriptors_from_smiles(
        df["SMILES_contaminant"],
        "Cont_",
        "contaminant",
    )
    desc_surfactant = compute_mordred_descriptors_from_smiles(
        df["SMILES_surfactant"],
        "Surf_",
        "surfactant",
    )

    # Merge base data with descriptors
    df_full = pd.concat(
        [
            df.reset_index(drop=True),
            desc_contaminant.reset_index(drop=True),
            desc_surfactant.reset_index(drop=True),
        ],
        axis=1,
    )

    pairs = pair_descriptors(df_full)
    df_full = pd.concat([df_full, pairs], axis=1)
    record_descriptor_provenance(df_full, list(desc_contaminant) + list(desc_surfactant) + list(pairs), counterion_policy)
    return df_full
