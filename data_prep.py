from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
from rdkit import Chem
from mordred import Calculator, descriptors
import warnings

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.ticker import MaxNLocator

from config import DATA_DIR, DATA_ARTIFACTS_DIR, DATA_FIGURES_DIR, PROCESSED_FULL_DF_PATH
from plot_style import set_plot_style, get_base_color, get_discrete_colors

# Silence specific noisy numerical warnings from NumPy used inside Mordred
warnings.filterwarnings(
    "ignore",
    category=RuntimeWarning,
    message="overflow encountered in reduce",
    module="numpy.core.fromnumeric",
)

# Based on excel file
RAW_XLSX_NAME = "data.xlsx"
SHEET_NAME = "full_dataset"
USECOLS = "A:I"
HEADER_ROW = 10

# Global Mordred calculator (2D descriptors only)
MORDRED_CALC = Calculator(descriptors, ignore_3D=True)



# Raw data loading and basic cleaning
def get_raw_data_path() -> Path:
    """Return the path to the raw Excel data file."""
    path = DATA_DIR / RAW_XLSX_NAME
    if not path.exists():
        raise FileNotFoundError(f"Raw data file not found: {path}")
    return path


def load_raw_dataframe() -> pd.DataFrame:
    """
    Load the raw Excel sheet into a DataFrame.
    """
    path = get_raw_data_path()
    df = pd.read_excel(
        path,
        sheet_name=SHEET_NAME,
        engine="openpyxl",
        usecols=USECOLS,
        header=HEADER_ROW,
    )
    return df

def clean_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    """
    Basic cleaning of the raw DataFrame:
    - Drop fully empty rows
    - Trim to first/last valid index
    - Drop unused columns by position
    - Normalize column names
    - Check that MSR is strictly positive
    """
    # Drop fully empty rows
    df = df.dropna(how="all")

    # Trim to valid index range
    first_idx, last_idx = df.first_valid_index(), df.last_valid_index()
    df = df.loc[first_idx:last_idx]

    # Normalize column names
    rename_map = {
        "Contaminant Type": "ContaminantType",
        "Surfactant Type": "SurfactantType",
        "Contaminant SMILES": "SMILES_contaminant",
        "Surfactant SMILES": "SMILES_surfactant",
        "CMC (mol/L)": "CMC",
        "Water solubility (g/L)": "WaterSolubility",
        "MSR": "MSR",
    }
    df = df.rename(columns=rename_map)

    # Convert MSR to numeric and keep only valid rows
    msr_numeric = pd.to_numeric(df["MSR"], errors="coerce")
    mask_valid = msr_numeric.notna()
    df = df.loc[mask_valid].copy()
    df["MSR"] = msr_numeric[mask_valid]

     # Ensure MSR > 0 for later log10 transforms
    msr = pd.to_numeric(df["MSR"], errors="coerce")
    if not (msr > 0).all():
        raise ValueError("MSR contains non-positive values; log10 transform would fail.")

    return df


def _save_log10_msr_histogram(
    df: pd.DataFrame,
    filename: str = "log10_msr_hist.pdf",
) -> None:
    """Save histogram of log10(MSR) to disk."""
    set_plot_style()
    if "MSR" not in df.columns:
        return

    msr = pd.to_numeric(df["MSR"], errors="coerce")
    msr = msr[msr > 0]

    if msr.empty:
        return

    log_msr = np.log10(msr.to_numpy(dtype=float))

    DATA_FIGURES_DIR.mkdir(parents=True, exist_ok=True)

    color = get_base_color()

    fig, ax = plt.subplots(figsize=(4.8, 4.0))
    ax.hist(log_msr, bins=20, color=color, edgecolor="black", linewidth=0.4)
    ax.set_xlabel("log10(MSR)")
    ax.set_ylabel("Count")

    ax.yaxis.set_major_locator(MaxNLocator(integer=True))

    fig.tight_layout()
    fig.savefig(DATA_FIGURES_DIR / filename)
    plt.close(fig)


def _save_log10_msr_violin_panels(
    df: pd.DataFrame,
    filename: str = "logMSR_violin_panels.pdf",
) -> None:
    """Save 1x2 violin panels of log10(MSR) by contaminant and surfactant types."""
    required_cols = {"MSR", "ContaminantType", "SurfactantType"}
    if not required_cols.issubset(df.columns):
        return

    msr = pd.to_numeric(df["MSR"], errors="coerce")
    mask = msr > 0
    if not mask.any():
        return

    df_plot = df.loc[mask, ["ContaminantType", "SurfactantType"]].copy()
    # Normalize categorical labels to avoid hidden spacing artifacts in category alignment.
    df_plot["ContaminantType"] = df_plot["ContaminantType"].astype(str).str.strip()
    df_plot["SurfactantType"] = df_plot["SurfactantType"].astype(str).str.strip()
    df_plot["log10_MSR"] = np.log10(msr.loc[mask].to_numpy(dtype=float))

    cont_order = (
        df_plot.groupby("ContaminantType")["log10_MSR"].median().sort_values().index.tolist()
    )
    surf_order = (
        df_plot.groupby("SurfactantType")["log10_MSR"].median().sort_values().index.tolist()
    )
    palette3 = get_discrete_colors(3)

    DATA_FIGURES_DIR.mkdir(parents=True, exist_ok=True)

    # Ensure this figure keeps manuscript style even when called standalone.
    set_plot_style()
    with plt.rc_context(
        {
            "font.size": 11,
            "axes.labelsize": 12,
            "xtick.labelsize": 12,
            "ytick.labelsize": 12,
        }
    ):
        def _overlay_white_medians(ax, x_col: str, order: list) -> None:
            """Draw visible white median marks over seaborn's inner box."""
            med_w = 0.024
            for i, cat in enumerate(order):
                vals = pd.to_numeric(
                    df_plot.loc[df_plot[x_col] == cat, "log10_MSR"],
                    errors="coerce",
                ).dropna()
                if vals.empty:
                    continue
                med = float(np.median(vals.to_numpy(dtype=float)))
                ax.plot(
                    [i - med_w / 2.0, i + med_w / 2.0],
                    [med, med],
                    color="white",
                    linewidth=0.75,
                    solid_capstyle="butt",
                    zorder=7,
                )

        fig, axes = plt.subplots(
            1,
            2,
            figsize=(10, 4),
            sharey=True,
            gridspec_kw={"wspace": 0.20},
        )
        ax1, ax2 = axes
        # Match notebook logic as closely as possible under seaborn 0.11.
        sns.violinplot(
            data=df_plot,
            x="ContaminantType",
            y="log10_MSR",
            hue="ContaminantType",
            order=cont_order,
            hue_order=cont_order,
            inner="box",
            dodge=False,
            width=0.8,
            scale="count",
            bw=0.65,
            cut=2,
            linewidth=0.5,
            palette=palette3,
            ax=ax1,
        )
        if ax1.get_legend() is not None:
            ax1.get_legend().remove()

        ax1.set_xlabel("Contaminant type")
        ax1.xaxis.labelpad = 10
        ax1.set_ylabel(r"$\log_{10}(\mathrm{MSR})$")
        ax1.set_ylim(-5.2, 3.2)
        _overlay_white_medians(ax1, "ContaminantType", cont_order)

        sns.violinplot(
            data=df_plot,
            x="SurfactantType",
            y="log10_MSR",
            hue="SurfactantType",
            order=surf_order,
            hue_order=surf_order,
            inner="box",
            dodge=False,
            width=0.8,
            scale="count",
            bw=0.65,
            cut=2,
            linewidth=0.5,
            palette=palette3,
            ax=ax2,
        )
        if ax2.get_legend() is not None:
            ax2.get_legend().remove()

        ax2.set_xlabel("Surfactant type")
        ax2.xaxis.labelpad = 10
        ax2.set_ylabel("")
        _overlay_white_medians(ax2, "SurfactantType", surf_order)

        for ax in (ax1, ax2):
            ax.set_axisbelow(True)
            ax.yaxis.grid(True, linestyle="--", linewidth=0.4, color="#B7B6B6", alpha=1.0)
            ax.xaxis.grid(False)

        fig.subplots_adjust(left=0.07, right=0.98, top=0.97, bottom=0.17, wspace=0.20)
        fig.savefig(DATA_FIGURES_DIR / filename)
        plt.close(fig)


def prepare_base_dataframe() -> pd.DataFrame:
    """
    Load and clean the raw data, returning the base DataFrame.

    This function does not compute descriptors or add RDKit Mol objects.
    """
    df_raw = load_raw_dataframe()
    df_clean = clean_dataframe(df_raw)

    # Histogram of log10(MSR)
    _save_log10_msr_histogram(df_clean)
    _save_log10_msr_violin_panels(df_clean)
    return df_clean.copy()



# SMILES canonicalization and Mordred descriptors
def canonical_main_organic_fragment(smiles: str) -> Optional[str]:
    """
    Return canonical SMILES of the main organic fragment using RDKit.

    - Splits into fragments
    - Keeps only fragments with at least one carbon atom
    - Chooses the largest organic fragment by number of heavy atoms
    - Returns its canonical SMILES
    """
    if pd.isna(smiles):
        return None

    try:
        mol = Chem.MolFromSmiles(str(smiles))
        if mol is None:
            return None

        frags = Chem.GetMolFrags(mol, asMols=True)

        if len(frags) == 1:
            return Chem.MolToSmiles(frags[0], canonical=True)

        organic_frags = []
        for f in frags:
            has_carbon = any(atom.GetAtomicNum() == 6 for atom in f.GetAtoms())
            if has_carbon:
                organic_frags.append(f)

        if not organic_frags:
            organic_frags = frags

        main_frag = max(organic_frags, key=lambda m: m.GetNumHeavyAtoms())
        return Chem.MolToSmiles(main_frag, canonical=True)

    except Exception:
        return None


def compute_mordred_descriptors(molecules: pd.Series, prefix: str) -> pd.DataFrame:
    """
    Compute Mordred descriptors for a Series of RDKit Mol objects.

    Returns a numeric DataFrame with the given prefix applied to column names.
    """
    dfx = MORDRED_CALC.pandas(molecules, quiet=True)
    dfx = dfx.add_prefix(prefix)
    dfx = dfx.apply(pd.to_numeric, errors="coerce")

    return dfx


def add_descriptors(df_base: pd.DataFrame) -> pd.DataFrame:
    """
    Add canonical SMILES and Mordred descriptors for contaminants and surfactants.

    Workflow:
    - Canonicalize SMILES_contaminant and SMILES_surfactant
    - Drop rows with invalid SMILES in either column
    - Build RDKit Mol objects
    - Compute Mordred descriptors for both sets
    - Concatenate base data with descriptors
    """
    df = df_base.copy()
    
    # Canonical SMILES
    df["SMILES_contaminant"] = df["SMILES_contaminant"].apply(
        canonical_main_organic_fragment
    )
    df["SMILES_surfactant"] = df["SMILES_surfactant"].apply(
        canonical_main_organic_fragment
    )

    # Drop rows with invalid SMILES
    n_invalid_cont = df["SMILES_contaminant"].isna().sum()
    n_invalid_surf = df["SMILES_surfactant"].isna().sum()
    if n_invalid_cont or n_invalid_surf:
        print(
            f"Dropping rows with invalid SMILES: "
            f"{n_invalid_cont} contaminants, {n_invalid_surf} surfactants."
        )

    df = df.dropna(subset=["SMILES_contaminant", "SMILES_surfactant"]).reset_index(
        drop=True
    )

    # RDKit Mol objects
    df["Mol_contaminant"] = df["SMILES_contaminant"].apply(Chem.MolFromSmiles)
    df["Mol_surfactant"] = df["SMILES_surfactant"].apply(Chem.MolFromSmiles)

    # Compute descriptors
    desc_contaminant = compute_mordred_descriptors(df["Mol_contaminant"], "Cont_")
    desc_surfactant = compute_mordred_descriptors(df["Mol_surfactant"], "Surf_")

    # Drop helper Mol columns
    df_base_clean = df.drop(columns=["Mol_contaminant", "Mol_surfactant"])

    # Merge base data with descriptors
    df_full = pd.concat(
        [
            df_base_clean.reset_index(drop=True),
            desc_contaminant.reset_index(drop=True),
            desc_surfactant.reset_index(drop=True),
        ],
        axis=1,
    )

    return df_full


def prepare_full_dataframe(
    use_cache: bool = True,
    force_recompute: bool = False,
) -> pd.DataFrame:
    """
    Return the full modeling DataFrame with descriptors included.

    If use_cache is True and a processed file exists, load it from disk.
    Otherwise compute descriptors, save, and return.
    """
    if use_cache and PROCESSED_FULL_DF_PATH.exists() and not force_recompute:
        df_full = pd.read_pickle(PROCESSED_FULL_DF_PATH)
        df_full = df_full.reset_index(drop=True)
        return df_full

    df_base = prepare_base_dataframe()
    df_full = add_descriptors(df_base)
    df_full = df_full.reset_index(drop=True)
    df_full.insert(0, "SampleID", np.arange(len(df_full)))

    DATA_ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)
    df_full.to_pickle(PROCESSED_FULL_DF_PATH)

    return df_full
