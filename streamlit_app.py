"""
streamlit_app.py - MSR Estimator web app.

Run from repo root:
    streamlit run streamlit_app.py
"""

from __future__ import annotations

import os
from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

def _resolve_artifacts_dir() -> Path:
    env_dir = os.getenv("MSR_ARTIFACTS_DIR")
    if env_dir:
        return Path(env_dir)
    return Path(__file__).resolve().parent / "artifacts"


ARTIFACTS = _resolve_artifacts_dir()

st.set_page_config(page_title="MSR Estimator", page_icon="MSR", layout="wide", initial_sidebar_state="collapsed")


# ---------------------------------------------------------------------------
# Data loading (cached)
# ---------------------------------------------------------------------------

@st.cache_resource(show_spinner=False)
def load_predictor():
    from app_core import MSRPredictor

    placeholder = st.empty()
    placeholder.markdown(
        """
<div style="display:flex; flex-direction:column; align-items:center; justify-content:center;
            min-height:60vh; gap:18px; font-family:'Space Grotesk',sans-serif;">
    <svg width="48" height="48" viewBox="0 0 48 48" xmlns="http://www.w3.org/2000/svg">
      <circle cx="24" cy="24" r="20" stroke="#d4e8e5" stroke-width="4" fill="none"/>
      <path d="M24 4 A20 20 0 0 1 44 24" stroke="#0c8a7f" stroke-width="4" fill="none" stroke-linecap="round">
        <animateTransform attributeName="transform" type="rotate" from="0 24 24" to="360 24 24" dur="0.9s" repeatCount="indefinite"/>
      </path>
    </svg>
    <div style="font-size:1.15rem; font-weight:600; color:#24445f;">Initialising MSR Estimator</div>
    <div style="font-size:0.88rem; color:#7b8a9e;">Loading model, training references and applicability domain</div>
</div>
        """,
        unsafe_allow_html=True,
    )

    try:
        predictor = MSRPredictor()
    except FileNotFoundError as exc:
        placeholder.empty()
        st.error("Required model artifacts are missing.")
        st.code(str(exc))
        st.info("Run `python msr_pipeline.py` from repository root, then restart this app.")
        st.stop()

    placeholder.empty()
    return predictor


@st.cache_data(show_spinner=False)
def _load_ui_smiles_df() -> pd.DataFrame:
    needed = ["Contaminant", "SMILES_contaminant", "Surfactant", "SMILES_surfactant"]

    # Prefer cleaned raw data so UI chemistry cards use full-input SMILES.
    try:
        from data_prep import clean_dataframe, load_raw_dataframe

        raw_df = clean_dataframe(load_raw_dataframe())
        if set(needed).issubset(raw_df.columns):
            return raw_df[needed].copy()
    except Exception:
        pass

    # Fallback to canonicalized artifact dataset if raw data is unavailable.
    path = ARTIFACTS / "data/full_dataset_with_descriptors.pkl"
    if not path.exists():
        return pd.DataFrame(columns=needed)

    df = pd.read_pickle(path)
    if not set(needed).issubset(df.columns):
        return pd.DataFrame(columns=needed)

    return df[needed].copy()


@st.cache_data(show_spinner=False)
def _load_compound_lookup() -> tuple[dict, dict]:
    df = _load_ui_smiles_df()
    if df.empty:
        return {}, {}
    # Normalize keys: lowercase and strip spaces for robust matching
    conts = (df[["Contaminant", "SMILES_contaminant"]]
             .drop_duplicates("Contaminant").sort_values("Contaminant"))
    surfs = (df[["Surfactant", "SMILES_surfactant"]]
             .drop_duplicates("Surfactant").sort_values("Surfactant"))
    cont_lookup = {str(k).strip().lower(): v for k, v in zip(conts["Contaminant"], conts["SMILES_contaminant"])}
    surf_lookup = {str(k).strip().lower(): v for k, v in zip(surfs["Surfactant"], surfs["SMILES_surfactant"])}
    return cont_lookup, surf_lookup


@st.cache_data(show_spinner=False)
def _load_example_pool() -> list[tuple[str, str, str, str, str]]:
    fallback = [
        ("Naphthalene + TX-100", "Naphthalene", "c1ccc2ccccc2c1", "Triton X-100", "CC(C)(C)CCc1ccc(OCCOCCOCCOCCOCCOCCOCCOCCO)cc1"),
        ("Pyrene + SDBS", "Pyrene", "c1cc2ccc3cccc4ccc(c1)c2c34", "SDBS", "CCCCCCCCCCCSc1ccc(S(=O)(=O)[O-])cc1"),
        ("Phenanthrene + Tween 80", "Phenanthrene", "c1ccc2c(c1)ccc3ccccc23", "Tween 80", "CCCCCCCCC=CCCCCCCCC(=O)OCC(CO)OCC"),
    ]
    df = _load_ui_smiles_df()
    if df.empty:
        return fallback

    pairs = df.dropna().drop_duplicates(["Contaminant", "Surfactant"])
    if pairs.empty:
        return fallback

    out: list[tuple[str, str, str, str, str]] = []
    for row in pairs.itertuples(index=False):
        cont = str(row.Contaminant).strip()
        surf = str(row.Surfactant).strip()
        cont_smi = str(row.SMILES_contaminant).strip()
        surf_smi = str(row.SMILES_surfactant).strip()
        if not cont or not surf or not cont_smi or not surf_smi:
            continue
        out.append((f"{cont} + {surf}", cont, cont_smi, surf, surf_smi))

    return out if out else fallback


def _pick_examples(k: int = 3) -> list[tuple[str, str, str, str, str]]:
    pool = _load_example_pool()
    if len(pool) <= k:
        return pool
    rng = np.random.default_rng()
    idx = rng.choice(len(pool), size=k, replace=False)
    return [pool[int(i)] for i in idx]


def _get_welcome_examples(k: int = 3) -> list[tuple[str, str, str, str, str]]:
    key = "_welcome_examples"
    examples = st.session_state.get(key)
    if isinstance(examples, list) and len(examples) == k:
        return examples
    picked = _pick_examples(k=k)
    st.session_state[key] = picked
    return picked


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    from app.styles import inject_css
    from app.components import inject_mol_modal, inject_compound_datalists, render_topbar
    from app.pages import render_left_panel, render_welcome, render_prediction_workspace

    inject_css()
    predictor = load_predictor()

    # Apply example payload before widgets are instantiated in this run.
    pending_example = st.session_state.pop("pending_example", None)
    if pending_example is not None:
        st.session_state["_cont_search"] = pending_example.get("cont_smiles", "")
        st.session_state["_surf_search"] = pending_example.get("surf_smiles", "")
        st.session_state["_example_cont_label"] = pending_example.get("cont_name", "Contaminant")
        st.session_state["_example_surf_label"] = pending_example.get("surf_name", "Surfactant")
        st.session_state["_run_example"] = True
    st.session_state.setdefault("last_inputs", None)
    st.session_state.setdefault("last_result", None)

    inject_mol_modal()

    cont_lookup, surf_lookup = _load_compound_lookup()
    inject_compound_datalists(cont_lookup, surf_lookup)

    render_topbar()
    st.markdown("<div style='height:10px'></div>", unsafe_allow_html=True)

    left, right = st.columns([2.8, 7.2], gap="medium", vertical_alignment="top")
    with left:
        render_left_panel(predictor, _load_compound_lookup)

    with right:
        if st.session_state["last_result"] is None:
            render_welcome(ARTIFACTS, _get_welcome_examples)
        else:
            render_prediction_workspace(st.session_state["last_inputs"], st.session_state["last_result"], predictor, ARTIFACTS)

    st.markdown(
        "<div class='app-footer'>© 2026 Juan Marcos Jara Elizeche</div>",
        unsafe_allow_html=True,
    )


if __name__ == "__main__":
    main()
