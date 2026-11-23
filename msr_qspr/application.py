"""
MSR Estimator inference application.

Run from repo root:
    python -m msr_qspr app
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import streamlit as st
from msr_qspr.inference import ARTIFACTS

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------


st.set_page_config(page_title="MSR Estimator", page_icon="MSR", layout="wide", initial_sidebar_state="collapsed")


# ---------------------------------------------------------------------------
# Data loading (cached)
# ---------------------------------------------------------------------------

@st.cache_resource(show_spinner=False)
def load_predictor():
    from msr_qspr.inference import MSRPredictor

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
    except (FileNotFoundError, ValueError) as exc:
        placeholder.empty()
        st.error("Compatible model artifacts and training references are required.")
        st.code(str(exc))
        st.stop()

    placeholder.empty()
    return predictor


@st.cache_data(show_spinner=False)
def _load_ui_smiles_df() -> pd.DataFrame:
    needed = ["Contaminant", "SMILES_contaminant", "Surfactant", "SMILES_surfactant"]

    from msr_qspr.data import corpus_metadata
    return corpus_metadata()[needed].copy()


@st.cache_data(show_spinner=False)
def _run_metadata_summary() -> dict[str, object]:
    from msr_qspr.inference import report_table
    bundle = load_predictor().bundle
    holdout = report_table(ARTIFACTS, 'holdout_predictions.csv')
    rmse = None
    if holdout is not None and {'actual_log10_MSR', 'predicted_log10_MSR'} <= set(holdout):
        residual = holdout['predicted_log10_MSR'] - holdout['actual_log10_MSR']
        rmse = float(np.sqrt(np.mean(np.square(residual))))
    return {'n_pairs': len(bundle['development_ids']),
            'n_features': len(bundle['effective_features']),
            'model': bundle['experiment']['estimator'], 'rmse_holdout': rmse}


@st.cache_data(show_spinner=False)
def _load_compound_lookup() -> tuple[dict, dict]:
    """Map normalized names to (dataset name, SMILES) for contaminants and surfactants."""
    df = _load_ui_smiles_df()
    if df.empty:
        return {}, {}

    def lookup(name_col: str, smiles_col: str) -> dict[str, tuple[str, str]]:
        rows = df[[name_col, smiles_col]].dropna().drop_duplicates(name_col).sort_values(name_col)
        return {str(name).strip().lower(): (str(name).strip(), str(smiles).strip())
                for name, smiles in zip(rows[name_col], rows[smiles_col])}

    return lookup("Contaminant", "SMILES_contaminant"), lookup("Surfactant", "SMILES_surfactant")


@st.cache_data(show_spinner=False)
def _load_example_pool() -> list[tuple[str, str, str, str, str]]:
    df = _load_ui_smiles_df()
    if df.empty:
        return []

    pairs = df.dropna().drop_duplicates(["Contaminant", "Surfactant"])
    if pairs.empty:
        return []

    out: list[tuple[str, str, str, str, str]] = []
    for row in pairs.itertuples(index=False):
        cont = str(row.Contaminant).strip()
        surf = str(row.Surfactant).strip()
        cont_smi = str(row.SMILES_contaminant).strip()
        surf_smi = str(row.SMILES_surfactant).strip()
        if not cont or not surf or not cont_smi or not surf_smi:
            continue
        out.append((f"{cont} + {surf}", cont, cont_smi, surf, surf_smi))

    return out


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
    from msr_qspr.ui.styles import inject_css
    from msr_qspr.ui.components import inject_mol_modal, inject_compound_datalists, render_topbar
    from msr_qspr.ui.pages import render_left_panel, render_welcome, render_prediction_workspace

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

    left, right = st.columns([2.8, 7.2], gap="medium")
    with left:
        render_left_panel(predictor, _load_compound_lookup)

    with right:
        if st.session_state["last_result"] is None:
            render_welcome(ARTIFACTS, _get_welcome_examples, _run_metadata_summary())
        else:
            render_prediction_workspace(st.session_state["last_inputs"], st.session_state["last_result"], predictor, ARTIFACTS)

    st.markdown(
        "<div class='app-footer'>© 2026 Juan Marcos Jara Elizeche</div>",
        unsafe_allow_html=True,
    )


if __name__ == "__main__":
    main()
