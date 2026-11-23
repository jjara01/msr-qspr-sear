"""Page renderers: welcome screen, left panel, prediction workspace and tabs."""

from __future__ import annotations

import base64
import html
import warnings
from pathlib import Path
import numpy as np
import streamlit as st
from msr_qspr.ui.charts import shap_bar, distribution, parity, global_shap
from msr_qspr.inference import report_table
from msr_qspr.ui.components import (
    molecule_image,
    molecule_props,
    pubchem_name,
    status,
    with_info,
    truncate_smiles,
    render_chart_card,
    render_example_molecule_panel,
    render_compound_search,
    descriptor_table,
)

SURFACTANT_MW = {
    'Tween 40':       1284.5,
    'Tween 80':       1310.0,
    'Tween 20':       1227.0,
    'Brij-58':        1123.0,
    'Brij-35':        1198.0,
    'Synperonic™ 13/8': 690.0,
    'Igepal CA-720':   735.0,
    'Igepal CO-720':   749.0,
    'Tergitol NP-10':  603.0,
}

_SURFACTANT_MW_LOWER = {name.lower(): mw for name, mw in SURFACTANT_MW.items()}


def _nominal_surfactant_mw(surf_name) -> float | None:
    return _SURFACTANT_MW_LOWER.get(str(surf_name).strip().lower())


def get_surfactant_mw(surf_name, mw_computed):
    nominal = _nominal_surfactant_mw(surf_name)
    return mw_computed if nominal is None else nominal


def _criterion_state(criterion) -> tuple[str, str]:
    """Display label and tone (pass, mid, fail) for one applicability criterion."""
    if criterion.availability_status in {"incomplete", "unavailable"}:
        return criterion.availability_status.upper(), "mid"
    return ("PASS", "pass") if criterion.pass_ else ("FAIL", "fail")


_STATUS_TONE = {"ad-ok": "ok", "ad-mid": "mid", "ad-bad": "bad"}


def _set_example(cont_name: str, cont_smiles: str, surf_name: str, surf_smiles: str) -> None:
    st.session_state["pending_example"] = {
        "cont_name": cont_name,
        "cont_smiles": cont_smiles,
        "surf_name": surf_name,
        "surf_smiles": surf_smiles,
    }


def _display_name(smiles: str) -> str:
    """Readable label for a structure entered as SMILES: PubChem IUPAC name or truncated SMILES."""
    name = pubchem_name(smiles)
    if name:
        return name[:60] + "…" if len(name) > 60 else name
    return truncate_smiles(smiles, head=20, tail=8)


def _predict_from_state(predictor, load_compound_lookup) -> tuple[dict | None, object | None]:
    cont_lookup, surf_lookup = load_compound_lookup()

    cont_raw = st.session_state.get("_cont_search", "").strip()
    surf_raw = st.session_state.get("_surf_search", "").strip()

    cont_name, cont_smiles = cont_lookup.get(cont_raw.lower(), (None, cont_raw))
    surf_name, surf_smiles = surf_lookup.get(surf_raw.lower(), (None, surf_raw))
    cont_name = st.session_state.pop("_example_cont_label", None) or cont_name
    surf_name = st.session_state.pop("_example_surf_label", None) or surf_name

    if not cont_smiles:
        st.error("Contaminant not found in database. Please enter a valid SMILES string for the contaminant.")
        return None, None
    if not surf_smiles:
        st.error("Surfactant not found in database. Please enter a valid SMILES string for the surfactant.")
        return None, None

    ok_c, msg_c = predictor.validate_smiles(cont_smiles)
    ok_s, msg_s = predictor.validate_smiles(surf_smiles, surfactant=True)
    if not ok_c:
        st.error(f"Contaminant input is not a valid SMILES string. Error: {msg_c}\nIf you intended to use a common name, please select one from the dropdown or check spelling.")
        return None, None
    if not ok_s:
        st.error(f"Surfactant input is not supported. {msg_s}")
        return None, None

    properties = {}
    for name, key, unit in (('CMC', '_reference_cmc', 'mol/L'),
                            ('WaterSolubility', '_reference_solubility', 'g/L')):
        if name not in getattr(predictor, 'required_reference_properties', ()):
            continue
        raw = str(st.session_state.get(key, '')).strip()
        if raw:
            try:
                value = float(raw)
            except ValueError:
                value = float('nan')
            if not np.isfinite(value) or value <= 0:
                st.error(f'{name} must be a finite positive number in {unit}.')
                return None, None
            properties['cmc' if name == 'CMC' else 'water_solubility'] = value
    with st.spinner("Computing descriptors and estimation..."):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            if cont_name:
                properties['contaminant_name'] = cont_name
            try:
                result = predictor.predict(cont_smiles=cont_smiles, surf_smiles=surf_smiles, **properties)
            except ValueError as error:
                st.error(f"Estimation is not available for this pair. {error}")
                return None, None

    inputs = {
        "cont_smiles": cont_smiles,
        "surf_smiles": surf_smiles,
        "cont_label": cont_name or _display_name(cont_smiles),
        "surf_label": surf_name or _display_name(surf_smiles),
    }
    return inputs, result


# ---------------------------------------------------------------------------
# Left panel
# ---------------------------------------------------------------------------

def render_left_panel(predictor, load_compound_lookup) -> None:
    with st.container(border=True):
        tip_text = (
            "As you type, a dropdown will suggest contaminants and surfactants "
            "registered in the database (by their common names). If your compound "
            "is not found, please enter a valid SMILES string instead."
        )
        st.markdown(f"### {with_info('New Estimation', tip_text)}", unsafe_allow_html=True)

        render_compound_search("Contaminant", "_cont_search")
        render_compound_search("Surfactant",   "_surf_search")

        required = getattr(predictor, 'required_reference_properties', ())
        if required:
            st.caption('This model also uses independent reference properties beyond molecular structures. Leave a value blank to use its fitted training imputation; the estimate will identify imputed inputs.')
            if 'CMC' in required:
                st.text_input('Reference CMC (mol/L)', key='_reference_cmc',
                              help='Independent critical micelle concentration; finite positive mol/L value.')
            if 'WaterSolubility' in required:
                st.text_input('Reference aqueous solubility (g/L)', key='_reference_solubility',
                              help='Independent aqueous contaminant solubility; finite positive g/L value.')

        run_now = st.session_state.pop("_run_example", False)
        if st.button("Estimate MSR", type="primary", width="stretch") or run_now:
            inputs, result = _predict_from_state(predictor, load_compound_lookup)
            if result is not None:
                st.session_state["last_inputs"] = inputs
                st.session_state["last_result"] = result

        st.markdown("<div style='height:6px'></div>", unsafe_allow_html=True)
        st.markdown(
            "<div class='warn-box'><b>Model scope.</b> Trained on a curated literature dataset. "
            "Use extra caution for out-of-domain estimates.</div>",
            unsafe_allow_html=True,
        )
        st.markdown("<div style='height:4px'></div>", unsafe_allow_html=True)

    if st.session_state.get("last_result") is not None:
        inp = st.session_state["last_inputs"]
        st.markdown("<div style='height:10px'></div>", unsafe_allow_html=True)
        st.markdown("<div class='label-overline' style='margin-bottom:6px;'>Selected system</div>", unsafe_allow_html=True)

        for title_key, smi_key in [("cont_label", "cont_smiles"), ("surf_label", "surf_smiles")]:
            title = inp[title_key]
            smi = inp[smi_key]
            role = "Contaminant" if title_key == "cont_label" else "Surfactant"

            safe_title = html.escape(str(title), quote=True)
            safe_role = html.escape(role)

            # Safe defaults in case RDKit fails
            mw, logp = None, None
            img, detail = None, None
            try:
                mw, logp = molecule_props(smi)
            except Exception:
                pass
            try:
                img = molecule_image(smi, (700, 430))
            except Exception:
                pass
            try:
                detail = molecule_image(smi, (980, 560))
            except Exception:
                pass

            with st.container(border=True):
                st.markdown(
                    f"""
<div class='selected-system-card'>
    <div class='selected-system-role'>{safe_role}</div>
    <div class='selected-system-name'>{safe_title}</div>
</div>
                    """,
                    unsafe_allow_html=True,
                )
                if img is not None:
                    img_uri = f"data:image/png;base64,{base64.b64encode(img).decode('ascii')}"
                    detail_bytes = detail if detail is not None else img
                    detail_uri = f"data:image/png;base64,{base64.b64encode(detail_bytes).decode('ascii')}"
                    st.markdown(
                        f"""
<div class='selected-system-mol-wrap'>
    <img class='selected-system-mol mol-zoomable'
             src='{img_uri}'
             data-detail='{detail_uri}'
             data-title="{safe_title}"
             alt="{safe_title}" />
</div>
                        """,
                        unsafe_allow_html=True,
                    )
                badge_line = ""
                if role == "Surfactant":
                    mw_display = get_surfactant_mw(title, mw)
                    if mw_display is not None:
                        badge_line += f"<span class='badge badge-teal'>MW {mw_display:.2f}</span>"
                else:
                    if mw is not None:
                        badge_line += f"<span class='badge badge-teal'>MW {mw:.2f}</span>"
                if logp is not None:
                    badge_line += f"<span class='badge badge-warn'>LogP {logp:.2f}</span>"
                if badge_line:
                    st.markdown(f"<div class='selected-system-metrics'>{badge_line}</div>", unsafe_allow_html=True)


# ---------------------------------------------------------------------------
# Welcome
# ---------------------------------------------------------------------------

def render_welcome(artifacts: Path, get_welcome_examples, study_summary: dict[str, object]) -> None:
    n_pairs = study_summary.get("n_pairs")
    n_features = study_summary.get("n_features")
    model_name = study_summary.get("model")
    rmse_holdout = study_summary.get("rmse_holdout")

    pairs_badge = f"{int(n_pairs)} pairs" if isinstance(n_pairs, int) else "Study dataset"
    feature_badge = f"{int(n_features)} selected features" if isinstance(n_features, int) else "Selected features"
    model_label = str(model_name) if model_name else "Reporting model"
    rmse_label = f"{float(rmse_holdout):.3f}" if isinstance(rmse_holdout, (int, float)) else "—"

    hero_cols = st.columns([3.2, 1.25])
    with hero_cols[0]:
        st.markdown(
            f"""
<div class='welcome-hero'>
    <div class='welcome-title'>MSR Estimator</div>
    <div class='welcome-sub'>QSPR-based estimation tool for contaminant–surfactant molar solubilization ratios</div>
    <div>
        <span class='badge badge-teal'>{pairs_badge}</span>
        <span class='badge badge-teal'>{feature_badge}</span>
        <span class='badge badge-teal'>Grouped CV</span>
    </div>
</div>
            """,
            unsafe_allow_html=True,
        )
    with hero_cols[1]:
        st.markdown(
            f"""
<div class='hero-mini-stack'>
    <div class='hero-mini-kpi'>
        <div class='k'>Model</div>
        <div class='v'>{html.escape(model_label)}</div>
    </div>
    <div class='hero-mini-kpi'>
        <div class='k'>Holdout RMSE</div>
        <div class='v'>{rmse_label}</div>
    </div>
</div>
            """,
            unsafe_allow_html=True,
        )

    st.markdown("<div style='height:14px'></div>", unsafe_allow_html=True)

    ex_cols = st.columns(3)
    examples = get_welcome_examples(k=3)

    for idx, (col, ex) in enumerate(zip(ex_cols, examples)):
        ex_title = html.escape(str(ex[0]))
        ex_cont_name = html.escape(str(ex[1]))
        ex_surf_name = html.escape(str(ex[3]))
        with col:
            with st.container(border=True):
                st.markdown(f"<div class='example-title'>{ex_title}</div>", unsafe_allow_html=True)
                name_cols = st.columns(2, gap="small")
                with name_cols[0]:
                    st.markdown("<div class='example-label'>Contaminant name</div>", unsafe_allow_html=True)
                    st.markdown(f"<div class='example-chem-name'>{ex_cont_name}</div>", unsafe_allow_html=True)
                with name_cols[1]:
                    st.markdown("<div class='example-label'>Surfactant name</div>", unsafe_allow_html=True)
                    st.markdown(f"<div class='example-chem-name'>{ex_surf_name}</div>", unsafe_allow_html=True)

                mol_cols = st.columns(2, gap="small")
                with mol_cols[0]:
                    render_example_molecule_panel(ex[2], ex[1], key_prefix=f"ex_{idx}_cont")
                with mol_cols[1]:
                    render_example_molecule_panel(ex[4], ex[3], key_prefix=f"ex_{idx}_surf")

                st.markdown("<div style='height:10px'></div>", unsafe_allow_html=True)
                cta_cols = st.columns([1, 1.45, 1])
                with cta_cols[1]:
                    clicked = st.button("Try this", key=f"try_{idx}", width="stretch", type="secondary")
                if clicked:
                    _set_example(ex[1], ex[2], ex[3], ex[4])
                    st.rerun()

    fig = parity(artifacts)
    if fig is not None:
        render_chart_card("Parity plot", fig)

    st.markdown("<div class='overview-note'>Enter SMILES in the left panel and click Estimate MSR to open full analysis tabs.</div>", unsafe_allow_html=True)


# ---------------------------------------------------------------------------
# Prediction workspace (header KPIs + tabs)
# ---------------------------------------------------------------------------

def _render_header_kpis(inputs: dict, result, artifacts: Path) -> None:
    status_txt, status_cls, n_pass = status(result.ad)
    pass_pct = (n_pass / 3.0) * 100.0
    tone = _STATUS_TONE[status_cls]

    cols = st.columns([4.6, 2, 2.4, 3])
    pair_name = html.escape(f"{inputs['cont_label']} + {inputs['surf_label']}")
    with cols[0]:
        st.markdown(
            f"""
<div class='metric-card'>
    <div class='pair-title'>{pair_name}</div>
    <div class='small'>Current estimation target pair</div>
</div>
            """,
            unsafe_allow_html=True,
        )

    with cols[1]:
        st.markdown(
            f"""
    <div class='metric-card'>
    <div class='metric-label'>{with_info('log10(MSR)', 'Model estimate in logarithmic scale (base 10).')}</div>
    <div class='metric-value'>{result.log10_msr:+.3f}</div>
    <div class='metric-sub'>Logarithmic prediction</div>
</div>
            """,
            unsafe_allow_html=True,
        )

    with cols[2]:
        unit_mode = st.session_state.get("_msr_card_unit", "M")

        msr_label = "MSR"
        msr_help = "Linear-scale estimate converted from log10(MSR), reported as mol/mol."
        msr_value = float(result.msr)
        msr_sub = "mol/mol"

        if unit_mode == "W":
            mw_cont, _ = molecule_props(str(inputs.get("cont_smiles", "")))
            mw_surf, _ = molecule_props(str(inputs.get("surf_smiles", "")))
            surf_label = str(inputs.get("surf_label", ""))
            mw_surf_final = get_surfactant_mw(surf_label, mw_surf)
            nominal_surf = _nominal_surfactant_mw(surf_label) is not None
            msr_label = "WSR"
            msr_help = "Weight solubilization ratio: WSR = MSR * (MW_contaminant / MW_surfactant). Contaminant MW source: SMILES (RDKit). "
            if nominal_surf:
                msr_help += "Surfactant MW source: commercial nominal value; WSR is approximate for this commercial mixture."
            else:
                msr_help += "Surfactant MW source: SMILES (RDKit); WSR is approximate if the SMILES is representative of a mixture."
            if all(isinstance(mw, (int, float, np.integer, np.floating))
                   and not isinstance(mw, (bool, np.bool_))
                   and np.isfinite(mw) and mw > 0 for mw in (mw_cont, mw_surf_final)):
                msr_value = msr_value * (mw_cont / mw_surf_final)
                msr_sub = "g/g (approx.; nominal surfactant MW)" if nominal_surf else "g/g"
            else:
                msr_value = None
                msr_help += " WSR cannot be calculated because a required molecular weight is unavailable or invalid (must be finite and strictly positive)."
                msr_sub = "g/g (MW unavailable)"

        msr_display = f"{msr_value:.4f}" if msr_value is not None else "N/A"
        m_active = "msr-pill-active" if unit_mode == "M" else "msr-pill-inactive"
        w_active = "msr-pill-active" if unit_mode == "W" else "msr-pill-inactive"

        with st.container(border=True, key="_msr_metric_card"):
            st.markdown(
                f"""
<div class='metric-label'>
  {with_info(msr_label, msr_help)}
  <span class='msr-pill-group'>
    <span class='msr-pill {m_active}' id='_msr_pill_m'>M</span><span class='msr-pill {w_active}' id='_msr_pill_w'>W</span>
  </span>
</div>
<div class='metric-value'>{msr_display}</div>
<div class='metric-sub'>{msr_sub}</div>
                """,
                unsafe_allow_html=True,
            )
            # Hidden Streamlit buttons handle state; clicked via JS from pills above
            btn_c = st.columns(2)
            with btn_c[0]:
                if st.button("M", key="_msr_unit_m", type="secondary"):
                    st.session_state["_msr_card_unit"] = "M"
                    st.rerun()
            with btn_c[1]:
                if st.button("W", key="_msr_unit_w", type="secondary"):
                    st.session_state["_msr_card_unit"] = "W"
                    st.rerun()

    with cols[3]:
        st.markdown(
            f"""
<div class='metric-card'>
    <div class='metric-label'>{with_info('Applicability', 'Reliability flag based on domain checks; not a probability of correctness.')}</div>
    <div class='status-kpi-main {tone}'>{status_txt}</div>
    <div class='status-kpi-sub'>{n_pass}/3 criteria passed</div>
    <div class='status-kpi-track'>
        <div class='status-kpi-fill {tone}' style='width:{pass_pct:.0f}%;'></div>
    </div>
</div>
            """,
            unsafe_allow_html=True,
        )


def _tab_overview(inputs: dict, result, predictor, artifacts: Path) -> None:
    row1 = st.columns(2)
    with row1[0]:
        if result.shap_status == 'available':
            render_chart_card(
                "Local SHAP contributions",
                shap_bar(result, n_show=5, fixed_height=300),
                info="Per-feature effects on this estimate. Positive bars increase estimated log10(MSR), negative bars decrease it.",
            )
        else:
            st.info(f'SHAP explanation {result.shap_status}. {result.shap_reason}')
    with row1[1]:
        render_chart_card(
            "Training distribution vs estimate",
            distribution(predictor._y_cv_log, result.log10_msr),
            info="Shows where this estimate sits relative to training targets; far-tail positions may indicate extrapolation risk.",
        )

    row2 = st.columns([3.2, 2.8])
    with row2[0]:
        n_pass = int(result.ad.structural.pass_) + int(result.ad.y_range.pass_) + int(result.ad.local_error.pass_)
        pass_frac = n_pass / 3.0
        semi_len = np.pi * 88.0
        progress_len = semi_len * pass_frac
        gauge = f"""
    <svg width="260" height="150" viewBox="0 0 260 150" xmlns="http://www.w3.org/2000/svg">
      <path d="M42 120 A88 88 0 0 1 218 120" stroke="#d4d9df" stroke-width="15" stroke-linecap="round" fill="none"/>
      <path d="M42 120 A88 88 0 0 1 218 120" stroke="#62cc7a" stroke-width="15" stroke-linecap="round" stroke-dasharray="{progress_len:.2f} {semi_len:.2f}" fill="none"/>
      <text x="130" y="96" text-anchor="middle" font-size="30" font-family="Space Grotesk" fill="#2e3445">{n_pass}/3</text>
</svg>
        """
        rows = [
            ("Input-space support", result.ad.structural),
            ("Response range", result.ad.y_range),
            ("Local reliability", result.ad.local_error),
        ]
        rows_html = ""
        for label, criterion in rows:
            value, value_cls = _criterion_state(criterion)
            rows_html += f"<div class='ad-score-row'><span class='k'>{label}</span><span class='v {value_cls}'>{value}</span></div>"
        with st.container(border=True):
            st.markdown(
                f"""
    <div class="overview-dual-card">
            <div class="overview-dual-top">
                <div class='ad-score-title'>{with_info('Applicability score', 'Summary of 3 checks: structural similarity, estimated range, and local reliability.')}</div>
        <div class='overview-main-center'>{gauge}</div>
      </div>
            <div class="overview-footer-pad">
        {rows_html}
      </div>
    </div>
                """,
                unsafe_allow_html=True,
            )

    with row2[1]:
        with st.container(border=True):
            st.markdown(
                f"""
<div class="overview-dual-card">
    <div class="overview-dual-top">
            <div class="card-title card-title-row">{with_info('Local predictive error', result.ad.local_error.description)}</div>
    <div class="overview-main-center-col">
        <div class="card-value-xl">{result.ad.local_error.value_label}</div>
        <div class="card-meta">log10(MSR)</div>
    </div>
    </div>
    <div class="overview-footer-pad-strong">
        <div class="card-footnote">{html.escape(result.ad.local_error.description)}</div>
        <div class="card-footnote" style="margin-top:2px;">This indicator is not a prediction interval.</div>
    </div>
</div>
                """,
                unsafe_allow_html=True,
            )


def _tab_feature_analysis(result, artifacts: Path) -> None:
    if result.shap_status != 'available':
        st.info(f'SHAP explanation {result.shap_status}. {result.shap_reason}')
        return
    shap_sum = float(np.sum(result.shap_values))
    if np.isfinite(result.imposed_offset):
        st.caption(f'The displayed base includes an imposed solubility offset of {result.imposed_offset:+.3f}; '
                   'the offset is not a learned SHAP contribution.')
    row = st.columns([4, 1])
    with row[0]:
        render_chart_card(
            "Top local SHAP profile",
            shap_bar(result, n_show=15, fixed_height=500),
            info="Detailed SHAP decomposition for this query. SHAP values are associative, not causal.",
        )
    with row[1]:
        with st.container(border=True):
            st.markdown(
                f"""
<div style="padding:2px 0; min-height:520px;">
    <div class="card-title card-title-row" style="margin-bottom:10px;">How to read</div>
    <div class="shap-key">
        <div class="sk">Direction</div>
        <div class="sm">Teal → increases estimate<br>Coral → decreases estimate</div>
    </div>
    <div class="shap-key">
        <div class="sk">Base value</div>
        <div class="sv">{result.shap_base:+.3f}</div>
    </div>
    <div class="shap-key">
        <div class="sk">SHAP sum</div>
        <div class="sv">{shap_sum:+.3f}</div>
    </div>
    <div class="shap-key">
        <div class="sk">Estimate</div>
        <div class="sv">{result.log10_msr:+.3f}</div>
    </div>
</div>
                """,
                unsafe_allow_html=True,
            )

    fig = global_shap(artifacts)
    if fig is not None:
        render_chart_card(
            "Global SHAP importance",
            fig,
            info="Average absolute SHAP magnitude across the training set, indicating global model dependence by feature.",
        )

    st.markdown(
        "<div class='small' style='color:#9aa4b2; margin-top:4px;'>"
        "SHAP values describe model associations in this dataset. They do not establish causality."
        "</div>",
        unsafe_allow_html=True,
    )


def _tab_ad(result, predictor, artifacts: Path) -> None:
    cols = st.columns(3)
    for col, crit in zip(cols, [result.ad.structural, result.ad.y_range, result.ad.local_error]):
        status_label, tone = _criterion_state(crit)
        cls = f"crit-{tone}"
        with col:
            st.markdown(
                f"""
<div class="crit-card {cls}">
  <div class="crit-header">
    <div class="crit-dot"></div>
    <div class="crit-name">{html.escape(crit.name)}</div>
  </div>
  <div class="crit-row"><span class="ck">Status</span><span class="cv">{status_label}</span></div>
  <div class="crit-row"><span class="ck">Value</span><span class="cv">{crit.value_label}</span></div>
  <div class="crit-row"><span class="ck">Threshold</span><span class="cv">{crit.threshold_label}</span></div>
  <div class="crit-desc">{html.escape(crit.description)}</div>
</div>
                """,
                unsafe_allow_html=True,
            )

    label, status_cls, n_pass = status(result.ad)
    vcls = {"ok": "verdict-pass", "mid": "verdict-mid", "bad": "verdict-fail"}[_STATUS_TONE[status_cls]]
    st.markdown(
        f"""
<div class="verdict-bar {vcls}">
  <div class="verdict-dot"></div>
  <span>{label}</span>
  <span style="font-weight:400; opacity:0.75;">·  {n_pass} / 3 criteria met</span>
</div>
        """,
        unsafe_allow_html=True,
    )

    st.caption('Support thresholds are training-derived. Local error availability is reported separately; no calibrated prediction interval is supplied.')


def _tab_descriptor_details(result, predictor, artifacts: Path) -> None:
    st.markdown(
        f"<div class='card-title card-title-row'>{with_info('Descriptor deviation table', 'Compares each descriptor to its training distribution. Extreme percentiles suggest out-of-domain behavior.')}</div>",
        unsafe_allow_html=True,
    )
    st.markdown(
        "<div class='small' style='margin-bottom:10px;'>Each descriptor value compared to the training distribution. "
        "Percentile outside 10–90 is flagged as outlying.</div>",
        unsafe_allow_html=True,
    )
    df = descriptor_table(result, predictor)
    shown = df.copy()
    for col in ["Query Value", "Training Mean", "Percentile", "Deviation"]:
        shown[col] = shown[col].map(lambda x: round(float(x), 4))

    st.dataframe(
        shown,
        width="stretch",
        hide_index=True,
        column_config={
            "Percentile": st.column_config.ProgressColumn(
                "Percentile", min_value=0, max_value=100, format="%.1f"
            ),
            "Status": st.column_config.TextColumn("Status"),
        },
    )
    st.download_button(
        "Export CSV",
        data=df.to_csv(index=False).encode("utf-8"),
        file_name="descriptor_deviation_table.csv",
        mime="text/csv",
        width="content",
    )

    st.markdown("<div style='height:10px'></div>", unsafe_allow_html=True)
    st.markdown(
        f"<div class='card-title card-title-row'>{with_info('Nearest training neighbors', 'Closest known training systems in feature space for this query.')}</div>",
        unsafe_allow_html=True,
    )
    neigh = predictor.get_neighbors(result.ad)
    st.dataframe(neigh.head(7), width="stretch", hide_index=True)


def _tab_model_evidence(result, predictor, artifacts: Path) -> None:
    st.caption('Completed reports are linked by the selected study and verified file receipts. Development evidence and reserved-holdout evidence are separate.')
    reports = [('Grouped validation', 'repetition_metrics.csv'),
               ('Applicability coverage and error', 'coverage_error.csv'),
               ('Complete-procedure target randomization', 'permutation_metrics.csv'),
               ('External-fold feature stability', 'selection_stability.csv')]
    for title, filename in reports:
        table = report_table(artifacts, filename)
        if table is not None:
            st.markdown(f'**{title}**')
            st.dataframe(table, width="stretch", hide_index=True)
        else:
            st.caption(f'{title}: report not available yet.')
    holdout = report_table(artifacts, 'holdout_predictions.csv')
    if holdout is None:
        st.caption('Reserved holdout: no separately completed evaluation is available.')
    else:
        st.markdown('**Reserved holdout predictions (separate evaluation)**')
        st.dataframe(holdout, width="stretch", hide_index=True)
    st.caption('Dependent repetitions are descriptive; target shuffling is not an exact significance test. Model associations are not causal.')


def render_prediction_workspace(inputs: dict, result, predictor, artifacts: Path) -> None:
    _render_header_kpis(inputs, result, artifacts)
    if result.abraham_descriptor_status != 'not_required':
        st.caption('Abraham descriptor status: ' + result.abraham_descriptor_status.replace('_', ' '))
    if result.imputed_reference_properties:
        st.warning('Reference input(s) imputed from training data: ' +
                   ', '.join(result.imputed_reference_properties) +
                   '. These values were not supplied as experimental measurements; the AD cannot be fully inside.')
    if result.ad.y_range.availability_status == 'outside':
        st.warning('Predicted log10(MSR) is outside the range of training responses. This is extrapolation, not a prediction interval.')

    tabs = st.tabs(["Overview", "Interpretability", "Applicability", "Local Diagnostics", "Model Validation"])
    with tabs[0]:
        _tab_overview(inputs, result, predictor, artifacts)
    with tabs[1]:
        _tab_feature_analysis(result, artifacts)
    with tabs[2]:
        _tab_ad(result, predictor, artifacts)
    with tabs[3]:
        _tab_descriptor_details(result, predictor, artifacts)
    with tabs[4]:
        _tab_model_evidence(result, predictor, artifacts)
