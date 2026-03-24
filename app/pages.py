from __future__ import annotations

import base64
import html
import json
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st

from app.charts import (
    shap_bar,
    distribution,
    parity,
    global_shap,
    feature_stability,
    ad_quantile,
    ad_dist_error,
    y_randomization,
    plotly_layout,
)
from app.components import (
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
    'Synperonic 13/8': 690.0,
    'Igepal CA-720':   735.0,
    'Igepal CO-720':   749.0,
    'Tergitol NP-10':  603.0,
}

def get_surfactant_mw(surf_name, mw_computed):
    return SURFACTANT_MW.get(surf_name, mw_computed)

"""Page renderers: welcome screen, left panel, prediction workspace and tabs."""


RMSE_SIGMA = 0.2256


def _set_example(cont_name: str, cont_smiles: str, surf_name: str, surf_smiles: str) -> None:
    st.session_state["pending_example"] = {
        "cont_name": cont_name,
        "cont_smiles": cont_smiles,
        "surf_name": surf_name,
        "surf_smiles": surf_smiles,
    }


def _predict_from_state(predictor, load_compound_lookup) -> tuple[dict | None, object | None]:
    cont_lookup, surf_lookup = load_compound_lookup()

    cont_raw = st.session_state.get("_cont_search", "").strip()
    surf_raw = st.session_state.get("_surf_search", "").strip()

    cont_key = cont_raw.lower()
    surf_key = surf_raw.lower()
    cont_smiles = cont_lookup.get(cont_key, cont_raw)
    surf_smiles = surf_lookup.get(surf_key, surf_raw)
    cont_name = st.session_state.pop("_example_cont_label", None) or (cont_raw if cont_key in cont_lookup else "Contaminant")
    surf_name = st.session_state.pop("_example_surf_label", None) or (surf_raw if surf_key in surf_lookup else "Surfactant")

    if not cont_smiles:
        st.error("Contaminant not found in database. Please enter a valid SMILES string for the contaminant.")
        return None, None
    if not surf_smiles:
        st.error("Surfactant not found in database. Please enter a valid SMILES string for the surfactant.")
        return None, None

    ok_c, msg_c = predictor.validate_smiles(cont_smiles)
    ok_s, msg_s = predictor.validate_smiles(surf_smiles)
    if not ok_c:
        st.error(f"Contaminant input is not a valid SMILES string. Error: {msg_c}\nIf you intended to use a common name, please select one from the dropdown or check spelling.")
        return None, None
    if not ok_s:
        st.error(f"Surfactant input is not a valid SMILES string. Error: {msg_s}\nIf you intended to use a common name, please select one from the dropdown or check spelling.")
        return None, None

    with st.spinner("Computing descriptors and estimation..."):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            result = predictor.predict(cont_smiles=cont_smiles, surf_smiles=surf_smiles)

    inputs = {
        "cont_smiles": cont_smiles,
        "surf_smiles": surf_smiles,
        "cont_label": cont_name,
        "surf_label": surf_name,
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

        run_now = st.session_state.pop("_run_example", False)
        if st.button("Estimate MSR", type="primary", use_container_width=True) or run_now:
            inputs, result = _predict_from_state(predictor, load_compound_lookup)
            if result is not None:
                st.session_state["last_inputs"] = inputs
                st.session_state["last_result"] = result

        st.markdown("<div style='height:6px'></div>", unsafe_allow_html=True)
        st.markdown(
            "<div class='warn-box'><b>Research prototype.</b> Trained on n=142 literature pairs. "
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

            # Resolve display name if still a generic placeholder
            if title in {"Contaminant", "Surfactant"}:
                looked_up = pubchem_name(smi)
                if looked_up:
                    title = looked_up[:60] + "…" if len(looked_up) > 60 else looked_up
                else:
                    title = truncate_smiles(smi, n=28)

            safe_title = html.escape(str(title))
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
             data-title={json.dumps(title)}
             alt={json.dumps(title)} />
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

def render_welcome(artifacts: Path, get_welcome_examples) -> None:
    hero_cols = st.columns([3.2, 1.25])
    with hero_cols[0]:
        st.markdown(
            """
<div class='welcome-hero'>
    <div class='welcome-title'>MSR Estimator</div>
    <div class='welcome-sub'>QSPR-based estimation tool for contaminant–surfactant molar solubilization ratios</div>
    <div>
        <span class='badge badge-teal'>142 pairs</span>
        <span class='badge badge-teal'>30 descriptors</span>
        <span class='badge badge-teal'>Nested CV</span>
    </div>
</div>
            """,
            unsafe_allow_html=True,
        )
    with hero_cols[1]:
        st.markdown(
            """
<div class='hero-mini-stack'>
    <div class='hero-mini-kpi'>
        <div class='k'>Model</div>
        <div class='v'>XGBoost</div>
    </div>
    <div class='hero-mini-kpi'>
        <div class='k'>Holdout RMSE</div>
        <div class='v'>0.2256</div>
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
                    clicked = st.button("Try this", key=f"try_{idx}", use_container_width=True, type="secondary")
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

def _render_header_kpis(inputs: dict, result) -> None:
    status_txt, status_cls, n_pass = status(result.ad)
    pass_pct = (n_pass / 3.0) * 100.0
    tone = "ok" if n_pass == 3 else ("mid" if n_pass == 2 else "bad")

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
    <div class='metric-sub'>+/- {RMSE_SIGMA:.3f} (1 sigma)</div>
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
            if mw_cont is not None and mw_surf_final is not None and mw_surf_final > 0:
                msr_value = msr_value * (mw_cont / mw_surf_final)
                msr_label = "WSR"
                msr_help = "Weight solubilization ratio in g/g, converted from MSR using molecular weights: WSR = MSR * (MW_contaminant / MW_surfactant)."
                msr_sub = "g/g"
            else:
                msr_label = "WSR"
                msr_help = "WSR (g/g) conversion requested, but molecular weights were unavailable."
                msr_sub = "g/g (MW unavailable)"

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
<div class='metric-value'>{msr_value:.4f}</div>
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
        render_chart_card(
            "Local SHAP contributions",
            shap_bar(result, n_show=5, fixed_height=300),
            info="Per-feature effects on this estimate. Positive bars increase estimated log10(MSR), negative bars decrease it.",
        )
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
            ("Structural similarity", result.ad.structural.pass_),
            ("Estimated range", result.ad.y_range.pass_),
            ("Local reliability", result.ad.local_error.pass_),
        ]
        rows_html = ""
        for label, passed in rows:
            value = "PASS" if passed else "FAIL"
            value_cls = "pass" if passed else "fail"
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
        lo = 10 ** (result.log10_msr - RMSE_SIGMA)
        hi = 10 ** (result.log10_msr + RMSE_SIGMA)
        with st.container(border=True):
            st.markdown(
                f"""
<div class="overview-dual-card">
    <div class="overview-dual-top">
            <div class="card-title card-title-row">{with_info('Uncertainty interval', 'Computed from holdout RMSE (1 sigma). This is not a calibrated confidence interval.')}</div>
    <div class="overview-main-center-col">
        <div class="card-value-xl">{lo:.4f} – {hi:.4f}</div>
        <div class="card-meta">mol / mol &nbsp;·&nbsp; 1σ interval</div>
    </div>
    </div>
    <div class="overview-footer-pad-strong">
        <div class="card-footnote">Based on holdout RMSE = {RMSE_SIGMA:.4f}</div>
        <div class="card-footnote" style="margin-top:2px;">Not a calibrated confidence interval.</div>
    </div>
</div>
                """,
                unsafe_allow_html=True,
            )


def _tab_feature_analysis(result, artifacts: Path) -> None:
    shap_sum = float(np.sum(result.shap_values))
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
        cls = "crit-pass" if crit.pass_ else "crit-fail"
        status_label = "PASS" if crit.pass_ else "FAIL"
        with col:
            st.markdown(
                f"""
<div class="crit-card {cls}">
  <div class="crit-header">
    <div class="crit-dot"></div>
    <div class="crit-name">{crit.name}</div>
  </div>
  <div class="crit-row"><span class="ck">Status</span><span class="cv">{status_label}</span></div>
  <div class="crit-row"><span class="ck">Value</span><span class="cv">{crit.value_label}</span></div>
  <div class="crit-row"><span class="ck">Threshold</span><span class="cv">{crit.threshold_label}</span></div>
  <div class="crit-desc">{crit.description}</div>
</div>
                """,
                unsafe_allow_html=True,
            )

    label, _, n_pass = status(result.ad)
    vcls = "verdict-pass" if n_pass == 3 else ("verdict-mid" if n_pass == 2 else "verdict-fail")
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

    row = st.columns(2)
    with row[0]:
        q = ad_quantile(artifacts)
        if q is not None:
            render_chart_card(
                "AD quantile scan",
                q,
                info="Tradeoff view between in-domain coverage and in-domain R2 as structural similarity threshold changes.",
            )
    with row[1]:
        de = ad_dist_error(result, artifacts)
        if de is not None:
            render_chart_card(
                "Distance-error map",
                de,
                info="Links structural distance to local cross-validated error; the star marks the current sample.",
            )


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
        use_container_width=True,
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
        use_container_width=False,
    )

    st.markdown("<div style='height:10px'></div>", unsafe_allow_html=True)
    st.markdown(
        f"<div class='card-title card-title-row'>{with_info('Nearest training neighbors', 'Closest known training systems in feature space for this query.')}</div>",
        unsafe_allow_html=True,
    )
    neigh = predictor.get_neighbors(result.ad)
    st.dataframe(neigh.head(7), use_container_width=True, hide_index=True)


def _tab_model_evidence(result, predictor, artifacts: Path) -> None:
    from app.charts import _read_csv

    metrics = _read_csv(artifacts / "final_model/metrics/xgboost_cv_holdout_metrics.csv")
    if metrics is not None and len(metrics):
        row_m = metrics.iloc[0]
        vals = [
            ("R² CV", row_m.get("R2_train_cv", np.nan)),
            ("RMSE CV", row_m.get("RMSE_train_cv", np.nan)),
            ("R² Holdout", row_m.get("R2_holdout", np.nan)),
            ("RMSE Holdout", row_m.get("RMSE_holdout", np.nan)),
        ]
        # Render four equal cards in a controlled flex row to tighten spacing.
        cards = []
        for name, v in vals:
            cards.append(
                f"""
<div class='metric-card' style='flex:1; margin:0;'>
    <div class='metric-label'>{name}</div>
    <div class='metric-value'>{float(v):.4f}</div>
</div>
                """
            )
        # Wrap with a page-specific class so CSS can target spacing precisely
        row_html = "<div class='metric-row model-validation-metrics' style='display:flex; gap:8px; align-items:stretch;'>" + "".join(cards) + "</div>"
        st.markdown(row_html, unsafe_allow_html=True)

    st.markdown("<div style='height:10px'></div>", unsafe_allow_html=True)

    row2 = st.columns([2.8, 2.2])
    with row2[0]:
        fig = parity(artifacts)
        if fig is not None:
            render_chart_card(
                "Parity plot",
                fig,
                info="Estimated vs observed values. Points near the diagonal indicate better agreement.",
            )

    with row2[1]:
        fold = _read_csv(artifacts / "model_selection/metrics/nested_outer_fold_metrics.csv")
        if fold is not None and {"Model", "R2_outer", "RMSE_outer"}.issubset(fold.columns):
            table = (
                fold.groupby("Model", as_index=False)
                .agg(R2_mean=("R2_outer", "mean"), R2_std=("R2_outer", "std"), RMSE_mean=("RMSE_outer", "mean"), RMSE_std=("RMSE_outer", "std"))
                .sort_values("R2_mean", ascending=False)
            )
            table["R² (mean ± sd)"] = table.apply(lambda r: f"{r['R2_mean']:.3f} ± {r['R2_std']:.3f}", axis=1)
            table["RMSE (mean ± sd)"] = table.apply(lambda r: f"{r['RMSE_mean']:.3f} ± {r['RMSE_std']:.3f}", axis=1)
            with st.container(border=True):
                st.markdown(
                    f"<div class='card-title card-title-row'>{with_info('Nested CV model comparison', 'Performance summarized over outer folds. Mean ± sd reflects both accuracy and stability.')}</div>",
                    unsafe_allow_html=True,
                )
                st.dataframe(table[["Model", "R² (mean ± sd)", "RMSE (mean ± sd)"]], use_container_width=True, hide_index=True, height=420)

    fig_y, real, n_perm = y_randomization(artifacts)
    if fig_y is not None:
        row3 = st.columns([3, 2])
        with row3[0]:
            render_chart_card(
                "Y-randomization",
                fig_y,
                info="Chance-correlation check using permuted targets. Real-model performance should exceed permuted baseline.",
            )
        with row3[1]:
            real_str = f"{real:.4f}" if real is not None else "\u2014"
            with st.container(border=True):
                st.markdown(
                    f"""
<div style="padding:2px 0; min-height:346px; display:flex; flex-direction:column; justify-content:flex-start;">
        <div class="card-title card-title-row" style="margin-bottom:10px;">{with_info('Y-randomization summary', 'If permuted-target scores remain poor while real-target score is strong, chance-correlation risk is lower.')}</div>
    <div class="shap-key">
        <div class="sk">Permuted runs</div>
        <div class="sv">{n_perm}</div>
    </div>
    <div class="shap-key">
        <div class="sk">Real model R²</div>
        <div class="sv">{real_str}</div>
    </div>
    <div class="shap-key">
        <div class="sk">Interpretation</div>
        <div class="sm">All permuted R² negative → suggests low chance-correlation risk</div>
    </div>
</div>
                    """,
                    unsafe_allow_html=True,
                )


def render_prediction_workspace(inputs: dict, result, predictor, artifacts: Path) -> None:
    _render_header_kpis(inputs, result)

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