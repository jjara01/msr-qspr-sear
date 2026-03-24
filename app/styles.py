"""CSS and JavaScript injection for the MSR Estimator Streamlit app."""

from __future__ import annotations

import streamlit as st
import streamlit.components.v1 as st_components


def inject_css() -> None:
    st.markdown(
        """
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&family=Space+Grotesk:wght@500;600;700&display=swap');

:root {
  --bg: #f3f5f7;
  --card: #ffffff;
  --line: #e6eaef;
    --line-strong: #d8dee6;
  --text: #2e3445;
  --muted: #7f8798;
  --teal: #0c8a7f;
    --teal-dark: #0a6f66;
  --teal-soft: #e4f5f2;
  --warn-soft: #fff4df;
  --warn-line: #ffd79c;
        --card-pad-y: 12px;
        --card-pad-x: 14px;
        --card-gap: 10px;
        --sel-mol-wrap-h: 100px;
        --sel-mol-h: 80px;
        --sel-mol-max-w: 270px;
        --sel-mol-pad: 6px;
    --space-1: 6px;
    --space-2: 10px;
    --space-3: 14px;
}

[data-testid="stSidebar"] { display: none; }
[data-testid="stMainBlockContainer"],
[data-testid="stAppViewBlockContainer"],
.block-container {
    max-width: 1460px;
    padding-top: 0 !important;
    padding-bottom: 2rem;
}
[data-testid="stAppViewContainer"] {
  background:
    radial-gradient(900px 420px at 102% -8%, rgba(12, 138, 127, 0.08), rgba(12,138,127,0)),
    radial-gradient(720px 350px at -10% 8%, rgba(63, 83, 173, 0.06), rgba(63,83,173,0)),
    var(--bg);
}

[data-testid="stAppViewContainer"] > .main,
[data-testid="stAppViewContainer"] .main,
section.main {
    padding-top: 0 !important;
    margin-top: 0 !important;
}
[data-testid="stHeader"] {
    background: transparent;
    height: 0;
    min-height: 0;
    display: none;
}

[data-testid="stToolbar"] {
    display: none;
}

html, body, [class*="css"] {
  font-family: 'Inter', sans-serif;
  color: var(--text);
}
h1, h2, h3, h4 {
  font-family: 'Space Grotesk', sans-serif;
  color: var(--text);
}

.topbar {
    background: rgba(255,255,255,0.82);
    border: 1px solid var(--line-strong);
  border-radius: 12px;
    padding: 12px 18px;
        margin-top: -35px;
    margin-bottom: 14px;
  backdrop-filter: blur(4px);
    box-shadow: 0 1px 4px rgba(32, 47, 69, 0.06);
}

.card {
  background: var(--card);
    border: 1px solid var(--line-strong);
  border-radius: 12px;
  padding: 14px 16px;
    box-shadow: 0 1px 3px rgba(32, 47, 69, 0.05);
}

.metric-card {
  background: var(--card);
    border: 1px solid var(--line-strong);
  border-radius: 12px;
  padding: 12px 14px;
    min-height: 132px;
    height: 132px;
    box-shadow: 0 1px 3px rgba(32, 47, 69, 0.05);
    display: flex;
    flex-direction: column;
    justify-content: space-between;
}

/* Tighter spacing specifically for Model Validation metric row */
.model-validation-metrics {
        display: flex !important;
        gap: 12px !important;
        align-items: stretch !important;
        margin-left: 0px; /* counter any container padding */
        margin-right: 0px;
}
.model-validation-metrics .metric-card {
    margin: 0 !important;
    gap: 6px !important;
    padding: 10px 10px !important;
    min-height: 64px !important;
    height: auto !important;
    box-shadow: 0 1px 2px rgba(32, 47, 69, 0.04) !important;
}
.model-validation-metrics .metric-card { border-radius:8px !important; }
.metric-label {
  font-size: 0.72rem;
  letter-spacing: 0.06em;
  text-transform: uppercase;
  color: var(--muted);
  font-weight: 700;
}
.metric-value {
  font-family: 'Space Grotesk', sans-serif;
  font-size: 2.0rem;
  font-weight: 700;
  color: var(--text);
  line-height: 1.05;
  margin-top: 5px;
}
.metric-sub {
  font-size: 0.80rem;
  color: var(--muted);
}

/* Tiny inline M/W buttons inside MSR card header */
.st-key-_msr_metric_card {
    min-height: 132px;
    padding-right: 10px !important;
}
.st-key-_msr_metric_card [data-testid="stVerticalBlock"] {
    row-gap: 0.12rem;
}

/* Hide the Streamlit M/W buttons — state only, clicked via JS from HTML pills */
.st-key-_msr_unit_m,
.st-key-_msr_unit_w {
    position: absolute !important;
    width: 0 !important;
    height: 0 !important;
    overflow: hidden !important;
    opacity: 0 !important;
    pointer-events: none !important;
}

/* HTML pill toggle inline with .metric-label */
.msr-pill-group {
    display: inline-flex;
    align-items: center;
    gap: 1px;
    margin-left: 45px;
    background: #edf2f7;
    border-radius: 999px;
    padding: 1px;
    vertical-align: text-bottom;
    line-height: 1;
}
.msr-pill {
    display: inline-flex;
    align-items: center;
    justify-content: center;
    min-width: 20px;
    height: 15px;
    padding: 0 5px;
    font-size: 0.58rem;
    font-weight: 700;
    border-radius: 999px;
    cursor: pointer;
    user-select: none;
    transition: background 0.15s, color 0.15s;
}
.msr-pill-active {
    background: linear-gradient(180deg, var(--teal) 0%, var(--teal-dark) 100%);
    color: #fff;
}
.msr-pill-inactive {
    background: transparent;
    color: var(--muted);
}
.msr-pill-inactive:hover {
    background: rgba(12, 138, 127, 0.12);
    color: var(--teal-dark);
}

.status-kpi-main {
    font-family: 'Space Grotesk', sans-serif;
    font-size: 1.10rem;
    font-weight: 700;
    line-height: 1.1;
    margin-top: 3px;
}

.status-kpi-main.ok { color: #187d43; }
.status-kpi-main.mid { color: #9c5a06; }
.status-kpi-main.bad { color: #9a1e1e; }

.status-kpi-sub {
    font-size: 0.82rem;
    color: #6f7890;
}

.status-kpi-track {
    height: 7px;
    width: 100%;
    border-radius: 999px;
    background: #edf2f7;
    border: 1px solid #dde4ee;
    overflow: hidden;
}

.status-kpi-fill {
    height: 100%;
    border-radius: 999px;
}

.status-kpi-fill.ok { background: #7ad19a; }
.status-kpi-fill.mid { background: #f3c56f; }
.status-kpi-fill.bad { background: #ec9999; }

.card-title {
    font-family: 'Space Grotesk', sans-serif;
    font-size: 1.02rem;
    font-weight: 700;
    color: #2b3f56;
    line-height: 1.25;
    margin-bottom: 8px;
}

.card-note {
    color: #8a93a5;
    font-size: 0.78rem;
    margin-top: 4px;
}

.info-dot {
    display: inline-flex;
    align-items: center;
    justify-content: center;
    width: 17px;
    height: 17px;
    margin-left: 6px;
    border-radius: 50%;
    border: 1px solid #c7d4e2;
    background: linear-gradient(180deg, #f8fbff 0%, #eef4fb 100%);
    color: #5f7288;
    font-size: 0.68rem;
    font-weight: 700;
    cursor: default;
    vertical-align: text-top;
    user-select: none;
    position: relative;
    line-height: 1;
}

.info-dot::after {
    content: attr(data-tip);
    position: absolute;
    left: 50%;
    bottom: calc(100% + 10px);
    transform: translateX(-50%) translateY(4px);
    width: min(320px, 70vw);
    padding: 8px 10px;
    border-radius: 10px;
    border: 1px solid #d5dde8;
    background: #ffffff;
    color: #4b5a6f;
    font-size: 0.76rem;
    font-weight: 500;
    line-height: 1.35;
    text-transform: none;
    letter-spacing: normal;
    box-shadow: 0 10px 26px rgba(24, 38, 56, 0.14);
    opacity: 0;
    pointer-events: none;
    transition: opacity 120ms ease, transform 120ms ease;
    z-index: 30;
    white-space: normal;
}

.info-dot::before {
    content: "";
    position: absolute;
    left: 50%;
    bottom: calc(100% + 4px);
    transform: translateX(-50%) translateY(4px);
    border-width: 6px 6px 0 6px;
    border-style: solid;
    border-color: #d5dde8 transparent transparent transparent;
    opacity: 0;
    transition: opacity 120ms ease, transform 120ms ease;
    z-index: 31;
}

.info-dot:hover::after,
.info-dot:hover::before {
    opacity: 1;
    transform: translateX(-50%) translateY(0);
}

.card-title-row {
    border-bottom: 1px solid #e8edf3;
    padding-bottom: 7px;
    margin-bottom: var(--card-gap);
}

.card-value-xl {
    font-family: 'Space Grotesk', sans-serif;
    font-size: 1.48rem;
    font-weight: 700;
    color: var(--text);
    line-height: 1.15;
    margin-bottom: 4px;
}

.card-meta {
    color: #7d8698;
    font-size: 0.82rem;
}

.card-footnote {
    color: #9aa4b2;
    font-size: 0.78rem;
    margin-top: 8px;
}

.overview-main-center {
    min-height: 150px;
    height: auto;
    padding: 2px 0;
    display: flex;
    align-items: center;
    justify-content: center;
    flex: 1;
}

.overview-main-center > * {
    margin: 0 !important;
}

.overview-main-center svg {
    display: block;
    margin: 0 auto;
}

.overview-main-center-col {
    min-height: 150px;
    display: flex;
    flex-direction: column;
    align-items: center;
    justify-content: center;
    text-align: center;
    flex: 1;
}

.pair-title {
    font-family: 'Space Grotesk', sans-serif;
    font-size: 1.78rem;
    font-weight: 700;
    color: #2e3445;
    line-height: 1.08;
    min-height: 2.2em;
    overflow: hidden;
    display: -webkit-box;
    -webkit-line-clamp: 2;
    -webkit-box-orient: vertical;
}

.overview-dual-card {
    min-height: 360px;
    display: flex;
    flex-direction: column;
}

.overview-dual-top {
    display: flex;
    flex-direction: column;
    flex: 1;
}

.app-footer {
    margin-top: 32px;
    text-align: center;
    font-size: 0.72rem;
    color: var(--muted);
    padding-bottom: 12px;
}

.overview-footer-pad {
    padding-bottom: 8px;
}

.overview-footer-pad-strong {
    padding-bottom: 14px;
}

.selected-system-card {
    padding-top: 6px;
}

.selected-system-role {
    font-size: 0.70rem;
    letter-spacing: 0.08em;
    text-transform: uppercase;
    color: #7d8698;
    font-weight: 700;
    margin-bottom: 4px;
}

.selected-system-name {
    font-family: 'Space Grotesk', sans-serif;
    font-size: 1.00rem;
    font-weight: 700;
    color: #2b3f56;
    line-height: 1.2;
    margin-bottom: 4px;
}

.selected-system-metrics {
    display: flex;
    gap: 6px;
    flex-wrap: wrap;
    margin-bottom: 10px;
}

.ad-score-title {
    font-family: 'Space Grotesk', sans-serif;
    font-size: 1.02rem;
    font-weight: 700;
    color: #2b3f56;
    line-height: 1.25;
    border-bottom: 1px solid #e8edf3;
    padding-bottom: 7px;
    margin-bottom: 8px;
}

.ad-score-row {
    display: flex;
    justify-content: space-between;
    align-items: center;
    border-top: 1px solid #e9edf2;
    padding: 7px 0;
    font-size: 0.82rem;
}

.ad-score-row .k {
    color: #616b7f;
}

.ad-score-row .v {
    font-weight: 700;
    letter-spacing: 0.02em;
}

.ad-score-row .v.pass { color: #187d43; }
.ad-score-row .v.fail { color: #9a1e1e; }

.badge {
  display: inline-block;
  border-radius: 999px;
  padding: 4px 10px;
  font-size: 0.76rem;
  font-weight: 700;
  margin-right: 6px;
  margin-bottom: 6px;
}
.badge-teal { background: var(--teal-soft); color: var(--teal); }
.badge-warn { background: #fff0e6; color: #b25a00; }

.ad-ok { background: #e7f8ef; color: #187d43; border: 1px solid #a8dfbd; }
.ad-mid { background: #fff6e8; color: #9c5a06; border: 1px solid #ffd79c; }
.ad-bad { background: #ffecec; color: #9a1e1e; border: 1px solid #f0abab; }

.warn-box {
  background: var(--warn-soft);
  border: 1px solid var(--warn-line);
  border-radius: 10px;
  padding: 9px 10px;
  color: #7d5620;
  font-size: 0.78rem;
}

.small {
  color: var(--muted);
  font-size: 0.80rem;
}

.label-overline {
    color: var(--muted);
    font-size: 0.70rem;
    letter-spacing: 0.07em;
    text-transform: uppercase;
    font-weight: 700;
    margin-bottom: 4px;
}

.toolbar-chip {
    border: 1px solid var(--line-strong);
    background: #f7f9fb;
    border-radius: 999px;
    padding: 5px 10px;
    color: #6e7587;
    font-size: 0.72rem;
}

.welcome-hero {
    background:
        linear-gradient(145deg, #ffffff 0%, #f7fbfa 46%, #f4f8ff 100%);
    border: 1px solid var(--line-strong);
    border-radius: 14px;
    padding: 16px 18px 12px 18px;
    box-shadow: 0 1px 4px rgba(32, 47, 69, 0.06);
    min-height: 150px;
    display: flex;
    flex-direction: column;
    justify-content: space-between;
}

.welcome-title {
    font-family: 'Space Grotesk', sans-serif;
    font-size: 2.05rem;
    font-weight: 700;
    color: #263850;
    letter-spacing: 0.01em;
    margin-bottom: 2px;
}

.welcome-sub {
    color: #6f7990;
    font-size: 0.96rem;
    margin-bottom: 8px;
}

.hero-mini-kpi {
    border: 1px solid #dfe7f0;
    border-radius: 12px;
    background: rgba(255,255,255,0.85);
    padding: 10px 12px;
    margin-top: 0;
}

.hero-mini-stack {
    min-height: 150px;
    display: flex;
    flex-direction: column;
    justify-content: space-between;
}

.hero-mini-kpi .k {
    font-size: 0.68rem;
    letter-spacing: 0.08em;
    text-transform: uppercase;
    color: #7f8798;
    font-weight: 700;
}

.hero-mini-kpi .v {
    font-family: 'Space Grotesk', sans-serif;
    font-size: 1.20rem;
    font-weight: 700;
    color: #24445f;
}

.example-card {
    background: #ffffff;
    border: 1px solid var(--line-strong);
    border-radius: 12px;
    padding: 12px 13px;
    box-shadow: 0 1px 3px rgba(32, 47, 69, 0.04);
    min-height: 198px;
}

.example-title {
  font-family: 'Space Grotesk', sans-serif;
  font-size: 1.05rem;
  font-weight: 700;
  color: #28354a;
  line-height: 1.2;
  height: 2.55em;
  overflow: hidden;
    margin-bottom: var(--space-1);
}

.example-smiles {
  font-family: ui-monospace, SFMono-Regular, Menlo, monospace;
  font-size: 0.73rem;
  color: #677188;
  line-height: 1.25;
  height: 2.35em;
  overflow: hidden;
    margin-bottom: var(--space-1);
  white-space: pre-wrap;
}

.example-label {
    font-size: 0.72rem;
    letter-spacing: 0.06em;
    text-transform: uppercase;
    color: #7b8395;
    font-weight: 700;
    margin-bottom: 4px;
}

.example-chem-name {
    font-family: 'Space Grotesk', sans-serif;
    font-size: 0.93rem;
    font-weight: 600;
    color: #2b3f56;
    min-height: 2.6em;
    line-height: 1.25;
}

.example-mol-wrap {
    border: 1px solid #e3e9f1;
    border-radius: 10px;
    background: linear-gradient(180deg, rgba(252, 254, 255, 0.86) 0%, rgba(247, 250, 253, 0.72) 100%);
    min-height: 126px;
    display: flex;
    align-items: center;
    justify-content: center;
    padding: 6px;
}

.example-thumb-link {
    display: block;
    width: fit-content;
    margin: 0 auto;
    border-radius: 10px;
}

.example-thumb-link img {
    display: block;
    width: 210px;
    height: auto;
    cursor: zoom-in;
    transition: transform 120ms ease, filter 120ms ease;
}

.example-thumb-link:hover img {
    transform: translateY(-1px);
    filter: saturate(1.06);
}

.selected-system-mol-wrap {
    border: 1px solid #e3e9f1;
    border-radius: 10px;
    background: linear-gradient(180deg, rgba(252, 254, 255, 0.9) 0%, rgba(246, 250, 253, 0.78) 100%);
    min-height: var(--sel-mol-wrap-h);
    display: flex;
    align-items: center;
    justify-content: center;
    padding: var(--sel-mol-pad);
    margin: 8px 0 10px 0;
}

.selected-system-mol {
    width: auto;
    max-width: min(100%, var(--sel-mol-max-w));
    height: var(--sel-mol-h);
    object-fit: contain;
}

.mol-modal {
    position: fixed;
    inset: 0;
    z-index: 1000;
    opacity: 0;
    pointer-events: none;
    transition: opacity 160ms ease;
    display: flex;
    align-items: center;
    justify-content: center;
}

.mol-modal:target {
    opacity: 1;
    pointer-events: auto;
}

.mol-modal-backdrop {
    position: absolute;
    inset: 0;
    background: rgba(20, 29, 42, 0.40);
    backdrop-filter: blur(2px);
}

.mol-modal-card {
    position: relative;
    width: min(940px, 94vw);
    max-height: 88vh;
    overflow: auto;
    background: #ffffff;
    border: 1px solid var(--line-strong);
    border-radius: 14px;
    box-shadow: 0 24px 64px rgba(20, 29, 42, 0.22);
    transform: translateY(10px) scale(0.985);
    transition: transform 180ms ease;
}

.mol-modal:target .mol-modal-card {
    transform: translateY(0) scale(1);
}

.mol-modal-head {
    display: flex;
    align-items: center;
    justify-content: space-between;
    border-bottom: 1px solid var(--line);
    padding: 12px 14px;
}

.mol-modal-title {
    font-family: 'Space Grotesk', sans-serif;
    font-size: 1.08rem;
    font-weight: 700;
    color: #28354a;
}

.mol-modal-close {
    text-decoration: none;
    color: #5f6678;
    border: 1px solid var(--line-strong);
    border-radius: 9px;
    padding: 4px 10px;
    background: #f7fafc;
    font-size: 0.86rem;
}

.mol-modal-body {
    padding: 10px 12px 14px 12px;
}

.mol-modal-body img {
    width: 100%;
    height: auto;
    display: block;
}

.mol-tab-input {
    display: none;
}

.mol-tab-labels {
    display: flex;
    gap: 8px;
    margin-bottom: 10px;
}

.mol-tab-label {
    border: 1px solid #d8dee6;
    border-radius: 9px;
    padding: 4px 12px;
    font-size: 0.84rem;
    color: #606a7e;
    background: #ffffff;
    cursor: pointer;
    user-select: none;
}

.mol-tab-panel {
    display: none;
}

.mol-tab-2d:checked ~ .mol-tab-labels .mol-tab-label-2d,
.mol-tab-3d:checked ~ .mol-tab-labels .mol-tab-label-3d {
    background: #e9f7f3;
    border-color: #bfe4de;
    color: #0a6f66;
}

.mol-tab-2d:checked ~ .mol-tab-panels .mol-panel-2d,
.mol-tab-3d:checked ~ .mol-tab-panels .mol-panel-3d {
    display: block;
}

.mol-panel-note {
    border: 1px solid #f0d8ac;
    background: #fff6e8;
    color: #9c5a06;
    border-radius: 10px;
    padding: 8px 10px;
    font-size: 0.85rem;
}

.summary-card {
    background: var(--card);
    border: 1px solid var(--line-strong);
    border-radius: 12px;
    box-shadow: 0 1px 3px rgba(32, 47, 69, 0.05);
    padding: 14px 16px;
    min-height: 274px;
}

.overview-note {
  background: linear-gradient(90deg, #e8f4f1 0%, #edf8f6 100%);
  border: 1px solid #cde7e1;
  border-radius: 10px;
  color: #2f6961;
  padding: 10px 12px;
  font-size: 0.86rem;
}

.mono {
  font-family: ui-monospace, SFMono-Regular, Menlo, monospace;
  font-size: 0.76rem;
  color: #5f6678;
  word-break: break-all;
}

[data-testid="stTabs"] button {
    font-size: 0.80rem;
    font-weight: 600;
    color: #606a7e;
}

button[data-baseweb="tab"][aria-selected="true"] {
    color: var(--teal) !important;
}

button[data-baseweb="tab"]:hover {
    color: var(--teal-dark) !important;
}

div[data-baseweb="tab-highlight"] {
    background-color: var(--teal) !important;
}

[data-testid="stButton"] button[kind="primary"] {
  background: var(--teal);
  border-color: var(--teal);
    color: #ffffff;
}

[data-testid="stButton"] button[kind="primary"]:hover {
    background: var(--teal-dark);
    border-color: var(--teal-dark);
}

[data-testid="stButton"] button[kind="secondary"] {
    background: #f5fbf9;
    border: 1px solid #bfe4de;
    color: var(--teal-dark);
    border-radius: 10px;
    font-weight: 600;
}

[data-testid="stButton"] button[kind="secondary"]:hover {
    background: #e9f7f3;
    border-color: #98d4cb;
    color: #095f58;
}

#MainMenu { visibility: hidden; }
footer { visibility: hidden; }
[data-testid="stDeployButton"] { display: none; }

/* Tab panel fade-in */
@keyframes fadeUp {
  from { opacity: 0; transform: translateY(6px); }
  to   { opacity: 1; transform: translateY(0); }
}
div[data-baseweb="tab-panel"] {
  animation: fadeUp 0.22s ease forwards;
}

/* Criterion cards */
.crit-card {
  border-radius: 12px;
  padding: 14px 16px;
  border: 1px solid;
  box-shadow: 0 1px 3px rgba(32,47,69,0.05);
    height: 268px;
}
.crit-pass { background: #f0faf5; border-color: #a8dfbd; }
.crit-fail { background: #fff3f3; border-color: #f0abab; }
.crit-header {
  display: flex;
  align-items: center;
  gap: 8px;
  margin-bottom: 10px;
}
.crit-dot {
  width: 9px; height: 9px;
  border-radius: 50%;
  flex-shrink: 0;
  margin-top: 1px;
}
.crit-pass .crit-dot { background: #187d43; }
.crit-fail .crit-dot { background: #9a1e1e; }
.crit-name {
  font-family: 'Space Grotesk', sans-serif;
  font-size: 0.90rem;
  font-weight: 700;
  color: var(--text);
}
.crit-row {
  display: flex;
  justify-content: space-between;
  align-items: baseline;
  font-size: 0.80rem;
  padding: 4px 0;
  border-bottom: 1px solid rgba(0,0,0,0.05);
}
.crit-row:last-of-type { border-bottom: none; }
.crit-row .ck { color: var(--muted); }
.crit-row .cv { font-weight: 600; color: var(--text); }
.crit-desc {
  margin-top: 9px;
  font-size: 0.74rem;
  color: var(--muted);
  line-height: 1.45;
    overflow: hidden;
    display: -webkit-box;
    -webkit-line-clamp: 4;
    -webkit-box-orient: vertical;
}

/* Verdict bar */
.verdict-bar {
  border-radius: 10px;
  padding: 10px 16px;
  display: flex;
  align-items: center;
  gap: 10px;
  font-size: 0.88rem;
  font-weight: 600;
  margin: 12px 0;
  border: 1px solid;
}
.verdict-pass { background: #e7f8ef; border-color: #a8dfbd; color: #187d43; }
.verdict-mid  { background: #fff6e8; border-color: #ffd79c; color: #9c5a06; }
.verdict-fail { background: #ffecec; border-color: #f0abab; color: #9a1e1e; }
.verdict-dot {
  width: 8px; height: 8px;
  border-radius: 50%;
  flex-shrink: 0;
}
.verdict-pass .verdict-dot { background: #187d43; }
.verdict-mid  .verdict-dot { background: #9c5a06; }
.verdict-fail .verdict-dot { background: #9a1e1e; }

/* SHAP legend card */
.shap-key {
  padding: 8px 0;
  border-bottom: 1px solid var(--line);
}
.shap-key:last-child { border-bottom: none; padding-bottom: 0; }
.shap-key .sk { font-size: 0.70rem; text-transform: uppercase; letter-spacing: 0.06em; color: var(--muted); font-weight: 700; margin-bottom: 2px; }
.shap-key .sv { font-family: 'Space Grotesk', sans-serif; font-size: 1.0rem; font-weight: 700; color: var(--text); }
.shap-key .sm { font-size: 0.78rem; color: var(--muted); }

/* st.container(border=True) — class stamped by JS injected via components.html */
.app-card {
  background-color: #ffffff !important;
  border-radius: 12px !important;
  border-color: #d8dee6 !important;
    padding: var(--card-pad-y) var(--card-pad-x) !important;
  box-shadow: 0 1px 3px rgba(32,47,69,0.05) !important;
}
.app-card:hover {
  box-shadow: 0 5px 14px rgba(32,47,69,0.10) !important;
}

/* Plotly charts get card treatment */
[data-testid="stPlotlyChart"] {
    background: transparent;
    border: none;
    border-radius: 10px;
    box-shadow: none;
  overflow: hidden;
    transition: none;
}
[data-testid="stPlotlyChart"]:hover { box-shadow: none; }

.app-card [data-testid="stPlotlyChart"] {
    margin-top: 0;
    margin-bottom: var(--card-gap);
}

.app-card [data-testid="stDataFrame"] {
    margin-top: 0;
    margin-bottom: var(--card-gap);
}

/* Card hover transitions */
.card { transition: box-shadow 0.18s ease, transform 0.18s ease; }
.card:hover { box-shadow: 0 5px 14px rgba(32,47,69,0.10); transform: translateY(-1px); }
.metric-card { transition: box-shadow 0.18s ease, transform 0.18s ease; }
.metric-card:hover { box-shadow: 0 5px 14px rgba(32,47,69,0.10); transform: translateY(-1px); }

/* ── Animations ──────────────────────────────────────────────────────────── */
@keyframes slideDown {
  from { opacity: 0; transform: translateY(-12px); }
  to   { opacity: 1; transform: translateY(0); }
}
@keyframes popIn {
  0%   { opacity: 0; transform: scale(0.92) translateY(5px); }
  70%  { transform: scale(1.02) translateY(-1px); }
  100% { opacity: 1; transform: scale(1) translateY(0); }
}
@keyframes badgePop {
  0%   { opacity: 0; transform: scale(0.70); }
  65%  { transform: scale(1.10); }
  100% { opacity: 1; transform: scale(1); }
}
@keyframes shimmer {
  from { left: -70%; }
  to   { left: 120%; }
}

/* Topbar entrance */
.topbar { animation: slideDown 0.38s cubic-bezier(0.22,1,0.36,1) both; }

/* Welcome hero */
.welcome-hero { animation: fadeUp 0.45s ease 0.05s both; }

/* Stat badges staggered pop */
.welcome-badge:nth-child(1) { animation: badgePop 0.38s ease 0.10s both; }
.welcome-badge:nth-child(2) { animation: badgePop 0.38s ease 0.20s both; }
.welcome-badge:nth-child(3) { animation: badgePop 0.38s ease 0.30s both; }

/* Example cards — stagger by column position */
[data-testid="column"]:nth-child(1) .app-card { animation: popIn 0.40s ease 0.08s both; }
[data-testid="column"]:nth-child(2) .app-card { animation: popIn 0.40s ease 0.18s both; }
[data-testid="column"]:nth-child(3) .app-card { animation: popIn 0.40s ease 0.28s both; }

/* app-card hover lift (bordered containers) */
.app-card {
  transition: box-shadow 0.20s ease, transform 0.20s ease;
}
.app-card:hover {
  box-shadow: 0 6px 18px rgba(32,47,69,0.11);
  transform: translateY(-2px);
}

/* KPI metric cards pop when prediction loads */
.metric-card { animation: popIn 0.36s ease both; }

/* Verdict bar entrance */
.verdict-bar { animation: slideDown 0.30s ease 0.08s both; }

/* Predict button shimmer on hover */
[data-testid="stBaseButton-primary"] {
  position: relative !important;
  overflow: hidden !important;
}
[data-testid="stBaseButton-primary"]::after {
  content: '';
  position: absolute;
  top: 0; bottom: 0;
  width: 55%;
  background: linear-gradient(90deg, transparent, rgba(255,255,255,0.22), transparent);
  left: -70%;
  pointer-events: none;
  transition: none;
}
[data-testid="stBaseButton-primary"]:hover::after {
  animation: shimmer 0.52s ease forwards;
}
</style>
        """,
        unsafe_allow_html=True,
    )
    # Inject JS via components.html (st.markdown strips <script> tags).
    # Accesses parent document from iframe to stamp bordered containers with
    # a stable .app-card class so CSS doesn't depend on emotion-cache hashes.
    st_components.html(
        """
<script>
(function () {
  var doc = window.parent.document;

    function normalize(s) {
        return (s || '').replace(/\\s+/g, ' ').trim().toLowerCase();
    }

    function titleText(el) {
        if (!el) return '';
        // Ignore tooltip icon glyphs so title matching survives header adornments.
        var copy = el.cloneNode(true);
        copy.querySelectorAll('.info-dot').forEach(function (n) { n.remove(); });
        return normalize(copy.textContent);
    }

    function findVisibleCardByTitle(title) {
        var want = normalize(title);
        var cards = Array.from(doc.querySelectorAll('.app-card'));
        for (var i = 0; i < cards.length; i++) {
            var card = cards[i];
            if (card.offsetParent === null) continue;
            var t = card.querySelector('.card-title, .ad-score-title');
            if (!t) continue;
            var got = titleText(t);
            if (got === want) return card;
            // Fallback: tolerate extra suffixes/punctuation from dynamic labels.
            if (got.indexOf(want) === 0) return card;
        }
        return null;
    }

    function equalizePair(titleA, titleB) {
        var a = findVisibleCardByTitle(titleA);
        var b = findVisibleCardByTitle(titleB);
        if (!a || !b) return;
        a.style.minHeight = '';
        b.style.minHeight = '';
        var h = Math.max(a.offsetHeight, b.offsetHeight);
        a.style.minHeight = h + 'px';
        b.style.minHeight = h + 'px';
    }

    function equalizeKnownRows() {
        equalizePair('Local SHAP contributions', 'Training distribution vs estimate');
        equalizePair('Top local SHAP profile', 'How to read');
        equalizePair('Applicability score', 'Uncertainty interval');
        equalizePair('Parity plot', 'Nested CV model comparison');
        equalizePair('Y-randomization', 'Y-randomization summary');
    }

    function equalizeExampleCards() {
        var cards = Array.from(doc.querySelectorAll('.app-card')).filter(function (card) {
            if (card.offsetParent === null) return false;
            return !!card.querySelector('.example-title');
        });
        if (cards.length < 2) return;

        cards.forEach(function (card) {
            card.style.minHeight = '';
        });

        var maxH = 0;
        cards.forEach(function (card) {
            maxH = Math.max(maxH, card.offsetHeight);
        });

        if (maxH > 0) {
            cards.forEach(function (card) {
                card.style.minHeight = maxH + 'px';
            });
        }
    }

  function tag() {
    doc.querySelectorAll('[data-testid="stVerticalBlock"]').forEach(function (el) {
      if (parseFloat(window.parent.getComputedStyle(el).borderTopWidth) > 0.4) {
        el.classList.add('app-card');
      }
    });
        equalizeKnownRows();
                equalizeExampleCards();
  }
  tag();
    setTimeout(tag, 0);
    setTimeout(tag, 60);
        setTimeout(tag, 180);
        setTimeout(tag, 420);
  new window.parent.MutationObserver(tag)
    .observe(doc.documentElement, { childList: true, subtree: true });
    // Remove Streamlit auto-anchor icons for headings that include our info-dot
    function removeAnchorsForInfoDot() {
        var heads = doc.querySelectorAll('h1,h2,h3,h4');
        heads.forEach(function(h){
            if (!h) return;
            if (!h.querySelector('.info-dot')) return;
            var anchors = Array.from(h.querySelectorAll('a'));
            anchors.forEach(function(a){
                if (a.querySelector('svg')) {
                    a.remove();
                }
            });
        });
    }
    removeAnchorsForInfoDot();
    setTimeout(removeAnchorsForInfoDot, 60);
    setTimeout(removeAnchorsForInfoDot, 420);
    new window.parent.MutationObserver(removeAnchorsForInfoDot).observe(doc.documentElement, { childList: true, subtree: true });

    // Wire HTML M/W pill clicks → hidden Streamlit buttons
    function wirePills() {
        doc.querySelectorAll('.msr-pill').forEach(function(pill) {
            if (pill._wired) return;
            pill._wired = true;
            pill.addEventListener('click', function() {
                var key = pill.id === '_msr_pill_m' ? '_msr_unit_m' : '_msr_unit_w';
                var el = doc.querySelector('.st-key-' + key);
                if (el) { var btn = el.querySelector('button'); if (btn) btn.click(); }
            });
        });
    }
    wirePills();
    new window.parent.MutationObserver(wirePills).observe(doc.documentElement, { childList: true, subtree: true });
})();
</script>
        """,
        height=0,
    )
