"""Plotly chart factories for the MSR Estimator app."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st


def _read_csv(path: Path) -> pd.DataFrame | None:
    return pd.read_csv(path) if path.exists() else None


def plotly_layout(**kwargs) -> dict:
    base = dict(
        paper_bgcolor="#ffffff",
        plot_bgcolor="#f8fafb",
        font=dict(color="#4f5565", size=12, family="Inter"),
        xaxis=dict(gridcolor="#e2e6eb", zerolinecolor="#dbe1e8"),
        yaxis=dict(gridcolor="#e2e6eb", zerolinecolor="#dbe1e8"),
        margin=dict(l=12, r=12, t=42, b=12),
        legend=dict(bgcolor="rgba(0,0,0,0)"),
    )
    base.update(kwargs)
    return base


def shap_bar(result, n_show: int, fixed_height: int | None = None) -> go.Figure:
    sv = np.asarray(result.shap_values)
    fv = np.asarray(result.feature_values)
    fn = list(result.feature_names)

    n = min(n_show, len(sv))
    idx = np.argsort(np.abs(sv))[-n:]
    sv_s = sv[idx]
    fv_s = fv[idx]
    fn_s = [fn[i] for i in idx]

    colors = ["#0c8a7f" if x >= 0 else "#e26b6b" for x in sv_s]
    labels = [f"{k} ({v:.3g})" for k, v in zip(fn_s, fv_s)]
    span = max(0.08, float(np.max(np.abs(sv_s))) * 0.25)

    fig = go.Figure(
        go.Bar(
            x=sv_s,
            y=labels,
            orientation="h",
            marker_color=colors,
            text=[f"{x:+.3f}" for x in sv_s],
            textposition="outside",
            cliponaxis=False,
        )
    )
    fig.add_vline(x=0, line_color="#9aa4b2", line_width=1)
    bar_height = fixed_height if fixed_height is not None else max(300, n * 28 + 76)
    fig.update_layout(
        **plotly_layout(
            xaxis_title="Impact on log10(MSR)",
            height=bar_height,
            xaxis=dict(range=[float(np.min(sv_s)) - span, float(np.max(sv_s)) + span], gridcolor="#e2e6eb"),
        )
    )
    return fig


def distribution(y_cv_log: np.ndarray, pred: float) -> go.Figure:
    fig = go.Figure(
        go.Histogram(
            x=y_cv_log,
            nbinsx=20,
            marker_color="rgba(12,138,127,0.58)",
            marker_line=dict(color="#0c8a7f", width=0.7),
            name="Training",
        )
    )
    fig.add_vline(
        x=pred,
        line_color="#d64545",
        line_width=2,
        annotation_text=f"Pred {pred:+.3f}",
        annotation_font_color="#d64545",
    )
    fig.update_layout(**plotly_layout(xaxis_title="log10(MSR)", yaxis_title="Count", height=300))
    return fig


def parity(artifacts: Path) -> go.Figure | None:
    train = _read_csv(artifacts / "final_model/preds/xgboost_train_predictions.csv")
    holdout = _read_csv(artifacts / "final_model/preds/xgboost_holdout_predictions.csv")
    if train is None or holdout is None:
        return None

    allv = pd.concat([train["y_true"], train["y_pred"], holdout["y_true"], holdout["y_pred"]])
    lo, hi = float(allv.min()) - 0.05, float(allv.max()) + 0.05

    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=train["y_true"], y=train["y_pred"], mode="markers", name="Train CV", marker=dict(color="rgba(12,138,127,0.55)", size=7)
        )
    )
    fig.add_trace(
        go.Scatter(
            x=holdout["y_true"], y=holdout["y_pred"], mode="markers", name="Holdout", marker=dict(color="#f39c2a", size=9, symbol="x")
        )
    )
    fig.add_trace(go.Scatter(x=[lo, hi], y=[lo, hi], mode="lines", name="Ideal", line=dict(color="#7c8795", dash="dash")))
    fig.update_layout(
        **plotly_layout(
            xaxis_title="True log10(MSR)",
            yaxis_title="Estimated log10(MSR)",
            xaxis=dict(range=[lo, hi], gridcolor="#e2e6eb"),
            yaxis=dict(range=[lo, hi], gridcolor="#e2e6eb"),
            height=420,
        )
    )
    return fig


def global_shap(artifacts: Path) -> go.Figure | None:
    df = _read_csv(artifacts / "diagnostics/shap/xgboost_shap_importance.csv")
    if df is None or not {"feature", "mean_abs_shap"}.issubset(df.columns):
        return None

    top = df.nlargest(12, "mean_abs_shap")[::-1]
    fig = go.Figure(go.Bar(x=top["mean_abs_shap"], y=top["feature"], orientation="h", marker_color="rgba(12,138,127,0.72)"))
    fig.update_layout(**plotly_layout(xaxis_title="Mean |SHAP|", height=360))
    return fig


def feature_stability(artifacts: Path) -> go.Figure | None:
    path = artifacts / "feature_selection/summaries/global_feature_frequency.csv"
    df = _read_csv(path)
    if df is None or "feature" not in df.columns:
        return None

    freq_col = "freq" if "freq" in df.columns else "count"
    top = df.nlargest(30, freq_col)[::-1]
    fig = go.Figure(
        go.Bar(
            x=top[freq_col],
            y=top["feature"],
            orientation="h",
            marker=dict(color=top[freq_col], colorscale="Teal", line=dict(color="#b6d9d6", width=0.4)),
        )
    )
    fig.update_layout(**plotly_layout(xaxis_title=freq_col, height=620))
    return fig


def ad_quantile(artifacts: Path) -> go.Figure | None:
    df = _read_csv(artifacts / "diagnostics/applicability_domain/xgboost_ad_quantiles.csv")
    if df is None or not {"quantile", "coverage", "R2_in_AD"}.issubset(df.columns):
        return None

    fig = go.Figure()
    fig.add_trace(
        go.Scatter(x=df["quantile"], y=df["coverage"] * 100, mode="lines+markers", name="Coverage (%)", line=dict(color="#0c8a7f"), yaxis="y1")
    )
    fig.add_trace(
        go.Scatter(x=df["quantile"], y=df["R2_in_AD"], mode="lines+markers", name="R2 in AD", line=dict(color="#f39c2a", dash="dot"), yaxis="y2")
    )
    fig.add_vline(x=0.95, line_color="#d64545", line_dash="dash", line_width=1)
    fig.update_layout(
        **plotly_layout(
            xaxis_title="Structural quantile",
            yaxis=dict(title="Coverage (%)", range=[0, 105], gridcolor="#e2e6eb"),
            yaxis2=dict(title="R2", overlaying="y", side="right", range=[0.85, 1.02]),
            height=320,
        )
    )
    return fig


def ad_dist_error(result, artifacts: Path) -> go.Figure | None:
    df = _read_csv(artifacts / "diagnostics/applicability_domain/xgboost_ad_dist_error.csv")
    if df is None or not {"mean_dist", "local_error"}.issubset(df.columns):
        return None

    fig = go.Figure()
    fig.add_trace(go.Scatter(x=df["mean_dist"], y=df["local_error"], mode="markers", marker=dict(color="rgba(120,125,132,0.42)", size=7), name="Training"))
    fig.add_trace(
        go.Scatter(
            x=[result.ad.structural.value], y=[result.ad.local_error.value], mode="markers", marker=dict(color="#0c8a7f", size=16, symbol="star"), name="Current"
        )
    )
    fig.add_vline(x=result.ad.structural.threshold, line_color="#d64545", line_dash="dash")
    fig.update_layout(**plotly_layout(xaxis_title="Mean distance", yaxis_title="Local CV error", height=320))
    return fig


def y_randomization(artifacts: Path) -> tuple[go.Figure | None, float | None, int]:
    df = _read_csv(artifacts / "diagnostics/y_randomization/xgboost_y_randomization_results.csv")
    if df is None or "R2" not in df.columns:
        return None, None, 0

    if "kind" in df.columns:
        perm = df[df["kind"] == "perm"]["R2"].to_numpy(dtype=float)
        real = df[df["kind"] == "real"]["R2"].to_numpy(dtype=float)
        real_val = float(real[0]) if len(real) else None
    else:
        perm = df["R2"].to_numpy(dtype=float)
        real_val = None

    fig = go.Figure(go.Histogram(x=perm, nbinsx=20, marker_color="rgba(227,97,97,0.58)", name="Permuted R2"))
    if real_val is not None:
        fig.add_vline(x=real_val, line_color="#2f8a42", line_width=2)
    fig.update_layout(**plotly_layout(xaxis_title="R2", yaxis_title="Count", height=300))
    return fig, real_val, int(len(perm))
