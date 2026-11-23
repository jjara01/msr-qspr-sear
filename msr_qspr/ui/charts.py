"""Plotly chart factories for the MSR Estimator app."""

from __future__ import annotations

from pathlib import Path
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from msr_qspr.inference import report_table

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


def global_shap(artifacts: Path) -> go.Figure | None:
    df = report_table(artifacts, 'interpretation.csv')
    if df is None or not {"feature", "mean_absolute_SHAP"}.issubset(df.columns):
        return None

    top = df.nlargest(12, "mean_absolute_SHAP")[::-1]
    fig = go.Figure(go.Bar(x=top["mean_absolute_SHAP"], y=top["feature"], orientation="h", marker_color="rgba(12,138,127,0.72)"))
    fig.update_layout(**plotly_layout(xaxis_title="Mean |SHAP|", height=360))
    return fig


def parity(artifacts: Path) -> go.Figure | None:
    train = report_table(artifacts, 'primary_oof_predictions.csv')
    holdout = report_table(artifacts, 'holdout_predictions.csv')
    actual, predicted = 'actual_log10_MSR', 'predicted_log10_MSR'
    if train is None or holdout is None or any(not {actual,predicted} <= set(t) for t in (train,holdout)):
        return None
    # Unseen-surfactant predictions averaged over repetitions, as in the manuscript parity figure.
    if {'Scenario', 'observation_id'} <= set(train):
        train = (train.loc[train['Scenario'].eq('unseen_surfactant')]
                 .groupby('observation_id', as_index=False)[[actual, predicted]].mean())
    values = pd.concat([t[c] for t in (train,holdout) for c in (actual,predicted)])
    lo,hi = float(values.min()),float(values.max())
    fig = go.Figure()
    for label,table in (('Development (unseen surfactants)',train),('Reserved holdout',holdout)):
        fig.add_trace(go.Scatter(x=table[actual],y=table[predicted],mode='markers',name=label))
    fig.add_trace(go.Scatter(x=[lo,hi],y=[lo,hi],mode='lines',name='Ideal'))
    fig.update_layout(**plotly_layout(xaxis_title='Observed log10(MSR)',
                                      yaxis_title='Predicted log10(MSR)',height=420))
    return fig
