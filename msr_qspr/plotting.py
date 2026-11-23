import matplotlib
matplotlib.use("Agg")
import matplotlib as mpl
import seaborn as sns
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from textwrap import fill

_PALETTE_NAME = "crest"
_THEME_N_COLORS = 6
_BASE_COLOR_INDEX = 0  # Anchor the palette at its light end.


def get_discrete_colors(n: int):
    """Return n discrete colors from the project palette, anchored at base color."""
    if n <= 1:
        return [get_base_color()]
    colors = sns.color_palette(_PALETTE_NAME, n_colors=n)
    colors[0] = get_base_color()
    return colors


def get_base_color():
    """Return the single-series base color (clear-end of the palette)."""
    return sns.color_palette(_PALETTE_NAME, n_colors=max(2, _THEME_N_COLORS))[_BASE_COLOR_INDEX]


def set_plot_style() -> None:
    """Configure global plotting style for the project."""

    palette = get_discrete_colors(_THEME_N_COLORS)

    sns.set_theme(
        context="paper",
        style="whitegrid",
        palette=palette,
    )

    mpl.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": [
                "Helvetica",
                "Arial",
                "Nimbus Sans",
                "Liberation Sans",
                "DejaVu Sans",
            ],
            "font.size": 11,
            "axes.titlesize": 12,
            "axes.labelsize": 11,
            "xtick.labelsize": 10,
            "ytick.labelsize": 10,
            "legend.fontsize": 10,
            "mathtext.fontset": "dejavusans",   # o "dejavusans"
            "mathtext.default": "regular",

            # PDF/PS font embedding
            "pdf.fonttype": 42,
            "ps.fonttype": 42,

            # Axes and lines
            "axes.labelpad": 10,
            "axes.linewidth": 0.8,
            "axes.edgecolor": "#555555",
            "axes.spines.top": True,
            "axes.spines.right": True,
            "lines.linewidth": 1.4,
            "lines.markersize": 4,

            # Grid
            "axes.grid": True,
            "grid.linestyle": "--",
            "grid.linewidth": 0.4,
            "grid.alpha": 1,
            "grid.color": "#B7B6B6",

            # Facecolors
            "figure.facecolor": "white",
            "axes.facecolor": "#F7F7F7",

            # Ticks
            "xtick.direction": "out",
            "ytick.direction": "out",
            "xtick.color": "#333333",
            "ytick.color": "#333333",

            # Legend
            "legend.framealpha": 0.9,
            "legend.edgecolor": "#AAAAAA",

            # Saving
            "savefig.dpi": 300,
            "savefig.bbox": "tight",
            "savefig.pad_inches": 0.02,

        }
    )

LOG_LABEL = r"$\log_{10}[\mathrm{MSR}/(\mathrm{mol\,mol^{-1}})]$"

# Display names shared by figures, tables and the README; identifiers stay unchanged in data.
CANDIDATE_LABELS = {
    "property_linear": "Reference properties, linear",
    "molecular_ridge": "Molecular descriptors, Ridge",
    "molecular_gb": "Molecular descriptors, GB",
    "enriched_rf": "Descriptors + properties, RF",
    "enriched_gb": "Descriptors + properties, GB",
    "enriched_xgb": "Descriptors + properties, XGBoost",
    "physicochemical_ridge": "Compact panel, Ridge",
    "physicochemical_gb": "Compact panel, GB",
    "solubility_anchored_gb": "Compact panel + anchor, GB",
    "abraham_anchored_ridge": "Compact + Abraham + anchor, Ridge",
    "abraham_anchored_gb": "Compact + Abraham + anchor, GB",
    "abraham_hybrid_gb": "Compact + Abraham + anchor, Ridge-init. GB",
}
# Representation groups, from reference properties to physically informed models.
CANDIDATE_GROUPS = {
    "property_linear": "Reference properties",
    "molecular_ridge": "Ranked descriptors", "molecular_gb": "Ranked descriptors",
    "enriched_rf": "Ranked descriptors", "enriched_gb": "Ranked descriptors", "enriched_xgb": "Ranked descriptors",
    "physicochemical_ridge": "Compact panel", "physicochemical_gb": "Compact panel",
    "solubility_anchored_gb": "Compact panel + anchor",
    "abraham_anchored_ridge": "Compact + Abraham + anchor", "abraham_anchored_gb": "Compact + Abraham + anchor",
    "abraham_hybrid_gb": "Compact + Abraham + anchor",
}
GROUP_ORDER = ("Reference properties", "Ranked descriptors", "Compact panel",
               "Compact panel + anchor", "Compact + Abraham + anchor")
SCENARIO_LABELS = {"unseen_surfactant": "Unseen surfactants", "unseen_contaminant": "Unseen contaminants",
                   "unseen_study": "Unseen studies"}
FEATURE_LABELS = {
    "Cont_MW": "Contaminant MW", "Cont_SLogP": "Contaminant SLogP", "Cont_TopoPSA": "Contaminant TPSA",
    "Surf_MW": "Surfactant MW", "Surf_SLogP": "Surfactant SLogP", "Surf_TopoPSA": "Surfactant TPSA",
    "Abraham_E": "Abraham E", "Abraham_S": "Abraham S", "Abraham_A": "Abraham A",
    "Abraham_B": "Abraham B", "Abraham_V": "Abraham V",
    "CMC": r"$\log_{10}\,$CMC", "WaterSolubility": r"$\log_{10}\,S_\mathrm{w}$",
}
# Marker per representation group, so groups stay distinguishable in greyscale.
GROUP_MARKERS = {"Reference properties": "s", "Ranked descriptors": "D", "Compact panel": "^",
                 "Compact panel + anchor": "v", "Compact + Abraham + anchor": "o"}


def candidate_label(name):
    return CANDIDATE_LABELS.get(name, name)


def feature_label(name):
    return FEATURE_LABELS.get(name, name)


def group_colors():
    """One palette colour per representation group, light (simple) to dark (physically informed)."""
    cmap = sns.color_palette(_PALETTE_NAME, as_cmap=True)
    return {group: cmap(position)[:3] for group, position in zip(GROUP_ORDER, (0.0, 0.28, 0.52, 0.76, 1.0))}


def legend_above(ax_or_fig, handles=None, ncol=None, **kwargs):
    """Frameless legend centred above the axes."""
    handles = handles if handles is not None else ax_or_fig.get_legend_handles_labels()[0]
    target = ax_or_fig
    anchor = (0.5, 0.99) if hasattr(target, "transAxes") else (0.5, 1.0)
    return target.legend(handles=handles, loc="lower center", bbox_to_anchor=anchor,
                         ncol=ncol or len(handles), frameon=False, **kwargs)


def save_figure(fig, directory, name):
    """Write PDF (embedded TrueType fonts) and 300-dpi PNG versions; return both paths."""
    from pathlib import Path
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    paths = []
    for suffix in ("pdf", "png"):
        path = directory / f"{name}.{suffix}"
        fig.savefig(path, dpi=300, metadata={"CreationDate": None} if suffix == "pdf" else None)
        paths.append(path)
    plt.close(fig)
    return paths


def class_label(value):
    """Readable display names; original class keys remain unchanged in tables."""
    return {"Non ionic": "Nonionic", "COCs": "Chlorinated organics",
            "PAHs": "PAHs", "BTEX / light aromatics": "BTEX / light aromatics",
            "Other": "Herbicides and other aromatics"}.get(value, value)

def plot_msr_histogram(ax, msr, *, log_scale=True):
    values = np.asarray(msr, dtype=float)
    values = values[np.isfinite(values) & (values > 0)]
    ax.hist(np.log10(values) if log_scale else values, bins="fd",
            color=get_base_color(), edgecolor="black", linewidth=.4)
    ax.set(xlabel=LOG_LABEL if log_scale else "MSR (mol/mol)", ylabel="Observations")


def plot_msr_ecdf(ax, msr):
    values = np.sort(np.log10(np.asarray(msr, dtype=float)))
    # The zero-probability endpoint makes the empirical step function explicit.
    ax.step(np.r_[values[0], values], np.r_[0, np.arange(1, len(values) + 1) / len(values)],
            where="post", color=get_base_color())
    ax.set(xlabel=LOG_LABEL, ylabel="Cumulative fraction", ylim=(0, 1.02))


def plot_msr_classes(df: pd.DataFrame, *, column=None):
    """Class violins with shared limits; classes ordered by size, sparse classes shown as points."""
    frame = df.loc[df.MSR.gt(0), ["ContaminantType", "SurfactantType", "MSR"]].copy()
    frame["log10_MSR"] = np.log10(frame.MSR)
    columns = [column] if column else ["ContaminantType", "SurfactantType"]
    widths = [6, 3] if len(columns) == 2 else [1]
    fig, axes = plt.subplots(1, len(columns), figsize=(10.5 if len(columns) == 2 else 6.0, 4.4), sharey=True,
                             squeeze=False, gridspec_kw={"width_ratios": widths})
    lower, upper = frame.log10_MSR.agg(["min", "max"])
    margin = max((upper - lower) * .05, .15)
    for i, (ax, column) in enumerate(zip(axes.flat, columns)):
        frame[column] = frame[column].astype(str).str.strip()
        counts = frame[column].value_counts()
        order = list(counts.sort_index().sort_values(ascending=False, kind="stable").index)
        colors = dict(zip(order, get_discrete_colors(len(order))))
        # A density is not informative for fewer than ten observations or a constant class.
        dense = [c for c in order if counts[c] >= 10 and frame.loc[frame[column].eq(c), "log10_MSR"].nunique() > 1]
        sns.violinplot(data=frame.loc[frame[column].isin(dense)], x=column, y="log10_MSR", order=order,
                       hue=column, hue_order=order, palette=colors, legend=False, inner="box",
                       density_norm="width", bw_adjust=.65, cut=0, linewidth=.6, ax=ax)
        for position, name in enumerate(order):
            if name not in dense:
                values = frame.loc[frame[column].eq(name), "log10_MSR"].sort_values().to_numpy()
                offsets = np.linspace(-.12, .12, len(values)) if len(values) > 1 else [0]
                ax.scatter(position + np.asarray(offsets), values, s=20, color="#444444", zorder=3)
        ax.set_xticks(range(len(order)), [fill(class_label(str(c)), 14) + f"\n(n = {counts[c]})" for c in order],
                      fontsize=9)
        label = "Contaminant class" if column == "ContaminantType" else "Surfactant class"
        ax.set(xlabel="", ylabel=LOG_LABEL if i == 0 else "", ylim=(lower - margin, upper + margin),
               title=(f"({chr(97 + i)}) " if len(columns) > 1 else "") + label)
        ax.xaxis.grid(False)
    fig.tight_layout()
    return fig


def plot_partition_ecdf(development, holdout):
    """log10(MSR) empirical cumulative distributions of the development and holdout populations."""
    fig, ax = plt.subplots(figsize=(5.2, 4.3))
    light, dark = get_discrete_colors(2)
    for values, label, color, style in ((development, "Development", light, "-"), (holdout, "Holdout", dark, "--")):
        values = np.sort(np.log10(np.asarray(values, dtype=float)))
        ax.step(np.r_[values[0], values], np.r_[0, np.arange(1, len(values) + 1) / len(values)], where="post",
                color=color, linestyle=style, linewidth=1.8, label=f"{label} (n = {len(values)})")
    ax.set(xlabel=LOG_LABEL, ylabel="Cumulative fraction", ylim=(0, 1.02))
    ax.legend(loc="upper left", frameon=True)
    fig.tight_layout()
    return fig


def plot_class_coverage(df: pd.DataFrame):
    """Observations per contaminant class and surfactant class, with row and column totals."""
    table = pd.crosstab(df.ContaminantType, df.SurfactantType)
    table = table.loc[table.sum(axis=1).sort_values(ascending=False).index,
                      table.sum(axis=0).sort_values(ascending=False).index]
    fig, ax = plt.subplots(figsize=(5.6, 4.0))
    sns.heatmap(table, annot=True, fmt="d", cmap=get_shap_cmap(), cbar=False, linewidths=1, linecolor="white",
                annot_kws={"size": 10}, ax=ax, mask=table.eq(0))
    for i, row in enumerate(table.index):
        for j, value in enumerate(table.loc[row]):
            if value == 0:
                ax.text(j + .5, i + .5, "0", ha="center", va="center", fontsize=10, color="#999999")
    ax.set_xticks(np.arange(table.shape[1]) + .5, [f"{class_label(c)}\n(n = {table[c].sum()})" for c in table.columns],
                  rotation=0, fontsize=9)
    ax.set_yticks(np.arange(table.shape[0]) + .5, [f"{class_label(r)} (n = {table.loc[r].sum()})" for r in table.index],
                  rotation=0, fontsize=9)
    ax.set(xlabel="Surfactant class", ylabel="Contaminant class")
    ax.grid(False)
    ax.tick_params(length=0)
    fig.tight_layout()
    return fig


def get_shap_cmap():
    """Continuous colormap of the project palette (light = low, dark = high)."""
    return sns.color_palette(_PALETTE_NAME, as_cmap=True)




# ---------------------------------------------------------------------------
# Result figures; each takes report tables and returns a matplotlib figure
# ---------------------------------------------------------------------------

RMSE_LABEL = r"RMSE, $\log_{10}(\mathrm{MSR})$"


def _group_style(name):
    group = CANDIDATE_GROUPS[name]
    return dict(color=group_colors()[group], marker=GROUP_MARKERS[group])


def _group_handles(names):
    import matplotlib.lines as mlines
    present = [g for g in GROUP_ORDER if g in {CANDIDATE_GROUPS[n] for n in names}]
    return [mlines.Line2D([], [], color=group_colors()[g], marker=GROUP_MARKERS[g], markersize=6.5,
                          markeredgecolor="white", markeredgewidth=0.6, linestyle="-", linewidth=2, label=g)
            for g in present]


def _point(ax, x, y, name, size=6.5):
    style = _group_style(name)
    ax.plot(x, y, linestyle="none", marker=style["marker"], markersize=size, color=style["color"],
            markeredgecolor="white", markeredgewidth=0.6, zorder=3)


def _label_box(ax, text, loc="lower right"):
    x, ha = (0.97, "right") if "right" in loc else (0.03, "left")
    y, va = (0.03, "bottom") if "lower" in loc else (0.97, "top")
    ax.text(x, y, text, transform=ax.transAxes, ha=ha, va=va, fontsize=9.5,
            bbox=dict(facecolor="white", edgecolor="#AAAAAA", boxstyle="round,pad=0.35", alpha=0.95))


def _figure_legend(fig, handles, ncol, title=None):
    """Frameless legend centred above all panels; returns the fraction of height it occupies."""
    rows = int(np.ceil(len(handles) / ncol)) + (1 if title else 0)
    fig.legend(handles=handles, loc="upper center", bbox_to_anchor=(0.5, 1.0), ncol=ncol, frameon=False,
               columnspacing=1.6, handletextpad=0.5, borderaxespad=0.2, title=title,
               title_fontproperties={"size": 11.5})
    return (0.23 * rows + 0.05) / fig.get_figheight()


def _summary_handles():
    import matplotlib.lines as mlines
    return [mlines.Line2D([], [], color="0.35", marker="o", linestyle="none", markersize=6.5,
                          markeredgecolor="white", label="Mean over repetitions"),
            mlines.Line2D([], [], color="0.35", linewidth=2.2, label="Range over repetitions")]


def _clipped_range(ax, low, high, y, limit, color, horizontal=True, linewidth=2.2, below=False):
    """Draw a range clipped at the axis limit; a clipped end gets an arrow and its value."""
    end = min(high, limit)
    if horizontal:
        ax.plot([low, end], [y, y], color=color, linewidth=linewidth, solid_capstyle="round", zorder=2)
    else:
        ax.plot([y, y], [low, end], color=color, linewidth=linewidth, solid_capstyle="round", zorder=2)
    if high > limit:
        xy = (limit, y) if horizontal else (y, limit)
        ax.plot(*xy, marker=">" if horizontal else "^", markersize=6, color=color, zorder=3, clip_on=False)
        offset, ha = ((-4, -8 if below else 7), "right") if horizontal else ((-5, -7), "right")
        ax.annotate(f"{high:.2f}", xy, xytext=offset, textcoords="offset points", fontsize=9.5,
                    ha=ha, va="center", color="#333333")


def plot_candidate_rmse(performance, order):
    """RMSE per candidate and validation axis: mean and range over repetitions.

    Values beyond the shared axis limit are marked with an arrow and their value.
    """
    set_plot_style()
    positions = np.arange(len(order))[::-1]
    scenarios = list(SCENARIO_LABELS)
    primary = performance[performance.Scenario.isin(scenarios[:2])]
    upper = float(primary.groupby("Configuration").RMSE_mean.max().sort_values().iloc[-2]) * 1.10
    lower = float(performance.RMSE_min.min()) - 0.03
    fig, axes = plt.subplots(1, 3, figsize=(10.5, 5.8), sharey=True)
    for i, (ax, scenario) in enumerate(zip(axes, scenarios)):
        frame = performance[performance.Scenario.eq(scenario)].set_index("Configuration").loc[order]
        for y, (name, row) in zip(positions, frame.iterrows()):
            color = _group_style(name)["color"]
            if row.n_repetitions > 1:
                _clipped_range(ax, row.RMSE_min, row.RMSE_max, y, upper, color)
            if row.RMSE_mean > upper:
                ax.plot(upper, y, marker=">", markersize=6, color=color, zorder=3, clip_on=False)
                ax.annotate(f"mean {row.RMSE_mean:.2f}", (upper, y), xytext=(-4, 7), textcoords="offset points",
                            ha="right", va="center", fontsize=9.5, color="#333333")
            else:
                _point(ax, row.RMSE_mean, y, name)
        ax.axvline(frame.RMSE_mean.min(), color="0.2", linestyle=":", linewidth=1.0, zorder=1)
        ax.set(xlim=(lower, upper), xlabel=RMSE_LABEL, title=f"({chr(97 + i)}) {SCENARIO_LABELS[scenario]}")
        ax.yaxis.grid(False)
        ax.tick_params(axis="y", length=0)
    axes[0].set_yticks(positions, [candidate_label(n) for n in order])
    height = _figure_legend(fig, _group_handles(order) + _summary_handles(), ncol=4)
    fig.tight_layout(rect=(0, 0, 1, 1 - height))
    return fig


def plot_paired_differences(paired, order, selected):
    """RMSE of the selected model minus each alternative, with group-bootstrap 95% intervals."""
    set_plot_style()
    others = [n for n in order if n != selected]
    positions = np.arange(len(others))[::-1]
    fig, axes = plt.subplots(1, 2, figsize=(9.5, 5.2), sharey=True)
    for i, (ax, scenario) in enumerate(zip(axes, list(SCENARIO_LABELS)[:2])):
        frame = paired[(paired.A == selected) & paired.Scenario.eq(scenario)].set_index("B").loc[others]
        for y, (name, row) in zip(positions, frame.iterrows()):
            ax.plot([row.dRMSE_low, row.dRMSE_high], [y, y], color=_group_style(name)["color"], linewidth=2.2,
                    solid_capstyle="round", zorder=2)
            _point(ax, row.dRMSE, y, name)
        ax.axvline(0, color="0.2", linestyle="--", linewidth=1.0, zorder=1)
        ax.set(title=f"({chr(97 + i)}) {SCENARIO_LABELS[scenario]}",
               xlabel=r"$\Delta$RMSE, selected − alternative")
        ax.yaxis.grid(False)
        ax.tick_params(axis="y", length=0)
    axes[0].set_yticks(positions, [candidate_label(n) for n in others])
    height = _figure_legend(fig, _group_handles(others), ncol=3)
    fig.tight_layout(rect=(0, 0, 1, 1 - height))
    return fig


def plot_axis_balance(performance, order):
    """Unseen-surfactant versus unseen-contaminant RMSE of every candidate (numbers: rank).

    Markers are means and whiskers ranges over repetitions; whiskers beyond the axis limits
    are clipped with an arrow and their value.
    """
    set_plot_style()
    surfactant = performance[performance.Scenario.eq("unseen_surfactant")].set_index("Configuration")
    contaminant = performance[performance.Scenario.eq("unseen_contaminant")].set_index("Configuration")
    worst = pd.concat([surfactant.RMSE_mean, contaminant.RMSE_mean], axis=1).max(axis=1)
    typical = [n for n in order if worst[n] <= worst.sort_values().iloc[-2] * 1.001]
    ranges = pd.concat([surfactant.loc[typical, ["RMSE_min", "RMSE_max"]],
                        contaminant.loc[typical, ["RMSE_min", "RMSE_max"]]])
    means = pd.concat([surfactant.RMSE_mean, contaminant.RMSE_mean])
    low = min(float(ranges.min().min()), float(means.min())) - 0.02
    high = max(float(ranges.max().max()), float(means.max())) + 0.02
    fig, ax = plt.subplots(figsize=(6.0, 6.6))
    ax.plot([low, high], [low, high], linestyle="--", linewidth=1.0, color="0.2", zorder=1)
    for name in order:
        x, y = surfactant.loc[name], contaminant.loc[name]
        color = _group_style(name)["color"]
        _clipped_range(ax, x.RMSE_min, x.RMSE_max, y.RMSE_mean, high, color, linewidth=1.0, below=True)
        _clipped_range(ax, y.RMSE_min, y.RMSE_max, x.RMSE_mean, high, color, horizontal=False, linewidth=1.0)
        _point(ax, x.RMSE_mean, y.RMSE_mean, name, size=9)
        ax.annotate(str(order.index(name) + 1), (x.RMSE_mean, y.RMSE_mean), xytext=(6, -10),
                    textcoords="offset points", fontsize=9, color="#222222", zorder=4)
    ax.set(xlim=(low, high), ylim=(low, high), aspect="equal",
           xlabel=r"Unseen-surfactant RMSE, $\log_{10}(\mathrm{MSR})$",
           ylabel=r"Unseen-contaminant RMSE, $\log_{10}(\mathrm{MSR})$")
    height = _figure_legend(fig, _group_handles(order) + _summary_handles(), ncol=2)
    fig.tight_layout(rect=(0, 0, 1, 1 - height))
    return fig


def plot_class_errors(class_errors, order, scenario="unseen_contaminant"):
    """RMSE by contaminant class (rows) and candidate (columns), pooled over repetitions."""
    set_plot_style()
    frame = class_errors[class_errors.Scenario.eq(scenario) & class_errors.class_field.eq("ContaminantType")]
    counts = frame.drop_duplicates("class_value").set_index("class_value").n_observations_per_repetition
    classes = counts.sort_values(ascending=False).index
    matrix = frame.pivot(index="class_value", columns="Configuration", values="RMSE").loc[classes, order]
    fig, ax = plt.subplots(figsize=(10.5, 4.8))
    sns.heatmap(matrix, annot=True, fmt=".2f", cmap=get_shap_cmap(), linewidths=1, linecolor="white",
                annot_kws={"size": 9.5}, cbar_kws={"label": RMSE_LABEL, "pad": 0.015, "shrink": 0.9}, ax=ax)
    ax.set_xticks(np.arange(len(order)) + .5, [fill(candidate_label(n), 18) for n in order], rotation=90, fontsize=9.5)
    ax.set_yticks(np.arange(len(classes)) + .5, [f"{class_label(c)} (n = {int(counts[c])})" for c in classes],
                  rotation=0, fontsize=9)
    ax.set(xlabel="", ylabel="")
    ax.tick_params(length=0)
    fig.tight_layout()
    return fig


def _beeswarm_offsets(values, width=0.38, bins=60):
    """Vertical offsets that spread points sharing similar values, like a beeswarm."""
    values = np.asarray(values, dtype=float)
    span = np.ptp(values) or 1.0
    index = np.floor((values - values.min()) / span * (bins - 1)).astype(int)
    offsets = np.zeros(len(values))
    for b in np.unique(index):
        members = np.flatnonzero(index == b)
        steps = np.arange(len(members))
        offsets[members] = np.where(steps % 2, -1, 1) * np.ceil(steps / 2)
    largest = np.abs(offsets).max()
    return offsets / largest * width if largest else offsets


def plot_interpretation(attributions, shap_values, anchored=False):
    """Learned attributions: (a) mean absolute SHAP per input, (b) SHAP per observation.

    Both panels share the feature order. For anchored models the attributions describe the
    learned correction; the imposed solubility offset is not attributed.
    """
    from matplotlib.colors import Normalize
    set_plot_style()
    table = attributions.dropna(subset=["mean_absolute_SHAP"]).sort_values("mean_absolute_SHAP")
    features = list(table.feature)
    positions = np.arange(len(features))
    fig, axes = plt.subplots(1, 2, figsize=(10.5, 0.36 * len(features) + 1.9), sharey=True,
                             gridspec_kw={"width_ratios": [1, 1.55]})
    width = float(table.mean_absolute_SHAP.max())
    bars = axes[0].barh(positions, table.mean_absolute_SHAP, color=get_discrete_colors(10)[-1], height=0.7)
    for bar, value in zip(bars, table.mean_absolute_SHAP):
        inside = value > 0.3 * width
        axes[0].text(value * 0.98 if inside else value + 0.015 * width, bar.get_y() + bar.get_height() / 2,
                     f"{value:.3f}", va="center", ha="right" if inside else "left",
                     color="white" if inside else "#333333", fontsize=9.5, fontweight="bold" if inside else "normal")
    axes[0].set(xlim=(0, width * 1.12), xlabel=r"Mean |SHAP|, $\log_{10}(\mathrm{MSR})$", title="(a) Global")
    cmap = get_shap_cmap()
    for i, feature in enumerate(features):
        attribution = shap_values[f"shap__{feature}"].to_numpy(dtype=float)
        value = shap_values[f"value__{feature}"].to_numpy(dtype=float)
        low, high = np.nanpercentile(value, [5, 95]) if np.isfinite(value).any() else (0, 1)
        colors = cmap(Normalize(low, high if high > low else low + 1, clip=True)(value))
        order = np.random.default_rng(0).permutation(len(attribution))
        axes[1].scatter(attribution[order], i + _beeswarm_offsets(attribution)[order], c=colors[order], s=9,
                        linewidths=0, alpha=0.9, zorder=3)
    axes[1].axvline(0, color="0.4", linewidth=1.0, zorder=2)
    axes[1].set(xlabel=r"SHAP value, $\log_{10}(\mathrm{MSR})$", title="(b) Per observation")
    colorbar = fig.colorbar(plt.cm.ScalarMappable(cmap=cmap), ax=axes[1], pad=0.015, aspect=30, ticks=[0, 1])
    colorbar.ax.set_yticklabels(["Low", "High"])
    colorbar.set_label("Input value", labelpad=-2)
    colorbar.outline.set_visible(False)
    axes[0].set_yticks(positions, [feature_label(f) for f in features])
    for ax in axes:
        ax.yaxis.grid(False)
        ax.tick_params(axis="y", length=0)
        ax.set_ylim(-0.6, len(features) - 0.4)
    if anchored:
        _label_box(axes[0], "Learned correction;\nsolubility offset\nreported separately")
    fig.tight_layout()
    return fig


def plot_applicability(components, distance):
    """OOF RMSE by applicability status and by distance to the training support."""
    set_plot_style()
    scenarios = list(SCENARIO_LABELS)
    fig, axes = plt.subplots(1, 2, figsize=(10.0, 4.4), sharey=True)
    status_rows = components[components.component.eq("ad_status")]
    statuses = [s for s in ("inside", "outside", "incomplete") if s in set(status_rows.status)]
    status_colors = dict(zip(("inside", "outside", "incomplete"), get_discrete_colors(6)[::2]))
    width = 0.8 / len(statuses)
    for k, status in enumerate(statuses):
        frame = status_rows[status_rows.status.eq(status)].set_index("Scenario").reindex(scenarios)
        x = np.arange(len(scenarios)) + (k - (len(statuses) - 1) / 2) * width
        bars = axes[0].bar(x, frame.RMSE, width=width * 0.95, color=status_colors[status],
                           label=status.capitalize())
        for bar, share in zip(bars, frame.share_of_rows):
            if np.isfinite(share):
                axes[0].text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.012, f"{100 * share:.0f}%",
                             ha="center", va="bottom", fontsize=9)
    axes[0].set_xticks(np.arange(len(scenarios)), [SCENARIO_LABELS[s] for s in scenarios])
    axes[0].set(ylabel=RMSE_LABEL, title="(a) Applicability domain")
    axes[0].xaxis.grid(False)
    axes[0].legend(loc="upper left", frameon=True, fontsize=9)
    quartiles = list(dict.fromkeys(distance.distance_quartile))
    quartile_colors = get_discrete_colors(len(quartiles))
    width = 0.8 / len(quartiles)
    for k, quartile in enumerate(quartiles):
        frame = distance[distance.distance_quartile.eq(quartile)].set_index("Scenario").reindex(scenarios)
        x = np.arange(len(scenarios)) + (k - (len(quartiles) - 1) / 2) * width
        axes[1].bar(x, frame.RMSE, width=width * 0.95, color=quartile_colors[k],
                    label=quartile.replace("D1", "Q1").replace("D2", "Q2").replace("D3", "Q3").replace("D4", "Q4"))
    axes[1].set_xticks(np.arange(len(scenarios)), [SCENARIO_LABELS[s] for s in scenarios])
    axes[1].set(title="(b) Distance to training support")
    axes[1].xaxis.grid(False)
    axes[1].legend(loc="upper left", frameon=True, fontsize=9, ncol=2)
    top = max(float(status_rows.RMSE.max()), float(distance.RMSE.max()))
    axes[0].set_ylim(0, top * 1.22)
    fig.tight_layout()
    return fig


def plot_within_contaminant_ranking(ranking, models, seed=42):
    """Within-contaminant Spearman correlation per unseen contaminant, by class, for two models.

    Points are contaminants (mean over repetitions); bars are class medians.
    """
    import matplotlib.lines as mlines
    set_plot_style()
    per_contaminant = ranking.groupby(["Configuration", "contaminant_group", "ContaminantType"]).agg(
        spearman=("spearman", "mean")).reset_index()
    selected = per_contaminant[per_contaminant.Configuration.eq(models[0])]
    classes = selected.groupby("ContaminantType").spearman.median().sort_values(ascending=False).index
    rng = np.random.default_rng(seed)
    dark, light = get_discrete_colors(6)[-1], get_discrete_colors(6)[1]
    styles = ((models[0], dark, dark, -0.18), (models[1], "white", light, 0.18))
    fig, ax = plt.subplots(figsize=(7.2, 4.4))
    for i, name in enumerate(classes):
        for model, face, edge, shift in styles:
            frame = per_contaminant[per_contaminant.ContaminantType.eq(name) & per_contaminant.Configuration.eq(model)]
            ax.scatter(i + shift + rng.uniform(-0.07, 0.07, len(frame)), frame.spearman, s=30, color=face,
                       edgecolor=edge, linewidth=1.0, zorder=3)
            ax.plot([i + shift - 0.13, i + shift + 0.13], [frame.spearman.median()] * 2, color=edge,
                    linewidth=2.2, zorder=4)
    ax.axhline(0, color="0.2", linestyle="--", linewidth=1.0)
    counts = selected.ContaminantType.value_counts()
    ax.set_xticks(range(len(classes)), [f"{fill(class_label(c), 14)}\n(n = {counts[c]})" for c in classes], fontsize=9)
    low = min(-0.1, float(per_contaminant.spearman.min()) - 0.15)
    ax.set(xlim=(-0.6, len(classes) - 0.4), ylim=(low, 1.05), ylabel="Within-contaminant Spearman ρ")
    ax.xaxis.grid(False)
    handles = [mlines.Line2D([], [], color=edge, markerfacecolor=face, marker="o", markersize=7, linestyle="none",
                             label=candidate_label(model)) for model, face, edge, _ in styles]
    height = _figure_legend(fig, handles, ncol=2)
    fig.tight_layout(rect=(0, 0, 1, 1 - height))
    return fig


def plot_screening_case(table, title=None):
    """Predicted MSR and WSR of candidate surfactants on log scales, coloured by applicability."""
    import matplotlib.lines as mlines
    set_plot_style()
    palette = get_discrete_colors(6)
    styles = {"within_applicability_domain": (palette[-1], "o", "Inside applicability domain"),
              "incomplete_reliability_information": (palette[2], "s", "Incomplete reliability information"),
              "outside_applicability_domain": ("#9A9A9A", "X", "Outside applicability domain")}
    table = table.sort_values("rank_by_MSR", ascending=False).reset_index(drop=True)
    y = np.arange(len(table))
    fig, axes = plt.subplots(1, 2, figsize=(9.0, 0.42 * len(table) + 2.0), sharey=True)
    panels = ((axes[0], "predicted_MSR_mol_per_mol", r"Predicted MSR (mol mol$^{-1}$)", "rank_by_MSR", "(a) Molar basis"),
              (axes[1], "predicted_WSR_g_per_g", r"Predicted WSR (g g$^{-1}$)", "rank_by_WSR", "(b) Mass basis"))
    for ax, column, xlabel, rank, title_panel in panels:
        values = table[column]
        ax.hlines(y, values.min() / 3, values, color="#CCCCCC", linewidth=1.0, zorder=1)
        for yi, (value, status, order) in enumerate(zip(values, table.decision_support, table[rank])):
            color, marker, _ = styles[status]
            ax.plot(value, yi, marker=marker, markersize=8, color=color, markeredgecolor="white",
                    markeredgewidth=0.6, linestyle="none", zorder=3)
            ax.annotate(f"#{order}", (value, yi), xytext=(7, 0), textcoords="offset points", va="center", fontsize=9.5)
        ax.set_xscale("log")
        ax.set_xlim(values.min() / 3, values.max() * 3)
        ax.set(xlabel=xlabel, title=title_panel)
        ax.yaxis.grid(False)
        ax.tick_params(axis="y", length=0)
    axes[0].set_yticks(y, table.surfactant)
    present = [s for s in styles if s in set(table.decision_support)]
    handles = [mlines.Line2D([], [], color=styles[s][0], marker=styles[s][1], linestyle="none", markersize=8,
                             markeredgecolor="white", label=styles[s][2]) for s in present]
    height = _figure_legend(fig, handles, ncol=len(handles), title=title)
    fig.tight_layout(rect=(0, 0, 1, 1 - height))
    return fig


def plot_y_randomization(permutations, summary):
    """Shuffled-target RMSE across permutations; the vertical line is the selected model."""
    import matplotlib.lines as mlines
    import matplotlib.patches as mpatches
    set_plot_style()
    summary = summary.set_index("Scenario")
    dark = get_discrete_colors(6)[-1]
    fig, axes = plt.subplots(1, 3, figsize=(10.0, 3.9), sharex=True, sharey=True)
    all_values = permutations.mean_repetition_RMSE
    bins = np.linspace(all_values.min(), all_values.max(), 16)
    for i, (ax, scenario) in enumerate(zip(axes, SCENARIO_LABELS)):
        values = permutations[permutations.Scenario.eq(scenario)].mean_repetition_RMSE
        ax.hist(values, bins=bins, color=get_base_color(), edgecolor="black", linewidth=0.4)
        selected = summary.loc[scenario, "selected_RMSE"]
        ax.axvline(selected, color=dark, linewidth=2.2)
        ax.set(title=f"({chr(97 + i)}) {SCENARIO_LABELS[scenario]}", xlabel=RMSE_LABEL)
    axes[0].set_ylabel("Permutations")
    axes[0].yaxis.set_major_locator(plt.MaxNLocator(integer=True))
    handles = [mpatches.Patch(facecolor=get_base_color(), edgecolor="black", linewidth=0.4, label="Shuffled targets"),
               mlines.Line2D([], [], color=dark, linewidth=2.2, label="Selected model")]
    height = _figure_legend(fig, handles, ncol=2)
    fig.tight_layout(rect=(0, 0, 1, 1 - height))
    return fig


def plot_holdout_parity(development, holdout, holdout_metrics):
    """Observed versus predicted log10(MSR) of the selected model.

    Development points are unseen-surfactant OOF predictions averaged over the three
    repetitions of each observation; the annotation gives the recorded holdout metrics.
    """
    set_plot_style()
    values = np.r_[development.actual, development.predicted, holdout.actual_log10_MSR, holdout.predicted_log10_MSR]
    pad = 0.03 * (values.max() - values.min())
    low, high = values.min() - pad, values.max() + pad
    light, dark = get_discrete_colors(2)
    fig, ax = plt.subplots(figsize=(6.0, 6.3))
    sns.scatterplot(x=development.actual, y=development.predicted, ax=ax, s=24, alpha=0.45, color=light,
                    edgecolor="none", label="Development, OOF (mean of 3 repetitions)")
    sns.scatterplot(x=holdout.actual_log10_MSR, y=holdout.predicted_log10_MSR, ax=ax, s=45, marker="X",
                    color=dark, edgecolor=dark, linewidth=0.7, label="Holdout")
    ax.plot([low, high], [low, high], linestyle="--", linewidth=1.0, color="0.2")
    _label_box(ax, f"Holdout: RMSE = {holdout_metrics['RMSE']:.2f}, " + r"R$^2$" + f" = {holdout_metrics['R2']:.2f}")
    ax.set(xlim=(low, high), ylim=(low, high), aspect="equal", xlabel=r"Observed $\log_{10}(\mathrm{MSR})$",
           ylabel=r"Predicted $\log_{10}(\mathrm{MSR})$")
    ax.set_axisbelow(True)
    ax.legend(loc="lower center", bbox_to_anchor=(0.5, 1.0), ncol=2, frameon=False)
    fig.tight_layout()
    return fig

