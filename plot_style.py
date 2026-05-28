import matplotlib as mpl
import seaborn as sns


_PALETTE_NAME = "crest"
_THEME_N_COLORS = 6
_BASE_COLOR_INDEX = 0  # clear-end anchor requested by user


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


def get_shap_cmap():
    """Return a continuous colormap derived from the seaborn palette."""
    return sns.color_palette(_PALETTE_NAME, as_cmap=True)
