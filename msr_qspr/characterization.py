"""Descriptive characterization of all 459 accepted MSR observations."""

from __future__ import annotations

import matplotlib
matplotlib.use("Agg")
import hashlib
import importlib.metadata
import json
import re
from textwrap import fill
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from rdkit import Chem, rdBase
from rdkit.Chem import Descriptors
from scipy import stats
import seaborn as sns
from msr_qspr.chemistry import canonicalize
from msr_qspr.plotting import get_base_color, get_discrete_colors, set_plot_style, save_figure
from msr_qspr.validation import _holdout_metadata, reserved_grouped_holdout
from msr_qspr.data import verify_scientific_inputs, corpus_metadata, _read_workbook, _load_observations
from msr_qspr.artifacts import fingerprint, record_cache
from msr_qspr.plotting import (plot_msr_histogram, plot_msr_ecdf, plot_msr_classes, plot_class_coverage,
                               plot_partition_ecdf, class_label, LOG_LABEL)


from msr_qspr.config import PROJECT_ROOT as ROOT, FIXED_MOLECULAR_PANEL, study_experiments
GROUPS = ("publication_group", "active_structure_group")
QUANTILES = (0, .01, .05, .25, .5, .75, .95, .99, 1)
FIGURE_CAPTIONS = {
    "target_distribution": "log10(MSR) histogram (Freedman–Diaconis bins) and ECDF; all observed tails retained.",
    "chemical_classes": "log10(MSR) by contaminant and surfactant class, ordered by descending observation count; n denotes observations.",
    "class_coverage": "observations per contaminant and surfactant class, with class totals.",
    "partition_ecdf": "log10(MSR) empirical cumulative distributions of the development and holdout populations; the holdout withholds complete surfactant groups.",
    "group_heterogeneity": "log10(MSR), 15 largest publications and active-structure groups; lines are observed minimum–maximum, points are medians, not confidence intervals. Bibliography numbers identify publications.",
    "condition_coverage": "assay-temperature provenance and numerical-temperature unavailability by contaminant class. Labels show unavailable/total observations; qualitative room temperature remains non-numerical.",
    "physicochemical_properties": "observation-weighted CMC ECDFs by surfactant class (mol/L), aqueous-solubility ECDFs by contaminant class (g/L), and descriptive Spearman correlations. Graph molecular weight is in g/mol.",
    "target_diagnostics": "raw MSR histogram and normal Q–Q plots on raw and log10 scales; reference lines aid comparison with normal quantiles.",
}


def load_corpus() -> pd.DataFrame:
    """Read all accepted responses and align verified chemical/group identities."""
    metadata = corpus_metadata()
    df = _load_observations(metadata, np.arange(len(metadata)))
    identity = _holdout_metadata(metadata).set_index('row_id').loc[df.row_id].reset_index()
    for column in GROUPS:
        df[column] = identity[column]
    for column, strip in (("surfactant_id", False), ("contaminant_id", True)):
        smiles = "SMILES_surfactant" if column == "surfactant_id" else "SMILES_contaminant"
        expected = df[smiles].map(lambda s: canonicalize(s, strip))
        df[column] = expected
    pair = df.surfactant_id + "__" + df.contaminant_id
    df['pair_id'] = pair
    if not pair.is_unique:
        raise ValueError("Full represented chemical pairs must be unique and consistent.")
    y = pd.to_numeric(df.MSR, errors="raise")
    if not np.isfinite(y).all() or not y.gt(0).all():
        raise ValueError("MSR must be finite and strictly positive.")
    df["log10_MSR"] = np.log10(y)
    df["active_pair"] = df.active_structure_group + "__" + df.contaminant_id
    df["chemical_cell"] = df.SurfactantType + " | " + df.ContaminantType
    return df


def summarize(values) -> dict:
    x = pd.Series(values, dtype=float)
    x = x[np.isfinite(x)]
    q = x.quantile(QUANTILES)
    return {"n": len(x), "mean": x.mean(), "std": x.std(),
            "iqr": q.loc[.75] - q.loc[.25], "skewness": x.skew(),
            "excess_kurtosis": x.kurt(),
            **{f"q{round(p * 100):02d}": value for p, value in q.items()}}


def group_summary(df: pd.DataFrame, column: str) -> pd.DataFrame:
    rows = []
    for key, group in df.groupby(column, sort=True):
        rows.append({"group": key, **summarize(group.log10_MSR),
                     "MSR_min": group.MSR.min(), "MSR_median": group.MSR.median(),
                     "MSR_max": group.MSR.max(), "publications": group.publication_group.nunique(),
                     "active_structures": group.active_structure_group.nunique(),
                     "contaminants": group.contaminant_id.nunique(),
                     "surfactant_labels": " | ".join(sorted(group.Surfactant.unique()))})
    return pd.DataFrame(rows)


def variance_fraction(df: pd.DataFrame, column: str) -> float:
    """Descriptive between-group sum of squares / total, with no inferential test."""
    y = df.log10_MSR
    grouped = df.groupby(column).log10_MSR.agg(["size", "mean"])
    return float((grouped["size"] * (grouped["mean"] - y.mean()) ** 2).sum()
                 / ((y - y.mean()) ** 2).sum())


def recorded_holdouts(df: pd.DataFrame) -> dict:
    return {name: reserved_grouped_holdout(df)
            for name in ('unseen_surfactant',)}


def partition_diagnostics(df, development, test, group_column) -> dict:
    a, b = df.iloc[development], df.iloc[test]
    if set(a[group_column]) & set(b[group_column]):
        raise ValueError("Partition violates group isolation.")
    result = {"n_development": len(a), "n_test": len(b), "isolated_group_overlap": 0}
    for role, part in (("development", a), ("test", b)):
        for col in (*GROUPS, "contaminant_id", "chemical_cell"):
            result[f"{role}_{col}_coverage"] = part[col].nunique()
    for col in ("ContaminantType", "SurfactantType", "chemical_cell"):
        p, q = a[col].value_counts(normalize=True).align(b[col].value_counts(normalize=True), fill_value=0)
        result[f"{col}_total_variation"] = .5 * (p - q).abs().sum()
        result[f"test_missing_{col}"] = " | ".join(sorted(set(df[col]) - set(b[col])))
    return result


def property_frame(df: pd.DataFrame) -> pd.DataFrame:
    properties = df[["CMC", "WaterSolubility"]].copy()
    for column in ("CMC", "WaterSolubility"):
        properties[f"log10_{column}"] = np.log10(properties[column])
    # Compute six inexpensive 2D quantities once per represented graph, without cache reuse.
    functions = {"MolWt": Descriptors.MolWt, "MolLogP": Descriptors.MolLogP,
                 "TPSA": Descriptors.TPSA, "HBD": Descriptors.NumHDonors,
                 "HBA": Descriptors.NumHAcceptors, "Rings": Descriptors.RingCount}
    for column, prefix in (("contaminant_id", "contaminant"), ("active_structure_group", "active")):
        values = {s: {name: fn(Chem.MolFromSmiles(s)) for name, fn in functions.items()}
                  for s in df[column].unique()}
        for name in functions:
            properties[f"{prefix}_{name}"] = df[column].map(lambda s: values[s][name])
    return properties


def feature_audit(df: pd.DataFrame, properties: pd.DataFrame) -> pd.DataFrame:
    """Separate generic builder inputs, configured candidate spaces and EDA quantities."""
    from msr_qspr.preprocessing import build_predictor_matrix, REFERENCE_PREDICTORS
    initial = build_predictor_matrix(df)
    experiments = study_experiments()
    rows = []
    for column in dict.fromkeys([*df.columns, *properties.columns, *FIXED_MOLECULAR_PANEL,
                                 "Cont_*/Surf_* (Mordred candidate space)"]):
        molecular = column in FIXED_MOLECULAR_PANEL or column.startswith("Cont_*/")
        candidates = [e.name for e in experiments
                      if column in e.required_reference_properties
                      or (column in (e.descriptor_panel or ()))
                      or (molecular and e.predictor_family in {"molecular", "enriched"})]
        if column in {"MSR", "log10_MSR"}:
            role = "response or target transform"
        elif column in {"CMC", "WaterSolubility"}:
            role = "reference property"
        elif molecular:
            role = "molecular descriptor candidate"
        elif column in properties:
            role = "descriptive reference-property transform or RDKit descriptor"
        elif column.startswith("temperature"):
            role = "assay context"
        elif column in {"ContaminantType", "SurfactantType"}:
            role = "chemical class metadata"
        else:
            role = "identifier, molecular structure or experimental source attribution"
        rows.append({"column": column, "role": role,
                     "initial_builder_available": column in REFERENCE_PREDICTORS or molecular,
                     "initial_builder_includes": column in initial,
                     "primary_candidate_input": bool(candidates),
                     "configured_candidates": " | ".join(candidates),
                     "missing": int(df[column].isna().sum()) if column in df else
                                int(properties[column].isna().sum()) if column in properties else np.nan,
                     "unique": df[column].nunique() if column in df else
                               properties[column].nunique() if column in properties else np.nan})
    return pd.DataFrame(rows)


def missingness_table(df):
    """Unavailable values among applicable records; blanks in inapplicable fields are separate."""
    rows = []
    for column in df:
        applicable = pd.Series(True, index=df.index)
        if column == "temperature_value_1":
            applicable = df.temperature_mode.isin(["reported_C", "reported_K", "range_C"])
        elif column == "temperature_value_2":
            applicable = df.temperature_mode.eq("range_C")
        unavailable = df[column].isna()
        if column in {"source_doi", "source_location"}:
            unavailable |= df[column].astype(str).str.strip().isin(["0", ""])
        rows.append({"column": column, "missing": int((applicable & unavailable).sum()),
                     "not_applicable": int((~applicable).sum()), "applicable_n": int(applicable.sum()),
                     "unavailable_or_placeholder": int((applicable & unavailable).sum()),
                     "unique": df.loc[applicable, column].nunique()})
    return pd.DataFrame(rows).set_index("column")


def publication_labels(df):
    """Read bibliography alone; retain reference numbers to disambiguate author-year labels."""
    _, references = _read_workbook(["row_id"])
    labels = {}
    for key, group in df.groupby("publication_group"):
        numbers = sorted(group.reference_number.unique())
        text = references[numbers[0]]
        match = re.match(r"(.+?)\s+[A-Z]\..*?\((\d{4})\)", text)
        short = f"{match[1]} ({match[2]})" if match else f"Reference {numbers[0]}"
        labels[key] = short + " [" + ", ".join(map(str, numbers)) + "]"
    return labels


def generate_figures(df, properties, directory):
    with plt.rc_context():
        set_plot_style()
        _generate_figures(df, properties, directory)


def _generate_figures(df, properties, directory):
    fig, axes = plt.subplots(1, 2, figsize=(9, 3.7), sharex=True)
    for i, (ax, draw, title) in enumerate(zip(axes, (plot_msr_histogram, plot_msr_ecdf),
                                            ("Histogram", "Empirical cumulative distribution"))):
        draw(ax, df.MSR)
        ax.set_title(f"({chr(97 + i)}) {title}")
    fig.tight_layout()
    save_figure(fig, directory, "target_distribution")
    fig, axes = plt.subplots(1, 3, figsize=(11, 3.7))
    plot_msr_histogram(axes[0], df.MSR, log_scale=False)
    axes[0].set_title("(a) Histogram, MSR")
    for ax, column, title in zip(axes[1:], ("MSR", "log10_MSR"),
                                 ("(b) Normal Q–Q, MSR", r"(c) Normal Q–Q, $\log_{10}(\mathrm{MSR})$")):
        (theoretical, ordered), (slope, intercept, _) = stats.probplot(df[column], dist="norm")
        ax.scatter(theoretical, ordered, s=9, color=get_discrete_colors(6)[-1])
        ax.plot(theoretical, slope * theoretical + intercept, color="0.2", linestyle="--", linewidth=1.0)
        ax.set(xlabel="Standard-normal quantile", ylabel="MSR (mol/mol)" if column == "MSR" else LOG_LABEL,
               title=title)
    fig.tight_layout()
    save_figure(fig, directory, "target_diagnostics")
    save_figure(plot_msr_classes(df), directory, "chemical_classes")
    save_figure(plot_class_coverage(df), directory, "class_coverage")
    development, holdout = recorded_holdouts(df.drop(columns=["MSR", "log10_MSR"]))["unseen_surfactant"]
    save_figure(plot_partition_ecdf(df.MSR.iloc[development], df.MSR.iloc[holdout]), directory, "partition_ecdf")

    fig, axes = plt.subplots(1, 2, figsize=(12, 8), sharex=True)
    pubs = publication_labels(df)
    for ax, column, label in zip(axes, GROUPS, ("Publication", "Active surfactant structure")):
        table = group_summary(df, column).sort_values(["n", "group"], ascending=[False, True]).head(15).sort_values("q50")
        names = table.group.map(pubs) if column == GROUPS[0] else table.surfactant_labels
        for i, (_, row) in enumerate(table.iterrows()):
            ax.plot([row.q00, row.q100], [i, i], color=get_base_color(), linewidth=1.4)
            ax.scatter([row.q50], [i], color=get_base_color(), s=20)
        ax.set_yticks(range(len(table)))
        ax.set_yticklabels([fill(s, 30, break_long_words=False, break_on_hyphens=False) + f"\n(n = {n})"
                            for s, n in zip(names, table.n)], fontsize=8)
        ax.set(xlabel=LOG_LABEL, title=f"{label}: 15 largest groups")
    fig.suptitle("Lines: observed minimum–maximum; points: median (not confidence intervals)", fontsize=11)
    fig.tight_layout(rect=(0, 0, 1, .94))
    save_figure(fig, directory, "group_heterogeneity")

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))
    modes = {"reported_C": "Reported in °C", "reported_K": "Reported in K (converted to °C)",
             "range_C": "Reported range (midpoint)", "room_temperature": "Room temperature (qualitative)",
             "not_reported": "Not reported"}
    counts = df.temperature_mode.value_counts().reindex(modes, fill_value=0).sort_values(ascending=False, kind="stable")
    bars = axes[0].barh([modes[m] for m in counts.index], counts.values, color=get_base_color())
    axes[0].bar_label(bars, padding=3, fontsize=9)
    axes[0].set(xlabel="Observations", title="(a) Assay-temperature provenance", xlim=(0, counts.max() * 1.16))
    axes[0].invert_yaxis()
    coverage = df.groupby("ContaminantType").temperature_C.agg(["size", "count"])
    coverage = coverage.loc[((coverage["size"] - coverage["count"]) / coverage["size"]).sort_values(
        ascending=False, kind="stable").index]
    missing = coverage["size"] - coverage["count"]
    labels = [fill(class_label(c), 24) + f" (n = {n})" for c, n in coverage["size"].items()]
    bars = axes[1].barh(labels, missing / coverage["size"] * 100, color=get_base_color())
    axes[1].bar_label(bars, labels=[f"{m}/{n}" for m, n in zip(missing, coverage["size"])], padding=3, fontsize=9)
    axes[1].set(xlabel="Unavailable numerical temperature (%)", xlim=(0, 100),
                title="(b) Coverage by contaminant class")
    axes[1].invert_yaxis()
    fig.tight_layout()
    save_figure(fig, directory, "condition_coverage")

    fig, axes = plt.subplots(1, 3, figsize=(14, 5.8), gridspec_kw={"width_ratios": [1, 1, 1.5]})
    for ax, col, grouping, label, title in zip(
            axes[:2], ["log10_CMC", "log10_WaterSolubility"], ["SurfactantType", "ContaminantType"],
            [r"$\log_{10}[\mathrm{CMC}/(\mathrm{mol\,L^{-1}})]$", r"$\log_{10}[S_\mathrm{w}/(\mathrm{g\,L^{-1}})]$"],
            ["(a) CMC by surfactant class", "(b) Solubility by contaminant class"]):
        groups = sorted(df.groupby(grouping).groups.items(), key=lambda item: -len(item[1]))
        styles = ("-", "--", ":", "-.", (0, (5, 1.5)), (0, (3, 1, 1, 1, 1, 1)))
        for color, style, (name, idx) in zip(get_discrete_colors(len(groups))[::-1], styles, groups):
            sns.ecdfplot(x=properties.loc[idx, col], ax=ax, linestyle=style, linewidth=1.6,
                         label=fill(class_label(name), 22), color=color)
        ax.set(xlabel=label, ylabel="Cumulative fraction", title=title, ylim=(0, 1.02))
        ax.legend(loc="upper left", fontsize=8, frameon=True)
    selected = properties[["log10_CMC", "log10_WaterSolubility", "active_MolWt", "contaminant_MolWt"]].copy()
    selected["log10_MSR"] = df.log10_MSR
    labels = [r"$\log_{10}\,$CMC", r"$\log_{10}\,S_\mathrm{w}$", "Active graph MW", "Contaminant MW",
              r"$\log_{10}\,$MSR"]
    sns.heatmap(selected.corr(method="spearman"), vmin=-1, vmax=1, cmap="vlag", center=0, annot=True,
                fmt=".2f", xticklabels=labels, yticklabels=labels, ax=axes[2], cbar=False,
                annot_kws={"size": 9}, square=True)
    axes[2].set_title("(c) Descriptive Spearman ρ")
    axes[2].tick_params(axis="x", labelsize=9, rotation=45)
    axes[2].tick_params(axis="y", labelsize=9, rotation=0)
    plt.setp(axes[2].get_xticklabels(), ha="right")
    fig.tight_layout()
    save_figure(fig, directory, "physicochemical_properties")


def run(output: Path):
    before = verify_scientific_inputs()
    df = load_corpus()
    output.mkdir(parents=True, exist_ok=True)
    tables = {}
    tables["target_summary"] = pd.DataFrame({c: summarize(df[c]) for c in ("MSR", "log10_MSR")}).T
    extremes = []
    sensitivity = []
    for column in ("MSR", "log10_MSR"):
        y = df[column]
        q1, q3 = y.quantile([.25, .75])
        low, high = q1 - 1.5 * (q3 - q1), q3 + 1.5 * (q3 - q1)
        flagged = df.loc[y.lt(low) | y.gt(high), ["row_id", "source_ref", "Surfactant", "Contaminant", "MSR", "log10_MSR"]].copy()
        flagged["flag_scale"] = column
        extremes.append(flagged)
        for label, values in (("complete", y), ("omit_largest_one_descriptive_only", y.drop(y.idxmax())),
                              ("central_98_percent_descriptive_only", y[y.between(*y.quantile([.01, .99]))])):
            sensitivity.append({"scale": column, "summary_scope": label, **summarize(values)})
    tables["extreme_observations"] = pd.concat(extremes, ignore_index=True)
    tables["tail_sensitivity"] = pd.DataFrame(sensitivity)
    for column in (*GROUPS, "surfactant_id", "contaminant_id", "Surfactant", "SurfactantType", "ContaminantType",
                   "chemical_cell", "temperature_mode"):
        tables[column] = group_summary(df, column)
    repeated_pairs = group_summary(df, "active_pair")
    tables["repeated_active_pairs"] = repeated_pairs.loc[repeated_pairs.n.gt(1)].reset_index(drop=True)
    labels = publication_labels(df)
    tables["publication_group"]["display_label"] = tables["publication_group"].group.map(labels)
    references = df.groupby("publication_group").agg(
        reference_numbers=("reference_number", lambda x: " | ".join(map(str, sorted(x.unique())))),
        source_references=("source_ref", lambda x: " | ".join(sorted(x.unique()))))
    tables["publication_group"] = tables["publication_group"].join(references, on="group")
    tables["missingness"] = missingness_table(df.drop(columns=["log10_MSR"]))
    tables["temperature_numeric"] = pd.DataFrame([
        {"scope": "reported_numeric_and_range_midpoints", **summarize(df.temperature_C)},
        {"scope": "reported_numeric_setpoints", **summarize(df.loc[df.temperature_mode.isin(["reported_C", "reported_K"]), "temperature_C"])}])
    properties = property_frame(df)
    tables["property_summary"] = pd.DataFrame({c: summarize(properties[c]) for c in properties}).T
    corr_frame = properties.assign(log10_MSR=df.log10_MSR)
    for method in ("pearson", "spearman"):
        tables[f"property_{method}"] = corr_frame.corr(method=method)
    problem = []
    for col in properties:
        counts = properties[col].value_counts(normalize=True)
        problem.append({"column": col, "constant": len(counts) <= 1,
                        "largest_value_share": counts.iloc[0], "near_constant_95pct": counts.iloc[0] >= .95})
    tables["feature_variation"] = pd.DataFrame(problem)
    duplicates = []
    numeric = df.drop(columns=["log10_MSR"]).select_dtypes(include=np.number)
    for i, col in enumerate(numeric):
        for other in numeric.columns[i + 1:]:
            if numeric[col].equals(numeric[other]):
                duplicates.append({"column": col, "identical_to": other})
    tables["duplicate_numeric_fields"] = pd.DataFrame(duplicates, columns=["column", "identical_to"]).reset_index(drop=True)
    tables["feature_roles"] = feature_audit(df, properties)
    tables["property_by_class"] = pd.DataFrame([
        {"class_axis": axis, "class": key, "property": col, **summarize(properties.loc[idx, col])}
        for axis in ("SurfactantType", "ContaminantType") for key, idx in df.groupby(axis).groups.items()
        for col in ("CMC", "WaterSolubility", "active_MolWt", "contaminant_MolWt")])
    tables["group_class_counts"] = pd.concat([
        df.groupby([column, "SurfactantType", "ContaminantType"]).size().rename("n").reset_index()
        .rename(columns={column: "group"}).assign(grouping=column) for column in GROUPS], ignore_index=True)
    tables["temperature_missingness"] = pd.concat([
        df.groupby(column).temperature_C.agg(["size", "count"]).assign(axis=column)
        .reset_index().rename(columns={column: "group"}) for column in ("source_ref", "SurfactantType", "ContaminantType")])
    tables["temperature_missingness"]["missing"] = tables["temperature_missingness"]["size"] - tables["temperature_missingness"]["count"]
    metadata = df.drop(columns=["MSR", "log10_MSR"])
    holdouts = recorded_holdouts(metadata)
    tables["holdout_diagnostics"] = pd.DataFrame([
        {"scenario": name, **partition_diagnostics(metadata, a, b, GROUPS[1])}
        for name, (a, b) in holdouts.items()])
    tables["holdout_class_counts"] = pd.DataFrame([
        {"scenario": name, "partition": role, "axis": column, "class": key, "n": n}
        for name, indices in holdouts.items() for role, idx in zip(("development", "holdout"), indices)
        for column in ("ContaminantType", "SurfactantType", "chemical_cell")
        for key, n in metadata.iloc[idx][column].value_counts().items()])
    for name, table in tables.items():
        table.to_csv(output / f"{name}.csv", index=not isinstance(table.index, pd.RangeIndex))
    generate_figures(df, properties, output / "figures")
    after = verify_scientific_inputs()
    if before != after:
        raise ValueError("Frozen input signatures changed during characterization.")
    record = {
        "scope": "Descriptive MSR dataset characterization",
        "table_rows": {name: len(table) for name, table in tables.items()},
        "observations": len(df), "publications": df.publication_group.nunique(),
        "full_structures": df.surfactant_id.nunique(), "active_structures": df.active_structure_group.nunique(),
        "contaminant_structures": df.contaminant_id.nunique(), "active_pairs": df.active_pair.nunique(),
        "orders_of_magnitude": float(np.ptp(df.log10_MSR)),
        "between_group_variance_fraction": {c: variance_fraction(df, c) for c in (*GROUPS, "ContaminantType", "SurfactantType")},
        "observations_without_doi": int(df.source_doi.eq("").sum()),
        "MSR_magnitude_counts": {"below_0.001": int(df.MSR.lt(.001).sum()),
                                 "above_1": int(df.MSR.gt(1).sum()), "above_10": int(df.MSR.gt(10).sum())},
        "freeze_sha256_before_and_after": after,
        "analysis_code_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "plotting_code_sha256": hashlib.sha256((ROOT / "msr_qspr/plotting.py").read_bytes()).hexdigest(),
        "grouping_code_sha256": hashlib.sha256((ROOT / "msr_qspr/validation.py").read_bytes()).hexdigest(),
        "versions": {**{name: importlib.metadata.version(name) for name in ("numpy", "pandas", "scipy", "scikit-learn", "matplotlib", "seaborn")},
                     "rdkit": rdBase.rdkitVersion},
    }
    summary = tables["target_summary"]
    compact_stats = [
        f"{scale}: median {summary.loc[scale, 'q50']:.4g}; IQR [{summary.loc[scale, 'q25']:.4g}, "
        f"{summary.loc[scale, 'q75']:.4g}]; range [{summary.loc[scale, 'q00']:.4g}, {summary.loc[scale, 'q100']:.4g}]"
        for scale in ("MSR", "log10_MSR")]
    corpus_counts = {"publications": df.publication_group.nunique(),
                     "active_structures": df.active_structure_group.nunique(),
                     "contaminant_structures": df.contaminant_id.nunique()}
    figure_inventory = [f"figures/{name}.{suffix}" for name in FIGURE_CAPTIONS for suffix in ("png", "pdf")]
    generated = [f"{name}.csv" for name in tables] + figure_inventory + ["provenance.json", "report.txt"]
    record["generated_files"] = generated
    record["figure_captions"] = FIGURE_CAPTIONS
    # Every output is signed with one identity of the inputs, code and software that produced it.
    record["identity"] = fingerprint({key: record[key] for key in (
        "freeze_sha256_before_and_after", "analysis_code_sha256", "plotting_code_sha256",
        "grouping_code_sha256", "versions")})
    (output / "provenance.json").write_text(json.dumps(record, indent=2) + "\n")
    status = [
        "MSR exploratory characterization",
        f"Dataset: {len(df)} observations.",
        "Groups: " + "; ".join(f"{n} {key.replace('_', ' ')}" for key, n in corpus_counts.items()) + ".",
        f"Chemical coverage: {record['full_structures']} full surfactant structures; "
        f"{df.ContaminantType.nunique()} contaminant classes; {df.SurfactantType.nunique()} surfactant classes.",
        "Target statistics (MSR in mol/mol; log10 relative to 1 mol/mol):",
        *compact_stats,
        "Integrity: frozen workbook checksum unchanged (PASS).",
        f"Generated {len(tables)} CSV tables and {len(FIGURE_CAPTIONS)} figure pairs (PNG at 300 dpi + embedded-font PDF), "
        "plus report.txt and provenance.json.",
        f"Output directory: {output}",
    ]
    details = [
        "Descriptive statistics",
        "n counts observations within a group or available values. Group-summary moments and quantiles "
        "describe log10(MSR); MSR_min/median/max are in mol/mol. Reference-property summaries and graph "
        "descriptors are observation-weighted. CMC is in mol/L, aqueous solubility in g/L, graph molecular "
        "weight in g/mol and numerical assay temperature in °C.",
        "Numerical assay-temperature coverage treats qualitative room temperature as unknown. Range bounds "
        "apply only to reported ranges. Missingness distinguishes unavailable applicable values from "
        "non-applicable records.",
        "The descriptive property panel comprises reference properties, their transforms and six RDKit "
        "quantities per molecular graph, rather than the complete Mordred predictor space.",
        "Scientific interpretation and limitations",
        f"MSR spans {record['orders_of_magnitude']:.2f} orders of magnitude. Raw-scale mean and SD "
        "are tail-sensitive; log-scale median and IQR summarize the central multiplicative variation. "
        "Tail-sensitivity rows compare the full distribution with trimmed subsets.",
        "Violin widths scale with observation counts; inner boxes show medians and interquartile ranges. "
        "Densities are bounded by observed extremes. PAHs denotes polycyclic aromatic hydrocarbons; "
        "BTEX denotes benzene, toluene, ethylbenzene and xylenes.",
        "Observed group minimum–median–maximum ranges are not confidence intervals. Class/group differences and "
        "correlations are descriptive, depend on composition and repeated chemical/source identities, and establish "
        "neither causality nor independent predictive relevance. CMC reference conditions are not MSR assay conditions.",
        "Figure captions:\n" + "\n".join(
            f"- figures/{name}.png and .pdf: {caption}" for name, caption in FIGURE_CAPTIONS.items()),
    ]
    (output / "report.txt").write_text("\n\n".join(status + details) + "\n")
    for name in generated:
        record_cache(output / name, record["identity"])
    return {"stage": "characterization", "output": str(output), "identity": record["identity"],
            "observations": record["observations"], "tables": len(tables), "figures": len(FIGURE_CAPTIONS)}
