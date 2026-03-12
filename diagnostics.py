from typing import List, Dict, Any, Optional

import io
import json
import warnings
import contextlib

import numpy as np
import pandas as pd
import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.markers import MarkerStyle
import seaborn as sns

from sklearn.base import clone
from sklearn.metrics import r2_score
from sklearn.model_selection import RepeatedStratifiedKFold, StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import StandardScaler

from config import (
    SEED,
    N_BINS,
    FINAL_MODELS_DIR,
    DIAG_YRAND_DIR,
    DIAG_SHAP_DIR,
    DIAG_AD_DIR,
    N_PERM,
    AD_ERROR_QUANTILE
)
from model_selection import (
    _make_estimators,
    _build_model_pipeline_for_fold,
    _safe_name,
    _rmse
)
from plot_style import get_shap_cmap, set_plot_style, get_discrete_colors



# Y-randomization
def run_y_randomization(
    X_cv: pd.DataFrame,
    y_cv_log: pd.Series,
    bins_cv: pd.Series,
    categorical_cols: List[str],
    log10_cols: List[str],
    final_features: List[str],
    best_model_name: str,
    n_perm: int = N_PERM,
    force_recompute: bool = False,
) -> pd.DataFrame:
    """
    Run Y-randomization for the chosen final model on the CV set.

    Uses:
      - fixed preprocessing + feature set (final_features)
      - fixed best hyperparameters
      - repeated stratified CV on log10(MSR)

    Returns a DataFrame with one row for the real target and one row per permutation.
    """
    DIAG_YRAND_DIR.mkdir(parents=True, exist_ok=True)
    set_plot_style()

    mslug = _safe_name(best_model_name)
    results_path = DIAG_YRAND_DIR / f"{mslug}_y_randomization_results.csv"

    if results_path.exists() and not force_recompute:
        df = pd.read_csv(results_path)
        print(f"[Y-rand] Loaded existing results from {results_path.name}")

        real_vals = df.loc[df["kind"] == "real", "R2"].to_numpy(dtype=float)
        perm_vals = df.loc[df["kind"] == "perm", "R2"].to_numpy(dtype=float)

        real_med = float(np.median(real_vals)) if real_vals.size else float("nan")
        perm_med = float(np.median(perm_vals)) if perm_vals.size else float("nan")
        perm_std = float(np.std(perm_vals, ddof=1)) if perm_vals.size > 1 else float ("nan")

        print(f"[Y-rand] Median R² (real y)     = {real_med:.3f}")
        print(f"[Y-rand] Median R² (permuted y) = {perm_med:.3f}, std = {perm_std:.3f}")

        return df

    estimators = _make_estimators()
    if best_model_name not in estimators:
        raise ValueError(f"Unknown best_model_name: {best_model_name}")

    base_est = estimators[best_model_name]

    params_path = FINAL_MODELS_DIR / f"{mslug}_final_params.json"
    if not params_path.exists():
        raise FileNotFoundError(
            f"Missing final params file for {best_model_name}: {params_path}"
        )

    with open(params_path, "r", encoding="utf-8") as f:
        best_params: Dict[str, Any] = json.load(f)

    model_pipe = _build_model_pipeline_for_fold(
        estimator=base_est,
        selected_features=final_features,
        categorical_cols=categorical_cols,
        log10_cols=log10_cols,
    )
    model_pipe.set_params(**best_params)

    strata = bins_cv
    min_per_class = pd.Series(strata).value_counts().min()
    n_splits_eff = max(2, min(5, int(min_per_class)))

    inner_cv = RepeatedStratifiedKFold(
        n_splits=n_splits_eff,
        n_repeats=3,
        random_state=SEED + 1234,
    )
    inner_splits = list(inner_cv.split(X_cv, strata.to_numpy()))

    def cv_median_r2_for_target(y_target: pd.Series) -> float:
        """Compute median R² over inner splits for a given target vector."""
        r2_vals: List[float] = []

        for itr, iva in inner_splits:
            X_tr = X_cv.iloc[itr]
            X_va = X_cv.iloc[iva]
            y_tr = y_target.iloc[itr]
            y_va = y_target.iloc[iva]

            m = clone(model_pipe).fit(X_tr, y_tr)
            pred = m.predict(X_va)
            r2_vals.append(float(r2_score(y_va.to_numpy(), pred)))

        return float(np.median(r2_vals))

    print("\n[Y-rand] Computing baseline CV R² with real targets...")
    r2_real = cv_median_r2_for_target(y_cv_log)
    print(f"[Y-rand] Baseline median R² (real y) = {r2_real:.3f}")

    rng = np.random.default_rng(SEED + 5678)
    rows: List[Dict[str, Any]] = []
    rows.append({"kind": "real", "perm_id": -1, "R2": r2_real})

    print(f"[Y-rand] Running {n_perm} permutations...")
    for i in range(n_perm):
        perm_idx = rng.permutation(len(y_cv_log))
        y_perm_vals = y_cv_log.to_numpy()[perm_idx]
        y_perm = pd.Series(
            y_perm_vals,
            index=y_cv_log.index,
            name=y_cv_log.name,
        )

        r2_perm = cv_median_r2_for_target(y_perm)
        rows.append({"kind": "perm", "perm_id": i, "R2": r2_perm})

    df_res = pd.DataFrame(rows)
    df_res.to_csv(results_path, index=False)
    print(f"[Y-rand] Saved results → {results_path.name}")

    # Histogram of permuted R² with real R² marked
    perm_vals = df_res.loc[df_res["kind"] == "perm", "R2"].to_numpy(dtype=float)

    fig_path = DIAG_YRAND_DIR / f"{mslug}_y_randomization_hist.png"
    plt.figure(figsize=(4.8, 3.6))
    plt.hist(perm_vals, bins=15, alpha=0.8)
    plt.axvline(r2_real, linestyle="--", linewidth=1.2)
    plt.xlabel("Median R² (permuted targets)")
    plt.ylabel("Count")
    plt.title(f"Y-randomization – {best_model_name}")
    plt.tight_layout()
    plt.savefig(str(fig_path), dpi=300)
    plt.close()
    print(f"[Y-rand] Saved histogram → {fig_path.name}")

    return df_res


# SHAP
def run_shap(
    best_model_name: str,
    X_data: pd.DataFrame,
    final_features: List[str],
    max_display: int = 11,
) -> pd.DataFrame:
    """
    Compute SHAP for the final fitted model.

    Loads the fitted pipeline from disk, computes SHAP values on X_data
    in the final feature space, and saves:
      - feature-level mean |SHAP| to CSV
      - bar summary plot
      - beeswarm summary plot

    Returns the importance DataFrame sorted by mean |SHAP|.
    """

    DIAG_SHAP_DIR.mkdir(parents=True, exist_ok=True)
    set_plot_style()

    mslug = _safe_name(best_model_name)
    model_path = FINAL_MODELS_DIR / f"{mslug}_final_model.joblib"
    if not model_path.exists():
        raise FileNotFoundError(
            f"Missing final model file for {best_model_name}: {model_path}"
        )

    print(f"\n[SHAP] Loading final model from {model_path.name}...")
    final_pipe = joblib.load(model_path)

    if "model" not in final_pipe.named_steps:
        raise ValueError("Loaded pipeline has no 'model' step.")

    preproc_pipe = Pipeline(final_pipe.steps[:-1])
    model = final_pipe.named_steps["model"]

    print("[SHAP] Transforming input data with fitted preprocessing...")
    X_pre = preproc_pipe.transform(X_data)
    if not isinstance(X_pre, pd.DataFrame):
        X_pre = pd.DataFrame(X_pre, columns=final_features)

    shap_cmap = get_shap_cmap()

    silent_out = io.StringIO()
    silent_err = io.StringIO()

    print("[SHAP] Building SHAP explainer...")
    with contextlib.redirect_stdout(silent_out), contextlib.redirect_stderr(silent_err):
        import shap
        explainer = shap.Explainer(model, X_pre)
        shap_values = explainer(X_pre, check_additivity=False)

    values = shap_values.values if hasattr(shap_values, "values") else np.asarray(shap_values)
    mean_abs = np.mean(np.abs(values), axis=0)

    importance_df = (
        pd.DataFrame(
            {
                "feature": X_pre.columns,
                "mean_abs_shap": mean_abs,
            }
        )
        .sort_values("mean_abs_shap", ascending=False)
        .reset_index(drop=True)
    )

    imp_path = DIAG_SHAP_DIR / f"{mslug}_shap_importance.csv"
    importance_df.to_csv(imp_path, index=False)
    print(f"[SHAP] Saved global importance → {imp_path.name}")

    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=UserWarning)

        # Bar plot
        bar_path = DIAG_SHAP_DIR / f"{mslug}_shap_bar.pdf"
        silent_out = io.StringIO()
        silent_err = io.StringIO()

        with contextlib.redirect_stdout(silent_out), contextlib.redirect_stderr(silent_err):
            plt.figure(figsize=(6.0, 4.0))

            shap.plots.bar(
                shap_values,
                max_display=max_display,
                show=False,
            )

            ax = plt.gca()

            palette = get_discrete_colors(max_display)
            bar_color = palette[-1]

            for patch in ax.patches:
                patch.set_facecolor(bar_color)

            if ax.patches:
                x_max = max(p.get_width() for p in ax.patches)
                ax.set_xlim(0, x_max * 1.10)

            for txt in ax.texts:
                txt.remove()

            for patch in ax.patches:
                value = patch.get_width()
                y = patch.get_y() + patch.get_height() / 2.0
                x_text = value * 0.98

                ax.text(
                    x_text,
                    y,
                    f"{value:.3f}",
                    va="center",
                    ha="right",
                    color="white",
                    fontsize=9,
                    fontweight="bold",
                )

            ax.set_xlabel("mean |SHAP value|")
            ax.set_ylabel("")
            plt.tight_layout()

        plt.savefig(str(bar_path))
        plt.close()
        print(f"[SHAP] Saved bar summary plot → {bar_path.name}")

        # Beeswarm plot
        beeswarm_path = DIAG_SHAP_DIR / f"{mslug}_shap_beeswarm.pdf"
        with contextlib.redirect_stdout(silent_out), contextlib.redirect_stderr(silent_err):
            shap.summary_plot(
                shap_values,
                X_pre,
                max_display=max_display,
                show=False,
                cmap=shap_cmap,
            )

            ax =  plt.gca()
            ax.set_xlabel("SHAP value")
        plt.gcf().savefig(str(beeswarm_path))
        plt.close()
        print(f"[SHAP] Saved beeswarm plot → {beeswarm_path.name}")

    return importance_df

def run_applicability_domain(
    best_model_name: str,
    X_cv: pd.DataFrame,
    y_cv_log: pd.Series,
    X_holdout: pd.DataFrame,
    y_holdout_log: pd.Series,
    k: int = 7,
    quantiles: Optional[np.ndarray] = None,
    q_ref: float = 0.95,
    force_recompute: bool = False,
) -> Dict[str, pd.DataFrame]:
    """
    Run kNN-based applicability domain diagnostics for the final model.

    The AD combines:
      - structural similarity in the preprocessed + scaled feature space
      - range of log10(MSR) observed in CV
      - local neighbourhood error learned from CV out-of-fold residuals

    For each structural quantile q it reports coverage and performance
    on the holdout set inside the AD.

    Saves:
      - CSV with quantile / coverage / R² / RMSE on holdout
      - CSV with distance vs |error| on holdout
      - two figures with the corresponding plots
    """
    if quantiles is None:
        quantiles = np.linspace(0.90, 0.99, 10)
    set_plot_style()

    mslug = _safe_name(best_model_name)

    scan_path = DIAG_AD_DIR / f"{mslug}_ad_quantiles.csv"
    dist_err_path = DIAG_AD_DIR / f"{mslug}_ad_dist_error.csv"
    fig_scan_path = DIAG_AD_DIR / f"{mslug}_ad_quantiles.pdf"
    fig_dist_path = DIAG_AD_DIR / f"{mslug}_ad_dist_error.pdf"

    if (
        scan_path.exists()
        and dist_err_path.exists()
        and fig_scan_path.exists()
        and fig_dist_path.exists()
        and not force_recompute
    ):
        scan_df = pd.read_csv(scan_path)
        dist_err_df = pd.read_csv(dist_err_path)
        print(f"[AD] Loaded existing diagnostics from {DIAG_AD_DIR.name}")
        return {"scan": scan_df, "dist_error": dist_err_df}

    model_path = FINAL_MODELS_DIR / f"{mslug}_final_model.joblib"
    if not model_path.exists():
        raise FileNotFoundError(
            f"Missing final model file for {best_model_name}: {model_path}"
        )

    print(f"\n[AD] Loading final model from {model_path.name}...")
    final_pipe = joblib.load(model_path)

    if "model" not in final_pipe.named_steps:
        raise ValueError("Loaded pipeline has no 'model' step.")

    # Preprocessing only (same as for final model)
    preproc_pipe = Pipeline(final_pipe.steps[:-1])
    model = final_pipe.named_steps["model"]

    print("[AD] Transforming CV and holdout features...")
    X_cv_pre = preproc_pipe.transform(X_cv)
    X_hold_pre = preproc_pipe.transform(X_holdout)

    X_cv_pre = np.asarray(X_cv_pre, dtype=float)
    X_hold_pre = np.asarray(X_hold_pre, dtype=float)
    y_cv_arr = y_cv_log.to_numpy(dtype=float)
    y_hold_arr = y_holdout_log.to_numpy(dtype=float)

    # Out-of-fold errors on CV to estimate local neighbourhood error
    print("[AD] Computing out-of-fold CV errors for local error AD...")

    strata = pd.qcut(
        y_cv_log,
        q=min(N_BINS, max(2, len(y_cv_log) // 5)),
        labels=False,
        duplicates="drop",
    )

    n_splits_eff = int(min(5, strata.value_counts().min()))
    skf = StratifiedKFold(
        n_splits=max(2, n_splits_eff),
        shuffle=True,
        random_state=SEED + 202,
    )

    y_cv_pred = np.empty_like(y_cv_arr)
    for itr, iva in skf.split(X_cv_pre, strata.to_numpy()):
        m = clone(model).fit(X_cv_pre[itr], y_cv_arr[itr])
        y_cv_pred[iva] = m.predict(X_cv_pre[iva])

    abs_err_cv = np.abs(y_cv_arr - y_cv_pred)

    # kNN in scaled preprocessed space
    print("[AD] Fitting kNN")

    scaler = StandardScaler().fit(X_cv_pre)
    X_cv_scaled = scaler.transform(X_cv_pre)
    X_hold_scaled = scaler.transform(X_hold_pre)

    nn = NearestNeighbors(n_neighbors=k, metric="euclidean")
    nn.fit(X_cv_scaled)

    dist_train, idx_train = nn.kneighbors(X_cv_scaled)
    mean_dist_train = dist_train.mean(axis=1)

    # Local error of each CV point: average abs-error of its neighbours
    local_err_train = abs_err_cv[idx_train].mean(axis=1)
    err_threshold = float(np.quantile(local_err_train, AD_ERROR_QUANTILE))

    ad_params = {
        "nn": nn,
        "mean_dist_train": mean_dist_train,
        "local_err_train": local_err_train,
        "err_threshold": err_threshold,
        "y_min": float(np.min(y_cv_arr)),
        "y_max": float(np.max(y_cv_arr)),
    }

    # Predictions on holdout and AD scan over quantiles
    print("[AD] Predicting on holdout...")
    y_pred_hold = final_pipe.predict(X_holdout)
    y_pred_hold = np.asarray(y_pred_hold, dtype=float)

    print("[AD] Scanning structural quantiles with fixed error threshold...")
    dist_hold, idx_hold = nn.kneighbors(X_hold_scaled)
    mean_dist_hold = dist_hold.mean(axis=1)
    local_err_hold = abs_err_cv[idx_hold].mean(axis=1)

    records: List[Dict[str, Any]] = []

    for q in quantiles:
        dist_threshold = float(np.quantile(ad_params["mean_dist_train"], float(q)))

        inside_struct = mean_dist_hold <= dist_threshold
        inside_y = (y_pred_hold >= ad_params["y_min"]) & (
            y_pred_hold <= ad_params["y_max"]
        )
        inside_err = local_err_hold <= ad_params["err_threshold"]

        inside_overall = inside_struct & inside_y & inside_err
        coverage = float(inside_overall.mean())

        if inside_overall.any():
            r2 = float(r2_score(y_hold_arr[inside_overall], y_pred_hold[inside_overall]))
            rmse = float(_rmse(y_hold_arr[inside_overall], y_pred_hold[inside_overall]))
        else:
            r2 = np.nan
            rmse = np.nan

        records.append(
            {
                "quantile": float(q),
                "coverage": coverage,
                "R2_in_AD": r2,
                "RMSE_in_AD": rmse,
            }
        )

    scan_df = pd.DataFrame.from_records(records).sort_values("quantile")
    scan_df.to_csv(scan_path, index=False)
    print(f"[AD] Saved quantile scan → {scan_path.name}")

    # Distance vs |error| on holdout
    print("[AD] Preparing distance vs |error| data...")
    abs_error_hold = np.abs(y_hold_arr - y_pred_hold)

    dist_err_df = pd.DataFrame(
        {
            "mean_dist": mean_dist_hold,
            "abs_error": abs_error_hold,
            "local_error": local_err_hold,
        }
    )
    dist_err_df.to_csv(dist_err_path, index=False)
    print(f"[AD] Saved distance vs error data → {dist_err_path.name}")

    # Global R² on holdout (no AD) – used as a baseline
    r2_all = float(r2_score(y_hold_arr, y_pred_hold))

    # If q_ref is not provided, choose it by a simple rule:
    # smallest quantile with coverage >= cov_target and R²_in_AD >= R²_all
    if q_ref is None:
        cov_target = 0.75  # desired minimum coverage
        eligible = scan_df[
            (scan_df["coverage"] >= cov_target) &
            (scan_df["R2_in_AD"] >= r2_all)
        ].sort_values("quantile")

        if not eligible.empty:
            q_ref = float(eligible["quantile"].iloc[0])
        else:
            # Fallback: quantile that maximizes R²_in_AD
            q_ref = float(
                scan_df.loc[scan_df["R2_in_AD"].idxmax(), "quantile"]
            )

        print(f"[AD] Auto-selected q_ref={q_ref:.3f}")

    FIGSIZE = (4.8, 3.6)
    AX_POS = [0.15, 0.20, 0.70, 0.68]

    # --------- Plot: coverage + R² inside AD vs quantile ---------
    palette = get_discrete_colors(3)
    color_cov = palette[-1]
    color_r2 = palette[0]

    fig = plt.figure(figsize=FIGSIZE)
    ax1 = fig.add_axes(AX_POS)

    ax1.plot(
        scan_df["quantile"],
        scan_df["coverage"],
        marker="o",
        linestyle="-",
        color=color_cov,
        label="Coverage",
    )
    ax1.set_xlabel("Structural quantile")
    ax1.set_ylabel("Coverage")
    ax1.set_ylim(0.6, 1.02) 
    ax1.set_yticks([0.6, 0.7, 0.8, 0.9, 1.0])

    ax2 = ax1.twinx()
    ax2.plot(
        scan_df["quantile"],
        scan_df["R2_in_AD"],
        marker="s",
        linestyle="--",
        color=color_r2,
        label="R² in AD",
    )
    ax2.set_ylabel("R² inside AD")
    ax2.set_ylim(0.6, 1.02)  
    ax2.set_yticks([0.6, 0.7, 0.8, 0.9, 1.0])

    # Horizontal line: global holdout R²
    ax2.axhline(
        r2_all,
        linestyle="--",
        linewidth=1.0,
        color="gray",
        alpha=0.8,
        label="R² all",
    )

    mask_q = np.isclose(scan_df["quantile"].to_numpy(dtype=float), float(q_ref), rtol=0, atol=1e-8)

    if mask_q.any():
        x_sel = scan_df.loc[mask_q, "quantile"].to_numpy(dtype=float)

        y_cov_sel = scan_df.loc[mask_q, "coverage"].to_numpy(dtype=float)
        ax1.scatter(
            x_sel,
            y_cov_sel,
            s=50,                      
            facecolor=color_cov,
            edgecolor="black",
            linewidth=0.7,
            zorder=4,
            marker=MarkerStyle("o"),
            label="_nolegend_",         
        )

        y_r2_sel = scan_df.loc[mask_q, "R2_in_AD"].to_numpy(dtype=float)
        ax2.scatter(
            x_sel,
            y_r2_sel,
            s=50,
            facecolor=color_r2,
            edgecolor="black",
            linewidth=0.7,
            zorder=4,
            marker=MarkerStyle("s"),
            label="_nolegend_",
        )


    ax1.grid(True, alpha=0.3)

    lines_1, labels_1 = ax1.get_legend_handles_labels()
    lines_2, labels_2 = ax2.get_legend_handles_labels()
    handles = lines_1 + lines_2
    labels = labels_1 + labels_2

    ax1.legend(
        handles,
        labels,
        loc="lower center",              
        bbox_to_anchor=(0.5, 0.98),      
        bbox_transform=ax1.transAxes,    
        ncol=3,
        frameon=False,
    )

    for ax_ in (ax1, ax2):
        ax_.tick_params(
            left=False,
            right=False,
        )

    fig.savefig(str(fig_scan_path))
    plt.close(fig)
    print(f"[AD] Saved quantile figure → {fig_scan_path.name}")

    # Distance vs |error| 
    palette2 = get_discrete_colors(5)
    color_in = palette2[-1]
    color_out = palette2[1]

    dist_thresh_ref = float(np.quantile(ad_params["mean_dist_train"], q_ref))
    inside_ref = dist_err_df["mean_dist"] <= dist_thresh_ref

    fig = plt.figure(figsize=FIGSIZE)
    ax = fig.add_axes(AX_POS)

    ax.scatter(
        dist_err_df.loc[inside_ref, "mean_dist"],
        dist_err_df.loc[inside_ref, "abs_error"],
        s=28,
        alpha=0.8,
        edgecolor="black",
        linewidth=0.4,
        color=color_in,
        label=f"Inside AD (q={q_ref:.2f})",
    )

    if (~inside_ref).any():
        ax.scatter(
            dist_err_df.loc[~inside_ref, "mean_dist"],
            dist_err_df.loc[~inside_ref, "abs_error"],
            s=32,
            alpha=0.8,
            edgecolor="black",
            linewidth=0.6,
            facecolors="none",
            color=color_out,
            label="Outside AD",
        )

    ax.axvline(
        dist_thresh_ref,
        linestyle="--",
        linewidth=1.0,
        color="gray",
        alpha=0.8,
    )

    ax.set_xlabel("Mean kNN distance")
    ax.set_ylabel("|Error| in log10(MSR)")
    ax.grid(True, alpha=0.3)

    ax.legend(
        loc="lower center",              
        bbox_to_anchor=(0.5, 0.98),      
        bbox_transform=ax.transAxes,    
        ncol=2,
        frameon=False,        
    )

    fig.savefig(str(fig_dist_path))
    plt.close(fig)
    print(f"[AD] Saved distance vs error figure → {fig_dist_path.name}")

    return {"scan": scan_df, "dist_error": dist_err_df}
