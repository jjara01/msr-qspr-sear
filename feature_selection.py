from typing import List, Tuple, Dict
from collections import Counter
import json

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


from sklearn.base import clone
from sklearn.ensemble import RandomForestRegressor
from sklearn.feature_selection import SelectFromModel
from sklearn.metrics import r2_score
from sklearn.model_selection import StratifiedKFold, RepeatedStratifiedKFold
from sklearn.pipeline import Pipeline



from config import (
    SEED,
    FEAT_STABILITY_DIR,
    FEAT_K_CURVES_DIR,
    FEAT_SUMMARY_DIR,
    FEAT_FIGURES_DIR,
    OUTER_N_SPLITS,
    INNER_MAX_SPLITS,
    INNER_REPEATS,
    B_REPEATS,
    MAX_K_RANK,
    RF_N_ESTIM,
    K_GRID,
    K_COVER,
    FREQ_MIN,
)
from transformers import (
    DropMostlyNaN,
    DataFrameImputer,
    SafeLog10,
    ColumnSubset,
    make_encoder_df,
)
from plot_style import set_plot_style


def make_outer_cv_splits(
    X_cv: pd.DataFrame,
    bins_cv: pd.Series,
    n_splits: int = OUTER_N_SPLITS,
) -> List[Tuple[np.ndarray, np.ndarray]]:
    """Create outer stratified folds on the CV set."""
    skf = StratifiedKFold(
        n_splits=n_splits,
        shuffle=True,
        random_state=SEED,
    )
    return list(skf.split(X_cv, bins_cv.to_numpy()))


def export_stage_counts(
    X_cv: pd.DataFrame,
    y_cv_log: pd.Series,
    preprocessor: Pipeline,
    outer_splits: List[Tuple[np.ndarray, np.ndarray]],
    force_recompute: bool = False,
) -> pd.DataFrame:
    """
    Export feature counts across preprocessing stages (global CV and per outer fold).

    Stages tracked:
    - initial
    - after_drop_sparse
    - after_ohe
    - after_variance
    - after_corr
    """
    FEAT_SUMMARY_DIR.mkdir(parents=True, exist_ok=True)
    out_csv = FEAT_SUMMARY_DIR / "stage_counts.csv"
    out_json = FEAT_SUMMARY_DIR / "stage_counts_summary.json"

    if out_csv.exists() and out_json.exists() and not force_recompute:
        print('[Stage counts] Loaded existing stage_counts artifacts.')
        return pd.read_csv(out_csv)

    def _counts_from_fitted(preproc_fitted: Pipeline, n_initial: int) -> Dict[str, int]:
        st = preproc_fitted.named_steps
        n_drop = len(st['drop_sparse'].keep_cols_)
        n_ohe = len(st['ohe'].get_feature_names_out())
        n_var = len(st['var'].get_feature_names_out())
        n_corr = len(st['corr'].keep_cols_)
        return {
            'initial': int(n_initial),
            'after_drop_sparse': int(n_drop),
            'after_ohe': int(n_ohe),
            'after_variance': int(n_var),
            'after_corr': int(n_corr),
        }

    def _enrich(scope: str, c: Dict[str, int]) -> Dict[str, float]:
        i = float(c['initial'])
        d = float(c['after_drop_sparse'])
        o = float(c['after_ohe'])
        v = float(c['after_variance'])
        r = float(c['after_corr'])
        row = {
            'scope': scope,
            **c,
            'drop_sparse_drop': int(i - d),
            'drop_sparse_drop_pct_initial': float((i - d) / i if i else 0.0),
            'ohe_delta': int(o - d),
            'ohe_delta_pct_prev': float((o - d) / d if d else 0.0),
            'variance_drop': int(o - v),
            'variance_drop_pct_prev': float((o - v) / o if o else 0.0),
            'corr_drop': int(v - r),
            'corr_drop_pct_prev': float((v - r) / v if v else 0.0),
            'final_keep_pct_initial': float(r / i if i else 0.0),
        }
        return row

    rows: List[Dict[str, float]] = []

    # Global counts on full CV split
    preproc_global = clone(preprocessor).fit(X_cv, y_cv_log)
    cg = _counts_from_fitted(preproc_global, n_initial=X_cv.shape[1])
    rows.append(_enrich('global_cv', cg))

    # Per-outer-fold counts (fit on each outer-train)
    for fold_id, (i_tr, i_te) in enumerate(outer_splits, start=1):
        X_tr = X_cv.iloc[i_tr]
        y_tr_log = y_cv_log.iloc[i_tr]
        preproc_fold = clone(preprocessor).fit(X_tr, y_tr_log)
        cf = _counts_from_fitted(preproc_fold, n_initial=X_tr.shape[1])
        rows.append(_enrich(f'fold_{fold_id}', cf))

    df = pd.DataFrame(rows)
    df.to_csv(out_csv, index=False)

    fold_df = df[df['scope'].str.startswith('fold_')]
    summary = {
        'n_outer_folds': int(len(fold_df)),
        'global_cv': df[df['scope'] == 'global_cv'].iloc[0].to_dict(),
        'fold_min': fold_df.drop(columns=['scope']).min(numeric_only=True).to_dict(),
        'fold_median': fold_df.drop(columns=['scope']).median(numeric_only=True).to_dict(),
        'fold_max': fold_df.drop(columns=['scope']).max(numeric_only=True).to_dict(),
    }

    with out_json.open('w', encoding='utf-8') as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    print(f"[Stage counts] Saved: {out_csv}")
    print(f"[Stage counts] Saved: {out_json}")
    return df


def _rf_estimator(seed: int = SEED, n_estimators: int = RF_N_ESTIM) -> RandomForestRegressor:
    """RandomForest used inside feature stability selection."""
    return RandomForestRegressor(
        n_estimators=n_estimators,
        max_depth=10,
        min_samples_leaf=3,
        max_features=0.5,
        bootstrap=True,
        random_state=seed,
        n_jobs=-1,
    )


def run_feature_stability(
    X_cv: pd.DataFrame,
    y_cv_log: pd.Series,
    preprocessor: Pipeline,
    outer_splits: List[Tuple[np.ndarray, np.ndarray]],
    force_recompute: bool = False,
) -> None:
    """
    Compute per-fold feature stability rankings using RF + SelectFromModel.

    For each outer fold:
    - fit preprocessing on outer-train
    - bootstrap outer-train in the preprocessed space
    - run SelectFromModel with RF
    - record selection frequency per feature
    """
    FEAT_STABILITY_DIR.mkdir(parents=True, exist_ok=True)

    for fold_id, (i_tr, i_te) in enumerate(outer_splits, start=1):
        out_path = FEAT_STABILITY_DIR / f"fold{fold_id}_freq_within_outer.csv"

        if out_path.exists() and not force_recompute:
            print(f"[Stability] Fold {fold_id}: loaded existing ranking.")
            continue

        print(f"[Stability] Fold {fold_id}/{len(outer_splits)}")

        X_tr = X_cv.iloc[i_tr]
        y_tr_log = y_cv_log.iloc[i_tr]

        preproc_fitted = clone(preprocessor).fit(X_tr, y_tr_log)
        X_tr_pre = preproc_fitted.transform(X_tr)

        rng = np.random.default_rng(SEED + fold_id)
        counts: Counter = Counter()

        for b in range(B_REPEATS):
            idx = rng.integers(0, len(X_tr_pre), size=len(X_tr_pre))
            X_rep = X_tr_pre.iloc[idx]
            y_rep = y_tr_log.iloc[idx]

            rf = _rf_estimator(seed=SEED + 1000 + b, n_estimators=RF_N_ESTIM)
            sfm = SelectFromModel(
                estimator=rf,
                max_features=MAX_K_RANK,
                threshold=-np.inf,
            )
            sfm.fit(X_rep, y_rep)

            feats_b = list(X_tr_pre.columns[sfm.get_support()])
            counts.update(set(feats_b))

        total = float(B_REPEATS)
        freq_df = (
            pd.DataFrame(
                {
                    "feature": list(X_tr_pre.columns),
                    "count": [counts.get(f, 0) for f in X_tr_pre.columns],
                }
            )
            .assign(frequency=lambda d: d["count"] / total)
            .sort_values(["frequency", "count"], ascending=[False, False])
            .reset_index(drop=True)
        )

        freq_df.to_csv(out_path, index=False)
        print(f"[Stability] Fold {fold_id}: saved ranking ({len(freq_df)} features).")


def select_k_per_fold(
    X_cv: pd.DataFrame,
    y_cv_log: pd.Series,
    bins_cv: pd.Series,
    outer_splits: List[Tuple[np.ndarray, np.ndarray]],
    categorical_cols: List[str],
    log10_cols: List[str],
    force_recompute: bool = False,
) -> Tuple[pd.DataFrame, int]:
    """
    For each outer fold, find k* (number of features) based on inner CV coverage.

    Uses a RF model on log10(MSR) and searches over K_GRID; for each k, it keeps
    the top-k features from the stability ranking of that fold and evaluates R²
    via repeated stratified CV. k* is the smallest k with median R² >= K_COVER * max(R²).
    """
    FEAT_K_CURVES_DIR.mkdir(parents=True, exist_ok=True)
    FEAT_SUMMARY_DIR.mkdir(parents=True, exist_ok=True)
    FEAT_FIGURES_DIR.mkdir(parents=True, exist_ok=True)

    k_records: List[Dict[str, int]] = []

    for fold_id, (i_tr, i_te) in enumerate(outer_splits, start=1):
        freq_path = FEAT_STABILITY_DIR / f"fold{fold_id}_freq_within_outer.csv"
        curve_path = FEAT_K_CURVES_DIR / f"fold{fold_id}_k_curve.csv"
        fig_path = FEAT_FIGURES_DIR / f"k_curve_fold{fold_id}.png"

        if not freq_path.exists():
            raise FileNotFoundError(
                f"Missing stability ranking for fold {fold_id}: {freq_path}"
            )

        freq_df = pd.read_csv(freq_path).sort_values(
            ["frequency", "count"], ascending=[False, False]
        )
        ranked_feats = freq_df["feature"].astype(str).tolist()

        X_tr_outer = X_cv.iloc[i_tr]
        y_tr_outer_log = y_cv_log.iloc[i_tr]
        strata_tr = bins_cv.iloc[i_tr]

        min_per_class = pd.Series(strata_tr).value_counts().min()
        n_splits_eff = max(2, min(INNER_MAX_SPLITS, int(min_per_class)))

        inner_cv = RepeatedStratifiedKFold(
            n_splits=n_splits_eff,
            n_repeats=INNER_REPEATS,
            random_state=SEED + fold_id,
        )

        max_k = min(len(ranked_feats), max(K_GRID))
        k_list = [k for k in K_GRID if k <= max_k] or [min(20, max_k)]

        if curve_path.exists() and not force_recompute:
            curve_df = pd.read_csv(curve_path).sort_values("k").reset_index(drop=True)
        else:
            rows = []
            est = _rf_estimator()

            for k in k_list:
                pipe = Pipeline(
                    steps=[
                        ("drop_sparse", DropMostlyNaN(threshold=0.95)),
                        ("imputer", DataFrameImputer()),
                        ("log10", SafeLog10(columns=log10_cols)),
                        ("ohe", make_encoder_df(categorical_cols)),
                        ("subset", ColumnSubset(ranked_feats[:k])),
                        ("model", est),
                    ]
                )

                r2_vals: List[float] = []
                for i_tr_in, i_va_in in inner_cv.split(X_tr_outer.to_numpy(), strata_tr.to_numpy()):
                    X_in_tr = X_tr_outer.iloc[i_tr_in]
                    X_in_va = X_tr_outer.iloc[i_va_in]
                    y_in_tr_log = y_tr_outer_log.iloc[i_tr_in]
                    y_in_va_log = y_tr_outer_log.iloc[i_va_in]

                    m = clone(pipe).fit(X_in_tr, y_in_tr_log)
                    y_pred_log = m.predict(X_in_va)
                    r2_vals.append(r2_score(y_in_va_log, y_pred_log))

                rows.append(
                    {
                        "k": int(k),
                        "R2_median": float(np.median(r2_vals)),
                        "R2_mean": float(np.mean(r2_vals)),
                    }
                )

            curve_df = (
                pd.DataFrame(rows).sort_values("k").reset_index(drop=True)
            )
            curve_df.to_csv(curve_path, index=False)

            set_plot_style()
            plt.figure(figsize=(5.5, 4.0))
            plt.plot(curve_df["k"], curve_df["R2_median"], marker="o")
            plt.xlabel("Number of features (k)")
            plt.ylabel("Median R² (inner CV)")
            plt.title(f"k selection – fold {fold_id}")
            plt.tight_layout()
            plt.savefig(fig_path, dpi=300)
            plt.close()

        r2_max = curve_df["R2_median"].max()
        thresh = K_COVER * r2_max
        eligible = curve_df[curve_df["R2_median"] >= thresh]

        if not eligible.empty:
            k_star = int(eligible["k"].iloc[0])
        else:
            k_star = int(
                curve_df.loc[curve_df["R2_median"].idxmax(), "k"]
            )

        k_records.append({"Fold": fold_id, "k_star": k_star})
        print(
            f"[k*] Fold {fold_id}: k*={k_star} "
            f"(max R²={r2_max:.3f}, threshold={thresh:.3f})"
        )

    k_summary = pd.DataFrame(k_records).sort_values("Fold").reset_index(drop=True)
    k_summary_path = FEAT_SUMMARY_DIR / "k_star_summary.csv"
    k_summary.to_csv(k_summary_path, index=False)

    k_global = int(np.median(k_summary["k_star"].to_numpy(dtype=float)))
    print(f"[k_global] median(k*) = {k_global}")

    return k_summary, k_global

def build_final_feature_set(
    k_summary: pd.DataFrame,
    k_global: int,
    freq_min: float = FREQ_MIN,
) -> List[str]:
    """
    Aggregate per-fold stability rankings to derive the final feature set.

    Uses:
    - per-fold k* from k_summary
    - per-fold stability rankings in FEAT_STABILITY_DIR
    """
    if k_summary.empty:
        raise ValueError("k_summary is empty; run select_k_per_fold first.")

    n_outer = int(k_summary["Fold"].nunique())

    freq = Counter()
    all_ranked: List[str] = []

    for _, row in k_summary.iterrows():
        fold_id = int(row["Fold"])
        k_star = int(row["k_star"])

        freq_path = FEAT_STABILITY_DIR / f"fold{fold_id}_freq_within_outer.csv"
        if not freq_path.exists():
            raise FileNotFoundError(
                f"Missing stability ranking for fold {fold_id}: {freq_path}"
            )

        fold_df = (
            pd.read_csv(freq_path)
            .sort_values(["frequency", "count"], ascending=[False, False])
            .reset_index(drop=True)
        )

        ranked_feats = fold_df["feature"].astype(str).tolist()
        top_k = ranked_feats[:k_star]

        freq.update(top_k)
        all_ranked.extend(ranked_feats)

    freq_df = (
        pd.DataFrame(
            {
                "feature": list(freq.keys()),
                "count": [freq[f] for f in freq],
            }
        )
        .assign(freq=lambda d: d["count"] / float(n_outer))
        .sort_values(["freq", "count"], ascending=[False, False])
        .reset_index(drop=True)
    )

    # primary selection by relative frequency
    final_features = freq_df.loc[freq_df["freq"] >= freq_min, "feature"].tolist()

    # pad or trim to exactly k_global
    if len(final_features) < k_global:
        pad_pool = [f for f in freq_df["feature"].tolist() if f not in final_features]
        final_features = final_features + pad_pool[: max(0, k_global - len(final_features))]
    elif len(final_features) > k_global:
        final_features = final_features[:k_global]

    print(f"\n[Final features] |S|={len(final_features)} (target k_global={k_global})")

    FEAT_SUMMARY_DIR.mkdir(parents=True, exist_ok=True)
    freq_df.to_csv(FEAT_SUMMARY_DIR / "global_feature_frequency.csv", index=False)

    with open(FEAT_SUMMARY_DIR / "final_features.json", "w", encoding="utf-8") as f:
        json.dump(
            {"k_global": int(k_global), "features": final_features},
            f,
            ensure_ascii=False,
            indent=2,
        )

    return final_features
