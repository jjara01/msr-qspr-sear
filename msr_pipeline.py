from config import ensure_directories, FORCE_RECOMPUTE
from plot_style import set_plot_style
from data_prep import prepare_full_dataframe
from split_preproc import create_cv_holdout_split, build_preprocessor
from feature_selection import (
    make_outer_cv_splits,
    export_stage_counts,
    run_feature_stability,
    select_k_per_fold,
    build_final_feature_set,
)
from model_selection import run_model_selection
from final_model import train_final_model
from diagnostics import run_y_randomization, run_shap, run_applicability_domain
import matplotlib as mpl


def main() -> None:
    
    ensure_directories()
    set_plot_style()

    # Data + descriptors
    df_full = prepare_full_dataframe(force_recompute=FORCE_RECOMPUTE["data_prep"])
    print(f"[Data] Full dataframe: {df_full.shape}\n")


    # Split CV / holdout + log10(MSR)
    split = create_cv_holdout_split(df_full)
    print(
        f"[Split] X_cv={split['X_cv'].shape}, "
        f"X_holdout={split['X_holdout'].shape}, "
        f"y_cv={split['y_cv'].shape}, "
        f"y_holdout={split['y_holdout'].shape}"
    )


    # Preprocessor built on CV features
    preprocessor, cat_cols, log_cols = build_preprocessor(split["X_cv"])
    print(f"\n[Preprocessor] categorical={cat_cols}")
    print(f"[Preprocessor] to log10={log_cols}")


    # Outer CV folds on CV set
    outer_splits = make_outer_cv_splits(split["X_cv"], split["bins_cv"])
    print(f"\n[Outer CV] Number of folds: {len(outer_splits)}\n")


    # Stage-wise feature counts (quick diagnostics for reporting)
    export_stage_counts(
        X_cv=split["X_cv"],
        y_cv_log=split["y_cv_log"],
        preprocessor=preprocessor,
        outer_splits=outer_splits,
        force_recompute=FORCE_RECOMPUTE["feature_stability"],
    )

    # Stability rankings per outer fold
    run_feature_stability(
        X_cv=split["X_cv"],
        y_cv_log=split["y_cv_log"],
        preprocessor=preprocessor,
        outer_splits=outer_splits,
        force_recompute=FORCE_RECOMPUTE["feature_stability"],
    )


    # k* per fold + k_global
    k_summary, k_global = select_k_per_fold(
        X_cv=split["X_cv"],
        y_cv_log=split["y_cv_log"],
        bins_cv=split["bins_cv"],
        outer_splits=outer_splits,
        categorical_cols=cat_cols,
        log10_cols=log_cols,
        force_recompute=FORCE_RECOMPUTE["k_selection"],
    )


    # Nested CV of models using k_global and stability rankings
    _, _, best_model_name = run_model_selection(
        X_cv=split["X_cv"],
        y_cv_log=split["y_cv_log"],
        bins_cv=split["bins_cv"],
        outer_splits=outer_splits,
        categorical_cols=cat_cols,
        log10_cols=log_cols,
        k_global=k_global,
        force_recompute=FORCE_RECOMPUTE["model_selection"],
    )

    # Final feature set
    final_features = build_final_feature_set(k_summary, k_global)

    # Refit on 90% CV with final feature set, and evaluate in holdout
    train_final_model(
        X_cv=split["X_cv"],
        y_cv_log=split["y_cv_log"],
        bins_cv=split["bins_cv"],
        X_holdout=split["X_holdout"],
        y_holdout_log=split["y_holdout_log"],
        categorical_cols=cat_cols,
        log10_cols=log_cols,
        final_features=final_features,
        best_model_name=best_model_name,
        force_recompute=FORCE_RECOMPUTE["final_model"]
    )

    # Y-randomization
    run_y_randomization(
        X_cv=split["X_cv"],
        y_cv_log=split["y_cv_log"],
        bins_cv=split["bins_cv"],
        categorical_cols=cat_cols,
        log10_cols=log_cols,
        final_features=final_features,
        best_model_name=best_model_name,
        force_recompute=FORCE_RECOMPUTE["y_randomization"],
    )

    # SHAP
    run_shap(
        best_model_name=best_model_name,
        X_data=split["X_cv"],
        final_features=final_features,
        max_display=11,
    ) 

    # Applicability domain
    with mpl.rc_context():
        mpl.rcParams["savefig.bbox"] = "standard"
        mpl.rcParams["savefig.pad_inches"] = 0.02
        run_applicability_domain(
            best_model_name=best_model_name,
            X_cv=split["X_cv"],
            y_cv_log=split["y_cv_log"],
            X_holdout=split["X_holdout"],
            y_holdout_log=split["y_holdout_log"],
            k=7,
            force_recompute=FORCE_RECOMPUTE["applicability_domain"],
            q_ref=0.95
        )


if __name__ == "__main__":
    main()
