"""Training-only descriptor rankings: bootstrap random forest or standardized Ridge."""

from msr_qspr import artifacts
from collections import Counter
import json
import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import SelectFromModel
from msr_qspr.config import SEED, B_REPEATS, MAX_K_RANK, RF_N_ESTIM
from msr_qspr.preprocessing import validate_predictor_matrix


def _rf_estimator(seed: int = SEED, n_estimators: int = RF_N_ESTIM) -> RandomForestRegressor:
    """RandomForest used inside feature stability selection."""
    from msr_qspr.config import _n_jobs
    return RandomForestRegressor(
        n_estimators=n_estimators,
        max_depth=10,
        min_samples_leaf=3,
        max_features=0.5,
        bootstrap=True,
        random_state=seed,
        n_jobs=_n_jobs(),
    )


def ranking_settings(preprocessor, seed, strategy='rf'):
    if strategy == 'linear':
        return {'preprocessor': preprocessor.get_params(), 'seed': seed,
                'strategy': strategy, 'ridge': Ridge(alpha=1.).get_params(),
                'scaler': StandardScaler().get_params(),
                'tie_break': ['absolute_standardized_coefficient', 'descriptor_name']}
    if strategy != 'rf':
        raise ValueError('Unknown feature ranking strategy.')
    return {'preprocessor': preprocessor.get_params(), 'seed': seed,
            'bootstrap_repeats': B_REPEATS, 'max_rank': MAX_K_RANK,
            'row_sampling': 'equal_mixture_inverse_surfactant_and_contaminant_frequency',
            'rf': _rf_estimator(n_estimators=RF_N_ESTIM).get_params()}


def balanced_row_probabilities(X):
    groups = X.attrs.get('training_structures')
    if not X.index.is_unique or not isinstance(groups, dict) or X.empty:
        raise ValueError('Balanced ranking requires aligned training group identities.')
    probability = np.zeros(len(X), dtype=float)
    for role in ('Surf_', 'Cont_'):
        mapping = groups.get(role, {})
        if any(i not in mapping or not isinstance(mapping[i], str) or not mapping[i] for i in X.index):
            raise ValueError('Balanced ranking requires aligned training group identities.')
        identities = pd.Series([mapping[i] for i in X.index])
        frequency = identities.map(identities.value_counts()).to_numpy(dtype=float)
        probability += .5 / (identities.nunique() * frequency)
    if not np.isfinite(probability).all() or (probability <= 0).any() or not np.isclose(probability.sum(), 1.):
        raise ValueError('Invalid balanced ranking probabilities.')
    return probability / probability.sum()


def rank_features(X, y, preprocessor, seed, strategy='rf'):
    """Fit preprocessing and supervised stability ranking on this training subset."""
    validate_predictor_matrix(X)
    if not isinstance(y, pd.Series) or not X.index.equals(y.index) or not X.index.is_unique:
        raise ValueError('Ranking predictors and targets must have aligned unique row indices.')
    ranking_settings(preprocessor, seed, strategy)
    fitted_preprocessor = clone(preprocessor)
    X_pre = fitted_preprocessor.fit_transform(X, y)
    if X_pre.shape[1] == 0:
        raise ValueError('No features remain after training-subset preprocessing.')
    if strategy == 'linear':
        standardized = StandardScaler().fit_transform(X_pre)
        coefficients = np.abs(Ridge(alpha=1.).fit(standardized, y).coef_)
        table = (pd.DataFrame({'feature': list(X_pre.columns), 'count': 1,
                              'frequency': 1., 'mean_importance': coefficients})
                .sort_values(['mean_importance', 'feature'], ascending=[False, True])
                .reset_index(drop=True))
        return _add_structure_support(table, fitted_preprocessor)
    counts = Counter()
    importances = np.zeros(X_pre.shape[1], dtype=float)
    rng = np.random.default_rng(seed)
    probability = balanced_row_probabilities(X)
    for b in range(B_REPEATS):
        idx = rng.choice(len(X_pre), size=len(X_pre), replace=True, p=probability)
        selector = SelectFromModel(
            _rf_estimator(seed=SEED + 1000 + b, n_estimators=RF_N_ESTIM),
            max_features=min(MAX_K_RANK, X_pre.shape[1]), threshold=-np.inf,
        ).fit(X_pre.iloc[idx], y.iloc[idx])
        counts.update(set(X_pre.columns[selector.get_support()]))
        importances += selector.estimator_.feature_importances_
    table = (pd.DataFrame({'feature': list(X_pre.columns),
                          'count': [counts.get(f, 0) for f in X_pre.columns]})
            .assign(frequency=lambda d: d['count'] / float(B_REPEATS),
                    mean_importance=importances / float(B_REPEATS))
            .sort_values(['frequency', 'mean_importance', 'feature'], ascending=[False, False, True])
            .reset_index(drop=True))
    return _add_structure_support(table, fitted_preprocessor)


def _add_structure_support(table, preprocessor):
    support = preprocessor.named_steps['structure_support'].support_
    table['training_finite_structures'] = table.feature.map(lambda name: support.get(name, {}).get('finite'))
    table['training_unique_structures'] = table.feature.map(lambda name: support.get(name, {}).get('structures'))
    table['training_finite_rows'] = table.feature.map(lambda name: support.get(name, {}).get('finite_rows'))
    table['training_rows'] = table.feature.map(lambda name: support.get(name, {}).get('rows'))
    return table


def outer_ranking(X, y, preprocessor, fold_id, force_recompute=False, *,
                  cache_dir, scientific_context=None, strategy='rf'):
    validate_predictor_matrix(X)
    suffix = '' if strategy == 'rf' else '_' + strategy
    path = cache_dir / f'fold{fold_id}_freq_within_outer{suffix}.csv'
    signature = artifacts.artifact_signature(X, y, ranking_settings(preprocessor, SEED + fold_id, strategy),
                                          scientific_context=scientific_context)
    with artifacts.exclusive_artifact(path):
        if not force_recompute and artifacts.cache_matches(path, signature):
            return pd.read_csv(path)
        ranking = rank_features(X, y, preprocessor, SEED + fold_id,
                                **({'strategy': strategy} if strategy != 'rf' else {}))
        with artifacts.atomic_artifact(path) as temporary:
            ranking.to_csv(temporary, index=False)
        artifacts.record_cache(path, signature)
        return ranking


def inner_rankings(X, y, preprocessor, splits, path, seed, force_recompute=False, *,
                   scientific_context=None, strategy='rf'):
    """Reuse rankings only for identical training inputs and inner partitions."""
    validate_predictor_matrix(X)
    settings = ranking_settings(preprocessor, seed, strategy)
    if strategy != 'rf':
        path = path.with_name(path.stem + '_' + strategy + path.suffix)
    settings['splits'] = [(tr.tolist(), va.tolist()) for tr, va in splits]
    signature = artifacts.artifact_signature(X, y, settings, scientific_context=scientific_context)
    with artifacts.exclusive_artifact(path):
        if not force_recompute and artifacts.cache_matches(path, signature):
            return json.loads(path.read_text())
        rankings = []
        for step, (tr, va) in enumerate(splits):
            if set(tr).intersection(va):
                raise ValueError('Inner training and validation rows overlap.')
            table = rank_features(X.iloc[tr], y.iloc[tr], preprocessor, seed + step,
                                  **({'strategy': strategy} if strategy != 'rf' else {}))
            rankings.append(table['feature'].astype(str).tolist())
        with artifacts.atomic_artifact(path) as temporary:
            temporary.write_text(json.dumps(rankings) + '\n')
        artifacts.record_cache(path, signature)
        return rankings
