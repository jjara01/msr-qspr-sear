"""Training-bound reliability, complete-procedure randomization and fitted-model interpretation.

All errors and response ranges use log10(MSR), with dimensionless molar MSR.
These interfaces never authorize scientific execution or reserved-holdout access.
"""

import json
import numpy as np
import pandas as pd
from sklearn.pipeline import Pipeline
from msr_qspr.artifacts import fingerprint
from msr_qspr.preprocessing import require_predictor_schema, validate_predictor_matrix
from msr_qspr.config import AD_POLICY

INTERPRETATION_LIMITATIONS = (
    'Associations are not causal. Correlated descriptors redistribute SHAP attribution and '
    'can reduce marginal permutation importance. Observation permutations do not preserve '
    'chemical grouping and can create unrealistic combinations. Reference properties are '
    'available inputs, not inferred experimental responses. Development importance is descriptive.'
)


def fitted_input_space(pipe, X):
    """Use the actual fitted transform and verify feature order and estimator predictions."""
    require_predictor_schema(pipe)
    validate_predictor_matrix(X)
    values = np.asarray(Pipeline(pipe.steps[:-1]).transform(X), dtype=float)
    subset = pipe.named_steps.get('subset')
    if subset is None or not getattr(subset, 'columns_', None):
        raise ValueError('Interpretation and AD require a fitted ordered feature subset.')
    names = list(subset.columns_)
    model = pipe.steps[-1][1]
    if (values.shape != (len(X), len(names)) or not np.isfinite(values).all()
            or values.shape[1] != model.n_features_in_
            or not np.allclose(model.predict(values), pipe.predict(X), rtol=1e-10, atol=1e-12)):
        raise ValueError('Fitted preprocessing, feature order and predictions are not aligned.')
    return pd.DataFrame(values, columns=names, index=X.index)


def reference_groups(metadata, scenario):
    """Canonical chemical units, independent of publication labels and row multiplicity."""
    columns = {'row_id', 'active_structure_group', 'contaminant_group'}
    if (not columns <= set(metadata) or metadata[list(columns)].isna().any().any()
            or not metadata.row_id.is_unique):
        raise ValueError('Reliability requires unique observations and canonical chemical identities.')
    if scenario == 'unseen_surfactant':
        return metadata.active_structure_group.astype(str).to_numpy()
    if scenario == 'unseen_contaminant':
        return metadata.contaminant_group.astype(str).to_numpy()
    if scenario == 'unseen_study':
        return np.array([fingerprint([a, b]) for a, b in
                         zip(metadata.active_structure_group, metadata.contaminant_group)])
    raise ValueError('Unknown chemical reliability setting.')


def feature_entity(feature):
    """Assign each effective fitted coordinate to its chemical information block.

    Pair descriptors are deterministic properties of one contaminant-surfactant
    combination and form their own block. Block weighting then gives every
    represented block (contaminant, surfactant, pair) equal total weight,
    independent of how many coordinates each contributes.
    """
    from msr_qspr.config import ABRAHAM_DESCRIPTORS
    from msr_qspr.chemistry import PAIR_NAMES
    if feature.startswith('Surf_') or feature in {'CMC', 'SurfactantType'}:
        return 'surfactant'
    if (feature.startswith('Cont_') or feature in {'WaterSolubility', 'ContaminantType'}
            or feature in ABRAHAM_DESCRIPTORS):
        return 'contaminant'
    if feature in PAIR_NAMES:
        return 'pair'
    raise ValueError(f'Unknown AD feature-to-entity assignment: {feature}')


class ReliabilityReference:
    """Stored training geometry plus response support; local errors require independent lineage."""

    def __init__(self, pipe, X, y_log, metadata, *, scenario, provenance, policy=None):
        self.policy = dict(AD_POLICY if policy is None else policy)
        k, q = self.policy['neighbors'], self.policy['distance_quantile']
        if type(k) is not int or k < 1 or not 0 < q < 1:
            raise ValueError('AD neighbors and training distance quantile are invalid.')
        self.ids = metadata.row_id.astype(str).tolist()
        self.groups = reference_groups(metadata, scenario)
        self.scenario, self.provenance = scenario, dict(provenance)
        self.metadata = metadata.reset_index(drop=True).copy()
        self.y_log = np.asarray(y_log, dtype=float)
        if (len(self.ids) != len(X) or self.y_log.shape != (len(X),)
                or not X.index.equals(metadata.index) or not np.isfinite(self.y_log).all()):
            raise ValueError('Training reliability inputs must be finite and aligned.')
        if hasattr(y_log, 'index') and not y_log.index.equals(X.index):
            raise ValueError('Training reliability response order differs from predictors.')
        effective = fitted_input_space(pipe, X)
        self.features = effective.columns.tolist()
        self.mean = effective.to_numpy().mean(axis=0)
        self.scale = effective.to_numpy().std(axis=0)
        self.scale[self.scale == 0] = 1.
        blocks = np.array([feature_entity(n) for n in self.features])
        self.feature_entities = blocks.tolist()
        self.block_weights = np.array([1 / np.sqrt(np.sum(blocks == b) * len(set(blocks)))
                                       for b in blocks])
        self.values = self.normalize(effective.to_numpy())
        scores = [self.neighbors(v, own_id=obs, own_group=group)[0]
                  for v, obs, group in zip(self.values, self.ids, self.groups)]
        grouped = pd.DataFrame({'group': self.groups, 'distance': scores}).groupby('group').distance.mean()
        self.distance_threshold = (float(np.quantile(grouped, q))
                                   if np.isfinite(grouped).all() else None)
        self.response_range = [float(self.y_log.min()), float(self.y_log.max())]
        self.local_status = 'unavailable'
        self.local_reason = 'Independent nested grouped residual references and threshold provenance are required.'
        self.local_atlas = None
        self.local_threshold = None
        self.local_policy = None

    def normalize(self, values):
        return (np.asarray(values, dtype=float) - self.mean) / self.scale * self.block_weights

    def neighbors(self, value, *, own_id=None, own_group=None, eligible_ids=None):
        distances = np.sqrt(np.sum((self.values - value) ** 2, axis=1))
        best = {}
        for i in np.argsort(distances, kind='stable'):
            group = self.groups[i]
            if (self.ids[i] == own_id or group == own_group
                    or (eligible_ids is not None and self.ids[i] not in eligible_ids)):
                continue
            if group not in best:
                best[group] = int(i)
        indices = list(best.values())[:self.policy['neighbors']]
        score = (float(np.mean(distances[indices]))
                 if len(indices) == self.policy['neighbors'] else float('nan'))
        return score, indices, distances[indices].tolist(), len(best)

    def attach_local_error(self, atlas, *, query_metadata, threshold_policy):
        """Validate every supervised boundary before using any local residual or threshold."""
        from msr_qspr.residuals import validate_residual_references
        combined = pd.concat([self.metadata, query_metadata], ignore_index=True)
        result = validate_residual_references(atlas,
            query_ids=query_metadata.row_id.tolist(), outer_training_ids=self.ids,
            outer_validation_ids=query_metadata.row_id.tolist(), scenario=self.scenario,
            threshold_policy=threshold_policy, canonical_metadata=combined, **self.provenance)
        self.local_reason = result['reason']
        if result['status'] != 'eligible':
            return result
        references = result['references']
        fit_ids = threshold_policy['fit_ids']
        if set(fit_ids) != set(references.observation_id):
            raise ValueError('Threshold calibration must bind the complete eligible atlas population.')
        q = threshold_policy.get('quantile')
        if not isinstance(q, (int, float)) or not 0 < q < 1:
            raise ValueError('An explicit training local-error quantile is required.')
        self._calibrate_local(references, threshold_policy)
        return result

    def attach_final_local_error(self, atlas, *, threshold_policy):
        """Bind complete grouped out-of-training references for unknown future queries."""
        from msr_qspr.residuals import validate_final_residual_references
        result = validate_final_residual_references(atlas, training_metadata=self.metadata,
            y_log=self.y_log, scenario=self.scenario, threshold_policy=threshold_policy,
            **self.provenance)
        self._calibrate_local(result['references'], threshold_policy)
        return result

    def _calibrate_local(self, references, threshold_policy):
        if set(references.observation_id) != set(self.ids) or len(references) != len(self.ids):
            raise ValueError('Local calibration requires exactly one residual for every reference observation.')
        q = threshold_policy['quantile']
        if q != self.policy['local_error_quantile']:
            raise ValueError('Local calibration quantile differs from the fixed AD policy.')
        residuals = references.set_index('observation_id').residual.abs().to_dict()
        scores = []
        for value, observation, group in zip(self.values, self.ids, self.groups):
            _, neighbors, _, _ = self.neighbors(value, own_id=observation, own_group=group,
                                                eligible_ids=set(residuals))
            scores.append(float(np.mean([residuals[self.ids[i]] for i in neighbors]))
                          if len(neighbors) == self.policy['neighbors'] else float('nan'))
        grouped = pd.DataFrame({'group': self.groups, 'local_score': scores}).groupby('group').local_score.mean()
        self.local_threshold = (float(np.quantile(grouped, q))
                                if np.isfinite(grouped).all() else None)
        self.local_atlas = residuals
        self.local_status = 'training_calibrated' if self.local_threshold is not None else 'unavailable'
        self.local_policy = dict(threshold_policy)
        self.local_reason = ('Selection-conditional grouped cross-fitted residual reference; '
                             'training-derived local-risk threshold; no prediction interval.'
                             if self.local_threshold is not None else 'Insufficient distinct reference groups.')

    def assess(self, pipe, X, query_metadata, *, unavailable_properties=()):
        """Component decisions use predictions and chemical metadata, never query targets."""
        if (len(X) != len(query_metadata) or not X.index.equals(query_metadata.index)
                or query_metadata.row_id.duplicated().any()):
            raise ValueError('Query predictors and identities must align.')
        values = fitted_input_space(pipe, X)
        if list(values) != self.features:
            raise ValueError('AD feature representation differs from fitted reference.')
        groups = reference_groups(query_metadata, self.scenario)
        prediction = np.asarray(pipe.predict(X), dtype=float)
        property_only = set(self.features) <= {'CMC', 'WaterSolubility'}
        from msr_qspr.config import ABRAHAM_DESCRIPTORS
        required = set(self.features) & ({'CMC', 'WaterSolubility'} | set(ABRAHAM_DESCRIPTORS))
        unknown = set(unavailable_properties) - required
        if unknown:
            raise ValueError(f'AD missing-property declarations do not match effective features: {sorted(unknown)}')
        # Structure-derived coordinates imputed from training values carry no query information.
        derived = [f for f in self.features if f not in required and f in X.columns]
        lower, upper = self.values.min(axis=0), self.values.max(axis=0)
        rows = []
        for pos, (obs, group, value, pred) in enumerate(zip(query_metadata.row_id, groups,
                                          self.normalize(values.to_numpy()), prediction)):
            score, indices, distances, count = self.neighbors(value, own_id=obs, own_group=group)
            available = self.distance_threshold is not None and np.isfinite(score)
            chemical = ('inside' if score <= self.distance_threshold else 'outside') if available else 'unavailable'
            response = 'inside' if self.response_range[0] <= pred <= self.response_range[1] else 'outside'
            local, error, local_count = 'unavailable', None, 0
            if self.local_atlas is not None:
                _, local_indices, _, local_count = self.neighbors(value, own_id=obs, own_group=group,
                                                                 eligible_ids=set(self.local_atlas))
                if len(local_indices) == self.policy['neighbors'] and self.local_threshold is not None:
                    error = float(np.mean([self.local_atlas[self.ids[i]] for i in local_indices]))
                    local = 'inside' if error <= self.local_threshold else 'outside'
                else:
                    local = 'unavailable'
            missing = sorted(set(unavailable_properties) | {p for p in required if pd.isna(X.iloc[pos][p])})
            imputed = sorted(f for f in derived if pd.isna(X.iloc[pos][f]))
            outside = (value < lower) | (value > upper)
            absolute_z = np.abs(value / self.block_weights)
            max_index = int(np.argmax(absolute_z))
            if (missing or imputed) and chemical == 'inside':
                chemical = 'unavailable'
            status = ('outside' if 'outside' in (chemical, response, local) else
                      'incomplete' if 'unavailable' in (chemical, response, local) else 'inside')
            rows.append({'observation_id': obs, 'support_group': group,
                'predicted_log10_MSR': float(pred), 'chemical_support': chemical,
                'chemical_reason': (f'Missing external inputs imputed from training: {missing}.' if missing and not imputed
                                    else f'Missing inputs imputed from training: {missing + imputed}.' if imputed
                                    else 'Training group-distance quantile (property space only; no molecular similarity claim).'
                                    if available and property_only else 'Training group-distance quantile.' if available
                                    else 'Insufficient distinct chemical reference groups.'),
                'missing_reference_properties': missing,
                'imputed_descriptors': imputed,
                'outside_training_range_count': int(outside.sum()),
                'outside_training_range_features': json.dumps(np.asarray(self.features)[outside].tolist()),
                'max_abs_training_z': float(absolute_z[max_index]),
                'max_abs_training_z_feature': self.features[max_index],
                'distance': float(score) if np.isfinite(score) else None,
                'distance_threshold': self.distance_threshold,
                'neighbor_ids': [self.ids[i] for i in indices],
                'neighbor_groups': [self.groups[i] for i in indices], 'neighbor_distances': distances,
                'neighbor_count': len(indices), 'available_support_groups': count,
                'response_range': response, 'response_reason': 'Training response range; not a prediction interval.',
                'response_min': self.response_range[0], 'response_max': self.response_range[1],
                'local_error': local, 'local_error_value': error, 'local_error_threshold': self.local_threshold,
                'local_error_reason': self.local_reason, 'local_support_groups': local_count,
                'local_calibration_status': self.local_status,
                'ad_status': status,
                'support_inside_available_components': chemical == response == 'inside',
                'independently_validated': False})
        return pd.DataFrame(rows)

    def specification(self):
        return {'features': self.features, 'feature_entities': self.feature_entities,
                'policy': self.policy, 'training_ids': self.ids,
                'reference_groups': self.groups.tolist(), 'scenario': self.scenario,
                'distance_threshold': self.distance_threshold, 'response_range': self.response_range,
                'normalization_mean': self.mean.tolist(), 'normalization_scale': self.scale.tolist(),
                'block_weights': self.block_weights.tolist(), 'provenance': self.provenance,
                'components': {'chemical_support': 'training_derived' if self.distance_threshold is not None
                               else 'insufficient_groups', 'response_range': 'training_derived',
                               'local_error': self.local_status},
                'local_threshold': self.local_threshold, 'local_policy': self.local_policy,
                'independently_validated': False, 'target': 'log10_MSR', 'MSR_units': 'mol/mol'}


def run_applicability_domain(*, pipe, X_train, y_train_log, training_metadata, X_query,
                             query_metadata, scenario, provenance, policy=None,
                             residual_atlas=None, threshold_policy=None):
    """Build training-only support and assess a separate population without its targets."""
    validate_predictor_matrix(X_train)
    validate_predictor_matrix(X_query)
    reference = ReliabilityReference(pipe, X_train, y_train_log, training_metadata,
                                     scenario=scenario, provenance=provenance, policy=policy)
    if residual_atlas is not None:
        reference.attach_local_error(residual_atlas, query_metadata=query_metadata,
                                     threshold_policy=threshold_policy)
    return reference, reference.assess(pipe, X_query, query_metadata)


def coverage_error_report(status, actual_log, canonical_groups, classes=None):
    """Report mixed-group membership explicitly; repeated OOF repetitions stay separate."""
    actual, groups = np.asarray(actual_log, dtype=float), np.asarray(canonical_groups)
    if len(status) != len(actual) or len(groups) != len(actual) or not np.isfinite(actual).all():
        raise ValueError('Coverage errors require aligned finite targets and complete groups.')
    if pd.isna(groups).any() or not status.observation_id.is_unique:
        raise ValueError('Report exactly one observation prediction per repetition and setting.')
    error = actual - status.predicted_log10_MSR.to_numpy(dtype=float)
    if classes is not None and (len(classes) != len(status) or
            not {'SurfactantType', 'ContaminantType'} <= set(classes) or
            classes[['SurfactantType', 'ContaminantType']].isna().any().any()):
        raise ValueError('Coverage classes require aligned complete chemical metadata.')
    strata = [('all', 'all', np.ones(len(status), dtype=bool))]
    if classes is not None:
        for column in ('SurfactantType', 'ContaminantType'):
            for label in sorted(classes[column].unique()):
                strata.append((column, label, classes[column].to_numpy() == label))
    rows = []
    for class_type, class_value, stratum in strata:
      for component in ('chemical_support', 'response_range', 'local_error', 'ad_status'):
        for state in ('inside', 'outside', 'incomplete', 'unavailable'):
            mask = stratum & status[component].eq(state).to_numpy()
            n = int(mask.sum())
            denominator = int(stratum.sum())
            counts = pd.DataFrame({'group': groups[stratum], 'selected': mask[stratum]}).groupby('group').selected.agg(['sum', 'size'])
            sufficient = n >= 10 and int((counts['sum'] > 0).sum()) >= 5
            local = status.loc[mask, 'local_error_value'].to_numpy(dtype=float)
            usable = np.isfinite(local)
            association = (float(pd.Series(local[usable]).corr(pd.Series(np.abs(error[mask][usable])),
                                                         method='spearman'))
                           if usable.sum() >= 3 and np.unique(local[usable]).size > 1 and
                           np.unique(np.abs(error[mask][usable])).size > 1 else None)
            rows.append({'class_type': class_type, 'class_value': class_value,
                'component': component, 'status': state, 'n_observations': n,
                'observation_denominator': denominator, 'observation_coverage': n / denominator,
                'n_groups_any': int((counts['sum'] > 0).sum()),
                'n_groups_all': int((counts['sum'] == counts['size']).sum()),
                'error_summary_support': 'descriptive' if sufficient else 'insufficient_support',
                'group_denominator': len(counts),
                'group_coverage_any': float((counts['sum'] > 0).mean()),
                'group_coverage_all': float((counts['sum'] == counts['size']).mean()),
                'RMSE_log10_MSR': float(np.sqrt(np.mean(error[mask] ** 2))) if n else None,
                'MAE_log10_MSR': float(np.mean(np.abs(error[mask]))) if n else None,
                'local_error_vs_absolute_error_spearman_descriptive': association,
                'n_local_association': int(usable.sum())})
    return pd.DataFrame(rows)


def descriptor_definitions(features):
    from msr_qspr.chemistry import MORDRED_CALC
    definitions = {str(d): {'definition': str(d.description()),
                            'implementation': f'{type(d).__module__}.{type(d).__name__}'}
                   for d in MORDRED_CALC.descriptors}
    result = []
    abraham = {'Abraham_E': 'excess molar refraction', 'Abraham_S': 'dipolarity/polarizability',
               'Abraham_A': 'overall hydrogen-bond acidity', 'Abraham_B': 'overall hydrogen-bond basicity',
               'Abraham_V': 'McGowan characteristic volume'}
    pairs = {'Pair_LogMWRatio': 'log10(Cont_MW / Surf_MW)',
             'Pair_DeltaSLogP': 'Cont_SLogP - Surf_SLogP',
             'Pair_DeltaPolarDensity': 'Cont_TopoPSA/Cont_MW - Surf_TopoPSA/Surf_MW',
             'Pair_SLogPProduct': 'Cont_SLogP * Surf_SLogP',
             'Pair_ContLogP_SurfPolar': 'Cont_SLogP * Surf_TopoPSA/Surf_MW',
             'Pair_SurfLogP_ContPolar': 'Surf_SLogP * Cont_TopoPSA/Cont_MW'}
    for feature in features:
        if feature in abraham:
            definition = {'definition': f'Abraham solute descriptor: {abraham[feature]}',
                          'implementation': 'msr_qspr.abraham (curated table, verified by checksum)',
                          'entity': 'contaminant',
                          'limitation': 'Curated literature or estimated values; unresolved identities are imputed and flagged.'}
        elif feature in pairs:
            definition = {'definition': pairs[feature], 'implementation': 'msr_qspr.chemistry.pair_descriptors',
                          'entity': 'contaminant-surfactant pair',
                          'limitation': 'Inherits limitations of represented structures, including proxies and UVCBs.'}
        elif feature in {'CMC', 'WaterSolubility'}:
            definition = {'definition': 'Independent reference CMC in mol/L' if feature == 'CMC'
                          else 'Independent reference contaminant water solubility in g/L',
                          'implementation': 'training median imputation then fitted SafeLog10'}
        elif feature[:5] in {'Cont_', 'Surf_'} and feature[5:] in definitions:
            definition = dict(definitions[feature[5:]], entity='contaminant' if feature.startswith('Cont_') else 'surfactant')
        else:
            raise ValueError(f'No scientific descriptor definition for {feature}')
        result.append(dict(feature=feature, **definition))
    return pd.DataFrame(result)


def _standardized_linear_terms(model, values):
    """Exact contributions of a standardized Ridge relative to its training mean."""
    scaler, ridge = model.named_steps['scaler'], model.named_steps['ridge']
    terms = np.asarray(scaler.transform(values), dtype=float) * np.asarray(ridge.coef_, dtype=float)
    return terms, np.full(len(values), float(ridge.intercept_))


def _tree_terms(model, values):
    import shap
    explanation = shap.TreeExplainer(model)(values)
    return (np.asarray(explanation.values, dtype=float),
            np.broadcast_to(np.asarray(explanation.base_values, dtype=float).reshape(-1), (len(values),)).copy())


def _anchored_attribution(pipe, values):
    """Separate the imposed solubility offset, learned linear terms and tree corrections.

    Each component is exact for its learner: standardized Ridge terms are linear
    contributions about the training mean; TreeSHAP explains only the boosting
    trees, after the Ridge initial estimator has been accounted for separately.
    """
    import copy
    from sklearn.dummy import DummyRegressor
    from sklearn.ensemble import GradientBoostingRegressor
    model = pipe.steps[-1][1]
    inner = model.estimator_
    array = values.to_numpy()
    offset = model.anchor_offset(array)
    linear = np.zeros_like(array)
    nonlinear = np.zeros_like(array)
    if isinstance(inner, Pipeline) and set(inner.named_steps) == {'scaler', 'ridge'}:
        linear, base = _standardized_linear_terms(inner, array)
    elif isinstance(inner, GradientBoostingRegressor) and isinstance(inner.init_, Pipeline):
        # Gradient boosting evaluates its initial estimator on float32 tree inputs.
        linear, base = _standardized_linear_terms(inner.init_, array.astype(np.float32))
        trees = copy.copy(inner)
        trees.init_ = DummyRegressor(strategy='constant', constant=0.).fit(array[:1], [0.])
        nonlinear, tree_base = _tree_terms(trees, array)
        base = base + tree_base
    elif isinstance(inner, GradientBoostingRegressor):
        nonlinear, base = _tree_terms(inner, array)
    else:
        return {'status': 'unsupported', 'reason': f'No prescribed attribution for anchored {type(inner).__name__}.'}
    return {'contributions': linear + nonlinear, 'base_values': base + offset,
            'components': {'imposed_offset': offset, 'learned_base': base,
                           'linear': linear, 'nonlinear': nonlinear,
                           'anchor_coefficient': float(model.anchor_coefficient)}}


def fitted_shap(pipe, X, background):
    """Explain the fitted estimator in its exact effective feature space, without refitting.

    For solubility-anchored estimators the imposed offset is reported separately and
    is included in the base values, so learned attributions describe only fitted terms.
    """
    from msr_qspr.estimators import SolubilityAnchoredRegressor
    model = pipe.steps[-1][1]
    name = type(model).__name__
    anchored = isinstance(model, SolubilityAnchoredRegressor)
    if not anchored and name not in {'LinearRegression', 'Ridge', 'RandomForestRegressor',
                                     'GradientBoostingRegressor', 'XGBRegressor'}:
        return {'status': 'unsupported', 'reason': f'No prescribed SHAP adapter for {name}.'}
    values = fitted_input_space(pipe, X)
    components = None
    if anchored:
        result = _anchored_attribution(pipe, values)
        if result.get('status') == 'unsupported':
            return result
        contributions, bases, components = result['contributions'], result['base_values'], result['components']
    else:
        import shap
        reference = (np.asarray(pipe.interpretation_background_, dtype=float) if background is None
                     else fitted_input_space(pipe, background).to_numpy())
        if reference.ndim != 2 or reference.shape[1] != len(values.columns) or not np.isfinite(reference).all():
            raise ValueError('SHAP background does not match the effective fitted feature space.')
        if name in {'LinearRegression', 'Ridge'}:
            explainer = shap.LinearExplainer(model, reference)
        else:
            explainer = shap.TreeExplainer(model)
        explanation = explainer(values.to_numpy())
        contributions = np.asarray(explanation.values, dtype=float)
        bases = np.asarray(explanation.base_values, dtype=float)
    if (contributions.shape != values.shape or not np.isfinite(contributions).all()
            or not np.allclose(contributions.sum(axis=1) + bases, pipe.predict(X), rtol=1e-4, atol=1e-5)):
        raise ValueError('SHAP preprocessing or additive prediction alignment failed.')
    result = {'status': 'available', 'values': contributions, 'base_values': bases,
              'features': list(values), 'feature_values': values.to_numpy()}
    if components is not None:
        result['components'] = components
    return result
