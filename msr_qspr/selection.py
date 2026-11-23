"""Exact normalized minimax selection from complete grouped OOF evidence."""

import json
import math
from msr_qspr import config
from msr_qspr.artifacts import fingerprint, membership_fingerprint
from msr_qspr.config import PRIMARY_SETTINGS

def comparison_table_fingerprint(table):
    """Canonical table identity independent of row order, column order and DataFrame index."""
    if not table.columns.is_unique:
        raise ValueError('Comparison columns must be unambiguous.')
    records = table.astype(object).where(table.notna(), None).to_dict('records')
    return fingerprint(sorted(records, key=fingerprint))


def select_common_configuration(repetition_metrics, oof_predictions, fold_selections, *,
                                experiments, development_ids):
    """Select one specification by normalized minimax error on two chemical settings.

    This pure function accepts synthetic evidence. Production callers must first
    verify aggregation receipts and run and membership provenance. Each candidate
    needs three exhaustive OOF repetitions and 30 signed external-fold feature
    selections. Complexity counts effective estimator inputs, including mandatory
    properties, rather than the raw descriptor pool or k alone. External references
    count inputs a user must supply beyond the two structures.
    Arithmetic uses binary64 and repetition-ordered math.fsum. Ties mean exact
    equality, including both error dimensions; no tolerance or equivalence margin
    affects selection. The 1e-12 check below only validates metric reconstruction.
    Complementary study metrics and all other diagnostics cannot change the score.
    """
    import numpy as np
    import statistics
    ids = list(development_ids)
    membership_fingerprint(ids)
    names = [e.name for e in experiments]
    if not names or len(names) != len(set(names)):
        raise ValueError('Expected candidate specifications must be unique and nonempty.')
    inputs = ((repetition_metrics, {'Scenario', 'Configuration', 'Repetition', 'Estimator',
                                  'PredictorFamily', 'RMSE_OOF', 'n_observations'}),
              (oof_predictions, {'Scenario', 'Configuration', 'Repetition', 'OuterFold',
                                 'Model', 'observation_id', 'actual_log10_MSR', 'predicted_log10_MSR'}),
              (fold_selections, {'Scenario', 'Configuration', 'Repetition', 'OuterFold',
                                 'Estimator', 'PredictorFamily', 'molecular_k', 'descriptors',
                                 'mandatory_reference_properties', 'selected_features',
                                 'n_effective_predictors'}))
    primary = []
    for frame, required in inputs:
        if not frame.columns.is_unique or not required <= set(frame):
            raise ValueError('Incomplete or ambiguous candidate evidence columns.')
        frame = frame.loc[frame.Scenario.isin(PRIMARY_SETTINGS)].copy()
        if set(frame.Configuration) != set(names) or set(frame.Scenario) != set(PRIMARY_SETTINGS):
            raise ValueError('Missing or unexpected candidates or primary settings.')
        primary.append(frame)
    metrics, oof, folds = primary

    def feature_list(value):
        value = json.loads(value) if isinstance(value, str) else value
        if (not isinstance(value, list) or any(not isinstance(v, str) or not v for v in value) or
                len(value) != len(set(value))):
            raise ValueError('Invalid or duplicated selected predictor metadata.')
        return value

    errors, complexities = {}, {}
    targets, memberships = None, {}
    for experiment in experiments:
        counts = []
        for setting in PRIMARY_SETTINGS:
            measured = metrics[metrics.Configuration.eq(experiment.name) & metrics.Scenario.eq(setting)]
            selected = folds[folds.Configuration.eq(experiment.name) & folds.Scenario.eq(setting)]
            predicted = oof[oof.Configuration.eq(experiment.name) & oof.Scenario.eq(setting)]
            for frame in (measured, selected):
                if (set(frame.Estimator) != {experiment.estimator} or
                        set(frame.PredictorFamily) != {experiment.predictor_family}):
                    raise ValueError('Candidate identities differ from verified specifications.')
            if (len(measured) != 3 or set(measured.Repetition) != {1, 2, 3} or
                    measured.Repetition.duplicated().any() or
                    not measured.n_observations.eq(len(ids)).all() or
                    len(selected) != 15 or selected.duplicated(['Repetition', 'OuterFold']).any() or
                    set(zip(selected.Repetition, selected.OuterFold)) !=
                    {(r, f) for r in (1, 2, 3) for f in range(1, 6)}):
                raise ValueError('Incomplete or duplicated repetition/fold records.')
            if (len(predicted) != 3 * len(ids) or
                    set(predicted.Repetition) != {1, 2, 3} or
                    set(predicted.Model) != {experiment.estimator}):
                raise ValueError('Incomplete or incompatible OOF repetitions.')
            values = []
            for repetition in (1, 2, 3):
                frame = predicted[predicted.Repetition.eq(repetition)].sort_values('observation_id')
                if (len(frame) != len(ids) or frame.observation_id.duplicated().any() or
                        set(frame.observation_id) != set(ids) or set(frame.OuterFold) != set(range(1, 6))):
                    raise ValueError('OOF repetition differs from the exact development population.')
                actual = frame.actual_log10_MSR.to_numpy(dtype=float)
                prediction = frame.predicted_log10_MSR.to_numpy(dtype=float)
                if targets is not None and not np.array_equal(actual, targets):
                    raise ValueError('Original observation responses differ across OOF comparisons.')
                targets = actual
                key = (setting, repetition)
                assignment = frame.OuterFold.tolist()
                if key in memberships and memberships[key] != assignment:
                    raise ValueError('Candidates use incompatible external-fold observation memberships.')
                memberships[key] = assignment
                value = float(measured.loc[measured.Repetition.eq(repetition), 'RMSE_OOF'].iloc[0])
                if (not np.isfinite(actual).all() or not np.isfinite(prediction).all() or
                        not math.isfinite(value) or value <= 0):
                    raise ValueError('RMSE and OOF values must be finite with strictly positive RMSE.')
                try:
                    pooled = math.sqrt(math.fsum(float(e) ** 2 for e in prediction - actual) / len(ids))
                except OverflowError as exc:
                    raise ValueError('OOF errors exceed finite RMSE arithmetic.') from exc
                if not math.isclose(value, pooled, rel_tol=1e-12, abs_tol=0):
                    raise ValueError('Repetition RMSE differs from pooled observation predictions.')
                values.append(value)
            errors[experiment.name, setting] = math.fsum(v / 3 for v in values)
            required = list(experiment.required_reference_properties)
            for row in selected.to_dict('records'):
                descriptors = feature_list(row['descriptors'])
                features = feature_list(row['selected_features'])
                if experiment.selection_policy == 'ranked_molecular':
                    k = row['molecular_k']
                    valid = (isinstance(k, (int, float)) and not isinstance(k, bool) and math.isfinite(k) and
                             k == len(descriptors) and experiment.k_min <= k <= experiment.k_max)
                    expected = descriptors + required
                else:
                    k = row['molecular_k']
                    valid = k is None or (isinstance(k, (int, float)) and math.isnan(k))
                    panel = list(experiment.descriptor_panel)
                    valid = valid and descriptors == panel
                    expected = panel + required
                if (not valid or feature_list(row['mandatory_reference_properties']) != required or
                        features != expected or isinstance(row['n_effective_predictors'], bool) or
                        row['n_effective_predictors'] != config.effective_input_count(experiment, features) or not features):
                    raise ValueError('Feature-count metadata differs from verified predictor requirements.')
                counts.append(config.effective_input_count(experiment, features))
        complexities[experiment.name] = statistics.median(counts)
    references = {s: min(errors[e.name, s] for e in experiments) for s in PRIMARY_SETTINGS}
    if not all(math.isfinite(v) and v > 0 for v in references.values()):
        raise ValueError('Reference RMSE must be finite and strictly positive.')
    scores = []
    for experiment in sorted(experiments, key=lambda e: e.name):
        relative = [errors[experiment.name, s] / references[s] - 1 for s in PRIMARY_SETTINGS]
        if not all(math.isfinite(v) for v in relative):
            raise ValueError('Invalid normalized reference RMSE.')
        scores.append({'configuration': experiment.name,
                       'setting_errors': {s: errors[experiment.name, s] for s in PRIMARY_SETTINGS},
                       'worst_relative_deterioration': max(relative),
                       'mean_relative_deterioration': math.fsum(relative) / 2,
                       'mandatory_external_reference_count': len(experiment.external_references),
                       'median_effective_selected_input_count': complexities[experiment.name]})
    winner = min(scores, key=lambda r: (r['worst_relative_deterioration'], r['mean_relative_deterioration'],
        r['mandatory_external_reference_count'], r['median_effective_selected_input_count'], r['configuration']))
    return {'configuration': winner['configuration'], 'reference_errors': references, 'scores': scores}


