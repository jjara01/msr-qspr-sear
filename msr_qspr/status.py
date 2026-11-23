"""Read-only inspection of study-run receipts and development metrics."""

import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

from msr_qspr import artifacts, config
from msr_qspr.reporting import COMPARISON_TABLES, _primary_metrics


def _receipt_valid(path, required=None, *, same_signature=False, allow_missing_sidecar=False):
    try:
        identity = json.loads(path.read_text())['identity']
        return artifacts.completed_unit_valid(path, identity, required,
            member_identity=identity if same_signature else None,
            allow_missing_sidecar=allow_missing_sidecar)
    except (OSError, ValueError, KeyError, TypeError):
        return False


def _fold_state(path, scenario, repetition, experiment, number, development_ids, *,
                allow_missing_sidecar=False):
    receipt = path / 'complete.json'
    if not receipt.exists():
        return {'state': 'incomplete' if path.exists() else 'pending/unknown'}
    expected = [path / name for name in ('parameters.json', 'model.joblib',
                                         'selection.json', 'predictions.csv', 'metrics.json')]
    if experiment.search_method != 'none':
        expected.append(path / 'trials.csv')
    try:
        if not _receipt_valid(receipt, expected, same_signature=True, allow_missing_sidecar=True):
            raise ValueError('Invalid signed outer-fold receipt')
        frame = pd.read_csv(path / 'predictions.csv', dtype={'observation_id': str})
        metrics = json.loads((path / 'metrics.json').read_text())
        selection = json.loads((path / 'selection.json').read_text())
        parameters = json.loads((path / 'parameters.json').read_text())
        if experiment.search_method != 'none':
            from msr_qspr.models import _search_history_valid
            if not _search_history_valid(path / 'trials.csv', selection.get('signature'),
                                         experiment.trial_budget,
                                         experiment.selection_policy == 'ranked_molecular'):
                raise ValueError('Incomplete or incompatible candidate trial history')
        provenance = json.loads((path.parent / 'run_provenance.json').read_text())
        context = {name: value for name, value in provenance.items() if name != 'identity_sha256'}
        effective_count = config.effective_input_count(experiment, selection['selected_features'])
        if (not frame.observation_id.is_unique or not len(frame)
                or set(frame.Scenario) != {scenario}
                or set(frame.Repetition) != {repetition}
                or set(frame.OuterFold) != {number}
                or set(frame.Configuration) != {experiment.name}
                or set(frame.Model) != {experiment.estimator}
                or selection.get('experiment') != json.loads(json.dumps(experiment.to_dict()))
                or selection.get('scenario') != scenario or selection.get('repetition') != repetition
                or selection.get('effective_input_count') != effective_count
                or metrics.get('n_feats') != effective_count
                or parameters.get('experiment') != selection.get('experiment')
                or parameters.get('model_params') != selection.get('model_params')
                or parameters.get('search') != selection.get('search')
                or parameters.get('k') != selection.get('molecular_k')
                or selection.get('scientific_context') != context
                or selection.get('signature') != artifacts.fingerprint({
                    'run': provenance['identity_sha256'], 'fold': selection.get('partition_membership_hashes')})
                or selection.get('signature') != json.loads(receipt.read_text())['identity']
                or selection.get('outer_fold') != number):
            raise ValueError('Outer-fold identity or prediction membership mismatch')
        membership = selection['partition_membership_hashes']['outer']
        if (membership['validation_ids_sha256'] != artifacts.membership_fingerprint(frame.observation_id)
                or membership['validation_groups_sha256'] != artifacts.membership_fingerprint(
                    frame.canonical_group.astype(str).unique())
                or membership['train_ids_sha256'] != artifacts.membership_fingerprint(
                    sorted(set(development_ids) - set(frame.observation_id)))
                or not set(frame.observation_id) <= set(development_ids)):
            raise ValueError('Outer-fold population differs from signed selection provenance')
        artifacts.validate_fold_metrics(frame, metrics)
        sidecar_missing = not receipt.with_name(receipt.name + '.provenance.json').exists()
        return {'state': 'incomplete' if sidecar_missing and not allow_missing_sidecar else 'complete',
                'n_predictions': len(frame),
                'RMSE_outer': metrics['RMSE_outer'], 'R2_outer': metrics['R2_outer']}
    except (OSError, ValueError, TypeError, KeyError, AttributeError) as error:
        return {'state': 'invalid', 'reason': str(error)}


class FoldVerificationCache:
    """Reuse read-only verification only while every constituent stat is unchanged."""

    def __init__(self):
        self.entries = {}

    def inspect(self, path, scenario, repetition, experiment, number, development_ids):
        files = [path / name for name in ('complete.json', 'model.joblib', 'parameters.json',
                 'selection.json', 'predictions.csv', 'metrics.json', 'trials.csv')]
        files.append(path.parent / 'run_provenance.json')
        files += [p.with_name(p.name + '.provenance.json') for p in list(files)]
        def versions():
            values = []
            for file in [path, *files]:
                try:
                    stat = file.stat()
                    values.append((stat.st_dev, stat.st_ino, stat.st_size, stat.st_mtime_ns, stat.st_ctime_ns))
                except OSError:
                    values.append(None)
            return tuple(values)
        key = (path, scenario, repetition, experiment, number, tuple(development_ids))
        before = versions()
        previous = self.entries.get(key)
        if previous is not None and previous[0] == before:
            return dict(previous[1])
        result = _fold_state(path, scenario, repetition, experiment, number, development_ids)
        if versions() == before:
            self.entries[key] = (before, dict(result))
        else:
            self.entries.pop(key, None)
            result = {'state': 'incomplete', 'reason': 'Artifacts changed during verification; retrying.'}
        return result


def _candidate_state(path, scenario, repetition, experiment, *, allow_missing_sidecar=False,
                     fold_cache=None):
    if path is None:
        return {'state': 'pending/unknown', 'folds': {str(i): {'state': 'pending/unknown'}
                                                 for i in range(1, config.GROUPED_OUTER_FOLDS + 1)}}
    try:
        provenance = path / 'run_provenance.json'
        info = json.loads(provenance.read_text())
        identity = info['identity_sha256']
        if (path != artifacts.candidate_directory(path.parents[2], scenario, repetition,
                                                  experiment.name, identity)
                or not artifacts.cache_matches(provenance, identity)
                or info.get('scientific_identity') != config.scientific_identity()
                or info.get('candidate_registry_sha256') != artifacts.candidate_registry_fingerprint()
                or info['scenario'] != scenario or info['repetition'] != repetition
                or info['experiment'] != json.loads(json.dumps(experiment.to_dict()))
                or artifacts.membership_fingerprint(info['development_ids']) !=
                    info['development_population_sha256']):
            raise ValueError('Candidate provenance mismatch')
    except (OSError, ValueError, KeyError, TypeError) as error:
        return {'state': 'invalid', 'reason': str(error), 'folds': {}}
    check = _fold_state if fold_cache is None else fold_cache.inspect
    folds = {str(i): check(path / f'fold-{i:02d}', scenario, repetition, experiment, i, info['development_ids'])
             for i in range(1, config.GROUPED_OUTER_FOLDS + 1)}
    receipt = path / 'repetition_complete.json'
    state = 'invalid' if any(fold['state'] == 'invalid' for fold in folds.values()) else 'incomplete'
    result = {'folds': folds}
    if receipt.exists():
        try:
            expected = [provenance, path / 'nested_outer_summary.csv',
                        path / 'outer_oof_predictions.csv', path / 'nested_outer_fold_metrics.csv']
            expected += [path / f'fold-{i:02d}/complete.json'
                         for i in range(1, config.GROUPED_OUTER_FOLDS + 1)]
            if (not _receipt_valid(receipt, expected, allow_missing_sidecar=True)
                    or json.loads(receipt.read_text())['identity'] != identity
                    or any(entry['state'] != 'complete' for entry in folds.values())):
                raise ValueError('Incomplete or corrupt repetition receipt')
            frame = pd.read_csv(path / 'outer_oof_predictions.csv', dtype={'observation_id': str})
            summary = pd.read_csv(path / 'nested_outer_summary.csv')
            fold_table = pd.read_csv(path / 'nested_outer_fold_metrics.csv')
            expected_rows = sum(entry['n_predictions'] for entry in folds.values())
            if (len(frame) != expected_rows or not frame.observation_id.is_unique
                    or set(frame.observation_id) != set(info['development_ids'])
                    or not frame.groupby('canonical_group').OuterFold.nunique().eq(1).all()
                    or set(frame.OuterFold) != set(range(1, config.GROUPED_OUTER_FOLDS + 1))
                    or len(summary) != 1 or len(fold_table) != config.GROUPED_OUTER_FOLDS
                    or set(fold_table.Fold) != set(range(1, config.GROUPED_OUTER_FOLDS + 1))
                    or set(frame.Configuration) != {experiment.name}
                    or set(frame.Scenario) != {scenario} or set(frame.Repetition) != {repetition}
                    or set(frame.Model) != {experiment.estimator}
                    or summary.Model.iloc[0] != experiment.estimator):
                raise ValueError('Incomplete repetition predictions or fold metrics')
            for number, fold in folds.items():
                current = frame[frame.OuterFold.eq(int(number))]
                signed = pd.read_csv(path / f'fold-{int(number):02d}/predictions.csv',
                                     dtype={'observation_id': str})
                if (len(current) != len(signed)
                        or set(current.observation_id) != set(signed.observation_id)
                        or not np.allclose(current.set_index('observation_id').sort_index()[
                            ['actual_log10_MSR', 'predicted_log10_MSR']],
                            signed.set_index('observation_id').sort_index()[
                            ['actual_log10_MSR', 'predicted_log10_MSR']], rtol=0, atol=1e-10)):
                    raise ValueError('Repetition OOF differs from signed fold predictions')
                row = fold_table[fold_table.Fold.eq(int(number))]
                if (len(row) != 1 or any(not np.isclose(row[field].iloc[0], fold[field], rtol=0, atol=1e-10)
                                         for field in ('RMSE_outer', 'R2_outer'))):
                    raise ValueError('Repetition fold table differs from signed fold metrics')
                artifacts.validate_fold_metrics(current, row.iloc[0].to_dict())
            metrics = _primary_metrics(frame)
            if (not np.isfinite(list(metrics.values())).all()
                    or any(not np.isclose(summary[field].iloc[0], value, rtol=0, atol=1e-10)
                           for field, value in metrics.items())):
                raise ValueError('Repetition OOF metrics differ from predictions')
            result['oof'] = metrics
            result['identity'] = identity
            sidecar_missing = not receipt.with_name(receipt.name + '.provenance.json').exists()
            state = 'incomplete' if sidecar_missing and not allow_missing_sidecar else 'complete'
        except (OSError, ValueError, KeyError, TypeError, AttributeError) as error:
            state = 'invalid'
            result['reason'] = str(error)
    result['state'] = state
    return result


def inspect_status(run_root, *, fold_cache=None):
    """No writes, training, partition regeneration or scheduler assumptions."""
    run_root = Path(run_root)
    experiments = config.study_experiments()
    scenarios = {}
    candidate_states = []
    for scenario, settings in config.VALIDATION_SETTINGS.items():
        repetitions = {}
        for number in range(1, len(settings['seeds']) + 1):
            candidates = {}
            rep_root = run_root / scenario / f'rep-{number:02d}'
            for experiment in experiments:
                paths = sorted(rep_root.glob(experiment.name + '__*'))
                if len(paths) > 1:
                    state = {'state': 'invalid', 'reason': 'Duplicate candidate namespaces', 'folds': {}}
                else:
                    state = _candidate_state(paths[0] if paths else None, scenario, number, experiment,
                                             fold_cache=fold_cache)
                candidates[experiment.name] = state
                candidate_states.append(state)
            repetitions[str(number)] = {'candidates': candidates,
                'complete': sum(v['state'] == 'complete' for v in candidates.values()),
                'expected': len(experiments)}
        scenarios[scenario] = {'repetitions': repetitions,
            'complete': sum(r['complete'] for r in repetitions.values()),
            'expected': len(repetitions) * len(experiments)}
    completed = sum(s['state'] == 'complete' for s in candidate_states)
    verified_folds = sum(f['state'] == 'complete' for s in candidate_states
                         for f in s['folds'].values())
    aggregate_root = run_root.parent / 'aggregation'
    receipt = aggregate_root / 'aggregation_complete.json'
    aggregation = 'pending/unknown'
    if receipt.exists():
        aggregation = 'invalid'
        try:
            sources = json.loads((aggregate_root / 'aggregation_sources.json').read_text())
            signed = json.loads(receipt.read_text())
            expected = {str((scenario, name, int(number))): candidate['identity']
                        for scenario, s in scenarios.items()
                        for number, rep in s['repetitions'].items()
                        for name, candidate in rep['candidates'].items()}
            identity = hashlib.sha256(json.dumps(expected, sort_keys=True).encode()).hexdigest()
            required = [aggregate_root / name for name in (*COMPARISON_TABLES, 'aggregation_sources.json')]
            if (len(expected) == completed and _receipt_valid(receipt, required)
                    and sources.get('format_version') == 1
                    and not Path(sources['runs_directory']).is_absolute()
                    and signed['identity'] == identity
                    and (aggregate_root / sources['runs_directory']).resolve() == run_root.resolve()):
                table = pd.read_csv(aggregate_root / 'repetition_metrics.csv')
                table = table.set_index(['Scenario', 'Configuration', 'Repetition'])
                if len(table) != completed or not table.index.is_unique:
                    raise ValueError('Aggregation has duplicate or missing candidate repetitions')
                for scenario, scenario_state in scenarios.items():
                    for number, repetition_state in scenario_state['repetitions'].items():
                        for name, candidate in repetition_state['candidates'].items():
                            row = table.loc[(scenario, name, int(number))]
                            if any(not np.isclose(row[field], value, rtol=0, atol=1e-10)
                                   for field, value in candidate['oof'].items()):
                                raise ValueError('Aggregated metrics differ from signed repetition OOF')
                aggregation = 'complete'
        except (OSError, ValueError, KeyError, TypeError, AttributeError):
            pass
    summary = {'complete': completed, 'expected': len(candidate_states),
               'verified_folds': verified_folds,
               'expected_folds': len(candidate_states) * config.GROUPED_OUTER_FOLDS,
               'incomplete': sum(s['state'] == 'incomplete' for s in candidate_states),
               'pending_unknown': sum(s['state'] == 'pending/unknown' for s in candidate_states),
               'invalid': sum(s['state'] == 'invalid' for s in candidate_states),
               'aggregation': aggregation}
    summary['state'] = ('invalid' if summary['invalid'] or aggregation == 'invalid'
                        else 'complete' if completed == len(candidate_states) and aggregation == 'complete'
                        else 'incomplete' if completed or summary['incomplete'] else 'pending/unknown')
    return {'run_root': str(run_root), 'available': run_root.is_dir(),
            'verification': 'Artifact integrity and metric consistency; scientific acceptance requires aggregate/select. '
                            'Pending/unknown and incomplete do not describe scheduler state.',
            'study_run': summary, 'scenarios': scenarios}
