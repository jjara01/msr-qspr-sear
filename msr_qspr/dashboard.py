"""Read-only, keyboard-driven study-run explorer."""

import ast
import csv
import curses
import hashlib
import io
import json
import math
from pathlib import Path
import textwrap
import threading
import time

from msr_qspr import config, progress
from msr_qspr.status import FoldVerificationCache, inspect_status

STATES = ('PENDING', 'RUNNING', 'COMPLETE', 'FAILED', 'INTERRUPTED', 'UNKNOWN/STALE')


def _bar(done, total, width=16):
    filled = round(width * done / total) if total else 0
    return '[' + '#' * filled + '.' * (width - filled) + f'] {done}/{total}'


def _duration(value):
    if value is None:
        return 'unknown'
    value = int(value)
    return f'{value // 3600:02d}:{value // 60 % 60:02d}:{value % 60:02d}'


def _value(value):
    if not isinstance(value, str):
        return value
    try:
        return ast.literal_eval(value)
    except (ValueError, SyntaxError):
        return value


class Explorer:
    """Navigation and evidence interpretation, independent of terminal rendering."""

    def __init__(self, root):
        self.root = Path(root)
        self.path = ()
        self.selection = {}
        self.filter = ''
        self.filtering = False
        self.detail_offsets = {}
        self.evidence = None
        self.jobs = []
        self.verified_at = None
        self.error = None
        self._trial_cache = {}

    def breadcrumbs(self):
        if self.path == ('jobs',):
            return self.root.parent.name + ' / Jobs'
        labels = [self.root.parent.name or 'Study run']
        for i, key in enumerate(self.path):
            labels.append(('Rep ' + key) if i == 1 else ('Fold ' + key) if i == 2 else ('Trial ' + key) if i == 4 else key)
        return ' / '.join(labels)

    def _scope(self, job, scope):
        return (job.get('scenario'), str(job.get('repetition')), str(job.get('fold')))[:len(scope)] == tuple(scope)

    def scope_jobs(self, scope):
        return [job for job in self.jobs if self._scope(job, scope)]

    def record(self, scope, candidate):
        matches = []
        for job in self.scope_jobs(scope):
            entry = job.get('candidates', {}).get(candidate)
            if entry is None and job.get('candidate') == candidate:
                entry = job
            if isinstance(entry, dict):
                matches.append((entry.get('updated', job.get('updated', '')), entry, job))
        if not matches:
            return {}, None
        _, entry, job = max(matches, key=lambda item: item[0])
        return entry, job

    def candidate_state(self, name, scope=None):
        scope = tuple(scope or self.path[:3])
        try:
            candidate = self.evidence['scenarios'][scope[0]]['repetitions'][scope[1]]['candidates'][name]
            if candidate['state'] == 'invalid':
                return {'state': 'invalid', 'reason': candidate.get('reason', 'Invalid candidate provenance')}
            return candidate['folds'].get(scope[2], {'state': 'pending/unknown'})
        except (TypeError, KeyError, IndexError):
            return {}

    def job_state(self, job):
        if job.get('state') == 'failed':
            return 'FAILED'
        if job.get('state') == 'interrupted':
            return 'INTERRUPTED'
        if job.get('state') == 'finished':
            return 'FINISHED'  # Operational only; receipts decide scientific completion.
        if any(all(other.get(key) == job.get(key) for key in ('scenario', 'repetition', 'fold', 'candidate'))
                and other.get('started', '') > job.get('started', '') for other in self.jobs):
            return 'UNKNOWN/STALE'
        age = progress.seconds(job.get('heartbeat', job.get('updated')))
        if age is not None and age <= 3 * progress.HEARTBEAT_SECONDS:
            return 'RUNNING'
        return 'UNKNOWN/STALE'

    def unit_state(self, scope, candidate):
        scientific = self.candidate_state(candidate, scope)
        if scientific.get('state') == 'complete':
            return 'COMPLETE'
        if scientific.get('state') == 'invalid':
            return 'FAILED'
        record, job = self.record(scope, candidate)
        if job and job.get('candidate') == candidate and record.get('phase') != 'published':
            state = self.job_state(job)
            return state if state in STATES else 'UNKNOWN/STALE'
        if record or scientific.get('state') == 'incomplete' or not self.evidence:
            return 'UNKNOWN/STALE'
        return 'PENDING'

    def units(self, scope=()):
        for scenario, settings in config.VALIDATION_SETTINGS.items():
            for rep in range(1, len(settings['seeds']) + 1):
                for fold in range(1, config.GROUPED_OUTER_FOLDS + 1):
                    key = (scenario, str(rep), str(fold))
                    if key[:len(scope)] == tuple(scope):
                        for experiment in config.study_experiments():
                            yield key, experiment.name, self.unit_state(key, experiment.name)

    def counts(self, scope=()):
        counts = dict.fromkeys(STATES, 0)
        for _, _, state in self.units(scope):
            counts[state] += 1
        return counts

    def summary(self, scenario, rep=None, fold=None):
        scope = tuple(str(v) for v in (scenario, rep, fold) if v is not None)
        counts = self.counts(scope)
        label = scenario if rep is None else 'Repetition ' + rep if fold is None else 'Fold ' + fold
        total = sum(counts.values())
        complete = counts['COMPLETE']
        state = ('COMPLETE' if complete == total else 'FAILED' if counts['FAILED'] else
                 'RUNNING' if counts['RUNNING'] else 'INTERRUPTED' if counts['INTERRUPTED'] else
                 'UNKNOWN/STALE' if counts['UNKNOWN/STALE'] else 'PENDING')
        latest = {}
        for job in self.scope_jobs(scope):
            key = (job['scenario'], str(job['repetition']), str(job['fold']))
            if key not in latest or job.get('updated', '') > latest[key].get('updated', ''):
                latest[key] = job
        if any(self.job_state(job) == 'FAILED' and
               any(unit_state != 'COMPLETE' for _, _, unit_state in self.units(key))
               for key, job in latest.items()):
            state = 'FAILED'
        return f'{label:<18} {_bar(complete, total, width=8)} {state}'

    def current_job(self, name=None):
        if len(self.path) < 3:
            return None
        if name:
            return self.record(self.path[:3], name)[1]
        jobs = self.scope_jobs(self.path[:3])
        return max(jobs, key=lambda job: job.get('updated', ''), default=None)

    def job_label(self, job):
        state = self.job_state(job)
        record = job.get('candidates', {}).get(job.get('candidate'), job)
        trials = (f"{record.get('completed_trials', '?')}/{record.get('trial_budget', '?')}"
                  if record.get('search_method') != 'none' else 'no tuning')
        elapsed = progress.seconds(job.get('started'), None if state == 'RUNNING' else job.get('finished') or job.get('updated'))
        age = progress.seconds(job.get('heartbeat', job.get('updated')))
        return (f"{job.get('candidate') or 'preparing'} | {state} | reported {job.get('state', '?')} | "
                f"heartbeat {_duration(age)} ago | {job.get('scenario')} R{job.get('repetition')} F{job.get('fold')} | "
                f"{trials} | {_duration(elapsed)}")

    def entries(self):
        p = self.path
        if p == ('jobs',):
            jobs = sorted(self.jobs, key=lambda j: (self.job_state(j) == 'RUNNING', j.get('updated', '')), reverse=True)
            return [(j['_snapshot_id'], self.job_label(j)) for j in jobs]
        if not p:
            return [(s, self.summary(s)) for s in config.VALIDATION_SETTINGS]
        if len(p) == 1:
            return [(str(i), self.summary(p[0], str(i))) for i in range(1, len(config.VALIDATION_SETTINGS[p[0]]['seeds']) + 1)]
        if len(p) == 2:
            return [(str(i), self.summary(p[0], p[1], str(i))) for i in range(1, config.GROUPED_OUTER_FOLDS + 1)]
        if len(p) == 3:
            return [(e.name, self.model_label(e)) for e in config.study_experiments()]
        if len(p) == 4:
            return [(str(row['number']), f"Trial {row['number']:02d} | {row.get('state')} | objective {row.get('value')}"
                     + (' | WARNING' if row.get('numerical_warnings') else '')) for row in self.trials()]
        return []

    def filtered_entries(self):
        return [(key, label) for key, label in self.entries() if self.filter.lower() in label.lower()]

    def selected(self):
        entries = self.filtered_entries()
        key = self.selection.get(self.path)
        selected = next((i for i, row in enumerate(entries) if row[0] == key), 0)
        if entries:
            self.selection.setdefault(self.path, entries[selected][0])
        return selected

    def model_label(self, experiment):
        record, _ = self.record(self.path[:3], experiment.name)
        state = self.unit_state(self.path[:3], experiment.name)
        trials = ('no tuning' if experiment.search_method == 'none' else
                  f"{record.get('completed_trials', '?')}/{experiment.trial_budget} {experiment.search_method.upper()}")
        return (f'{experiment.name:<34} {state:<13} {trials} | '
                f'{experiment.estimator}/{experiment.predictor_family} | {record.get("phase", "awaiting evidence")}')

    def _folder(self):
        if len(self.path) < 4:
            return None
        scenario, rep, fold, name = self.path[:4]
        parent = self.root / scenario / f'rep-{int(rep):02d}'
        paths = list(parent.glob(name + '__*'))
        if len(paths) != 1:
            return None
        folder = paths[0] / f'fold-{int(fold):02d}'
        return folder if folder.resolve().is_relative_to(self.root.resolve()) else None

    def _parameters(self):
        if len(self.path) < 4 or self.candidate_state(self.path[3]).get('state') != 'complete':
            return {}
        folder = self._folder()
        try:
            value = json.loads((folder / 'parameters.json').read_text()) if folder else {}
            return value if isinstance(value, dict) else {}
        except (OSError, ValueError):
            return {}

    def trials(self):
        if len(self.path) < 4:
            return []
        record, _ = self.record(self.path[:3], self.path[3])
        folder = self._folder()
        complete = self.candidate_state(self.path[3]).get('state') == 'complete'
        try:
            if complete and folder:
                history = folder / 'trials.csv'
                if not history.exists() or not (folder / 'complete.json').exists():
                    return []
                sidecar = history.with_name(history.name + '.provenance.json')
                stat, receipt_stat = history.stat(), sidecar.stat()
                key = (history, stat.st_mtime_ns, stat.st_ctime_ns, stat.st_size,
                       receipt_stat.st_mtime_ns, receipt_stat.st_ctime_ns, receipt_stat.st_size)
                if self._trial_cache.get('key') == key:
                    return self._trial_cache['rows']
                raw = history.read_bytes()
                receipt = json.loads(sidecar.read_text())
                if (receipt.get('format_version') != 1
                        or receipt.get('signature') != json.loads((folder / 'complete.json').read_text())['identity']
                        or hashlib.sha256(raw).hexdigest() != receipt.get('sha256')):
                    return []
                with io.StringIO(raw.decode()) as handle:
                    rows = []
                    for raw in csv.DictReader(handle):
                        row = {'number': int(raw['number']), 'state': raw['state'], 'value': float(raw['value']),
                               'params': {k.removeprefix('params_'): _value(v) for k, v in raw.items() if k.startswith('params_') and v},
                               'recovered': _value(raw.get('recovered_from_previous_execution', 'False')),
                               'duration': raw.get('duration') or 'not recorded'}
                        row.update({k.removeprefix('user_attrs_'): _value(v) for k, v in raw.items() if k.startswith('user_attrs_') and v})
                        rows.append(row)
            else:
                checkpoint = Path(record['checkpoint'])
                checkpoint = (checkpoint if checkpoint.is_absolute() else self.root / checkpoint).resolve()
                if (not checkpoint.is_relative_to(self.root.resolve())
                        or checkpoint.name != 'search_trials.study.joblib'):
                    return []
                sidecar = checkpoint.with_name(checkpoint.name + '.provenance.json')
                stat, receipt_stat = checkpoint.stat(), sidecar.stat()
                key = (checkpoint, stat.st_mtime_ns, stat.st_ctime_ns, stat.st_size, receipt_stat.st_mtime_ns,
                       receipt_stat.st_ctime_ns, receipt_stat.st_size, record.get('recovered_trials', 0))
                if self._trial_cache.get('key') == key:
                    return self._trial_cache['rows']
                raw = checkpoint.read_bytes()
                receipt = json.loads(sidecar.read_text())
                if receipt.get('format_version') != 1 or hashlib.sha256(raw).hexdigest() != receipt.get('sha256'):
                    return []
                if record.get('search_signature', receipt.get('signature')) != receipt.get('signature'):
                    return []
                import joblib
                try:
                    saved = joblib.load(io.BytesIO(raw))
                except Exception:
                    return []
                if saved.get('signature') != receipt.get('signature'):
                    return []
                rows = [dict(number=t.number, state=t.state.name, value=t.value, params=t.params,
                             **t.user_attrs, duration=str(t.duration),
                             recovered=t.number < record.get('recovered_trials', 0))
                        for t in saved['study'].trials if t.state.name == 'COMPLETE']
            self._trial_cache = {'key': key, 'rows': rows}
            return rows
        except (OSError, ValueError, TypeError, KeyError, EOFError, AttributeError, csv.Error):
            return []

    def _scope_details(self, scope):
        counts = self.counts(scope)
        total = sum(counts.values())
        folds = {}
        for key, _, state in self.units(scope):
            folds.setdefault(key, []).append(state)
        completed = sum(all(state == 'COMPLETE' for state in values) for values in folds.values())
        jobs = self.scope_jobs(scope)
        active = [j for j in jobs if self.job_state(j) == 'RUNNING']
        starts = [j['started'] for j in jobs if j.get('started')]
        ends = [j.get('finished') or j.get('updated') for j in jobs if j.get('updated')]
        elapsed = progress.seconds(min(starts), None if active else max(ends)) if starts and ends else None
        warning_count = sum(j.get('warning_count', 0) for j in jobs)
        numerical_count = sum(j.get('numerical_warning_count', 0) for j in jobs)
        lines = [f"Scientific completion {_bar(counts['COMPLETE'], total)} ({100 * counts['COMPLETE'] / total:.1f}%)",
                 f'Outer folds {completed}/{len(folds)} | reported active tasks {len(active)} | numerical warnings {numerical_count} / all {warning_count}',
                 ' | '.join(f'{name} {count}' for name, count in counts.items()),
                 f'Observed wall span {_duration(elapsed)} (includes interruption gaps; not an ETA)']
        if len(scope) == 1:
            reps = len(config.VALIDATION_SETTINGS[scope[0]]['seeds'])
            complete_reps = sum(all(state == 'COMPLETE' for _, _, state in self.units((scope[0], str(rep))))
                                for rep in range(1, reps + 1))
            lines.append(f'Repetitions {complete_reps}/{reps}')
        if len(scope) == 3:
            lines.append('Inner tuning: 3 surfactant-grouped + 3 contaminant-grouped folds, training only')
        for job in active:
            lines.append(f"Active: {job.get('candidate') or 'preparing'} | "
                         f"{job.get('scenario')} R{job.get('repetition')} F{job.get('fold')}")
        for job in jobs:
            if job.get('error'):
                lines.append('Error: ' + job['error'])
        return lines

    def details(self):
        p = self.path
        if p == ('jobs',):
            lines = ['Recent heartbeats report activity; COMPLETE requires a verified receipt.']
            entries = self.filtered_entries()
            if entries:
                selected_id = entries[self.selected()][0]
                job = next(j for j in self.jobs if j['_snapshot_id'] == selected_id)
                lines.append(f"Last activity {job.get('updated', '?')} | heartbeat {job.get('heartbeat', '?')}")
                if job.get('error'):
                    lines.append('Error: ' + job['error'])
            return lines
        if len(p) <= 3:
            lines = self._scope_details(p)
            if not p:
                lines.insert(0, f'Study run {self.root.parent.name} | protocol {config.SCIENTIFIC_PROTOCOL}')
                lines.append('Receipt verification: ' + (self.verified_at or 'pending'))
                if self.error:
                    lines.append('Verification unavailable: ' + self.error)
            return lines
        if len(p) == 4:
            experiment = next(e for e in config.study_experiments() if e.name == p[3])
            record, job = self.record(p[:3], p[3])
            result = self.candidate_state(p[3])
            rows = self.trials() if experiment.search_method != 'none' else []
            parameters = self._parameters()
            best = min(rows, key=lambda t: (-float(t['value']), float(t.get('mean_relative_rmse', 0)), t['number']), default=None)
            running = job and job.get('candidate') == p[3] and self.job_state(job) == 'RUNNING'
            elapsed = progress.seconds(record.get('started'), None if running else record.get('finished') or record.get('updated'))
            lines = [f'{config.study_experiments().index(experiment) + 1:02d}/{len(config.study_experiments())} '
                     f'{experiment.estimator} | {experiment.predictor_family} | {self.unit_state(p[:3], p[3])}',
                     f"Phase {record.get('phase', 'not recorded')} | elapsed {_duration(elapsed)} | outer receipt {result.get('state', 'unknown')}"]
            if experiment.search_method == 'none':
                lines.append('No hyperparameter tuning required.')
            else:
                done = len(rows) if rows else record.get('completed_trials', 0)
                lines.append(f'{experiment.search_method.upper()} search {_bar(done, experiment.trial_budget)} confirmed trials')
                if best:
                    lines.append(f"Best trial {best['number']} | objective {best['value']:+.6g} | axis RMSE {best.get('axis_rmse', 'not recorded')}")
                elif record.get('best_objective') is not None:
                    lines.append(f"Best trial {record.get('best_trial')} | objective {record['best_objective']:+.6g} | axis RMSE {record.get('axis_rmse')}")
                if record.get('current_trial') is not None:
                    lines.append(f"Reported current trial {record['current_trial']} (not counted as complete)")
                lines.append(f"Recovered trials {record.get('recovered_trials', sum(t.get('recovered') is True for t in rows) if rows else 'not recorded')}")
            params = parameters.get('model_params', record.get('final_parameters'))
            if params is not None:
                lines.append('Final fitted parameters: ' + str(params))
            if result.get('state') == 'complete':
                lines.append(f"Outer RMSE {result.get('RMSE_outer'):.6g} | R2 {result.get('R2_outer'):.6g}")
            if result.get('reason'):
                lines.append('Receipt issue: ' + result['reason'])
            lines.append(f"Numerical warnings {record.get('numerical_warning_count', 0)} / all warnings {record.get('warning_count', 0)}")
            if experiment.search_method != 'none':
                lines.append('Checkpoint/history: ' + ('available' if rows else 'not currently readable'))
            lines.extend(record.get('warnings', []))
            if job:
                lines.append('Job: ' + self.job_label(job))
            return lines
        rows = self.trials()
        trial = next((t for t in rows if str(t['number']) == p[4]), None)
        if trial is None:
            return ['Trial evidence unavailable; no computation is triggered by this view.']
        lines = [f"Trial {trial['number']} | {trial['state']} | duration {trial.get('duration', 'not recorded')}",
                 f"Hyperparameters: {trial.get('params')} | molecular k: {trial.get('params', {}).get('k', 'fixed panel')}",
                 f"Objective {trial['value']} | pooled axis RMSE {trial.get('axis_rmse', 'not recorded')}",
                 f"Recovered from checkpoint: {trial.get('recovered', 'not recorded')}"]
        lines.extend(f"Inner fold {i}: RMSE {trial.get(f'inner_fold_{i}_rmse', 'not recorded')} "
                     f"({config.PRIMARY_SETTINGS[(i - 1) // config.GROUPED_INNER_FOLDS]}, "
                     f"fold {(i - 1) % config.GROUPED_INNER_FOLDS + 1})"
                     for i in range(1, 2 * config.GROUPED_INNER_FOLDS + 1))
        lines.extend(trial.get('numerical_warnings', []) or ['Extreme-error warnings: none recorded'])
        lines.extend('Warning: ' + message for message in trial.get('warnings', [])
                     if message not in trial.get('numerical_warnings', []))
        return lines

    def open(self, key):
        if self.path == ('jobs',):
            job = next((j for j in self.jobs if j['_snapshot_id'] == key), None)
            if job:
                self.path = (job['scenario'], str(job['repetition']), str(job['fold']))
                if job.get('candidate') in {e.name for e in config.study_experiments()}:
                    self.path += (job['candidate'],)
        elif key in {k for k, _ in self.entries()}:
            self.path += (key,)
        self.filter = ''
        self.filtering = False

    def handle(self, key):
        """Return False only on quit; navigation remains stable across refreshes."""
        if self.filtering:
            if key in ('ENTER', 'ESC'):
                self.filtering = False
            elif key == 'BACKSPACE':
                self.filter = self.filter[:-1]
            elif len(key) == 1 and key.isprintable():
                self.filter += key
            return True
        if key == 'q':
            return False
        if key == 'g':
            self.path = (); self.filter = ''
        elif key == 'a':
            self.path = ('jobs',); self.filter = ''
        elif key == 'j':
            active = [j for j in self.jobs if self.job_state(j) == 'RUNNING']
            if active:
                job = max(active, key=lambda j: j.get('updated', ''))
                self.path = ('jobs',); self.open(job['_snapshot_id'])
        elif key == '/':
            self.filtering = True
        elif key in ('LEFT', 'ESC'):
            self.path = self.path[:-1]; self.filter = ''
        elif key in ('PGUP', 'PGDOWN'):
            self.detail_offsets[self.path] = max(0, self.detail_offsets.get(self.path, 0) + (5 if key == 'PGDOWN' else -5))
        else:
            entries = self.filtered_entries()
            if not entries:
                return True
            selected = self.selected()
            if key in ('UP', 'DOWN'):
                selected = max(0, min(len(entries) - 1, selected + (1 if key == 'DOWN' else -1)))
                self.selection[self.path] = entries[selected][0]
            elif key in ('ENTER', 'RIGHT'):
                self.selection[self.path] = entries[selected][0]
                self.open(entries[selected][0])
        return True


class Monitor:
    """Filesystem I/O stays off the terminal thread."""

    def __init__(self, explorer, interval=2):
        self.explorer = explorer
        self.interval = interval
        self.stop = threading.Event()
        self.refresh = threading.Event()
        self.cache = FoldVerificationCache()
        self.thread = None

    def start(self):
        def poll():
            last_scientific = 0.
            publication = None
            while not self.stop.is_set():
                now = time.monotonic()
                self.explorer.jobs = progress.snapshots(self.explorer.root)
                published = tuple((j['_snapshot_id'], tuple((k, v.get('finished')) for k, v in j.get('candidates', {}).items()))
                                  for j in self.explorer.jobs)
                force = self.refresh.is_set()
                if force:
                    self.refresh.clear()
                    self.cache.entries.clear()
                if force or now - last_scientific >= 30 or published != publication:
                    try:
                        evidence = inspect_status(self.explorer.root, fold_cache=self.cache)
                        self.explorer.evidence = evidence
                        self.explorer.verified_at = progress.timestamp()
                        self.explorer.error = None
                    except Exception as error:
                        self.explorer.evidence = None
                        self.explorer.error = str(error)
                    publication = published
                    last_scientific = now
                self.explorer.jobs = progress.snapshots(self.explorer.root)
                self.refresh.wait(self.interval)
        self.thread = threading.Thread(target=poll, daemon=True)
        self.thread.start()

    def close(self):
        self.stop.set()
        self.refresh.set()
        if self.thread:
            self.thread.join(timeout=.2)


def render(win, explorer, colors=None):
    height, width = win.getmaxyx()
    win.erase()
    colors = colors or {}
    def line(y, value, attr=0):
        if 0 <= y < height and width > 1:
            try:
                win.addnstr(y, 0, value, width - 1, attr)
            except curses.error:
                pass
    if height < 6 or width < 24:
        line(0, 'Terminal too small; resize')
        line(height - 1, 'q quit')
        win.noutrefresh()
        curses.doupdate()
        return
    line(0, ' ML / MSR   |   ' + explorer.breadcrumbs(),
         curses.A_BOLD | colors.get('header', 0))
    entries = explorer.filtered_entries()
    details = [part for value in explorer.details() for part in textwrap.wrap(value, max(1, width - 1)) or ['']]
    detail_height = min(len(details), max(0, (height - 5) // 2)) if entries else max(0, height - 4)
    offset = min(explorer.detail_offsets.get(explorer.path, 0), max(0, len(details) - detail_height))
    explorer.detail_offsets[explorer.path] = offset
    for i, value in enumerate(details[offset:offset + detail_height]):
        line(i + 1, value)
    if len(details) > detail_height and detail_height:
        line(detail_height, details[offset + detail_height - 1] + ' [PgUp/PgDn]')
    line(detail_height + 1, '  ' + '-' * max(0, width - 4), colors.get('muted', 0))
    start = detail_height + 2
    visible = max(0, height - start - 2)
    selected = explorer.selected()
    scroll = max(0, selected - visible + 1)
    for i, (_, label) in enumerate(entries[scroll:scroll + visible]):
        chosen = scroll + i == selected
        line(start + i, ('> ' if chosen else '  ') + label,
             (curses.A_REVERSE | curses.A_BOLD) if chosen else colors.get('item', 0))
    if height >= 2:
        hints = 'Arrows Enter open | Esc back | g home | a jobs | j active | / filter | r refresh | q quit'
        if width < 80:
            hints = 'Enter open  Esc back  g home  a jobs  / search  q quit'
        line(height - 2, ('Filter: ' + explorer.filter + '_') if explorer.filtering else hints,
             colors.get('header', 0))
    footer = ('PgUp/PgDn scroll details.' if width < 80 else
              'Scientific units are not wall-time estimates. PgUp/PgDn scroll details.')
    line(height - 1, ('Filter: ' + explorer.filter) if explorer.filter and not explorer.filtering else
             footer, colors.get('muted', 0))
    win.noutrefresh()
    curses.doupdate()


def run(root, interval=2):
    if not math.isfinite(interval) or interval <= 0:
        raise ValueError('Refresh interval must be finite and positive.')
    explorer = Explorer(root)
    monitor = Monitor(explorer, interval)
    monitor.start()
    def screen(win):
        colors = {}
        try:
            if curses.has_colors():
                curses.start_color()
                curses.use_default_colors()
                curses.init_pair(1, curses.COLOR_CYAN, -1)
                colors = {'header': curses.color_pair(1), 'muted': curses.A_DIM}
        except curses.error:
            pass
        try:
            curses.curs_set(0)
        except curses.error:
            pass
        win.keypad(True)
        win.timeout(max(50, min(250, int(interval * 1000))))
        keys = {curses.KEY_UP: 'UP', curses.KEY_DOWN: 'DOWN', curses.KEY_LEFT: 'LEFT',
                curses.KEY_RIGHT: 'RIGHT', curses.KEY_PPAGE: 'PGUP', curses.KEY_NPAGE: 'PGDOWN',
                curses.KEY_BACKSPACE: 'BACKSPACE', 127: 'BACKSPACE', 8: 'BACKSPACE',
                10: 'ENTER', 13: 'ENTER', 27: 'ESC'}
        last = 0.
        while True:
            now = time.monotonic()
            if now - last >= interval:
                render(win, explorer, colors)
                last = now
            code = win.getch()
            if code == -1:
                continue
            key = keys.get(code, chr(code) if 0 <= code < 256 else '')
            if key == 'r' and not explorer.filtering:
                monitor.refresh.set()
            if not explorer.handle(key):
                return
            render(win, explorer, colors)
            last = time.monotonic()
    try:
        curses.wrapper(screen)
    finally:
        monitor.close()
