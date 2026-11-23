"""Atomic operational reporting; scientific receipts remain authoritative."""

from contextlib import contextmanager
from contextvars import ContextVar
from datetime import datetime, timezone
import sys
import json
import math
import os
from pathlib import Path
import tempfile
import threading
import uuid
import warnings

from msr_qspr.config import SCIENTIFIC_PROTOCOL
from msr_qspr import artifacts, config

_current = ContextVar('study_run_progress', default=None)
HEARTBEAT_SECONDS = 15
_warning_lock = threading.Lock()
_warning_users = 0
_original_showwarning = None


def _showwarning(message, category, filename, lineno, file=None, line=None):
    reporter = _current.get()
    if reporter is not None:
        numerical = issubclass(category, RuntimeWarning) or category.__name__ in ('ConvergenceWarning', 'LinAlgWarning')
        reporter.warn(category.__name__ + ': ' + str(message), numerical=numerical)
    _original_showwarning(message, category, filename, lineno, file=file, line=line)


@contextmanager
def _observe_warnings():
    global _warning_users, _original_showwarning
    with _warning_lock:
        if _warning_users == 0:
            _original_showwarning = warnings.showwarning
            warnings.showwarning = _showwarning
        _warning_users += 1
    try:
        yield
    finally:
        with _warning_lock:
            _warning_users -= 1
            if _warning_users == 0 and warnings.showwarning is _showwarning:
                warnings.showwarning = _original_showwarning


def timestamp():
    return datetime.now(timezone.utc).isoformat()


def seconds(start, end=None):
    try:
        end = datetime.fromisoformat(end) if end else datetime.now(timezone.utc)
        return max(0., (end - datetime.fromisoformat(start)).total_seconds())
    except (TypeError, ValueError):
        return None


class Reporter:
    """One writer namespace per fold attempt, with a small candidate history."""

    def __init__(self, root, scenario, repetition, fold, candidate=None):
        self.path = Path(root) / '.progress' / (uuid.uuid4().hex + '.json')
        self.state = dict(version=1, scenario=scenario, repetition=repetition, fold=fold,
                          started=timestamp(),
                          protocol=SCIENTIFIC_PROTOCOL, scientific_identity=config.scientific_identity(),
                          candidate_registry_sha256=artifacts.candidate_registry_fingerprint(),
                          state='running', phase='preparing',
                          candidates={}, warning_count=0, numerical_warning_count=0, warnings=[])
        if candidate is not None:
            self.state['candidate'] = candidate
        self.failed = False
        self.lock = threading.RLock()
        self.stop = threading.Event()
        self.thread = None
        self.trial_warnings = []

    def _write(self):
        try:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            fd, name = tempfile.mkstemp(prefix='.progress-', dir=self.path.parent)
            try:
                with os.fdopen(fd, 'w') as handle:
                    json.dump(self.state, handle, allow_nan=False)
                    handle.flush()
                os.replace(name, self.path)
            finally:
                if os.path.exists(name):
                    os.unlink(name)
        except (OSError, ValueError, TypeError) as error:
            if not self.failed:
                self.failed = True
                # Logging must not turn a missing operational snapshot into a failed fit.
                print('[Progress unavailable] ' + str(error), flush=True, file=sys.stderr)

    def update(self, **values):
        with self.lock:
            for field in ('checkpoint', 'fold_directory'):
                if values.get(field):
                    try:
                        values[field] = str(Path(values[field]).resolve().relative_to(self.path.parent.parent.resolve()))
                    except (OSError, ValueError):
                        values[field] = None
            now = timestamp()
            self.state.update(values, updated=now, heartbeat=now)
            if values.get('current_trial') is not None:
                self.trial_warnings = []
            candidate = self.state.get('candidate')
            if candidate:
                record = self.state['candidates'].setdefault(candidate, {'started': now, 'warnings': [], 'warning_count': 0})
                if 'phase' in values and values['phase'] != record.get('phase'):
                    record.setdefault('phases', {})[values['phase']] = now
                record.update(values, updated=now)
                if values.get('phase') == 'published':
                    record['finished'] = now
            if values.get('state') in ('failed', 'interrupted', 'finished'):
                self.state['finished'] = now
            self._write()

    def warn(self, message, *, numerical=False):
        with self.lock:
            if self.state.get('current_trial') is not None and message not in self.trial_warnings:
                self.trial_warnings.append(message)
            self.state['warning_count'] += 1
            if numerical:
                self.state['numerical_warning_count'] += 1
            if message not in self.state['warnings'] and len(self.state['warnings']) < 8:
                self.state['warnings'].append(message)
            candidate = self.state.get('candidate')
            if candidate:
                record = self.state['candidates'].setdefault(candidate, {'started': timestamp(), 'warnings': [], 'warning_count': 0})
                record['warning_count'] += 1
                if numerical:
                    record['numerical_warning_count'] = record.get('numerical_warning_count', 0) + 1
                if message not in record['warnings'] and len(record['warnings']) < 8:
                    record['warnings'].append(message)
            self._write()

    def start(self):
        self.update()
        def heartbeat():
            while not self.stop.wait(HEARTBEAT_SECONDS):
                with self.lock:
                    self.state['heartbeat'] = timestamp()
                    self._write()
        self.thread = threading.Thread(target=heartbeat, daemon=True)
        self.thread.start()

    def close(self):
        self.stop.set()
        if self.thread:
            self.thread.join(timeout=1)


@contextmanager
def fold(root, scenario, repetition, number, *, candidate=None):
    reporter = Reporter(root, scenario, repetition, number, candidate=candidate)
    token = _current.set(reporter)
    reporter.start()
    try:
        with _observe_warnings():
            yield reporter
    except BaseException as error:
        state = 'interrupted' if isinstance(error, (KeyboardInterrupt, SystemExit)) else 'failed'
        reporter.update(state=state, phase=state, error=type(error).__name__ + ': ' + str(error))
        raise
    else:
        reporter.update(state='finished', phase='finished')
    finally:
        reporter.close()
        _current.reset(token)


def update(**values):
    reporter = _current.get()
    if reporter is not None:
        reporter.update(**values)


def log(message):
    reporter = _current.get()
    if reporter is not None:
        scope = reporter.state
        prefix = f"[{scope['scenario']} | rep {scope['repetition']} | fold {scope['fold']}] "
    else:
        prefix = ''
    print(prefix + message, flush=True, file=sys.stderr)


def numerical_warning(message):
    """Record and log a diagnostic without rejecting or changing a trial."""
    reporter = _current.get()
    if reporter is not None:
        reporter.warn(message, numerical=True)
    log('[Numerical warning] ' + message)


def trial_warnings():
    reporter = _current.get()
    return list(reporter.trial_warnings) if reporter is not None else []


def snapshots(root):
    """Read valid known-scope snapshots, without repairing or writing anything."""
    root = Path(root)
    names = {e.name for e in config.study_experiments()}
    def valid_record(record):
        if not isinstance(record, dict):
            return False
        for key in ('warning_count', 'numerical_warning_count', 'completed_trials', 'recovered_trials', 'trial_budget', 'current_trial', 'best_trial'):
            value = record.get(key)
            if value is not None and (type(value) is not int or value < 0):
                return False
        for key in ('started', 'updated', 'heartbeat', 'finished', 'checkpoint', 'phase', 'error'):
            if record.get(key) is not None and not isinstance(record[key], str):
                return False
        value = record.get('best_objective')
        if value is not None and (type(value) not in (int, float) or not math.isfinite(value)):
            return False
        messages = record.get('warnings', [])
        return isinstance(messages, list) and all(isinstance(message, str) for message in messages)
    result = []
    try:
        paths = list((Path(root) / '.progress').glob('*.json'))
    except OSError:
        return result
    for path in paths:
        try:
            value = json.loads(path.read_text())
            scenario = value.get('scenario')
            if (value.get('version') != 1 or value.get('protocol') != SCIENTIFIC_PROTOCOL
                    or value.get('scientific_identity') != config.scientific_identity()
                    or value.get('candidate_registry_sha256') != artifacts.candidate_registry_fingerprint()
                    or scenario not in config.VALIDATION_SETTINGS
                    or type(value.get('repetition')) is not int or type(value.get('fold')) is not int
                    or not 1 <= value['repetition'] <= len(config.VALIDATION_SETTINGS[scenario]['seeds'])
                    or not 1 <= value['fold'] <= config.GROUPED_OUTER_FOLDS
                    or value.get('state') not in ('running', 'failed', 'interrupted', 'finished')
                    or not isinstance(value.get('candidates', {}), dict)
                    or any(k not in names or not valid_record(v) for k, v in value.get('candidates', {}).items())
                    or not valid_record(value)):
                continue
            value['_snapshot_id'] = path.stem
            result.append(value)
        except (OSError, ValueError, TypeError, AttributeError):
            continue
    return sorted(result, key=lambda value: (value.get('started', ''), value['_snapshot_id']))
