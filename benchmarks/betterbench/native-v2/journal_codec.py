"""Portable adapter of the exact reviewed journal functions.

This standalone module was not transported as a module. Function ASTs match
the consumed runtime/readout; full native/model ownership validation is separate.
"""
from __future__ import annotations
import hashlib,json,os,stat
from pathlib import Path
from typing import Any

class ServingRuntimeError(ValueError):
    """A serving-screen binding cannot safely start native HTTP."""

def _require(value: bool, message: str) -> None:
    if not value:
        raise ServingRuntimeError(message)

def _canonical(value: Any) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode("utf-8")

def _bb_write_observation(native, row):
    phase, ordinal = row['phase'], row['ordinal']
    _require(phase in ('before', 'after') and type(ordinal) is int and 0 <= ordinal < 53,
             'bounded BetterBench observation ordinal/phase')
    path = native / f'request-{ordinal:02d}.jsonl'
    raw = _canonical(row) + b'\n'
    _require(len(raw) <= 8192, 'BetterBench observation line bound')
    before = prior = None
    if phase == 'after':
        descriptor = os.open(path, os.O_RDONLY | os.O_NOFOLLOW)
        try:
            prior = os.fstat(descriptor)
            _require(stat.S_ISREG(prior.st_mode) and prior.st_nlink == 1 and prior.st_uid == os.getuid()
                     and 0 < prior.st_size <= 8192, 'owned single before observation')
            with os.fdopen(descriptor, 'rb', closefd=False) as stream:
                old = stream.read(8193)
            before = json.loads(old)
            _require(_canonical(before) + b'\n' == old and before.get('phase') == 'before'
                     and before.get('ordinal') == ordinal and before.get('body_sha256') == row['body_sha256'],
                     'canonical matching BetterBench before observation')
        finally:
            os.close(descriptor)
    flags = os.O_WRONLY | os.O_NOFOLLOW | (os.O_CREAT | os.O_EXCL if phase == 'before' else os.O_APPEND)
    descriptor = os.open(path, flags, 0o600)
    try:
        current = os.fstat(descriptor)
        _require(stat.S_ISREG(current.st_mode) and current.st_nlink == 1 and current.st_uid == os.getuid(),
                 'owned regular BetterBench journal')
        if prior is not None:
            _require((current.st_dev, current.st_ino, current.st_size) ==
                     (prior.st_dev, prior.st_ino, prior.st_size), 'BetterBench before journal changed')
        with os.fdopen(descriptor, 'wb', closefd=False) as stream:
            stream.write(raw)
            stream.flush()
            os.fsync(descriptor)
    finally:
        os.close(descriptor)
    return before

def _need(value, message):
    if not value:
        raise ValueError('HTTP integration: ' + message)

def _json(raw):

    def pairs(items):
        value = {}
        for key, item in items:
            _need(key not in value, 'duplicate JSON key')
            value[key] = item
        return value
    return json.loads(raw, object_pairs_hook=pairs, parse_constant=lambda _: (_ for _ in ()).throw(ValueError('nonfinite JSON')))

def _bb_expand_observations(observations):
    result = dict(observations)
    storage = {}
    for ordinal in range(53):
        name = f'request-{ordinal:02d}.jsonl'
        raw = result.pop(name, None)
        _need(type(raw) is bytes and 0 < len(raw) <= 16384, 'bounded complete BetterBench journal')
        lines = raw.splitlines(keepends=True)
        _need(len(lines) == 2, 'exact two BetterBench observation records')
        logical = {}
        for phase, line in zip(('before', 'after'), lines):
            _need(0 < len(line) <= 8192 and line.endswith(b'\n'), 'bounded terminated BetterBench observation')
            row = _json(line)
            canonical = json.dumps(row, sort_keys=True, separators=(',', ':'), allow_nan=False).encode()
            _need(canonical + b'\n' == line and type(row) is dict and row.get('phase') == phase
                  and type(row.get('ordinal')) is int and row['ordinal'] == ordinal,
                  'canonical ordered BetterBench observation')
            old_name = f'request-{ordinal:02d}-{phase}.json'
            _need(old_name not in result, 'mixed legacy/journal BetterBench observations')
            result[old_name] = canonical
            logical[old_name] = hashlib.sha256(canonical).hexdigest()
        storage[name] = dict(bytes=len(raw), sha256=hashlib.sha256(raw).hexdigest(), logical_sha256=logical)
    _need(not any(name.startswith('request-') and name.endswith('.jsonl') for name in result),
          'unexpected BetterBench journal ordinal')
    return result, dict(mode='betterbench-two-record-observation-journal-v1', journals=storage,
                        logical_records=106, lossless=True, protected_export_limits_changed=False)

write_observation = _bb_write_observation
expand_observations = _bb_expand_observations
