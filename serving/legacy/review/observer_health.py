"""Bounded host-only observer state and fail-closed final receipts."""
import json
import os
from pathlib import Path


def canonical(value):
    return json.dumps(value, sort_keys=True, separators=(',', ':'), allow_nan=False).encode() + b'\n'


class Saturated(ValueError):
    """An evidence counter or byte allowance is exhausted."""


class Health:
    def __init__(self, observer, rank):
        self.observer = observer
        self.rank = rank
        self.failed = False
        self.saturated = False
        self.failure_type = None
        self.closed = False

    def fail(self, error):
        if self.failed:
            return
        self.failed = True
        self.saturated = isinstance(error, Saturated)
        self.failure_type = type(error).__name__
        try:
            print('SPRY_OBSERVER_FAILED ' + canonical(dict(
                observer=self.observer, rank=self.rank, observer_failed=True,
                saturated=self.saturated, failure_type=self.failure_type)).decode().strip(), flush=True)
        except Exception:
            pass

    def call(self, callback, *args, **kwargs):
        if self.failed or self.closed:
            return None
        try:
            return callback(*args, **kwargs)
        except Exception as error:
            self.fail(error)
            return None

    def finish(self, manifest, worker, helper_source, counters):
        if self.closed:
            return
        self.closed = True
        try:
            record = {key: manifest[key] for key in RUN_FIELDS}
            record.update({key: worker[key] for key in WORKER_FIELDS})
            record.update(schema=1, observer=self.observer,
                worker_identity_sha256=worker['identity_sha256'], helper_source=helper_source,
                counters=counters, observer_failed=self.failed, saturated=self.saturated,
                failure_type=self.failure_type, complete=True)
            raw = canonical(record)
            if len(raw) > 4096:
                raise Saturated('health receipt bytes')
            path = Path(manifest['evidence_dir']) / filename(self.observer, self.rank)
            pending = path.with_suffix('.pending')
            fd = os.open(pending, os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW, 0o600)
            with os.fdopen(fd, 'wb') as stream:
                if stream.write(raw) != len(raw):
                    raise OSError('short observer health write')
                stream.flush()
                os.fsync(stream.fileno())
            # Never overwrite a prior receipt. A pending file or extra link
            # remains on error and is rejected by collection/readout.
            os.link(pending, path)
            pending.unlink()
        except Exception as error:
            self.fail(error)


RUN_FIELDS = ('mode', 'arm', 'session_sha256', 'spec_sha256', 'source_manifest_sha256')
WORKER_FIELDS = ('rank', 'pid', 'start_ticks', 'boot_id')


def filename(observer, rank):
    if observer not in ('graph', 'target_kv') or type(rank) is not int or rank not in (0, 1):
        raise ValueError('observer health name/rank')
    return f'OBSERVER-HEALTH-{observer}-rank-{rank}.json'


def validate(record, *, binding, worker, observer, helper_source, limits):
    fields = {*RUN_FIELDS, *WORKER_FIELDS, 'schema', 'observer', 'worker_identity_sha256',
              'helper_source', 'counters', 'observer_failed', 'saturated', 'failure_type', 'complete'}
    if type(record) is not dict or set(record) != fields:
        raise ValueError('observer health schema')
    if type(record['schema']) is not int or record['schema'] != 1 or record['observer'] != observer:
        raise ValueError('observer health version/name')
    for keys, expected in ((RUN_FIELDS, binding), (WORKER_FIELDS, worker)):
        if any(type(record[key]) is not type(expected[key]) or record[key] != expected[key] for key in keys):
            raise ValueError('observer health run/worker binding')
    if record['worker_identity_sha256'] != worker['identity_sha256'] or record['helper_source'] != helper_source:
        raise ValueError('observer health worker/source binding')
    if (record['complete'] is not True or record['observer_failed'] is not False or
            record['saturated'] is not False or record['failure_type'] is not None):
        raise ValueError('observer health incomplete or failed')
    counters = record['counters']
    if (type(counters) is not dict or set(counters) != set(limits) or
            any(type(counters[key]) is not int or not 0 <= counters[key] <= limit
                for key, limit in limits.items())):
        raise ValueError('observer health counter bound')
    return counters
