"""Expand authenticated journal bytes before unchanged logical witness checks."""
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
