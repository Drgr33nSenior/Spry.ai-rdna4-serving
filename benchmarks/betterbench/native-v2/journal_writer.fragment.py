"""BetterBench-only durable two-record request journals; no witness is dropped."""
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
