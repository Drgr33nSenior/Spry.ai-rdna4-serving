"""Read-only OCI-to-expanded-runtime verification. Never imports the runtime.

The content inventory covers names, types, file bytes/executable bits and exact
link targets. It excludes timestamps and numeric archive owners: the retained
rootless extractor intentionally did not preserve those owners or permissions.
No archive member is extracted, executed or followed outside the inventory.
"""
import argparse
import gzip
import hashlib
import json
import os
from pathlib import Path, PurePosixPath
import signal
import stat
import sys
import time


MANIFEST_SHA256 = '1d3ebef7da1df35f0c103ed038521933ebaeaf6f8c8d710168d8c294efdc59d9'
CONFIG_SHA256 = '6ce2bb86684c1b0551d2efa87c765966845f784bf8b038929bbae7c3dbe95f14'
BASE_CONTENT_SHA256 = '1808051ef37965fc1f7ca4a454f5db28241f72a277824d22c0027d2d67248a9d'
BASE_MEMBERS = 96087
PREFIX_SHA256 = 'f251329192588a73a5c74af6e33a37284e7317146e3d66fcdf9be123b0b8ccf8'
PREFIX_MEMBERS = 44881
DERIVATION_REPORT_SHA256 = '567de77c86387d2d830c13652e7c7bbbe35ab0fa8aa315298afbcda2bac9b9c2'
DERIVATION_SOURCE_SHA256 = 'a632e0b2b8a059ec3f4c175f0a5baff78bed3a47249ac3576f8cfd41c5f504c2'
MAX_MEMBERS = 500000
MAX_PAYLOAD = 150 << 30
MAX_OUTPUT = 64 << 20


def need(value, message):
    if not value:
        raise ValueError(message)


def canonical(value):
    return (json.dumps(value, sort_keys=True, separators=(',', ':'), allow_nan=False) + '\n').encode()


def sha(raw):
    return hashlib.sha256(raw).hexdigest()


def pairs(items):
    result = {}
    for name, value in items:
        need(name not in result, 'duplicate JSON key')
        result[name] = value
    return result


def json_bytes(raw):
    return json.loads(raw, object_pairs_hook=pairs,
        parse_constant=lambda _: (_ for _ in ()).throw(ValueError('nonfinite JSON')))


def deadline_check(deadline):
    need(time.monotonic() < deadline, 'runtime verification deadline')


def relative(value, *, root=False):
    need(type(value) is str and '\x00' not in value and '\\' not in value, 'unsafe member name')
    while value.startswith('./'):
        value = value[2:]
    value = value.rstrip('/')
    path = PurePosixPath(value)
    need(not path.is_absolute() and '..' not in path.parts, 'unsafe member path')
    need(root or value not in ('', '.'), 'empty member path')
    return '' if value in ('', '.') else str(path)


def stamp(value):
    return tuple(getattr(value, name) for name in ('st_dev', 'st_ino', 'st_mode', 'st_nlink',
        'st_uid', 'st_gid', 'st_size', 'st_mtime_ns', 'st_ctime_ns'))


def hash_stream(stream, deadline, *, maximum=MAX_PAYLOAD):
    result, size = hashlib.sha256(), 0
    while True:
        deadline_check(deadline)
        raw = stream.read(8 << 20)
        if not raw:
            break
        size += len(raw)
        need(size <= maximum, 'stream size bound')
        result.update(raw)
    return result.hexdigest(), size


def read_pinned_json(path, pin, deadline, maximum=1 << 20):
    fd = os.open(path, os.O_RDONLY | os.O_NOFOLLOW)
    with os.fdopen(fd, 'rb') as stream:
        before = os.fstat(fd)
        need(stat.S_ISREG(before.st_mode) and before.st_size <= maximum, 'bounded image metadata')
        deadline_check(deadline)
        raw = stream.read(maximum + 1)
        need(sha(raw) == pin and len(raw) == before.st_size
            and stamp(before) == stamp(os.fstat(fd)) == stamp(Path(path).lstat()), 'metadata pin changed')
        return json_bytes(raw)


class HashReader:
    def __init__(self, stream, deadline):
        self.stream, self.deadline = stream, deadline
        self.hash, self.count = hashlib.sha256(), 0

    def read(self, size=-1):
        deadline_check(self.deadline)
        raw = self.stream.read(size)
        self.count += len(raw)
        need(self.count <= MAX_PAYLOAD, 'layer stream bound')
        self.hash.update(raw)
        return raw


def erase(tree, name, *, contents_only=False):
    for key in list(tree):
        if key.startswith(name + '/') or not contents_only and key == name:
            del tree[key]


def apply_layer(tree, entries, whiteouts):
    """Apply whiteouts only to the lower tree, then ordered upper entries."""
    for path in whiteouts:
        parent, _, leaf = path.rpartition('/')
        if leaf == '.wh..wh..opq':
            if parent:
                erase(tree, parent, contents_only=True)
            else:
                tree.clear()
        else:
            need(leaf.startswith('.wh.') and leaf[4:] not in ('', '.', '..'), 'invalid whiteout')
            erase(tree, (parent + '/' if parent else '') + leaf[4:])
    deferred = []
    for name, record in entries:
        parents = PurePosixPath(name).parents
        for parent in reversed(parents):
            key = str(parent)
            if key == '.':
                continue
            need(key not in tree or tree[key]['kind'] == 'directory', 'layer writes through non-directory parent')
            tree.setdefault(key, {'kind': 'directory'})
        if record['kind'] == 'hardlink':
            target = tree.get(record['target'])
            if target is None:
                deferred.append((name, record))
                continue
            need(target['kind'] == 'file', 'hardlink does not target a regular file')
            record = dict(target)
        if name in tree and tree[name]['kind'] == 'directory' and record['kind'] != 'directory':
            erase(tree, name)
        tree[name] = record
    for _ in range(len(deferred) + 1):
        if not deferred:
            return
        pending = []
        for name, record in deferred:
            target = tree.get(record['target'])
            if target is None:
                pending.append((name, record))
            else:
                need(target['kind'] == 'file', 'hardlink target is not regular')
                if name in tree and tree[name]['kind'] == 'directory':
                    erase(tree, name)
                tree[name] = dict(target)
        need(len(pending) < len(deferred), 'unresolved or cyclic hardlink')
        deferred = pending


def derive_image(image_root, deadline, *, manifest_pin=MANIFEST_SHA256, config_pin=CONFIG_SHA256,
                 progress=lambda value: None):
    import tarfile
    root = Path(image_root)
    manifest = read_pinned_json(root / 'base-manifest.json', manifest_pin, deadline)
    config = read_pinned_json(root / 'base-config.json', config_pin, deadline)
    need(manifest['config']['digest'] == 'sha256:' + config_pin and
        0 < len(manifest['layers']) == len(config['rootfs']['diff_ids']) <= 64, 'image/config/layer binding')
    tree, records, count, payload = {}, [], 0, 0
    for index, (layer, diff_id) in enumerate(zip(manifest['layers'], config['rootfs']['diff_ids'])):
        pin = layer['digest'].removeprefix('sha256:')
        need(len(pin) == 64 and all(value in '0123456789abcdef' for value in pin), 'layer digest shape')
        blob = root / 'base/blobs/sha256' / pin
        fd = os.open(blob, os.O_RDONLY | os.O_NOFOLLOW)
        entries, whiteouts = [], []
        with os.fdopen(fd, 'rb') as stream:
            before = os.fstat(fd)
            need(stat.S_ISREG(before.st_mode) and before.st_size == layer['size'], 'layer type/size differs')
            compressed = HashReader(stream, deadline)
            with gzip.GzipFile(fileobj=compressed, mode='rb') as unzipped:
                plain = HashReader(unzipped, deadline)
                with tarfile.open(fileobj=plain, mode='r|') as archive:
                    for item in archive:
                        deadline_check(deadline)
                        count, payload = count + 1, payload + item.size
                        need(count <= MAX_MEMBERS and payload <= MAX_PAYLOAD, 'archive count/payload bound')
                        name = relative(item.name, root=True)
                        if not name:
                            need(item.isdir(), 'non-directory root member')
                            continue
                        if PurePosixPath(name).name.startswith('.wh.'):
                            whiteouts.append(name)
                            continue
                        if item.isfile():
                            value, size = hash_stream(archive.extractfile(item), deadline, maximum=item.size)
                            need(size == item.size, 'short regular member')
                            record = {'kind': 'file', 'bytes': size, 'sha256': value, 'executable': item.mode & 0o111}
                        elif item.isdir():
                            record = {'kind': 'directory'}
                        elif item.issym():
                            need('\x00' not in item.linkname and len(item.linkname) <= 16384, 'link target bound')
                            record = {'kind': 'symlink', 'target': item.linkname}
                        elif item.islnk():
                            record = {'kind': 'hardlink', 'target': relative(item.linkname)}
                        else:
                            raise ValueError('unexpected base special file')
                        entries.append((name, record))
                while plain.read(8 << 20):
                    pass
            # Gzip normally consumes the complete source. Hash any trailing
            # bytes too; the independently pinned compressed digest decides.
            while compressed.read(8 << 20):
                pass
            need(compressed.hash.hexdigest() == pin and compressed.count == layer['size']
                and 'sha256:' + plain.hash.hexdigest() == diff_id
                and stamp(before) == stamp(os.fstat(fd)) == stamp(blob.lstat()), 'layer/diff-id identity changed')
        apply_layer(tree, entries, whiteouts)
        record = {'index': index, 'compressed_sha256': pin, 'diff_id': diff_id,
            'compressed_bytes': compressed.count, 'uncompressed_bytes': plain.count,
            'entries': len(entries), 'whiteouts': len(whiteouts), 'result_members': len(tree)}
        records.append(record)
        progress(record)
    return {'image_manifest_sha256': manifest_pin, 'image_config_sha256': config_pin,
        'layers': records, 'archive_members': count, 'archive_payload_bytes': payload,
        'content_inventory': tree, 'content_inventory_sha256': sha(canonical(tree))}


def current_tree(root, deadline):
    """Walk held directory descriptors; never follow a symlink while hashing."""
    result, stamps = {}, {}
    root = Path(root)
    need(root.is_absolute() and root.resolve() == root, 'canonical tree root')
    root_fd = os.open(root, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW)
    def walk(directory, prefix):
        before_dir = os.fstat(directory)
        with os.scandir(directory) as iterator:
            names = sorted(entry.name for entry in iterator)
        for name in names:
            deadline_check(deadline)
            need(len(result) < MAX_MEMBERS, 'expanded member bound')
            key = relative(prefix + name)
            before = os.stat(name, dir_fd=directory, follow_symlinks=False)
            if stat.S_ISLNK(before.st_mode):
                record = {'kind': 'symlink', 'target': os.readlink(name, dir_fd=directory)}
            elif stat.S_ISDIR(before.st_mode):
                record = {'kind': 'directory'}
                child = os.open(name, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW, dir_fd=directory)
                try:
                    need(stamp(before) == stamp(os.fstat(child)), 'directory substituted')
                    walk(child, key + '/')
                finally:
                    os.close(child)
            elif stat.S_ISREG(before.st_mode):
                child = os.open(name, os.O_RDONLY | os.O_NOFOLLOW, dir_fd=directory)
                with os.fdopen(child, 'rb') as stream:
                    need(stamp(before) == stamp(os.fstat(child)), 'file substituted')
                    value, size = hash_stream(stream, deadline)
                    need(size == before.st_size and stamp(before) == stamp(os.fstat(child)), 'file changed during hash')
                record = {'kind': 'file', 'bytes': size, 'sha256': value, 'executable': before.st_mode & 0o111}
            else:
                raise ValueError('unexpected expanded special file')
            need(stamp(before) == stamp(os.stat(name, dir_fd=directory, follow_symlinks=False)), 'tree entry changed')
            result[key], stamps[key] = record, stamp(before)
        need(stamp(before_dir) == stamp(os.fstat(directory)), 'directory changed during walk')
    try:
        walk(root_fd, '')
    finally:
        os.close(root_fd)
    return result, stamps


def compare(expected, actual):
    missing, extra = sorted(set(expected) - set(actual)), sorted(set(actual) - set(expected))
    changed = sorted(name for name in set(expected) & set(actual) if expected[name] != actual[name])
    return {'matches': not (missing or extra or changed), 'missing_count': len(missing),
        'extra_count': len(extra), 'changed_count': len(changed), 'missing': missing[:100], 'extra': extra[:100],
        'changed': [{'path': name, 'expected': expected[name], 'actual': actual[name]} for name in changed[:100]]}


def proof_record():
    """Frozen provenance, not a replacement for current file verification."""
    return {'schema': 1, 'scope': 'oci-derived-retained-runtime-v1',
        'image_manifest_sha256': MANIFEST_SHA256, 'image_config_sha256': CONFIG_SHA256,
        'content_inventory_sha256': BASE_CONTENT_SHA256, 'expanded_members': BASE_MEMBERS,
        'prefix_inventory_sha256': PREFIX_SHA256, 'prefix_members': PREFIX_MEMBERS,
        'derivation_report_sha256': DERIVATION_REPORT_SHA256,
        'derivation_source_sha256': DERIVATION_SOURCE_SHA256}


def verify_retained(retained_root, facts, deadline):
    """Authenticate current base and complete prefix before loading any runtime."""
    started = time.monotonic()
    root = Path(retained_root)
    actual, base_stamps = current_tree(root / 'base-rootfs', deadline)
    need(len(actual) == BASE_MEMBERS and sha(canonical(actual)) == BASE_CONTENT_SHA256,
        'expanded base-rootfs differs from authenticated OCI-derived content')
    base_seconds = time.monotonic() - started
    prefix, prefix_stamps = current_tree(root / 'build-output3/build-venv', deadline)
    expected = facts.get('installed_prefix_files')
    need(type(expected) is dict and len(expected) == PREFIX_MEMBERS and
        sha(canonical(expected).rstrip(b'\n')) == facts.get('installed_inventory_sha256') == PREFIX_SHA256,
        'authenticated complete installed-prefix inventory required')
    compact = {}
    for name, record in prefix.items():
        if record['kind'] == 'file':
            compact[name] = record['sha256']
        elif record['kind'] == 'symlink':
            compact[name] = {'symlink': record['target']}
    need(compact == expected, 'installed-prefix content or link inventory changed')
    stamps = {str(root / subdir / name): stamp for subdir, mapping in
        (('base-rootfs', base_stamps), ('build-output3/build-venv', prefix_stamps))
        for name, stamp in mapping.items()}
    return {'scope': 'current-retained-runtime-verified-v1', 'base_members': len(actual),
        'base_content_sha256': BASE_CONTENT_SHA256, 'prefix_members': len(compact),
        'prefix_inventory_sha256': PREFIX_SHA256, 'base_seconds': base_seconds,
        'elapsed_seconds': time.monotonic() - started, 'stamps': stamps}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--derive-and-check', action='store_true')
    parser.add_argument('--check-retained', action='store_true')
    parser.add_argument('--image-root')
    parser.add_argument('--base-root')
    parser.add_argument('--retained-root')
    parser.add_argument('--facts')
    parser.add_argument('--seconds', required=True, type=int)
    args = parser.parse_args(argv)
    need(args.derive_and_check != args.check_retained and 1 <= args.seconds <= 1800, 'bounded read-only verification required')
    retained = '/home/developer/kv4-serving-integration-20261014.M4uQHI-base2'
    need(os.getuid() == 1000 and Path('/proc/sys/kernel/hostname').read_text().strip() == 'threadripper-ai'
        and (args.derive_and_check and args.image_root == '/home/developer/rocm101-build.sdwQZaU0/image-assembly3'
            and args.base_root == retained + '/base-rootfs' or args.check_retained
            and args.retained_root == retained and args.facts == retained + '/serving-composition9/COMPOSITION-FACTS.json'),
        'exact approved read-only target paths required')
    started, deadline = time.monotonic(), time.monotonic() + args.seconds
    def interrupted(number, frame):
        raise TimeoutError('verification signal ' + str(number))
    for number in (signal.SIGTERM, signal.SIGINT, signal.SIGALRM):
        signal.signal(number, interrupted)
    signal.setitimer(signal.ITIMER_REAL, args.seconds)
    output = {'schema': 1, 'scope': 'read-only-runtime-integrity-v1', 'pid': os.getpid(),
        'uid': os.getuid(), 'model_started': False, 'gpu_work_performed': False, 'writes_to_runtime': False}
    process_stat = Path('/proc/self/stat').read_bytes()
    output['start_ticks'] = int(process_stat[process_stat.rfind(b')') + 2:].split()[19])
    output['boot_id'] = Path('/proc/sys/kernel/random/boot_id').read_text().strip()
    print(json.dumps({'remote_started': {key: output[key] for key in ('pid', 'start_ticks', 'uid', 'boot_id')}}),
        file=sys.stderr, flush=True)
    try:
        if args.derive_and_check:
            output['derivation'] = derive_image(args.image_root, deadline,
                progress=lambda record: print(json.dumps({'layer_verified': record}), file=sys.stderr, flush=True))
            derived = time.monotonic()
            actual, _ = current_tree(args.base_root, deadline)
            output.update(comparison=compare(output['derivation']['content_inventory'], actual),
                actual_inventory_sha256=sha(canonical(actual)), actual_members=len(actual),
                derivation_seconds=derived - started, expanded_check_seconds=time.monotonic() - derived)
            output['status'] = 'RUNTIME_CONTENT_MATCH' if output['comparison']['matches'] else 'RUNTIME_CONTENT_MISMATCH'
        else:
            facts = read_pinned_json(args.facts,
                '539441ba42ac450d7b03cc47dc5462beda174937931e2a0f2fd832ac102cc5a5', deadline, maximum=16 << 20)
            record = verify_retained(args.retained_root, facts, deadline)
            output['retained'] = {name: value for name, value in record.items() if name != 'stamps'}
            output['status'] = 'RETAINED_RUNTIME_MATCH'
    except BaseException as error:
        output.update(status='INCOMPLETE', failure=type(error).__name__ + ': ' + str(error)[:1000])
    finally:
        signal.setitimer(signal.ITIMER_REAL, 0)
    output['elapsed_seconds'] = time.monotonic() - started
    raw = canonical(output)
    need(len(raw) <= MAX_OUTPUT, 'bounded derivation output')
    sys.stdout.buffer.write(raw)
    sys.stdout.buffer.flush()
    return 0 if output['status'] in ('RUNTIME_CONTENT_MATCH', 'RETAINED_RUNTIME_MATCH') else 1


if __name__ == '__main__':
    raise SystemExit(main())
