"""Project one pinned attention source to a private host-phase diagnostic copy.

No build, load, GPU call, package, or canonical source update is performed.
The only source edits are an isolated helper, a host lambda wrapper, and two
host clock boundaries. recover_source verifies byte-exact reversibility.
"""
import argparse
import hashlib
import os
from pathlib import Path
import stat


PINNED_SOURCE_SHA256 = "2206e70d233cacd86bea286be36c04a42d504e5807fe05985b2b2fa3319db88d"
MARKER = b"UQ_CACHE_PHASE_PROJECTION_V1"
SIGNATURE = b'extern "C" int uq_bf16_attention_v2('
HELPER = b'''// BEGIN UQ_CACHE_PHASE_PROJECTION_V1
#include "cache_phase_profile.h"
#include <chrono>

namespace uq_cache_phase_detail {
using Clock = std::chrono::steady_clock;
static_assert(Clock::is_steady, "phase timing requires a steady clock");
static_assert(sizeof(uq_cache_phase_profile_v1) == 64, "phase ABI size");
thread_local uq_cache_phase_profile_v1 last{};
thread_local uint64_t sequence = 0;

class Scope {
    Clock::time_point start;
    uint64_t host_end = 0;
    uint64_t validation_end = 0;
    unsigned phase = 0;
    uint64_t elapsed() const {
        return static_cast<uint64_t>(std::chrono::duration_cast<std::chrono::nanoseconds>(
            Clock::now() - start).count());
    }
public:
    Scope() : start(Clock::now()) {
        const uint64_t next = sequence < UINT64_MAX ? ++sequence : 0;
        last = {1, next, 0, -1, 0, 0, 0, 0};
    }
    void host_done() {
        if (phase == 0) { host_end = elapsed(); phase = 1; }
    }
    void validation_done() {
        if (phase == 1) { validation_end = elapsed(); phase = 2; }
    }
    int finish(int result) {
        const uint64_t total = elapsed();
        last.api_result = result;
        last.host_checks_ns = phase == 0 ? total : host_end;
        last.validation_status_ns = phase == 0 ? 0 :
            (phase == 1 ? total : validation_end) - host_end;
        last.reader_sync_ns = phase == 2 ? total - validation_end : 0;
        last.total_ns = total;
        last.complete = result == 0 && phase == 2 && last.sequence != 0;
        return result;
    }
};
}

extern "C" int uq_cache_phase_profile_last_v1(uq_cache_phase_profile_v1* out, uint64_t bytes) {
    if (!out || bytes != sizeof(*out) ||
        reinterpret_cast<uintptr_t>(out) % alignof(uq_cache_phase_profile_v1) != 0 ||
        uq_cache_phase_detail::last.sequence == 0) return 1;
    *out = uq_cache_phase_detail::last;
    return 0;
}
// END UQ_CACHE_PHASE_PROJECTION_V1
'''
WRAPPER_OPEN = b"\n    uq_cache_phase_detail::Scope phase_profile;\n    const int phase_result = [&]() -> int {"
WRAPPER_CLOSE = b"    }();\n    return phase_profile.finish(phase_result);\n"
HOST_BOUNDARY = b"    phase_profile.host_done();\n"
VALIDATION_BOUNDARY = b"    phase_profile.validation_done();\n"
HOST_ANCHOR = b"    auto stream=static_cast<hipStream_t>(opaque);\n"
VALIDATION_ANCHOR = b"    if(c->protected_layer) {\n"


def _one(source, old, new):
    if source.count(old) != 1:
        raise ValueError("missing or ambiguous pinned projection boundary")
    return source.replace(old, new, 1)


def project_source(source):
    """Return deterministic bytes; refuse all input except the pinned source."""
    if not isinstance(source, bytes):
        raise TypeError("source must be bytes")
    if MARKER in source:
        raise ValueError("source is already instrumented")
    if hashlib.sha256(source).hexdigest() != PINNED_SOURCE_SHA256:
        raise ValueError("attention source pin mismatch")
    if source.count(SIGNATURE) != 1 or not source.endswith(b"}\n"):
        raise ValueError("pinned attention function shape mismatch")
    start = source.index(SIGNATURE)
    opening = source.index(b"{", start)
    body = source[opening + 1:-2]
    body = _one(body, HOST_ANCHOR, HOST_BOUNDARY + HOST_ANCHOR)
    body = _one(body, VALIDATION_ANCHOR, VALIDATION_BOUNDARY + VALIDATION_ANCHOR)
    return HELPER + source[:opening + 1] + WRAPPER_OPEN + body + WRAPPER_CLOSE + b"}\n"


def recover_source(candidate):
    """Recover exact canonical bytes only from a verified projection."""
    if not isinstance(candidate, bytes) or not candidate.startswith(HELPER):
        raise ValueError("candidate helper mismatch")
    recovered = candidate[len(HELPER):]
    for old in (WRAPPER_OPEN, WRAPPER_CLOSE, HOST_BOUNDARY, VALIDATION_BOUNDARY):
        recovered = _one(recovered, old, b"")
    if hashlib.sha256(recovered).hexdigest() != PINNED_SOURCE_SHA256:
        raise ValueError("recovered attention source pin mismatch")
    if project_source(recovered) != candidate:
        raise ValueError("candidate differs from deterministic projection")
    return recovered


def project_file(source, output):
    """Create one new mode-0600 copy; never replace a source or existing file."""
    source, output = Path(source), Path(output)
    if source.is_symlink() or not stat.S_ISREG(source.stat().st_mode):
        raise ValueError("source must be a regular non-symlink file")
    if source.resolve() == output.resolve():
        raise ValueError("projection must not overwrite source")
    candidate = project_source(source.read_bytes())
    recover_source(candidate)
    descriptor = os.open(output, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    with os.fdopen(descriptor, "wb") as stream:
        stream.write(candidate)
    return hashlib.sha256(candidate).hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source", type=Path)
    parser.add_argument("output", type=Path)
    arguments = parser.parse_args()
    try:
        digest = project_file(arguments.source, arguments.output)
    except (OSError, ValueError) as error:
        parser.exit(1, f"refused: {error}\n")
    print(digest)


if __name__ == "__main__":
    main()
