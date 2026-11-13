#!/usr/bin/env python3
"""Run one SGLang unittest module on a machine without a GPU.

Upstream skips the unified radix cache test classes unless CUDA is available.
This runner clears that class-level skip, points the test kits' get_device()
at the CPU and turns cudaHostRegister into a no-op, so the device-tier cases
run against CPU tensors. Cases that move KV between device and host (HiCache)
still need CUDA and report errors here; that is expected.

Run it from the test file's directory with the patched SGLang tree on
PYTHONPATH, inside the serving image's Python environment:

    PYTHONPATH=<sglang>/python:. SGLANG_UNIFIED_RADIX_TREE_CORE_BACKEND=python \
        python run_sglang_unit_on_cpu.py <module> [unittest arguments]
"""

import importlib
import sys
import unittest


def main() -> None:
    if len(sys.argv) < 2:
        raise SystemExit(__doc__)
    name = sys.argv[1]
    module = importlib.import_module(name)
    for loaded_name, loaded in list(sys.modules.items()):
        if loaded_name.startswith(("sglang.test", "test_")) and hasattr(
            loaded, "get_device"
        ):
            loaded.get_device = lambda *args, **kwargs: "cpu"
    for value in vars(module).values():
        if isinstance(value, type) and issubclass(value, unittest.TestCase):
            if getattr(value, "__unittest_skip__", False):
                value.__unittest_skip__ = False

    from sglang.srt.mem_cache.pool_host import common as host_common

    host_common._cuda_host_register = lambda *args, **kwargs: None
    unittest.main(module=module, argv=[name, *sys.argv[2:]])


if __name__ == "__main__":
    main()
