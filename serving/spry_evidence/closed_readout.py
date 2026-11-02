"""Source-bound HTTP owner readout; no native imports or standalone launch."""
import hashlib
import importlib.util
import json
from pathlib import Path

_ENTRY = Path(__file__).resolve().with_name('model_benchmark.py')
_MODULE_SPEC = importlib.util.spec_from_file_location('spry_benchmark_contract', _ENTRY)
bench = importlib.util.module_from_spec(_MODULE_SPEC)
_MODULE_SPEC.loader.exec_module(bench)
MAXIMUM_BYTES = 8 << 20


def _object(pairs):
    value = {}
    for key, item in pairs:
        if key in value:
            raise ValueError("duplicate JSON member")
        value[key] = item
    return value


def _constant(value):
    raise ValueError("nonfinite JSON constant")


def read_json(path, expected_sha256=None):
    raw = bench.read_bounded_file(Path(path), MAXIMUM_BYTES)
    digest = hashlib.sha256(raw).hexdigest()
    if expected_sha256 is not None and digest != expected_sha256:
        raise ValueError("spec digest mismatch")
    value = json.loads(raw, object_pairs_hook=_object, parse_constant=_constant)
    if not isinstance(value, dict):
        raise ValueError("JSON object required")
    return value, digest


def checked_http_arm(result, arm, spec_sha256):
    """Check the narrow owner-facing result from the raw HTTP evidence reader.

    Full wire/gate/source validation happens in ``serving_integration``.  This
    function deliberately does not reinterpret HTTP latency as model quality.
    """
    contract_spec = importlib.util.spec_from_file_location('final_readout_contract', Path(__file__).with_name('mtp_contract.py'))
    final_contract = importlib.util.module_from_spec(contract_spec)
    contract_spec.loader.exec_module(final_contract)
    logical = final_contract.owner_arm_map()
    if (not isinstance(result, dict) or result.get("schema") != 1 or
            result.get("mode") != "native-http-fp8-sprint01-v1" or
            result.get("status") != "HTTP_ARM_COMPLETE" or result.get("owner_arm") != arm or
            result.get("logical_arm") != logical[arm] or result.get("qualification") is not False or
            result.get("binding", {}).get("spec_sha256") != spec_sha256 or
            result.get("wire", {}).get("status") != "HTTP_WIRE_COMPLETE" or
            result.get("wire", {}).get("wire_complete") is not True):
        raise ValueError("incomplete or unbound HTTP arm")
    if final_contract.contract_for_owner_arm(arm)['enabled']:
        telemetry = result["wire"].get("speculation")
        if not (isinstance(telemetry, dict) and type(telemetry.get("spec_verify_ct")) is int and
                telemetry["spec_verify_ct"] > 0):
            raise ValueError("MTP arm lacks verified speculation telemetry")
    return result
