# Startup configuration checks in both modes

Production previously skipped `_mtp_preflight`, so device identity could pass
without checking the effective speculative width, overlap policy or selected
runner. This source change adds `_validate_serving_startup` to the shared
pre-weight path before device queries and observer installation.

The check uses the pinned native `resolving_view`, rather than the raw parser
namespace. It requires the declared DFlash algorithm and width, tensor and
pipeline ranks, disabled overlap, graph backends and exact graph batch values.
Boolean aliases cannot stand in for integers. It joins every selected runtime
overlay to its module, target, proposed hash and original hash, then verifies the
actual overlay file. A selected model-runner overlay is mandatory.

The original deadline remains in force after these metadata checks. If that
deadline expires, the worker cannot begin device identity queries. Instrumented
preflight and its existing receipts remain unchanged. Production adds no observer
import, tensor operation, model forward or configuration receipt.

[SOURCE.patch](SOURCE.patch) changes only the canonical worker source. Its
baseline matches the exact worker pin from the earlier host-binding proposal.
The protected helper bytes and their hash set remain identical to the boot-
approval stage. The retained `D2.patch` is that prior exact protected proposal;
this increment changes no protected helper.

Nine focused startup tests and all eight host-binding tests pass. The existing
host-binding fixtures mock the new configuration boundary while checking device
identity; they now require configuration to precede identity and delegation.
The dedicated tests exercise the actual configuration check. Private composition
checks additionally use the actual W5 contracts, both arm templates and all nine
overlay files. Native resolution, device and sysfs observations remain explicit
CPU substitutes. Both arms and ranks pass; eight bad resolved configurations
refuse before identity work.

The full composed suite passes 33 checks. The repository suite passes 334 cleanup
tests and 771 Python unit tests. These CPU results do not establish native startup
or complete Phase 1. Production evidence mounts and unconditional parity imports
remain separate cleanup work. No inference performance gain has been measured.
