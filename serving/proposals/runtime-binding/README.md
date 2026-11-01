# MXFP4 runtime binding at startup

This source proposal resolves the MXFP4 operator modules and checks the HIP
architecture when the native loader constructs a private quantization scheme.
Quantizer and GEMM calls then use that binding. They still check the current
device, tensor metadata and distinct storage before each kernel launch.

The binding retains only module references for the two admitted device indices.
It stores no tensors, buffers or stream handles. The kernels continue to use the
current stream. The quantizer and GEMM frontend bodies, launch arguments, tile
policy and kernel sources are unchanged.

The new operator source requires `bind_runtime(card)` before use. A cold call
refuses without loading a runtime or launching a kernel. Invalid device or
architecture checks cannot publish a partial binding. The worker authenticates
the new binding functions along with the existing operator callables before
creating the scheme class. Scheme instances bind after native ModelRunner has
selected the rank's device. Creating the class alone must not bind the parent
process's default device. The retained control source keeps its previous behavior.

## Source and assembly

`spry_serving/ops.py`, `scheme.py` and `worker_entry.py` hold the candidate source.
`runtime_contract.py` binds the new operator and scheme hashes. The pure prefill
helper is also canonical; its bytes are unchanged. The original operator,
scheme and prefill helper are retained under `legacy/review` and authenticated
against the sealed donor.
These additions replace existing package roles and add no archive members.

`RUNTIME.patch` records the operator, worker and contract changes. `SOURCE.patch`
updates the proposer so it loads the contract from its declared, pinned source
path. This preserves the previous configuration proposal and permits the new
contract variant without rewriting it.

The protected `D2.patch` changes only the operator and scheme hashes in the host
transport's existing source inventory. Resource ceilings, archive membership and release
predicates are unchanged. Exact owner review is still required before native
adoption. Publication of this proposal grants no execution authority.

## Verification and remaining gates

CPU tests exercise the actual worker to the scheme-factory boundary with inert
external modules. They check rank-one construction after device selection,
bad-architecture refusal, one resolution per device, per-call
device/layout/storage refusal, and unchanged
kernel frontend bodies. These descriptor tests perform no native arithmetic.

The separate packet checks authenticate the original donor, call the real
archive inspector, reject operator rollback even with self-consistent package
hashes, and compare every package member with the configuration control. Model,
stack, requests, settings and all active overlays must remain byte identical.
The configured prelaunch and producer/reader checks also remain required.

Native token equality and quality have not been established for this proposal.
It changes runtime setup timing and needs a fresh one-use serving qualification.
No inference speedup is claimed. Repeated startup source hashing, the remaining
module closure, the runner hook and replay metadata work are still pending.
Memory recovery follows those refactors.
