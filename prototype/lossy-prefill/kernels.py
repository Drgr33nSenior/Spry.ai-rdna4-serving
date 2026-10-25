"""Block-256 codec kernels. Importing this module does not load a framework."""


def build(triton):
    """Compile lazily with the caller's authenticated Triton installation."""
    import importlib

    # JIT resolves the function's module globals; do not depend on closure lookup.
    global tl
    tl = importlib.import_module("triton.language")

    @triton.jit
    def pack(X, Codes, Scales, N: tl.constexpr, FLOOR: tl.constexpr):
        block = tl.program_id(0)
        lane = tl.arange(0, 256)
        index = block * 256 + lane
        x = tl.load(X + index, index < N, other=0).to(tl.float32)
        scale = tl.maximum(tl.div_rn(tl.max(tl.abs(x), 0), 127.0), FLOOR)
        scaled = tl.div_rn(x, scale)
        # Explicit round-half-away-from-zero, followed by signed INT8 clipping.
        rounded = tl.where(
            scaled >= 0, tl.floor(scaled + 0.5), -tl.floor(-scaled + 0.5)
        )
        codes = tl.minimum(tl.maximum(rounded, -127), 127).to(tl.int8)
        tl.store(Codes + block * 260 + lane, codes)
        tl.store(Scales + block * 65 + 64, scale)

    @triton.jit
    def unpack_sum(Codes, Scales, Y, N: tl.constexpr, BLOCKS: tl.constexpr):
        block = tl.program_id(0)
        lane = tl.arange(0, 256)
        index = block * 256 + lane
        q0 = tl.load(Codes + block * 260 + lane).to(tl.float32)
        q1 = tl.load(Codes + (BLOCKS + block) * 260 + lane).to(tl.float32)
        s0 = tl.load(Scales + block * 65 + 64)
        s1 = tl.load(Scales + (BLOCKS + block) * 65 + 64)
        # Launch disables FP fusion so the CPU reference has identical FP32 steps.
        value = q0 * s0 + q1 * s1
        tl.store(Y + index, value, index < N)

    return pack, unpack_sum
