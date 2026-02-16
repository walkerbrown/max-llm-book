"""Reproduction: Metal GPU slice offset ignored in kernel computations.

When a tensor slice with non-zero offset is consumed by any computation on Metal
GPU, the kernel reads from buffer offset 0 instead of the correct offset. The
slice metadata is tracked through MLIR but not applied in the generated Metal
shader.

Environment:
    MAX 26.2.0.dev2026021505, macOS, Apple M2 Max, Metal GPU

Usage:
    pixi run python repro_metal_slice_offset.py
"""

import sys

import numpy as np
from max.driver import Accelerator, CPU, accelerator_count
from max.dtype import DType
from max.graph import DeviceRef
from max.nn import Module
from max.tensor import Tensor, TensorType, default_device, default_dtype


def require_metal_gpu():
    """Skip gracefully if no Metal GPU is available."""
    if accelerator_count() == 0:
        print("SKIP: No GPU available")
        sys.exit(0)
    from max.driver import accelerator_api

    if accelerator_api() != "metal":
        print(f"SKIP: GPU is {accelerator_api()!r}, not metal")
        sys.exit(0)


def make_test_input(gpu):
    """Create a (1, 2, 12) tensor with distinct values per 4-element region.

    Region [:4] = 1.0, [4:8] = 2.0, [8:12] = 3.0
    """
    data = np.zeros((1, 2, 12), dtype=np.float32)
    data[:, :, :4] = 1.0
    data[:, :, 4:8] = 2.0
    data[:, :, 8:] = 3.0
    return data, Tensor.constant(data.tolist(), dtype=DType.float32, device=gpu)


def compile_module(cls, input_shape, device):
    with default_device(CPU()), default_dtype(DType.float32):
        m = cls()
    m.to(device)
    tt = TensorType(DType.float32, input_shape, device=DeviceRef.from_device(device))
    return m.compile(tt)


def readback(tensor):
    return np.from_dlpack(tensor.to(CPU()))


# ---------------------------------------------------------------------------
# Test modules
# ---------------------------------------------------------------------------


class SliceOffset0(Module):
    """Slice at offset 0 consumed by multiply."""

    def forward(self, x):
        return x[:, :, :4] * 10.0


class SliceOffset4(Module):
    """Slice at offset 4 consumed by multiply."""

    def forward(self, x):
        return x[:, :, 4:8] * 10.0


class SliceOffset8(Module):
    """Slice at offset 8 consumed by multiply."""

    def forward(self, x):
        return x[:, :, 8:] * 10.0


class SliceBatchAxis(Module):
    """Slice on batch axis (axis 0) at non-zero offset."""

    def forward(self, x):
        return x[1:2, :, :] * 10.0


class SliceSeqAxis(Module):
    """Slice on sequence axis (axis 1) at non-zero offset."""

    def forward(self, x):
        return x[:, 2:3, :] * 10.0


class TwoSlicesConsumed(Module):
    """Two slices of same buffer consumed in addition."""

    def forward(self, x):
        a = x[:, :, :4]
        b = x[:, :, 4:8]
        return a + b


class SliceReturnOnly(Module):
    """Return a non-zero-offset slice WITHOUT computation (output path)."""

    def forward(self, x):
        return x[:, :, 4:8]


# ---------------------------------------------------------------------------
# Runner
# ---------------------------------------------------------------------------


def run_tests():
    require_metal_gpu()

    gpu = Accelerator()
    cpu = CPU()
    data, t_gpu = make_test_input(gpu)
    _, t_cpu = make_test_input(cpu)

    passed = 0
    failed = 0
    results = []

    def check(name, actual, expected, tolerance=1e-5):
        nonlocal passed, failed
        ok = np.allclose(actual, expected, atol=tolerance)
        status = "PASS" if ok else "FAIL"
        if not ok:
            failed += 1
        else:
            passed += 1
        results.append((status, name, actual, expected))

    # -- Test 1: Offset 0 (baseline, should work) --
    c = compile_module(SliceOffset0, (1, 2, 12), gpu)
    out = readback(c(t_gpu))
    check(
        "slice[:4]*10 (offset=0)",
        out.flat[0],
        10.0,
    )

    # -- Test 2: Offset 4 --
    c = compile_module(SliceOffset4, (1, 2, 12), gpu)
    out = readback(c(t_gpu))
    check(
        "slice[4:8]*10 (offset=4, expect 20)",
        out.flat[0],
        20.0,
    )

    # -- Test 3: Offset 8 --
    c = compile_module(SliceOffset8, (1, 2, 12), gpu)
    out = readback(c(t_gpu))
    check(
        "slice[8:]*10 (offset=8, expect 30)",
        out.flat[0],
        30.0,
    )

    # -- Test 4: Slice on batch axis --
    batch_data = np.zeros((4, 2, 4), dtype=np.float32)
    batch_data[0] = 1.0
    batch_data[1] = 2.0
    batch_data[2] = 3.0
    batch_data[3] = 4.0
    t_batch = Tensor.constant(batch_data.tolist(), dtype=DType.float32, device=gpu)
    c = compile_module(SliceBatchAxis, (4, 2, 4), gpu)
    out = readback(c(t_batch))
    check(
        "slice batch[1:2]*10 (axis=0, expect 20)",
        out.flat[0],
        20.0,
    )

    # -- Test 5: Slice on sequence axis --
    seq_data = np.zeros((1, 4, 4), dtype=np.float32)
    seq_data[:, 0, :] = 1.0
    seq_data[:, 1, :] = 2.0
    seq_data[:, 2, :] = 3.0
    seq_data[:, 3, :] = 4.0
    t_seq = Tensor.constant(seq_data.tolist(), dtype=DType.float32, device=gpu)
    c = compile_module(SliceSeqAxis, (1, 4, 4), gpu)
    out = readback(c(t_seq))
    check(
        "slice seq[2:3]*10 (axis=1, expect 30)",
        out.flat[0],
        30.0,
    )

    # -- Test 6: Two slices consumed in one computation --
    c = compile_module(TwoSlicesConsumed, (1, 2, 12), gpu)
    out_gpu = readback(c(t_gpu))
    c_cpu = compile_module(TwoSlicesConsumed, (1, 2, 12), cpu)
    out_cpu = readback(c_cpu(t_cpu))
    check(
        "a[:4]+b[4:8] (expect 3.0=1+2)",
        out_gpu.flat[0],
        out_cpu.flat[0],
    )

    # -- Test 7: Return slice without computation (should work) --
    c = compile_module(SliceReturnOnly, (1, 2, 12), gpu)
    out = readback(c(t_gpu))
    check(
        "return slice[4:8] as output (no computation, expect 2)",
        out.flat[0],
        2.0,
    )

    # -- Print results --
    print()
    print("=" * 70)
    print("Metal GPU Slice Offset Bug — Reproduction Results")
    print("=" * 70)
    for status, name, actual, expected in results:
        marker = "  " if status == "PASS" else ">>"
        print(f"  {marker} [{status}] {name}")
        if status == "FAIL":
            print(f"           actual={actual}, expected={expected}")
    print()
    print(f"  {passed} passed, {failed} failed")
    print()

    if failed > 0:
        print("BUG CONFIRMED: Slice offsets are ignored in Metal GPU kernels.")
        print("Kernels read from buffer base address (offset 0) regardless of")
        print("the slice start position. Only offset=0 slices and the output")
        print("path (return without computation) are unaffected.")
    else:
        print("All tests passed — bug may be fixed in this version.")

    print()
    return failed == 0


if __name__ == "__main__":
    success = run_tests()
    sys.exit(0 if success else 1)
