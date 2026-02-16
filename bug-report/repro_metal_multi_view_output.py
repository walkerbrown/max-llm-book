"""Reproduction: Metal GPU returns zeros for 3+ slice view outputs from same buffer.

When a compiled graph on Metal GPU returns 3 or more output tensors that are all
views (slices) of the same source buffer, every output contains zeros. The output
buffers are allocated but never populated. This is independent from the
slice-offset-in-computation bug — that bug reads real data from offset 0, while
this bug produces zeros even when the entire buffer is non-zero.

Environment:
    MAX 26.2.0.dev2026021505, macOS, Apple M2 Max, Metal GPU

Usage:
    pixi run python repro_metal_multi_view_output.py
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


class OneSliceOutput(Module):
    """Single slice output (baseline)."""

    def forward(self, x):
        return x[:, :, 4:8]


class TwoSliceOutputs(Module):
    """Two slice outputs from same buffer."""

    def forward(self, x):
        return x[:, :, :4], x[:, :, 4:8]


class ThreeSliceOutputs(Module):
    """Three slice outputs from same buffer."""

    def forward(self, x):
        return x[:, :, :4], x[:, :, 4:8], x[:, :, 8:]


class FourSliceOutputs(Module):
    """Four slice outputs from same buffer."""

    def forward(self, x):
        return x[:, :, :4], x[:, :, 4:8], x[:, :, 8:12], x[:, :, 12:]


class ThreeComputedOutputs(Module):
    """Three computed (non-view) outputs from same input — control test."""

    def forward(self, x):
        return x + 1.0, x + 2.0, x + 3.0


class FiveComputedOutputs(Module):
    """Five computed outputs — control test for output count."""

    def forward(self, x):
        return x + 1.0, x + 2.0, x + 3.0, x + 4.0, x + 5.0


class ThreeSlicesConsumedInternally(Module):
    """Three slices used internally, single output — tests interaction."""

    def forward(self, x):
        a = x[:, :, :4]
        b = x[:, :, 4:8]
        c = x[:, :, 8:]
        return a + b + c


# ---------------------------------------------------------------------------
# Runner
# ---------------------------------------------------------------------------


def run_tests():
    require_metal_gpu()

    gpu = Accelerator()
    cpu = CPU()

    # Use non-zero fill so we can distinguish "offset 0 data" from "uninitialized"
    data_12 = np.zeros((1, 2, 12), dtype=np.float32)
    data_12[:, :, :4] = 1.0
    data_12[:, :, 4:8] = 2.0
    data_12[:, :, 8:] = 3.0
    t_gpu_12 = Tensor.constant(data_12.tolist(), dtype=DType.float32, device=gpu)
    t_cpu_12 = Tensor.constant(data_12.tolist(), dtype=DType.float32, device=cpu)

    data_16 = np.zeros((1, 2, 16), dtype=np.float32)
    data_16[:, :, :4] = 1.0
    data_16[:, :, 4:8] = 2.0
    data_16[:, :, 8:12] = 3.0
    data_16[:, :, 12:] = 4.0
    t_gpu_16 = Tensor.constant(data_16.tolist(), dtype=DType.float32, device=gpu)

    # All-42 buffer to prove zeros aren't from buffer contents
    data_42 = np.full((1, 2, 12), 42.0, dtype=np.float32)
    t_gpu_42 = Tensor.constant(data_42.tolist(), dtype=DType.float32, device=gpu)

    small = Tensor.constant([[[10.0, 10.0, 10.0, 10.0]]], dtype=DType.float32, device=gpu)

    passed = 0
    failed = 0
    results = []

    def check(name, actual_vals, expected_vals, tolerance=1e-5):
        nonlocal passed, failed
        ok = all(
            abs(a - e) < tolerance for a, e in zip(actual_vals, expected_vals)
        )
        status = "PASS" if ok else "FAIL"
        if ok:
            passed += 1
        else:
            failed += 1
        results.append((status, name, actual_vals, expected_vals))

    # -- Test 1: 1 slice output (baseline) --
    c = compile_module(OneSliceOutput, (1, 2, 12), gpu)
    out = readback(c(t_gpu_12))
    check("1 slice output", [out.flat[0]], [2.0])

    # -- Test 2: 2 slice outputs --
    c = compile_module(TwoSliceOutputs, (1, 2, 12), gpu)
    a, b = c(t_gpu_12)
    check(
        "2 slice outputs",
        [readback(a).flat[0], readback(b).flat[0]],
        [1.0, 2.0],
    )

    # -- Test 3: 3 slice outputs (BUG) --
    c = compile_module(ThreeSliceOutputs, (1, 2, 12), gpu)
    a, b, cc = c(t_gpu_12)
    check(
        "3 slice outputs",
        [readback(a).flat[0], readback(b).flat[0], readback(cc).flat[0]],
        [1.0, 2.0, 3.0],
    )

    # -- Test 4: 4 slice outputs (BUG) --
    c = compile_module(FourSliceOutputs, (1, 2, 16), gpu)
    a, b, cc, d = c(t_gpu_16)
    check(
        "4 slice outputs",
        [readback(r).flat[0] for r in (a, b, cc, d)],
        [1.0, 2.0, 3.0, 4.0],
    )

    # -- Test 5: 3 slice outputs from all-42 buffer (proves zeros, not offset-0) --
    c = compile_module(ThreeSliceOutputs, (1, 2, 12), gpu)
    a, b, cc = c(t_gpu_42)
    vals = [readback(r).flat[0] for r in (a, b, cc)]
    check("3 slice outputs (all-42 buffer)", vals, [42.0, 42.0, 42.0])

    # -- Test 6: 3 computed outputs, NOT slices (control) --
    c = compile_module(ThreeComputedOutputs, (1, 1, 4), gpu)
    a, b, cc = c(small)
    check(
        "3 computed outputs (not slices)",
        [readback(r).flat[0] for r in (a, b, cc)],
        [11.0, 12.0, 13.0],
    )

    # -- Test 7: 5 computed outputs (control for output count) --
    c = compile_module(FiveComputedOutputs, (1, 1, 4), gpu)
    outs = c(small)
    check(
        "5 computed outputs (not slices)",
        [readback(r).flat[0] for r in outs],
        [11.0, 12.0, 13.0, 14.0, 15.0],
    )

    # -- Test 8: 3 slices consumed internally, single output --
    c_gpu = compile_module(ThreeSlicesConsumedInternally, (1, 2, 12), gpu)
    c_cpu = compile_module(ThreeSlicesConsumedInternally, (1, 2, 12), cpu)
    out_gpu = readback(c_gpu(t_gpu_12)).flat[0]
    out_cpu = readback(c_cpu(t_cpu_12)).flat[0]
    check(
        "3 slices consumed internally (expect 6.0=1+2+3)",
        [out_gpu],
        [out_cpu],
    )

    # -- Print results --
    print()
    print("=" * 70)
    print("Metal GPU Multi-View Output Bug — Reproduction Results")
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
        print("BUG CONFIRMED: 3+ slice view outputs from the same buffer")
        print("return zeros on Metal GPU. Output buffers are allocated but")
        print("never populated. Non-slice outputs and <=2 slice outputs work.")
        # Check if test 8 also failed to note interaction with slice offset bug
        t8 = next(
            (r for r in results if "consumed internally" in r[1]), None
        )
        if t8 and t8[0] == "FAIL":
            print()
            print(
                "NOTE: Test 8 failure (internal consumption) is caused by the"
            )
            print("separate slice-offset bug, not this output-materialization bug.")
    else:
        print("All tests passed — bug may be fixed in this version.")

    print()
    return failed == 0


if __name__ == "__main__":
    success = run_tests()
    sys.exit(0 if success else 1)
