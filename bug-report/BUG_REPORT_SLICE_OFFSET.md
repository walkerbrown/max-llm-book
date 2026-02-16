# Metal GPU: Slice offset ignored when consumed by computation kernel

## Environment

| | |
|---|---|
| **MAX version** | 26.2.0.dev2026021505 |
| **OS** | macOS 26.2 (25C56) |
| **Hardware** | Apple M2 Max, 38-core GPU |
| **Metal** | Metal 4 |
| **Python** | 3.14.3 |

## Summary

When a tensor slice with a non-zero byte offset is consumed by a downstream computation kernel on Metal GPU, the kernel reads from buffer offset 0 instead of the correct offset. This affects all axes and all downstream operations. The bug is **not** triggered when a slice is returned directly as an output (the `execute()` fallback path materializes correctly). CPU execution is unaffected.

## Minimal reproduction

```python
from max.driver import Accelerator, CPU
from max.dtype import DType
from max.graph import DeviceRef
from max.nn import Module
from max.tensor import Tensor, TensorType, default_device, default_dtype
import numpy as np

class SliceOffset4(Module):
    def forward(self, x):
        return x[:, :, 4:8] * 10.0  # Should select region filled with 2.0

gpu = Accelerator()
with default_device(CPU()), default_dtype(DType.float32):
    m = SliceOffset4()
m.to(gpu)
tt = TensorType(DType.float32, (1, 2, 12), device=DeviceRef.from_device(gpu))
compiled = m.compile(tt)

data = np.zeros((1, 2, 12), dtype=np.float32)
data[:, :, :4] = 1.0
data[:, :, 4:8] = 2.0
data[:, :, 8:] = 3.0
t = Tensor.constant(data.tolist(), dtype=DType.float32, device=gpu)

result = np.from_dlpack(compiled(t).to(CPU()))
print(result.flat[0])  # Expected: 20.0 (2.0 * 10), Actual: 10.0 (1.0 * 10)
```

Full reproduction script: `repro_metal_slice_offset.py` (run via `pixi run slice-offset`)

## Observed output

```
======================================================================
Metal GPU Slice Offset Bug — Reproduction Results
======================================================================
     [PASS] slice[:4]*10 (offset=0)
  >> [FAIL] slice[4:8]*10 (offset=4, expect 20)
           actual=10.0, expected=20.0
  >> [FAIL] slice[8:]*10 (offset=8, expect 30)
           actual=10.0, expected=30.0
  >> [FAIL] slice batch[1:2]*10 (axis=0, expect 20)
           actual=10.0, expected=20.0
  >> [FAIL] slice seq[2:3]*10 (axis=1, expect 30)
           actual=10.0, expected=30.0
  >> [FAIL] a[:4]+b[4:8] (expect 3.0=1+2)
           actual=2.0, expected=3.0
     [PASS] return slice[4:8] as output (no computation, expect 2)

  2 passed, 5 failed

BUG CONFIRMED: Slice offsets are ignored in Metal GPU kernels.
```

### Key observations

- **Offset 0 always works** — `x[:, :, :4] * 10` correctly returns 10.0
- **Non-zero offsets read from offset 0** — `x[:, :, 4:8] * 10` returns 10.0 (should be 20.0), `x[:, :, 8:] * 10` returns 10.0 (should be 30.0)
- **Axis-independent** — happens on axis 0, 1, and 2
- **Operation-independent** — happens with multiply, add, and any downstream op
- **Output path works** — returning a slice directly (no computation) returns correct data
- **CPU unaffected** — all tests pass on CPU

## Root cause analysis

`Slice` and `SliceDim` in `MOGGKernelAPI.mojo` are decorated with `@compiler.view_kernel`:

```mojo
# MOGGKernelAPI.mojo ~line 2099
@compiler.view_kernel
fn Slice(...):
    ...
    result = slice_as_view(input, starts, sizes)  # adjusts base pointer

# MOGGKernelAPI.mojo ~line 2244
@compiler.view_kernel
fn SliceDim(...):
    ...
    result = slice_as_view(input, starts, sizes)  # adjusts base pointer
```

The `slice_as_view` helper (in `slice.mojo` ~line 74, ~line 143) computes `ptr + offset` to produce a new base pointer into the existing buffer. This is the **only** view kernel that adjusts the base pointer — other view kernels (`Broadcast`, `Reshape`, `Transpose`) keep `input.unsafe_ptr()` unchanged and only modify shape/strides metadata.

When the MOGG compiler fuses a view kernel with a downstream computation kernel, the fused kernel receives the offset pointer. On Metal GPU, the runtime (`AsyncRT_DeviceContext_enqueueFunctionDirect`) binds the underlying `MTLBuffer` at offset 0 instead of computing and applying the byte offset via Metal's `setBuffer:offset:atIndex:` API.

The `execute()` fallback path (used when a slice is a graph output) goes through `view_copy_impl` (`managed_tensor_slice.mojo` ~line 1756), which materializes the slice into a new buffer, bypassing the issue.

### Evidence

- Offset 0 works → the buffer binding itself is correct, only the offset is lost
- All non-zero offsets read offset-0 data → the kernel always reads from buffer base
- Output-only path works → `view_copy_impl` materializes correctly
- CPU works → the pointer arithmetic in `slice_as_view` is correct; it's the Metal buffer binding that drops the offset

## Impact

This breaks **any** compiled model that uses `F.split`, tensor slicing, or any operation that produces a non-zero-offset slice consumed by a downstream kernel on Metal GPU. This includes multi-head attention (which splits Q/K/V), mixture-of-experts routing, and many other common patterns.

## Relevant source files

- `MOGGKernelAPI.mojo` lines ~2099-2101 (`Slice` view kernel)
- `MOGGKernelAPI.mojo` lines ~2244-2246 (`SliceDim` view kernel)
- `slice.mojo` lines ~74, ~143 (`slice_as_view` pointer adjustment)
- `managed_tensor_slice.mojo` line ~1756 (`view_copy_impl` — the working fallback)

## Suggested fix

In `AsyncRT_DeviceContext_enqueueFunctionDirect` (or wherever Metal kernel arguments are bound), when a kernel argument is a pointer that is offset from a `MTLBuffer` base address:

1. Compute `byte_offset = arg_ptr - mtl_buffer.contents()`
2. Use `setBuffer:offset:atIndex:` with the computed byte offset instead of `setBuffer:offset:atIndex:` with offset 0

### Workaround

Removing the `@compiler.view_kernel` decorator from `Slice` and `SliceDim` in `MOGGKernelAPI.mojo` forces materialization via `view_copy_impl`, which produces correct results at the cost of an extra copy.
