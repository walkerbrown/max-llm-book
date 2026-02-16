# Metal GPU: 3+ slice view outputs from same buffer return zeros

## Environment

| | |
|---|---|
| **MAX version** | 26.2.0.dev2026021505 |
| **OS** | macOS 26.2 (25C56) |
| **Hardware** | Apple M2 Max, 38-core GPU |
| **Metal** | Metal 4 |
| **Python** | 3.14.3 |

## Summary

When a compiled graph on Metal GPU returns 3 or more output tensors that are all views (slices) of the same source buffer, every output contains zeros. The output buffers are allocated (zero-initialized) but never populated. 1-2 slice outputs work correctly. Non-slice outputs (e.g., `x+1, x+2, x+3`) work with any count. CPU execution is unaffected.

This is **independent** from the [slice offset bug](./BUG_REPORT_SLICE_OFFSET.md) — that bug reads real data from offset 0, while this bug produces zeros even when the entire source buffer is filled with non-zero values (e.g., 42.0).

## Minimal reproduction

```python
from max.driver import Accelerator, CPU
from max.dtype import DType
from max.graph import DeviceRef
from max.nn import Module
from max.tensor import Tensor, TensorType, default_device, default_dtype
import numpy as np

class ThreeSliceOutputs(Module):
    def forward(self, x):
        return x[:, :, :4], x[:, :, 4:8], x[:, :, 8:]

gpu = Accelerator()
with default_device(CPU()), default_dtype(DType.float32):
    m = ThreeSliceOutputs()
m.to(gpu)
tt = TensorType(DType.float32, (1, 2, 12), device=DeviceRef.from_device(gpu))
compiled = m.compile(tt)

data = np.full((1, 2, 12), 42.0, dtype=np.float32)
t = Tensor.constant(data.tolist(), dtype=DType.float32, device=gpu)

a, b, c = compiled(t)
print([np.from_dlpack(r.to(CPU())).flat[0] for r in (a, b, c)])
# Expected: [42.0, 42.0, 42.0]
# Actual:   [0.0, 0.0, 0.0]
```

Full reproduction script: `repro_metal_multi_view_output.py` (run via `pixi run multi-view`)

## Observed output

```
======================================================================
Metal GPU Multi-View Output Bug — Reproduction Results
======================================================================
     [PASS] 1 slice output
     [PASS] 2 slice outputs
  >> [FAIL] 3 slice outputs
           actual=[np.float32(0.0), np.float32(0.0), np.float32(0.0)], expected=[1.0, 2.0, 3.0]
  >> [FAIL] 4 slice outputs
           actual=[np.float32(0.0), np.float32(0.0), np.float32(0.0), np.float32(0.0)], expected=[1.0, 2.0, 3.0, 4.0]
  >> [FAIL] 3 slice outputs (all-42 buffer)
           actual=[np.float32(0.0), np.float32(0.0), np.float32(0.0)], expected=[42.0, 42.0, 42.0]
     [PASS] 3 computed outputs (not slices)
     [PASS] 5 computed outputs (not slices)
  >> [FAIL] 3 slices consumed internally (expect 6.0=1+2+3)
           actual=[np.float32(3.0)], expected=[np.float32(6.0)]

  4 passed, 4 failed

BUG CONFIRMED: 3+ slice view outputs from the same buffer
return zeros on Metal GPU. Output buffers are allocated but
never populated. Non-slice outputs and <=2 slice outputs work.

NOTE: Test 8 failure (internal consumption) is caused by the
separate slice-offset bug, not this output-materialization bug.
```

### Key observations

- **1 slice output** — works correctly
- **2 slice outputs** — works correctly
- **3+ slice outputs** — all outputs are zeros (buffers allocated but never written)
- **All-42 buffer test** — proves zeros aren't from buffer contents; the entire source is 42.0 but outputs are 0.0
- **Computed (non-view) outputs** — 3 and 5 non-slice outputs all work correctly
- **Test 8 note** — the "3 slices consumed internally" test fails with value 3.0 instead of 6.0; this is caused by the separate [slice offset bug](./BUG_REPORT_SLICE_OFFSET.md), not this bug (the three slices all read from offset 0, getting 1.0+1.0+1.0=3.0)

## Analysis

The threshold is exactly 3 outputs. The output buffers are allocated (they contain zeros from initialization) but the copy from the source buffer never executes. This suggests:

1. **Not a Metal argument buffer limit** — 5 computed outputs work fine, so the issue isn't the number of output bindings
2. **View-specific** — only happens when outputs are slice views of the same buffer, not when outputs are computed tensors
3. **Possible scheduling/optimization bug** — the runtime may incorrectly optimize away the `view_copy_impl` calls when it detects that 3+ outputs alias the same source buffer, perhaps assuming the copies are redundant
4. **Possible Metal command encoder issue** — the copy commands for 3+ aliased outputs may be batched into a single command encoder that gets incorrectly elided or reordered

The fact that 2 outputs work but 3 don't points to a specific threshold in the output scheduling logic, possibly a heuristic that kicks in at 3+ aliased outputs.

## Impact

This breaks any compiled model returning 3 or more slices from the same tensor on Metal GPU. Common patterns affected:

- `F.split(x, 3)` (e.g., splitting Q/K/V in attention)
- Any graph that returns multiple views of the same buffer
- Multi-head attention implementations that split heads

## Relationship to slice offset bug

These are **independent bugs** with different symptoms:

| | Slice offset bug | Multi-view output bug |
|---|---|---|
| **Symptom** | Reads data from offset 0 | Returns zeros |
| **Trigger** | Non-zero slice consumed by computation | 3+ slice outputs from same buffer |
| **Output path** | Works correctly | Fails (this bug) |
| **Threshold** | Any non-zero offset | Exactly 3+ outputs |

Both bugs must be fixed independently for correct Metal GPU execution of models using tensor slicing.
