# MAX aarch64 GPU conv bug reports (drafts for modular/modular)

**Date:** 2026-06-25
**Reporter context:** mojo-audio RVC voice-conversion pipeline on DGX Spark.
**Reproducers:** `scripts/probe_native_conv.py` (this branch). Full session
notes in `.superpowers/sdd/task-1-report.md`.

Shared environment block (paste into both issues):

```
MAX:    26.4.0.dev2026061006
Mojo:   1.0.0b2.dev2026061006
GPU:    NVIDIA GB10 (Grace Blackwell), CUDA 13.0
Arch:   linux-aarch64 (DGX Spark, Ubuntu 24.04)
cuDNN:  nvidia_cudnn_cu13 == 9.1.9
```

> cuDNN env note (not a bug, but needed to reproduce): the pixi env runs Python
> 3.12 but `nvidia_cudnn_cu13` installs `libcudnn.so.9` under the 3.13
> site-packages. Without `LD_LIBRARY_PATH` pointing at that dir, MAX hard-aborts
> with "symbol not found: cudnnCreate" before reaching the bug.

---

## Issue 1 — `[BUG]: ops.conv2d with dilation>1 fails at runtime on GPU (aarch64): "Non-unit dilation is not supported yet"`

**Summary.** `ops.conv2d` with a non-unit `dilation` builds the graph
successfully but fails at execution time inside the compiled GPU kernel on
aarch64 (GB10). `dilation=(1,1)` works; `dilation=(3,1)` and `(5,1)` fail.

**Expected.** Dilated conv2d produces the same result as PyTorch
`F.conv2d(..., dilation=d)` (verified equivalence: a kernel-expansion
workaround with `dilation=(1,1)` matches PyTorch at max_diff ~3.8e-5).

**Actual.** Runtime kernel error:

```
An error occurred in kernel entry point named "region_7":
... (dilation lowering) "Non-unit dilation is not supported yet"
```

The graph compiles; the failure is at `model.execute()`, thrown from the
compiled CUDA kernel — suggesting the GPU conv kernel path does not implement
non-unit dilation.

**Minimal reproducer.**

```python
import numpy as np
from max.driver import Accelerator
from max.dtype import DType
from max.engine import InferenceSession
from max.graph import Graph, TensorType, DeviceRef, ops

DEV = DeviceRef.GPU()
SESSION = InferenceSession(devices=[Accelerator()])
C_in = C_out = 64; K = 3; T = 50; d = 3
x = np.random.default_rng(0).standard_normal((1, T, 1, C_in)).astype(np.float32)
w = np.random.default_rng(1).standard_normal((C_out, C_in, K)).astype(np.float32)
w_rscf = np.transpose(w[..., None], (2, 3, 1, 0)).copy()  # [K,1,C_in,C_out]
pad = d * (K - 1) // 2
with Graph("d", input_types=[TensorType(DType.float32, [1, T, 1, C_in], DEV)]) as g:
    out = ops.conv2d(g.inputs[0], ops.constant(w_rscf, device=DEV),
                     stride=(1, 1), dilation=(d, 1), padding=(pad, pad, 0, 0))
    g.output(out)
SESSION.load(g).execute(<x as device tensor>)  # raises at execute()
```

**Impact.** Blocks every dilated conv on GPU (WaveNet-style residual stacks,
HiFiGAN ResBlocks). Workaround is kernel expansion (insert d-1 zeros between
taps, run as `dilation=(1,1)`) — correct, but adds graph ops and regresses
full-graph performance.

**Question for the team.** Is non-unit dilation simply not yet implemented in
the GPU conv kernel (a feature gap), or a bug? If unimplemented, is it on the
roadmap?

---

## Issue 2 — `[BUG]: ops.conv2d_transpose fails on GPU (aarch64) with cuDNN ALLOC_FAILED even for 1x1x1 inputs`

**Summary.** `ops.conv2d_transpose` fails at execution on aarch64 (GB10) with a
cuDNN `ALLOC_FAILED`, even for a minimal 1×1 kernel / `C_in=1` / `T=5` input —
so it is not memory exhaustion. The CPU fallback also fails (cannot lower to
Mojo). The graph builds and the output shape is inferred correctly; the failure
is at execution.

**Expected.** `ops.conv2d_transpose` produces the same result as PyTorch
`F.conv_transpose2d` / `conv_transpose1d`.

**Actual (GPU).** With `CUDNN_LOGINFO_DBG=1`, cuDNN is called for
`cudnnGetConvolutionBackwardDataWorkspaceSize` and returns `ALLOC_FAILED`. The
filter descriptor is created as `CUDNN_TENSOR_NCHW` even though MAX's input is
NHWC — i.e. a layout/descriptor mismatch in MAX's cuDNN dispatch for NHWC
transpose conv on aarch64. Fails even at `C_in=1, T=5, K=1`.

**Actual (CPU fallback).**
```
Failed to infer parameter `num_groups` — Could not lower this operation to Mojo.
```

**Minimal reproducer.**

```python
import numpy as np
from max.driver import Accelerator
from max.dtype import DType
from max.engine import InferenceSession
from max.graph import Graph, TensorType, DeviceRef, ops

DEV = DeviceRef.GPU()
SESSION = InferenceSession(devices=[Accelerator()])
C_in = C_out = 64; K = 4; S = 2; T = 20
x = np.random.default_rng(1).standard_normal((1, T, 1, C_in)).astype(np.float32)
w = np.random.default_rng(2).standard_normal((C_in, C_out, K)).astype(np.float32)
w_rscf = np.transpose(w[..., None], (2, 3, 1, 0)).copy()  # [K,1,C_out,C_in]
with Graph("t", input_types=[TensorType(DType.float32, [1, T, 1, C_in], DEV)]) as g:
    out = ops.conv2d_transpose(g.inputs[0], ops.constant(w_rscf, device=DEV),
                               stride=(S, 1), padding=((K - S) // 2, (K - S) // 2, 0, 0),
                               output_paddings=(0, 0))
    g.output(out)
SESSION.load(g).execute(<x as device tensor>)  # raises cuDNN ALLOC_FAILED at execute()
```

(Also reproduces with a 2D transpose: input `[1,8,8,32]`, weight `[32,32,3,3]`,
`stride=(2,2)`, `padding=(1,1,1,1)`.)

**Additional note.** MAX 26.4 only accepts `output_paddings=(0,0)`; non-zero
output padding (needed to match PyTorch `output_padding=1`, e.g. RMVPE's
decoder doubling H→2H) is rejected at build time. Worth confirming whether
non-zero `output_paddings` is intended to be supported.

**Impact.** Blocks all transposed convolutions on GPU (vocoder upsamplers,
U-Net decoders). Workaround is zero-interleave + flipped-kernel plain conv2d —
numerically correct (max_diff ~1.1–2.3e-5) but the multi-op chain is not fused
and regresses full-graph performance badly.
