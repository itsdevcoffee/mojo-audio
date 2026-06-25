# Native conv2d Rewrite Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Replace the im2col+matmul convolution workaround with native MAX `ops.conv2d` / `ops.conv2d_transpose` in RMVPE, HiFiGAN, and VITS on the DGX Spark, closing the GPU perf gap vs Applio and removing the im2col numerical drift.

**Architecture:** A single new module `src/models/_conv.py` provides native conv primitives (`conv1d`, `conv2d`, `conv_transpose1d`, `conv_transpose2d`). Each model's existing im2col helper is rewritten to delegate to it, keeping call-site signatures unchanged. A correctness probe validates the two unverified variants (dilated, transpose) on aarch64 before any swap; any variant that fails the probe keeps its im2col body.

**Tech Stack:** Python, MAX `max.graph` (26.4 nightly), NumPy, PyTorch (ground truth), pytest, pixi.

## Global Constraints

- **Platform: Spark only.** All builds, tests, and benchmarks run on the DGX Spark (aarch64 GB10) via SSH. The Fedora x64 box is NOT used for anything (conv2d is broken on x64 per #6248). Never run tests locally.
- **MAX version: 26.4.** `max = "==26.4.0.dev2026061006"`, `mojo = "==1.0.0b2.dev2026061006"`. The pipeline already runs on 26.4.
- **torchvision stays stashed** at `/tmp/tv-stash` on the Spark (moved out of site-packages) so the Applio comparison harness imports cleanly. Do not restore it.
- **Native weight layout:** PyTorch Conv `[C_out, C_in, K]` → MAX RSCF `[K, 1, C_in, C_out]` via the existing `_pt_weight_to_max` in `src/models/_feature_extractor.py:35`. Reuse it; do not reimplement.
- **Correctness threshold:** native vs PyTorch `max_diff < 1e-5`.
- **Probe-failure rule:** if a variant fails the probe, that variant keeps its im2col body, a MAX bug is filed referencing #6248, and the rest proceed. Never block the whole rewrite on one variant.
- **Reference impl already in tree:** `AudioEncoder` (`src/models/audio_encoder.py:174,246`) already calls native `ops.conv2d` with `stride=` and `groups=16` in production — follow its pattern.

### SYNC + TEST convention (used by every task)

Code is edited locally, synced to the Spark, and tested there.

**SYNC** (run from the local repo root after editing):
```bash
rsync -az src/models/ visage@visage-spark:~/repos/mojo-audio/src/models/
rsync -az scripts/   visage@visage-spark:~/repos/mojo-audio/scripts/
rsync -az tests/     visage@visage-spark:~/repos/mojo-audio/tests/
```

**TEST/RUN** (template — substitute the pixi task or script):
```bash
ssh visage@visage-spark 'export PATH=$HOME/.pixi/bin:$PATH; cd ~/repos/mojo-audio && pixi run <task>'
```

---

### Task 0: Put the Spark checkout on the feature branch

**Files:** none (environment setup).

**Interfaces:**
- Produces: a Spark checkout at `~/repos/mojo-audio` on branch `feat/native-conv2d`, MAX 26.4 installed, torchvision stashed.

- [ ] **Step 1: Push the local branch**

Run (local):
```bash
git push -u origin feat/native-conv2d
```
Expected: branch pushed to origin.

- [ ] **Step 2: Check out the branch on the Spark and bump to 26.4**

Run:
```bash
ssh visage@visage-spark 'export PATH=$HOME/.pixi/bin:$PATH; cd ~/repos/mojo-audio
  git fetch origin && git checkout feat/native-conv2d && git pull
  sed -i "s|^max = .*|max = \"==26.4.0.dev2026061006\"|; s|^mojo = .*|mojo = \"==1.0.0b2.dev2026061006\"|" pixi.toml
  pixi install'
```
Expected: branch checked out, `pixi install` ends with "The default environment has been installed."

- [ ] **Step 3: Verify env (26.4, torchvision stashed, pipeline imports)**

Run:
```bash
ssh visage@visage-spark 'export PATH=$HOME/.pixi/bin:$PATH; cd ~/repos/mojo-audio
  pixi list 2>/dev/null | grep -E "^(max|mojo) "
  pixi run python -c "from transformers import HubertModel; print(\"imports OK\")"'
```
Expected: `max 26.4.0.dev2026061006`, `mojo 1.0.0b2...`, and `imports OK`. If `torchvision::nms` error appears, move it aside: `mv ~/repos/mojo-audio/.pixi/envs/default/lib/python3.13/site-packages/torchvision* /tmp/tv-stash/`.

---

### Task 1: Correctness probe for native conv variants

**Files:**
- Create: `scripts/probe_native_conv.py`

**Interfaces:**
- Produces: a printed PASS/FAIL table for `plain`, `dilated(1,3,5)`, `convT1d`, `convT2d`. The results decide which variants the migration tasks swap vs keep on im2col.

- [ ] **Step 1: Write the probe script**

Create `scripts/probe_native_conv.py`:
```python
"""Probe native MAX conv ops vs PyTorch ground truth on aarch64.

Validates the conv variants the pipeline uses that were NOT already verified
(dilated conv1d, ConvTranspose1d, ConvTranspose2d). Plain/strided/grouped are
already proven in AudioEncoder. Threshold: max_diff < 1e-5.

Run on the Spark:  pixi run python scripts/probe_native_conv.py
"""
import numpy as np
import torch
import torch.nn.functional as F
from max.driver import Accelerator
from max.dtype import DType
from max.engine import InferenceSession
from max.graph import Graph, TensorType, DeviceRef, ops

DEV = DeviceRef.GPU()
SESSION = InferenceSession(devices=[Accelerator()])
THRESH = 1e-5


def _pt_to_rscf(w):  # PyTorch conv [C_out, C_in, K] -> MAX RSCF [K, 1, C_in, C_out]
    return np.transpose(w[..., None], (2, 3, 1, 0)).copy()


def _run(graph):
    model = SESSION.load(graph)
    return model


def probe_dilated(C_in=64, C_out=64, K=3, T=50, dilation=1):
    rng = np.random.default_rng(0)
    x = rng.standard_normal((1, T, 1, C_in)).astype(np.float32)
    w = rng.standard_normal((C_out, C_in, K)).astype(np.float32)
    # PyTorch ground truth (NCT)
    xt = torch.from_numpy(x).squeeze(2).transpose(1, 2)  # [1, C_in, T]
    pad = dilation * (K - 1) // 2
    yt = F.conv1d(xt, torch.from_numpy(w), padding=pad, dilation=dilation)
    yt = yt.transpose(1, 2).unsqueeze(2).numpy()  # [1, T, 1, C_out]
    # MAX native
    with Graph("d", input_types=[TensorType(DType.float32, [1, T, 1, C_in], DEV)]) as g:
        out = ops.conv2d(
            g.inputs[0], ops.constant(_pt_to_rscf(w), device=DEV),
            stride=(1, 1), dilation=(dilation, 1), padding=(pad, pad, 0, 0),
        )
        g.output(out)
    ym = _run(g).execute(x)[0].to_numpy()
    d = float(np.abs(ym - yt).max())
    print(f"dilated d={dilation:>1}: max_diff={d:.3e}  {'PASS' if d < THRESH else 'FAIL'}")
    return d < THRESH


def probe_convT1d(C_in=64, C_out=64, K=4, S=2, T=20):
    rng = np.random.default_rng(1)
    x = rng.standard_normal((1, T, 1, C_in)).astype(np.float32)
    w = rng.standard_normal((C_in, C_out, K)).astype(np.float32)  # ConvT weight [C_in,C_out,K]
    pad = (K - S) // 2
    xt = torch.from_numpy(x).squeeze(2).transpose(1, 2)
    yt = F.conv_transpose1d(xt, torch.from_numpy(w), stride=S, padding=pad)
    yt = yt.transpose(1, 2).unsqueeze(2).numpy()
    # RSCF for transpose: [K, 1, C_in, C_out] from [C_in, C_out, K]
    w_rscf = np.transpose(w[..., None], (2, 3, 0, 1)).copy()
    with Graph("t1", input_types=[TensorType(DType.float32, [1, T, 1, C_in], DEV)]) as g:
        out = ops.conv2d_transpose(
            g.inputs[0], ops.constant(w_rscf, device=DEV),
            stride=(S, 1), padding=(pad, pad, 0, 0), output_paddings=(0, 0),
        )
        g.output(out)
    ym = _run(g).execute(x)[0].to_numpy()
    n = min(ym.shape[1], yt.shape[1])
    d = float(np.abs(ym[:, :n] - yt[:, :n]).max())
    print(f"convT1d S={S}: max_diff={d:.3e}  {'PASS' if d < THRESH else 'FAIL'}  shapes ym={ym.shape} yt={yt.shape}")
    return d < THRESH


def probe_convT2d(C_in=32, C_out=32, K=3, S=2, H=8, W=8):
    rng = np.random.default_rng(2)
    x = rng.standard_normal((1, H, W, C_in)).astype(np.float32)
    w = rng.standard_normal((C_in, C_out, K, K)).astype(np.float32)
    xt = torch.from_numpy(x).permute(0, 3, 1, 2)  # NCHW
    yt = F.conv_transpose2d(xt, torch.from_numpy(w), stride=S, padding=1, output_padding=1)
    yt = yt.permute(0, 2, 3, 1).numpy()  # NHWC
    w_rscf = np.transpose(w, (2, 3, 0, 1)).copy()  # [K,K,C_in,C_out]
    with Graph("t2", input_types=[TensorType(DType.float32, [1, H, W, C_in], DEV)]) as g:
        out = ops.conv2d_transpose(
            g.inputs[0], ops.constant(w_rscf, device=DEV),
            stride=(S, S), padding=(1, 1, 1, 1), output_paddings=(1, 1),
        )
        g.output(out)
    ym = _run(g).execute(x)[0].to_numpy()
    d = float(np.abs(ym - yt).max())
    print(f"convT2d S={S}: max_diff={d:.3e}  {'PASS' if d < THRESH else 'FAIL'}  shapes ym={ym.shape} yt={yt.shape}")
    return d < THRESH


if __name__ == "__main__":
    results = {
        "dilated_1": probe_dilated(dilation=1),
        "dilated_3": probe_dilated(dilation=3),
        "dilated_5": probe_dilated(dilation=5),
        "convT1d": probe_convT1d(),
        "convT2d": probe_convT2d(),
    }
    print("\nSUMMARY:", {k: ("PASS" if v else "FAIL") for k, v in results.items()})
```

- [ ] **Step 2: Sync to the Spark**

Run the SYNC block (see Global Constraints).

- [ ] **Step 3: Run the probe on the Spark**

Run:
```bash
ssh visage@visage-spark 'export PATH=$HOME/.pixi/bin:$PATH; cd ~/repos/mojo-audio && pixi run python scripts/probe_native_conv.py'
```
Expected: a table ending in `SUMMARY: {...}`. Record which variants PASS. The exact transpose padding/output_padding mapping that yields PASS is the mapping used in Task 2's transpose helpers. If a transpose shape mismatches, adjust `padding`/`output_paddings` until shapes match PyTorch and re-run (this is the documented purpose of the probe).

- [ ] **Step 4: Commit**

```bash
git add scripts/probe_native_conv.py
git commit -m "feat(probe): native conv variant correctness probe for aarch64"
```

---

### Task 2: Shared native conv module `_conv.py`

**Files:**
- Create: `src/models/_conv.py`
- Test: `tests/test_conv.py`

**Interfaces:**
- Consumes: `_pt_weight_to_max` from `src/models/_feature_extractor.py`.
- Produces:
  - `conv1d(x, w_pt, b_np, dilation=1, groups=1, device_ref=None) -> TensorValue` — input NHWC `[B,T,1,C_in]`, PyTorch weight `[C_out,C_in,K]`, output `[B,T,1,C_out]` (same padding).
  - `conv2d(x, w_max, b_np, stride=(1,1), padding=(0,0,0,0), groups=1, device_ref=None) -> TensorValue` — weight already MAX RSCF `[kH,kW,C_in,C_out]`.
  - `conv_transpose1d(x, w_pt, b_np, stride, device_ref=None) -> TensorValue` — PyTorch ConvT weight `[C_in,C_out,K]`, output `[B,T*stride,1,C_out]`.
  - `conv_transpose2d(x, w_pt, b_np, stride=2, device_ref=None) -> TensorValue` — stride-2 K3 pad1 output_pad1, PyTorch weight `[C_in,C_out,3,3]`.

- [ ] **Step 1: Write the failing test**

Create `tests/test_conv.py`:
```python
import numpy as np
import pytest
import torch
import torch.nn.functional as F
from max.driver import Accelerator
from max.dtype import DType
from max.engine import InferenceSession
from max.graph import Graph, TensorType, DeviceRef, ops
from models import _conv

DEV = DeviceRef.GPU()
SESSION = InferenceSession(devices=[Accelerator()])


def _exec(build, x):
    with Graph("t", input_types=[TensorType(DType.float32, list(x.shape), DEV)]) as g:
        g.output(build(g.inputs[0]))
    return SESSION.load(g).execute(x)[0].to_numpy()


def test_conv1d_dilated_matches_torch():
    rng = np.random.default_rng(0)
    C_in, C_out, K, T, d = 64, 96, 3, 40, 3
    x = rng.standard_normal((1, T, 1, C_in)).astype(np.float32)
    w = rng.standard_normal((C_out, C_in, K)).astype(np.float32)
    b = rng.standard_normal((C_out,)).astype(np.float32)
    ym = _exec(lambda inp: _conv.conv1d(inp, w, b, dilation=d, device_ref=DEV), x)
    xt = torch.from_numpy(x).squeeze(2).transpose(1, 2)
    yt = F.conv1d(xt, torch.from_numpy(w), torch.from_numpy(b),
                  padding=d * (K - 1) // 2, dilation=d).transpose(1, 2).unsqueeze(2).numpy()
    assert np.abs(ym - yt).max() < 1e-5
```

- [ ] **Step 2: Run the test, verify it fails**

Run SYNC, then:
```bash
ssh visage@visage-spark 'export PATH=$HOME/.pixi/bin:$PATH; cd ~/repos/mojo-audio && pixi run pytest tests/test_conv.py -v'
```
Expected: FAIL with `ModuleNotFoundError: ... _conv` or `AttributeError: conv1d`.

- [ ] **Step 3: Write `_conv.py`**

Create `src/models/_conv.py`:
```python
"""Native MAX conv primitives (ops.conv2d / ops.conv2d_transpose).

Replaces the im2col+matmul workaround now that conv2d is correct on aarch64
(modular/modular#6129 fixed; #6248 fixed on aarch64). Weight layout helper is
shared with AudioEncoder via _pt_weight_to_max.
"""
import numpy as np
from max.graph import ops
from ._feature_extractor import _pt_weight_to_max  # [C_out,C_in,K] -> RSCF [K,1,C_in,C_out]


def _bias(b_np, device_ref):
    return ops.constant(np.asarray(b_np, dtype=np.float32), device=device_ref) if b_np is not None else None


def conv1d(x, w_pt, b_np, dilation=1, groups=1, device_ref=None):
    """Conv1d as conv2d with kernel (K,1). x: NHWC [B,T,1,C_in]. w_pt: [C_out,C_in,K]."""
    C_out, C_in, K = w_pt.shape
    w_max = _pt_weight_to_max(w_pt)  # [K,1,C_in,C_out]
    pad = dilation * (K - 1) // 2
    return ops.conv2d(
        x, ops.constant(w_max, device=device_ref),
        stride=(1, 1), dilation=(dilation, 1),
        padding=(pad, pad, 0, 0), groups=groups, bias=_bias(b_np, device_ref),
    )


def conv2d(x, w_max, b_np, stride=(1, 1), padding=(0, 0, 0, 0), groups=1, device_ref=None):
    """Direct conv2d. w_max already MAX RSCF [kH,kW,C_in,C_out]."""
    return ops.conv2d(
        x, ops.constant(np.asarray(w_max, dtype=np.float32), device=device_ref),
        stride=stride, padding=padding, groups=groups, bias=_bias(b_np, device_ref),
    )


def conv_transpose1d(x, w_pt, b_np, stride, device_ref=None):
    """ConvTranspose1d as conv2d_transpose. w_pt: [C_in,C_out,K]. stride upsamples T."""
    C_in, C_out, K = w_pt.shape
    pad = (K - stride) // 2
    w_rscf = np.transpose(w_pt[..., None], (2, 3, 0, 1)).copy()  # [K,1,C_in,C_out]
    return ops.conv2d_transpose(
        x, ops.constant(w_rscf, device=device_ref),
        stride=(stride, 1), padding=(pad, pad, 0, 0),
        output_paddings=(0, 0), bias=_bias(b_np, device_ref),
    )


def conv_transpose2d(x, w_pt, b_np, stride=2, device_ref=None):
    """ConvTranspose2d stride=2 K3 pad1 output_pad1. w_pt: [C_in,C_out,3,3]."""
    w_rscf = np.transpose(w_pt, (2, 3, 0, 1)).copy()  # [3,3,C_in,C_out]
    return ops.conv2d_transpose(
        x, ops.constant(w_rscf, device=device_ref),
        stride=(stride, stride), padding=(1, 1, 1, 1),
        output_paddings=(1, 1), bias=_bias(b_np, device_ref),
    )
```

> NOTE: if Task 1's probe found a different padding/output_paddings mapping for the transpose variants, use that exact mapping here. If a transpose variant FAILED the probe, leave its helper body raising `NotImplementedError("keep im2col — probe failed")` and the migration task keeps the old im2col helper for it.

- [ ] **Step 4: Run the test, verify it passes**

Run SYNC, then:
```bash
ssh visage@visage-spark 'export PATH=$HOME/.pixi/bin:$PATH; cd ~/repos/mojo-audio && pixi run pytest tests/test_conv.py -v'
```
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add src/models/_conv.py tests/test_conv.py
git commit -m "feat(conv): shared native ops.conv2d primitives module"
```

---

### Task 3: Migrate RMVPE to native conv2d (+ fix accumulation drift)

**Files:**
- Modify: `src/models/_rmvpe.py` — `_conv2d` (line 108), `_conv_transpose_2x` (line 192).
- Test: `tests/test_pitch_extractor.py` (existing).

**Interfaces:**
- Consumes: `conv2d`, `conv_transpose2d` from `src/models/_conv.py`.
- Produces: RMVPE U-Net running on native conv. Expectation: `test_salience_matches_pytorch` flips xfail → pass (im2col drift removed).

- [ ] **Step 1: Capture the baseline (xfail) on the Spark**

Run:
```bash
ssh visage@visage-spark 'export PATH=$HOME/.pixi/bin:$PATH; cd ~/repos/mojo-audio && pixi run test-pitch-extractor-full 2>&1 | tail -15'
```
Expected: 28 passed, 1 xfailed (`test_salience_matches_pytorch`). Record this.

- [ ] **Step 2: Rewrite `_conv2d` to delegate to native**

In `src/models/_rmvpe.py`, replace the body of `_conv2d(x, w_np, b_np, stride, padding, device_ref)` (lines 108–189) with a delegation. Keep the signature identical. The weight `w_np` is already MAX `[kH,kW,C_in,C_out]`:
```python
def _conv2d(x, w_np, b_np, stride, padding, device_ref):
    """Native conv2d (was im2col; conv2d fixed on aarch64). Signature unchanged."""
    from ._conv import conv2d as _native_conv2d
    return _native_conv2d(x, w_np, b_np, stride=stride, padding=padding, device_ref=device_ref)
```

- [ ] **Step 3: Rewrite `_conv_transpose_2x` to delegate to native**

Replace the body of `_conv_transpose_2x(x, w_pt, b_np, device_ref)` (starts line 192) with:
```python
def _conv_transpose_2x(x, w_pt, b_np, device_ref):
    """Native ConvTranspose2d stride=2 (was hand-rolled). Signature unchanged."""
    from ._conv import conv_transpose2d
    return conv_transpose2d(x, w_pt, b_np, stride=2, device_ref=device_ref)
```

- [ ] **Step 4: Run the RMVPE suite on the Spark**

Run SYNC, then:
```bash
ssh visage@visage-spark 'export PATH=$HOME/.pixi/bin:$PATH; cd ~/repos/mojo-audio && pixi run test-pitch-extractor-full 2>&1 | tail -20'
```
Expected: all prior passes still green. `test_salience_matches_pytorch` now PASSES (or stays xfail if drift wasn't the sole cause — acceptable, not a regression). If any previously-passing test FAILS, revert `_rmvpe.py` and investigate before continuing.

- [ ] **Step 5: If salience now passes, remove its xfail marker**

If `test_salience_matches_pytorch` passed, delete the `@pytest.mark.xfail(...)` decorator on it in `tests/test_pitch_extractor.py` (around lines 536–541 per the audit) so the suite enforces it going forward. Re-run Step 4 to confirm it passes unmarked.

- [ ] **Step 6: Commit**

```bash
git add src/models/_rmvpe.py tests/test_pitch_extractor.py
git commit -m "perf(rmvpe): native ops.conv2d — removes im2col U-Net accumulation drift"
```

---

### Task 4: Migrate HiFiGAN to native conv

**Files:**
- Modify: `src/models/_hifigan_graph.py` — `conv1d` (line 151), `conv_transpose_1d` (line 36).
- Test: `tests/test_hifigan.py` (existing).

**Interfaces:**
- Consumes: `conv1d`, `conv_transpose1d` from `src/models/_conv.py`.
- Produces: HiFiGAN vocoder on native conv. Call-sites (lines 255–506) unchanged.

- [ ] **Step 1: Capture the baseline on the Spark**

Run:
```bash
ssh visage@visage-spark 'export PATH=$HOME/.pixi/bin:$PATH; cd ~/repos/mojo-audio && pixi run test-hifigan-full 2>&1 | tail -15'
```
Expected: 15 passed, 1 xfailed (batch>1). Record.

- [ ] **Step 2: Rewrite `conv1d` to delegate (with dilated fallback if probe failed)**

In `src/models/_hifigan_graph.py`, replace the body of `conv1d(x, w_np, b_np, dilation=1, device_ref=None)` (lines 151–end of function) with:
```python
def conv1d(x, w_np, b_np, dilation=1, device_ref=None):
    """Native Conv1d (was im2col). Signature unchanged."""
    from ._conv import conv1d as _native_conv1d
    return _native_conv1d(x, w_np, b_np, dilation=dilation, device_ref=device_ref)
```
IF the probe found dilated conv FAILS on aarch64: keep the original im2col body, rename it `_conv1d_im2col`, and make `conv1d` branch: `return _native_conv1d(...) if dilation == 1 else _conv1d_im2col(x, w_np, b_np, dilation, device_ref)`.

- [ ] **Step 3: Rewrite `conv_transpose_1d` to delegate**

Replace the body of `conv_transpose_1d(x, w_pt, b_np, *, stride, device_ref)` (line 36) with:
```python
def conv_transpose_1d(x, w_pt, b_np, *, stride, device_ref):
    """Native ConvTranspose1d (was zero-interleave im2col). Signature unchanged."""
    from ._conv import conv_transpose1d
    return conv_transpose1d(x, w_pt, b_np, stride=stride, device_ref=device_ref)
```
IF the probe found convT1d FAILS: skip this step, keep the im2col transpose.

- [ ] **Step 4: Run the HiFiGAN suite on the Spark**

Run SYNC, then:
```bash
ssh visage@visage-spark 'export PATH=$HOME/.pixi/bin:$PATH; cd ~/repos/mojo-audio && pixi run test-hifigan-full 2>&1 | tail -20'
```
Expected: 15 passed, 1 xfailed (batch>1 unchanged). The HiFiGAN-vs-PyTorch correlation test must stay ≥0.999. If a pass regresses, revert and investigate.

- [ ] **Step 5: Commit**

```bash
git add src/models/_hifigan_graph.py
git commit -m "perf(hifigan): native ops.conv2d/conv2d_transpose for vocoder convs"
```

---

### Task 5: Migrate VITS to native conv

**Files:**
- Modify: `src/models/_vits_graph.py` — `_conv1d_bct` (line 45).
- Test: `tests/test_vits.py` (existing).

**Interfaces:**
- Consumes: `conv1d` from `src/models/_conv.py`.
- Produces: VITS enc_p/flow convs on native conv. All call-sites unchanged.

- [ ] **Step 1: Capture the baseline on the Spark**

Run:
```bash
ssh visage@visage-spark 'export PATH=$HOME/.pixi/bin:$PATH; cd ~/repos/mojo-audio && pixi run test-vits-full 2>&1 | tail -15'
```
Expected: 50 passed (flow corr 1.0, enc_p corr 1.0). Record.

- [ ] **Step 2: Rewrite `_conv1d_bct` to delegate**

Inspect `_conv1d_bct` (line 45): it takes `[B,C,T]` ("bct"), converts to NHWC, calls the local `conv1d`, converts back. Replace its internal `conv1d(...)` call (line ~63, `out_nhwc = conv1d(x_nhwc, ...)`) with the native module:
```python
    from ._conv import conv1d as _native_conv1d
    out_nhwc = _native_conv1d(x_nhwc, w_np, b_np, dilation=dilation, device_ref=device_ref)
```
Leave the surrounding BCT↔NHWC layout conversion untouched. IF the probe found dilated conv FAILS: branch on `dilation == 1` exactly as in Task 4 Step 2, keeping the original im2col `conv1d` for dilation>1.

- [ ] **Step 3: Run the VITS suite on the Spark**

Run SYNC, then:
```bash
ssh visage@visage-spark 'export PATH=$HOME/.pixi/bin:$PATH; cd ~/repos/mojo-audio && pixi run test-vits-full 2>&1 | tail -20'
```
Expected: 50 passed, enc_p and flow correlation still 1.0. If regressed, revert and investigate.

- [ ] **Step 4: Run the FULL suite to confirm no cross-model regression**

Run:
```bash
ssh visage@visage-spark 'export PATH=$HOME/.pixi/bin:$PATH; cd ~/repos/mojo-audio
  pixi run test-models-full && pixi run test-pitch-extractor-full && pixi run test-hifigan-full && pixi run test-vits-full 2>&1 | tail -8'
```
Expected: full green (128 pass baseline, ideally 129 if RMVPE salience flipped; ≤1 xfail for batch>1).

- [ ] **Step 5: Commit**

```bash
git add src/models/_vits_graph.py
git commit -m "perf(vits): native ops.conv2d for enc_p/flow convs"
```

---

### Task 6: Benchmark, quality check, and record results

**Files:**
- Modify: `docs/project/04-17-2026-backlog-radar.md` (update Priority 2 #1 status).
- Create: `docs/benchmarks/06-19-2026-native-conv2d-results.md`

**Interfaces:**
- Consumes: the migrated pipeline on the Spark.
- Produces: a measured GPU RTF vs the 0.42 im2col baseline, a quality delta, and an updated radar.

- [ ] **Step 1: Run the GPU benchmark on the Spark (same models as the controlled baseline)**

Run:
```bash
ssh visage@visage-spark 'export PATH=$HOME/.pixi/bin:$PATH; cd ~/repos/mojo-audio
  pixi run python scripts/benchmark_suite.py run --mojo-only --mojo-device gpu \
    --models melodic-male-singer-1,falsetto-male-soul-singer-1 2>&1 | tail -15'
```
Expected: a Summary with Mean RTF. Compare to the im2col baseline (26.3: 0.423; 26.4 im2col: ~0.52). Target ≤0.15.

- [ ] **Step 2: Run the head-to-head quality comparison vs Applio**

Run:
```bash
ssh visage@visage-spark 'export PATH=$HOME/.pixi/bin:$PATH; cd ~/repos/mojo-audio
  pixi run python scripts/compare_vs_applio.py \
    --model /home/visage/repos/shade/models/melodic-male-singer-1/model.pth \
    --audio /home/visage/repos/Applio/logs/10f05a97-efae-4ea8-ae87-2c256fa42be5/sliced_audios/0_0_0.wav \
    --mojo-device gpu 2>&1 | tail -25'
```
Expected: waveform / mel / F0 correlation numbers. They should hold or improve vs the im2col baseline (F0 especially, given the RMVPE drift fix).

- [ ] **Step 3: Write the benchmark results doc**

Create `docs/benchmarks/06-19-2026-native-conv2d-results.md` with: the per-model RTF before (im2col) and after (native), the % improvement, whether ≤0.15 was reached, the probe results table (which variants went native), the RMVPE salience xfail outcome, and the Applio quality correlations from Step 2.

- [ ] **Step 4: Update the backlog radar**

In `docs/project/04-17-2026-backlog-radar.md`, mark Priority 2 #1 (im2col→native conv2d) done with the measured RTF, and update the "switch criteria" line with whether mojo-audio now meets the Shade perf bar.

- [ ] **Step 5: Commit and push**

```bash
git add docs/benchmarks/06-19-2026-native-conv2d-results.md docs/project/04-17-2026-backlog-radar.md
git commit -m "docs(bench): native conv2d results — RTF <measured> vs 0.42 im2col"
git push
```

---

## Self-Review notes

- **Spec coverage:** probe (§Probe) → Task 1; shared `_conv.py` (§Architecture) → Task 2; RMVPE/HiFiGAN/VITS migration order (§Migration) → Tasks 3/4/5; perf+correctness+quality success criteria (§Success) → Task 6; per-variant im2col fallback (§probe-failure rule) → Task 2 NOTE + Task 4/5 branches; 26.4 + torchvision (§Platform) → Task 0.
- **Probe-failure handling** is threaded through every migration task (explicit "IF the probe found … FAILS" branches), satisfying the Global Constraint.
- **Type consistency:** `_conv.conv1d/conv2d/conv_transpose1d/conv_transpose2d` signatures in Task 2 match every call in Tasks 3–5.
