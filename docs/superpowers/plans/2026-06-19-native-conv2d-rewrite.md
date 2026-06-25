# Native conv2d Rewrite Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Replace the im2col+matmul convolution workaround with native MAX `ops.conv2d` / `ops.conv2d_transpose` in RMVPE, HiFiGAN, and VITS on the DGX Spark, closing the GPU perf gap vs Applio and removing the im2col numerical drift.

**Architecture:** A single new module `src/models/_conv.py` provides native conv primitives (`conv1d`, `conv2d`, `conv_transpose1d`, `conv_transpose2d`). Each model's existing im2col helper is rewritten to delegate to it, keeping call-site signatures unchanged.

**PROBE OUTCOME (2026-06-25 — drives this plan):** native `ops.conv2d` on the Spark GPU (MAX 26.4) supports ONLY plain dilation=1. Native dilated conv ("non-unit dilation not supported yet") and native `ops.conv2d_transpose` (cuDNN ALLOC_FAILED) both FAIL. **Strategy: express-as-plain.** Every variant is reduced to native plain conv2d, validated correct (max_diff ~1–4e-5) and ~1.8x faster than im2col in Probe 2:
- **Dilated conv** → expand kernel with `(d-1)` zeros between taps, then native dilation=1 conv2d with the larger kernel.
- **ConvTranspose1d/2d** → zero-interleave the input, flip the kernel, then native plain conv2d.

Because express-as-plain covers ALL variants, im2col is removed entirely — no fallback path remains. The exact validated constructions (weight layouts, kernel flips, zero-interleave op sequences, padding) are in `.superpowers/sdd/task-1b-report.md` §"Implementation Details (Reuse These in Migration)" — the migration MUST follow them verbatim.

**Tech Stack:** Python, MAX `max.graph` (26.4 nightly), NumPy, PyTorch (ground truth), pytest, pixi.

## Global Constraints

- **Platform: Spark only.** All builds, tests, and benchmarks run on the DGX Spark (aarch64 GB10) via SSH. The Fedora x64 box is NOT used for anything (conv2d is broken on x64 per #6248). Never run tests locally.
- **MAX version: 26.4.** `max = "==26.4.0.dev2026061006"`, `mojo = "==1.0.0b2.dev2026061006"`. The pipeline already runs on 26.4.
- **torchvision stays stashed** at `/tmp/tv-stash` on the Spark (moved out of site-packages) so the Applio comparison harness imports cleanly. Do not restore it.
- **Native weight layout:** PyTorch Conv `[C_out, C_in, K]` → MAX RSCF `[K, 1, C_in, C_out]` via the existing `_pt_weight_to_max` in `src/models/_feature_extractor.py:35`. Reuse it; do not reimplement.
- **Correctness threshold:** native vs PyTorch `max_diff < 1e-4` (GPU fp32 accumulation gives ~2–4e-5 at high fan-in; this is not a bug, confirmed in Probe 2 by linear scaling with C_in). A real failure is orders of magnitude larger or a kernel exception.
- **Express-as-plain is mandatory** (native dilation + native transpose are broken on aarch64/26.4): dilated→kernel-expansion, transpose→zero-interleave, both via native plain conv2d. im2col is fully removed; there is no fallback path. Follow the validated constructions in `.superpowers/sdd/task-1b-report.md` §"Implementation Details" verbatim.
- **`ops.transpose` is a 2-axis swap only** in MAX 26.4 — never use it for 4-axis permutations; use the reshape+pad+reshape chain from the probe report instead.
- **MAX bugs to file** (tracked, not blocking): native dilated conv ("non-unit dilation not supported yet") and `ops.conv2d_transpose` (cuDNN ALLOC_FAILED on aarch64 NHWC). File after the rewrite lands.
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

> ✅ **DONE (2026-06-25, commits 05cb6be + 8752376).** Probe 1 found native dilated + transpose FAIL on aarch64; Probe 2 validated express-as-plain (all PASS, ~1.8x faster than im2col). Findings in `.superpowers/sdd/task-1-report.md` and `task-1b-report.md`. The architecture note and Task 2 above already reflect this. Kept below for provenance.

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

### Task 2: Shared native conv module `_conv.py` (express-as-plain)

**Files:**
- Create: `src/models/_conv.py`
- Test: `tests/test_conv.py`

**Authoritative reference:** `.superpowers/sdd/task-1b-report.md` §"Implementation Details (Reuse These in Migration)" contains the exact validated constructions (weight layouts, kernel flips, zero-interleave op sequences, padding). The code below transcribes them; if any detail differs, the report's validated version wins.

**Interfaces:**
- Consumes: `_pt_weight_to_max` from `src/models/_feature_extractor.py` (`[C_out,C_in,K]` → RSCF `[K,1,C_in,C_out]`).
- Produces:
  - `conv1d(x, w_pt, b_np, dilation=1, groups=1, device_ref=None) -> TensorValue` — NHWC `[B,T,1,C_in]`, PyTorch weight `[C_out,C_in,K]`, output `[B,T,1,C_out]`. Dilation via kernel expansion (native dilation is broken).
  - `conv2d(x, w_max, b_np, stride=(1,1), padding=(0,0,0,0), groups=1, device_ref=None) -> TensorValue` — weight already MAX RSCF `[kH,kW,C_in,C_out]`.
  - `conv_transpose1d(x, w_pt, b_np, stride, device_ref=None) -> TensorValue` — PyTorch ConvT weight `[C_in,C_out,K]`, output `[B,T*stride,1,C_out]`. Zero-interleave + native plain conv2d.
  - `conv_transpose2d(x, w_pt, b_np, stride=2, device_ref=None) -> TensorValue` — stride-2 K3 pad1 output_pad1, PyTorch weight `[C_in,C_out,3,3]`, H/W static. 2D zero-interleave + native plain conv2d.

- [ ] **Step 1: Write the failing tests**

Create `tests/test_conv.py` (follow the existing test files' import/path convention — check `tests/test_hifigan.py` for how `models` is made importable; the MAX exec idiom must match how `scripts/probe_native_conv.py` runs):
```python
import numpy as np
import torch
import torch.nn.functional as F
from max.driver import Accelerator
from max.dtype import DType
from max.engine import InferenceSession
from max.graph import Graph, TensorType, DeviceRef
from models import _conv

DEV = DeviceRef.GPU()
SESSION = InferenceSession(devices=[Accelerator()])
THRESH = 1e-4  # GPU fp32 accumulation ~2-4e-5 at high fan-in (Probe 2 confirmed)


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
    assert np.abs(ym - yt).max() < THRESH


def test_conv_transpose1d_matches_torch():
    rng = np.random.default_rng(1)
    C_in, C_out, K, S, T = 64, 64, 4, 2, 20
    x = rng.standard_normal((1, T, 1, C_in)).astype(np.float32)
    w = rng.standard_normal((C_in, C_out, K)).astype(np.float32)
    b = rng.standard_normal((C_out,)).astype(np.float32)
    ym = _exec(lambda inp: _conv.conv_transpose1d(inp, w, b, stride=S, device_ref=DEV), x)
    xt = torch.from_numpy(x).squeeze(2).transpose(1, 2)
    yt = F.conv_transpose1d(xt, torch.from_numpy(w), torch.from_numpy(b),
                            stride=S, padding=(K - S) // 2).transpose(1, 2).unsqueeze(2).numpy()
    n = min(ym.shape[1], yt.shape[1])
    assert np.abs(ym[:, :n] - yt[:, :n]).max() < THRESH


def test_conv_transpose2d_matches_torch():
    rng = np.random.default_rng(2)
    C_in, C_out, H, W = 32, 32, 8, 8
    x = rng.standard_normal((1, H, W, C_in)).astype(np.float32)
    w = rng.standard_normal((C_in, C_out, 3, 3)).astype(np.float32)
    b = rng.standard_normal((C_out,)).astype(np.float32)
    ym = _exec(lambda inp: _conv.conv_transpose2d(inp, w, b, stride=2, device_ref=DEV), x)
    xt = torch.from_numpy(x).permute(0, 3, 1, 2)
    yt = F.conv_transpose2d(xt, torch.from_numpy(w), torch.from_numpy(b),
                            stride=2, padding=1, output_padding=1).permute(0, 2, 3, 1).numpy()
    assert ym.shape == yt.shape and np.abs(ym - yt).max() < THRESH
```

- [ ] **Step 2: Run the tests, verify they fail**

Run SYNC, then (note the cuDNN `LD_LIBRARY_PATH` from the probe report may be needed):
```bash
ssh visage@visage-spark 'export PATH=$HOME/.pixi/bin:$PATH; cd ~/repos/mojo-audio && pixi run pytest tests/test_conv.py -v'
```
Expected: FAIL with `ModuleNotFoundError: ... _conv`.

- [ ] **Step 3: Write `_conv.py`**

Create `src/models/_conv.py`:
```python
"""Native MAX conv primitives via express-as-plain.

Native ops.conv2d on aarch64/26.4 supports ONLY plain dilation=1 (native
dilated conv and ops.conv2d_transpose are broken — see
.superpowers/sdd/task-1b-report.md). So dilation is done by kernel expansion
and transpose by zero-interleave, both reduced to native plain conv2d.
"""
import numpy as np
from max.graph import ops
from ._feature_extractor import _pt_weight_to_max  # [C_out,C_in,K] -> RSCF [K,1,C_in,C_out]


def _bias(b_np, device_ref):
    return ops.constant(np.asarray(b_np, dtype=np.float32), device=device_ref) if b_np is not None else None


def _dilate_kernel(w, d):
    """Insert (d-1) zeros between taps: [C_out,C_in,K] -> [C_out,C_in,(K-1)*d+1]."""
    if d == 1:
        return np.asarray(w, dtype=np.float32)
    C_out, C_in, K = w.shape
    w_eff = np.zeros((C_out, C_in, (K - 1) * d + 1), dtype=np.float32)
    w_eff[:, :, ::d] = w
    return w_eff


def conv1d(x, w_pt, b_np, dilation=1, groups=1, device_ref=None):
    """Conv1d as native plain conv2d; dilation handled by kernel expansion.
    x: NHWC [B,T,1,C_in]. w_pt: PyTorch [C_out,C_in,K]."""
    w_eff = _dilate_kernel(np.asarray(w_pt, dtype=np.float32), dilation)
    K_eff = w_eff.shape[2]
    w_max = _pt_weight_to_max(w_eff)  # [K_eff,1,C_in,C_out]
    pad = (K_eff - 1) // 2
    return ops.conv2d(
        x, ops.constant(w_max, device=device_ref),
        stride=(1, 1), dilation=(1, 1),
        padding=(pad, pad, 0, 0), groups=groups, bias=_bias(b_np, device_ref),
    )


def conv2d(x, w_max, b_np, stride=(1, 1), padding=(0, 0, 0, 0), groups=1, device_ref=None):
    """Direct native conv2d. w_max already MAX RSCF [kH,kW,C_in,C_out]."""
    return ops.conv2d(
        x, ops.constant(np.asarray(w_max, dtype=np.float32), device=device_ref),
        stride=stride, dilation=(1, 1), padding=padding, groups=groups,
        bias=_bias(b_np, device_ref),
    )


def conv_transpose1d(x, w_pt, b_np, stride, device_ref=None):
    """ConvTranspose1d via zero-interleave + native plain conv2d.
    w_pt: PyTorch [C_in,C_out,K]. x: NHWC [B,T,1,C_in]. B must be 1."""
    w_pt = np.asarray(w_pt, dtype=np.float32)
    C_in, C_out, K = w_pt.shape
    S = stride
    w_flipped = w_pt[:, :, ::-1].copy()                          # convolution = flipped corr
    w_max = np.transpose(w_flipped[..., None], (2, 3, 0, 1)).copy()  # [K,1,C_in,C_out]
    T = x.shape[1]
    x_sq = ops.squeeze(x, 0)                                     # [T,1,C_in]
    x_ins = ops.unsqueeze(x_sq, 1)                               # [T,1,1,C_in]
    x_pad = ops.pad(x_ins, [0, 0, 0, S - 1, 0, 0, 0, 0])        # [T,S,1,C_in]
    x_merged = ops.reshape(x_pad, [T * S, 1, C_in])             # [T*S,1,C_in]
    x_zi = ops.unsqueeze(x_merged, 0)                           # [1,T*S,1,C_in]
    pad_left = (K + S - 2) // 2
    pad_right = (K - S) // 2
    return ops.conv2d(
        x_zi, ops.constant(w_max, device=device_ref),
        stride=(1, 1), dilation=(1, 1),
        padding=(pad_left, pad_right, 0, 0), bias=_bias(b_np, device_ref),
    )


def conv_transpose2d(x, w_pt, b_np, stride=2, device_ref=None):
    """ConvTranspose2d (S=2,K=3,P=1,output_pad=1) via 2D zero-interleave +
    native plain conv2d. w_pt: PyTorch [C_in,C_out,3,3]. H,W must be STATIC
    (RMVPE decoder dims are fixed). ops.transpose is 2-axis only — the
    reshape/pad/reshape chain below is the validated 4D interleave."""
    w_pt = np.asarray(w_pt, dtype=np.float32)
    C_in, C_out, Kh, Kw = w_pt.shape
    S = stride
    w_flipped = w_pt[:, :, ::-1, ::-1].copy()
    w_max = np.transpose(w_flipped, (2, 3, 0, 1)).copy()        # [Kh,Kw,C_in,C_out]
    H = int(x.shape[1]); W = int(x.shape[2]); C = C_in
    H_zi = (H - 1) * S + 1
    W_zi = (W - 1) * S + 1
    # H-axis interleave of [1,H,W,C]
    xf = ops.reshape(x, [1, H, W * C])
    xf = ops.unsqueeze(xf, 2)                                   # [1,H,1,W*C]
    xf = ops.pad(xf, [0, 0, 0, 0, 0, S - 1, 0, 0])            # [1,H,S,W*C]
    xf = ops.reshape(xf, [1, H * S, W * C])
    xf = ops.slice_tensor(xf, [slice(None), slice(0, H_zi), slice(None)])
    x_h = ops.reshape(xf, [1, H_zi, W, C])
    # W-axis interleave (H_zi as batch axis)
    xr = ops.reshape(x_h, [H_zi, W, 1, C])
    xr = ops.pad(xr, [0, 0, 0, 0, 0, S - 1, 0, 0])            # [H_zi,W,S,C]
    xr = ops.reshape(xr, [H_zi, W * S, C])
    xr = ops.slice_tensor(xr, [slice(None), slice(0, W_zi), slice(None)])
    x_zi = ops.reshape(xr, [1, H_zi, W_zi, C])
    pad = Kh - 1 - 1  # K-1-P; for K=3,P=1 -> 1
    return ops.conv2d(
        x_zi, ops.constant(w_max, device=device_ref),
        stride=(1, 1), dilation=(1, 1),
        padding=(pad, pad, pad, pad), bias=_bias(b_np, device_ref),
    )
```

- [ ] **Step 4: Run the tests, verify they pass**

Run SYNC, then:
```bash
ssh visage@visage-spark 'export PATH=$HOME/.pixi/bin:$PATH; cd ~/repos/mojo-audio && pixi run pytest tests/test_conv.py -v'
```
Expected: 3 PASS. If a transpose test shows a shape mismatch, reconcile against the probe report's validated padding (the report ran these exact shapes green).

- [ ] **Step 5: Commit**

```bash
git add src/models/_conv.py tests/test_conv.py
git commit -m "feat(conv): native express-as-plain conv primitives (dilation=expansion, transpose=interleave)"
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

- [ ] **Step 2: Rewrite `conv1d` to delegate**

In `src/models/_hifigan_graph.py`, replace the body of `conv1d(x, w_np, b_np, dilation=1, device_ref=None)` (lines 151–end of function) with the delegation below. The native `_conv.conv1d` handles dilation internally via kernel expansion, so no dilation branch is needed:
```python
def conv1d(x, w_np, b_np, dilation=1, device_ref=None):
    """Native Conv1d via express-as-plain (was im2col). Signature unchanged."""
    from ._conv import conv1d as _native_conv1d
    return _native_conv1d(x, w_np, b_np, dilation=dilation, device_ref=device_ref)
```
Once `conv1d` and `conv_transpose_1d` delegate, delete the now-dead im2col helpers in this file (`_dilate_kernel` if duplicated, the old im2col body, and the `conv_transpose_1d` zero-interleave/im2col body) — express-as-plain leaves no fallback path. Keep `_pt_..._to_max`-style weight helpers only if still referenced.

- [ ] **Step 3: Rewrite `conv_transpose_1d` to delegate**

Replace the body of `conv_transpose_1d(x, w_pt, b_np, *, stride, device_ref)` (line 36) with:
```python
def conv_transpose_1d(x, w_pt, b_np, *, stride, device_ref):
    """Native ConvTranspose1d via zero-interleave + plain conv2d (was im2col). Signature unchanged."""
    from ._conv import conv_transpose1d
    return conv_transpose1d(x, w_pt, b_np, stride=stride, device_ref=device_ref)
```

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
Leave the surrounding BCT↔NHWC layout conversion untouched. The native `_conv.conv1d` handles dilation via kernel expansion, so the local im2col `conv1d` / `_conv1d_numpy` helpers in this file become dead — delete them after the delegation works.

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
