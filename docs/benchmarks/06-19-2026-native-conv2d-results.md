# Native conv2d Rewrite: GPU Benchmark Results

**Date:** 2026-06-25
**Branch:** feat/native-conv2d
**Commit:** 6702ae1 (perf(vits): native ops.conv2d for enc_p/flow convs)
**Host:** visage-spark — NVIDIA GB10 (Grace Blackwell), CUDA 13.0, aarch64
**MAX version:** 26.4.0.dev2026061006 / Mojo 1.0.0b2
**Benchmark script:** `scripts/benchmark_suite.py`
**Result file:** `benchmarks/results/20260625T114052.json`

---

## Summary

The express-as-plain rewrite is **numerically correct** but caused a **7.8x
performance regression** vs the im2col baseline. The switch to native
`ops.conv2d` did not unlock cuDNN for dilated and transpose convolutions — those
remained in the zero-interleave + reshape path — and those helper chains are
dramatically slower end-to-end than the original im2col matmul approach.

**Target (Applio): RTF ≤ 0.15. Not met. Regression from 0.42 to 3.28.**

---

## Per-Model RTF: Before vs After

| Model | im2col MAX 26.3 RTF | native MAX 26.4 RTF | Δ |
|-------|---------------------|---------------------|---|
| melodic-male-singer-1 | 0.377 | 3.101 | **+8.2x slower** |
| falsetto-male-soul-singer-1 | 0.469 | 3.462 | **+7.4x slower** |
| **Mean** | **0.423** | **3.282** | **+7.8x slower** |

For reference: im2col on MAX 26.4 was ~0.52 (from radar context). The native
rewrite is ~6.3x slower than the 26.4 im2col baseline and ~22x slower than
Applio (0.15).

---

## Per-Stage Breakdown (3.0 s audio clip)

### melodic-male-singer-1

| Stage | im2col 26.3 (s) | native 26.4 (s) | Δ |
|-------|-----------------|-----------------|---|
| HuBERT / ContentVec | 0.528 | 0.347 | -34% (faster) |
| RMVPE (F0) | 0.069 | 3.670 | **+53x slower** |
| VITS synth | 0.504 | 5.285 | **+10x slower** |
| **Total inference** | **1.101** | **9.302** | **+8.5x slower** |
| **RTF** | **0.367** | **3.101** | — |

### falsetto-male-soul-singer-1

| Stage | im2col 26.3 (s) | native 26.4 (s) | Δ |
|-------|-----------------|-----------------|---|
| HuBERT / ContentVec | 0.529 | 0.349 | -34% (faster) |
| RMVPE (F0) | 0.101 | 3.655 | **+36x slower** |
| VITS synth | 0.592 | 6.380 | **+11x slower** |
| **Total inference** | **1.225** | **10.385** | **+8.5x slower** |
| **RTF** | **0.408** | **3.462** | — |

HuBERT improved (it was already on native matmul). RMVPE and VITS regressed
severely. These are precisely the models that received the express-as-plain
migration in Tasks 3–5.

---

## What Was Migrated (express-as-plain summary)

All tasks ran green on the test suite — correctness is not the issue.

| Conv variant | Strategy | Used in | Test result |
|---|---|---|---|
| Plain conv2d (d=1, any stride/groups) | Native `ops.conv2d` directly | RMVPE residual blocks, VITS enc_p/flow, HiFiGAN plain | PASS |
| Dilated conv (d>1) | Kernel expansion: zeros inserted between taps, then plain conv2d | HiFiGAN ResBlocks (d=1,3,5) | PASS (max_diff ~3.8e-05) |
| ConvTranspose1d (S=2) | Zero-interleave: squeeze→unsqueeze→pad→reshape chain, then plain conv2d | HiFiGAN decoder upsampling | PASS (max_diff ~2.3e-05) |
| ConvTranspose2d (S=2) | 2D zero-interleave: 14-op squeeze/unsqueeze/pad/reshape/transpose chain, then plain conv2d | RMVPE decoder upsampling | PASS (max_diff ~1.1e-05) |

**Root cause of regression:** the zero-interleave chains for ConvTranspose1d
and ConvTranspose2d, and the kernel-expansion for dilated conv, add 10–14 MAX
graph ops per conv call. These ops are not fused by the JIT. Each op is a
separate GPU kernel dispatch with full synchronization overhead. The im2col
approach (matmul after explicit column extraction) was faster because it
collapsed the same computation into fewer large matrix multiplications.

The probe microbenchmark (Task 1b) showed 1.8x speedup for a single
HiFiGAN-scale dilated conv in isolation. In the full graph, the overhead of
many sequential small ops dominates and reverses the advantage.

**ops.conv2d_transpose** is still broken in MAX 26.4 on aarch64 (cuDNN
ALLOC_FAILED on layout mismatch — see task-1-report.md). The express-as-plain
strategy was the only available path.

---

## Quality vs Applio

**BLOCKED**: `scripts/compare_vs_applio.py` requires `torchaudio` (via Applio's
`rvc/lib/algorithm/generators/refinegan.py`). Torchaudio is not installed in
the pixi default env and Applio's `.venv` is absent on visage-spark. The pixi
env only has `torch 2.10.0` (no torchaudio).

**Quality inference from test suite**: All model unit tests pass (`test-models`,
`test-vits`, `test-hifigan`, `test-pitch`). Numerical correctness for each
conv variant was verified by the probe (Task 1b max_diff < 1e-4 for all). The
rewrite is quality-neutral — no degradation expected vs im2col baseline.

Prior quality context from audit (04-11 handoff): RMVPE U-Net output corr ≈
0.98 vs PyTorch on real mel input; individual layer corr = 1.000000.

---

## Applio Target: RTF ≤ 0.15

**Not met.** Not close.

| Engine | Mean RTF | vs Applio |
|--------|----------|-----------|
| Applio (PyTorch CUDA) | 0.152 | 1.0x (target) |
| mojo-audio im2col MAX 26.3 | 0.423 | 2.8x slower |
| mojo-audio native MAX 26.4 | 3.282 | 21.6x slower |

The express-as-plain approach is counterproductive for GPU performance. **The
im2col baseline is the better performing implementation** and should be reverted
for production development.

---

## Diagnosis and Next Steps

The express-as-plain approach hit a fundamental wall: MAX 26.4 does not fuse
chains of reshape/pad/transpose ops, and the overhead of dispatching each op as
an independent GPU kernel far exceeds the benefit of eventually calling a native
`ops.conv2d`. The approach that would actually unlock cuDNN's dilated and
transposed conv kernels — native `ops.conv2d` with `dilation>1` and
`ops.conv2d_transpose` — is blocked by MAX 26.4 GPU bugs (region_7 kernel crash
for dilated; cuDNN ALLOC_FAILED for transpose).

**Recommended path forward:**

1. **Revert feat/native-conv2d to im2col** (or continue on main with im2col).
   The test suite is green on main; this branch's regression makes it
   unsuitable for merge.

2. **Track MAX nightly** for when `ops.conv2d` with `dilation>1` and
   `ops.conv2d_transpose` are fixed on aarch64. These are the real unlock.
   Reference: MAX bugs #6129 (groups conv), #6248 (conv2d C_in<8).

3. **File a MAX bug** for the `conv2d_transpose` cuDNN ALLOC_FAILED on
   aarch64 (not yet filed as of this session). The task-1-report.md has the
   full reproducer.

4. **Alternative**: investigate whether MAX's `ops.conv1d` (if available) or a
   pure-Mojo custom kernel could match cuDNN performance for the dilated/
   transpose cases without the graph-op-chain overhead.

---

## Benchmark Environment

```
Commit:    6702ae1 (feat/native-conv2d)
MAX:       26.4.0.dev2026061006
Host:      visage-spark (aarch64)
Audio:     0_0_0.wav (3.0 s, 16 kHz)
Models:    2 (both OK, 0 failed)
Run type:  mojo-only, gpu
```

### Full Benchmark Output

```
============================================================
  mojo-audio Benchmark Suite
============================================================
  Models:    2
  Audio:     0_0_0.wav (3.0s)
  Run type:  mojo-only
  Device:    mojo=gpu  applio=auto
  Commit:    6702ae1 (feat/native-conv2d)
  MAX:       26.4.0.dev2026061006
  Host:      visage-spark (aarch64)
============================================================

[1/2] melodic-male-singer-1...
 OK (301.6s)  RTF=3.101
[2/2] falsetto-male-soul-singer-1...
 OK (188.6s)  RTF=3.462

============================================================
  Summary
============================================================
  Models OK / total:  2 / 2
  Mean RTF (mojo):    3.282
  RTF range:          3.101 - 3.462
  Results saved to:   benchmarks/results/20260625T114052.json
============================================================
```
