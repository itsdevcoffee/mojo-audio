# Design: im2col → native `ops.conv2d` rewrite

**Date:** 2026-06-19
**Status:** Approved design — ready for implementation plan
**Goal:** Replace the im2col+matmul convolution workaround with native MAX
`ops.conv2d` / `ops.conv2d_transpose` on the DGX Spark (aarch64 GB10), closing
the 2.4x GPU perf gap vs Applio and removing the im2col numerical drift — the
first step toward making mojo-audio good enough to run Shade in production.

## Why

mojo-audio replaced every `ops.conv2d` call with an im2col+matmul workaround to
dodge MAX bug `modular/modular#6248` (conv2d wrong for C_in≥8). Two costs:

1. **Performance.** im2col thrashes GPU memory. Measured GPU RTF is ~0.42
   (26.3) vs Applio's 0.15 — **2.4x slower**. The im2col matmuls never hit
   MAX's native conv kernels (incl. 26.4's autotuned cuDNN selection).
2. **Correctness.** The 04-11 audit found ~3% RMS / ~10% relative drift
   accumulating across the ~120 stacked im2col convs in the RMVPE U-Net — the
   reason `test_salience_matches_pytorch` is xfail. Native conv2d removes this,
   improving pitch accuracy (a direct voice-conversion quality input).

The blocker is gone on aarch64: the 04-11 audit verified native `ops.conv2d`
correct on the Spark (C_in≥8, K=3, stride=1, max_diff <1e-7), and AudioEncoder
already runs native strided + grouped (groups=16, #6129) conv2d in production.

## Scope

**In:** swap im2col → native conv ops in RMVPE, HiFiGAN, VITS. Spark/aarch64
only. Target MAX 26.4.

**Out (separate specs):** RMS volume normalization, FAISS index retrieval, RVC
v1 (256-ch) support, batch>1. These are the *quality/compat* track; this spec
is the *perf+conv-correctness* track.

**Platform:** Spark (aarch64) is the only target. All development, tests, and
benchmarks run on the Spark via `ssh visage@visage-spark`, repo at
`~/repos/mojo-audio`, pixi at `~/.pixi/bin`. The Fedora x64 box is explicitly
disregarded — do not build, test, or benchmark there (conv2d is still broken on
x64 per #6248, and it is not a target).

## Architecture

One shared module `src/models/_conv.py` replaces the per-file im2col helpers:

| Helper | Wraps | Replaces |
|---|---|---|
| `conv1d(x, w_pt, b, dilation, groups)` | `ops.conv2d`, kernel `(K,1)`, `dilation=(d,1)`, same-pad `(d·(K-1)//2, …, 0,0)` | `_hifigan_graph.conv1d`, `_vits_graph._conv1d_bct` |
| `conv2d(x, w_max, b, stride, padding, groups)` | `ops.conv2d` directly | `_rmvpe._conv2d` |
| `conv_transpose1d(x, w_pt, b, stride, padding)` | `ops.conv2d_transpose` | `_hifigan_graph.conv_transpose_1d` |
| `conv_transpose2d(x, w_pt, b, stride, padding, output_padding)` | `ops.conv2d_transpose` | `_rmvpe._conv_transpose_2x` |

**Weight layout.** PyTorch Conv1d `[C_out, C_in, K]` → MAX RSCF
`[K, 1, C_in, C_out]` via `np.transpose(w[...,None], (2,3,1,0))`. AudioEncoder
already performs this conversion; lift it into `_conv.py` as the single tested
implementation. Input stays NHWC (the helpers already produce `[B,T,1,C_in]` /
`[1,H,W,C_in]`).

**No platform branching.** Native ops are used unconditionally. im2col code is
retained *only per-variant* if that variant fails the probe (§Probe). If all
variants pass, the im2col helpers are deleted entirely.

## The probe (run first, on the Spark)

`scripts/probe_native_conv.py` checks native ops vs PyTorch ground truth for
exactly the variants the pipeline uses. Threshold: `max_diff < 1e-5`.

| Variant | Config | Used by | Prior status |
|---|---|---|---|
| plain | k×k, stride 1, groups 1, C_in 8→192 | RMVPE, VITS, HiFiGAN | ✅ verified 04-11 |
| strided | stride (s,1) | (AudioEncoder) | ✅ in prod |
| grouped | groups=16, K=128 | (AudioEncoder) | ✅ in prod (#6129) |
| **dilated** | dilation (1,1)/(3,1)/(5,1) | HiFiGAN/VITS WaveNet | ❓ unverified |
| **convT1d** | stride ∈ upsample rates, pad=(K-S)//2 | HiFiGAN upsampler | ❓ unverified |
| **convT2d** | stride 2, K3, pad 1, output_pad 1 | RMVPE decoder | ❓ unverified |

The probe also pins down the exact `ops.conv2d_transpose` parameter mapping
(stride / padding / output_padding semantics) against PyTorch, since that is the
riskiest translation.

**On failure:** the failing variant keeps its im2col helper, a MAX bug is filed
(reference #6248), and the rewrite proceeds for all passing variants. Partial
wins are shipped — the whole rewrite is not blocked on one variant.

## Migration order — file-by-file, test-gated on the Spark

After each file, run the full suite on the Spark; revert that file if it
regresses (128 pass, 2 xfail baseline).

1. **RMVPE** — `_conv2d`, `_conv_transpose_2x`. Highest value: removes the
   ~120-conv accumulation drift, so `test_salience_matches_pytorch` is expected
   to flip **xfail → pass**. That flip is the primary correctness signal.
2. **HiFiGAN** — `conv1d` (dilated + plain), `conv_transpose_1d`. The vocoder;
   expected runtime hotspot, so the biggest single RTF contributor.
3. **VITS** — `_conv1d_bct` (mostly dilation=1). Straightforward; finishes the
   single-conv-path consolidation.

## Success criteria

- **Perf:** re-run `scripts/benchmark_suite.py run --mojo-only --mojo-device
  gpu` on the Spark, same models/audio as the controlled baseline (melodic-male-
  singer-1, falsetto-male-soul-singer-1). Target mean RTF **≤0.15**. Report the
  measured number regardless of whether it hits target.
- **Correctness:** full suite stays green; RMVPE salience xfail ideally → pass.
- **Quality:** `scripts/compare_vs_applio.py` waveform / mel / F0 correlation
  holds or improves vs the im2col baseline.

## Risks & mitigations

| Risk | Mitigation |
|---|---|
| Dilated or transpose conv2d numerically wrong on aarch64 | Probe catches it before any swap; keep im2col for that variant + file bug |
| ConvTranspose param mapping mismatch (pad/output_pad) | Probe validates exact mapping vs PyTorch before use |
| A swap regresses the test suite | File-by-file, test-gated on Spark; revert the offending file |
| 26.4 changes break the pipeline | Pipeline already confirmed running on 26.4 (06-10); keep torchvision stashed at `/tmp/tv-stash` so the Applio harness imports |
| Perf gain smaller than hoped | Still ship: correctness (drift) win stands; informs whether Spec 2 (quality) alone can justify the Shade switch |

## References

- Backlog radar Priority 2 #1: `docs/project/04-17-2026-backlog-radar.md`
- conv2d aarch64 verification: `docs/handoff/04-11-2026-audit-results.md` §4
- MAX 26.4 evaluation (no im2col perf win; native rewrite is the unlock): memory `project_max_264_evaluation`
- MAX bugs: #6129 (grouped, closed/fixed), #6248 (C_in≥8 x64, open)
