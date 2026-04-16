# GPU Recon Failure Matrix — 2026-04-16

**Operator:** Claude (Opus 4.6) + Chris
**Hardware:** visage-spark, NVIDIA GB10
**MAX:** `26.3.0.dev2026032005` (Spark production)
**Repo state:** `0ab1fdd` on `main`
**Recon script:** `scripts/gpu_recon_2026_04_16.py` (committed)
**Spark log:** `/tmp/gpu_recon.log`, `/tmp/gpu_recon_results.json`

> Forced `device="gpu"` on each pipeline stage in isolation. Captured compile, run,
> output shape, and error class. This is the campaign map.

---

## TL;DR

| # | Stage | Compile | Run | Compile (s) | Run (s) | Notes |
|---|---|---|---|---|---|---|
| 1 | AudioEncoder | ✅ | ✅ | 64.85 | 0.477 | **Works.** RTF 0.24 on 2s @ 16kHz. |
| 2 | PitchExtractor | ✅ | ✅ | 93.09 | 0.799 | **FIXED 2026-04-16** — see §6. RTF 0.40, perfect numerical match vs CPU. |
| 3 | HiFiGAN | ✅ | ✅ | 143.08 | 4.307 | Works but **slower than CPU** — RTF 2.15. Perf problem. |
| 4 | VITS (enc_p + flow) | ✅ | ❌ | 240.86 | — | Compiles. Runtime fails on **input tensor placement** — fixable in our code. |
| 5 | Full VoiceConverter | ⏳ | — | — | — | PitchExtractor cascade unblocked; still hits VITS placement bug at runtime. |

**Net (post-fix):** 3 of 5 stages run cleanly on GPU, 1 needs the placement fix we own,
1 still needs perf work. **No outstanding MAX Engine compiler bugs blocking us.**

---

## 1. AudioEncoder — ✅ works on GPU

```json
{
  "compile_time_s": 64.85,
  "run_time_s": 0.477,
  "output_shape": [1, 99, 768],
  "output_stats": {"min": -3.55, "max": 4.82, "mean": 0.001, "std": 0.405}
}
```

- **RTF 0.24** on a 2s clip — already a meaningful speedup vs the 0.63 CPU baseline that the whole pipeline was at.
- **Caveat:** the compile log shows the `bmm.mojo` KGEN errors (same family as #2 below), but the engine recovers via a fallback path and ships a working binary. The errors are visible noise on stderr — not fatal here, but they indicate the same underlying compiler issue is present in this graph too. If MAX ever stops falling back, this stage will regress.

---

## 2. PitchExtractor — ❌ MAX Engine compiler bug

```
RuntimeError: Failed to compile the model.
…
max/kernels/src/Mogg/MOGGKernelAPI/MOGGKernelAPI.mojo:3917:60:
  error: rebind input type '!pop.array<3, scalar<si64>>'
         does not match result type '!pop.array<4, scalar<si64>>'
-:1:1: error: The graph compiler could not elaborate the generated KGEN
```

**The error chain:**
```
bmm.mojo:725 (instantiation failed)
  → _multistage_gemm_gpu.mojo:691 (function instantiation failed)
    → _multistage_gemm_gpu.mojo:946 (call expansion failed: width=2, alignment=8)
      → bmm.mojo:774 (rank=4, _width=2, _rank=3)
        → MOGGKernelAPI.mojo:3911 (rebind rank 3 → rank 4 mismatch)
```

**Reading:** the compiler is trying to dispatch a batched matmul to the multistage GEMM
GPU kernel, the kernel asks for a tensor with one rank, and the rebind it inserts is
producing a tensor with a different rank. The same kernel is invoked for AudioEncoder
(stage 1), but in that graph the MAX engine has another viable code path and falls
back. For PitchExtractor's specific graph shape, the fallback is exhausted.

**This is almost certainly the same family as the unfiled "ops.rebind overwrites static dimensions"
bug listed in `docs/project/03-06-2026-roadmap.md:244`** — surfacing again, this time as a
GPU compile failure rather than a CPU silent overwrite.

**Where to dig:**
- `src/models/_rmvpe.py` uses `ops.rebind` in `_conv2d` and after `out_cnn` (added in the PAD_T fix).
- The bmm path implies an im2col + matmul somewhere is producing a rank-3 result that gets rebound to rank-4 (or vice versa) — almost certainly one of the `_conv2d` calls.
- Bisection target: build a stripped-down RMVPE graph with just the encoder, see if it compiles. If yes, add bottleneck. If yes, decoder. Pinpoint the offending layer.

**Workaround likely takes the form:** explicit `ops.reshape` to the target rank before the matmul, or explicit shape annotation that prevents the engine from inserting the bad rebind.

---

## 3. HiFiGAN — ✅ runs, but slower than CPU

```json
{
  "compile_time_s": 143.08,
  "run_time_s": 4.307,
  "output_shape": [1, 96000]   // 2s @ 48 kHz — correct
}
```

- Output is finite, shape is right, it does the work.
- **RTF 2.15 on GPU** — for context, the whole CPU pipeline runs at RTF 0.63. This stage *alone* on GPU is taking 3.4× longer than the entire CPU pipeline takes.
- **Hypothesis:** the im2col + matmul workaround for `ops.conv2d` (Sprint 3) creates massive intermediate tensors. On CPU that's fine — cache-friendly. On GPU, those intermediates don't fit cleanly in shared memory and you pay HBM round-trips for every conv. This was the predicted GPU cost of the Sprint 3 workarounds.
- **Mitigation sequence:**
  1. Profile to confirm im2col is the bottleneck (vs e.g. zero-interleave conv_transpose).
  2. Try native `ops.conv2d` per-layer on GPU (the C_in≥8 bug was x86 only — aarch64 is clean per the 04-11 audit). May be a one-line swap to reclaim native perf.
  3. If GPU conv2d is slow too, explore `ops.conv2d` with grouped/batched dispatch tuning.

---

## 4. VITS enc_p + flow — ✅ compiles, ❌ tensor placement bug in our code

```
TypeError: expected argument 0 to be on device Device(type=gpu,id=0),
           but was on device Device(type=cpu,id=0)
  at voice_converter.py:434  enc_result = self._enc_p_model.execute(*inputs)
```

**This is our bug, not MAX's.** When the graph is compiled for GPU, `model.execute()`
expects GPU tensors. `convert_from_features()` passes raw numpy arrays. MAX 26.3 enforces
the placement strictly now.

**Fix shape:**
```python
# in voice_converter.py near the execute() calls
def _to_device(self, arr: np.ndarray):
    from max.driver import Tensor
    return Tensor.from_numpy(arr).to(self._device)

# then
inputs = [self._to_device(features), self._to_device(pitch_i32), self._to_device(lengths)]
for i in range(...):
    inputs.append(self._to_device(biases_k[i]))
    inputs.append(self._to_device(biases_v[i]))
```

Same treatment for the `flow_model.execute(z_p, enc_mask)` call. AudioEncoder, PitchExtractor,
and HiFiGAN already handle this internally — VoiceConverter is the orchestrator that punts.

**Estimated effort:** 30 minutes. Probably enables flow_model on GPU too once landed.

---

## 5. Full VoiceConverter — ❌ cascades from #2

`from_pretrained` calls `PitchExtractor.from_pretrained` internally → same KGEN failure.
Once #2 is solved this stage should at least reach runtime; will then hit the same
tensor-placement bug as #4.

---

## Strategic read

**What we feared:** widespread compile failures across the board. **What we found:** one
real MAX bug, one self-inflicted code bug, one perf optimization problem.

**Recommended sequence:**

1. **Land the VoiceConverter tensor-placement fix** (#4). 30 min. Unblocks VITS GPU and
   sets up the orchestrator for the day PitchExtractor lands.
2. **Bisect the PitchExtractor KGEN failure** (#2). Build a layer-by-layer probe graph,
   find the offending op, add a shape-annotation workaround. Concurrently file a minimal
   repro — this almost certainly closes the still-unfiled `ops.rebind` bug from the
   roadmap. ETA: 1–2 days, depending on how cleanly the layer bisects.
3. **HiFiGAN GPU perf** (#3). After #1 and #2 are landed and full pipeline runs end-to-end
   on GPU, profile and try native `ops.conv2d` swap on aarch64. RTF improvement here is
   the actual demo win. ETA: probably a week of perf grind.
4. **End-to-end RTF measurement** on real Shade input, report numbers, update roadmap.

**The good news for the meeting:** AudioEncoder GPU alone is already RTF 0.24 — once
PitchExtractor compiles and HiFiGAN's perf is sorted, end-to-end RTF on Spark GPU should
land somewhere in the 0.10–0.25 range, which is the kind of "live voice conversion at
real-time speed" number the demo story needs.

---

## 6. PitchExtractor GPU fix — same-day resolution

**Branch:** `main` (working tree)
**Files:** `src/models/_rmvpe.py`, `tests/test_pitch_extractor.py`, `scripts/gpu_pitch_probe_2026_04_16.py`

### Diagnosis

Comparing the failing RMVPE `_conv2d` to HiFiGAN's working im2col exposed
the structural difference:

```
HiFiGAN  (works on GPU):  ops.matmul([1, T, K·C_in],     [K·C_in, C_out])  — rank-3 × rank-2
RMVPE    (fails on GPU):  ops.matmul([1, H, W, K²·C_in], [K²·C_in, C_out]) — rank-4 × rank-2
```

The rank-4 × rank-2 form routes through `_multistage_gemm_gpu`, which
inserts a rank-3↔rank-4 rebind that doesn't reconcile and trips KGEN.
HiFiGAN's rank-3 form skips that path entirely.

Notably: **isolated `_conv2d` calls (even with symbolic H) compile fine on
GPU.** The bug only fires inside the deeper U-Net stack — likely a
specific shape combination triggers the broken kernel selection. Bisecting
to the offending layer wasn't necessary because the structural fix
unblocks all call sites at once.

### Fix

`src/models/_rmvpe.py` `_conv2d` k×k path: collapse `[1, H, W, K²·C_in]`
to `[1, H*W, K²·C_in]` via `ops.reshape(..., [1, -1, K²·C_in])` before the
matmul, then reshape back to `[1, H, W, C_out]`. Existing `ops.rebind` for
the symbolic-dim reconciliation kept untouched. ~6-line diff.

The 1×1 path (lines 125-129) was already rank-3 × rank-2 and untouched.

### Verification

| Check | Result |
|---|---|
| New `test_unet_graph_compiles_on_gpu` (pre-fix) | RED — `RuntimeError` from KGEN bmm rebind, ~4 min compile failure |
| Same test (post-fix) | GREEN — compile in 92s |
| Full `pixi run test-pitch-extractor-full` (30 tests) | **30 passed** in 158s — zero regressions |
| `test_salience_matches_pytorch` (CPU correctness) | PASSED — corr 1.000000, max_diff 0.000000 |
| `scripts/gpu_pitch_probe_2026_04_16.py` (real RMVPE weights, GPU vs CPU) | GPU compile 93s, GPU extract 0.8s, **GPU↔CPU output diff = 0.0** |

### Perf observation

GPU extract took 0.799s on 2s of audio (RTF 0.40); CPU was 0.224s (RTF 0.11)
with a warm cache. So PitchExtractor GPU is **~3.5× slower than CPU on this
input** — same im2col-on-GPU tax HiFiGAN exhibits. Perf optimization
(switch to native `ops.conv2d` on aarch64, which the 04-11 audit verified
clean) is its own follow-up. Compile-blocking issue resolved; perf war is
where the demo win lives.

### Pending follow-ups from this fix

1. **File the MAX bug** with a minimal repro. Shape: full `build_unet_graph`
   on GPU triggers; isolated `_conv2d` does not. Worth a clean repro that
   shows what specific layer combination forces the bad kernel selection.
   Already known to persist across 26 daily nightlies (dev2026032005 →
   dev2026041520) with line-number drift but identical symptom.
2. **Apply the same im2col reshape pattern to HiFiGAN** if any rank-4
   matmul exists there too (likely the cause of HiFiGAN's GPU perf tax —
   though it currently compiles, the dispatch may still go through the
   slow path).
3. **VITS tensor-placement fix** (§4) is now the next blocker for full
   end-to-end GPU pipeline. Was estimated at 30 min; nothing in this fix
   changes that estimate.

## Artifacts

- Script: `scripts/gpu_recon_2026_04_16.py` (local + scp'd to `/tmp/` on Spark)
- Raw log: `/tmp/gpu_recon.log` on visage-spark
- Structured JSON: `/tmp/gpu_recon_results.json` on visage-spark
- Test inputs: deterministic numpy seed 42, 2s @ 16 kHz random low-amplitude noise

## What this recon did NOT cover

- Numerical correctness vs CPU (only checked shape + finiteness). Need a follow-up that
  runs the same input through CPU and GPU and diffs.
- Cold-start vs warm-cache compile times (these are cold; warm should be near-zero).
- Real audio input through the full pipeline (used random noise). Once #2 is fixed, run
  the Weeknd checkpoint on a real 30s vocal clip and listen.
- GPU memory headroom — `nvidia-smi --query-gpu=memory.used` returned `[N/A]` on Spark
  (sysfs reporting issue), so we can't currently see how much VRAM the compiled graphs
  consume. Worth solving as a side quest.
