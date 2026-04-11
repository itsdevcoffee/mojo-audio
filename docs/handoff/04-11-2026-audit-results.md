# Audit Results — 04-11-2026

**Auditor:** Claude (Opus 4.6) — independent review session
**Scope:** PitchExtractor (RMVPE) fixes + MAX Engine `ops.conv2d` retest
**Status of repo at audit time:** `1d97a36` on `main` (local), `3d9b1ef` on Spark
**MAX version under test:** `26.3.0.dev2026041020 (83c8cb24)` in isolated repro envs
**Production MAX:** `26.3.0.dev2026032005` (unchanged)

> Short version: the 7 architectural RMVPE fixes are all correct and the
> conv2d retest is accurate, BUT the handoff's headline claim of "U-Net matches
> PyTorch at correlation 1.0, max diff 2.6e-5" is **not reproducible** on either
> random noise or real mel input. Actual U-Net output is corr ≈ 0.98, max_diff
> ≈ 2.6. This doesn't break the pipeline, but the debugging attribution for the
> remaining xfail gap is wrong — the gap is in the U-Net, not the numpy BiGRU.
>
> **Follow-up (same day): root cause of the ~0.98 drift found and fixed. See
> §10 for the investigation and §11 for the fix. After the fix, the U-Net
> matches PyTorch to float32 precision at every layer (corr = 1.000000,
> max_diff ~1e-5 — genuine float32 noise now, not a misquoted metric).**

---

## TL;DR verdict

| Finding | Verdict |
|---|---|
| Fix 1: ReLU placement in residual block | ✅ correct — matches `ConvBlockRes` line-for-line |
| Fix 2: Decoder upsample ReLU | ✅ correct — matches `ResDecoderBlock.conv1` sequential |
| Fix 3: Flatten order (NHWC→NCHW) | ✅ correct — produces same (channel, freq) layout as PyTorch |
| Fix 4: GRU gate ordering `[r \| z \| n]` | ✅ verified bit-exact vs real PyTorch `nn.GRU` |
| Fix 5: GRU update `(1-z)*n + z*h` | ✅ verified bit-exact vs real PyTorch `nn.GRU` |
| Fix 6: Bins-to-Hz `cents = bin*20 + 1997.3794…` | ✅ correct — matches Applio `RMVPE0Predictor` |
| Fix 7: im2col + matmul `_conv2d` | ✅ correct — no `ops.conv2d` calls; structure verified |
| Test suite regressions | ✅ none — 128 pass, 2 expected xfails (PitchExtractor correctness + HiFiGAN batch>1) |
| conv2d C_in≥8 bug on x64 (still broken) | ✅ reproduced, byte-identical to claimed values |
| conv2d bug on aarch64 (fixed) | ✅ reproduced, max_diff < 1e-7 for all cases |
| Multi-stride conv2d crash (fixed on both) | ✅ reproduced |
| K=7 worst-case: 0.191 on x64 | ✅ reproduced |
| **"U-Net matches PyTorch at correlation 1.0, max_diff 2.6e-5"** | ❌ **not reproducible** — see §3 |
| **"xfail gap is float32 accumulation in numpy BiGRU"** | ❌ **wrong attribution** — BiGRU is bit-exact; gap is in U-Net |
| Mel divergence is full cause of F0 cent error | ❌ no — mel is now fixed (3d9b1ef), residual error is in U-Net drift |

---

## 1. PitchExtractor fixes — theory-level review

Cross-referenced each fix against `/home/visage/repos/Applio/rvc/lib/predictors/RMVPE.py` (PyTorch ground truth) and `src/models/_rmvpe.py` (current state).

### Fix 1 — ReLU placement in `_residual_block`

PyTorch `ConvBlockRes.conv` is `Sequential(Conv2d, BN, ReLU, Conv2d, BN, ReLU)` (RMVPE.py:25-46). `forward` returns `self.conv(x) + [self.shortcut(x) | x]` — **no final ReLU on the sum**.

Current `_residual_block` in `_rmvpe.py:235-265`:
```
h = conv1 → BN1 → relu → conv2 → BN2 → relu
sc = shortcut(x) or x
return h + sc         # no final relu
```

**Verdict: correct.**

**Minor doc bug:** the module-level docstring at `_rmvpe.py:37-42` still shows the OLD (incorrect) residual-block structure:
```
h = relu(BN1(conv1(x)))
h = BN2(conv2(h))
...
output = relu(h + sc)
```
This is stale and contradicts the actual code. Should be updated to match the inline docstring at line 236-242.

### Fix 2 — Missing ReLU after decoder upsample BN

PyTorch `ResDecoderBlock.conv1 = Sequential(ConvTranspose2d, BN, ReLU)` (RMVPE.py:191-203). `forward` applies `conv1(x)` then `torch.cat((x, concat_tensor), dim=1)`.

Current `build_unet_graph` at `_rmvpe.py:316-345`:
```python
x = _conv_transpose_2x(x, up_w_pt, up_b, device_ref)
if f"dec.{L}.up.scale" in weights:
    x = _bn_add(x, ...)
x = ops.relu(x)                 # ← added by Fix 2
skip = ops.slice_tensor(...)
x = ops.concat([x, skip], axis=3)
```

**Verdict: correct.**

**Minor concern:** `ops.relu(x)` is applied unconditionally, but the BN step is gated on `"dec.{L}.up.scale" in weights`. In practice all 5 decoder levels have BN weights (invariant of RMVPE), so both branches always run together — but structurally the ReLU-without-BN path would be wrong if BN were ever missing. Benign today.

### Fix 3 — Flatten order

PyTorch E2E.forward (RMVPE.py:337): `self.cnn(self.unet(mel)).transpose(1, 2).flatten(-2)`.
- `self.cnn` outputs NCHW `[B, 3, T, 128]`
- `transpose(1, 2)` → `[B, T, 3, 128]`
- `flatten(-2)` → `[B, T, 384]`, row-major so order is `(c=0,f=0), (c=0,f=1), …, (c=0,f=127), (c=1,f=0), …`

Current `build_unet_graph` at `_rmvpe.py:365-371`:
```python
x_sq = ops.squeeze(x, 0)          # [T, 128, 3]       (was NHWC)
x_t = ops.transpose(x_sq, 1, 2)   # [T, 3, 128]       ← same layout as PyTorch
x = ops.unsqueeze(x_t, 0)         # [1, T, 3, 128]
x = ops.reshape(x, [1, Dim("T"), 384])
```

After the transpose, the last-two-dims are `(channel, freq)` identical to PyTorch. C-contiguous reshape flattens the same way. **Verdict: correct.**

### Fix 4 — GRU gate ordering and Fix 5 — GRU update formula

PyTorch `nn.GRU` weight layout: `[W_ir | W_iz | W_in]` (reset, update, new). Update rule: `h_t = (1 - z_t) ⊙ n_t + z_t ⊙ h_{t-1}`.

Current `bigru_forward` at `_rmvpe.py:386-440`:
```python
r = _sigmoid(gi[:, :H]    + gh[:, :H])
z = _sigmoid(gi[:, H:2*H] + gh[:, H:2*H])
n = np.tanh(gi[:, 2*H:]   + r * gh[:, 2*H:])
return ((1.0 - z) * n + z * h).astype(np.float32)
```

**Verified by standalone test** (`/tmp/audit_gru_test.py`, ran on Spark with real rmvpe.pt GRU weights):

```
T=   1  max_diff=8.05e-07  mean_diff=9.69e-08  corr=1.00000000
T=   4  max_diff=1.49e-06  mean_diff=1.44e-07  corr=1.00000000
T=  16  max_diff=3.40e-06  mean_diff=1.68e-07  corr=1.00000000
T=  64  max_diff=3.34e-06  mean_diff=1.74e-07  corr=1.00000000
```

This is float32 precision noise over 64 timesteps. The numpy BiGRU is effectively bit-exact with PyTorch `nn.GRU`. **Both Fix 4 and Fix 5 are correct.**

**Important corollary:** this invalidates the handoff's explanation for the xfail correlation gap. The handoff says the 0.022 gap is "float32 accumulation through numpy BiGRU over 64 timesteps". That's wrong — the BiGRU contributes ~3e-6, not ~0.022. The actual gap is in the U-Net (§3).

### Fix 6 — Bins-to-Hz

Applio uses `cents_mapping = 20 * np.arange(N_CLASS) + 1997.3794084376191` and `Hz = 10 * (2 ** (cents / 1200))` (RMVPE.py:442, 493).

Current `_bins_to_hz` at `_rmvpe.py:462-465`:
```python
cents = bin_indices.astype(np.float64) * 20.0 + 1997.3794084376191
return (10.0 * (2.0 ** (cents / 1200.0))).astype(np.float32)
```

Bin 0 → 31.72 Hz. Matches Applio's pitch floor. **Verdict: correct.**

On the "where does 1997.3794… come from" question: it's just `1200 * log2(31.70/10)`, i.e. the cents offset that puts bin 0 at ~31.7 Hz (a sensible low-F0 floor for speech, around B0). Chosen by the RMVPE model authors as the training grid lower edge. Not suspicious.

### Fix 7 — im2col replacement for `ops.conv2d`

`grep ops\.conv2d src/models/_rmvpe.py` finds **zero** calls — all mentions are in comments/docstrings. Structure of `_conv2d` at `_rmvpe.py:98-172`:
- 1×1 path: reshape weight to `[C_in, C_out]` → matmul → add bias. Mathematically equivalent to 1×1 conv.
- k×k path: pad, then for each `(kh, kw)` slice `[:, kh:kh+H, kw:kw+W, :]`, concat `kH*kW` slices along channel axis → `[1, H, W, kH*kW*C_in]`, then matmul with `weight.reshape(kH*kW*C_in, C_out)` → `[1, H, W, C_out]`, rebind, add bias.

This is the standard im2col decomposition of a conv. **Verdict: correct in principle.** The `test_conv_transpose_numerically_equivalent` test passes in the full suite, and the test-pitch-extractor run below confirms the graph produces correct shape/non-NaN output.

**Caveat discovered in §3:** while each individual `_conv2d` call is mathematically correct, I observed ~10% relative drift on the full U-Net output vs PyTorch, which likely comes from accumulation of float32 rounding across ~120 stacked conv calls through im2col+matmul. This is different from the claimed "matches perfectly".

---

## 2. Test suite results (Spark, caches cleared)

All run on `visage-spark` at commit `3d9b1ef`, after clearing `~/.cache/modular/.max_cache`, `.pixi/envs/default/share/max/.max_cache`, and `__pycache__`.

| Suite | Result | Notes |
|---|---|---|
| `test-pitch-extractor-full` | **28 passed, 1 xfailed** | xfail on `test_salience_matches_pytorch` as expected. `test_mel_matches_applio` **PASSED** with max_diff 2e-6 and correlation 1.0 — mel fix in 3d9b1ef landed. |
| `test-models-full` | **35 passed** | AudioEncoder numerical match max_diff 7.3e-5 |
| `test-hifigan-full` | **15 passed, 1 xfailed** | xfail on batch>1 as expected |
| `test-vits-full` | **50 passed** | flow correlation 1.0, enc_p correlation 1.0 |
| **Total** | **128 pass, 2 xfail** | matches expected count |

**No regressions introduced.** The session's fixes do not break any passing test.

---

## 3. ⚠️ The "U-Net matches perfectly" claim does NOT reproduce

This is the single biggest finding of the audit. The handoff (`docs/handoff/04-09-2026-pitch-extractor-fixes.md:57`) claims:

> U-Net graph (encoder+bottleneck+decoder+CNN) | **Perfect match** | Correlation 1.0, max diff 0.000026 vs PyTorch

And the audit context file (`04-11-2026-audit-context.md:33`) repeats:

> U-Net graph itself matches PyTorch at correlation 1.0, max diff 2.6e-5

**I was unable to reproduce either number** on Spark CPU with real RMVPE weights. My test (`/tmp/audit_unet_compare2.py`) hooks PyTorch E2E to capture `self.cnn` output (the tensor fed into the BiGRU), reconstructs the same pre-GRU layout as `max_unet_out`, and compares.

### Random noise input (`rng.randn(1, 128, 64)`)

```
PT  preflatten: min=-6.65  max=5.90  std=0.99
MAX preflatten: min=-6.54  max=5.97  std=1.01
max_diff=2.8145  mean_diff=0.0914  corr=0.98235749
```

Per-output-channel (the 3 RMVPE logits):
```
ch0: max_diff=2.81 mean_diff=0.106 corr=0.9784
ch1: max_diff=2.47 mean_diff=0.107 corr=0.9728
ch2: max_diff=1.97 mean_diff=0.062 corr=0.9782
```

Per-feature correlation stats: min 0.849, mean 0.972, median 0.980. No features are catastrophically wrong, but the whole output is uniformly noisy at the few-percent level.

### Real voice audio (Applio-style mel of a 3s voice clip from Applio's sliced_audios)

```
PT  preflatten: [-13.573, 10.501] std=0.816
MAX preflatten: [-13.573, 10.501] std=0.809
max_diff=2.598e+00  mean_diff=8.96e-02  corr=0.97688122

Final salience (post-sigmoid):
  max_diff=6.07e-02  mean_diff=4.61e-04  corr=0.99913140
  peak-bin exact agreement: 62.2%
  peak-bin within 2 bins:   67.7%
  median bin offset: 0
```

So on the exact in-distribution case the handoff implies was measured, the **U-Net pre-GRU correlation is 0.977, not 1.0, and max_diff is 2.6 (not 2.6e-5)**. The final sigmoid output correlates at 0.999 (good enough that the extracted F0 is usable), but peak-bin agreement is only ~62% on this clip.

### Where does the divergence come from?

I ruled out the BiGRU numerically:
- Feeding `pt_preflatten` → `bigru_forward` vs feeding the same into PyTorch `nn.GRU`: max_diff 5.2e-6, corr 1.0.
- Feeding `max_unet_out` → both implementations: max_diff 5.2e-6, corr 1.0.
- Standalone GRU test (above): max_diff 3.3e-6 at T=64.

So the numpy BiGRU is bit-exact. The ~10% drift is in the U-Net itself.

My best hypothesis: **accumulation error from the im2col + matmul conv path across ~120 stacked conv calls**. Each conv introduces a few ULPs of rounding noise, and stacking them through 5 encoder levels × 4 blocks × 2 convs + 16 bottleneck × 2 + 5 decoder levels × (1 transpose + 4 blocks × 2) + 1 output CNN compounds to ~1% per layer. This is consistent with the observation that per-feature correlations are all in the 0.85–0.99 range (uniform noise, no specific bad channel/layer).

**This does not break the current pipeline** — the final salience correlation is 0.999, the peak bin agreement is acceptable, and the claimed F0 figures are plausible. But:

1. **The documentation is wrong.** The U-Net does not match at 1.0 correlation. Both `04-09-2026-pitch-extractor-fixes.md:57` and `04-11-2026-audit-context.md:33` should be corrected.
2. **The xfail reason in `tests/test_pitch_extractor.py:536-541` is wrong.** It blames "baked batch-norm or im2col accumulation error in deep U-Net" (actually half right on the im2col part) and claims the U-Net matches now — it doesn't. Update to: "U-Net output diverges from PyTorch by ~3% RMS; final salience correlation only ~0.978 on random noise and ~0.999 on real mel; likely im2col accumulation across 120 stacked convs. Not fatal but not zero."
3. **If anything downstream becomes more sensitive** (e.g. retraining F0 quantizer) this drift could start to matter. Worth tracking as a known limitation.

---

## 4. MAX `ops.conv2d` retest — reproduced exactly

### Version and platform checks

Both repro envs at `Mojo 0.26.3.0.dev2026041020 (83c8cb24)` — new build hash, confirmed not `4362bfeb` from March.

- Spark: `uname -m` → `aarch64`, Ubuntu 24.04, NVIDIA DGX Spark
- Local: `uname -m` → `x86_64`, Fedora 43

### `conv2d_repro.py` — C_in sweep

**Spark (aarch64)** — all PASS with max_diff ≤ 9e-8 (float32 noise):
```
C_in=  2, C_out=  4, K=3: max_diff=0.00000000  [PASS]
C_in=  4, C_out=  8, K=3: max_diff=0.00000000  [PASS]
C_in=  6, C_out= 12, K=3: max_diff=0.00000000  [PASS]
C_in=  8, C_out= 16, K=3: max_diff=0.00000000  [PASS]
C_in= 16, C_out= 32, K=3: max_diff=0.00000000  [PASS]
C_in= 32, C_out= 64, K=3: max_diff=0.00000001  [PASS]
C_in= 64, C_out=128, K=3: max_diff=0.00000002  [PASS]
C_in=192, C_out=384, K=3: max_diff=0.00000009  [PASS]
```

**Local (x64)** — **byte-identical** to the handoff's claimed values:
```
C_in=  2, C_out=  4, K=3: max_diff=0.00000000  [PASS]
C_in=  4, C_out=  8, K=3: max_diff=0.00000000  [PASS]
C_in=  6, C_out= 12, K=3: max_diff=0.00000000  [PASS]
C_in=  8, C_out= 16, K=3: max_diff=0.01723001  [FAIL]   ← same
C_in= 16, C_out= 32, K=3: max_diff=0.02368748  [FAIL]   ← same
C_in= 32, C_out= 64, K=3: max_diff=0.03833004  [FAIL]   ← same
C_in= 64, C_out=128, K=3: max_diff=0.09013467  [FAIL]   ← same
C_in=192, C_out=384, K=3: max_diff=0.11577050  [FAIL]   ← same
```

### `conv2d_repro_k7.py` — worst case

```
Spark aarch64:
  C_in=192, C_out=512, K=7: max_diff=0.00000013  [PASS]
  C_in= 64, C_out=128, K=7: max_diff=0.00000005  [PASS]
  C_in= 32, C_out= 64, K=5: max_diff=0.00000001  [PASS]

Local x64:
  C_in=192, C_out=512, K=7: max_diff=0.19144066  [FAIL]   ← matches claim
  C_in= 64, C_out=128, K=7: max_diff=0.08989511  [FAIL]
  C_in= 32, C_out= 64, K=5: max_diff=0.05662702  [FAIL]
```

The handoff's note about the K=7 case being "slightly worse" (0.191 on current build vs 0.165 in the original #6248 filing) is accurate. Since the rng seed (42) is identical and the weight shapes match, the delta really reflects a backend numerical change — not run-to-run noise. Worth mentioning in the issue comment (see §5).

### `multi_stride_repro.py` — compiler crash

Both platforms:
```
=== Test 1: Two conv2d with different strides ===
Graph built OK, attempting compilation...
PASS: graph compiled successfully
Output shape: (1, 5, 1, 64)

=== Test 2: Two conv2d with different paddings ===
Graph built OK, attempting compilation...
PASS: graph compiled successfully
```

**Multi-stride crash is fixed on BOTH x64 and aarch64**. Nothing to flag there.

### Suspicion checklist from the handoff

1. **Repro actually uses `ops.conv2d`?** Verified: `conv2d_repro.py:37` calls `ops.conv2d(padded, …, stride=(1,1))` directly; no interception. ✅
2. **Envs at new version?** Both show `0.26.3.0.dev2026041020 (83c8cb24)`, `pixi list max` confirms `26.3.0.dev2026041020`. ✅
3. **Platforms differ?** `uname -m` → aarch64 vs x86_64. ✅
4. **CPU vs GPU?** Both repros use `DeviceRef.CPU()` and `CPU()` driver. **The bug may or may not exist on GPU — NOT TESTED.** This is correctly flagged in the handoff. I did not add GPU tests — low priority but worth doing if Chris has time, because the original #6248 was filed citing "RTX 4060 Ti" even though the filed repro was CPU-only.
5. **Float32 ground truth?** `numpy_conv2d` uses `np.float32` and `np.zeros(…, dtype=np.float32)` with scalar accumulation in Python float (which is float64, then stored as f32). No dtype gotcha — both sides effectively round to f32 at storage. ✅
6. **K=7 "slightly worse" claim?** Seed is identical; delta is a real backend numeric change. The K=7 scripts in `/tmp/max-repro*` are NOT identical to the original issue's script (a separate file was added), so there's a minor caveat — the input shapes and seed match but the exact code path differs slightly. Good enough for a "still broken" claim but not for "precisely the same regression".

### Conclusion

The platform-split finding is real and byte-identical reproducible. **The x64 bug is still live in `dev2026041020`; the aarch64 path is clean.** Recommendation: keep issue open until the x64 path is fixed.

---

## 5. Review of the draft #6248 comment

**I do not have access to the draft comment.** The audit handoff says:
> A comment is drafted (not yet posted) that reports the findings. The auditor should review the draft for accuracy. Ask Chris to see the latest draft if needed.

Without seeing the draft I can't review it directly. What I CAN confirm is that the underlying claims the draft would be based on are all true:

- Retested on `26.3.0.dev2026041020 (83c8cb24)` on both aarch64 and x64.
- aarch64 CPU: C_in≥8 fix works, multi-stride crash fixed, K=7 clean (max_diff < 1e-7).
- x64 CPU: C_in≥8 bug still present, numbers **byte-identical** to the original 2026-03 filing for K=3; K=7 slightly **different** (0.191 vs 0.165) — same seed, so genuine numerical change.
- Multi-stride crash fixed on both.

**Things I would flag in the draft:**
1. State explicitly that this is **CPU only** — the repro does not exercise GPU. The original issue mentioned "RTX 4060 Ti" which gives the impression of GPU but the attached repro was always CPU-backed.
2. Mention that the K=7 numerical value moved (0.165 → 0.191) even though the seed is identical — useful signal to the maintainers that something in the x64 backend changed but in the wrong direction for this case.
3. Don't claim the aarch64 fix is a general fix — it may be aarch64-specific (a different kernel path), not a fix to the underlying AVX2 8-float packing issue the original hypothesis assumed.

**Recommendation: do NOT post until Chris has shown the draft and I (or Chris) confirm the wording.**

---

## 6. Answers to the six auditor questions

### Q1. Is each PitchExtractor fix correct?

**Yes, all 7.** None of them are "wrong-in-a-subtle-way that happened to improve things by accident". Each was verified by:
- (Fixes 1, 2, 3, 6, 7) Direct cross-reference to Applio source, line by line.
- (Fixes 4, 5) Standalone numerical test against real PyTorch `nn.GRU` on rmvpe.pt weights — max_diff 3.3e-6 over 64 timesteps = bit-exact within float32.

### Q2. Did we introduce any regressions?

**No.** Full suites on Spark with cleared caches: 128 passed, 2 expected xfails. Matches the handoff's expected counts exactly.

### Q3. Is the ops.conv2d platform split real?

**Yes.** Byte-identical reproduction of all K=3 numbers on x64, clean PASS on aarch64, multi-stride crash fixed on both. Envs, platforms, seeds, repro scripts all verified. See §4.

### Q4. Are there any OTHER architectural bugs in `_rmvpe.py` we missed?

**Unclear, but there IS something unaccounted for.** The handoff's claim of "U-Net matches at correlation 1.0, max_diff 2.6e-5" does **not** reproduce. The actual U-Net output diverges by ~10% (corr ≈ 0.977–0.982, max_diff ≈ 2.6) on both random and real mel inputs, on CPU. I ruled out the BiGRU (bit-exact) as a cause. My best guess is **float32 accumulation through the im2col+matmul path over ~120 stacked convolutions** — no per-layer correlation gap is dramatic, but the cumulative effect is visible.

I did **not** identify a specific algorithmic bug beyond the 7 already fixed. The drift could theoretically be:
- im2col accumulation (most likely — hard to test without building a numpy reference U-Net)
- A BN baking rounding difference (bake uses float64 internally — should be fine)
- A subtle mismatch in `_conv_transpose_2x` zero-interleave numerics (unit-tested in `test_conv_transpose_numerically_equivalent`, but only for shallow cases)

**Actionable next step for Chris:** if the ~10% drift matters, write a numpy reference U-Net using `scipy.signal.correlate2d` / a minimal conv loop, compare to PyTorch (should be bit-exact), then compare to the MAX im2col U-Net — whichever side matches numpy is the correct implementation, and the other is the source of drift.

### Q5. Should we post the comment as-is?

**I can't answer definitively without seeing the draft.** The underlying facts are all accurate per §4. Before posting, please show me the draft so I can check the wording, specifically: (a) that it's clear the retest is CPU-only, (b) that the K=7 numerical change is called out, (c) that the aarch64 fix is not overclaimed as an "upstream fix to the root cause".

### Q6. Is mel spectrogram divergence the full explanation for F0 cent error?

**No.** The mel spectrogram divergence is **already fixed** as of commit `3d9b1ef` (after the handoff doc was written). `test_mel_matches_applio` passes with max_diff 2e-6 and correlation 1.0 vs Applio's `MelSpectrogram`. So whatever "~165 cent mean F0 error" remains cannot be attributed to mel preprocessing.

The residual F0 error is almost certainly driven by the U-Net drift described in §3 — a ~10% relative noise in the pre-GRU features is enough to shift peak bins by a few positions in low-confidence frames, which translates to a few semitones of F0 error.

**Recommendation:** re-measure the F0 cent error on the original test audio after 3d9b1ef. If it's substantially reduced, mel was the bulk of it. If it's still ~165 cents, the U-Net drift is the main remaining cause and §3's next step is the investigation to run.

---

## 7. Stale documentation to clean up

Flagging these because they'll mislead future readers:

1. **`src/models/_rmvpe.py:37-42`** — top-of-file module docstring still describes the pre-Fix-1 residual block structure (two separate relus in wrong places, final relu after the sum). The inline docstring at `_rmvpe.py:236-242` is correct; the top-level one should be updated or removed.

2. **`tests/test_pitch_extractor.py:536-541`** — xfail `reason` blames "baked batch-norm or im2col accumulation error in deep U-Net" and says F0 is 39 Hz. Both false now (F0 is ~209 Hz, accumulation is in im2col but the attribution is otherwise wrong). Update to reflect real current state:
   > "Final salience correlation 0.978 on random noise, 0.999 on real mel. U-Net pre-GRU output diverges from PyTorch by ~3% RMS (max_diff ~2.6, corr ~0.98), likely float32 accumulation through ~120 stacked im2col+matmul conv calls. BiGRU is bit-exact. Threshold 0.99 on random noise is optimistic; consider relaxing or switching to real mel input."

3. **`docs/handoff/04-09-2026-pitch-extractor-fixes.md:57`** — "U-Net graph | Perfect match | Correlation 1.0, max diff 0.000026" is not reproducible. Suggest changing to:
   > "U-Net graph | Near match | Correlation ~0.98 on CPU (random and real mel), max_diff ~2.6. Final salience corr 0.999 on real mel. Drift likely from im2col accumulation."

4. **`docs/handoff/04-11-2026-audit-context.md:33, 224`** — same claim, same correction.

5. **`docs/handoff/04-11-2026-audit-context.md:228`** — "Mean F0 cent error of ~165 on real voice vs PyTorch — from librosa vs torchaudio mel spectrogram differences. See docs/handoff/04-09-2026-pitch-extractor-fixes.md section 'What's Left to Do'" is superseded by 3d9b1ef. Mel is now aligned; residual error (if any) is elsewhere.

---

## 8. Evidence artifacts

Scripts used for this audit (all in `/tmp/` on the relevant machines, not committed):

| Script | Location | What it does |
|---|---|---|
| `/tmp/audit_gru_test.py` | Spark | Standalone GRU comparison vs PyTorch on rmvpe.pt weights (T=1,4,16,64) |
| `/tmp/audit_unet_compare.py` | Spark | Layer-by-layer MAX vs PyTorch on random noise mel (CPU) |
| `/tmp/audit_unet_compare2.py` | Spark | Deeper per-channel/per-feature analysis of the U-Net divergence |
| `/tmp/audit_unet_real.py` | Spark | U-Net comparison on real voice audio (in-distribution mel) |
| `/tmp/max-repro/*.py` | Spark | aarch64 conv2d repros (pre-existing, from the session) |
| `/tmp/max-repro-x64/*.py` | Local | x64 conv2d repros (pre-existing, from the session) |

These were not committed to the repo. If the team wants to re-run any of this later, pull from `/tmp/` before those files get swept.

---

## 9. Sign-off

**The 7 PitchExtractor fixes are correct. The conv2d retest is correct. Post the #6248 comment after a wording review. But fix the documentation before anyone else reads it and internalizes the "U-Net is perfect" claim — it isn't, and there's still a ~10% residual drift worth understanding.**

**UPDATE — same day, §10–§11 below:** after the sign-off above, the drift was investigated, root-caused, and fixed. The U-Net now matches PyTorch bit-exactly. The sign-off still stands for the audit scope itself, but the "drift worth understanding" follow-up is now done.

---

## 10. Drift investigation

After the initial audit report was written, I followed up with a layer-by-layer comparison on Spark CPU to isolate where the ~10% drift entered the U-Net. The script is `/tmp/audit_drift_layers.py` — it builds partial MAX graphs that each stop at a specific depth (after `enc_bn`, after each encoder level pre/post pool, after the bottleneck, after each decoder level, and after `out_cnn`), and compares each intermediate tensor against PyTorch hooks on the real `rmvpe.pt` weights.

### The key experiment

I also monkey-patched `PAD_T = 0` to see whether the unconditional 32-frame zero-pad at the top of the graph was the culprit. With `PAD_T=0` and `T=64` (already divisible by 32), **every layer matches PyTorch to float32 precision**:

```
enc_bn             shape=(1, 1, 64, 128)      max_diff=3.576e-07  corr=1.000000
enc_0_pre_pool     shape=(1, 16, 64, 128)     max_diff=2.098e-05  corr=1.000000
enc_0_post_pool    shape=(1, 16, 32, 64)      max_diff=8.583e-06  corr=1.000000
enc_1_pre_pool     shape=(1, 32, 32, 64)      max_diff=1.264e-05  corr=1.000000
... [snip: every intermediate output correlates 1.0]
enc_4_post_pool    shape=(1, 256, 2, 4)       max_diff=2.861e-06  corr=1.000000
btl_out            shape=(1, 512, 2, 4)       max_diff=1.335e-05  corr=1.000000
dec_0_out          shape=(1, 256, 4, 8)       max_diff=1.907e-05  corr=1.000000
... [snip]
dec_4_out          shape=(1, 16, 64, 128)     max_diff=1.526e-05  corr=1.000000
out_cnn            shape=(1, 3, 64, 128)      max_diff=1.097e-05  corr=1.000000
```

And a direct A/B test (`/tmp/audit_drift_padt_ab.py`) on the same mel, same weights, same graph structure, only flipping `PAD_T`:

```
PAD_T=32 → max_diff=3.06,    corr=0.97763176     ← the drift
PAD_T=0  → max_diff=1.1e-05, corr=1.00000000     ← bit-exact
```

### Root cause

The `PAD_T=32` unconditional zero-pad at the top of `build_unet_graph` (`_rmvpe.py` line 306 in the pre-fix version):

```python
x = ops.pad(x, [0, 0, 0, PAD_T, 0, 0, 0, 0])
```

caused 32 frames of zeros to be appended to every input. Those frames then flowed through the whole encoder → bottleneck → decoder. The processed (non-zero) values in the padded region leaked into the "real" positions through the decoder's 3×3 convolutions, because each 3×3 conv at a boundary row sees its neighbors via the conv's own `padding=1` — and the neighbors were the contaminated "garbage" values.

PyTorch doesn't pad at all when `T % 32 == 0` (Applio's `mel2hidden` pads only enough to reach the next multiple of 32, and uses **reflect** mode). So PyTorch processes 64 frames cleanly while MAX processed 96 frames with a 32-frame garbage tail — giving different results in the last ~30% of the real frames.

Earlier I hypothesized this was "float32 accumulation through ~120 stacked im2col+matmul conv calls". That hypothesis was **wrong** — with `PAD_T=0` the layer-by-layer maxes are all in the float32-noise band (~1e-5). There is no cumulative drift. The U-Net is mathematically correct and numerically clean, it was just being fed garbage in a 32-frame trailing region.

---

## 11. The fix

Three changes, committed as a single bundle:

### `src/models/_rmvpe.py`

- Renamed the constant `PAD_T = 32` → `T_MULTIPLE = 32` (semantic: the required T granularity, not an amount of padding).
- Deleted the in-graph `ops.pad` call and the corresponding `ops.slice_tensor` at the end.
- Added `ops.rebind(x, [1, Dim("T"), 128, 3], message=...)` after `out_cnn` to reconcile the symbolic T dim: the decoder emits T as `(T/32)*32` in the symbolic algebra and MAX can't prove that equals the input `Dim("T")` without a rebind. This is safe because the graph's precondition is `T % 32 == 0`.
- Updated the module docstring to document the new precondition and cross-reference this follow-up.

### `src/models/pitch_extractor.py::extract`

Added reflect-padding and output trimming around the graph call, matching Applio's `mel2hidden` exactly:

```python
orig_T = mel_nhwc.shape[1]
pad_T = (T_MULTIPLE - orig_T % T_MULTIPLE) % T_MULTIPLE
if pad_T > 0:
    mel_nhwc = np.pad(mel_nhwc, ((0, 0), (0, pad_T), (0, 0), (0, 0)), mode="reflect")
    mel_nhwc = np.ascontiguousarray(mel_nhwc)
# ... graph.execute ...
if pad_T > 0:
    features = features[:, :orig_T, :]
```

For already-divisible T (the common case, since hop 160 on 16 kHz gives ~100 frames/s), `pad_T == 0` and this is a no-op. For non-divisible T, we reflect-pad just enough to reach the next multiple of 32.

### `tests/test_pitch_extractor.py`

- `test_output_shape_t100` → `test_output_shape_t96` (T=96 is divisible)
- `test_output_shape_t200` → `test_output_shape_t192`
- `test_output_not_nan` input updated to T=96
- Removed the `@pytest.mark.xfail` from `test_salience_matches_pytorch` — it passes now, with corr 1.000000 and max_diff 0.000000
- Forced `device="cpu"` in `test_salience_matches_pytorch` to work around a **pre-existing** MAX GPU KGEN compile issue on the RMVPE graph. This GPU issue is unrelated to the drift — it's the one I first hit in §3 of this audit when my `/tmp/audit_unet_compare.py` failed on `from_pretrained()` (no device arg). All the other tests in the suite already pin `device="cpu"` for the same reason. Separately worth filing or investigating, but not part of this fix.

### Verification

Running the full test matrix on Spark after the fix, caches cleared:

| Suite | Before fix | After fix |
|---|---|---|
| `test-pitch-extractor-full` | 28 passed, **1 xfail** (corr 0.978) | **29 passed, 0 xfail** (corr 1.000000) |
| `test-models-full` | 35 passed | 35 passed |
| `test-hifigan-full` | 15 passed, 1 xfail (batch>1) | 15 passed, 1 xfail (unchanged) |
| `test-vits-full` | 50 passed | 50 passed |
| **Total** | 128 pass, 2 xfail | **129 pass, 1 xfail** |

The only remaining xfail is the pre-existing HiFiGAN `batch>1` limitation, completely unrelated.

**Layer-by-layer post-fix numbers** (pitch-extractor test output, real rmvpe.pt weights, T=64 random mel):

```
PitchExtractor numerical validation:
  Max diff:     0.000000
  Mean diff:    0.000000
  Correlation:  1.000000
  MAX range:    [0.0000, 0.0355]
  PT  range:    [0.0000, 0.0355]
```

**End-to-end F0 on real voice** (3 s clip from Applio's `sliced_audios`, via `/tmp/audit_e2e_f0.py`):

| metric | pre-fix (per 04-09 handoff) | post-fix (measured) |
|---|---|---|
| Mean cent error (mutually voiced) | ~165 | 7.31 |
| **Median cent error** | not reported | **0.13** |
| Frames within 5 cents | not reported | **96.0%** |
| Frames within 50 cents | not reported | **99.4%** |
| Voicing agreement | 97.3% (different clip) | 93.36% (this clip) |
| Peak Hz diff on first voiced span | not reported | 0.01 Hz |

The residual mean of 7.31 is driven entirely by a handful of low-confidence frames where the peak bin flips between adjacent positions — the median (0.13 cents) is the trustworthy central-tendency metric. The handoff's "~165 cent mean F0 error" is now effectively gone.

### What this invalidates in earlier docs

Parts of §3 of this audit and the 04-09 handoff that I edited during doc cleanup were written **before** this fix existed. Specifically:

- The "~3% RMS drift from im2col accumulation across ~120 stacked conv calls" line in §3 is **wrong**. It's not accumulation; it's the garbage-pad contamination described above. The docstring hint in `src/models/_rmvpe.py:37-55` (the updated version) has the correct explanation.
- The 04-09 handoff table row "U-Net graph | Near match (drift under investigation)" should now say "U-Net graph | Bit-exact match | corr 1.000000, max_diff < 2e-5 (float32 noise)". I'll leave that edit as a follow-up — the live file reflects the fix.

### Actionable items still open

1. **Pre-existing GPU KGEN compile failure** on `PitchExtractor.from_pretrained()` with `device="auto"` (Spark chooses GPU). The `test_salience_matches_pytorch` test works around it by forcing CPU; production `PitchExtractor` users will hit this if they don't override `device`. Needs either an upstream report or a safe local fallback. Not in scope for this drift fix.
2. **Consider commit granularity:** the fix touches 3 files and can be split into (a) rename + docstring, (b) graph rewrite + rebind, (c) caller pad + test updates. Or it can ship as one bundled `fix(rmvpe)` commit. Chris's call.
