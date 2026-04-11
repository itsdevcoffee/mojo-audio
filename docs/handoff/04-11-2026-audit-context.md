# Audit Handoff — PitchExtractor Fixes + MAX Conv2d Bug Retest

> For an auditor agent tasked with validating findings from the April 9–11 session.
> Your job is to verify everything below is sound, catch any bugs or false conclusions,
> and flag anything that looks suspicious. Be skeptical — don't rubber-stamp.

**Date:** 2026-04-11
**Branch:** `main`
**Working directory:** `/home/maskkiller/dev-coffee/repos/mojo-audio`
**Spark:** `ssh visage@visage-spark` → `/home/visage/repos/mojo-audio`
**Local:** `/home/maskkiller/dev-coffee/repos/mojo-audio`

---

## Your Audit Mission

Validate two categories of findings:

1. **PitchExtractor (RMVPE) bug fixes** — 9 bugs identified and fixed in `src/models/_rmvpe.py`. You need to verify each fix is correct, the reasoning is sound, and nothing was broken.

2. **MAX Engine `ops.conv2d` retest** — we retested the `modular/modular#6248` bug report on the latest nightly (`26.3.0.dev2026041020`) across both aarch64 and x64. Findings claim platform-specific fix: aarch64 fixed, x64 still broken, multi-stride crash fixed on both. Verify this by re-running the repros yourself.

**Do NOT take anything below on faith. Re-run, re-check, question assumptions.**

---

## Part 1 — PitchExtractor Fixes

### Background

Before this session: PitchExtractor produced wildly wrong F0 values (39 Hz on 188 Hz voice audio, 1% voicing agreement with PyTorch RMVPE reference). The whole voice conversion pipeline was degraded because of this.

After this session: F0 now 208.9 Hz on same audio, 97.3% voicing agreement, mean cent error ~165. The 04-11 audit traced the residual cent error to a PAD_T zero-pad contamination bug in `build_unet_graph` and fixed it: moved padding outside the graph (reflect-pad in `PitchExtractor.extract` matching Applio's `mel2hidden`). **Post-fix:** U-Net correlation 1.000000 with PyTorch at every layer, salience max_diff 0.000000, end-to-end median cent error on real voice 0.13. See `docs/handoff/04-11-2026-audit-results.md` §3 (original audit finding) and §10–§11 (root-cause investigation + fix).

### Commits to audit (in order)

```
39049dd fix(rmvpe): replace ops.conv2d with im2col+matmul in U-Net graph
eedf6fb fix(rmvpe): correct ReLU placement in residual block
16c4548 fix(rmvpe): add missing ReLU after decoder upsample BN
6715e77 fix(rmvpe): correct flatten order in U-Net output reshape
5d0337c fix(rmvpe): correct GRU gate ordering — swap r and z gates
6c11390 fix(rmvpe): correct GRU update formula — (1-z)*n + z*h
29b6414 fix(rmvpe): correct bins-to-Hz mapping to match Applio/RMVPE training
```

Read each commit's diff and verify the reasoning below matches the actual change.

### Findings to verify

#### Fix 1: ReLU placement in residual block (`_residual_block`)

**Claim:** PyTorch `ConvBlockRes.forward` applies ReLU as part of the conv sequential (`conv→BN→relu→conv→BN→relu`) then adds the shortcut WITHOUT a final relu. Our code had ReLU in the wrong place — second ReLU was missing, and we applied relu after the add instead.

**How to verify:**
1. Read `/home/visage/repos/Applio/rvc/lib/predictors/RMVPE.py` lines 12-58 (`ConvBlockRes` class). Look at `forward()`:
   ```python
   def forward(self, x):
       if self.is_shortcut:
           return self.conv(x) + self.shortcut(x)
       else:
           return self.conv(x) + x
   ```
   And `self.conv` is `nn.Sequential(Conv2d, BN, ReLU, Conv2d, BN, ReLU)`.
2. Read our current `_residual_block` in `src/models/_rmvpe.py` (around line 235). Verify we now do `conv→BN→relu→conv→BN→relu`, then `h + shortcut` with no final relu.
3. Confirm the fix is correct.

**Risks to check:**
- Is there ANY case where PyTorch's `.conv` sequential doesn't end in ReLU? (Should be "no" but verify)
- Did we break something in the intermediate (bottleneck) blocks which use a similar class? They share weights naming `btl.{I}.0.w` etc — verify via isolated test.

#### Fix 2: Missing ReLU after decoder upsample BN

**Claim:** PyTorch `ResDecoderBlock.forward` does `conv1(x)` where `conv1` is `ConvTranspose2d → BN → ReLU`. The ReLU is baked into the sequential. Our code did the ConvTranspose, the BN, but skipped the ReLU before concatenating with the skip.

**How to verify:**
1. Read `/home/visage/repos/Applio/rvc/lib/predictors/RMVPE.py` lines 175-214 (`ResDecoderBlock`):
   ```python
   self.conv1 = nn.Sequential(
       nn.ConvTranspose2d(...),
       nn.BatchNorm2d(...),
       nn.ReLU(),
   )
   def forward(self, x, concat_tensor):
       x = self.conv1(x)
       x = torch.cat((x, concat_tensor), dim=1)
       ...
   ```
2. Read `build_unet_graph` in `src/models/_rmvpe.py` around line 320. Verify the decoder loop has a `ops.relu(x)` after the BN and before `ops.concat`.

#### Fix 3: Flatten order bug (NHWC vs NCHW ordering)

**Claim:** PyTorch outputs `[1, C=3, T, W=128]`, does `transpose(1,2).flatten(-2)` → `[1, T, 384]` where 384 features are ordered `[channel_major, freq]`. Our NHWC output was `[1, T, W=128, C=3]`, and we reshaped directly to `[1, T, 384]` which put 384 features in `[freq_major, channel]` order. **All 384 features feeding the BiGRU were in wrong positions.**

**How to verify:**
1. Read PyTorch `DeepUnet` (search for `def forward` in the E2E class or DeepUnet class, around lines 265-290 in RMVPE.py):
   ```python
   x = self.cnn(self.unet(mel)).transpose(1, 2).flatten(-2)
   ```
   `self.cnn` outputs NCHW `[1, 3, T, 128]`. After `transpose(1,2)` → `[1, T, 3, 128]`. After `flatten(-2)` → `[1, T, 384]`. So the 384 features are laid out as `(ch=0, freq=0), (ch=0, freq=1), ..., (ch=0, freq=127), (ch=1, freq=0), ...`.
2. Read our current `build_unet_graph` output section (around line 355-370):
   ```python
   x_sq = ops.squeeze(x, 0)          # [T, 128, 3]
   x_t = ops.transpose(x_sq, 1, 2)   # [T, 3, 128]
   x = ops.unsqueeze(x_t, 0)         # [1, T, 3, 128]
   x = ops.reshape(x, [1, Dim("T"), 384])
   ```
3. Confirm the final 384 features are in the same `(channel, freq)` order as PyTorch.

**Risks:**
- Is `ops.transpose` in MAX the same semantics as PyTorch (swaps dims 1 and 2)? Verify by test.
- Could the extra squeeze/unsqueeze cause issues with dynamic T dim? (We saw it work in tests, but double-check.)

#### Fix 4: GRU gate ordering (reset/update swap)

**Claim:** PyTorch GRU weight layout is `[W_ir | W_iz | W_in]` — reset first, then update, then new gate. Our numpy code sliced as if the first block were update (z), not reset (r).

**Source of truth:** https://docs.pytorch.org/docs/stable/generated/torch.nn.GRU.html

**How to verify:**
1. Read current `bigru_forward` in `src/models/_rmvpe.py` (around line 386-440). Verify gate slicing:
   ```python
   r = _sigmoid(gi[:, :H]    + gh[:, :H])       # reset gate
   z = _sigmoid(gi[:, H:2*H] + gh[:, H:2*H])    # update gate
   n = np.tanh(gi[:, 2*H:]   + r * gh[:, 2*H:]) # new gate
   ```
2. Write a standalone test: load the real RMVPE checkpoint, feed a small `(1, 4, 384)` input through both PyTorch `nn.GRU` and our `bigru_forward`, and compare outputs. They should match within float32 precision.

#### Fix 5: GRU update formula

**Claim:** PyTorch GRU: `h_t = (1 - z_t) * n_t + z_t * h_{t-1}`. When z=1 the old state passes through. Our code had `(1-z) * h + z * n` — when z=1, the new candidate passes through, which is the opposite semantic.

**Source of truth:** Same PyTorch GRU doc page — the equations section.

**How to verify:**
1. Read the current code around line 410 in `_rmvpe.py`:
   ```python
   return ((1.0 - z) * n + z * h).astype(np.float32)
   ```
2. Cross-check against PyTorch doc equations.
3. The standalone GRU test from Fix 4 will catch this if the formula is wrong.

#### Fix 6: Bins-to-Hz mapping

**Claim:** RMVPE uses cents mapping `bin * 20 + 1997.3794084376191`, Hz = `10 * 2^(cents/1200)`. Our formula used `440 * 2^((bin*20 - 6900)/1200)` which gives bin 0 = 8.2 Hz instead of 31.7 Hz — a ~2 octave systematic error.

**Source of truth:** Applio `RMVPE0Predictor`:
```python
cents_mapping = 20 * np.arange(N_CLASS) + 1997.3794084376191
# Hz = 10 * (2 ** (cents / 1200))
```
See `/home/visage/repos/Applio/rvc/lib/predictors/RMVPE.py` around line 440 and 490.

**How to verify:**
1. Read current `_bins_to_hz` in `src/models/_rmvpe.py` (around line 460). Verify it uses:
   ```python
   cents = bin_indices.astype(np.float64) * 20.0 + 1997.3794084376191
   return (10.0 * (2.0 ** (cents / 1200.0))).astype(np.float32)
   ```
2. Compute the expected Hz for a few known peaks and cross-check against the PyTorch RMVPE output on the same audio (they should agree on F0 when both are voiced).

**Risks:**
- Why does Applio use `1997.3794084376191` — where does that magic number come from? Check if there's any doc/reference. This is just a cents offset but worth understanding. (Bin 0 = 31.7 Hz is a sensible low-frequency pitch floor.)

#### Fix 7: im2col replacement for ops.conv2d

**Claim:** We replaced `ops.conv2d` calls in `src/models/_rmvpe.py::_conv2d` with 2D im2col + matmul as a safeguard against the `modular/modular#6248` C_in≥8 bug. This is mathematically equivalent to conv2d. We verified in isolation that it produces correct results (max_diff < 1e-4 for all channel sizes).

**How to verify:**
1. Read current `_conv2d` function (around line 98). Verify:
   - 1×1 case uses matmul path
   - k×k case uses im2col: extract `kH*kW` shifted slices, concat along channel axis, matmul with reshaped weight
   - Bias is added after
   - `ops.rebind` reconciles symbolic dims after slicing
2. Write a standalone test: random input, random 3×3 weights, C_in values of 8, 16, 32, 64. Compare `_conv2d` output against a pure numpy conv2d. They should match to within 1e-4.
3. Verify the function ONLY calls `ops.slice_tensor`, `ops.concat`, `ops.matmul`, `ops.pad`, `ops.constant`, `ops.add`, `ops.rebind`, `ops.squeeze`, `ops.unsqueeze` — no `ops.conv2d`.

**IMPORTANT CONTEXT:** Based on the Part 2 findings below, the im2col workaround is NOT strictly needed on aarch64 anymore (bug is fixed upstream). But we haven't removed it. Verify that the im2col path is still correct — we may remove it later after proper regression testing.

### Where to look for PyTorch ground truth

Applio's RMVPE source: `/home/visage/repos/Applio/rvc/lib/predictors/RMVPE.py` (on Spark). Key classes:
- `ConvBlockRes` (lines ~12-58)
- `ResEncoderBlock` (lines ~61-90)
- `Encoder` (lines ~94-145)
- `Intermediate` (lines ~148-175)
- `ResDecoderBlock` (lines ~175-215)
- `Decoder` (lines ~218-250)
- `DeepUnet` (lines ~250-290)
- `E2E` (lines ~290-340) — this is the full model
- `BiGRU` (lines ~543-567) — wraps `nn.GRU`
- `RMVPE0Predictor` (lines ~420-530) — full inference including decode

### How to run tests

On Spark (where all PyTorch-based tests work):
```bash
ssh visage@visage-spark
cd /home/visage/repos/mojo-audio
git pull   # make sure you're at f8aa188 or later

# CLEAR CACHES FIRST — this is critical
rm -rf /home/visage/.cache/modular/.max_cache
rm -rf /home/visage/repos/mojo-audio/.pixi/envs/default/share/max/.max_cache
find src tests -name '__pycache__' -exec rm -rf {} + 2>/dev/null

# Run all tests
~/.pixi/bin/pixi run test-pitch-extractor-full  # 28 tests (1 xfail)
~/.pixi/bin/pixi run test-models-full           # 35 tests
~/.pixi/bin/pixi run test-hifigan-full          # 16 tests (1 xfail)
~/.pixi/bin/pixi run test-vits-full             # 50 tests
```

Expected: all tests pass except 2 xfails (PitchExtractor salience correlation ~0.978 threshold 0.99, HiFiGAN batch>1).

### What the xfail means

`tests/test_pitch_extractor.py::TestPitchExtractorCorrectness::test_salience_matches_pytorch` is marked xfail. Current behavior:
- Uses random noise input (`rng.randn(1, 128, 64)`)
- Compares sigmoid(MAX output) vs PyTorch E2E output
- Achieves correlation 0.978, max_diff 0.007
- xfail threshold requires correlation > 0.99

The xfail reason previously said "MAX U-Net output diverges from PyTorch — likely baked BN or im2col accumulation error" and attributed the remaining gap to the numpy BiGRU. The 04-11 audit re-tested both of these and found: (a) the U-Net really is diverged (~0.98 corr, max_diff ~2.6) — the "matches perfectly" correction in the earlier draft of this doc was wrong; (b) the numpy BiGRU is bit-exact vs PyTorch nn.GRU (max_diff 3.3e-6 at T=64), so the gap is NOT there — it really is in the U-Net. The xfail reason was rewritten on 2026-04-11 to reflect this. See `docs/handoff/04-11-2026-audit-results.md` §3 for full numbers.

### Known remaining issues (not this session's scope)

- Mean F0 cent error of ~165 on real voice vs PyTorch. **Originally** attributed to librosa vs torchaudio mel spectrogram differences. That has since been **ruled out**: `test_mel_matches_applio` passes with max_diff 2e-6 and correlation 1.0 after commit `3d9b1ef`. The residual cent error is almost certainly the U-Net drift described above (~3% RMS noise in the pre-GRU features).
- `test_mel_matches_applio` test was added to `tests/test_pitch_extractor.py` (lines 433-471) and is marked `@pytest.mark.slow`. ✅ Now PASSING after commit `3d9b1ef` (mel alignment fix).

---

## Part 2 — MAX `ops.conv2d` Retest

### Background

Chris filed `modular/modular#6248` earlier about `ops.conv2d` producing wrong values when C_in ≥ 8. Claimed root cause: AVX2 8-float micro-kernel packing bug. Workaround: im2col + matmul. Used throughout HiFiGAN vocoder.

This session we retested on the latest MAX nightly (`26.3.0.dev2026041020`, April 10) to see if anything changed before adding a new comment to the issue.

### Findings claimed

**Platform split:**

| Platform | ops.conv2d C_in≥8 | Multi-stride crash |
|----------|-------------------|--------------------|
| linux-aarch64 (DGX Spark) | FIXED (all C_in, all K) | FIXED |
| linux-x64 (Fedora 43, RTX 4060 Ti, CUDA 13) | STILL BROKEN (same error values as March) | FIXED |

**Specific claimed values on x64:**
```
C_in=  8, C_out= 16, K=3: max_diff=0.01723001  [FAIL]
C_in= 16, C_out= 32, K=3: max_diff=0.02368748  [FAIL]
C_in= 32, C_out= 64, K=3: max_diff=0.03833004  [FAIL]
C_in= 64, C_out=128, K=3: max_diff=0.09013467  [FAIL]
C_in=192, C_out=384, K=3: max_diff=0.11577050  [FAIL]
C_in=192, C_out=512, K=7: max_diff=0.19144066  [FAIL]
```

The K=3 numbers are byte-identical to the original report. K=7 case is slightly worse (0.191 vs 0.165 originally).

### How to reproduce

**Repro environments already set up (both use isolated pixi envs — do NOT touch production):**
- Spark: `/tmp/max-repro/` — aarch64 env pinned to `max==26.3.0.dev2026041020`
- Local: `/tmp/max-repro-x64/` — x64 env pinned to `max==26.3.0.dev2026041020`

Both contain:
- `conv2d_repro.py` — original C_in sweep from issue #6248
- `conv2d_repro_k7.py` — K=7 worst case
- `multi_stride_repro.py` — multi-stride compiler crash test

**Run on Spark (aarch64):**
```bash
ssh visage@visage-spark
cd /tmp/max-repro
~/.pixi/bin/pixi run mojo --version   # should show 0.26.3.0.dev2026041020
~/.pixi/bin/pixi run python conv2d_repro.py
~/.pixi/bin/pixi run python conv2d_repro_k7.py
~/.pixi/bin/pixi run python multi_stride_repro.py
```

**Run on local (x64):**
```bash
cd /tmp/max-repro-x64
pixi run mojo --version
pixi run python conv2d_repro.py
pixi run python conv2d_repro_k7.py
pixi run python multi_stride_repro.py
```

**Expected output (what we observed):**
- Spark: all conv2d tests PASS, multi-stride PASS
- Local x64: C_in ≤ 6 PASS, C_in ≥ 8 FAIL (see numbers above), multi-stride PASS

### Things to verify / be suspicious of

1. **Is the repro actually using `ops.conv2d` and not some other op?** Read `conv2d_repro.py` carefully. Make sure the call to `ops.conv2d` is not being intercepted or replaced by something else.

2. **Is the pixi env REALLY at the new version?** Run `pixi run mojo --version` AND check `pixi list max`. The build hash should be `83c8cb24`, not `4362bfeb` (the March 20 build).

3. **Are the platforms ACTUALLY different?** Check `uname -m` on both — should be `aarch64` on Spark, `x86_64` on local.

4. **Test on GPU vs CPU?** The repro uses `DeviceRef.CPU()` and `CPU()` device. The original issue was filed on RTX 4060 Ti, so user assumed GPU, but the repro is actually CPU-only. **This is worth flagging.** If you have time, try running it with `Accelerator()` / `DeviceRef.GPU(0)` to see if the bug also exists on GPU — that could change our understanding.

5. **Float32 precision ambiguity?** The claim is that `diff=0.0` means perfect. Verify — our numpy ground truth computation also uses float32, so both should have the same rounding behavior. Check `conv2d_repro.py::numpy_conv2d` for any dtype gotchas.

6. **K=7 "slightly worse" claim (0.191 vs 0.165):** The numbers we observed are on the CURRENT build, the original numbers are from March. Is this difference meaningful or just random seed / weight init? The repro uses `rng = np.random.default_rng(42)` and the original issue also used seed 42. If the seed is identical, the input should be identical, so a numerical difference genuinely reflects a backend change. **Worth verifying by reading the original #6248 repro script more carefully.** (Our K=7 tests are in `conv2d_repro_k7.py`, not exactly the original repro script.)

### Key files

| File | What |
|------|------|
| `/tmp/max-repro/conv2d_repro.py` (Spark) | C_in sweep repro |
| `/tmp/max-repro-x64/conv2d_repro.py` (local) | same, x64 |
| `/tmp/max-repro/multi_stride_repro.py` (Spark) | Two-conv2d compiler crash test |
| `/tmp/max-repro-x64/multi_stride_repro.py` (local) | same, x64 |

### The draft comment for #6248

A comment is drafted (not yet posted) that reports the findings. The auditor should review the draft for accuracy. Key claims to verify:
- Platform-specific fix
- C_in=8 threshold still holds (AVX2 packing theory)
- Multi-stride crash fixed on both
- Recommendation to keep issue open until x64 fixed

Ask Chris to see the latest draft if needed. Main thing: **don't let Chris post the comment if anything in the findings is wrong.**

---

## Environment Details

### Versions

| | aarch64 (Spark) | x64 (Fedora local) |
|---|---|---|
| Production MAX | 26.3.0.dev2026032005 | 26.3.0.dev2026032005 |
| Retest MAX (isolated env) | 26.3.0.dev2026041020 | 26.3.0.dev2026041020 |
| Mojo | 0.26.3 | 0.26.3 |
| Python | 3.13 | 3.13 |
| Platform | NVIDIA DGX Spark (GB10, 121GB) | Fedora 43, RTX 4060 Ti, CUDA 13.0 |

### Git state

```
Current branch: main
Latest commit when this doc was written: f8aa188 (docs: PitchExtractor bug fix handoff)
```

### Session commits chronology

```
90dcb65 fix: Mojo syntax updates (tests/test_fft.mojo, src/wav_io.mojo)
fc10156 fix(tests): rewrite PitchExtractor correctness test
471728c fix(tests): apply sigmoid to MAX RMVPE output
f132cb4 fix(tests): mark PitchExtractor correctness test as xfail
39049dd fix(rmvpe): im2col + matmul for conv2d (safeguard)
eedf6fb fix(rmvpe): ReLU placement in residual block
16c4548 fix(rmvpe): ReLU after decoder upsample BN
6715e77 fix(rmvpe): flatten order
5d0337c fix(rmvpe): GRU gate ordering
6c11390 fix(rmvpe): GRU update formula
29b6414 fix(rmvpe): bins-to-Hz mapping
f8aa188 docs: handoff for the PitchExtractor fixes
```

### CRITICAL: MAX graph caching gotcha

MAX Engine caches compiled graphs to disk. If you change `_rmvpe.py` and rerun tests, the OLD compiled graph may be reused. Symptoms: test runs in ~3 seconds instead of ~25+ seconds, results unchanged despite code changes.

**Always clear caches before re-testing:**
```bash
rm -rf /home/visage/.cache/modular/.max_cache
rm -rf /home/visage/repos/mojo-audio/.pixi/envs/default/share/max/.max_cache
find src tests -name '__pycache__' -exec rm -rf {} + 2>/dev/null
```

(Same paths apply on local, substituting `/home/maskkiller` for `/home/visage`.)

---

## Questions for the Auditor to Answer

Please provide explicit answers to each:

1. **Is each PitchExtractor fix correct?** For each of the 7 fixes, is the change we made actually what PyTorch does? Are there any fixes that are "wrong in a subtle way that happened to improve correlation but isn't the real fix"?

2. **Did we introduce any regressions?** Run the full test suite on Spark and confirm all tests still pass (27 PitchExtractor + 35 AudioEncoder + 15 HiFiGAN + 50 VITS + the 2 known xfails).

3. **Is the ops.conv2d platform split real?** Re-run both repros yourself and confirm the numbers match. Pay attention to: env version, device (CPU vs GPU), seed determinism.

4. **Are there any OTHER architectural bugs in `_rmvpe.py` we missed?** Run a more exhaustive layer-by-layer comparison vs PyTorch on REAL mel input (not random noise) and see if the remaining correlation gap of ~0.022 is explained by float32 accumulation or is there a residual bug.

5. **Should we post the comment as-is?** Review the draft comment for #6248. Any wording changes, missing caveats, or mis-statements?

6. **Is the mel spectrogram divergence (mentioned as "next issue") actually the full explanation for the F0 cent error?** Or could there still be a subtle issue in our decode logic (`salience_to_hz`)?

Report in: a markdown file at `docs/handoff/04-11-2026-audit-results.md`. Be specific. If something is wrong, show the code / test output that proves it.
