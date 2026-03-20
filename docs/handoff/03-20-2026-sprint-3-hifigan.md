# Sprint 3 Handoff — NSF-HiFiGAN Vocoder

> For a coding agent picking up Sprint 3 of mojo-audio.
> Read this fully before starting. The WIP issue is well-scoped.

**Current date:** 2026-03-20
**Branch:** `main`
**Working directory:** `/home/maskkiller/dev-coffee/repos/mojo-audio`

---

## Context

Sprint 3 implements the NSF-HiFiGAN neural vocoder for RVC v2 voice conversion. The vocoder takes 192-dim latent features + F0 pitch → audio waveform. Supports 32k/40k/48k sample rates.

**Spec:** `docs/superpowers/specs/2026-03-20-nsf-hifigan-vocoder-design.md`
**Plan:** `docs/superpowers/plans/2026-03-20-nsf-hifigan-vocoder.md`

---

## What's Done

All 6 plan tasks are complete and tests pass with random weights:

| Task | Status | Files |
|------|--------|-------|
| 1. Weight loader + config parsing | ✅ | `src/models/_hifigan_weight_loader.py` |
| 2. Generalized ConvTranspose1d | ✅ | `src/models/_hifigan_graph.py` (conv_transpose_1d) |
| 3. ResBlock + LeakyReLU | ✅ | `src/models/_hifigan_graph.py` (build_resblock, conv1d, leaky_relu) |
| 4. Full HiFiGAN MAX graph | ✅ | `src/models/_hifigan_graph.py` (build_hifigan_graph) |
| 5. NSFHiFiGAN class + harmonic source | ✅ | `src/models/hifigan.py` |
| 6. Test suite + pixi task | ✅ | `tests/test_hifigan.py`, `pixi.toml` |

**Test results:** 15 pass, 1 xfail (batch>1 blocked by ConvTranspose B=1 constraint)

**Additional work completed:**
- Filed `modular/modular#6248` — `ops.conv2d` produces incorrect results when C_in >= 8
- Replaced `conv1d` with im2col + matmul workaround (avoids buggy `ops.conv2d`)
- Created PyTorch reference helper: `tests/_rvc_pytorch_reference.py`
- ResBlock parallel averaging bug fixed (was sequential, now parallel + average)
- Config parsing fixed for real RVC checkpoints (18-element list, not 17)
- Added `load_hifigan_weights()` function for loading from `.pth` files

---

## What's Remaining — ONE Issue

### Symbolic shape mismatch with real RVC weights

**The problem:** Tests pass with random weights (graph compiles fine). But loading a real RVC checkpoint and building the graph fails at compilation:

```
ValueError: Failed to create op 'reshape':
  error: [reshape] input and output number of elements must match
```

**Root cause:** The im2col `conv1d` implementation uses `ops.slice_tensor` to extract K shifted copies of the padded input. Each slice creates a new symbolic dimension. Even though we added `ops.rebind` at the end of `conv1d` to reconcile T, the subsequent `conv_transpose_1d` function does `ops.squeeze(x, 0)` → `ops.unsqueeze` → `ops.pad` → `ops.reshape` to zero-interleave. The reshape `[T, S, 1, C] → [T*S, 1, C]` fails because MAX can't prove the element count matches after the rebind chain.

**The fix:** Add `ops.rebind` calls at strategic points in the upsample path inside `build_hifigan_graph`. Specifically, before each `conv_transpose_1d` call, rebind x to have a clean T dimension that MAX can reason about. The pattern is:

```python
# Before conv_transpose_1d:
x = ops.rebind(x, [batch_size, expected_T, 1, ch], message="pre-upsample rebind")
```

Where `expected_T` is the symbolic `Dim("T")` scaled by previous upsample factors. This may require tracking the current T dimension through the upsample chain.

**Alternative approach:** Modify `conv_transpose_1d` itself to use `ops.rebind` on its input before the squeeze/reshape chain. The key is that after im2col conv1d, the output's T dim is technically correct but symbolically opaque to MAX.

**Where to look:**
- `src/models/_hifigan_graph.py:conv_transpose_1d` (lines 34-118) — the reshape at line 87
- `src/models/_hifigan_graph.py:build_hifigan_graph` (lines 330-430) — the upsample loop
- `src/models/_hifigan_graph.py:conv1d` (lines 147-235) — the rebind at line 233

**Why tests pass but real weights don't:** The test uses `_make_full_hifigan_weights` which generates weights with the same config, and the graph builds in one shot. With real weights loaded from `.pth`, the weight shapes are identical but the graph construction path may trigger different MAX compiler behavior. The actual divergence point needs debugging.

### After the shape fix: Numerical correctness comparison

Once the graph compiles with real weights, run:

```python
import sys; sys.path.insert(0, 'tests'); sys.path.insert(0, 'src')
import numpy as np
from _rvc_pytorch_reference import run_pytorch_reference
from models.hifigan import NSFHiFiGAN

ckpt = '/home/maskkiller/Downloads/voice files/extracted/theweeknd biggest data set/theweekv1.pth'
rng = np.random.default_rng(42)
latents = rng.standard_normal((1, 192, 20)).astype(np.float32) * 0.1
f0 = np.full((1, 20), 220.0, dtype=np.float32)

pt_audio = run_pytorch_reference(ckpt, latents, f0)
vocoder = NSFHiFiGAN.from_pretrained(ckpt, device='cpu')
max_audio = vocoder.synthesize(latents, f0)

diff = np.abs(pt_audio - max_audio)
corr = np.corrcoef(pt_audio.flatten(), max_audio.flatten())[0, 1]
print(f'Max diff: {diff.max():.6f}, Correlation: {corr:.6f}')
```

**Expected outcome:** The neural filter (conv_pre → upsample → resblocks → conv_post) should match PyTorch closely (< 1e-3 diff). The harmonic source will differ (we use simplified single-sine vs PyTorch's multi-harmonic SineGen), so overall correlation may not be 1.0, but the neural filter path should be numerically identical.

**If correlation is still low after shape fix:** The issue is in the harmonic source difference. To isolate, extract PyTorch's excitation signal and feed it directly to the MAX graph (bypassing our numpy harmonic source).

---

## RVC Voice Model Files

Drew's production models are at `/home/maskkiller/Downloads/voice files/extracted/`:

| Artist | SR | File |
|--------|-----|------|
| The Weeknd | 48k | `theweeknd biggest data set/theweekv1.pth` |
| Drake | 40k | `drake/drake_e1000_s14000.pth` |
| Gunna | 48k | `Gunna (best model) - Weights Model (1)/model.pth` |
| + 14 more | mixed | See full list in extracted/ |

---

## Key Files

| File | Purpose |
|------|---------|
| `src/models/hifigan.py` | Public API: NSFHiFiGAN class |
| `src/models/_hifigan_graph.py` | MAX graph builder — **this is where the fix goes** |
| `src/models/_hifigan_weight_loader.py` | RVC checkpoint loader |
| `tests/test_hifigan.py` | 16 tests (15 pass, 1 xfail) |
| `tests/_rvc_pytorch_reference.py` | PyTorch reference for comparison |
| `experiments/max-bug-repro/conv2d_cin_bug.py` | Repro for modular/modular#6248 |

---

## MAX Workarounds in This Codebase

| Issue | Workaround | Filed |
|-------|-----------|-------|
| `ops.conv2d` wrong for C_in >= 8 | im2col + matmul in `conv1d` | modular/modular#6248 |
| `ops.conv2d` wrong for groups > 1 | Fixed in MAX 26.3 | modular/modular#6129 (closed) |
| No `ops.conv_transpose` | Zero-interleave + conv2d | N/A (known limitation) |
| No dilated conv support | Pre-expand kernel with zeros | N/A |
| K=1 C=1 conv2d layout failure | Zero-pad kernel to K=3 | N/A |
| Symbolic dim mismatch | `ops.rebind` at boundaries | N/A |
| No `ops.batch_norm` | Bake at weight load time | N/A |

---

## Environment

```bash
cd /home/maskkiller/dev-coffee/repos/mojo-audio
pixi run test-hifigan          # 15 pass, 1 xfail
pixi run test-models           # 34 pass (AudioEncoder regression check)
pixi run test-pitch            # 7 pass (Mojo DSP)

# DGX Spark
ssh visage@visage-spark
cd /home/visage/repos/mojo-audio
git pull && ~/.pixi/bin/pixi run test-hifigan
```

MAX version: `26.3.0.dev2026032005`

---

## What NOT to Do

- Do not change the public `NSFHiFiGAN` API (synthesize signature is final)
- Do not change the weight loader (verified against real checkpoints)
- Do not use `ops.conv2d` for conv1d — it's broken for C_in >= 8
- Do not compile separate graphs per sample rate in tests (OOM on 16-32GB machines)
- Do not implement speaker conditioning (gin_channels) — not in Sprint 3 scope
