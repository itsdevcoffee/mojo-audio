# Do Immediately — Handoff Context

> For a coding agent picking up the two highest-priority items from mojo-audio.
> Read this fully before starting. Both items are independent and can be done in parallel.

---

## Item 1: File MAX Engine conv2d Groups Bug with Modular

### What the bug is

`ops.conv2d` in MAX Engine v26.1 produces incorrect results when:
- `groups > 1` (grouped convolution)
- `K > 1` (non-trivial kernel size)
- Input uses a dynamic `Dim(...)` shape

Specifically: the `pos_conv` layer in our HuBERT implementation
(`groups=16, K=128, C_in/groups=48`) returns values that differ from the
numpy reference by several orders of magnitude (max diff > 100.0 vs expected < 0.001).

This forces the pos_conv stage to run on CPU via a numpy bridge, meaning the full
MAX GPU pipeline cannot be realized. The CNN (Stage 1) and all 12 transformer blocks
(Stage 4) run on GPU correctly — only Stage 3 is broken.

### Where to report

1. **Modular Discord** (`#bugs` or `#max` channel): https://discord.gg/modular
2. **GitHub Issues**: https://github.com/modular/max (if public issues are open)
3. **Modular Forum**: https://forum.modular.com

### Minimal repro script

Save as `experiments/max-bug-repro/conv2d_groups_bug.py` and run with
`pixi run python experiments/max-bug-repro/conv2d_groups_bug.py` on the DGX Spark.

```python
#!/usr/bin/env python3
"""
Minimal repro for MAX Engine conv2d groups bug.

Expected: MAX grouped conv2d output ≈ numpy reference (max diff < 0.001)
Actual:   MAX grouped conv2d output differs wildly (max diff > 100.0)

System: MAX 26.1.0.dev2026010718, DGX Spark GB10 SM_121 ARM64, CUDA 13.0
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../../src"))

import numpy as np
from max import engine
from max.driver import CPU, Accelerator, Tensor, accelerator_count
from max.graph import Graph, TensorType, ops, DeviceRef, Dim
from max.dtype import DType


def numpy_grouped_conv1d_reference(x, filt, groups, padding):
    """Pure numpy grouped conv1d. Ground truth."""
    B, L, _, C_in = x.shape
    K = filt.shape[0]
    C_out = filt.shape[3]
    cin_g = C_in // groups
    cout_g = C_out // groups

    x_pad = np.pad(x, ((0,0), (padding, padding), (0,0), (0,0)))
    L_out = L + 2*padding - K + 1
    out = np.zeros((B, L_out, 1, C_out), dtype=np.float32)

    for g in range(groups):
        w_g = filt[:, :, :, g*cout_g:(g+1)*cout_g]  # [K, 1, cin_g, cout_g]
        w_flat = w_g.reshape(-1, cout_g)
        for l in range(L_out):
            patch = x_pad[:, l:l+K, :, g*cin_g:(g+1)*cin_g]
            out[:, l, 0, g*cout_g:(g+1)*cout_g] = patch.reshape(B, -1) @ w_flat

    return out


def run_max_grouped_conv(x_np, filt_np, groups, padding, use_gpu=False):
    """Run grouped conv2d via MAX Graph."""
    B, L, _, C_in = x_np.shape

    if use_gpu and accelerator_count() > 0:
        dev = Accelerator()
        dev_ref = DeviceRef.GPU(0)
    else:
        dev = CPU()
        dev_ref = DeviceRef.CPU()

    with Graph(
        "grouped_conv_test",
        input_types=[TensorType(DType.float32, [B, Dim("L"), 1, C_in], dev_ref)],
    ) as g:
        x = g.inputs[0]
        w = ops.constant(filt_np, device=dev_ref)
        out = ops.conv2d(x, w, stride=(1, 1), padding=(padding, padding, 0, 0), groups=groups)
        g.output(out)

    session = engine.InferenceSession(devices=[dev])
    model = session.load(g)

    if use_gpu and accelerator_count() > 0:
        inp = Tensor.from_numpy(x_np).to(dev)
    else:
        inp = x_np

    result = model.execute(inp)
    tensor = list(result.values())[0] if isinstance(result, dict) else result[0]
    return tensor.to_numpy()


def main():
    print("MAX Engine conv2d groups bug repro")
    print(f"MAX version: {engine.__version__}")
    print(f"GPU available: {accelerator_count() > 0}")
    print()

    rng = np.random.default_rng(42)

    # ===== Case 1: K=1 (control — should work) =====
    print("Case 1: groups=16, K=1, C_in=768 (control)")
    B, L, C_in, C_out, K, G = 1, 49, 768, 768, 1, 16
    cin_g = C_in // G
    x1 = rng.standard_normal((B, L, 1, C_in)).astype(np.float32)
    w1 = rng.standard_normal((K, 1, cin_g, C_out)).astype(np.float32)
    ref1 = numpy_grouped_conv1d_reference(x1, w1, G, padding=0)
    max1 = run_max_grouped_conv(x1, w1, G, padding=0, use_gpu=True)
    diff1 = np.abs(max1 - ref1).max()
    print(f"  Max diff (K=1):  {diff1:.6f}  {'PASS' if diff1 < 0.01 else 'FAIL'}")

    # ===== Case 2: K=3, small kernel =====
    print()
    print("Case 2: groups=16, K=3, C_in=768")
    K2 = 3
    w2 = rng.standard_normal((K2, 1, cin_g, C_out)).astype(np.float32)
    ref2 = numpy_grouped_conv1d_reference(x1, w2, G, padding=1)
    max2 = run_max_grouped_conv(x1, w2, G, padding=1, use_gpu=True)
    diff2 = np.abs(max2 - ref2[:, :L, :, :]).max()
    print(f"  Max diff (K=3):  {diff2:.6f}  {'PASS' if diff2 < 0.01 else 'FAIL'}")

    # ===== Case 3: K=128, large kernel — the failing case =====
    print()
    print("Case 3: groups=16, K=128, C_in=768 (pos_conv from HuBERT — BUG)")
    K3 = 128
    w3 = rng.standard_normal((K3, 1, cin_g, C_out)).astype(np.float32)
    ref3 = numpy_grouped_conv1d_reference(x1, w3, G, padding=64)
    max3_cpu = run_max_grouped_conv(x1, w3, G, padding=64, use_gpu=False)
    max3_gpu = run_max_grouped_conv(x1, w3, G, padding=64, use_gpu=True)
    diff_cpu = np.abs(max3_cpu[:, :L, :, :] - ref3[:, :L, :, :]).max()
    diff_gpu = np.abs(max3_gpu[:, :L, :, :] - ref3[:, :L, :, :]).max()
    print(f"  Max diff CPU (K=128): {diff_cpu:.6f}  {'PASS' if diff_cpu < 0.01 else 'BUG CONFIRMED'}")
    print(f"  Max diff GPU (K=128): {diff_gpu:.6f}  {'PASS' if diff_gpu < 0.01 else 'BUG CONFIRMED'}")

    # ===== Summary =====
    print()
    print("=" * 60)
    print("SUMMARY")
    print(f"  K=1   CPU diff: {diff1:.2e}  (expected < 1e-3)")
    print(f"  K=3   GPU diff: {diff2:.2e}  (expected < 1e-3)")
    print(f"  K=128 CPU diff: {diff_cpu:.2e}  (expected < 1e-3)")
    print(f"  K=128 GPU diff: {diff_gpu:.2e}  (expected < 1e-3)")
    print()
    if diff_cpu > 0.01 or diff_gpu > 0.01:
        print("BUG: ops.conv2d with groups > 1 and K > small threshold")
        print("     produces incorrect results. K=1 works, large K does not.")
        print()
        print("IMPACT: HuBERT/ContentVec pos_conv layer (groups=16, K=128)")
        print("        cannot run correctly on GPU. Workaround: numpy bridge.")
        print()
        print("ENVIRONMENT:")
        import platform, subprocess
        print(f"  Platform: {platform.machine()}")
        try:
            gpu = subprocess.run(
                ["nvidia-smi", "--query-gpu=name,compute_cap", "--format=csv,noheader"],
                capture_output=True, text=True, timeout=5
            ).stdout.strip()
            print(f"  GPU: {gpu}")
        except Exception:
            pass
        print(f"  MAX version: {engine.__version__}")


if __name__ == "__main__":
    main()
```

### What to include in the bug report

```
Title: ops.conv2d produces incorrect results with groups > 1 and large kernel (K=128)

Environment:
- MAX 26.1.0.dev2026010718
- NVIDIA GB10 (DGX Spark, SM_121, ARM64) + NVIDIA RTX 4060 Ti (SM_89, x86_64)
- CUDA 13.0 / 12.8

Description:
ops.conv2d with groups > 1 returns incorrect values when kernel size K is large
(K=128 in our case). K=1 and small K appear to work correctly. The bug reproduces
on both CPU and GPU sessions, and on both x86_64 and aarch64.

This blocks the HuBERT pos_conv layer (groups=16, K=128, C_in=768)
from running correctly, forcing a numpy CPU bridge in our audio encoder.

Repro: [paste the script above]
Expected: max absolute diff vs numpy reference < 0.001
Actual: max diff >> 1.0 (several orders of magnitude wrong)

Workaround: run pos_conv in numpy outside the MAX graph.
```

---

## Item 2: RMVPE Pitch Extractor in MAX Graph

### What RMVPE is

RMVPE (Robust Melody Via Pitch Estimation) is the pitch extraction model used in RVC v2.
It takes audio and outputs a pitch (F0) value per time frame in Hz.

Why it matters: RMVPE is the step between HuBERT (content encoding) and VITS (synthesis).
Without it on Spark, the VC pipeline falls back to PyTorch for pitch extraction,
which fails on Spark due to ARM64/CUDA issues.

### CONFIRMED architecture (from live checkpoint inspection on DGX Spark, 2026-03-09)

**The architecture is a U-Net, not a simple CNN+BiGRU as previously assumed.**
741 total keys. Key structure:

```
Input: mel spectrogram [B, 1, T, 128]  (image: batch, 1-channel, time, freq)
  ↓
unet.encoder.bn            → BatchNorm(1) initial normalization
  ↓
unet.encoder.layers.0-4    → 5-level encoder with 2× downsampling per level
  channels: 1 → 16 → 32 → 64 → 128 → 256
  each level: 4 residual blocks (Conv2D 3×3 + BatchNorm + ReLU, with shortcut)
  ↓
unet.intermediate.layers.0-3 → bottleneck: 4 residual blocks at 512 channels
  ↓
unet.decoder.layers.0-4    → 5-level decoder with skip connections
  channels: 512 → 256 → 128 → 64 → 32 → 16
  each level: ConvTranspose (upsample) + skip concat + 4 residual blocks
  ↓
cnn.weight (3, 16, 3, 3)   → final Conv2D: 16 → 3 channels
  ↓
Reshape: [B, 3, T, 128] → [B, T, 384]   (3 * 128 = 384 flattened freq features)
  ↓
fc.0.gru (BiGRU)           → input=384, hidden=256, bidirectional → output [B, T, 512]
  weight_ih_l0: (768, 384)    ← forward GRU input weights (768 = 3 gates × 256)
  weight_hh_l0: (768, 256)    ← forward GRU hidden weights
  weight_ih_l0_reverse: (768, 384)   ← reverse GRU (bidirectional)
  weight_hh_l0_reverse: (768, 256)
  ↓
fc.1.weight (360, 512)     → linear output: 512 → 360 pitch bins
  ↓
Argmax + post-processing   → F0 in Hz per frame
```

### Key implementation notes

**Good news — U-Net ops are all MAX-friendly:**
All encoder/decoder convolutions are `Conv2D` with `groups=1`, kernel `3×3`.
No grouped convolutions, no large kernels — NONE of the bugs we hit with HuBERT pos_conv.
`ops.conv2d` (encoder) and `ops.conv_transpose` (decoder) both exist in MAX ops.

**BatchNorm in inference mode = simple linear transform:**
There is no `ops.batch_norm` in MAX. At inference, BatchNorm is:
```
y = (x - running_mean) / sqrt(running_var + eps) * weight + bias
  = x * (weight / sqrt(running_var + eps)) + (bias - running_mean * scale)
```
Bake this into `scale` and `offset` constants from the checkpoint's `running_mean`
and `running_var` tensors. Then just: `ops.add(ops.mul(x, scale_const), offset_const)`.

**BiGRU strategy — split at the GRU boundary:**
`max.nn` has no GRU (confirmed both Fedora and DGX Spark). The GRU is small and at
the very end on `[B, T, 384]`. Use the same split pattern as HuBERT's pos_conv:

1. MAX Graph: U-Net encoder + intermediate + decoder + output CNN → `[B, T, 384]`
2. Numpy: BiGRU on `[B, T, 384]` → `[B, T, 512]`
3. MAX Graph or numpy: Linear `[B, T, 512]` → `[B, T, 360]`

The U-Net is the heavy compute (GPU wins here); the GRU is sequential and small.

**BiGRU numpy implementation:**
```python
import numpy as np

def sigmoid(x): return 1 / (1 + np.exp(-x))

def bigru_forward(x, sd, hidden_size=256):
    """x: [B, T, 384]. Returns [B, T, 512]."""
    w_ih_f  = sd["fc.0.gru.weight_ih_l0"]         # (768, 384)
    w_hh_f  = sd["fc.0.gru.weight_hh_l0"]         # (768, 256)
    b_ih_f  = sd["fc.0.gru.bias_ih_l0"]           # (768,)
    b_hh_f  = sd["fc.0.gru.bias_hh_l0"]           # (768,)
    w_ih_r  = sd["fc.0.gru.weight_ih_l0_reverse"] # (768, 384)
    w_hh_r  = sd["fc.0.gru.weight_hh_l0_reverse"] # (768, 256)
    b_ih_r  = sd["fc.0.gru.bias_ih_l0_reverse"]
    b_hh_r  = sd["fc.0.gru.bias_hh_l0_reverse"]

    def gru_step(x_t, h, w_ih, w_hh, b_ih, b_hh):
        H = hidden_size
        gi = x_t @ w_ih.T + b_ih   # [B, 3H]
        gh = h @ w_hh.T + b_hh     # [B, 3H]
        z = sigmoid(gi[:, :H]    + gh[:, :H])     # update gate
        r = sigmoid(gi[:, H:2*H] + gh[:, H:2*H]) # reset gate
        n = np.tanh(gi[:, 2*H:]  + r * gh[:, 2*H:]) # new gate
        return (1 - z) * h + z * n

    B, T, _ = x.shape
    # Forward
    h = np.zeros((B, hidden_size), np.float32)
    fwd = []
    for t in range(T):
        h = gru_step(x[:, t], h, w_ih_f, w_hh_f, b_ih_f, b_hh_f)
        fwd.append(h)
    # Reverse
    h = np.zeros((B, hidden_size), np.float32)
    rev = [None] * T
    for t in range(T-1, -1, -1):
        h = gru_step(x[:, t], h, w_ih_r, w_hh_r, b_ih_r, b_hh_r)
        rev[t] = h

    fwd_out = np.stack(fwd, axis=1)  # [B, T, H]
    rev_out = np.stack(rev, axis=1)  # [B, T, H]
    return np.concatenate([fwd_out, rev_out], axis=-1)  # [B, T, 512]
```

### Where the weights are

The checkpoint is at `lj1995/VoiceConversionWebUI`, file `rmvpe.pt` (~181MB).
Check if already cached (may have been downloaded by Applio):
```bash
find ~/.cache -name "rmvpe.pt" 2>/dev/null
```

If not cached:
```python
from huggingface_hub import hf_hub_download
path = hf_hub_download("lj1995/VoiceConversionWebUI", "rmvpe.pt")
```

### Validated weight key examples (from live inspection)

```
unet.encoder.bn.weight: (1,)
unet.encoder.layers.0.conv.0.conv.0.weight: (16, 1, 3, 3)     ← first residual block
unet.encoder.layers.0.conv.0.conv.1.weight: (16,)              ← BN weight
unet.encoder.layers.0.conv.0.conv.1.running_mean: (16,)        ← BN running stats
unet.encoder.layers.0.conv.0.shortcut.weight: (16, 1, 1, 1)   ← residual shortcut
unet.decoder.layers.0.conv1.0.weight: (512, 256, 3, 3)         ← ConvTranspose
unet.decoder.layers.0.conv2.0.conv.0.weight: (256, 512, 3, 3) ← after skip concat
cnn.weight: (3, 16, 3, 3)
cnn.bias: (3,)
fc.0.gru.weight_ih_l0: (768, 384)
fc.1.weight: (360, 512)
fc.1.bias: (360,)
```

### MAX Graph API facts (all verified on DGX Spark SM_121)

```python
from max.graph import Graph, TensorType, ops, DeviceRef, Dim
from max.dtype import DType

# Dynamic dims: use Dim(), NOT -1
TensorType(DType.float32, [1, 1, Dim("T"), 128], device_ref)

# Conv2D — works correctly for groups=1 (no bug)
ops.conv2d(x, w, stride=(2,2), padding=(1,1,1,1), groups=1)

# ConvTranspose — available for decoder upsampling
ops.conv_transpose(x, w, stride=(2,2), ...)

# All standard activations: ops.relu, ops.gelu, ops.sigmoid, ops.tanh
# Layer ops: ops.layer_norm, ops.matmul, ops.reshape(-1 ok), ops.softmax
# ops.transpose(x, a, b) — swaps only 2 axes, not a permutation list
# results: tensor.to_numpy()  NOT np.array(tensor) which returns shape ()
```

### Suggested implementation structure

```
src/models/
  _rmvpe.py          # U-Net encoder/decoder, BatchNorm baking, BiGRU, output
  pitch_extractor.py # PitchExtractor class (mirrors AudioEncoder pattern)
```

```python
from mojo_audio.models import PitchExtractor

model = PitchExtractor.from_pretrained("lj1995/VoiceConversionWebUI", filename="rmvpe.pt")
f0_hz = model.extract(audio_np)  # float32 [1, 16000] → float32 [T_frames] Hz, 0=unvoiced
```

### Validation target

Compare against Applio's RMVPE on the same audio:
- `rvc/lib/predictors/RMVPE.py` → `RMVPE.infer_from_audio(audio, thred=0.03)`
- Our output must match within ±5 cents on voiced frames (~0.3% Hz error)
- 1s @16kHz → ~100 F0 frames

### Files to read first

1. `src/models/audio_encoder.py` — the HuBERT template to follow
2. `src/models/_weight_loader.py` — extend this for RMVPE weight keys
3. `docs/plans/03-05-2026-hubert-contentvec-max-graph.md` — the approach we used

### pixi tasks to add when done

```toml
test-pitch-extractor = "pytest tests/test_pitch_extractor.py -v -m 'not slow'"
test-pitch-extractor-full = "pytest tests/test_pitch_extractor.py -v -s"
```

---

## Environment Setup

Both items run in the mojo-audio pixi environment:

```bash
# Local machine
cd /home/maskkiller/dev-coffee/repos/mojo-audio
pixi run python <script>

# DGX Spark
ssh visage@visage-spark
cd /home/visage/repos/mojo-audio
~/.pixi/bin/pixi run python <script>
```

All dependencies already installed: MAX Engine, PyTorch, transformers, safetensors,
huggingface_hub, onnxruntime, pytest.

The HuBERT model (~360MB) is already cached from Task 7:
`~/.cache/mojo-audio/models/facebook--hubert-base-ls960/`
