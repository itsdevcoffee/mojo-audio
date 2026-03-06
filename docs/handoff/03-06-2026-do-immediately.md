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
It takes a mel spectrogram and outputs a pitch (F0) value per time frame.

- Input: mel spectrogram `[B, n_mels=128, T]`
- Output: pitch probability distribution `[B, 360, T]` then argmax → Hz per frame
- The 360 pitch bins cover C1–B7 at 20-cent resolution (6 octaves × 60 bins)

Why it matters: RMVPE is the step between HuBERT (content) and VITS (synthesis).
Without it on Spark, the VC pipeline falls back to PyTorch for pitch extraction
(which fails on Spark due to ARM64/CUDA issues).

### Where the weights are

```python
# HuggingFace:
"lj1995/VoiceConversionWebUI"  # file: rmvpe.pt (~181MB) or rmvpe.onnx (~362MB)

# Download:
from huggingface_hub import hf_hub_download
hf_hub_download("lj1995/VoiceConversionWebUI", "rmvpe.pt")
```

### Architecture (from paper + RVC source)

RMVPE is a deep CNN-based pitch salience estimator adapted from E2E-ROFORMER.
From reading the Applio source at `rvc/lib/predictors/RMVPE.py`:

```
Input: raw audio → mel spectrogram [B, 128, T]
  ↓
Stack of residual CNN blocks (conv2d, batch norm, ReLU)
  ↓
Bidirectional GRU layers
  ↓
Linear output layer → [B, 360, T]  (pitch salience per bin)
  ↓
Post-processing → peak Hz per frame
```

Key architecture details to verify by reading the model:
```python
import torch
from huggingface_hub import hf_hub_download
ckpt = torch.load(hf_hub_download("lj1995/VoiceConversionWebUI", "rmvpe.pt"), map_location="cpu")
print(list(ckpt.keys())[:20])          # see top-level structure
print(ckpt.get("model", ckpt).keys())  # see layer names
```

This will reveal the actual layer structure, which is what drives the MAX Graph implementation.

### How to implement (follow the HuBERT pattern)

The HuBERT implementation in `src/models/audio_encoder.py` is the template.
The same MAX Graph API patterns apply:

```python
# Key MAX API facts already discovered:
from max.graph import Graph, TensorType, ops, DeviceRef, Dim
from max.dtype import DType

TensorType(DType.float32, [1, Dim("T"), 128], device_ref)  # dynamic T, NOT -1
ops.conv2d(x, w, stride=(1,1), padding=(0,0,0,0), groups=1)  # groups=1 works correctly
ops.layer_norm(x, gamma, beta, 1e-5)
ops.gelu(x), ops.relu(x), ops.sigmoid(x)
ops.matmul(x, w_transposed)   # w must be .T of PyTorch [out, in] format
ops.transpose(x, axis_a, axis_b)  # ONLY 2 axes at a time, not a permutation list
ops.reshape(x, [1, -1, C])    # -1 works in reshape even though TensorType needs Dim()
result = model.execute(inp)   # positional arg, not keyword
tensor.to_numpy()              # NOT np.array(tensor) — returns shape ()
```

**If RMVPE uses GRU/LSTM:** These are not directly available as `ops.*` primitives.
Options:
1. Check if `max.nn` has a GRU — look at `dir(max.nn)` for "GRU", "LSTM", "RNN"
2. If not: implement GRU manually using `ops.matmul`, `ops.sigmoid`, `ops.tanh`, `ops.add`, `ops.mul`
   (GRU has 3 gate operations — update, reset, candidate — all expressible with basic ops)
3. Alternatively: export RMVPE to ONNX and test via `onnxruntime` to validate the architecture
   before implementing in MAX (use the already-installed `onnxruntime` in the pixi env)

**Suggested implementation structure:**

```
src/models/
  _rmvpe.py              # CNN blocks, GRU, output layer
  pitch_extractor.py     # PitchExtractor class (mirrors AudioEncoder pattern)
```

```python
from mojo_audio.models import PitchExtractor

model = PitchExtractor.from_pretrained("lj1995/VoiceConversionWebUI/rmvpe.pt")
f0_hz = model.extract(audio_np)   # [1, T_frames] in Hz, 0 where unvoiced
```

### Validation target

Compare against `torchcrepe` or Applio's RMVPE inference:
- Same input → output F0 values should match within 5 cents (~0.3% Hz error)
- Run on 1s @16kHz → expect ~100 F0 values (one per 10ms frame)

### Files to read first

Before writing any code, read:
1. `src/models/audio_encoder.py` — the HuBERT implementation is the template
2. `src/models/_weight_loader.py` — extend this to handle RMVPE weight keys
3. `docs/plans/03-05-2026-hubert-contentvec-max-graph.md` — the plan we followed

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
