# NSF-HiFiGAN Vocoder Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Implement NSF-HiFiGAN neural vocoder in MAX Graph — takes 192-dim latents + F0 pitch → audio waveform, supporting RVC v2 checkpoints at 32k/40k/48k sample rates.

**Architecture:** Two-stage: (1) harmonic source in numpy generates pitched excitation from F0, (2) MAX Graph runs the neural filter (conv_pre → 4× upsample blocks with residual dilated convs → conv_post → tanh). Weight-norm reconstructed at load time. ConvTranspose1d via generalized zero-interleave + conv2d.

**Tech Stack:** Python, MAX Engine (MAX Graph API), numpy, torch (weight loading only)

**Spec:** `docs/superpowers/specs/2026-03-20-nsf-hifigan-vocoder-design.md`

---

## File Map

| File | Responsibility |
|---|---|
| `src/models/_hifigan_weight_loader.py` | Parse RVC `.pth`, extract `dec.*` keys, reconstruct weight-norm, parse config list |
| `src/models/_hifigan_graph.py` | MAX Graph construction: generalized ConvTranspose, ResBlock, upsample blocks, full graph |
| `src/models/hifigan.py` | `NSFHiFiGAN` class: harmonic source (numpy), `synthesize()`, `from_pretrained()` |
| `src/models/__init__.py` | Export `NSFHiFiGAN` |
| `tests/test_hifigan.py` | All tests: weight loader, shape, NaN, config detection, correctness |

**Reference files** (read, don't modify):
- `src/models/_rmvpe_weight_loader.py` — BatchNorm baking pattern, weight extraction
- `src/models/_rmvpe.py:100-181` — Conv1d K=1 matmul workaround, ConvTranspose stride=2 zero-interleave
- `src/models/pitch_extractor.py` — Class API pattern (from_pretrained, extract)
- `src/models/audio_encoder.py` — Class API pattern (from_pretrained, encode, batch_size)

---

### Task 1: Weight loader + config parsing

**Files:**
- Create: `src/models/_hifigan_weight_loader.py`
- Create: `tests/test_hifigan.py` (initial test file)

The weight loader extracts decoder weights from RVC checkpoints, reconstructs weight-normalized layers, and parses the config list to determine sample rate and architecture params.

- [ ] **Step 1: Write config parsing tests**

Add to `tests/test_hifigan.py`:

```python
"""Tests for NSF-HiFiGAN vocoder."""

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import numpy as np
import pytest


class TestHiFiGANWeightLoader:
    """Tests for _hifigan_weight_loader — no download required."""

    def test_parse_config_48k(self):
        """48kHz config list → correct model config dict."""
        from models._hifigan_weight_loader import parse_rvc_config
        # Minimal RVC v2 48k config list (17 elements)
        config_list = [
            256, 256, 8192, 192, 192, 256, 2, 6, 3, 0.0,
            "1", [3, 7, 11], [[1,3,5],[1,3,5],[1,3,5]],
            [12, 10, 2, 2], 512, [24, 20, 4, 4], 0.5
        ]
        cfg = parse_rvc_config(config_list, sr=48000)
        assert cfg["sample_rate"] == 48000
        assert cfg["upsample_rates"] == [12, 10, 2, 2]
        assert cfg["upsample_kernel_sizes"] == [24, 20, 4, 4]
        assert cfg["upsample_initial_channel"] == 512
        assert cfg["inter_channels"] == 192
        assert cfg["resblock_kernel_sizes"] == [3, 7, 11]
        assert cfg["hop_length"] == 480  # product of upsample_rates

    def test_parse_config_40k(self):
        """40kHz config list → correct hop_length=400."""
        from models._hifigan_weight_loader import parse_rvc_config
        config_list = [
            256, 256, 8192, 192, 192, 256, 2, 6, 3, 0.0,
            "1", [3, 7, 11], [[1,3,5],[1,3,5],[1,3,5]],
            [10, 10, 2, 2], 512, [16, 16, 4, 4], 0.5
        ]
        cfg = parse_rvc_config(config_list, sr=40000)
        assert cfg["sample_rate"] == 40000
        assert cfg["upsample_rates"] == [10, 10, 2, 2]
        assert cfg["hop_length"] == 400

    def test_weight_norm_reconstruction(self):
        """weight_g + weight_v → reconstructed weight."""
        from models._hifigan_weight_loader import reconstruct_weight_norm
        rng = np.random.default_rng(42)
        # Simulate weight_v [C_out, C_in, K] and weight_g [C_out, 1, 1]
        weight_v = rng.standard_normal((64, 32, 7)).astype(np.float32)
        weight_g = rng.standard_normal((64, 1, 1)).astype(np.float32)
        weight = reconstruct_weight_norm(weight_v, weight_g)
        assert weight.shape == (64, 32, 7)
        assert weight.dtype == np.float32
        # Verify: weight = weight_v * (weight_g / norm(weight_v))
        norm_v = np.linalg.norm(weight_v.reshape(64, -1), axis=1, keepdims=True)
        expected = weight_v * (weight_g.reshape(64, 1, 1) / norm_v.reshape(64, 1, 1))
        np.testing.assert_allclose(weight, expected, atol=1e-6)

    def test_extract_dec_keys(self):
        """Only dec.* keys are extracted, prefix stripped."""
        from models._hifigan_weight_loader import extract_decoder_weights
        fake_sd = {
            "enc.some.weight": np.zeros(10, dtype=np.float32),
            "dec.conv_pre.weight_v": np.zeros((512, 192, 7), dtype=np.float32),
            "dec.conv_pre.weight_g": np.zeros((512, 1, 1), dtype=np.float32),
            "dec.ups.0.weight_v": np.zeros((512, 256, 24), dtype=np.float32),
            "flow.some.weight": np.zeros(10, dtype=np.float32),
        }
        weights = extract_decoder_weights(fake_sd)
        assert "conv_pre.weight" in weights  # weight_norm reconstructed
        assert "ups.0.weight" in weights or "ups.0.weight_v" in weights
        assert "enc.some.weight" not in weights
        assert "flow.some.weight" not in weights
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pixi run python -m pytest tests/test_hifigan.py::TestHiFiGANWeightLoader -v 2>&1 | tail -10`
Expected: ModuleNotFoundError (module doesn't exist yet)

- [ ] **Step 3: Implement weight loader**

Create `src/models/_hifigan_weight_loader.py` with:

```python
"""Weight loading for NSF-HiFiGAN from RVC v2 checkpoints.

Extracts dec.* keys from the SynthesizerTrn state dict, reconstructs
weight-normalized layers, and parses the RVC config list.

Internal key convention:
  conv_pre.weight / .bias
  ups.{i}.weight / .bias                    — ConvTranspose1d upsampling
  noise_convs.{i}.weight / .bias            — excitation injection
  resblocks.{i}.convs1.{j}.weight / .bias   — ResBlock dilated conv (first)
  resblocks.{i}.convs2.{j}.weight / .bias   — ResBlock dilated conv (second)
  conv_post.weight / .bias
"""
from __future__ import annotations
import numpy as np
from functools import reduce
import operator


def parse_rvc_config(config_list: list, sr: int) -> dict:
    """Parse RVC v2 positional config list into a model config dict."""
    upsample_rates = config_list[13]
    return {
        "inter_channels": config_list[3],
        "resblock": config_list[10],
        "resblock_kernel_sizes": config_list[11],
        "resblock_dilation_sizes": config_list[12],
        "upsample_rates": upsample_rates,
        "upsample_initial_channel": config_list[14],
        "upsample_kernel_sizes": config_list[15],
        "sample_rate": sr,
        "hop_length": reduce(operator.mul, upsample_rates, 1),
    }


def reconstruct_weight_norm(weight_v: np.ndarray, weight_g: np.ndarray) -> np.ndarray:
    """Reconstruct weight from weight_v (direction) and weight_g (magnitude).

    weight = weight_v * (weight_g / ||weight_v||)
    where ||weight_v|| is computed per output channel (dim 0).
    """
    v = weight_v.astype(np.float64)
    g = weight_g.astype(np.float64)
    # Norm over all dims except dim 0 (output channels)
    norm_v = np.linalg.norm(v.reshape(v.shape[0], -1), axis=1, keepdims=True)
    norm_v = norm_v.reshape(v.shape[0], *([1] * (v.ndim - 1)))
    weight = v * (g.reshape(v.shape[0], *([1] * (v.ndim - 1))) / norm_v)
    return weight.astype(np.float32)


def extract_decoder_weights(state_dict: dict) -> dict[str, np.ndarray]:
    """Extract dec.* keys, strip prefix, reconstruct weight-normalized layers.

    Weight-norm keys (e.g., conv_pre.weight_v + conv_pre.weight_g) are
    combined into a single conv_pre.weight entry.
    """
    # Extract dec.* keys
    dec_sd = {}
    for k, v in state_dict.items():
        if k.startswith("dec."):
            dec_sd[k[4:]] = np.asarray(v, dtype=np.float32)

    # Reconstruct weight-normalized layers
    result = {}
    seen_wn = set()
    for key in sorted(dec_sd.keys()):
        if key.endswith(".weight_v"):
            base = key[:-len(".weight_v")]
            g_key = f"{base}.weight_g"
            if g_key in dec_sd:
                result[f"{base}.weight"] = reconstruct_weight_norm(dec_sd[key], dec_sd[g_key])
                seen_wn.add(key)
                seen_wn.add(g_key)
            else:
                result[key] = dec_sd[key]
                seen_wn.add(key)
        elif key.endswith(".weight_g"):
            continue  # handled with weight_v
        elif key not in seen_wn:
            result[key] = dec_sd[key]

    # Defensive BatchNorm baking: if any keys have .running_mean/.running_var,
    # bake them using the pattern from _rmvpe_weight_loader.bake_batch_norm().
    # Standard RVC NSF-HiFiGAN uses weight_norm (not BN), but handle gracefully.
    baked = {}
    for key in list(result.keys()):
        if key.endswith(".running_mean"):
            base = key[:-len(".running_mean")]
            if f"{base}.running_var" in result:
                from ._rmvpe_weight_loader import bake_batch_norm
                scale, offset = bake_batch_norm(
                    result.get(f"{base}.weight", np.ones_like(result[key])),
                    result.get(f"{base}.bias", np.zeros_like(result[key])),
                    result[key], result[f"{base}.running_var"],
                )
                baked[f"{base}.scale"] = scale
                baked[f"{base}.offset"] = offset
    result.update(baked)

    return result


def load_hifigan_weights(checkpoint_path: str) -> tuple[dict[str, np.ndarray], dict]:
    """Load NSF-HiFiGAN weights from an RVC v2 .pth checkpoint.

    Args:
        checkpoint_path: Path to RVC .pth file.

    Returns:
        (weights_dict, config_dict) where weights_dict has internal flat keys
        and config_dict has model architecture params.
    """
    import torch
    ckpt = torch.load(checkpoint_path, map_location="cpu", weights_only=False)

    # Handle nested state dict
    if "weight" in ckpt:
        sd = {k: v.numpy() if hasattr(v, 'numpy') else np.asarray(v) for k, v in ckpt["weight"].items()}
    else:
        sd = {k: v.numpy() if hasattr(v, 'numpy') else np.asarray(v) for k, v in ckpt.items()}

    sr = ckpt.get("sr", 48000)
    config_list = ckpt.get("config", [])
    if not config_list:
        raise ValueError("RVC checkpoint missing 'config' key")

    weights = extract_decoder_weights(sd)
    config = parse_rvc_config(config_list, sr)

    return weights, config
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `pixi run python -m pytest tests/test_hifigan.py::TestHiFiGANWeightLoader -v`
Expected: All 4 tests PASS

- [ ] **Step 5: Commit**

```bash
git add src/models/_hifigan_weight_loader.py tests/test_hifigan.py
git commit -m "feat(hifigan): add weight loader + config parsing for RVC v2 checkpoints"
```

---

### Task 2: Generalized ConvTranspose1d for arbitrary stride

**Files:**
- Create: `src/models/_hifigan_graph.py` (initial file with ConvTranspose helper)
- Modify: `tests/test_hifigan.py` (add ConvTranspose shape tests)

The RMVPE ConvTranspose is hardcoded for stride=2 with H/W interleaving. HiFiGAN needs strides of 2, 8, 10, 12. We need a generalized version that inserts (S-1) zeros between each sample along the time axis only (Conv1d, so W=1).

- [ ] **Step 1: Write ConvTranspose shape test**

Add `TestConvTranspose` class to `tests/test_hifigan.py`:

```python
class TestConvTranspose:
    """Test generalized ConvTranspose1d via zero-interleave."""

    @pytest.fixture(scope="class")
    def cpu_device(self):
        from max.driver import CPU
        return CPU()

    def _build_and_run(self, cpu_device, C_in, C_out, stride, kernel_size, T_in):
        """Build a graph with a single ConvTranspose1d, execute, return output shape."""
        import numpy as np
        from max import engine
        from max.graph import Graph, TensorType, DeviceRef, Dim
        from max.dtype import DType
        from models._hifigan_graph import conv_transpose_1d

        cpu_ref = DeviceRef.CPU()
        rng = np.random.default_rng(42)
        w_pt = rng.standard_normal((C_in, C_out, kernel_size)).astype(np.float32) * 0.02
        b = np.zeros(C_out, dtype=np.float32)

        with Graph(
            "ct_test",
            input_types=[TensorType(DType.float32, [1, Dim("T"), 1, C_in], cpu_ref)],
        ) as g:
            x = g.inputs[0]
            out = conv_transpose_1d(x, w_pt, b, stride=stride, device_ref=cpu_ref)
            g.output(out)

        model = engine.InferenceSession(devices=[cpu_device]).load(g)
        inp = rng.standard_normal((1, T_in, 1, C_in)).astype(np.float32)
        result = model.execute(inp)
        tensor = list(result.values())[0] if isinstance(result, dict) else result[0]
        return tensor.to_numpy().shape

    def test_stride_2(self, cpu_device):
        """ConvTranspose stride=2: T=10 → T_out=20."""
        shape = self._build_and_run(cpu_device, C_in=64, C_out=32, stride=2, kernel_size=4, T_in=10)
        assert shape[1] == 20, f"Expected T_out=20, got {shape[1]}"

    def test_stride_10(self, cpu_device):
        """ConvTranspose stride=10: T=10 → T_out=100."""
        shape = self._build_and_run(cpu_device, C_in=256, C_out=128, stride=10, kernel_size=16, T_in=10)
        assert shape[1] == 100, f"Expected T_out=100, got {shape[1]}"

    def test_stride_12(self, cpu_device):
        """ConvTranspose stride=12: T=10 → T_out=120."""
        shape = self._build_and_run(cpu_device, C_in=512, C_out=256, stride=12, kernel_size=24, T_in=10)
        assert shape[1] == 120, f"Expected T_out=120, got {shape[1]}"
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pixi run python -m pytest tests/test_hifigan.py::TestConvTranspose -v 2>&1 | tail -10`
Expected: ImportError (models._hifigan_graph doesn't exist)

- [ ] **Step 3: Implement generalized ConvTranspose1d**

Create `src/models/_hifigan_graph.py` with the `conv_transpose_1d` function. Key algorithm for arbitrary stride S on Conv1d (input is NHWC with W=1):

```python
"""MAX Graph builder for NSF-HiFiGAN vocoder.

All Conv1d operations are implemented as Conv2d with NHWC layout and W=1.
ConvTranspose1d uses zero-interleave + regular conv2d, generalized for
arbitrary stride (not just stride=2 as in RMVPE).
"""
from __future__ import annotations
import numpy as np
from max.graph import ops, Dim


def conv_transpose_1d(x, w_pt: np.ndarray, b_np: np.ndarray | None,
                      stride: int, device_ref):
    """Generalized ConvTranspose1d via zero-interleave + conv2d.

    Equivalent to PyTorch ConvTranspose1d(C_in, C_out, kernel_size=K,
    stride=S, padding=K//2-S//2) — which produces output T_out = T_in * S.

    Implementation:
      1. Zero-interleave: insert (S-1) zeros between each time step → [B, T*S, 1, C]
      2. Regular conv2d with flipped weights → [B, T*S, 1, C_out]

    Args:
        x: NHWC input [B, T, 1, C_in].
        w_pt: PyTorch ConvTranspose1d weight [C_in, C_out, K].
        b_np: Bias [C_out] or None.
        stride: Upsample factor S.
        device_ref: MAX DeviceRef.
    """
    C_in = w_pt.shape[0]
    C_out = w_pt.shape[1]
    K = w_pt.shape[2]

    # Padding to achieve T_out = T_in * S:
    #
    # IMPORTANT: The zero-interleave produces T_in*S - (S-1) samples (the last
    # sample has no trailing zeros). After conv with kernel K, stride 1, and
    # asymmetric padding (pad_left, pad_right):
    #   out = (T_in*S - S + 1) - K + 1 + pad_left + pad_right
    # We want out = T_in * S.
    #
    # PyTorch ConvTranspose1d(K, stride=S, padding=(K-S)//2) achieves this.
    # The equivalent after zero-interleave is:
    #   pad_left = (K - S) // 2
    #   pad_right = (K - S + 1) // 2   (handles odd K-S)
    # Verify: 48k block 0: K=24, S=12 → pad_left=6, pad_right=6 ✓
    #         40k block 0: K=16, S=10 → pad_left=3, pad_right=3 ✓
    #         block 3:     K=4,  S=2  → pad_left=1, pad_right=1 ✓
    pad_left = (K - stride) // 2
    pad_right = (K - stride + 1) // 2

    # --- Step 1: Zero-interleave along T (dim 1) ---
    # x: [B, T, 1, C_in]
    # Strategy: use ops.pad to insert S-1 zeros after each time step,
    # then trim the trailing zeros.
    # Reshape to [B, T, 1, 1, C] → pad dim 3 with (0, S-1) → [B, T, 1, S, C]
    # → reshape [B, T*S, 1, C] → trim last S-1 samples → [B, T*S - (S-1), 1, C]
    # Actually simpler: unsqueeze a slot, pad it, reshape to merge.

    _T = x.shape[1]  # symbolic dynamic dim
    x_4d = ops.reshape(x, [1, _T, 1, C_in])  # ensure [1, T, 1, C]

    if stride == 1:
        # No upsampling needed
        x_zi = x_4d
    else:
        # Insert (S-1) zeros after each time step:
        # [B, T, 1, C] → unsqueeze → [B, T, 1, 1, C] → pad → [B, T, S, 1, C]
        # → reshape → [B, T*S, 1, C]
        x_exp = ops.unsqueeze(x_4d, 2)  # [B, T, 1, 1, C]
        # Pad dim 2 (the new dim) with 0 before, S-1 after
        x_padded = ops.pad(x_exp, [0, 0, 0, 0, 0, stride - 1, 0, 0, 0, 0])  # [B, T, S, 1, C]
        x_zi = ops.reshape(x_padded, [1, _T * stride, 1, C_in])  # [B, T*S, 1, C]

    # --- Step 2: Flip weights and apply regular conv2d ---
    # PyTorch ConvTranspose1d weight: [C_in, C_out, K]
    # Flip kernel (reverse K dim) for cross-correlation:
    w_flipped = w_pt[:, :, ::-1].copy()
    # Convert to MAX RSCF: [K, 1, C_in, C_out]  (but swap C_in/C_out for transpose)
    # Actually for ConvTranspose, input channels become output and vice versa:
    # The flipped weight needs to be [K, 1, C_in, C_out] where C_in is the
    # input to the forward conv (which is C_in of the transpose).
    w_max = np.transpose(w_flipped, (2, 0, 1))[:, :, np.newaxis, :]  # [K, C_in, 1, C_out]
    w_max = np.transpose(w_max, (0, 2, 1, 3))  # [K, 1, C_in, C_out]
    w_const = ops.constant(w_max.astype(np.float32), device=device_ref)

    out = ops.conv2d(x_zi, w_const, stride=(1, 1), padding=(pad_left, pad_right, 0, 0))

    if b_np is not None:
        b_const = ops.constant(b_np.reshape(1, 1, 1, -1).astype(np.float32), device=device_ref)
        out = ops.add(out, b_const)

    return out  # [B, T*S, 1, C_out]
```

**Important:** The exact weight transposition and padding math may need adjustment during implementation. The key invariant is: input `[B, T, 1, C_in]` → output `[B, T*S, 1, C_out]`. Verify this with the shape tests.

- [ ] **Step 4: Run tests and iterate on the weight transposition until shapes are correct**

Run: `pixi run python -m pytest tests/test_hifigan.py::TestConvTranspose -v`
Expected: All 3 stride tests PASS with correct output T dimensions.

If shapes are wrong, adjust the padding formula. The target is `T_out = T_in * stride` exactly.

- [ ] **Step 5: Commit**

```bash
git add src/models/_hifigan_graph.py tests/test_hifigan.py
git commit -m "feat(hifigan): add generalized ConvTranspose1d for arbitrary stride"
```

---

### Task 3: ResBlock + LeakyReLU graph ops

**Files:**
- Modify: `src/models/_hifigan_graph.py` (add ResBlock, conv1d, leaky_relu helpers)
- Modify: `tests/test_hifigan.py` (add ResBlock shape test)

- [ ] **Step 1: Write ResBlock shape test**

Add to `tests/test_hifigan.py`:

```python
class TestResBlock:
    """Test ResBlock graph construction."""

    @pytest.fixture(scope="class")
    def cpu_device(self):
        from max.driver import CPU
        return CPU()

    def test_resblock_preserves_shape(self, cpu_device):
        """ResBlock1 output shape == input shape [1, T, 1, 256]."""
        import numpy as np
        from max import engine
        from max.graph import Graph, TensorType, DeviceRef, Dim
        from max.dtype import DType
        from models._hifigan_graph import build_resblock

        cpu_ref = DeviceRef.CPU()
        rng = np.random.default_rng(42)
        channels = 256
        # ResBlock weights: 3 pairs of (convs1, convs2), kernel sizes [3,7,11], dilation [1,3,5]/[1,1,1]
        weights = _make_resblock_weights(rng, channels, [3, 7, 11], [[1,3,5],[1,3,5],[1,3,5]])

        with Graph(
            "rb_test",
            input_types=[TensorType(DType.float32, [1, Dim("T"), 1, channels], cpu_ref)],
        ) as g:
            x = g.inputs[0]
            out = build_resblock(x, weights, device_ref=cpu_ref)
            g.output(out)

        model = engine.InferenceSession(devices=[cpu_device]).load(g)
        inp = rng.standard_normal((1, 50, 1, channels)).astype(np.float32)
        result = model.execute(inp)
        tensor = list(result.values())[0] if isinstance(result, dict) else result[0]
        out_arr = tensor.to_numpy()
        assert out_arr.shape == (1, 50, 1, 256), f"Expected (1,50,1,256) got {out_arr.shape}"


def _make_resblock_weights(rng, channels, kernel_sizes, dilations):
    """Generate random ResBlock weights matching the internal key convention."""
    w = {}
    for j, (ks, dils) in enumerate(zip(kernel_sizes, dilations)):
        for d_idx, d in enumerate(dils):
            pad = (ks * d - d) // 2  # dilated padding
            w[f"convs1.{j * len(dils) + d_idx}.weight"] = rng.standard_normal((channels, channels, ks)).astype(np.float32) * 0.02
            w[f"convs1.{j * len(dils) + d_idx}.bias"] = np.zeros(channels, dtype=np.float32)
        w[f"convs2.{j}.weight"] = rng.standard_normal((channels, channels, ks)).astype(np.float32) * 0.02
        w[f"convs2.{j}.bias"] = np.zeros(channels, dtype=np.float32)
    return w
```

**Note:** The exact weight key convention for ResBlock convolutions may need adjustment during implementation. The critical test is that the ResBlock preserves spatial dimensions.

- [ ] **Step 2: Implement ResBlock, conv1d helper, and leaky_relu**

Add to `src/models/_hifigan_graph.py`:

- `leaky_relu(x, alpha=0.1)` → `ops.maximum(x, ops.mul(x, alpha_const))`
- `conv1d(x, w_np, b_np, dilation, device_ref)` → pads, calls `ops.conv2d` with NHWC layout
- `build_resblock(x, weights, device_ref)` → 3 pairs of dilated conv1d with leaky_relu + residual

The ResBlock structure per kernel size k with dilations [d1, d2, d3]:
```
for each (kernel_size, dilation_list) in zip(kernel_sizes, dilation_sizes):
    x_res = x
    for d in dilation_list:
        x = leaky_relu(x)
        x = conv1d(x, convs1 weights, dilation=d)
    x = leaky_relu(x)
    x = conv1d(x, convs2 weights, dilation=1)
    x = x + x_res
```

Wait — re-reading the spec more carefully. The ResBlock type "1" has this structure for each kernel size:

```
x_res = x
x = leaky_relu(x) → conv1d(K, dilation=d) → leaky_relu(x) → conv1d(K, dilation=1) → x + x_res
```

This is ONE pair per dilation value. With 3 kernel sizes × 3 dilations = 9 total conv pairs across 3 sub-blocks. Each sub-block handles one kernel size with all its dilations sequentially.

Clarify the exact structure by examining RVC source during implementation if needed. The shape test verifies the residual connection works regardless.

- [ ] **Step 3: Run test, iterate until ResBlock shape is preserved**

Run: `pixi run python -m pytest tests/test_hifigan.py::TestResBlock -v`
Expected: PASS with shape (1, 50, 1, 256)

- [ ] **Step 4: Commit**

```bash
git add src/models/_hifigan_graph.py tests/test_hifigan.py
git commit -m "feat(hifigan): add ResBlock + conv1d + leaky_relu graph ops"
```

---

### Task 4: Full HiFiGAN MAX graph

**Files:**
- Modify: `src/models/_hifigan_graph.py` (add `build_hifigan_graph`)
- Modify: `tests/test_hifigan.py` (add full graph shape test)

This task wires together conv_pre → 4× upsample blocks (ConvTranspose + noise_conv + 3× ResBlock) → conv_post → tanh into a single MAX graph.

- [ ] **Step 1: Write full graph shape test**

Add to `tests/test_hifigan.py`:

```python
class TestHiFiGANGraph:
    """Test full HiFiGAN graph construction."""

    @pytest.fixture(scope="class")
    def cpu_device(self):
        from max.driver import CPU
        return CPU()

    @pytest.fixture(scope="class")
    def model_48k(self, cpu_device):
        """Build HiFiGAN graph with random weights, 48kHz config. Compiled once."""
        import numpy as np
        from models._hifigan_graph import build_hifigan_graph
        from max import engine

        rng = np.random.default_rng(42)
        config = {
            "inter_channels": 192,
            "upsample_rates": [12, 10, 2, 2],
            "upsample_initial_channel": 512,
            "upsample_kernel_sizes": [24, 20, 4, 4],
            "resblock_kernel_sizes": [3, 7, 11],
            "resblock_dilation_sizes": [[1,3,5],[1,3,5],[1,3,5]],
            "sample_rate": 48000,
            "hop_length": 480,
        }
        weights = _make_full_hifigan_weights(rng, config)
        graph = build_hifigan_graph(weights, config, device="cpu", batch_size=1)
        return engine.InferenceSession(devices=[cpu_device]).load(graph)

    def test_output_shape_48k(self, model_48k):
        """48kHz: latents [1, 192, 10] + excitation [1, 1, 4800] → audio [1, 1, 4800]."""
        import numpy as np
        rng = np.random.default_rng(42)
        T = 10
        T_audio = T * 480  # 4800
        latents = rng.standard_normal((1, T, 1, 192)).astype(np.float32)  # NHWC
        excitation = rng.standard_normal((1, T_audio, 1, 1)).astype(np.float32)  # NHWC
        result = model_48k.execute(latents, excitation)
        tensor = list(result.values())[0] if isinstance(result, dict) else result[0]
        out = tensor.to_numpy()
        # Output should be [1, T_audio, 1, 1] in NHWC → T_audio = 4800
        assert out.shape[1] == T_audio, f"Expected T_audio={T_audio}, got {out.shape[1]}"

    def test_output_not_nan(self, model_48k):
        """Output must not contain NaN or Inf."""
        import numpy as np
        rng = np.random.default_rng(123)
        T = 10
        latents = rng.standard_normal((1, T, 1, 192)).astype(np.float32) * 0.1
        excitation = rng.standard_normal((1, T * 480, 1, 1)).astype(np.float32) * 0.1
        result = model_48k.execute(latents, excitation)
        tensor = list(result.values())[0] if isinstance(result, dict) else result[0]
        out = tensor.to_numpy()
        assert not np.isnan(out).any(), "Output contains NaN"
        assert not np.isinf(out).any(), "Output contains Inf"
```

Add a helper `_make_full_hifigan_weights(rng, config)` that generates all weights matching the internal key convention for the full architecture.

- [ ] **Step 2: Implement `build_hifigan_graph`**

Add to `src/models/_hifigan_graph.py`:

```python
def build_hifigan_graph(weights, config, device="cpu", batch_size=1):
    """Build the full NSF-HiFiGAN MAX graph.

    Two inputs:
      - latents: [B, T, 1, inter_channels] NHWC (time=T, channels last)
      - excitation: [B, T_audio, 1, 1] NHWC

    Output:
      - audio: [B, T_audio, 1, 1] NHWC (squeezed to [B, T_audio] by caller)

    The graph implements:
      conv_pre → 4× (ConvTranspose + noise_conv + 3× ResBlock) → conv_post → tanh
    """
    from max.graph import Graph, TensorType, DeviceRef, Dim
    from max.dtype import DType

    cpu_ref = DeviceRef.CPU()  # adjust for GPU

    inter_ch = config["inter_channels"]
    upsample_rates = config["upsample_rates"]
    upsample_ks = config["upsample_kernel_sizes"]
    init_ch = config["upsample_initial_channel"]
    hop = config["hop_length"]

    with Graph(
        "hifigan",
        input_types=[
            TensorType(DType.float32, [batch_size, Dim("T"), 1, inter_ch], cpu_ref),
            TensorType(DType.float32, [batch_size, Dim("T") * hop, 1, 1], cpu_ref),
        ],
    ) as g:
        latents = g.inputs[0]   # [B, T, 1, 192]
        excitation = g.inputs[1]  # [B, T*hop, 1, 1]

        # Conv Pre: [B, T, 1, 192] → [B, T, 1, 512]
        x = conv1d(latents, weights["conv_pre.weight"], weights.get("conv_pre.bias"),
                    dilation=1, device_ref=cpu_ref)

        # 4× Upsample blocks
        ch = init_ch  # 512
        for i, (stride, ks) in enumerate(zip(upsample_rates, upsample_ks)):
            ch_next = ch // 2

            # LeakyReLU before upsample
            x = leaky_relu(x)

            # ConvTranspose1d: [B, T_i, 1, ch] → [B, T_i*stride, 1, ch_next]
            x = conv_transpose_1d(x, weights[f"ups.{i}.weight"], weights.get(f"ups.{i}.bias"),
                                  stride=stride, device_ref=cpu_ref)

            # Noise conv: excitation [B, T_audio, 1, 1] → [B, T_i*stride, 1, ch_next]
            # This is a strided Conv1d that downsamples the excitation to match
            # the current feature resolution. Implemented as ops.conv2d with
            # stride=(noise_stride, 1) — forward strided conv is natively supported.
            # noise_stride = product of remaining upsample rates: prod(upsample_rates[i:])
            noise_stride = reduce(operator.mul, upsample_rates[i:], 1)
            nc_w = _pt_conv1d_to_nhwc(weights[f"noise_convs.{i}.weight"])  # [K, 1, 1, C_out]
            nc_out = ops.conv2d(excitation, ops.constant(nc_w, device=device_ref),
                                stride=(noise_stride, 1))
            if f"noise_convs.{i}.bias" in weights:
                nc_b = ops.constant(weights[f"noise_convs.{i}.bias"].reshape(1,1,1,-1), device=device_ref)
                nc_out = ops.add(nc_out, nc_b)
            x = ops.add(x, nc_out)

            # 3× ResBlock
            for j in range(3):  # 3 kernel sizes
                rb_idx = i * 3 + j
                rb_weights = extract_resblock_weights(weights, rb_idx)
                x = build_resblock(x, rb_weights, device_ref=cpu_ref)

            ch = ch_next

        # Conv Post: [B, T_audio, 1, ch] → [B, T_audio, 1, 1]
        x = leaky_relu(x)
        x = conv1d(x, weights["conv_post.weight"], weights.get("conv_post.bias"),
                    dilation=1, device_ref=cpu_ref)
        x = ops.tanh(x)

        g.output(x)

    return g
```

**Important notes for implementer:**
- The `Dim("T") * hop` for excitation input may need to be a separate `Dim("T_audio")` if MAX doesn't support dim arithmetic. In that case, the relationship T_audio = T × hop is enforced by the caller, not the graph.
- `conv1d_strided` is a new helper for the noise_convs (strided Conv1d for downsampling).
- `compute_noise_conv_stride(upsample_rates, i)` computes the product of remaining strides: `prod(upsample_rates[i:])`.
- `extract_resblock_weights(weights, idx)` extracts the weight subset for ResBlock `idx`.
- The NHWC layout means latents arrive as `[B, T, 1, C]` not `[B, C, T]`. The `synthesize()` method will handle the transpose.

- [ ] **Step 3: Run tests and iterate**

Run: `pixi run python -m pytest tests/test_hifigan.py::TestHiFiGANGraph -v`
Expected: Both shape and NaN tests pass.

This is the hardest task — expect multiple iterations on weight transposition, padding math, and dimension ordering. Use the shape tests as your guide.

- [ ] **Step 4: Commit**

```bash
git add src/models/_hifigan_graph.py tests/test_hifigan.py
git commit -m "feat(hifigan): full NSF-HiFiGAN MAX graph with upsample blocks + ResBlocks"
```

---

### Task 5: NSFHiFiGAN class + harmonic source

**Files:**
- Create: `src/models/hifigan.py`
- Modify: `src/models/__init__.py` (add export)
- Modify: `tests/test_hifigan.py` (add integration tests)

- [ ] **Step 1: Write integration tests**

Add to `tests/test_hifigan.py`:

```python
class TestNSFHiFiGAN:
    """Integration tests for NSFHiFiGAN class."""

    @staticmethod
    def _make_full_weights_and_config(sr=48000):
        """Generate random weights + config for a given sample rate."""
        rng = np.random.default_rng(42)
        sr_to_rates = {
            32000: ([10,8,2,2], [20,16,4,4]),
            40000: ([10,10,2,2], [16,16,4,4]),
            48000: ([12,10,2,2], [24,20,4,4]),
        }
        rates, ks = sr_to_rates[sr]
        config = {
            "inter_channels": 192,
            "upsample_rates": rates,
            "upsample_initial_channel": 512,
            "upsample_kernel_sizes": ks,
            "resblock_kernel_sizes": [3, 7, 11],
            "resblock_dilation_sizes": [[1,3,5],[1,3,5],[1,3,5]],
            "sample_rate": sr,
            "hop_length": reduce(operator.mul, rates, 1),
        }
        weights = _make_full_hifigan_weights(rng, config)
        return weights, config

    @pytest.fixture(scope="class")
    def vocoder_48k(self):
        from models.hifigan import NSFHiFiGAN
        w, cfg = self._make_full_weights_and_config(48000)
        return NSFHiFiGAN._from_weights(w, cfg, device="cpu")

    @pytest.fixture(scope="class")
    def vocoder_40k(self):
        from models.hifigan import NSFHiFiGAN
        w, cfg = self._make_full_weights_and_config(40000)
        return NSFHiFiGAN._from_weights(w, cfg, device="cpu")

    @pytest.fixture(scope="class")
    def vocoder_32k(self):
        from models.hifigan import NSFHiFiGAN
        w, cfg = self._make_full_weights_and_config(32000)
        return NSFHiFiGAN._from_weights(w, cfg, device="cpu")

    def test_synthesize_shape_48k(self, vocoder_48k):
        """48kHz: T=10 → T_audio=4800."""
        latents = np.random.randn(1, 192, 10).astype(np.float32)
        f0 = np.full((1, 10), 440.0, dtype=np.float32)
        out = vocoder_48k.synthesize(latents, f0)
        assert out.shape == (1, 4800), f"Expected (1,4800) got {out.shape}"

    def test_synthesize_shape_40k(self, vocoder_40k):
        """40kHz: T=10 → T_audio=4000."""
        latents = np.random.randn(1, 192, 10).astype(np.float32)
        f0 = np.full((1, 10), 440.0, dtype=np.float32)
        out = vocoder_40k.synthesize(latents, f0)
        assert out.shape == (1, 4000), f"Expected (1,4000) got {out.shape}"

    def test_synthesize_shape_32k(self, vocoder_32k):
        """32kHz: T=10 → T_audio=3200."""
        latents = np.random.randn(1, 192, 10).astype(np.float32)
        f0 = np.full((1, 10), 440.0, dtype=np.float32)
        out = vocoder_32k.synthesize(latents, f0)
        assert out.shape == (1, 3200), f"Expected (1,3200) got {out.shape}"

    def test_synthesize_not_nan(self, vocoder_48k):
        """Output must not contain NaN or Inf."""
        latents = np.random.randn(1, 192, 10).astype(np.float32) * 0.1
        f0 = np.full((1, 10), 440.0, dtype=np.float32)
        out = vocoder_48k.synthesize(latents, f0)
        assert not np.isnan(out).any()
        assert not np.isinf(out).any()

    def test_synthesize_unvoiced_not_nan(self, vocoder_48k):
        """All-unvoiced (f0=0) must not produce NaN."""
        latents = np.random.randn(1, 192, 10).astype(np.float32) * 0.1
        f0 = np.zeros((1, 10), dtype=np.float32)
        out = vocoder_48k.synthesize(latents, f0)
        assert not np.isnan(out).any()
        assert not np.isinf(out).any()

    def test_synthesize_batch2_shape(self):
        """Batch=2: output shape is [2, T_audio]."""
        from models.hifigan import NSFHiFiGAN
        w, cfg = self._make_full_weights_and_config(48000)
        vocoder = NSFHiFiGAN._from_weights(w, cfg, device="cpu", batch_size=2)
        latents = np.random.randn(2, 192, 10).astype(np.float32)
        f0 = np.full((2, 10), 440.0, dtype=np.float32)
        out = vocoder.synthesize(latents, f0)
        assert out.shape == (2, 4800), f"Expected (2,4800) got {out.shape}"
```

- [ ] **Step 2: Implement NSFHiFiGAN class**

Create `src/models/hifigan.py`:

```python
"""NSFHiFiGAN: Neural Source-Filter HiFiGAN vocoder via MAX Graph.

Converts latent features + F0 pitch → audio waveform for RVC v2 voice conversion.
Supports 32kHz, 40kHz, and 48kHz sample rates (auto-detected from checkpoint).

Example:
    vocoder = NSFHiFiGAN.from_pretrained("path/to/rvc_model.pth")
    audio = vocoder.synthesize(latents, f0)  # [B, 192, T], [B, T] → [B, T_audio]
"""
from __future__ import annotations
import numpy as np


class NSFHiFiGAN:

    def __init__(self, _model, _device, _config, _batch_size=1):
        self._model = _model
        self._device = _device
        self._config = _config
        self._batch_size = _batch_size

    @classmethod
    def _from_weights(cls, weights, config, device="auto", batch_size=1):
        from max import engine
        from max.driver import Accelerator, CPU, accelerator_count
        from ._hifigan_graph import build_hifigan_graph

        use_gpu = accelerator_count() > 0 if device == "auto" else device == "gpu"
        dev = Accelerator() if use_gpu else CPU()

        graph = build_hifigan_graph(weights, config, device=device, batch_size=batch_size)
        session = engine.InferenceSession(devices=[dev])
        model = session.load(graph)

        return cls(_model=model, _device=dev, _config=config, _batch_size=batch_size)

    @classmethod
    def from_pretrained(cls, checkpoint_path, device="auto", batch_size=1):
        from ._hifigan_weight_loader import load_hifigan_weights
        weights, config = load_hifigan_weights(checkpoint_path)
        return cls._from_weights(weights, config, device=device, batch_size=batch_size)

    def synthesize(self, latents: np.ndarray, f0: np.ndarray) -> np.ndarray:
        """Synthesize audio from latent features and F0.

        Args:
            latents: [B, 192, T] float32 — from VITS encoder/flow.
            f0: [B, T] float32 — F0 in Hz at frame rate, 0 = unvoiced.

        Returns:
            [B, T_audio] float32 — audio waveform [-1, 1].
        """
        from max.driver import Accelerator, Buffer

        B = latents.shape[0]
        T = latents.shape[2]
        hop = self._config["hop_length"]
        sr = self._config["sample_rate"]
        T_audio = T * hop

        # Stage 1: Harmonic source (numpy)
        excitation = self._harmonic_source(f0, T_audio, sr)  # [B, 1, T_audio]

        # Transpose latents from [B, C, T] to NHWC [B, T, 1, C]
        latents_nhwc = np.ascontiguousarray(
            latents.transpose(0, 2, 1)[:, :, np.newaxis, :]  # [B, T, 1, C]
        ).astype(np.float32)

        # Excitation from [B, 1, T_audio] to NHWC [B, T_audio, 1, 1]
        exc_nhwc = np.ascontiguousarray(
            excitation.transpose(0, 2, 1)[:, :, :, np.newaxis]  # [B, T_audio, 1, 1]
        ).astype(np.float32)

        # Execute MAX graph
        if isinstance(self._device, Accelerator):
            inp_lat = Buffer.from_numpy(latents_nhwc).to(self._device)
            inp_exc = Buffer.from_numpy(exc_nhwc).to(self._device)
        else:
            inp_lat = latents_nhwc
            inp_exc = exc_nhwc

        result = self._model.execute(inp_lat, inp_exc)
        out = list(result.values())[0] if isinstance(result, dict) else result[0]
        out_np = out.to_numpy()  # [B, T_audio, 1, 1]

        # Squeeze NHWC → [B, T_audio]
        return out_np.reshape(B, -1)

    @staticmethod
    def _harmonic_source(f0: np.ndarray, T_audio: int, sr: int) -> np.ndarray:
        """Generate excitation signal from F0.

        Args:
            f0: [B, T] F0 in Hz at frame rate, 0 = unvoiced.
            T_audio: Target audio length in samples.
            sr: Sample rate.

        Returns:
            [B, 1, T_audio] excitation signal.
        """
        B, T = f0.shape
        hop = T_audio // T

        # Upsample F0 from frame rate to audio sample rate
        f0_up = np.repeat(f0, hop, axis=-1)[:, :T_audio]  # [B, T_audio]

        # Voiced/unvoiced mask
        uv = (f0_up > 0).astype(np.float32)

        # Phase accumulation + sine generation
        phase = np.cumsum(2.0 * np.pi * f0_up / sr, axis=-1)
        sine = np.sin(phase) * uv

        # Add noise
        noise = np.random.randn(B, T_audio).astype(np.float32) * 0.003
        excitation = (sine + noise).astype(np.float32)

        return excitation[:, np.newaxis, :]  # [B, 1, T_audio]
```

- [ ] **Step 3: Update `__init__.py`**

```python
"""mojo_audio.models — MAX Graph audio encoder, pitch extractor, and vocoder."""

from .audio_encoder import AudioEncoder
from .pitch_extractor import PitchExtractor
from .hifigan import NSFHiFiGAN

__all__ = ["AudioEncoder", "PitchExtractor", "NSFHiFiGAN"]
```

- [ ] **Step 4: Run integration tests**

Run: `pixi run python -m pytest tests/test_hifigan.py::TestNSFHiFiGAN -v`
Expected: All 6 tests pass (3 shape × 3 sample rates, NaN, unvoiced NaN)

- [ ] **Step 5: Commit**

```bash
git add src/models/hifigan.py src/models/__init__.py tests/test_hifigan.py
git commit -m "feat(hifigan): NSFHiFiGAN class with harmonic source + synthesize()"
```

---

### Task 6: Full test suite + pixi task

**Files:**
- Modify: `pixi.toml` (add test-hifigan task)
- Modify: `tests/test_hifigan.py` (add config detection tests, ensure all fixtures are class-scoped)

- [ ] **Step 1: Add pixi task**

Add to `pixi.toml` in the `[tasks]` section:

```toml
test-hifigan = "pytest tests/test_hifigan.py -v -m 'not slow'"
test-hifigan-full = "pytest tests/test_hifigan.py -v -s"
```

- [ ] **Step 2: Add config detection tests**

```python
class TestConfigDetection:
    """Test RVC checkpoint config auto-detection."""

    def test_config_detection_48k(self):
        from models._hifigan_weight_loader import parse_rvc_config
        config_list = [
            256, 256, 8192, 192, 192, 256, 2, 6, 3, 0.0,
            "1", [3, 7, 11], [[1,3,5],[1,3,5],[1,3,5]],
            [12, 10, 2, 2], 512, [24, 20, 4, 4], 0.5
        ]
        cfg = parse_rvc_config(config_list, sr=48000)
        assert cfg["upsample_rates"] == [12, 10, 2, 2]
        assert cfg["hop_length"] == 480

    def test_config_detection_40k(self):
        from models._hifigan_weight_loader import parse_rvc_config
        config_list = [
            256, 256, 8192, 192, 192, 256, 2, 6, 3, 0.0,
            "1", [3, 7, 11], [[1,3,5],[1,3,5],[1,3,5]],
            [10, 10, 2, 2], 512, [16, 16, 4, 4], 0.5
        ]
        cfg = parse_rvc_config(config_list, sr=40000)
        assert cfg["upsample_rates"] == [10, 10, 2, 2]
        assert cfg["hop_length"] == 400

    def test_config_detection_32k(self):
        from models._hifigan_weight_loader import parse_rvc_config
        config_list = [
            256, 256, 8192, 192, 192, 256, 2, 6, 3, 0.0,
            "1", [3, 7, 11], [[1,3,5],[1,3,5],[1,3,5]],
            [10, 8, 2, 2], 512, [20, 16, 4, 4], 0.5
        ]
        cfg = parse_rvc_config(config_list, sr=32000)
        assert cfg["upsample_rates"] == [10, 8, 2, 2]
        assert cfg["hop_length"] == 320
```

- [ ] **Step 3: Run full test suite**

Run: `pixi run test-hifigan`
Expected: All tests pass. Count should be ~15+ tests across all classes.

- [ ] **Step 4: Run existing tests for regression**

Run: `pixi run test-models`
Expected: All 34 AudioEncoder tests still pass (no regression).

- [ ] **Step 5: Commit**

```bash
git add pixi.toml tests/test_hifigan.py
git commit -m "feat(hifigan): complete test suite + pixi task"
```

---

### Known Challenges

1. **ConvTranspose weight transposition** — The exact mapping from PyTorch ConvTranspose1d weights to the flipped conv2d kernel is the trickiest part. Task 2 step 3 may need multiple iterations. Validate against PyTorch output for a small case.

2. **Dim arithmetic** — MAX Graph may not support `Dim("T") * hop` for the excitation input. If so, use a separate unrelated `Dim("T_audio")` and enforce the relationship in the `synthesize()` method.

3. **NHWC layout bookkeeping** — All Conv1d ops go through Conv2d with W=1. The `synthesize()` method handles the PyTorch `[B, C, T]` → MAX `[B, T, 1, C]` transpose. Keep this conversion at the boundary, not inside the graph.

4. **ResBlock internal structure** — The spec describes the structure but the exact correspondence between weight key indices and dilation/kernel pairs needs verification against a real checkpoint during Task 3.

5. **`ops.tanh` availability** — If MAX doesn't have `ops.tanh`, implement via `(exp(2x) - 1) / (exp(2x) + 1)` using available ops.
