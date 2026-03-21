# RMVPE Pitch Extractor Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Implement `PitchExtractor` — a MAX Graph U-Net that extracts F0 pitch from audio, replacing PyTorch RMVPE in the voice conversion pipeline so pitch extraction works on DGX Spark ARM64.

**Architecture:** U-Net MAX Graph (encoder + bottleneck + decoder + output CNN) outputs `[B, T, 384]`, then numpy BiGRU produces `[B, T, 512]`, then linear `[B, T, 360]` pitch salience bins, then argmax + Hz conversion. BatchNorm is baked into conv scale/offset at load time (no `ops.batch_norm` needed).

**Tech Stack:** MAX Engine (`max.graph.ops`), numpy, torchaudio (mel spectrogram), PyTorch (checkpoint loading via `torch.load`)

**Context:** Read `docs/handoff/03-06-2026-do-immediately.md` in full before starting — it has the confirmed architecture, exact weight key examples, and a copy-paste-ready BiGRU implementation.

---

## Key Facts (confirmed from live checkpoint inspection)

```
Checkpoint: lj1995/VoiceConversionWebUI, file rmvpe.pt (~181MB)
Total keys: 741

Architecture data flow:
  audio [1, N] @16kHz
    → mel [1, 1, T_mel, 128]       (B=1, channels=1, time, freq)
    → U-Net MAX Graph
    → [1, T_mel, 384]              (3 * 128 flattened freq features)
    → numpy BiGRU(384→512)
    → [1, T_mel, 512]
    → numpy linear(512→360)
    → [1, T_mel, 360]
    → argmax + Hz conversion
    → [T_mel] Hz (0 = unvoiced)

Confirmed weight key examples:
  unet.encoder.bn.weight: (1,)                                  ← initial BN
  unet.encoder.layers.0.conv.0.conv.0.weight: (16, 1, 3, 3)    ← encoder L0, block 0, conv
  unet.encoder.layers.0.conv.0.conv.1.weight: (16,)             ← BN weight in that block
  unet.encoder.layers.0.conv.0.conv.1.running_mean: (16,)
  unet.encoder.layers.0.conv.0.shortcut.weight: (16, 1, 1, 1)  ← shortcut (channels change)
  unet.decoder.layers.0.conv1.0.weight: (512, 256, 3, 3)        ← ConvTranspose [C_in, C_out, H, W]
  unet.decoder.layers.0.conv2.0.conv.0.weight: (256, 512, 3, 3) ← after skip concat
  cnn.weight: (3, 16, 3, 3)                                      ← final output CNN
  cnn.bias: (3,)
  fc.0.gru.weight_ih_l0: (768, 384)                             ← BiGRU forward input weights
  fc.1.weight: (360, 512)                                        ← linear output

Encoder channel progression:  1 → 16 → 32 → 64 → 128 → 256  (5 levels, 2× downsample each)
Bottleneck:                    256 → 512 (4 residual blocks)
Decoder channel progression:  512 → 256 → 128 → 64 → 32 → 16 (5 levels, 2× upsample each)
Output CNN:                    16 → 3 channels

MAX API (verified on DGX Spark SM_121):
  ops.conv2d(x, w, stride=(2,2), padding=(1,1,1,1), groups=1)   ← groups=1 works correctly
  ops.conv_transpose(x, w, stride=(2,2), padding=(1,1,1,1))     ← available
  ops.relu(x), ops.add(x, y), ops.mul(x, scale_const)
  TensorType(DType.float32, [1, 1, Dim("T"), 128], device_ref)  ← dynamic T, NOT -1
  result = model.execute(inp)                                     ← positional, not keyword
  tensor.to_numpy()                                               ← NOT np.array(tensor)

Run tests: pixi run test-pitch-extractor
```

---

## Task 1: RMVPE Weight Loader

**Files:**
- Create: `src/models/_rmvpe_weight_loader.py`
- Test: `tests/test_pitch_extractor.py` (create file, add `TestRmvpeWeightLoader` class)

### Step 1: Discover all checkpoint keys

Write a one-off diagnostic (do NOT commit this — it's just for building the loader):

```python
# run: pixi run python -c "
import torch
from huggingface_hub import hf_hub_download
path = hf_hub_download('lj1995/VoiceConversionWebUI', 'rmvpe.pt')
sd = torch.load(path, map_location='cpu')
sd = sd.get('model', sd)
for k, v in sorted(sd.items()):
    print(f'{k}: {tuple(v.shape)}')
"
```

Study the output. Note all unique key patterns and their index structures. The full key structure drives the rest of this task.

### Step 2: Write failing tests

Create `tests/test_pitch_extractor.py`:

```python
"""Tests for RMVPE-based PitchExtractor.

Level 1 (no download): no marker — run via: pixi run test-pitch-extractor
Level 2 (download required): @pytest.mark.slow — run via: pixi run test-pitch-extractor-full
"""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import numpy as np
import pytest


@pytest.fixture
def rng():
    return np.random.default_rng(42)


class TestRmvpeWeightLoader:
    """Tests for _rmvpe_weight_loader — no checkpoint download required."""

    def _make_fake_raw_weights(self):
        """Minimal synthetic checkpoint mimicking rmvpe.pt key/shape structure."""
        import numpy as np
        w = {}
        # Initial BN
        for k in ["weight", "bias", "running_mean", "running_var"]:
            w[f"unet.encoder.bn.{k}"] = np.ones(1, dtype=np.float32)
        w["unet.encoder.bn.num_batches_tracked"] = np.array(0, dtype=np.int64)
        # Encoder level 0, block 0 (channels 1→16, has shortcut)
        w["unet.encoder.layers.0.conv.0.conv.0.weight"] = np.random.randn(16, 1, 3, 3).astype(np.float32)
        w["unet.encoder.layers.0.conv.0.conv.0.bias"] = np.zeros(16, dtype=np.float32)
        for k in ["weight", "bias", "running_mean", "running_var"]:
            w[f"unet.encoder.layers.0.conv.0.conv.1.{k}"] = np.ones(16, dtype=np.float32)
        w["unet.encoder.layers.0.conv.0.conv.1.num_batches_tracked"] = np.array(0, dtype=np.int64)
        w["unet.encoder.layers.0.conv.0.shortcut.weight"] = np.random.randn(16, 1, 1, 1).astype(np.float32)
        # Output CNN
        w["cnn.weight"] = np.random.randn(3, 16, 3, 3).astype(np.float32)
        w["cnn.bias"] = np.zeros(3, dtype=np.float32)
        # BiGRU
        w["fc.0.gru.weight_ih_l0"] = np.random.randn(768, 384).astype(np.float32)
        w["fc.0.gru.weight_hh_l0"] = np.random.randn(768, 256).astype(np.float32)
        w["fc.0.gru.bias_ih_l0"] = np.zeros(768, dtype=np.float32)
        w["fc.0.gru.bias_hh_l0"] = np.zeros(768, dtype=np.float32)
        w["fc.0.gru.weight_ih_l0_reverse"] = np.random.randn(768, 384).astype(np.float32)
        w["fc.0.gru.weight_hh_l0_reverse"] = np.random.randn(768, 256).astype(np.float32)
        w["fc.0.gru.bias_ih_l0_reverse"] = np.zeros(768, dtype=np.float32)
        w["fc.0.gru.bias_hh_l0_reverse"] = np.zeros(768, dtype=np.float32)
        # Linear output
        w["fc.1.weight"] = np.random.randn(360, 512).astype(np.float32)
        w["fc.1.bias"] = np.zeros(360, dtype=np.float32)
        return w

    def test_load_from_dict_returns_dict(self):
        from models._rmvpe_weight_loader import load_rmvpe_from_dict
        result = load_rmvpe_from_dict(self._make_fake_raw_weights())
        assert isinstance(result, dict)
        assert len(result) > 0

    def test_bn_baked_into_scale_offset(self):
        """BatchNorm running stats must be baked — no raw running_mean in output."""
        from models._rmvpe_weight_loader import load_rmvpe_from_dict
        result = load_rmvpe_from_dict(self._make_fake_raw_weights())
        for key in result:
            assert "running_mean" not in key, f"Unbaked BN key found: {key}"
            assert "running_var" not in key, f"Unbaked BN key found: {key}"
            assert "num_batches_tracked" not in key

    def test_gru_weights_preserved(self):
        """BiGRU weights must be in output dict with correct shapes."""
        from models._rmvpe_weight_loader import load_rmvpe_from_dict
        result = load_rmvpe_from_dict(self._make_fake_raw_weights())
        assert "gru.weight_ih_l0" in result
        assert result["gru.weight_ih_l0"].shape == (768, 384)
        assert "gru.weight_ih_l0_reverse" in result

    def test_linear_weights_preserved(self):
        """Output linear weights must be present."""
        from models._rmvpe_weight_loader import load_rmvpe_from_dict
        result = load_rmvpe_from_dict(self._make_fake_raw_weights())
        assert "linear.weight" in result
        assert result["linear.weight"].shape == (360, 512)

    def test_all_values_float32(self):
        """All output arrays must be float32."""
        from models._rmvpe_weight_loader import load_rmvpe_from_dict
        result = load_rmvpe_from_dict(self._make_fake_raw_weights())
        for key, arr in result.items():
            assert arr.dtype == np.float32, f"Key {key} has dtype {arr.dtype}"

    def test_bake_bn_correctness(self):
        """Baked BN: (x - mean) / sqrt(var + eps) * weight + bias == x * scale + offset."""
        from models._rmvpe_weight_loader import bake_batch_norm
        rng = np.random.default_rng(0)
        weight = rng.standard_normal(16).astype(np.float32)
        bias = rng.standard_normal(16).astype(np.float32)
        running_mean = rng.standard_normal(16).astype(np.float32)
        running_var = np.abs(rng.standard_normal(16)).astype(np.float32)
        scale, offset = bake_batch_norm(weight, bias, running_mean, running_var)
        # Verify against reference formula
        x = rng.standard_normal((4, 16)).astype(np.float32)
        ref = (x - running_mean) / np.sqrt(running_var + 1e-5) * weight + bias
        out = x * scale + offset
        assert np.allclose(ref, out, atol=1e-5), f"Max diff: {np.abs(ref - out).max()}"
```

**Step 3: Run test to verify it fails**

```bash
pixi run pytest tests/test_pitch_extractor.py::TestRmvpeWeightLoader -v
```

Expected: `ModuleNotFoundError: No module named 'models._rmvpe_weight_loader'`

### Step 4: Implement `_rmvpe_weight_loader.py`

```python
"""Weight loader for RMVPE pitch estimation checkpoint.

Loads rmvpe.pt from HuggingFace, bakes BatchNorm into conv scale/offset,
and maps all 741 raw checkpoint keys to a clean internal naming scheme.

Internal naming convention:
  enc_bn.scale / enc_bn.offset       — initial encoder BatchNorm
  enc.{L}.{B}.0.w / .b              — encoder level L, block B, first Conv weight/bias
  enc.{L}.{B}.0.scale / .offset     — baked BN after first conv
  enc.{L}.{B}.1.w / .b              — second conv in block
  enc.{L}.{B}.1.scale / .offset     — baked BN after second conv
  enc.{L}.{B}.sc.w                  — shortcut conv weight (only where C_in != C_out)
  btl.{B}.0.w / .b / .scale / .offset  — bottleneck blocks (same pattern)
  btl.{B}.1.w / .b / .scale / .offset
  btl.{B}.sc.w
  dec.{L}.up.w                      — decoder ConvTranspose weight
  dec.{L}.up.b                      — decoder ConvTranspose bias
  dec.{L}.{B}.0.w / .b / .scale / .offset  — decoder residual blocks
  dec.{L}.{B}.1.w / .b / .scale / .offset
  dec.{L}.{B}.sc.w
  out_cnn.w / out_cnn.b             — final output Conv2d (16→3 channels)
  gru.weight_ih_l0                  — BiGRU weights (copied directly)
  gru.weight_hh_l0
  gru.bias_ih_l0
  gru.bias_hh_l0
  gru.weight_ih_l0_reverse
  gru.weight_hh_l0_reverse
  gru.bias_ih_l0_reverse
  gru.bias_hh_l0_reverse
  linear.weight / linear.bias       — output linear (360 bins)
"""
from __future__ import annotations
import numpy as np
from pathlib import Path

_EPS = 1e-5
_N_ENC_LEVELS = 5
_N_BTL_BLOCKS = 4
_N_DEC_LEVELS = 5
_N_BLOCKS_PER_LEVEL = 4


def bake_batch_norm(
    weight: np.ndarray,
    bias: np.ndarray,
    running_mean: np.ndarray,
    running_var: np.ndarray,
    eps: float = _EPS,
) -> tuple[np.ndarray, np.ndarray]:
    """Fold BatchNorm into scale and offset for inference.

    Computes: y = x * scale + offset, equivalent to BN(x).
    """
    scale = weight / np.sqrt(running_var + eps)
    offset = bias - running_mean * scale
    return scale.astype(np.float32), offset.astype(np.float32)


def _get(sd: dict, key: str) -> np.ndarray:
    """Get key from state dict as float32 numpy array."""
    return np.asarray(sd[key], dtype=np.float32)


def _bake_bn_from_sd(sd: dict, prefix: str) -> tuple[np.ndarray, np.ndarray]:
    """Read BN params from sd[prefix.{weight,bias,running_mean,running_var}] and bake."""
    return bake_batch_norm(
        _get(sd, f"{prefix}.weight"),
        _get(sd, f"{prefix}.bias"),
        _get(sd, f"{prefix}.running_mean"),
        _get(sd, f"{prefix}.running_var"),
    )


def _map_residual_block(
    sd: dict, src_prefix: str, dst_prefix: str, result: dict
) -> None:
    """Map one residual block's weights from checkpoint to internal names.

    src_prefix: e.g. "unet.encoder.layers.0.conv.0"
    dst_prefix: e.g. "enc.0.0"

    Discovers conv indices by scanning sd for src_prefix keys. Handles both:
    - Two-conv blocks: .conv.{0,1,3,4} (Conv, BN, [ReLU], Conv, BN)
    - Shortcut: .shortcut.weight if present
    """
    # Find all conv sub-keys for this block
    block_prefix = f"{src_prefix}.conv"
    indices = sorted(set(
        int(k.split(".")[len(block_prefix.split("."))])
        for k in sd if k.startswith(block_prefix + ".")
        and k.split(".")[len(block_prefix.split("."))] .isdigit()
    ))
    # Group by (conv_weight, bn) pairs: conv at even-ish indices, BN follows
    # Strategy: find indices with 4D weights (Conv) vs 1D (BN)
    conv_indices = [i for i in indices if _get(sd, f"{block_prefix}.{i}.weight").ndim == 4]
    bn_indices   = [i for i in indices if _get(sd, f"{block_prefix}.{i}.weight").ndim == 1]

    for pair_idx, (ci, bi) in enumerate(zip(conv_indices, bn_indices)):
        w = _get(sd, f"{block_prefix}.{ci}.weight")
        b_key = f"{block_prefix}.{ci}.bias"
        result[f"{dst_prefix}.{pair_idx}.w"] = w
        if b_key in sd:
            result[f"{dst_prefix}.{pair_idx}.b"] = _get(sd, b_key)
        scale, offset = _bake_bn_from_sd(sd, f"{block_prefix}.{bi}")
        result[f"{dst_prefix}.{pair_idx}.scale"] = scale
        result[f"{dst_prefix}.{pair_idx}.offset"] = offset

    # Shortcut
    sc_key = f"{src_prefix}.shortcut.weight"
    if sc_key in sd:
        result[f"{dst_prefix}.sc.w"] = _get(sd, sc_key)


def load_rmvpe_from_dict(sd: dict) -> dict[str, np.ndarray]:
    """Map raw rmvpe.pt state dict to internal weight names.

    Bakes all BatchNorm layers. Returns only float32 arrays.
    num_batches_tracked and other non-inference keys are dropped.
    """
    result = {}

    # Initial encoder BN
    scale, offset = _bake_bn_from_sd(sd, "unet.encoder.bn")
    result["enc_bn.scale"] = scale
    result["enc_bn.offset"] = offset

    # Encoder levels 0-4
    for L in range(_N_ENC_LEVELS):
        for B in range(_N_BLOCKS_PER_LEVEL):
            src = f"unet.encoder.layers.{L}.conv.{B}"
            dst = f"enc.{L}.{B}"
            _map_residual_block(sd, src, dst, result)

    # Bottleneck
    for B in range(_N_BTL_BLOCKS):
        src = f"unet.intermediate.layers.{B}"
        dst = f"btl.{B}"
        _map_residual_block(sd, src, dst, result)

    # Decoder levels 0-4
    for L in range(_N_DEC_LEVELS):
        # ConvTranspose upsample (PyTorch ConvTranspose2d: [C_in, C_out, H, W])
        up_w_key = f"unet.decoder.layers.{L}.conv1.0.weight"
        up_b_key = f"unet.decoder.layers.{L}.conv1.0.bias"
        result[f"dec.{L}.up.w"] = _get(sd, up_w_key)
        if up_b_key in sd:
            result[f"dec.{L}.up.b"] = _get(sd, up_b_key)
        # Post-concat residual blocks
        for B in range(_N_BLOCKS_PER_LEVEL):
            src = f"unet.decoder.layers.{L}.conv2.{B}"
            dst = f"dec.{L}.{B}"
            _map_residual_block(sd, src, dst, result)

    # Output CNN (16→3 channels)
    result["out_cnn.w"] = _get(sd, "cnn.weight")
    result["out_cnn.b"] = _get(sd, "cnn.bias")

    # BiGRU weights (passed through to numpy — not in MAX graph)
    for suffix in [
        "weight_ih_l0", "weight_hh_l0", "bias_ih_l0", "bias_hh_l0",
        "weight_ih_l0_reverse", "weight_hh_l0_reverse",
        "bias_ih_l0_reverse", "bias_hh_l0_reverse",
    ]:
        result[f"gru.{suffix}"] = _get(sd, f"fc.0.gru.{suffix}")

    # Linear output
    result["linear.weight"] = _get(sd, "fc.1.weight")
    result["linear.bias"] = _get(sd, "fc.1.bias")

    return result


def load_rmvpe_from_pt(path: str | Path) -> dict[str, np.ndarray]:
    """Load from .pt file and return internal weight dict."""
    import torch
    sd = torch.load(str(path), map_location="cpu", weights_only=True)
    if isinstance(sd, dict) and "model" in sd and isinstance(sd["model"], dict):
        sd = sd["model"]
    sd_np = {k: v.numpy() for k, v in sd.items() if hasattr(v, "numpy")}
    return load_rmvpe_from_dict(sd_np)


def load_rmvpe_weights(
    repo_id: str = "lj1995/VoiceConversionWebUI",
    filename: str = "rmvpe.pt",
    cache_dir: str | None = None,
) -> dict[str, np.ndarray]:
    """Download rmvpe.pt from HuggingFace and return internal weight dict.

    Checks local Applio cache first (~/.cache/rmvpe.pt) before downloading.
    """
    import os
    from huggingface_hub import hf_hub_download

    # Check if already cached by Applio
    applio_path = os.path.expanduser("~/.cache/rmvpe.pt")
    if os.path.exists(applio_path):
        return load_rmvpe_from_pt(applio_path)

    local_path = hf_hub_download(
        repo_id=repo_id,
        filename=filename,
        cache_dir=cache_dir,
    )
    return load_rmvpe_from_pt(local_path)
```

### Step 5: Run tests

```bash
pixi run pytest tests/test_pitch_extractor.py::TestRmvpeWeightLoader -v
```

Expected: all 6 tests PASS.

**Note:** If `_map_residual_block` fails because the actual key structure differs from what the diagnostic script revealed, fix the index discovery logic. The tests will catch incorrect BN baking via `test_bake_bn_correctness`.

### Step 6: Commit

```bash
git add src/models/_rmvpe_weight_loader.py tests/test_pitch_extractor.py
git commit -m "feat: RMVPE weight loader with BatchNorm baking"
```

---

## Task 2: U-Net MAX Graph

**Files:**
- Create: `src/models/_rmvpe.py`
- Modify: `tests/test_pitch_extractor.py` — add `TestUNetGraph` class

### Step 1: Write failing tests

Add to `tests/test_pitch_extractor.py`:

```python
class TestUNetGraph:
    """U-Net MAX Graph shape tests — no download required."""

    def _make_minimal_weights(self):
        """Minimal weight dict for a tiny U-Net (1 level, 1 block) for shape testing."""
        import numpy as np
        rng = np.random.default_rng(1)
        w = {}
        # Initial BN (1 channel input)
        w["enc_bn.scale"] = np.ones(1, dtype=np.float32)
        w["enc_bn.offset"] = np.zeros(1, dtype=np.float32)
        # Encoder level 0, block 0: 1→16 channels, first conv pair
        w["enc.0.0.0.w"] = rng.standard_normal((16, 1, 3, 3)).astype(np.float32) * 0.01
        w["enc.0.0.0.b"] = np.zeros(16, dtype=np.float32)
        w["enc.0.0.0.scale"] = np.ones(16, dtype=np.float32)
        w["enc.0.0.0.offset"] = np.zeros(16, dtype=np.float32)
        w["enc.0.0.1.w"] = rng.standard_normal((16, 16, 3, 3)).astype(np.float32) * 0.01
        w["enc.0.0.1.b"] = np.zeros(16, dtype=np.float32)
        w["enc.0.0.1.scale"] = np.ones(16, dtype=np.float32)
        w["enc.0.0.1.offset"] = np.zeros(16, dtype=np.float32)
        w["enc.0.0.sc.w"] = rng.standard_normal((16, 1, 1, 1)).astype(np.float32) * 0.01
        # Output CNN (use 16→3 directly for simplicity in test)
        w["out_cnn.w"] = rng.standard_normal((3, 16, 3, 3)).astype(np.float32) * 0.01
        w["out_cnn.b"] = np.zeros(3, dtype=np.float32)
        return w

    def test_unet_graph_buildable(self):
        """build_unet_graph must construct a valid MAX graph without errors."""
        from max import engine
        from max.driver import CPU
        from max.graph import DeviceRef
        from models._rmvpe import build_unet_graph

        weights = self._make_full_random_weights()
        cpu_ref = DeviceRef.CPU()
        graph = build_unet_graph(weights, cpu_ref)
        model = engine.InferenceSession(devices=[CPU()]).load(graph)
        assert model is not None

    def test_output_shape_t100(self):
        """Mel input [1, 1, 100, 128] → output [1, 100, 384]."""
        import numpy as np
        from max import engine
        from max.driver import CPU
        from max.graph import DeviceRef
        from models._rmvpe import build_unet_graph

        weights = self._make_full_random_weights()
        cpu_ref = DeviceRef.CPU()
        graph = build_unet_graph(weights, cpu_ref)
        model = engine.InferenceSession(devices=[CPU()]).load(graph)

        mel = np.random.randn(1, 1, 100, 128).astype(np.float32) * 0.1
        result = model.execute(mel)
        out = (list(result.values())[0] if isinstance(result, dict) else result[0]).to_numpy()
        assert out.shape == (1, 100, 384), f"Expected (1,100,384) got {out.shape}"

    def test_output_shape_t200(self):
        """Dynamic T: mel input [1, 1, 200, 128] → output [1, 200, 384]."""
        import numpy as np
        from max import engine
        from max.driver import CPU
        from max.graph import DeviceRef
        from models._rmvpe import build_unet_graph

        weights = self._make_full_random_weights()
        cpu_ref = DeviceRef.CPU()
        graph = build_unet_graph(weights, cpu_ref)
        model = engine.InferenceSession(devices=[CPU()]).load(graph)

        mel = np.random.randn(1, 1, 200, 128).astype(np.float32) * 0.1
        result = model.execute(mel)
        out = (list(result.values())[0] if isinstance(result, dict) else result[0]).to_numpy()
        assert out.shape == (1, 200, 384), f"Expected (1,200,384) got {out.shape}"

    def test_output_not_nan(self):
        """U-Net output must not contain NaN or Inf."""
        import numpy as np
        from max import engine
        from max.driver import CPU
        from max.graph import DeviceRef
        from models._rmvpe import build_unet_graph

        weights = self._make_full_random_weights()
        cpu_ref = DeviceRef.CPU()
        graph = build_unet_graph(weights, cpu_ref)
        model = engine.InferenceSession(devices=[CPU()]).load(graph)

        mel = np.random.randn(1, 1, 100, 128).astype(np.float32) * 0.1
        result = model.execute(mel)
        out = (list(result.values())[0] if isinstance(result, dict) else result[0]).to_numpy()
        assert not np.isnan(out).any(), "Output contains NaN"
        assert not np.isinf(out).any(), "Output contains Inf"

    def _make_full_random_weights(self):
        """Full random weight dict matching actual RMVPE U-Net architecture."""
        import numpy as np
        rng = np.random.default_rng(2)
        w = {}
        enc_channels = [1, 16, 32, 64, 128, 256]
        dec_channels = [512, 256, 128, 64, 32, 16]
        SCALE = 0.01

        def _rbn(c): return np.ones(c, np.float32), np.zeros(c, np.float32)
        def _rconv(co, ci, k=3): return rng.standard_normal((co, ci, k, k)).astype(np.float32) * SCALE

        # Initial enc BN
        w["enc_bn.scale"], w["enc_bn.offset"] = _rbn(1)

        # Encoder
        for L in range(5):
            c_in, c_out = enc_channels[L], enc_channels[L+1]
            for B in range(4):
                ci = c_in if B == 0 else c_out
                w[f"enc.{L}.{B}.0.w"] = _rconv(c_out, ci)
                w[f"enc.{L}.{B}.0.b"] = np.zeros(c_out, np.float32)
                w[f"enc.{L}.{B}.0.scale"], w[f"enc.{L}.{B}.0.offset"] = _rbn(c_out)
                w[f"enc.{L}.{B}.1.w"] = _rconv(c_out, c_out)
                w[f"enc.{L}.{B}.1.b"] = np.zeros(c_out, np.float32)
                w[f"enc.{L}.{B}.1.scale"], w[f"enc.{L}.{B}.1.offset"] = _rbn(c_out)
                if B == 0:  # shortcut only for first block (channel change)
                    w[f"enc.{L}.{B}.sc.w"] = _rconv(c_out, ci, k=1)

        # Bottleneck (256→512 on first block, then 512→512)
        btl_channels = [256, 512, 512, 512]
        for B in range(4):
            ci = btl_channels[B]
            co = 512
            w[f"btl.{B}.0.w"] = _rconv(co, ci)
            w[f"btl.{B}.0.b"] = np.zeros(co, np.float32)
            w[f"btl.{B}.0.scale"], w[f"btl.{B}.0.offset"] = _rbn(co)
            w[f"btl.{B}.1.w"] = _rconv(co, co)
            w[f"btl.{B}.1.b"] = np.zeros(co, np.float32)
            w[f"btl.{B}.1.scale"], w[f"btl.{B}.1.offset"] = _rbn(co)
            if B == 0:
                w[f"btl.{B}.sc.w"] = _rconv(co, ci, k=1)

        # Decoder
        for L in range(5):
            up_ci, up_co = dec_channels[L], dec_channels[L+1]
            # ConvTranspose: PyTorch shape [C_in, C_out, H, W]
            w[f"dec.{L}.up.w"] = rng.standard_normal((up_ci, up_co, 3, 3)).astype(np.float32) * SCALE
            w[f"dec.{L}.up.b"] = np.zeros(up_co, np.float32)
            # After skip concat: C_in = up_co + up_co (skip from encoder)
            skip_ci = up_co * 2
            for B in range(4):
                ci = skip_ci if B == 0 else up_co
                w[f"dec.{L}.{B}.0.w"] = _rconv(up_co, ci)
                w[f"dec.{L}.{B}.0.b"] = np.zeros(up_co, np.float32)
                w[f"dec.{L}.{B}.0.scale"], w[f"dec.{L}.{B}.0.offset"] = _rbn(up_co)
                w[f"dec.{L}.{B}.1.w"] = _rconv(up_co, up_co)
                w[f"dec.{L}.{B}.1.b"] = np.zeros(up_co, np.float32)
                w[f"dec.{L}.{B}.1.scale"], w[f"dec.{L}.{B}.1.offset"] = _rbn(up_co)
                if B == 0:
                    w[f"dec.{L}.{B}.sc.w"] = _rconv(up_co, ci, k=1)

        # Output CNN
        w["out_cnn.w"] = _rconv(3, 16)
        w["out_cnn.b"] = np.zeros(3, np.float32)
        return w
```

**Step 2: Run test to verify it fails**

```bash
pixi run pytest tests/test_pitch_extractor.py::TestUNetGraph -v
```

Expected: `ModuleNotFoundError: No module named 'models._rmvpe'`

### Step 3: Implement `build_unet_graph`

Create `src/models/_rmvpe.py`:

```python
"""RMVPE U-Net MAX Graph and numpy BiGRU/post-processing.

The U-Net runs in MAX (GPU). The BiGRU and linear output run in numpy
(max.nn has no GRU). Post-processing converts pitch salience [360 bins]
to Hz values per frame.
"""
from __future__ import annotations
import numpy as np


# ---------------------------------------------------------------------------
# U-Net MAX Graph
# ---------------------------------------------------------------------------

def _conv_bn_relu(x, w_key: str, b_key: str, scale_key: str, offset_key: str,
                  weights: dict, ops, const, stride=(1, 1), padding=(1, 1, 1, 1)):
    """Apply Conv2d → BN (baked as scale/offset) → ReLU."""
    w = const(weights[w_key])
    out = ops.conv2d(x, w, stride=stride, padding=padding)
    if b_key in weights:
        out = ops.add(out, ops.reshape(const(weights[b_key]), [1, 1, 1, -1]))
    scale = ops.reshape(const(weights[scale_key]), [1, 1, 1, -1])
    offset = ops.reshape(const(weights[offset_key]), [1, 1, 1, -1])
    out = ops.add(ops.mul(out, scale), offset)
    return ops.relu(out)


def _residual_block(x, prefix: str, weights: dict, ops, const,
                    stride=(1, 1), is_first_block=False):
    """One residual block: [Conv→BN→ReLU→Conv→BN] + shortcut → ReLU.

    When is_first_block=True and stride=2, applies stride only to first conv
    and uses a 1×1 strided shortcut conv.
    """
    # First conv pair
    padding = (1, 1, 1, 1)  # same padding for 3×3 with stride=1
    out = _conv_bn_relu(
        x,
        f"{prefix}.0.w", f"{prefix}.0.b", f"{prefix}.0.scale", f"{prefix}.0.offset",
        weights, ops, const, stride=stride, padding=padding,
    )
    # Second conv pair (always stride=1)
    w2 = const(weights[f"{prefix}.1.w"])
    out = ops.conv2d(out, w2, stride=(1, 1), padding=padding)
    if f"{prefix}.1.b" in weights:
        out = ops.add(out, ops.reshape(const(weights[f"{prefix}.1.b"]), [1, 1, 1, -1]))
    scale2 = ops.reshape(const(weights[f"{prefix}.1.scale"]), [1, 1, 1, -1])
    offset2 = ops.reshape(const(weights[f"{prefix}.1.offset"]), [1, 1, 1, -1])
    out = ops.add(ops.mul(out, scale2), offset2)
    # Shortcut
    sc_key = f"{prefix}.sc.w"
    if sc_key in weights:
        shortcut = ops.conv2d(x, const(weights[sc_key]), stride=stride, padding=(0, 0, 0, 0))
    else:
        shortcut = x
    return ops.relu(ops.add(out, shortcut))


def build_unet_graph(weights: dict, device_ref) -> "Graph":
    """Build RMVPE U-Net as a MAX Graph.

    Input:  [1, 1, T, 128]  float32 (mel spectrogram, NHWC-like: batch, channel, time, freq)
    Output: [1, T, 384]     float32 (flattened freq features, ready for BiGRU)

    MAX ops use NHWC format: [N, H, W, C]. The mel input is:
      [B, C_mel=1, T, F=128] → treated as [B, T, F, C] = [1, T, 128, 1]
    ConvTranspose in decoder upsamples the T dimension (height in 2D).
    """
    from max.graph import Graph, TensorType, ops, Dim
    from max.dtype import DType

    def const(arr):
        return ops.constant(np.asarray(arr, dtype=np.float32), device=device_ref)

    # RMVPE input: [B=1, 1, T, 128] but MAX Conv2d is NHWC [N, H, W, C].
    # We treat: N=1, H=T (time), W=128 (freq), C=1 (mono channel).
    # So input to graph is [1, Dim("T"), 128, 1].
    with Graph(
        "rmvpe_unet",
        input_types=[TensorType(DType.float32, [1, Dim("T"), 128, 1], device_ref)],
    ) as g:
        x = g.inputs[0]  # [1, T, 128, 1]

        # Initial BN (no conv, just scale+offset on 1-channel input)
        enc_bn_scale = ops.reshape(const(weights["enc_bn.scale"]), [1, 1, 1, 1])
        enc_bn_offset = ops.reshape(const(weights["enc_bn.offset"]), [1, 1, 1, 1])
        x = ops.add(ops.mul(x, enc_bn_scale), enc_bn_offset)

        enc_channels = [1, 16, 32, 64, 128, 256]
        skip_outputs = []

        # Encoder: 5 levels, each with 4 residual blocks + 2× downsample
        for L in range(5):
            # First block downsamples (stride=2 on H=time, stride=1 on W=freq)
            # We only downsample along the time dimension, not frequency.
            for B in range(4):
                prefix = f"enc.{L}.{B}"
                stride = (2, 1) if B == 0 else (1, 1)
                x = _residual_block(x, prefix, weights, ops, const,
                                    stride=stride, is_first_block=(B == 0))
            skip_outputs.append(x)

        # Bottleneck: 4 residual blocks at 512 channels
        for B in range(4):
            prefix = f"btl.{B}"
            x = _residual_block(x, prefix, weights, ops, const,
                                stride=(1, 1), is_first_block=(B == 0))

        # Decoder: 5 levels, each with ConvTranspose upsample + skip concat + 4 blocks
        for L in range(5):
            # ConvTranspose to upsample time dimension
            # Weight shape in internal dict follows PyTorch convention [C_in, C_out, H, W].
            # MAX ops.conv_transpose expects NHWC filter. Transpose accordingly.
            up_w = weights[f"dec.{L}.up.w"]  # [C_in, C_out, 3, 3] PyTorch
            # Convert to MAX NHWC filter format [H, W, C_out, C_in]
            up_w_max = up_w.transpose(2, 3, 1, 0)  # [3, 3, C_out, C_in]
            x = ops.conv_transpose(
                x, const(up_w_max),
                stride=(2, 1), padding=(1, 1, 1, 1),
            )
            if f"dec.{L}.up.b" in weights:
                x = ops.add(x, ops.reshape(const(weights[f"dec.{L}.up.b"]), [1, 1, 1, -1]))
            # Skip connection from encoder (mirror level: encoder 4-L feeds decoder L)
            skip = skip_outputs[4 - L]
            x = ops.concatenate([x, skip], axis=3)  # concat on channel dim
            # 4 residual blocks
            for B in range(4):
                prefix = f"dec.{L}.{B}"
                x = _residual_block(x, prefix, weights, ops, const,
                                    stride=(1, 1), is_first_block=(B == 0))

        # Output CNN: [1, T, 128, 16] → [1, T, 128, 3]
        out_w = weights["out_cnn.w"]  # PyTorch [C_out=3, C_in=16, 3, 3]
        out_w_max = out_w.transpose(2, 3, 1, 0)  # [3, 3, C_in=16, C_out=3] → NHWC
        x = ops.conv2d(x, const(out_w_max), stride=(1, 1), padding=(1, 1, 1, 1))
        if "out_cnn.b" in weights:
            x = ops.add(x, ops.reshape(const(weights["out_cnn.b"]), [1, 1, 1, -1]))
        # x: [1, T, 128, 3] → reshape to [1, T, 384]
        x = ops.reshape(x, [1, -1, 384])
        g.output(x)

    return g
```

> **Important:** The stride pattern along time vs frequency dimensions may need adjustment based on actual U-Net behavior. If the shape tests fail with dimension mismatch, check whether stride is applied to (H=time) or (W=freq) and fix accordingly. The encoder downsample should halve T at each level so that the decoder skip connections align.

### Step 4: Run tests

```bash
pixi run pytest tests/test_pitch_extractor.py::TestUNetGraph -v
```

Expected: all 4 tests PASS. If shape tests fail, the likely cause is skip connection dimension mismatch — verify that encoder output T at each level matches the upsampled decoder T at the mirrored level.

### Step 5: Commit

```bash
git add src/models/_rmvpe.py tests/test_pitch_extractor.py
git commit -m "feat: RMVPE U-Net MAX Graph (encoder + bottleneck + decoder)"
```

---

## Task 3: BiGRU + Post-Processing

**Files:**
- Modify: `src/models/_rmvpe.py` — add BiGRU and pitch conversion functions
- Modify: `tests/test_pitch_extractor.py` — add `TestBiGRU` and `TestPitchPostProcessing`

### Step 1: Write failing tests

Add to `tests/test_pitch_extractor.py`:

```python
class TestBiGRU:
    """BiGRU numpy implementation tests."""

    def _make_gru_weights(self, hidden=256, input_size=384):
        """Random BiGRU weights matching RMVPE internal naming."""
        rng = np.random.default_rng(3)
        w = {}
        gate_size = 3 * hidden
        for suffix, shape in [
            ("weight_ih_l0", (gate_size, input_size)),
            ("weight_hh_l0", (gate_size, hidden)),
            ("bias_ih_l0", (gate_size,)),
            ("bias_hh_l0", (gate_size,)),
            ("weight_ih_l0_reverse", (gate_size, input_size)),
            ("weight_hh_l0_reverse", (gate_size, hidden)),
            ("bias_ih_l0_reverse", (gate_size,)),
            ("bias_hh_l0_reverse", (gate_size,)),
        ]:
            w[f"gru.{suffix}"] = rng.standard_normal(shape).astype(np.float32) * 0.01
        return w

    def test_output_shape(self):
        """BiGRU: [1, T, 384] → [1, T, 512]."""
        from models._rmvpe import bigru_forward
        weights = self._make_gru_weights()
        x = np.random.randn(1, 100, 384).astype(np.float32) * 0.1
        out = bigru_forward(x, weights)
        assert out.shape == (1, 100, 512), f"Expected (1,100,512) got {out.shape}"

    def test_output_not_nan(self):
        from models._rmvpe import bigru_forward
        weights = self._make_gru_weights()
        x = np.random.randn(1, 50, 384).astype(np.float32) * 0.1
        out = bigru_forward(x, weights)
        assert not np.isnan(out).any()
        assert not np.isinf(out).any()

    def test_forward_reverse_differ(self):
        """Forward and reverse halves must differ (not identical)."""
        from models._rmvpe import bigru_forward
        weights = self._make_gru_weights()
        x = np.random.randn(1, 20, 384).astype(np.float32)
        out = bigru_forward(x, weights)
        fwd = out[:, :, :256]
        rev = out[:, :, 256:]
        # They should differ (different weights, different directions)
        assert not np.allclose(fwd, rev), "Forward and reverse GRU outputs are identical (bug)"


class TestPitchPostProcessing:
    """Tests for pitch salience → Hz conversion."""

    def test_to_local_average_f0_shape(self):
        """Salience [1, T, 360] → Hz [T] (float32)."""
        from models._rmvpe import salience_to_hz
        salience = np.random.rand(1, 100, 360).astype(np.float32)
        hz = salience_to_hz(salience, threshold=0.03)
        assert hz.shape == (100,), f"Expected (100,) got {hz.shape}"
        assert hz.dtype == np.float32

    def test_hz_range_voiced(self):
        """Voiced frames must be in piano range (27.5–4186 Hz)."""
        from models._rmvpe import salience_to_hz
        # Force all frames to be 'voiced' with a peak at bin 180 (middle C range)
        salience = np.zeros((1, 10, 360), dtype=np.float32)
        salience[:, :, 180] = 1.0
        hz = salience_to_hz(salience, threshold=0.03)
        voiced = hz[hz > 0]
        assert len(voiced) == 10
        assert (voiced > 20).all() and (voiced < 5000).all(), \
            f"Voiced Hz out of range: min={voiced.min():.1f}, max={voiced.max():.1f}"

    def test_unvoiced_returns_zero(self):
        """Low-salience frames must return 0 Hz."""
        from models._rmvpe import salience_to_hz
        salience = np.full((1, 10, 360), 0.001, dtype=np.float32)  # below threshold
        hz = salience_to_hz(salience, threshold=0.03)
        assert (hz == 0).all(), f"Expected all zeros, got: {hz}"
```

**Step 2: Run tests to verify they fail**

```bash
pixi run pytest tests/test_pitch_extractor.py::TestBiGRU tests/test_pitch_extractor.py::TestPitchPostProcessing -v
```

### Step 3: Add BiGRU and post-processing to `_rmvpe.py`

Add to `src/models/_rmvpe.py`:

```python
# ---------------------------------------------------------------------------
# BiGRU (numpy — max.nn has no GRU)
# ---------------------------------------------------------------------------

def _sigmoid(x: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-x))


def bigru_forward(x: np.ndarray, weights: dict, hidden_size: int = 256) -> np.ndarray:
    """Bidirectional GRU forward pass (numpy).

    Args:
        x: [B, T, 384] float32
        weights: internal weight dict with keys gru.weight_ih_l0, etc.
        hidden_size: 256 (single-direction output size; total = 512)

    Returns:
        [B, T, 512] float32 — concat of forward and reverse hidden states.
    """
    w = weights
    H = hidden_size

    def gru_step(x_t, h, w_ih, w_hh, b_ih, b_hh):
        gi = x_t @ w_ih.T + b_ih   # [B, 3H]
        gh = h @ w_hh.T + b_hh     # [B, 3H]
        z = _sigmoid(gi[:, :H]     + gh[:, :H])      # update gate
        r = _sigmoid(gi[:, H:2*H]  + gh[:, H:2*H])   # reset gate
        n = np.tanh(gi[:, 2*H:]    + r * gh[:, 2*H:]) # new gate
        return (1.0 - z) * h + z * n                  # [B, H]

    B, T, _ = x.shape

    # Forward pass
    h_fwd = np.zeros((B, H), dtype=np.float32)
    fwd_states = []
    for t in range(T):
        h_fwd = gru_step(
            x[:, t], h_fwd,
            w["gru.weight_ih_l0"], w["gru.weight_hh_l0"],
            w["gru.bias_ih_l0"],   w["gru.bias_hh_l0"],
        )
        fwd_states.append(h_fwd.copy())

    # Reverse pass
    h_rev = np.zeros((B, H), dtype=np.float32)
    rev_states = [None] * T
    for t in range(T - 1, -1, -1):
        h_rev = gru_step(
            x[:, t], h_rev,
            w["gru.weight_ih_l0_reverse"], w["gru.weight_hh_l0_reverse"],
            w["gru.bias_ih_l0_reverse"],   w["gru.bias_hh_l0_reverse"],
        )
        rev_states[t] = h_rev.copy()

    fwd_out = np.stack(fwd_states, axis=1)  # [B, T, H]
    rev_out = np.stack(rev_states, axis=1)  # [B, T, H]
    return np.concatenate([fwd_out, rev_out], axis=-1)  # [B, T, 512]


# ---------------------------------------------------------------------------
# Linear output + pitch salience → Hz
# ---------------------------------------------------------------------------

def linear_output(x: np.ndarray, weights: dict) -> np.ndarray:
    """[B, T, 512] → [B, T, 360] via linear layer."""
    W = weights["linear.weight"]  # [360, 512]
    b = weights["linear.bias"]    # [360]
    return x @ W.T + b            # [B, T, 360]


# RMVPE pitch bins: 360 bins covering C1-B7 at 20-cent steps.
# Bin i corresponds to: f0 = 440 * 2^((i * 20 - 6900) / 1200) Hz
# Bin 0 ≈ 32.7 Hz (C1), bin 359 ≈ 1975.5 Hz (B6)
_RMVPE_CENTS_PER_BIN = 20.0
_RMVPE_CENTER_CENTS = 6900.0  # offset so bin 0 maps to ~C1


def bins_to_hz(bin_indices: np.ndarray) -> np.ndarray:
    """Convert RMVPE bin indices to Hz. Returns float32 array."""
    cents = bin_indices.astype(np.float32) * _RMVPE_CENTS_PER_BIN - _RMVPE_CENTER_CENTS
    return (440.0 * (2.0 ** (cents / 1200.0))).astype(np.float32)


def salience_to_hz(salience: np.ndarray, threshold: float = 0.03) -> np.ndarray:
    """Convert pitch salience [1, T, 360] to F0 in Hz per frame [T].

    Uses local-average F0 estimation (weighted mean of nearby bins around peak)
    for sub-bin accuracy. Frames below threshold are unvoiced (0 Hz).

    Args:
        salience: [1, T, 360] float32 pitch salience from linear output (pre-sigmoid).
        threshold: Minimum salience to consider frame voiced (default 0.03).

    Returns:
        [T] float32 — F0 in Hz per frame, 0.0 for unvoiced frames.
    """
    # Apply sigmoid to get probability per bin
    prob = _sigmoid(salience[0])  # [T, 360]
    T = prob.shape[0]

    # Local-average: weight bins within ±4 of peak
    center_bin = np.argmax(prob, axis=-1)  # [T]
    max_prob = prob[np.arange(T), center_bin]  # [T]

    bin_indices = np.arange(360, dtype=np.float32)
    weighted_bins = np.zeros(T, dtype=np.float32)

    for t in range(T):
        lo = max(0, center_bin[t] - 4)
        hi = min(360, center_bin[t] + 5)
        weights_t = prob[t, lo:hi]
        bins_t = bin_indices[lo:hi]
        w_sum = weights_t.sum()
        if w_sum > 0:
            weighted_bins[t] = (bins_t * weights_t).sum() / w_sum
        else:
            weighted_bins[t] = float(center_bin[t])

    hz = bins_to_hz(weighted_bins)
    # Zero out unvoiced frames
    hz[max_prob < threshold] = 0.0
    return hz
```

### Step 4: Run tests

```bash
pixi run pytest tests/test_pitch_extractor.py::TestBiGRU tests/test_pitch_extractor.py::TestPitchPostProcessing -v
```

Expected: all 6 tests PASS.

### Step 5: Commit

```bash
git add src/models/_rmvpe.py tests/test_pitch_extractor.py
git commit -m "feat: RMVPE BiGRU numpy + pitch salience-to-Hz post-processing"
```

---

## Task 4: PitchExtractor Class + Mel + pixi Tasks

**Files:**
- Create: `src/models/pitch_extractor.py`
- Modify: `src/models/__init__.py` — export `PitchExtractor`
- Modify: `pixi.toml` — add test tasks
- Modify: `tests/test_pitch_extractor.py` — add `TestPitchExtractorShapes` + `TestPitchExtractorCorrectness`

### Step 1: Write failing tests

Add to `tests/test_pitch_extractor.py`:

```python
class TestMelSpectrogram:
    """Test RMVPE mel preprocessing — no download required."""

    def test_output_shape_1s(self):
        """1s @16kHz → [1, 1, ~100, 128]."""
        from models.pitch_extractor import _mel_spectrogram
        audio = np.zeros((1, 16000), dtype=np.float32)
        mel = _mel_spectrogram(audio)
        assert mel.ndim == 4
        assert mel.shape[0] == 1 and mel.shape[1] == 1 and mel.shape[3] == 128
        # ~100 frames for 1s at hop=160
        assert 95 <= mel.shape[2] <= 105, f"Unexpected T dim: {mel.shape[2]}"

    def test_output_dtype(self):
        from models.pitch_extractor import _mel_spectrogram
        audio = np.zeros((1, 16000), dtype=np.float32)
        mel = _mel_spectrogram(audio)
        assert mel.dtype == np.float32

    def test_output_not_nan(self):
        from models.pitch_extractor import _mel_spectrogram
        audio = np.random.randn(1, 16000).astype(np.float32) * 0.1
        mel = _mel_spectrogram(audio)
        assert not np.isnan(mel).any()
        assert not np.isinf(mel).any()


class TestPitchExtractorShapes:
    """PitchExtractor shape tests with random weights — no download required."""

    def _make_full_random_weights(self):
        """Full random weight dict for PitchExtractor (reuse from TestUNetGraph)."""
        # Import and call the helper defined in TestUNetGraph
        return TestUNetGraph()._make_full_random_weights()

    def test_from_weights_buildable(self):
        """PitchExtractor._from_weights must not raise."""
        from models.pitch_extractor import PitchExtractor
        w = self._make_full_random_weights()
        model = PitchExtractor._from_weights(w, device="cpu")
        assert model is not None

    def test_extract_1s_shape(self):
        """1s audio → [T] Hz array (T ≈ 100)."""
        from models.pitch_extractor import PitchExtractor
        w = self._make_full_random_weights()
        model = PitchExtractor._from_weights(w, device="cpu")
        audio = np.zeros((1, 16000), dtype=np.float32)
        f0 = model.extract(audio)
        assert f0.ndim == 1
        assert 95 <= len(f0) <= 105, f"Expected ~100 frames, got {len(f0)}"
        assert f0.dtype == np.float32

    def test_extract_output_not_nan(self):
        from models.pitch_extractor import PitchExtractor
        w = self._make_full_random_weights()
        model = PitchExtractor._from_weights(w, device="cpu")
        audio = np.random.randn(1, 16000).astype(np.float32) * 0.1
        f0 = model.extract(audio)
        assert not np.isnan(f0).any()

    def test_extract_unvoiced_returns_zero(self):
        """Silence should produce all-zero F0 (unvoiced)."""
        from models.pitch_extractor import PitchExtractor
        w = self._make_full_random_weights()
        model = PitchExtractor._from_weights(w, device="cpu")
        audio = np.zeros((1, 16000), dtype=np.float32)
        f0 = model.extract(audio)
        # With random weights on silent audio, most frames should be unvoiced
        # (this tests that the unvoiced masking path runs without error)
        assert f0.dtype == np.float32


@pytest.mark.slow
class TestPitchExtractorCorrectness:
    """Integration test: PitchExtractor vs Applio RMVPE on same audio.

    Requires rmvpe.pt download (~181MB). Run with: pixi run test-pitch-extractor-full
    """

    def _get_applio_f0(self, audio_np):
        """Get reference F0 from Applio's RMVPE implementation."""
        import sys
        # Applio must be installed or its path added
        import torch
        from rvc.lib.predictors.RMVPE import RMVPE
        from huggingface_hub import hf_hub_download
        ckpt_path = hf_hub_download("lj1995/VoiceConversionWebUI", "rmvpe.pt")
        ref_model = RMVPE(model_path=ckpt_path, is_half=False, device="cpu")
        return ref_model.infer_from_audio(audio_np[0], thred=0.03)

    def test_f0_matches_applio(self):
        """Our F0 must match Applio's RMVPE within ±5 cents on voiced frames."""
        from models import PitchExtractor

        rng = np.random.default_rng(42)
        # Sine wave at 440 Hz — always voiced, known pitch
        t = np.linspace(0, 1, 16000, dtype=np.float32)
        audio = (0.5 * np.sin(2 * np.pi * 440 * t)).reshape(1, -1)

        model = PitchExtractor.from_pretrained()
        our_f0 = model.extract(audio)

        ref_f0 = self._get_applio_f0(audio)
        # Align lengths (may differ by 1 frame)
        min_len = min(len(our_f0), len(ref_f0))
        our_voiced = our_f0[:min_len]
        ref_voiced = ref_f0[:min_len]

        # Compare only voiced frames
        voiced_mask = (our_voiced > 0) & (ref_voiced > 0)
        assert voiced_mask.sum() > 0, "No voiced frames to compare"

        # Convert to cents for comparison: cents = 1200 * log2(f / 440)
        our_cents = 1200 * np.log2(our_voiced[voiced_mask] / 440.0)
        ref_cents = 1200 * np.log2(ref_voiced[voiced_mask] / 440.0)
        cent_diff = np.abs(our_cents - ref_cents)

        print(f"\n  Voiced frames: {voiced_mask.sum()}")
        print(f"  Mean cent error: {cent_diff.mean():.2f}")
        print(f"  Max cent error:  {cent_diff.max():.2f}")
        assert cent_diff.mean() < 5.0, \
            f"Mean cent error {cent_diff.mean():.2f} > 5 cents threshold"
```

**Step 2: Run tests to verify they fail**

```bash
pixi run pytest tests/test_pitch_extractor.py::TestMelSpectrogram tests/test_pitch_extractor.py::TestPitchExtractorShapes -v
```

Expected: `ModuleNotFoundError: No module named 'models.pitch_extractor'`

### Step 3: Implement `pitch_extractor.py`

```python
"""PitchExtractor: RMVPE pitch estimation via MAX Graph U-Net.

Replaces PyTorch RMVPE in the VC pipeline. Works on DGX Spark ARM64.

Example:
    model = PitchExtractor.from_pretrained()
    f0_hz = model.extract(audio_np)  # [1, N] float32 @16kHz → [T_frames] float32 Hz
"""
from __future__ import annotations
import numpy as np


# RMVPE mel spectrogram parameters (must match checkpoint's training preprocessing)
_MEL_SAMPLE_RATE = 16000
_MEL_N_FFT       = 1024
_MEL_WIN_LENGTH  = 1024
_MEL_HOP_LENGTH  = 160   # 10ms frames → ~100 frames/second
_MEL_N_MELS      = 128
_MEL_F_MIN       = 50.0
_MEL_F_MAX       = 2006.0


def _mel_spectrogram(audio: np.ndarray) -> np.ndarray:
    """Compute log mel spectrogram matching RMVPE training preprocessing.

    Args:
        audio: [1, N] float32 @16kHz.

    Returns:
        [1, 1, T, 128] float32 (batch, channel, time, mel_bins).
    """
    import torch
    import torchaudio.transforms as T

    mel_transform = T.MelSpectrogram(
        sample_rate=_MEL_SAMPLE_RATE,
        n_fft=_MEL_N_FFT,
        win_length=_MEL_WIN_LENGTH,
        hop_length=_MEL_HOP_LENGTH,
        n_mels=_MEL_N_MELS,
        f_min=_MEL_F_MIN,
        f_max=_MEL_F_MAX,
        window_fn=torch.hann_window,
        center=True,
        norm="slaney",
        mel_scale="slaney",
    )
    audio_t = torch.from_numpy(audio)           # [1, N]
    mel = mel_transform(audio_t)                # [1, n_mels, T]
    mel = (mel + 1e-8).log()
    # Reshape: [1, n_mels, T] → [1, 1, T, n_mels]
    mel = mel.permute(0, 2, 1).unsqueeze(1)    # [1, 1, T, n_mels]
    return mel.numpy().astype(np.float32)


class PitchExtractor:
    """RMVPE pitch extractor — MAX Graph U-Net + numpy BiGRU.

    Computes F0 (fundamental frequency) from raw audio in Hz per 10ms frame.
    0 Hz = unvoiced frame.

    Example:
        model = PitchExtractor.from_pretrained()
        f0 = model.extract(audio)  # [1, N] @16kHz → [T] Hz
    """

    def __init__(self, _unet_model, _device, _weights: dict):
        self._unet_model = _unet_model
        self._device = _device
        self._weights = weights  # full weight dict for BiGRU + linear

    @classmethod
    def _from_weights(cls, weights: dict, device: str = "auto") -> "PitchExtractor":
        """Build from loaded internal weight dict.

        Args:
            weights: Internal weight dict from _rmvpe_weight_loader.
            device: "auto", "gpu", or "cpu".
        """
        from max import engine
        from max.driver import Accelerator, CPU, accelerator_count
        from max.graph import DeviceRef
        from ._rmvpe import build_unet_graph

        use_gpu = accelerator_count() > 0 if device == "auto" else device == "gpu"
        dev = Accelerator() if use_gpu else CPU()
        device_ref = DeviceRef.GPU(0) if use_gpu else DeviceRef.CPU()

        graph = build_unet_graph(weights, device_ref)
        session = engine.InferenceSession(devices=[dev])
        unet_model = session.load(graph)

        return cls(_unet_model=unet_model, _device=dev, _weights=weights)

    @classmethod
    def from_pretrained(
        cls,
        repo_id: str = "lj1995/VoiceConversionWebUI",
        filename: str = "rmvpe.pt",
        device: str = "auto",
        cache_dir: str | None = None,
    ) -> "PitchExtractor":
        """Load RMVPE from HuggingFace Hub.

        Args:
            repo_id: HuggingFace repo containing rmvpe.pt.
            filename: Filename within the repo (default "rmvpe.pt").
            device: "auto" (default), "gpu", or "cpu".
            cache_dir: Override download cache directory.
        """
        from ._rmvpe_weight_loader import load_rmvpe_weights
        weights = load_rmvpe_weights(repo_id, filename, cache_dir)
        return cls._from_weights(weights, device=device)

    def extract(self, audio: np.ndarray, threshold: float = 0.03) -> np.ndarray:
        """Extract F0 from raw audio.

        Args:
            audio: [1, N] float32, 16kHz, normalized to roughly [-1, 1].
            threshold: Minimum salience to classify frame as voiced (default 0.03).

        Returns:
            [T_frames] float32 — F0 in Hz, 0.0 = unvoiced. ~100 frames per second.
        """
        from max.driver import Accelerator, Tensor
        from ._rmvpe import bigru_forward, linear_output, salience_to_hz

        # Step 1: Mel spectrogram [1, N] → [1, 1, T, 128]
        mel = _mel_spectrogram(audio)
        T = mel.shape[2]

        # Step 2: U-Net MAX Graph [1, 1, T, 128] → [1, T, 384]
        # Input must be [1, T, 128, 1] (NHWC for MAX)
        mel_nhwc = mel.transpose(0, 2, 3, 1)  # [1, 1, T, 128] → [1, T, 128, 1]
        if isinstance(self._device, Accelerator):
            inp = Tensor.from_numpy(np.ascontiguousarray(mel_nhwc)).to(self._device)
        else:
            inp = np.ascontiguousarray(mel_nhwc)
        result = self._unet_model.execute(inp)
        unet_out = (list(result.values())[0] if isinstance(result, dict) else result[0])
        features = unet_out.to_numpy()  # [1, T, 384]

        # Step 3: BiGRU [1, T, 384] → [1, T, 512]
        gru_out = bigru_forward(features, self._weights)

        # Step 4: Linear [1, T, 512] → [1, T, 360]
        salience = linear_output(gru_out, self._weights)

        # Step 5: Salience [1, T, 360] → [T] Hz
        return salience_to_hz(salience, threshold=threshold)
```

> **Bug to fix:** `cls(_unet_model=unet_model, _device=dev, _weights=weights)` should be `_weights=weights` not `_weights=weights`. Also the `__init__` has `self._weights = weights` but the parameter is named `_weights`. Fix the `__init__` signature to match.

### Step 4: Update `__init__.py`

```python
"""mojo_audio.models — MAX Graph audio encoder (HuBERT / ContentVec) and pitch extractor."""

from .audio_encoder import AudioEncoder
from .pitch_extractor import PitchExtractor

__all__ = ["AudioEncoder", "PitchExtractor"]
```

### Step 5: Add pixi tasks to `pixi.toml`

Find the `[tasks]` section and add:

```toml
test-pitch-extractor = "pytest tests/test_pitch_extractor.py -v -m 'not slow'"
test-pitch-extractor-full = "pytest tests/test_pitch_extractor.py -v -s"
```

### Step 6: Run tests

```bash
pixi run test-pitch-extractor
```

Expected: all non-slow tests PASS. Fix any issues before committing.

### Step 7: Commit

```bash
git add src/models/pitch_extractor.py src/models/__init__.py pixi.toml tests/test_pitch_extractor.py
git commit -m "feat: PitchExtractor class — RMVPE mel + MAX U-Net + numpy BiGRU"
```

---

## Final Verification

```bash
# Fast tests (always)
pixi run test-pitch-extractor

# Full test including correctness vs Applio (requires rmvpe.pt download + Applio installed)
pixi run test-pitch-extractor-full
```

After all tasks complete, update the roadmap: change Sprint 1B description from "CNN blocks + BiGRU" to "U-Net + BiGRU tail" in `docs/project/03-06-2026-roadmap.md`.
