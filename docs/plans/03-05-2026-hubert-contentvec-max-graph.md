# HuBERT / ContentVec in MAX Graph — Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Implement HuBERT and ContentVec audio feature extraction as a Python module (`mojo_audio.models.AudioEncoder`) using MAX Graph ops, so the full voice conversion pipeline runs natively on DGX Spark SM_121 ARM64 with GPU acceleration.

**Architecture:** Pure `max.graph` low-level ops (no `max.nn` dependency risk). Conv1D implemented as Conv2D with H=1 (NHWC layout). Attention built from `ops.matmul` + `ops.softmax`. Weights loaded from HuggingFace safetensors or legacy `.pt` format, mapped via explicit name dict. Lives in `src/models/` as a Python package inside mojo-audio.

**Tech Stack:** Python 3.13, MAX Engine 26.1, `max.graph.ops` (conv2d, layer_norm, gelu, matmul, softmax, reshape, transpose), `huggingface_hub` (weight download), `safetensors` (weight loading), `pytest` (tests). All via `pixi run python`.

---

## Environment Context

- **Local machine:** `/home/maskkiller/dev-coffee/repos/mojo-audio` — RTX 4060 Ti, x86_64
- **DGX Spark:** `visage@visage-spark:/home/visage/repos/mojo-audio` — GB10 SM_121, aarch64
- **Run all Python with:** `pixi run python` (never bare `python3`)
- **Run tests with:** `pixi run pytest tests/test_audio_encoder.py -v`
- **Verified working ops on SM_121:** `ops.conv2d`, `ops.layer_norm`, `ops.gelu`, `ops.matmul`, `ops.softmax`, `ops.reshape`, `ops.transpose`, `ops.add`, `ops.constant`

## Ops API Quick Reference (verified v26.1)

```python
from max.graph import ops, DeviceRef
from max.dtype import DType

# Conv1D via Conv2D (treat sequence length as height, width=1)
# Input: [B, L, 1, C_in] (NHWC)   Filter: [K, 1, C_in/groups, C_out] (RSCF)
ops.conv2d(x, filter, stride=(s,1), padding=(pl,pr,0,0), groups=g)

# Layer norm
ops.layer_norm(x, gamma, beta, epsilon=1e-5)

# GELU (HuBERT uses standard, not approximate)
ops.gelu(x, approximate="none")

# Linear: matmul + optional add for bias
# Weight must be shape [in_features, out_features] for x=[B, T, in_features]
ops.matmul(x, weight_constant)

# Softmax (attention weights)
ops.softmax(x, axis=-1)

# Reshape: -1 allowed for one dynamic dim
ops.reshape(x, [B, -1, C])

# Transpose last two dims (for attention scores: [B,H,T,T])
ops.transpose(x, [-1, -2])

# Constant from numpy
ops.constant(np_array, device=device_ref)
```

---

## Task 1: Package Scaffold + pytest Setup

**Files:**
- Create: `src/models/__init__.py`
- Create: `src/models/audio_encoder.py` (stub only)
- Create: `src/models/_feature_extractor.py` (empty)
- Create: `src/models/_attention.py` (empty)
- Create: `src/models/_weight_loader.py` (empty)
- Create: `tests/test_audio_encoder.py`
- Modify: `pixi.toml` — add `pytest`, `safetensors` deps + test tasks

---

**Step 1: Add to `pixi.toml` under `[dependencies]`**

```toml
pytest = ">=8.0.0,<9"
safetensors = ">=0.4.0,<1"
```

Add under `[tasks]`:
```toml
test-models = "pytest tests/test_audio_encoder.py -v -m 'not slow'"
test-models-full = "pytest tests/test_audio_encoder.py -v"
bench-models = "python experiments/contentvec-max/benchmark.py"
```

**Step 2: Create `src/models/__init__.py`**

```python
"""mojo_audio.models — MAX Graph audio encoder (HuBERT / ContentVec)."""

from .audio_encoder import AudioEncoder

__all__ = ["AudioEncoder"]
```

**Step 3: Create `src/models/audio_encoder.py` stub**

```python
"""AudioEncoder: HuBERT / ContentVec feature extraction via MAX Graph."""

from __future__ import annotations
import numpy as np
from pathlib import Path


class AudioEncoder:
    """MAX Graph implementation of HuBERT / ContentVec audio encoder.

    Supports facebook/hubert-base-ls960 and lengyue233/content-vec-best.
    Automatically selects GPU if available, falls back to CPU.

    Example:
        model = AudioEncoder.from_pretrained("facebook/hubert-base-ls960")
        features = model.encode(audio_np)  # [1, seq] → [1, frames, 768]
    """

    def __init__(self, _model, _device):
        self._model = _model
        self._device = _device

    @classmethod
    def from_pretrained(
        cls,
        model_id: str,
        device: str = "auto",
        cache_dir: str | None = None,
    ) -> "AudioEncoder":
        """Load model from HuggingFace Hub or local path.

        Args:
            model_id: HuggingFace model ID or local path to .safetensors/.pt file.
            device: "auto" (default), "gpu", or "cpu".
            cache_dir: Override default cache (~/.cache/mojo-audio/models/).
        """
        raise NotImplementedError("Task 7 implements this")

    def encode(self, audio: np.ndarray) -> np.ndarray:
        """Encode raw audio waveform to feature vectors.

        Args:
            audio: Float32 numpy array, shape [1, samples], 16kHz, normalized [-1, 1].

        Returns:
            Float32 numpy array, shape [1, time_frames, 768].
            For 1s audio: [1, 49, 768].
        """
        raise NotImplementedError("Task 7 implements this")
```

**Step 4: Create `tests/test_audio_encoder.py`**

```python
"""Tests for mojo_audio.models.AudioEncoder.

Level 1 tests (no download): marked with no marker — run by default.
Level 2 tests (download required): marked @pytest.mark.slow — skipped by default.
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import numpy as np
import pytest
from max.driver import accelerator_count


# --- Fixtures ---

@pytest.fixture
def rng():
    return np.random.default_rng(42)


@pytest.fixture
def audio_1s(rng):
    """1 second of synthetic 16kHz audio, normalized."""
    return rng.standard_normal((1, 16000)).astype(np.float32)


@pytest.fixture
def cpu_device():
    from max.driver import CPU
    return CPU()


@pytest.fixture
def gpu_available():
    return accelerator_count() > 0


# --- Placeholder tests (will be filled per task) ---

def test_package_importable():
    """mojo_audio.models must be importable."""
    from models import AudioEncoder
    assert AudioEncoder is not None


def test_max_engine_importable():
    """MAX Engine must be accessible."""
    from max import engine
    from max.driver import accelerator_count
    assert True  # if we got here, imports work


def test_gpu_session_creatable():
    """GPU InferenceSession must work if GPU is available."""
    from max import engine
    from max.driver import Accelerator, CPU, accelerator_count
    if accelerator_count() > 0:
        session = engine.InferenceSession(devices=[Accelerator()])
    else:
        session = engine.InferenceSession(devices=[CPU()])
    assert session is not None
```

**Step 5: Run to verify structure is correct**

```bash
pixi run pytest tests/test_audio_encoder.py -v -m "not slow"
```

Expected:
```
test_package_importable PASSED
test_max_engine_importable PASSED
test_gpu_session_creatable PASSED
3 passed
```

**Step 6: Commit**

```bash
git add src/models/ tests/test_audio_encoder.py pixi.toml
git commit -m "chore: scaffold mojo_audio.models package and pytest setup"
```

---

## Task 2: Weight Loader (`_weight_loader.py`)

Maps HuggingFace weight keys → internal flat names. Supports `.safetensors` (HF format) and `.pt` (legacy RVC format). Auto-downloads from HuggingFace Hub.

**Files:**
- Implement: `src/models/_weight_loader.py`
- Test: `tests/test_audio_encoder.py` — add weight loader tests

---

**Step 1: Add weight loader tests (write first, before implementation)**

Add to `tests/test_audio_encoder.py`:

```python
class TestWeightLoader:
    """Tests for _weight_loader — no model download required."""

    def test_detect_hubert_prefix(self):
        """Detect 'hubert' prefix in HuBERT checkpoint keys."""
        from models._weight_loader import _detect_prefix
        keys = [
            "hubert.feature_extractor.conv_layers.0.conv.weight",
            "hubert.encoder.layers.0.attention.q_proj.weight",
        ]
        assert _detect_prefix(keys) == "hubert"

    def test_detect_contentvec_prefix(self):
        """Detect 'model' prefix in ContentVec checkpoint keys."""
        from models._weight_loader import _detect_prefix
        keys = [
            "model.feature_extractor.conv_layers.0.conv.weight",
            "model.encoder.layers.0.attention.q_proj.weight",
        ]
        assert _detect_prefix(keys) == "model"

    def test_map_single_weight(self):
        """Single weight key maps to correct internal name."""
        from models._weight_loader import _map_key
        result = _map_key("hubert.feature_extractor.conv_layers.3.conv.weight", "hubert")
        assert result == "cnn.3.weight"

    def test_map_transformer_layer_key(self):
        """Transformer layer key maps correctly."""
        from models._weight_loader import _map_key
        result = _map_key("hubert.encoder.layers.5.attention.q_proj.weight", "hubert")
        assert result == "blocks.5.attn.q.weight"

    def test_map_unknown_key_returns_none(self):
        """Unknown key returns None (will be skipped)."""
        from models._weight_loader import _map_key
        result = _map_key("some.unknown.weight", "hubert")
        assert result is None

    def test_load_from_dict(self):
        """load_weights_from_dict maps a synthetic weight dict."""
        from models._weight_loader import load_weights_from_dict
        fake_weights = {
            "hubert.feature_extractor.conv_layers.0.conv.weight": np.zeros((512, 1, 10), dtype=np.float32),
            "hubert.encoder.layers.0.attention.q_proj.weight": np.zeros((768, 768), dtype=np.float32),
        }
        result = load_weights_from_dict(fake_weights)
        assert "cnn.0.weight" in result
        assert "blocks.0.attn.q.weight" in result
```

**Step 2: Run to confirm they fail**

```bash
pixi run pytest tests/test_audio_encoder.py::TestWeightLoader -v
```

Expected: ImportError or AttributeError — `_weight_loader` doesn't exist yet.

**Step 3: Implement `src/models/_weight_loader.py`**

```python
"""Weight loading for HuBERT / ContentVec checkpoints.

Supports:
  - .safetensors (HuggingFace format)
  - .pt / .bin (legacy PyTorch, used by RVC's hubert_base.pt)

Internal naming convention (flat):
  cnn.{i}.weight / cnn.{i}.bias          — CNN feature extractor
  cnn.{i}.norm.weight / .bias            — CNN group norm
  proj.weight / proj.bias                — feature projection
  proj.norm.weight / proj.norm.bias      — feature projection layernorm
  pos_conv.weight / pos_conv.bias        — convolutional position embedding
  blocks.{i}.norm1.weight / .bias        — pre-attention layernorm
  blocks.{i}.attn.q.weight / .bias       — attention Q projection
  blocks.{i}.attn.k.weight / .bias       — attention K projection
  blocks.{i}.attn.v.weight / .bias       — attention V projection
  blocks.{i}.attn.out.weight / .bias     — attention output projection
  blocks.{i}.norm2.weight / .bias        — pre-FFN layernorm
  blocks.{i}.ffn.fc1.weight / .bias      — FFN first linear
  blocks.{i}.ffn.fc2.weight / .bias      — FFN second linear
"""

from __future__ import annotations
from pathlib import Path
import numpy as np

# HuBERT has 7 CNN layers, 12 transformer blocks
_N_CNN = 7
_N_BLOCKS = 12

# Explicit weight name mapping template.
# {prefix} = "hubert" or "model" (ContentVec)
# {i} = layer index
_PATTERN_MAP = {
    # CNN feature extractor
    "{p}.feature_extractor.conv_layers.{i}.conv.weight": "cnn.{i}.weight",
    "{p}.feature_extractor.conv_layers.{i}.conv.bias": "cnn.{i}.bias",
    "{p}.feature_extractor.conv_layers.{i}.layer_norm.weight": "cnn.{i}.norm.weight",
    "{p}.feature_extractor.conv_layers.{i}.layer_norm.bias": "cnn.{i}.norm.bias",
    # Feature projection
    "{p}.feature_projection.projection.weight": "proj.weight",
    "{p}.feature_projection.projection.bias": "proj.bias",
    "{p}.feature_projection.layer_norm.weight": "proj.norm.weight",
    "{p}.feature_projection.layer_norm.bias": "proj.norm.bias",
    # Convolutional position embeddings
    "{p}.encoder.pos_conv_embed.conv.weight": "pos_conv.weight",
    "{p}.encoder.pos_conv_embed.conv.bias": "pos_conv.bias",
    # Transformer encoder layer norm (before attention)
    "{p}.encoder.layers.{i}.layer_norm.weight": "blocks.{i}.norm1.weight",
    "{p}.encoder.layers.{i}.layer_norm.bias": "blocks.{i}.norm1.bias",
    # Attention projections
    "{p}.encoder.layers.{i}.attention.q_proj.weight": "blocks.{i}.attn.q.weight",
    "{p}.encoder.layers.{i}.attention.q_proj.bias": "blocks.{i}.attn.q.bias",
    "{p}.encoder.layers.{i}.attention.k_proj.weight": "blocks.{i}.attn.k.weight",
    "{p}.encoder.layers.{i}.attention.k_proj.bias": "blocks.{i}.attn.k.bias",
    "{p}.encoder.layers.{i}.attention.v_proj.weight": "blocks.{i}.attn.v.weight",
    "{p}.encoder.layers.{i}.attention.v_proj.bias": "blocks.{i}.attn.v.bias",
    "{p}.encoder.layers.{i}.attention.out_proj.weight": "blocks.{i}.attn.out.weight",
    "{p}.encoder.layers.{i}.attention.out_proj.bias": "blocks.{i}.attn.out.bias",
    # Transformer encoder layer norm (before FFN)
    "{p}.encoder.layers.{i}.final_layer_norm.weight": "blocks.{i}.norm2.weight",
    "{p}.encoder.layers.{i}.final_layer_norm.bias": "blocks.{i}.norm2.bias",
    # FFN
    "{p}.encoder.layers.{i}.feed_forward.intermediate_dense.weight": "blocks.{i}.ffn.fc1.weight",
    "{p}.encoder.layers.{i}.feed_forward.intermediate_dense.bias": "blocks.{i}.ffn.fc1.bias",
    "{p}.encoder.layers.{i}.feed_forward.output_dense.weight": "blocks.{i}.ffn.fc2.weight",
    "{p}.encoder.layers.{i}.feed_forward.output_dense.bias": "blocks.{i}.ffn.fc2.bias",
}


def _build_key_map(prefix: str) -> dict[str, str]:
    """Build the full src→dst key mapping for a given prefix."""
    result = {}
    for pattern_src, pattern_dst in _PATTERN_MAP.items():
        # Patterns with {i} need to be expanded for each layer
        if "{i}" in pattern_src:
            n = _N_CNN if "conv_layers" in pattern_src else _N_BLOCKS
            for i in range(n):
                src = pattern_src.replace("{p}", prefix).replace("{i}", str(i))
                dst = pattern_dst.replace("{i}", str(i))
                result[src] = dst
        else:
            src = pattern_src.replace("{p}", prefix)
            result[src] = pattern_dst
    return result


def _detect_prefix(keys: list[str]) -> str:
    """Detect whether keys use 'hubert' or 'model' prefix."""
    for key in keys:
        if key.startswith("hubert."):
            return "hubert"
        if key.startswith("model."):
            return "model"
    raise ValueError(f"Cannot detect prefix from keys: {keys[:3]}")


def _map_key(key: str, prefix: str) -> str | None:
    """Map a single HuggingFace key to internal name. Returns None if unknown."""
    key_map = _build_key_map(prefix)
    return key_map.get(key)


def load_weights_from_dict(
    weights: dict[str, np.ndarray],
) -> dict[str, np.ndarray]:
    """Map a {hf_key: array} dict to {internal_key: array}."""
    prefix = _detect_prefix(list(weights.keys()))
    key_map = _build_key_map(prefix)
    result = {}
    for src_key, array in weights.items():
        dst_key = key_map.get(src_key)
        if dst_key is not None:
            result[dst_key] = np.array(array, dtype=np.float32)
    return result


def load_from_safetensors(path: str | Path) -> dict[str, np.ndarray]:
    """Load weights from a .safetensors file."""
    from safetensors import safe_open
    weights = {}
    with safe_open(str(path), framework="numpy") as f:
        for key in f.keys():
            weights[key] = f.get_tensor(key)
    return load_weights_from_dict(weights)


def load_from_pt(path: str | Path) -> dict[str, np.ndarray]:
    """Load weights from a .pt or .bin PyTorch checkpoint."""
    import torch
    state_dict = torch.load(str(path), map_location="cpu")
    # Handle nested state dicts (some checkpoints wrap in {"model": ...})
    if "model" in state_dict and isinstance(state_dict["model"], dict):
        state_dict = state_dict["model"]
    weights = {k: v.numpy() for k, v in state_dict.items() if hasattr(v, "numpy")}
    return load_weights_from_dict(weights)


def download_and_load(
    model_id: str,
    cache_dir: str | None = None,
) -> dict[str, np.ndarray]:
    """Download from HuggingFace Hub and load weights.

    Caches to ~/.cache/mojo-audio/models/ by default.
    """
    from huggingface_hub import snapshot_download
    import os

    default_cache = os.path.expanduser("~/.cache/mojo-audio/models")
    cache = cache_dir or default_cache
    os.makedirs(cache, exist_ok=True)

    # Sanitize model_id for filesystem: "org/model" → "org--model"
    safe_id = model_id.replace("/", "--")
    local_dir = os.path.join(cache, safe_id)

    snapshot_download(
        repo_id=model_id,
        local_dir=local_dir,
        ignore_patterns=["*.msgpack", "*.h5", "flax_model*", "tf_model*", "rust_model*"],
    )

    # Find weight file: prefer safetensors, fall back to .bin/.pt
    local_path = Path(local_dir)
    for pattern in ["model.safetensors", "*.safetensors", "pytorch_model.bin", "*.pt"]:
        matches = list(local_path.glob(pattern))
        if matches:
            weight_path = matches[0]
            break
    else:
        raise FileNotFoundError(f"No weight file found in {local_dir}")

    ext = weight_path.suffix.lower()
    if ext == ".safetensors":
        return load_from_safetensors(weight_path)
    else:
        return load_from_pt(weight_path)


def load_weights(
    model_id_or_path: str,
    cache_dir: str | None = None,
) -> dict[str, np.ndarray]:
    """Load weights from HuggingFace ID or local file path.

    Args:
        model_id_or_path: HuggingFace model ID (e.g. "facebook/hubert-base-ls960")
                          or local path to .safetensors/.pt file.
        cache_dir: Override default download cache.

    Returns:
        Dict mapping internal weight names to float32 numpy arrays.
    """
    path = Path(model_id_or_path)
    if path.exists():
        ext = path.suffix.lower()
        if ext == ".safetensors":
            return load_from_safetensors(path)
        else:
            return load_from_pt(path)
    else:
        return download_and_load(model_id_or_path, cache_dir)
```

**Step 4: Run tests**

```bash
pixi run pytest tests/test_audio_encoder.py::TestWeightLoader -v
```

Expected: all 6 weight loader tests pass.

**Step 5: Commit**

```bash
git add src/models/_weight_loader.py tests/test_audio_encoder.py
git commit -m "feat: weight loader with HuBERT/ContentVec key mapping and HF download"
```

---

## Task 3: CNN Feature Extractor (`_feature_extractor.py`)

7 Conv1D layers (implemented as Conv2D with H=1). Each followed by GroupNorm + GELU. Converts raw waveform `[B, samples]` → feature vectors `[B, T, 512]`.

**Files:**
- Implement: `src/models/_feature_extractor.py`
- Test: add `TestFeatureExtractor` to `tests/test_audio_encoder.py`

---

**Step 1: Add tests first**

```python
class TestFeatureExtractor:
    """Tests for CNN feature extractor — no model download required."""

    def _make_extractor(self, device_ref, weights=None):
        """Helper: build feature extractor graph with random weights."""
        from models._feature_extractor import build_feature_extractor_graph
        import numpy as np
        if weights is None:
            # Generate random weights matching HuBERT architecture
            weights = {}
            configs = [(1,512,10),(512,512,3),(512,512,3),(512,512,3),(512,512,3),(512,512,2),(512,512,2)]
            for i, (c_in, c_out, k) in enumerate(configs):
                weights[f"cnn.{i}.weight"] = np.random.randn(k,1,c_in,c_out).astype(np.float32)
                weights[f"cnn.{i}.norm.weight"] = np.ones(c_out, dtype=np.float32)
                weights[f"cnn.{i}.norm.bias"] = np.zeros(c_out, dtype=np.float32)
        return build_feature_extractor_graph(weights, device_ref)

    def test_output_shape_1s(self, cpu_device):
        """1s @16kHz → (1, 49, 512)."""
        from max.graph import DeviceRef
        from max import engine
        import numpy as np

        cpu_ref = DeviceRef.CPU()
        graph = self._make_extractor(cpu_ref)
        model = engine.InferenceSession(devices=[cpu_device]).load(graph)

        audio = np.zeros((1, 16000, 1, 1), dtype=np.float32)
        result = model.execute(audio)
        out = np.array(list(result.values())[0] if isinstance(result, dict) else result[0])
        assert out.shape == (1, 49, 512), f"Expected (1,49,512) got {out.shape}"

    def test_output_shape_2s(self, cpu_device):
        """2s @16kHz → (1, 99, 512)."""
        from max.graph import DeviceRef
        from max import engine
        import numpy as np

        cpu_ref = DeviceRef.CPU()
        graph = self._make_extractor(cpu_ref)
        model = engine.InferenceSession(devices=[cpu_device]).load(graph)

        audio = np.zeros((1, 32000, 1, 1), dtype=np.float32)
        result = model.execute(audio)
        out = np.array(list(result.values())[0] if isinstance(result, dict) else result[0])
        assert out.shape == (1, 99, 512), f"Expected (1,99,512) got {out.shape}"

    def test_output_not_nan(self, cpu_device, audio_1s):
        """Output must not contain NaN or Inf."""
        from max.graph import DeviceRef
        from max import engine
        import numpy as np

        cpu_ref = DeviceRef.CPU()
        graph = self._make_extractor(cpu_ref)
        model = engine.InferenceSession(devices=[cpu_device]).load(graph)

        # Reshape to [B, L, 1, C_in]
        audio_in = audio_1s.reshape(1, 16000, 1, 1)
        result = model.execute(audio_in)
        out = np.array(list(result.values())[0] if isinstance(result, dict) else result[0])
        assert not np.isnan(out).any(), "Output contains NaN"
        assert not np.isinf(out).any(), "Output contains Inf"
```

**Step 2: Run tests to confirm they fail**

```bash
pixi run pytest tests/test_audio_encoder.py::TestFeatureExtractor -v
```

Expected: ImportError — `_feature_extractor` doesn't exist.

**Step 3: Implement `src/models/_feature_extractor.py`**

```python
"""CNN feature extractor for HuBERT / ContentVec.

7 Conv1D layers (implemented as Conv2D with H=1, NHWC layout).
Each layer: Conv1D → GroupNorm (via LayerNorm per-group) → GELU.

Input:  np.ndarray [B, L, 1, 1]  (audio reshaped to NHWC with C_in=1)
Output: MAX graph output [B, T, 512]

Downsampling schedule:
  Layer 0: kernel=10, stride=5  (C_in=1  → C_out=512)
  Layers 1-4: kernel=3, stride=2  (512 → 512)
  Layers 5-6: kernel=2, stride=2  (512 → 512)
Total downsampling: 5 × 2^6 = 320× → 16000/320 = 50 → -1 for padding = 49 frames/s
"""

from __future__ import annotations
import numpy as np
from max.graph import Graph, TensorType, ops, DeviceRef
from max.dtype import DType

# HuBERT CNN architecture: (in_channels, out_channels, kernel_size, stride)
_CNN_CONFIG = [
    (1, 512, 10, 5),
    (512, 512, 3, 2),
    (512, 512, 3, 2),
    (512, 512, 3, 2),
    (512, 512, 3, 2),
    (512, 512, 2, 2),
    (512, 512, 2, 2),
]


def _conv1d_layer(
    x,
    weight: np.ndarray,
    norm_weight: np.ndarray,
    norm_bias: np.ndarray,
    stride: int,
    device_ref,
):
    """One Conv1D + GroupNorm + GELU block.

    x shape: [B, L, 1, C_in]  (NHWC)
    weight shape: [K, 1, C_in, C_out]  (RSCF)

    Returns: [B, L', 1, C_out]
    """
    w = ops.constant(weight, device=device_ref)
    # Conv2D with width=1 implements Conv1D
    conv_out = ops.conv2d(x, w, stride=(stride, 1))  # [B, L', 1, C_out]

    # Squeeze spatial dims to apply LayerNorm: [B, L', C_out]
    B_dim = conv_out.shape[0] if hasattr(conv_out, "shape") else -1
    conv_out = ops.reshape(conv_out, [1, -1, weight.shape[3]])

    # GroupNorm via LayerNorm (HuBERT uses groups=C_out, equivalent to LayerNorm per channel)
    g = ops.constant(norm_weight, device=device_ref)
    b = ops.constant(norm_bias, device=device_ref)
    norm_out = ops.layer_norm(conv_out, g, b, 1e-5)

    # GELU activation
    gelu_out = ops.gelu(norm_out)

    # Reshape back to [B, L', 1, C_out] for next conv
    C_out = weight.shape[3]
    gelu_out = ops.reshape(gelu_out, [1, -1, 1, C_out])

    return gelu_out


def build_feature_extractor_graph(
    weights: dict[str, np.ndarray],
    device_ref,
    input_len: int = -1,
) -> Graph:
    """Build a MAX Graph for the 7-layer CNN feature extractor.

    Args:
        weights: Internal weight dict (from _weight_loader).
        device_ref: DeviceRef.CPU() or DeviceRef.GPU(0).
        input_len: Known input length in samples (-1 for dynamic).

    Returns:
        Compiled MAX Graph. Input: [1, L, 1, 1] float32. Output: [1, T, 512] float32.
    """
    # Dynamic shape for the sequence dimension
    input_shape = [1, input_len if input_len > 0 else -1, 1, 1]

    with Graph(
        "feature_extractor",
        input_types=[TensorType(DType.float32, input_shape, device_ref)],
    ) as g:
        x = g.inputs[0]

        for i, (c_in, c_out, kernel, stride) in enumerate(_CNN_CONFIG):
            w = weights[f"cnn.{i}.weight"]  # [C_out, C_in, K] → need [K, 1, C_in, C_out]
            # PyTorch Conv1D stores [C_out, C_in, K]; convert to MAX RSCF [K, 1, C_in, C_out]
            w_max = w.transpose(2, 0, 1)[..., np.newaxis, :]  # [K, C_in, 1] ... wrong
            # Correct: PyTorch [C_out, C_in, K] → MAX RSCF [K, W=1, C_in, C_out]
            w_max = np.transpose(w, (2, 1, 0))[:, np.newaxis, :, :]
            # shape: [K, 1, C_in, C_out] ✓

            norm_w = weights[f"cnn.{i}.norm.weight"]
            norm_b = weights[f"cnn.{i}.norm.bias"]

            x = _conv1d_layer(x, w_max, norm_w, norm_b, stride, device_ref)

        # Final reshape: [B, T, 1, 512] → [B, T, 512]
        x = ops.reshape(x, [1, -1, 512])
        g.output(x)

    return g
```

**Step 4: Run tests**

```bash
pixi run pytest tests/test_audio_encoder.py::TestFeatureExtractor -v
```

Expected:
```
test_output_shape_1s PASSED
test_output_shape_2s PASSED
test_output_not_nan PASSED
3 passed
```

**Step 5: Commit**

```bash
git add src/models/_feature_extractor.py tests/test_audio_encoder.py
git commit -m "feat: CNN feature extractor (7-layer Conv1D + GroupNorm + GELU via MAX Graph)"
```

---

## Task 4: Attention Module (`_attention.py`)

HuBERT-style multi-head self-attention. No RoPE — just standard scaled dot-product. Built from `ops.matmul`, `ops.softmax`, `ops.reshape`, `ops.transpose`.

**Files:**
- Implement: `src/models/_attention.py`
- Test: add `TestAttention` to `tests/test_audio_encoder.py`

---

**Step 1: Add tests first**

```python
class TestAttention:
    """Tests for multi-head attention — no download required."""

    def _make_attn_graph(self, device_ref, seq_len=49, hidden=768, heads=12):
        """Build attention graph with random weights."""
        from models._attention import build_attention_graph
        import numpy as np
        weights = {
            "q.weight": np.random.randn(hidden, hidden).astype(np.float32),
            "q.bias": np.zeros(hidden, dtype=np.float32),
            "k.weight": np.random.randn(hidden, hidden).astype(np.float32),
            "k.bias": np.zeros(hidden, dtype=np.float32),
            "v.weight": np.random.randn(hidden, hidden).astype(np.float32),
            "v.bias": np.zeros(hidden, dtype=np.float32),
            "out.weight": np.random.randn(hidden, hidden).astype(np.float32),
            "out.bias": np.zeros(hidden, dtype=np.float32),
        }
        return build_attention_graph(weights, device_ref, heads=heads, hidden=hidden)

    def test_output_shape(self, cpu_device):
        """Attention output shape must match input [1, T, 768]."""
        from max.graph import DeviceRef
        from max import engine
        import numpy as np

        cpu_ref = DeviceRef.CPU()
        graph = self._make_attn_graph(cpu_ref)
        model = engine.InferenceSession(devices=[cpu_device]).load(graph)

        x = np.random.randn(1, 49, 768).astype(np.float32)
        result = model.execute(x)
        out = np.array(list(result.values())[0] if isinstance(result, dict) else result[0])
        assert out.shape == (1, 49, 768), f"Expected (1,49,768) got {out.shape}"

    def test_different_seq_lengths(self, cpu_device):
        """Attention must handle different sequence lengths."""
        from max.graph import DeviceRef
        from max import engine
        import numpy as np

        cpu_ref = DeviceRef.CPU()
        graph = self._make_attn_graph(cpu_ref)
        model = engine.InferenceSession(devices=[cpu_device]).load(graph)

        for seq_len in [49, 99, 149]:
            x = np.random.randn(1, seq_len, 768).astype(np.float32)
            result = model.execute(x)
            out = np.array(list(result.values())[0] if isinstance(result, dict) else result[0])
            assert out.shape == (1, seq_len, 768)
```

**Step 2: Run to confirm fails**

```bash
pixi run pytest tests/test_audio_encoder.py::TestAttention -v
```

**Step 3: Implement `src/models/_attention.py`**

```python
"""HuBERT multi-head self-attention via MAX Graph ops.

Standard scaled dot-product attention. No RoPE.
12 heads × 64 dim = 768 total (HuBERT base).

Input:  [B, T, hidden]
Output: [B, T, hidden]
"""

from __future__ import annotations
import numpy as np
from max.graph import Graph, TensorType, ops, DeviceRef
from max.dtype import DType


def _linear(x, weight: np.ndarray, bias: np.ndarray | None, device_ref):
    """Apply a linear transformation: y = xW + b.

    weight: [in_features, out_features]  (already transposed from PyTorch's [out, in])
    """
    w = ops.constant(weight.T, device=device_ref)  # [in, out] transposed back
    out = ops.matmul(x, ops.constant(weight, device=device_ref))
    if bias is not None:
        b = ops.constant(bias, device=device_ref)
        out = ops.add(out, b)
    return out


def build_attention_graph(
    weights: dict[str, np.ndarray],
    device_ref,
    heads: int = 12,
    hidden: int = 768,
) -> Graph:
    """Build a MAX Graph for multi-head self-attention.

    Args:
        weights: Dict with keys q.weight, k.weight, v.weight, out.weight + .bias variants.
                 All shapes [hidden, hidden]. Weights are in HuggingFace format [out, in].
        device_ref: DeviceRef.CPU() or DeviceRef.GPU(0).
        heads: Number of attention heads (12 for HuBERT base).
        hidden: Hidden dimension (768 for HuBERT base).

    Input:  [1, T, hidden]
    Output: [1, T, hidden]
    """
    head_dim = hidden // heads
    scale = float(head_dim) ** -0.5

    with Graph(
        "attention",
        input_types=[TensorType(DType.float32, [1, -1, hidden], device_ref)],
    ) as g:
        x = g.inputs[0]  # [1, T, 768]

        # Q, K, V projections — weights are [out_features, in_features] in PyTorch
        # ops.matmul expects [B, T, in] @ [in, out] = [B, T, out]
        q_w = weights["q.weight"]  # [768, 768]
        k_w = weights["k.weight"]
        v_w = weights["v.weight"]
        q_b = weights.get("q.bias")
        k_b = weights.get("k.bias")
        v_b = weights.get("v.bias")

        q = ops.matmul(x, ops.constant(q_w.T, device=device_ref))
        k = ops.matmul(x, ops.constant(k_w.T, device=device_ref))
        v = ops.matmul(x, ops.constant(v_w.T, device=device_ref))

        if q_b is not None:
            q = ops.add(q, ops.constant(q_b, device=device_ref))
            k = ops.add(k, ops.constant(k_b, device=device_ref))
            v = ops.add(v, ops.constant(v_b, device=device_ref))

        # Reshape to [B, heads, T, head_dim]
        q = ops.reshape(q, [1, -1, heads, head_dim])
        q = ops.transpose(q, [0, 2, 1, 3])  # [1, heads, T, head_dim]
        k = ops.reshape(k, [1, -1, heads, head_dim])
        k = ops.transpose(k, [0, 2, 1, 3])
        v = ops.reshape(v, [1, -1, heads, head_dim])
        v = ops.transpose(v, [0, 2, 1, 3])

        # Scaled dot-product attention
        # scores: [1, heads, T, T]
        k_t = ops.transpose(k, [0, 1, 3, 2])  # [1, heads, head_dim, T]
        scores = ops.matmul(q, k_t)            # [1, heads, T, T]
        scores = ops.mul(scores, ops.constant(
            np.array(scale, dtype=np.float32), device=device_ref
        ))
        attn_weights = ops.softmax(scores, axis=-1)

        # Context: [1, heads, T, head_dim]
        context = ops.matmul(attn_weights, v)

        # Merge heads: [1, T, hidden]
        context = ops.transpose(context, [0, 2, 1, 3])   # [1, T, heads, head_dim]
        context = ops.reshape(context, [1, -1, hidden])   # [1, T, 768]

        # Output projection
        out_w = weights["out.weight"]
        out_b = weights.get("out.bias")
        out = ops.matmul(context, ops.constant(out_w.T, device=device_ref))
        if out_b is not None:
            out = ops.add(out, ops.constant(out_b, device=device_ref))

        g.output(out)

    return g
```

**Step 4: Run tests**

```bash
pixi run pytest tests/test_audio_encoder.py::TestAttention -v
```

**Step 5: Commit**

```bash
git add src/models/_attention.py tests/test_audio_encoder.py
git commit -m "feat: HuBERT multi-head attention via MAX Graph (no RoPE, scaled dot-product)"
```

---

## Task 5: Transformer Block (inline in `audio_encoder.py`)

Wire LayerNorm → Attention → residual → LayerNorm → FFN → residual. Each of the 12 transformer blocks is the same structure. Implemented as a helper function building within the main graph (not a separate sub-graph — MAX compiles it all together).

No separate file needed — the transformer block logic lives in `audio_encoder.py` since it's not independently useful.

**Files:**
- Modify: `src/models/audio_encoder.py` — add `_transformer_block()` helper
- Test: add `TestTransformerBlock` to `tests/test_audio_encoder.py`

---

**Step 1: Add tests**

```python
class TestTransformerBlock:
    """Test a single transformer block preserves shape and applies residual."""

    def test_output_shape(self, cpu_device):
        """Block output shape must match input [1, 49, 768]."""
        from max.graph import DeviceRef, Graph, TensorType, ops
        from max.dtype import DType
        from max import engine
        from models.audio_encoder import _transformer_block_ops
        import numpy as np

        hidden, heads = 768, 12
        # Build minimal weight dict for one block
        block_w = {
            "norm1.weight": np.ones(hidden, dtype=np.float32),
            "norm1.bias": np.zeros(hidden, dtype=np.float32),
            "attn.q.weight": np.random.randn(hidden, hidden).astype(np.float32),
            "attn.q.bias": np.zeros(hidden, dtype=np.float32),
            "attn.k.weight": np.random.randn(hidden, hidden).astype(np.float32),
            "attn.k.bias": np.zeros(hidden, dtype=np.float32),
            "attn.v.weight": np.random.randn(hidden, hidden).astype(np.float32),
            "attn.v.bias": np.zeros(hidden, dtype=np.float32),
            "attn.out.weight": np.random.randn(hidden, hidden).astype(np.float32),
            "attn.out.bias": np.zeros(hidden, dtype=np.float32),
            "norm2.weight": np.ones(hidden, dtype=np.float32),
            "norm2.bias": np.zeros(hidden, dtype=np.float32),
            "ffn.fc1.weight": np.random.randn(hidden, hidden * 4).astype(np.float32),
            "ffn.fc1.bias": np.zeros(hidden * 4, dtype=np.float32),
            "ffn.fc2.weight": np.random.randn(hidden * 4, hidden).astype(np.float32),
            "ffn.fc2.bias": np.zeros(hidden, dtype=np.float32),
        }
        cpu_ref = DeviceRef.CPU()

        with Graph("block_test", input_types=[TensorType(DType.float32, [1, 49, hidden], cpu_ref)]) as g:
            x = g.inputs[0]
            out = _transformer_block_ops(x, block_w, cpu_ref, heads=heads, hidden=hidden)
            g.output(out)

        model = engine.InferenceSession(devices=[cpu_device]).load(g)
        inp = np.random.randn(1, 49, hidden).astype(np.float32)
        result = model.execute(inp)
        out_arr = np.array(list(result.values())[0] if isinstance(result, dict) else result[0])
        assert out_arr.shape == (1, 49, hidden)
```

**Step 2: Add `_transformer_block_ops` to `src/models/audio_encoder.py`**

```python
# Add these imports at top of audio_encoder.py:
import numpy as np
from max.graph import Graph, TensorType, ops, DeviceRef
from max.dtype import DType


def _transformer_block_ops(x, block_weights: dict, device_ref, heads=12, hidden=768):
    """Apply one HuBERT transformer block (pre-norm architecture).

    Structure: LayerNorm → Attention → residual → LayerNorm → FFN → residual

    Args:
        x: MAX graph value, shape [1, T, hidden].
        block_weights: Dict with keys: norm1.*, attn.*, norm2.*, ffn.*.
                       All weights as numpy arrays.
        device_ref: DeviceRef for constant placement.
        heads: Number of attention heads.
        hidden: Hidden dimension.

    Returns:
        MAX graph value, shape [1, T, hidden].
    """
    head_dim = hidden // heads
    scale = float(head_dim) ** -0.5

    # --- Pre-attention LayerNorm ---
    ln1_w = ops.constant(block_weights["norm1.weight"], device=device_ref)
    ln1_b = ops.constant(block_weights["norm1.bias"], device=device_ref)
    normed = ops.layer_norm(x, ln1_w, ln1_b, 1e-5)

    # --- Multi-head Self-Attention ---
    def proj(tensor, w_key, b_key):
        w = ops.constant(block_weights[w_key].T, device=device_ref)
        out = ops.matmul(tensor, w)
        if b_key in block_weights:
            b = ops.constant(block_weights[b_key], device=device_ref)
            out = ops.add(out, b)
        return out

    q = proj(normed, "attn.q.weight", "attn.q.bias")
    k = proj(normed, "attn.k.weight", "attn.k.bias")
    v = proj(normed, "attn.v.weight", "attn.v.bias")

    q = ops.transpose(ops.reshape(q, [1, -1, heads, head_dim]), [0, 2, 1, 3])
    k = ops.transpose(ops.reshape(k, [1, -1, heads, head_dim]), [0, 2, 1, 3])
    v = ops.transpose(ops.reshape(v, [1, -1, heads, head_dim]), [0, 2, 1, 3])

    attn = ops.softmax(
        ops.mul(
            ops.matmul(q, ops.transpose(k, [0, 1, 3, 2])),
            ops.constant(np.array(scale, dtype=np.float32), device=device_ref),
        ),
        axis=-1,
    )
    context = ops.reshape(
        ops.transpose(ops.matmul(attn, v), [0, 2, 1, 3]),
        [1, -1, hidden],
    )
    attn_out = proj(context, "attn.out.weight", "attn.out.bias")

    # Residual connection
    x = ops.add(x, attn_out)

    # --- Pre-FFN LayerNorm ---
    ln2_w = ops.constant(block_weights["norm2.weight"], device=device_ref)
    ln2_b = ops.constant(block_weights["norm2.bias"], device=device_ref)
    normed2 = ops.layer_norm(x, ln2_w, ln2_b, 1e-5)

    # --- Feed-Forward Network ---
    ffn1_out = ops.gelu(proj(normed2, "ffn.fc1.weight", "ffn.fc1.bias"))
    ffn2_out = proj(ffn1_out, "ffn.fc2.weight", "ffn.fc2.bias")

    # Residual connection
    return ops.add(x, ffn2_out)
```

**Step 3: Run tests**

```bash
pixi run pytest tests/test_audio_encoder.py::TestTransformerBlock -v
```

**Step 4: Commit**

```bash
git add src/models/audio_encoder.py tests/test_audio_encoder.py
git commit -m "feat: HuBERT transformer block (pre-norm, attention + FFN + residuals)"
```

---

## Task 6: Wire Full `AudioEncoder` + `from_pretrained` + `encode`

Connect all components into the full MAX Graph: feature extractor → projection → position conv → 12 transformer blocks. Implement `from_pretrained()` and `encode()`.

**Files:**
- Implement: `src/models/audio_encoder.py` — replace stubs with full implementation
- Test: add `TestAudioEncoderShapes` (no download) to `tests/test_audio_encoder.py`

---

**Step 1: Add shape tests (no download)**

```python
class TestAudioEncoderShapes:
    """Test full AudioEncoder with random weights — no download required."""

    def _make_random_weights(self):
        """Generate a complete random weight dict matching HuBERT architecture."""
        import numpy as np
        w = {}
        # CNN weights
        configs = [(1,512,10),(512,512,3),(512,512,3),(512,512,3),(512,512,3),(512,512,2),(512,512,2)]
        for i, (c_in, c_out, k) in enumerate(configs):
            # PyTorch format: [C_out, C_in, K]
            w[f"cnn.{i}.weight"] = np.random.randn(c_out, c_in, k).astype(np.float32)
            w[f"cnn.{i}.norm.weight"] = np.ones(c_out, dtype=np.float32)
            w[f"cnn.{i}.norm.bias"] = np.zeros(c_out, dtype=np.float32)
        # Feature projection
        w["proj.weight"] = np.random.randn(768, 512).astype(np.float32)
        w["proj.bias"] = np.zeros(768, dtype=np.float32)
        w["proj.norm.weight"] = np.ones(768, dtype=np.float32)
        w["proj.norm.bias"] = np.zeros(768, dtype=np.float32)
        # Position conv (groups=16, kernel=128, padding=64)
        w["pos_conv.weight"] = np.random.randn(768, 768//16, 128).astype(np.float32)
        w["pos_conv.bias"] = np.zeros(768, dtype=np.float32)
        # 12 transformer blocks
        for i in range(12):
            for name in ["norm1", "norm2"]:
                w[f"blocks.{i}.{name}.weight"] = np.ones(768, dtype=np.float32)
                w[f"blocks.{i}.{name}.bias"] = np.zeros(768, dtype=np.float32)
            for proj in ["attn.q", "attn.k", "attn.v", "attn.out"]:
                w[f"blocks.{i}.{proj}.weight"] = np.random.randn(768, 768).astype(np.float32)
                w[f"blocks.{i}.{proj}.bias"] = np.zeros(768, dtype=np.float32)
            w[f"blocks.{i}.ffn.fc1.weight"] = np.random.randn(768, 3072).astype(np.float32)
            w[f"blocks.{i}.ffn.fc1.bias"] = np.zeros(3072, dtype=np.float32)
            w[f"blocks.{i}.ffn.fc2.weight"] = np.random.randn(3072, 768).astype(np.float32)
            w[f"blocks.{i}.ffn.fc2.bias"] = np.zeros(768, dtype=np.float32)
        return w

    def test_encode_1s_shape(self, cpu_device):
        """1s audio → (1, 49, 768)."""
        from models.audio_encoder import AudioEncoder
        import numpy as np
        weights = self._make_random_weights()
        model = AudioEncoder._from_weights(weights, device="cpu")
        audio = np.zeros((1, 16000), dtype=np.float32)
        out = model.encode(audio)
        assert out.shape == (1, 49, 768), f"Expected (1,49,768) got {out.shape}"

    def test_encode_2s_shape(self, cpu_device):
        """2s audio → (1, 99, 768)."""
        from models.audio_encoder import AudioEncoder
        import numpy as np
        weights = self._make_random_weights()
        model = AudioEncoder._from_weights(weights, device="cpu")
        audio = np.zeros((1, 32000), dtype=np.float32)
        out = model.encode(audio)
        assert out.shape == (1, 99, 768), f"Expected (1,99,768) got {out.shape}"
```

**Step 2: Implement the full `audio_encoder.py`**

Replace the stub `AudioEncoder` class with the full implementation. The `_from_weights` classmethod builds the MAX graph from a weight dict. `from_pretrained` calls `load_weights` then `_from_weights`. `encode` runs inference.

```python
"""AudioEncoder: HuBERT / ContentVec feature extraction via MAX Graph."""

from __future__ import annotations
import numpy as np
from pathlib import Path


def _transformer_block_ops(x, block_weights, device_ref, heads=12, hidden=768):
    """(Already implemented in Task 5 — copy here)"""
    # ... (paste full implementation from Task 5)


class AudioEncoder:
    """MAX Graph HuBERT / ContentVec encoder."""

    def __init__(self, _model, _device, _device_ref, _session):
        self._model = _model
        self._device = _device
        self._device_ref = _device_ref
        self._session = _session

    @classmethod
    def _from_weights(cls, weights: dict, device: str = "auto") -> "AudioEncoder":
        """Build MAX Graph from loaded weight dict."""
        from max import engine
        from max.driver import Accelerator, CPU, accelerator_count
        from max.graph import Graph, TensorType, ops, DeviceRef
        from max.dtype import DType

        # Device selection
        if device == "auto":
            use_gpu = accelerator_count() > 0
        else:
            use_gpu = (device == "gpu")

        if use_gpu:
            dev = Accelerator()
            device_ref = DeviceRef.GPU(0)
        else:
            dev = CPU()
            device_ref = DeviceRef.CPU()

        # Build full graph
        with Graph(
            "audio_encoder",
            input_types=[TensorType(DType.float32, [1, -1, 1, 1], device_ref)],
        ) as g:
            x = g.inputs[0]  # [1, L, 1, 1] — audio in NHWC format

            # Stage 1: CNN Feature Extractor
            cnn_configs = [
                (1, 512, 10, 5), (512, 512, 3, 2), (512, 512, 3, 2),
                (512, 512, 3, 2), (512, 512, 3, 2), (512, 512, 2, 2), (512, 512, 2, 2),
            ]
            for i, (c_in, c_out, kernel, stride) in enumerate(cnn_configs):
                w = weights[f"cnn.{i}.weight"]  # PyTorch [C_out, C_in, K]
                # Convert to MAX RSCF [K, W=1, C_in, C_out]
                w_max = np.transpose(w, (2, 1, 0))[:, np.newaxis, :, :]
                conv_out = ops.conv2d(
                    x,
                    ops.constant(w_max, device=device_ref),
                    stride=(stride, 1),
                )
                # Reshape for LayerNorm: [1, T, 1, C_out] → [1, T, C_out]
                conv_out = ops.reshape(conv_out, [1, -1, c_out])
                conv_out = ops.layer_norm(
                    conv_out,
                    ops.constant(weights[f"cnn.{i}.norm.weight"], device=device_ref),
                    ops.constant(weights[f"cnn.{i}.norm.bias"], device=device_ref),
                    1e-5,
                )
                conv_out = ops.gelu(conv_out)
                # Reshape back for next conv
                x = ops.reshape(conv_out, [1, -1, 1, c_out])

            # x is now [1, T, 1, 512]; reshape to [1, T, 512]
            x = ops.reshape(x, [1, -1, 512])

            # Stage 2: Feature Projection (512 → 768) + LayerNorm
            x = ops.add(
                ops.matmul(x, ops.constant(weights["proj.weight"].T, device=device_ref)),
                ops.constant(weights["proj.bias"], device=device_ref),
            )
            x = ops.layer_norm(
                x,
                ops.constant(weights["proj.norm.weight"], device=device_ref),
                ops.constant(weights["proj.norm.bias"], device=device_ref),
                1e-5,
            )

            # Stage 3: Convolutional Position Embeddings
            # pos_conv: groups=16, kernel=128, padding=64
            # PyTorch [C_out, C_in/groups, K] → MAX RSCF [K, 1, C_in/groups, C_out]
            pw = weights["pos_conv.weight"]  # [768, 768//16, 128]
            pw_max = np.transpose(pw, (2, 1, 0))[:, np.newaxis, :, :]  # [128, 1, 48, 768]
            pos_x = ops.reshape(x, [1, -1, 1, 768])  # [1, T, 1, 768] for conv2d
            pos_out = ops.conv2d(
                pos_x,
                ops.constant(pw_max, device=device_ref),
                stride=(1, 1),
                padding=(64, 64, 0, 0),
                groups=16,
            )
            pos_out = ops.reshape(pos_out, [1, -1, 768])
            pos_out = ops.gelu(pos_out)
            if "pos_conv.bias" in weights:
                pos_out = ops.add(
                    pos_out,
                    ops.constant(weights["pos_conv.bias"], device=device_ref),
                )
            x = ops.add(x, pos_out)

            # Stage 4: 12 Transformer Encoder Blocks
            for i in range(12):
                block_w = {
                    k.replace(f"blocks.{i}.", ""): weights[k]
                    for k in weights if k.startswith(f"blocks.{i}.")
                }
                x = _transformer_block_ops(x, block_w, device_ref)

            g.output(x)

        session = engine.InferenceSession(devices=[dev])
        model = session.load(g)

        return cls(_model=model, _device=dev, _device_ref=device_ref, _session=session)

    @classmethod
    def from_pretrained(
        cls,
        model_id: str,
        device: str = "auto",
        cache_dir: str | None = None,
    ) -> "AudioEncoder":
        """Load from HuggingFace Hub or local path."""
        from ._weight_loader import load_weights
        weights = load_weights(model_id, cache_dir)
        return cls._from_weights(weights, device=device)

    def encode(self, audio: np.ndarray) -> np.ndarray:
        """Encode audio waveform to feature vectors.

        Args:
            audio: Float32 numpy array [1, samples] at 16kHz.

        Returns:
            Float32 numpy array [1, time_frames, 768].
        """
        from max.driver import Tensor

        # Reshape to [1, L, 1, 1] for NHWC conv2d input
        audio_in = audio.reshape(1, -1, 1, 1).astype(np.float32)

        # Transfer to device if GPU
        if hasattr(self._device, 'id'):  # Accelerator has .id, CPU doesn't
            inp = Tensor.from_numpy(audio_in).to(self._device)
        else:
            inp = audio_in

        result = self._model.execute(inp)
        out = list(result.values())[0] if isinstance(result, dict) else result[0]
        return np.array(out)
```

**Step 3: Run shape tests**

```bash
pixi run pytest tests/test_audio_encoder.py::TestAudioEncoderShapes -v
```

Expected:
```
test_encode_1s_shape PASSED
test_encode_2s_shape PASSED
2 passed
```

**Step 4: Commit**

```bash
git add src/models/audio_encoder.py tests/test_audio_encoder.py
git commit -m "feat: full AudioEncoder with from_pretrained and encode via MAX Graph"
```

---

## Task 7: Integration Correctness Test (vs PyTorch HuBERT)

Download real HuBERT weights, compare MAX output against PyTorch reference. This is the proof it works.

**Files:**
- Test: add `TestAudioEncoderCorrectness` (marked `@pytest.mark.slow`) to `tests/test_audio_encoder.py`

---

**Step 1: Add correctness test**

```python
@pytest.mark.slow
class TestAudioEncoderCorrectness:
    """Integration test comparing MAX output to PyTorch HuBERT.

    Requires internet access to download ~360MB on first run.
    Run with: pixi run test-models-full
    """

    MODEL_ID = "facebook/hubert-base-ls960"

    def test_output_matches_pytorch_cpu(self):
        """MAX CPU output matches PyTorch within 1e-3."""
        import torch
        from transformers import HubertModel
        from models import AudioEncoder

        rng = np.random.default_rng(42)
        audio_np = rng.standard_normal((1, 16000)).astype(np.float32)

        # PyTorch reference
        pt_model = HubertModel.from_pretrained(self.MODEL_ID).eval()
        with torch.no_grad():
            pt_out = pt_model(torch.from_numpy(audio_np)).last_hidden_state.numpy()

        # MAX output
        max_model = AudioEncoder.from_pretrained(self.MODEL_ID, device="cpu")
        max_out = max_model.encode(audio_np)

        assert max_out.shape == pt_out.shape, \
            f"Shape mismatch: MAX {max_out.shape} vs PyTorch {pt_out.shape}"

        max_diff = np.abs(max_out - pt_out).max()
        mean_diff = np.abs(max_out - pt_out).mean()
        print(f"\n  PyTorch shape: {pt_out.shape}")
        print(f"  MAX shape:     {max_out.shape}")
        print(f"  Max diff:      {max_diff:.6f}")
        print(f"  Mean diff:     {mean_diff:.6f}")

        assert max_diff < 1e-3, \
            f"Max diff {max_diff:.4e} exceeds threshold 1e-3"

    @pytest.mark.skipif(
        not (lambda: __import__('max.driver', fromlist=['accelerator_count']).accelerator_count() > 0)(),
        reason="GPU not available"
    )
    def test_output_matches_pytorch_gpu(self):
        """MAX GPU output matches PyTorch within 1e-3."""
        import torch
        from transformers import HubertModel
        from models import AudioEncoder

        rng = np.random.default_rng(42)
        audio_np = rng.standard_normal((1, 16000)).astype(np.float32)

        pt_model = HubertModel.from_pretrained(self.MODEL_ID).eval()
        with torch.no_grad():
            pt_out = pt_model(torch.from_numpy(audio_np)).last_hidden_state.numpy()

        max_model = AudioEncoder.from_pretrained(self.MODEL_ID, device="gpu")
        max_out = max_model.encode(audio_np)

        max_diff = np.abs(max_out - pt_out).max()
        assert max_diff < 1e-3, f"GPU max diff {max_diff:.4e} exceeds 1e-3"
        print(f"\n  GPU Max diff: {max_diff:.6f} ✓")
```

**Step 2: Run correctness test (downloads ~360MB first time)**

```bash
pixi run test-models-full
```

Also run on the Spark:
```bash
ssh visage@visage-spark "cd /home/visage/repos/mojo-audio && git pull && ~/.pixi/bin/pixi run test-models-full"
```

**Step 3: If test fails (diff > 1e-3):** Check weight transpose directions. PyTorch linear layers store weights as `[out_features, in_features]` — the `.T` transpose in `ops.matmul(x, weight.T)` must be consistent everywhere. Also verify CNN weight reshape from `[C_out, C_in, K]` to `[K, 1, C_in, C_out]`.

**Step 4: Commit when passing**

```bash
git add tests/test_audio_encoder.py
git commit -m "test: integration correctness test vs PyTorch HuBERT (slow, requires download)"
```

---

## Task 8: Benchmark + Results Document

Run on both machines, write the comparison that will be posted publicly.

**Files:**
- Create: `experiments/contentvec-max/benchmark.py`
- Create: `experiments/contentvec-max/benchmark_results.md` (written by the script)

---

**Step 1: Create `experiments/contentvec-max/benchmark.py`**

```python
#!/usr/bin/env python3
"""
Benchmark AudioEncoder (MAX Graph) vs PyTorch HuBERT on all available hardware.

Writes results to experiments/contentvec-max/benchmark_results.md.
Run with: pixi run bench-models
"""

import sys, os, time, platform
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../../src"))

import numpy as np
import torch
from transformers import HubertModel
from models import AudioEncoder

MODEL_ID = "facebook/hubert-base-ls960"
N_WARMUP = 3
N_ITERS = 20
AUDIO = np.random.default_rng(42).standard_normal((1, 16000)).astype(np.float32)


def bench(fn, n_warmup=N_WARMUP, n_iters=N_ITERS):
    for _ in range(n_warmup):
        fn()
    times = []
    for _ in range(n_iters):
        t0 = time.perf_counter()
        fn()
        times.append((time.perf_counter() - t0) * 1000)
    return np.array(times)


def main():
    results = {}

    print("Benchmarking PyTorch CPU...")
    pt_cpu = HubertModel.from_pretrained(MODEL_ID).eval()
    pt_in = torch.from_numpy(AUDIO)
    lat = bench(lambda: pt_cpu(pt_in))
    results["PyTorch CPU"] = lat
    print(f"  {lat.mean():.1f} ms")

    if torch.cuda.is_available():
        print("Benchmarking PyTorch GPU...")
        pt_gpu = pt_cpu.cuda()
        pt_in_gpu = pt_in.cuda()
        lat = bench(lambda: (pt_gpu(pt_in_gpu), torch.cuda.synchronize()))
        results["PyTorch GPU"] = lat
        print(f"  {lat.mean():.1f} ms")

    print("Benchmarking MAX Engine CPU...")
    max_cpu = AudioEncoder.from_pretrained(MODEL_ID, device="cpu")
    lat = bench(lambda: max_cpu.encode(AUDIO))
    results["MAX Engine CPU"] = lat
    print(f"  {lat.mean():.1f} ms")

    from max.driver import accelerator_count
    if accelerator_count() > 0:
        print("Benchmarking MAX Engine GPU...")
        max_gpu = AudioEncoder.from_pretrained(MODEL_ID, device="gpu")
        lat = bench(lambda: max_gpu.encode(AUDIO))
        results["MAX Engine GPU"] = lat
        print(f"  {lat.mean():.1f} ms")

    # Print table
    print("\n" + "=" * 65)
    print(f"AudioEncoder Benchmark — {MODEL_ID}")
    print(f"Platform: {platform.machine()}, Python {platform.python_version()}")
    print(f"Input: 16000 samples (1s @16kHz), batch=1")
    print(f"Iters: {N_ITERS} (after {N_WARMUP} warmup)")
    print()
    baseline = list(results.values())[0].mean()
    print(f"{'Backend':<22} {'Mean ms':>9} {'Std ms':>8} {'P95 ms':>8} {'vs PT-CPU':>10}")
    print("-" * 62)
    for name, lats in results.items():
        speedup = baseline / lats.mean()
        print(f"{name:<22} {lats.mean():>9.1f} {lats.std():>8.1f} {np.percentile(lats,95):>8.1f} {speedup:>9.2f}x")
    print("=" * 65)

    # Write markdown
    out_path = os.path.join(os.path.dirname(__file__), "benchmark_results.md")
    with open(out_path, "w") as f:
        import subprocess
        gpu_info = subprocess.run(["nvidia-smi", "--query-gpu=name,compute_cap", "--format=csv,noheader"],
                                  capture_output=True, text=True).stdout.strip()
        f.write(f"# AudioEncoder Benchmark Results\n\n")
        f.write(f"**Model:** `{MODEL_ID}`  \n")
        f.write(f"**Date:** {time.strftime('%Y-%m-%d')}  \n")
        f.write(f"**Platform:** `{platform.machine()}` Python {platform.python_version()}  \n")
        f.write(f"**GPU:** {gpu_info}  \n\n")
        f.write("| Backend | Mean (ms) | Std (ms) | P95 (ms) | vs PyTorch CPU |\n")
        f.write("|---|---|---|---|---|\n")
        for name, lats in results.items():
            speedup = baseline / lats.mean()
            f.write(f"| {name} | {lats.mean():.1f} | {lats.std():.1f} | {np.percentile(lats,95):.1f} | {speedup:.2f}x |\n")
    print(f"\nResults written to {out_path}")


if __name__ == "__main__":
    main()
```

**Step 2: Run benchmark on local machine**

```bash
pixi run bench-models
```

**Step 3: Run on Spark**

```bash
ssh visage@visage-spark "cd /home/visage/repos/mojo-audio && git pull && ~/.pixi/bin/pixi run bench-models"
```

**Step 4: Copy Spark results back**

```bash
scp visage@visage-spark:/home/visage/repos/mojo-audio/experiments/contentvec-max/benchmark_results.md \
    experiments/contentvec-max/benchmark_results_spark.md
```

**Step 5: Commit**

```bash
git add experiments/contentvec-max/ tests/test_audio_encoder.py
git commit -m "feat: AudioEncoder benchmark + results (local + DGX Spark)"
```

---

## Task 9: `pixi.toml` + Final Run + Push

**Step 1: Verify final `pixi.toml` tasks section**

```toml
test-models = "pytest tests/test_audio_encoder.py -v -m 'not slow'"
test-models-full = "pytest tests/test_audio_encoder.py -v"
bench-models = "python experiments/contentvec-max/benchmark.py"
```

**Step 2: Run full test suite to confirm nothing broke**

```bash
pixi run test
pixi run test-models
```

**Step 3: Run on Spark to confirm ARM64 still works**

```bash
ssh visage@visage-spark "cd /home/visage/repos/mojo-audio && git pull && ~/.pixi/bin/pixi run test && ~/.pixi/bin/pixi run test-models"
```

**Step 4: Final commit + push**

```bash
git add -u
git commit -m "chore: finalize pixi tasks for AudioEncoder tests and benchmarks"
git push origin main
```

---

## Summary

After all tasks complete, mojo-audio has:

| Component | File | What it does |
|---|---|---|
| Package entry | `src/models/__init__.py` | `from mojo_audio.models import AudioEncoder` |
| Main class | `src/models/audio_encoder.py` | `from_pretrained()`, `encode()`, full graph |
| Weight loading | `src/models/_weight_loader.py` | HF download, .safetensors, .pt, name mapping |
| Tests | `tests/test_audio_encoder.py` | Unit + integration + GPU |
| Benchmark | `experiments/contentvec-max/benchmark.py` | Multi-backend timing |
| Results | `experiments/contentvec-max/benchmark_results.md` | Public-ready numbers |

**Supported checkpoints:**
- `facebook/hubert-base-ls960` (HuBERT base)
- `lengyue233/content-vec-best` (ContentVec — what Shade/RVC uses)
- Local `.pt` / `.safetensors` files

**Runs on:**
- RTX 4060 Ti (x86_64, CUDA 12.8) via `pixi run`
- DGX Spark GB10 SM_121 (aarch64, CUDA 13.0) via `~/.pixi/bin/pixi run`
- Any CPU (fallback)
