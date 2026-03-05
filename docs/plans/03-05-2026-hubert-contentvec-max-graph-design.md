# HuBERT / ContentVec in MAX Graph — Design

**Date:** 2026-03-05
**Status:** Approved

---

## Goal

Implement HuBERT and ContentVec audio feature extraction in MAX Graph so the full voice conversion pipeline (Shade) runs natively on DGX Spark SM_121 ARM64 — bypassing the torchaudio/PyTorch CUDA extension ecosystem that blocks all existing tools on that hardware.

---

## Context

- **DGX Spark (SM_121, ARM64):** MAX Engine installs cleanly, GPU sessions work, 16x speedup at HuBERT scale (confirmed live on `visage-spark`). PyTorch audio ecosystem (torchaudio, xformers, Flash Attention) is broken on this hardware.
- **mojo-audio:** Already runs on ARM64 (all 8 DSP test suites pass on Spark). mojovoice already uses `libmojo_audio.so` via FFI. Adding MAX model inference here creates a single "audio AI library" narrative for the open source / DGX Spark community.
- **Shade / RVC:** Uses `embedder_model="contentvec"` — ContentVec 768L12, architecturally identical to HuBERT base (12 transformer layers, 768 hidden dim) with different pretrained weights. Both checkpoints are supported by one implementation.
- **MAX v26.1 ONNX path:** Not viable — `session.load(path)` only accepts MAX's own Graph IR. Approach B ruled out. This implementation uses Approach A: native MAX Graph via `max.nn` primitives.

---

## Architecture

### Approach: Thin MAX Graph Wrapper (Approach A)

Build the model using `max.nn` primitives wired to match HuBERT's architecture exactly. Load weights from HuggingFace safetensors or PyTorch `.pt` files. Device selection is automatic (GPU preferred, CPU fallback).

### Module Structure

```
mojo-audio/
  src/
    models/                         # new Python package
      __init__.py                   # exports AudioEncoder
      audio_encoder.py              # main AudioEncoder class + from_pretrained()
      _feature_extractor.py         # 7-layer CNN (private)
      _attention.py                 # HuBERT-style attention without RoPE (private)
      _weight_loader.py             # HuggingFace/PT weight → MAX params (private)
```

Private modules (prefixed `_`) are implementation details. Only `AudioEncoder` is public API.

### Public API

```python
from mojo_audio.models import AudioEncoder

# HuBERT base (broader community)
model = AudioEncoder.from_pretrained("facebook/hubert-base-ls960")

# ContentVec 768L12 (what Shade/RVC actually uses)
model = AudioEncoder.from_pretrained("lengyue233/content-vec-best")

# Local file, explicit device
model = AudioEncoder.from_pretrained("/path/to/model.safetensors", device="cpu")

# Inference: np.ndarray [batch, seq_len] → np.ndarray [batch, time_frames, 768]
features = model.encode(audio_np)  # 1s @16kHz → [1, 49, 768]
```

CLI smoke test:
```bash
pixi run python -m mojo_audio.models.audio_encoder --model contentvec --input audio.wav
```

---

## Layer-by-Layer Design

### Stage 1: CNN Feature Extractor (`_feature_extractor.py`)

7 `max.nn.Conv1D` layers. Converts raw waveform `[batch, samples]` → feature vectors `[batch, 512, time]` at 50 Hz (320x downsampling for 16kHz input).

| Layer | In ch | Out ch | Kernel | Stride | Norm |
|-------|-------|--------|--------|--------|------|
| 0     | 1     | 512    | 10     | 5      | GroupNorm |
| 1–4   | 512   | 512    | 3      | 2      | GroupNorm |
| 5–6   | 512   | 512    | 2      | 2      | GroupNorm |

Each layer followed by GELU activation.

**GroupNorm implementation:** `max.nn` has no GroupNorm. Implement using `ops.reshape` + `max.nn.LayerNorm` per group + `ops.reshape` back. Mathematically identical to PyTorch's GroupNorm.

### Stage 2: Feature Projection

`max.nn.Linear(512 → 768)` + `max.nn.LayerNorm(768)`. Straightforward mapping.

### Stage 3: Convolutional Position Embeddings

`max.nn.Conv1D(768 → 768, kernel=128, padding=64, groups=16)` + GELU.

HuBERT encodes position via a depthwise convolution over the sequence — not sinusoidal, not RoPE. Output is added to projected features before the transformer blocks.

### Stage 4: Transformer Encoder — 12 Identical Blocks

Each block: `LayerNorm → Attention → residual → LayerNorm → FFN → residual` (pre-norm architecture).

**Attention (`_attention.py`):** Does NOT use `max.nn.AttentionWithRope` (which assumes RoPE position encoding). Built directly from `max.nn.Linear` + `ops`:

```python
# Four Linear projections
q_proj = max.nn.Linear(768, 768)   # 12 heads × 64 dim
k_proj = max.nn.Linear(768, 768)
v_proj = max.nn.Linear(768, 768)
out_proj = max.nn.Linear(768, 768)

# Scaled dot-product attention in the MAX graph
scores = ops.matmul(q, ops.transpose(k, [-1, -2])) * (1.0 / 8.0)  # sqrt(64)
weights = ops.softmax(scores, axis=-1)
context = ops.matmul(weights, v)
```

**FFN:** `max.nn.Linear(768 → 3072)` + GELU + `max.nn.Linear(3072 → 768)`.

---

## Weight Loading (`_weight_loader.py`)

### Supported Formats
- `.safetensors` — HuggingFace default, preferred
- `.pt` / `.bin` — legacy PyTorch checkpoints (RVC's `hubert_base.pt`, ContentVec)

Auto-detected by file extension. `.pt` files loaded with `torch.load(..., map_location="cpu")` then converted to NumPy.

### Key Name Mapping

HuggingFace uses `hubert.*` prefix; ContentVec uses `model.*`. Both mapped to an internal flat namespace:

```python
WEIGHT_MAP = {
    # Feature extractor
    "{prefix}.feature_extractor.conv_layers.{i}.conv.weight":       "cnn.{i}.weight",
    "{prefix}.feature_extractor.conv_layers.{i}.layer_norm.weight": "cnn.{i}.norm.weight",
    "{prefix}.feature_extractor.conv_layers.{i}.layer_norm.bias":   "cnn.{i}.norm.bias",
    # Feature projection
    "{prefix}.feature_projection.projection.weight":                 "proj.weight",
    "{prefix}.feature_projection.projection.bias":                   "proj.bias",
    "{prefix}.feature_projection.layer_norm.weight":                 "proj.norm.weight",
    # Position conv
    "{prefix}.encoder.pos_conv_embed.conv.weight":                   "pos_conv.weight",
    # Transformer layers (12 × ~8 entries each)
    "{prefix}.encoder.layers.{i}.attention.q_proj.weight":          "blocks.{i}.attn.q.weight",
    "{prefix}.encoder.layers.{i}.attention.k_proj.weight":          "blocks.{i}.attn.k.weight",
    "{prefix}.encoder.layers.{i}.attention.v_proj.weight":          "blocks.{i}.attn.v.weight",
    "{prefix}.encoder.layers.{i}.attention.out_proj.weight":        "blocks.{i}.attn.out.weight",
    "{prefix}.encoder.layers.{i}.layer_norm.weight":                "blocks.{i}.norm1.weight",
    "{prefix}.encoder.layers.{i}.final_layer_norm.weight":          "blocks.{i}.norm2.weight",
    "{prefix}.encoder.layers.{i}.feed_forward.intermediate_dense.weight": "blocks.{i}.ffn.fc1.weight",
    "{prefix}.encoder.layers.{i}.feed_forward.output_dense.weight": "blocks.{i}.ffn.fc2.weight",
    # ... ~50 total entries including biases
}
```

`{prefix}` is `hubert` or `model` detected automatically from checkpoint keys.

### Cache

Downloaded weights cached to `~/.cache/mojo-audio/models/{org}--{model}/`.
First `from_pretrained()` call: downloads + MAX graph compilation (~10–30s).
Subsequent calls: kernel cache hit, starts in under 1s.

---

## Device Selection

```python
AudioEncoder.from_pretrained(model_id)              # auto: GPU if available, else CPU
AudioEncoder.from_pretrained(model_id, device="gpu") # force GPU
AudioEncoder.from_pretrained(model_id, device="cpu") # force CPU
```

Device detection:
```python
from max.driver import accelerator_count
device = Accelerator() if accelerator_count() > 0 else CPU()
```

Works identically on RTX 4060 Ti (x86_64, CUDA) and DGX Spark GB10 (aarch64, SM_121).

---

## Testing Strategy

New pixi tasks:
- `pixi run test-models` — unit tests only, no download required (~fast)
- `pixi run test-models-full` — includes integration test (downloads model ~360MB)
- `pixi run bench-models` — benchmark CPU vs GPU, writes results to `experiments/contentvec-max/`

### Level 1: Unit Tests (no download)

Test each layer in isolation with synthetic inputs. Check shapes, check that residuals are applied, check GroupNorm equivalence.

### Level 2: Integration Test

Compare MAX output against PyTorch HuBERT reference on the same fixed random input. Acceptance threshold: max absolute difference < 1e-3.

### Level 3: Benchmark

20 iterations (3 warmup) on both CPU and GPU. Target: GPU faster than CPU by at least 5x. Results written to `experiments/contentvec-max/benchmark_results.md` for public posting.

---

## Supported Checkpoints

| Model ID | Type | Notes |
|---|---|---|
| `facebook/hubert-base-ls960` | HuBERT base | Broad community use, speech research |
| `lengyue233/content-vec-best` | ContentVec 768L12 | What RVC/Applio/Shade uses |
| Local `.pt` file | Either | RVC's `hubert_base.pt` or ContentVec `.pt` |
| Local `.safetensors` | Either | HuggingFace format |

---

## What This Unlocks

1. **Shade on DGX Spark** — ContentVec running natively on SM_121 via MAX, bypassing torchaudio entirely. First step toward full VC pipeline on Spark.
2. **Open source story** — "mojo-audio: audio AI DSP + inference that works on DGX Spark." Post to NVIDIA forums and Modular Discord with benchmark numbers.
3. **Modular visibility** — A voice conversion use case on DGX Spark using MAX Graph is exactly the story Modular wants to tell. Benchmark doc ready to share.

---

## Out of Scope (this implementation)

- RMVPE pitch extractor (next after HuBERT)
- VITS synthesis in MAX (after RMVPE)
- FAISS index retrieval (stays in Python/C++)
- Training (inference only)
- Multi-GPU / tensor parallelism (single GPU for now)
- RVC v3 (when/if released)
