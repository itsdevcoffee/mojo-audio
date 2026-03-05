"""Weight loading for HuBERT / ContentVec checkpoints.

Supports:
  - .safetensors (HuggingFace format, preferred)
  - .pt / .bin (legacy PyTorch, used by RVC's hubert_base.pt)
  - HuggingFace Hub auto-download

Internal naming convention (flat):
  cnn.{i}.weight / cnn.{i}.bias            — CNN feature extractor conv
  cnn.{i}.norm.weight / cnn.{i}.norm.bias  — CNN group norm
  proj.weight / proj.bias                   — feature projection linear
  proj.norm.weight / proj.norm.bias         — feature projection layernorm
  pos_conv.weight / pos_conv.bias           — convolutional position embedding
  blocks.{i}.norm1.weight / .bias           — pre-attention layernorm
  blocks.{i}.attn.q.weight / .bias          — attention Q projection
  blocks.{i}.attn.k.weight / .bias          — attention K projection
  blocks.{i}.attn.v.weight / .bias          — attention V projection
  blocks.{i}.attn.out.weight / .bias        — attention output projection
  blocks.{i}.norm2.weight / .bias           — pre-FFN layernorm
  blocks.{i}.ffn.fc1.weight / .bias         — FFN first linear
  blocks.{i}.ffn.fc2.weight / .bias         — FFN second linear
"""

from __future__ import annotations
from pathlib import Path
import numpy as np

_N_CNN = 7
_N_BLOCKS = 12

# Weight name mapping templates.
# {p} = prefix ("hubert" or "model")
# {i} = layer index
_PATTERN_MAP = {
    # CNN feature extractor
    "{p}feature_extractor.conv_layers.{i}.conv.weight":       "cnn.{i}.weight",
    "{p}feature_extractor.conv_layers.{i}.conv.bias":         "cnn.{i}.bias",
    "{p}feature_extractor.conv_layers.{i}.layer_norm.weight": "cnn.{i}.norm.weight",
    "{p}feature_extractor.conv_layers.{i}.layer_norm.bias":   "cnn.{i}.norm.bias",
    # Feature projection
    "{p}feature_projection.projection.weight": "proj.weight",
    "{p}feature_projection.projection.bias":   "proj.bias",
    "{p}feature_projection.layer_norm.weight": "proj.norm.weight",
    "{p}feature_projection.layer_norm.bias":   "proj.norm.bias",
    # Convolutional position embeddings
    # Standard weight (used by RVC-style checkpoints and some HF variants)
    "{p}encoder.pos_conv_embed.conv.weight": "pos_conv.weight",
    "{p}encoder.pos_conv_embed.conv.bias":   "pos_conv.bias",
    # Weight-normalized form (standard HuggingFace facebook/hubert-base-ls960)
    "{p}encoder.pos_conv_embed.conv.weight_g": "pos_conv.weight_g",
    "{p}encoder.pos_conv_embed.conv.weight_v": "pos_conv.weight_v",
    # Encoder layer norm (applied after pos_conv, before transformer blocks)
    "{p}encoder.layer_norm.weight": "enc_norm.weight",
    "{p}encoder.layer_norm.bias":   "enc_norm.bias",
    # Transformer block: pre-attention layernorm
    "{p}encoder.layers.{i}.layer_norm.weight": "blocks.{i}.norm1.weight",
    "{p}encoder.layers.{i}.layer_norm.bias":   "blocks.{i}.norm1.bias",
    # Transformer block: attention
    "{p}encoder.layers.{i}.attention.q_proj.weight":   "blocks.{i}.attn.q.weight",
    "{p}encoder.layers.{i}.attention.q_proj.bias":     "blocks.{i}.attn.q.bias",
    "{p}encoder.layers.{i}.attention.k_proj.weight":   "blocks.{i}.attn.k.weight",
    "{p}encoder.layers.{i}.attention.k_proj.bias":     "blocks.{i}.attn.k.bias",
    "{p}encoder.layers.{i}.attention.v_proj.weight":   "blocks.{i}.attn.v.weight",
    "{p}encoder.layers.{i}.attention.v_proj.bias":     "blocks.{i}.attn.v.bias",
    "{p}encoder.layers.{i}.attention.out_proj.weight": "blocks.{i}.attn.out.weight",
    "{p}encoder.layers.{i}.attention.out_proj.bias":   "blocks.{i}.attn.out.bias",
    # Transformer block: pre-FFN layernorm
    "{p}encoder.layers.{i}.final_layer_norm.weight": "blocks.{i}.norm2.weight",
    "{p}encoder.layers.{i}.final_layer_norm.bias":   "blocks.{i}.norm2.bias",
    # Transformer block: FFN
    "{p}encoder.layers.{i}.feed_forward.intermediate_dense.weight": "blocks.{i}.ffn.fc1.weight",
    "{p}encoder.layers.{i}.feed_forward.intermediate_dense.bias":   "blocks.{i}.ffn.fc1.bias",
    "{p}encoder.layers.{i}.feed_forward.output_dense.weight": "blocks.{i}.ffn.fc2.weight",
    "{p}encoder.layers.{i}.feed_forward.output_dense.bias":   "blocks.{i}.ffn.fc2.bias",
}


def _build_key_map(prefix: str) -> dict[str, str]:
    """Expand all pattern templates for the given prefix into a flat key map.

    prefix is the literal string to substitute for {p}, e.g.:
      - "hubert." for RVC-style checkpoints
      - "model."  for ContentVec checkpoints
      - ""        for standard HuggingFace checkpoints (no wrapper prefix)
    """
    result = {}
    for pattern_src, pattern_dst in _PATTERN_MAP.items():
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
    """Detect checkpoint prefix style and return the prefix string for _build_key_map.

    Returns:
        - "hubert." for RVC-style checkpoints (keys start with "hubert.")
        - "model."  for ContentVec checkpoints (keys start with "model.")
        - ""        for standard HuggingFace checkpoints (keys start with "feature_extractor." etc.)

    Raises ValueError if none of the known patterns match.
    """
    for key in keys:
        if key.startswith("hubert."):
            return "hubert."
        if key.startswith("model."):
            return "model."
        # Standard HuggingFace format: keys begin directly with the module name
        if key.startswith("feature_extractor.") or key.startswith("encoder.") or key.startswith("feature_projection."):
            return ""
    raise ValueError(
        f"Cannot detect HuBERT/ContentVec prefix from keys. "
        f"Expected keys starting with 'hubert.', 'model.', 'feature_extractor.', or 'encoder.'. "
        f"First keys: {keys[:3]}"
    )


def _map_key(key: str, prefix: str) -> str | None:
    """Map a single HuggingFace key to internal name. Returns None if unknown."""
    return _build_key_map(prefix).get(key)


def load_weights_from_dict(
    weights: dict[str, np.ndarray],
) -> dict[str, np.ndarray]:
    """Map a {hf_key: array} dict to {internal_key: float32_array}.

    Unknown keys are silently skipped.

    Handles weight-normalized pos_conv: reconstructs pos_conv.weight from
    pos_conv.weight_g and pos_conv.weight_v when the direct weight is absent.
    """
    prefix = _detect_prefix(list(weights.keys()))
    key_map = _build_key_map(prefix)
    result = {}
    for src_key, array in weights.items():
        dst_key = key_map.get(src_key)
        if dst_key is not None:
            result[dst_key] = np.asarray(array, dtype=np.float32)

    # Reconstruct weight-normalized pos_conv weight if needed.
    # Standard HuggingFace HuBERT checkpoints use nn.utils.weight_norm which stores
    # weight_g [1,1,K] and weight_v [C_out, C_in/groups, K] instead of weight directly.
    # Reconstruction: weight = weight_g * weight_v / norm(weight_v, dim=(0,1))
    if "pos_conv.weight" not in result and "pos_conv.weight_g" in result and "pos_conv.weight_v" in result:
        wg = result.pop("pos_conv.weight_g")
        wv = result.pop("pos_conv.weight_v")
        norm_v = np.sqrt((wv ** 2).sum(axis=(0, 1), keepdims=True))
        norm_v = np.maximum(norm_v, 1e-12)  # avoid div by zero
        result["pos_conv.weight"] = (wg * (wv / norm_v)).astype(np.float32)

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
    state_dict = torch.load(str(path), map_location="cpu", weights_only=True)
    # Handle nested state dicts (some checkpoints wrap in {"model": ...})
    if isinstance(state_dict, dict) and "model" in state_dict and isinstance(state_dict["model"], dict):
        state_dict = state_dict["model"]
    weights = {
        k: v.numpy()
        for k, v in state_dict.items()
        if hasattr(v, "numpy")
    }
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

    # Sanitize model_id for filesystem: "org/model" -> "org--model"
    safe_id = model_id.replace("/", "--")
    local_dir = os.path.join(cache, safe_id)

    snapshot_download(
        repo_id=model_id,
        local_dir=local_dir,
        ignore_patterns=["*.msgpack", "*.h5", "flax_model*", "tf_model*", "rust_model*"],
    )

    # Find weight file: prefer safetensors, fall back to .bin/.pt
    local_path = Path(local_dir)
    for pattern in ["model.safetensors", "*.safetensors", "pytorch_model.bin", "*.pt", "*.bin"]:
        matches = list(local_path.glob(pattern))
        if matches:
            weight_path = matches[0]
            break
    else:
        raise FileNotFoundError(
            f"No weight file found in {local_dir}. "
            f"Expected .safetensors, .bin, or .pt"
        )

    return _load_by_extension(weight_path)


def _load_by_extension(path: Path) -> dict[str, np.ndarray]:
    """Load weights from a file, dispatching by extension."""
    ext = path.suffix.lower()
    if ext == ".safetensors":
        return load_from_safetensors(path)
    else:
        return load_from_pt(path)


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
        return _load_by_extension(path)
    return download_and_load(model_id_or_path, cache_dir)
