"""Weight loading for RMVPE pitch extraction checkpoints.

Supports:
  - .pt (PyTorch checkpoint, used by rmvpe.pt from lj1995/VoiceConversionWebUI)
  - HuggingFace Hub auto-download

Internal naming convention (flat):
  enc_bn.scale / enc_bn.offset                          — initial encoder BN (baked)
  enc.{L}.{B}.0.w / .b                                 — encoder level L, block B, first conv weight/bias
  enc.{L}.{B}.0.scale / .offset                        — baked BN after first conv
  enc.{L}.{B}.1.w / .b                                 — encoder level L, block B, second conv weight/bias
  enc.{L}.{B}.1.scale / .offset                        — baked BN after second conv
  enc.{L}.{B}.sc.w / .b                                — shortcut conv weight/bias (only block 0)
  btl.{B}.0.w / .b / .scale / .offset                  — bottleneck block B (0..15), first conv
  btl.{B}.1.w / .b / .scale / .offset                  — bottleneck block B (0..15), second conv
  btl.{B}.sc.w / .b                                    — bottleneck shortcut (only block 0, raw layer 0)
  dec.{L}.up.w / .b                                    — decoder level L, ConvTranspose weight/bias
  dec.{L}.up.scale / .offset                           — decoder upsample BN (baked, confirmed present)
  dec.{L}.{B}.0.w / .b / .scale / .offset              — decoder level L, block B, first conv
  dec.{L}.{B}.1.w / .b / .scale / .offset              — decoder level L, block B, second conv
  dec.{L}.{B}.sc.w / .b                                — decoder shortcut (only block 0)
  out_cnn.w / out_cnn.b                                — final output CNN
  gru.weight_ih_l0 / .weight_hh_l0 / ...              — BiGRU weights (8 keys)
  linear.weight / linear.bias                          — output linear

BatchNorm baking:
  In inference mode BN is a linear transform:
    y = (x - mean) / sqrt(var + eps) * weight + bias
      = x * scale + offset
  where:
    scale  = weight / sqrt(var + eps)
    offset = bias - mean * scale

  We precompute scale and offset once at load time. MAX Engine has no ops.batch_norm.

Real checkpoint structure (confirmed by diagnostic):
  Each residual block has TWO conv-BN pairs:
    first:  .conv.0 (Conv2d weight/bias), .conv.1 (BN)
    second: .conv.3 (Conv2d weight),      .conv.4 (BN)  — indices skip ReLU at index 2
  Shortcut has both weight AND bias (block 0 only, where channel count changes).
  Decoder conv1 = [ConvTranspose2d (idx 0), BN (idx 1)] — BN confirmed present on all 5 levels.
  Bottleneck: 4 raw PyTorch "layers" (L=0..3), each with 4 residual blocks (B=0..3) = 16 blocks.
    Internal naming flattens to btl.{L*4+B}.*, i.e. btl.0 through btl.15.
"""

from __future__ import annotations
from pathlib import Path
import numpy as np

# Number of encoder/decoder levels and blocks per level
_N_ENC_LEVELS = 5
_N_DEC_LEVELS = 5
_N_BLOCKS_PER_LEVEL = 4

# Bottleneck: 4 raw PyTorch "layers" × 4 blocks each = 16 blocks total.
# Internal names use a global sequential index: btl.{L*4+B}.* (btl.0 through btl.15).
_N_BTL_LEVELS = 4
_N_BTL_BLOCKS_PER_LEVEL = 4

# GRU suffix names (copied directly, stripped of "fc.0.gru." prefix)
_GRU_SUFFIXES = [
    "weight_ih_l0",
    "weight_hh_l0",
    "bias_ih_l0",
    "bias_hh_l0",
    "weight_ih_l0_reverse",
    "weight_hh_l0_reverse",
    "bias_ih_l0_reverse",
    "bias_hh_l0_reverse",
]


def bake_batch_norm(
    weight: np.ndarray,
    bias: np.ndarray,
    running_mean: np.ndarray,
    running_var: np.ndarray,
    eps: float = 1e-5,
) -> tuple[np.ndarray, np.ndarray]:
    """Bake BatchNorm running statistics into scale and offset arrays.

    In inference mode, BN is equivalent to:
        y = (x - mean) / sqrt(var + eps) * weight + bias
          = x * scale + offset

    Args:
        weight: BN learnable weight (gamma), shape (C,).
        bias: BN learnable bias (beta), shape (C,).
        running_mean: BN running mean, shape (C,).
        running_var: BN running variance, shape (C,).
        eps: Small constant for numerical stability.

    Returns:
        (scale, offset) as float32 arrays, each shape (C,).
    """
    std = np.sqrt(running_var.astype(np.float64) + eps)
    scale = (weight.astype(np.float64) / std).astype(np.float32)
    offset = (bias.astype(np.float64) - running_mean.astype(np.float64) * scale.astype(np.float64)).astype(np.float32)
    return scale, offset


def _bake_bn_from_dict(sd: dict, prefix: str) -> tuple[np.ndarray, np.ndarray]:
    """Extract and bake a BatchNorm layer from state dict using key prefix.

    Args:
        sd: Full state dict (numpy arrays).
        prefix: Key prefix before .weight/.bias/.running_mean/.running_var,
                e.g. "unet.encoder.bn".

    Returns:
        (scale, offset) float32 arrays.
    """
    weight = np.asarray(sd[f"{prefix}.weight"], dtype=np.float32)
    bias = np.asarray(sd[f"{prefix}.bias"], dtype=np.float32)
    running_mean = np.asarray(sd[f"{prefix}.running_mean"], dtype=np.float32)
    running_var = np.asarray(sd[f"{prefix}.running_var"], dtype=np.float32)
    return bake_batch_norm(weight, bias, running_mean, running_var)


def _get(sd: dict, key: str, dtype=np.float32) -> np.ndarray | None:
    """Get key from state dict as float32 array, or None if missing."""
    v = sd.get(key)
    if v is None:
        return None
    return np.asarray(v, dtype=dtype)


def load_rmvpe_from_dict(sd: dict) -> dict[str, np.ndarray]:
    """Map a {checkpoint_key: array} dict to {internal_key: float32_array}.

    Bakes all BatchNorm layers. Maps all layer names to the internal
    naming convention. Returns NO raw running_mean, running_var, or
    num_batches_tracked keys.

    Args:
        sd: Raw state dict from torch.load (values may be tensors or numpy arrays).

    Returns:
        Dict mapping internal weight names to float32 numpy arrays.
    """
    # Convert any torch tensors to numpy up front
    raw: dict[str, np.ndarray] = {}
    for k, v in sd.items():
        if hasattr(v, "numpy"):
            raw[k] = v.numpy()
        elif hasattr(v, "detach"):
            raw[k] = v.detach().numpy()
        else:
            raw[k] = np.asarray(v)

    result: dict[str, np.ndarray] = {}

    # --- Initial encoder BN ---
    enc_bn_prefix = "unet.encoder.bn"
    if f"{enc_bn_prefix}.weight" in raw:
        scale, offset = _bake_bn_from_dict(raw, enc_bn_prefix)
        result["enc_bn.scale"] = scale
        result["enc_bn.offset"] = offset

    # --- Encoder levels ---
    # Each level L has _N_BLOCKS_PER_LEVEL residual blocks.
    # Each residual block has TWO conv-BN pairs:
    #   first:  .conv.0 (Conv2d), .conv.1 (BN)
    #   second: .conv.3 (Conv2d), .conv.4 (BN)  — ReLU at index 2 skipped
    # Block 0 of each level has a shortcut (1×1 conv, different channels).
    for L in range(_N_ENC_LEVELS):
        for B in range(_N_BLOCKS_PER_LEVEL):
            pfx = f"unet.encoder.layers.{L}.conv.{B}.conv"

            # First conv-BN pair (indices 0, 1)
            w = _get(raw, f"{pfx}.0.weight")
            if w is None:
                break  # this block doesn't exist in the (possibly truncated) fake weights
            result[f"enc.{L}.{B}.0.w"] = w
            b = _get(raw, f"{pfx}.0.bias")
            if b is not None:
                result[f"enc.{L}.{B}.0.b"] = b
            bn1_prefix = f"unet.encoder.layers.{L}.conv.{B}.conv.1"
            if f"{bn1_prefix}.weight" in raw:
                scale, offset = _bake_bn_from_dict(raw, bn1_prefix)
                result[f"enc.{L}.{B}.0.scale"] = scale
                result[f"enc.{L}.{B}.0.offset"] = offset

            # Second conv-BN pair (indices 3, 4) — only present in real checkpoint
            w2 = _get(raw, f"{pfx}.3.weight")
            if w2 is not None:
                result[f"enc.{L}.{B}.1.w"] = w2
                b2 = _get(raw, f"{pfx}.3.bias")
                if b2 is not None:
                    result[f"enc.{L}.{B}.1.b"] = b2
                bn2_prefix = f"unet.encoder.layers.{L}.conv.{B}.conv.4"
                if f"{bn2_prefix}.weight" in raw:
                    scale2, offset2 = _bake_bn_from_dict(raw, bn2_prefix)
                    result[f"enc.{L}.{B}.1.scale"] = scale2
                    result[f"enc.{L}.{B}.1.offset"] = offset2

            # Shortcut (block 0 only, where channel count changes)
            sc_w = _get(raw, f"unet.encoder.layers.{L}.conv.{B}.shortcut.weight")
            if sc_w is not None:
                result[f"enc.{L}.{B}.sc.w"] = sc_w
            sc_b = _get(raw, f"unet.encoder.layers.{L}.conv.{B}.shortcut.bias")
            if sc_b is not None:
                result[f"enc.{L}.{B}.sc.b"] = sc_b

    # --- Bottleneck (unet.intermediate) ---
    # 4 raw PyTorch "layers" × 4 blocks each = 16 blocks total.
    # Internal index I = L * _N_BTL_BLOCKS_PER_LEVEL + B (0..15).
    # Only raw layer 0, block 0 has a shortcut (channel change 256→512).
    for L in range(_N_BTL_LEVELS):
        for B in range(_N_BTL_BLOCKS_PER_LEVEL):
            I = L * _N_BTL_BLOCKS_PER_LEVEL + B
            pfx = f"unet.intermediate.layers.{L}.conv.{B}.conv"

            w = _get(raw, f"{pfx}.0.weight")
            if w is None:
                break
            result[f"btl.{I}.0.w"] = w
            b = _get(raw, f"{pfx}.0.bias")
            if b is not None:
                result[f"btl.{I}.0.b"] = b
            bn1_prefix = f"unet.intermediate.layers.{L}.conv.{B}.conv.1"
            if f"{bn1_prefix}.weight" in raw:
                scale, offset = _bake_bn_from_dict(raw, bn1_prefix)
                result[f"btl.{I}.0.scale"] = scale
                result[f"btl.{I}.0.offset"] = offset

            w2 = _get(raw, f"{pfx}.3.weight")
            if w2 is not None:
                result[f"btl.{I}.1.w"] = w2
                b2 = _get(raw, f"{pfx}.3.bias")
                if b2 is not None:
                    result[f"btl.{I}.1.b"] = b2
                bn2_prefix = f"unet.intermediate.layers.{L}.conv.{B}.conv.4"
                if f"{bn2_prefix}.weight" in raw:
                    scale2, offset2 = _bake_bn_from_dict(raw, bn2_prefix)
                    result[f"btl.{I}.1.scale"] = scale2
                    result[f"btl.{I}.1.offset"] = offset2

            sc_w = _get(raw, f"unet.intermediate.layers.{L}.conv.{B}.shortcut.weight")
            if sc_w is not None:
                result[f"btl.{I}.sc.w"] = sc_w
            sc_b = _get(raw, f"unet.intermediate.layers.{L}.conv.{B}.shortcut.bias")
            if sc_b is not None:
                result[f"btl.{I}.sc.b"] = sc_b

    # --- Decoder levels ---
    # Each level L has:
    #   conv1[0]: ConvTranspose2d weight
    #   conv1[1]: BN (baked → dec.{L}.up.scale / dec.{L}.up.offset)
    #   conv2: 4 residual blocks with same two-pair structure as encoder
    for L in range(_N_DEC_LEVELS):
        # Upsample ConvTranspose2d + BN
        up_w = _get(raw, f"unet.decoder.layers.{L}.conv1.0.weight")
        if up_w is not None:
            result[f"dec.{L}.up.w"] = up_w
        up_b = _get(raw, f"unet.decoder.layers.{L}.conv1.0.bias")
        if up_b is not None:
            result[f"dec.{L}.up.b"] = up_b
        bn_up_prefix = f"unet.decoder.layers.{L}.conv1.1"
        if f"{bn_up_prefix}.weight" in raw:
            scale, offset = _bake_bn_from_dict(raw, bn_up_prefix)
            result[f"dec.{L}.up.scale"] = scale
            result[f"dec.{L}.up.offset"] = offset

        # Residual blocks
        for B in range(_N_BLOCKS_PER_LEVEL):
            pfx = f"unet.decoder.layers.{L}.conv2.{B}.conv"

            w = _get(raw, f"{pfx}.0.weight")
            if w is None:
                break
            result[f"dec.{L}.{B}.0.w"] = w
            b = _get(raw, f"{pfx}.0.bias")
            if b is not None:
                result[f"dec.{L}.{B}.0.b"] = b
            bn1_prefix = f"unet.decoder.layers.{L}.conv2.{B}.conv.1"
            if f"{bn1_prefix}.weight" in raw:
                scale, offset = _bake_bn_from_dict(raw, bn1_prefix)
                result[f"dec.{L}.{B}.0.scale"] = scale
                result[f"dec.{L}.{B}.0.offset"] = offset

            w2 = _get(raw, f"{pfx}.3.weight")
            if w2 is not None:
                result[f"dec.{L}.{B}.1.w"] = w2
                b2 = _get(raw, f"{pfx}.3.bias")
                if b2 is not None:
                    result[f"dec.{L}.{B}.1.b"] = b2
                bn2_prefix = f"unet.decoder.layers.{L}.conv2.{B}.conv.4"
                if f"{bn2_prefix}.weight" in raw:
                    scale2, offset2 = _bake_bn_from_dict(raw, bn2_prefix)
                    result[f"dec.{L}.{B}.1.scale"] = scale2
                    result[f"dec.{L}.{B}.1.offset"] = offset2

            sc_w = _get(raw, f"unet.decoder.layers.{L}.conv2.{B}.shortcut.weight")
            if sc_w is not None:
                result[f"dec.{L}.{B}.sc.w"] = sc_w
            sc_b = _get(raw, f"unet.decoder.layers.{L}.conv2.{B}.shortcut.bias")
            if sc_b is not None:
                result[f"dec.{L}.{B}.sc.b"] = sc_b

    # --- Output CNN ---
    cnn_w = _get(raw, "cnn.weight")
    if cnn_w is not None:
        result["out_cnn.w"] = cnn_w
    cnn_b = _get(raw, "cnn.bias")
    if cnn_b is not None:
        result["out_cnn.b"] = cnn_b

    # --- BiGRU ---
    for suffix in _GRU_SUFFIXES:
        v = _get(raw, f"fc.0.gru.{suffix}")
        if v is not None:
            result[f"gru.{suffix}"] = v

    # --- Linear output ---
    lin_w = _get(raw, "fc.1.weight")
    if lin_w is not None:
        result["linear.weight"] = lin_w
    lin_b = _get(raw, "fc.1.bias")
    if lin_b is not None:
        result["linear.bias"] = lin_b

    return result


def load_rmvpe_from_pt(path: str | Path) -> dict[str, np.ndarray]:
    """Load weights from a .pt PyTorch checkpoint file.

    Args:
        path: Path to the .pt file.

    Returns:
        Dict mapping internal weight names to float32 numpy arrays.
    """
    import torch

    state_dict = torch.load(str(path), map_location="cpu", weights_only=True)
    # Handle nested state dicts (rmvpe.pt wraps in {"model": ...})
    state_dict = state_dict.get("model", state_dict)
    weights = {k: v.numpy() for k, v in state_dict.items() if hasattr(v, "numpy")}
    return load_rmvpe_from_dict(weights)


def load_rmvpe_weights(
    repo_id: str = "lj1995/VoiceConversionWebUI",
    filename: str = "rmvpe.pt",
    cache_dir: str | None = None,
) -> dict[str, np.ndarray]:
    """Download RMVPE weights from HuggingFace Hub and load them.

    Checks ~/.cache/rmvpe.pt first as a fast path. Downloads to HuggingFace
    Hub cache otherwise.

    Args:
        repo_id: HuggingFace repository ID.
        filename: Filename within the repository.
        cache_dir: Override the HuggingFace Hub cache directory.

    Returns:
        Dict mapping internal weight names to float32 numpy arrays.
    """
    import os

    # Fast path: check legacy cache location
    legacy_cache = os.path.expanduser("~/.cache/rmvpe.pt")
    if os.path.isfile(legacy_cache):
        return load_rmvpe_from_pt(legacy_cache)

    from huggingface_hub import hf_hub_download

    path = hf_hub_download(repo_id=repo_id, filename=filename, cache_dir=cache_dir)
    return load_rmvpe_from_pt(path)
