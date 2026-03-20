"""Weight loading for NSF-HiFiGAN decoder from RVC v2 checkpoints.

Supports:
  - Parsing the RVC v2 config list (positional, 17 elements) into a model config dict
  - Extracting dec.* keys from the full checkpoint state dict
  - Reconstructing weight-normalized layers (weight_g + weight_v → plain weight)
  - Defensively baking BatchNorm if present (unlikely but handled gracefully)

Internal key convention after extraction (dec.* prefix stripped):
  conv_pre.weight / .bias                           — input projection
  ups.{i}.weight                                    — upsampling ConvTranspose1d (weight-norm reconstructed)
  noise_convs.{i}.weight / .bias                    — noise injection convolutions
  resblocks.{i}.convs1.{j}.weight / .bias           — resblock conv bank 1
  resblocks.{i}.convs2.{j}.weight / .bias           — resblock conv bank 2
  conv_post.weight / .bias                          — output projection

Weight norm reconstruction:
  weight = weight_v * (weight_g / norm(weight_v, per_output_channel))
  where norm is computed over all dims except dim 0 (the output channel dim).
"""

from __future__ import annotations

import math
from functools import reduce
from operator import mul

import numpy as np


def parse_hifigan_config(config_list: list, sr: int) -> dict:
    """Parse an RVC v2 config list into a HiFiGAN model config dict.

    RVC v2 config is an 18-element positional list. Key indices:
      [2]  = inter_channels (192)
      [10] = resblock_kernel_sizes ([3, 7, 11])
      [11] = resblock_dilation_sizes ([[1,3,5],...])
      [12] = upsample_rates (e.g. [12, 10, 2, 2])
      [13] = upsample_initial_channel (512)
      [14] = upsample_kernel_sizes (e.g. [24, 20, 4, 4])
      [17] = sample_rate (if present, else use sr param)

    Args:
        config_list: 18-element positional config list from RVC v2 checkpoint.
        sr: Sample rate (from checkpoint["sr"], used as fallback).

    Returns:
        Dict with named config values including computed hop_length.
    """
    upsample_rates = config_list[12]
    # Sample rate: prefer config[17] (int) over sr param (may be string like "48k")
    sample_rate = sr
    if len(config_list) > 17 and isinstance(config_list[17], int):
        sample_rate = config_list[17]
    elif isinstance(sr, str):
        # Handle "48k" -> 48000 style strings
        sample_rate = int(sr.replace("k", "000"))
    return {
        "sr": sample_rate,
        "inter_channels": config_list[2],
        "resblock_kernel_sizes": config_list[10],
        "resblock_dilation_sizes": config_list[11],
        "upsample_rates": upsample_rates,
        "upsample_initial_channel": config_list[13],
        "upsample_kernel_sizes": config_list[14],
        "hop_length": reduce(mul, upsample_rates, 1),
    }


def reconstruct_weight_norm(
    weight_v: np.ndarray,
    weight_g: np.ndarray,
) -> np.ndarray:
    """Reconstruct a plain weight tensor from weight-norm parameterization.

    weight = weight_v * (weight_g / norm(weight_v, per_output_channel))

    The norm is computed over all dimensions except dim 0 (output channels).

    Args:
        weight_v: Direction tensor, shape (out_channels, ...).
        weight_g: Magnitude tensor, shape (out_channels, 1, ...) or broadcastable.

    Returns:
        Reconstructed weight as float32.
    """
    weight_v = np.asarray(weight_v, dtype=np.float32)
    weight_g = np.asarray(weight_g, dtype=np.float32)

    out_channels = weight_v.shape[0]
    # Flatten all dims except dim 0 to compute per-output-channel norm
    flat = weight_v.reshape(out_channels, -1)
    norm = np.linalg.norm(flat, axis=1)
    # Reshape norm to broadcast: (out_channels, 1, 1, ...)
    norm_shape = (out_channels,) + (1,) * (weight_v.ndim - 1)
    norm = norm.reshape(norm_shape)

    return (weight_v * (weight_g / norm)).astype(np.float32)


def extract_hifigan_weights(state_dict: dict) -> dict[str, np.ndarray]:
    """Extract and process HiFiGAN decoder weights from an RVC v2 state dict.

    1. Filters to only dec.* keys
    2. Strips the dec. prefix
    3. Reconstructs weight-normalized layers (weight_g + weight_v → weight)
    4. Defensively bakes BatchNorm if present
    5. Converts all values to float32

    Args:
        state_dict: Full RVC checkpoint state dict (values may be tensors or arrays).

    Returns:
        Dict mapping internal weight names to float32 numpy arrays.
    """
    # Step 1: Filter dec.* keys and convert to numpy float32
    dec_prefix = "dec."
    raw: dict[str, np.ndarray] = {}
    for k, v in state_dict.items():
        if not k.startswith(dec_prefix):
            continue
        stripped = k[len(dec_prefix):]
        if hasattr(v, "numpy"):
            raw[stripped] = v.numpy()
        elif hasattr(v, "detach"):
            raw[stripped] = v.detach().numpy()
        else:
            raw[stripped] = np.asarray(v)

    # Step 2: Identify weight-norm pairs and reconstruct
    # Weight-norm keys come as <prefix>.weight_v and <prefix>.weight_g
    wn_prefixes: set[str] = set()
    for k in list(raw.keys()):
        if k.endswith(".weight_v"):
            wn_prefixes.add(k[: -len(".weight_v")])
        elif k.endswith(".weight_g"):
            wn_prefixes.add(k[: -len(".weight_g")])

    result: dict[str, np.ndarray] = {}

    # Reconstruct weight-norm pairs
    for pfx in sorted(wn_prefixes):
        v_key = f"{pfx}.weight_v"
        g_key = f"{pfx}.weight_g"
        if v_key in raw and g_key in raw:
            result[f"{pfx}.weight"] = reconstruct_weight_norm(raw[v_key], raw[g_key])
        else:
            # Partial weight-norm (shouldn't happen, but keep what we have)
            if v_key in raw:
                result[v_key] = np.asarray(raw[v_key], dtype=np.float32)
            if g_key in raw:
                result[g_key] = np.asarray(raw[g_key], dtype=np.float32)

    # Step 3: Copy all non-weight-norm keys
    wn_keys = set()
    for pfx in wn_prefixes:
        wn_keys.add(f"{pfx}.weight_v")
        wn_keys.add(f"{pfx}.weight_g")

    for k, v in raw.items():
        if k in wn_keys:
            continue
        # Step 4: Defensively bake BatchNorm if present
        if k.endswith(".running_mean") or k.endswith(".running_var") or k.endswith(".num_batches_tracked"):
            continue  # Skip BN stats; they get consumed by baking below
        result[k] = np.asarray(v, dtype=np.float32)

    # Step 4 (cont): Find and bake any BatchNorm layers
    # BN layers have .running_mean, .running_var, .weight, .bias
    bn_prefixes: set[str] = set()
    for k in raw:
        if k.endswith(".running_mean"):
            bn_prefixes.add(k[: -len(".running_mean")])

    if bn_prefixes:
        from models._rmvpe_weight_loader import bake_batch_norm

        for bn_pfx in sorted(bn_prefixes):
            weight_key = f"{bn_pfx}.weight"
            bias_key = f"{bn_pfx}.bias"
            mean_key = f"{bn_pfx}.running_mean"
            var_key = f"{bn_pfx}.running_var"

            if all(k in raw for k in [weight_key, bias_key, mean_key, var_key]):
                scale, offset = bake_batch_norm(
                    np.asarray(raw[weight_key], dtype=np.float32),
                    np.asarray(raw[bias_key], dtype=np.float32),
                    np.asarray(raw[mean_key], dtype=np.float32),
                    np.asarray(raw[var_key], dtype=np.float32),
                )
                # Replace the raw weight/bias with baked scale/offset
                result[f"{bn_pfx}.scale"] = scale
                result[f"{bn_pfx}.offset"] = offset
                # Remove the original weight/bias since they're now baked
                result.pop(weight_key, None)
                result.pop(bias_key, None)

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
        sd = {k: v.numpy() if hasattr(v, 'numpy') else np.asarray(v)
              for k, v in ckpt["weight"].items()}
    else:
        sd = {k: v.numpy() if hasattr(v, 'numpy') else np.asarray(v)
              for k, v in ckpt.items() if hasattr(v, 'shape')}

    sr = ckpt.get("sr", 48000)
    config_list = ckpt.get("config", [])
    if not config_list:
        raise ValueError("RVC checkpoint missing 'config' key")

    weights = extract_hifigan_weights(sd)
    config = parse_hifigan_config(config_list, sr)

    return weights, config
