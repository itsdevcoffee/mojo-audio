"""Weight loading for the VITS synthesis pipeline (TextEncoder + Flow) from RVC v2 checkpoints.

Supports:
  - Extracting enc_p.* keys (TextEncoder) from the full checkpoint state dict
  - Extracting flow.* keys (ResidualCouplingBlock) — coupling layers only, Flip layers skipped
  - Extracting emb_g.weight (speaker embedding) and returning speaker vector [256, 1]
  - Reconstructing weight-normalized layers in both formats:
      old: .weight_g + .weight_v  ->  plain .weight
      new: .parametrizations.weight.original0 (weight_g) / .original1 (weight_v)  ->  plain .weight
  - bake_hifigan_cond: folds cond(g) into conv_pre.bias in-place
  - Full loading pipeline via load_vits_weights

Internal key convention after extraction (enc_p.* / flow.* prefix stripped):

enc_p.*:
  emb_phone.weight [192, 768], emb_phone.bias [192]
  emb_pitch.weight [256, 192]
  encoder.attn_layers.{i}.conv_{q,k,v,o}.weight [192, 192, 1], .bias [192]
  encoder.attn_layers.{i}.emb_rel_k [1, 21, 96], emb_rel_v [1, 21, 96]
  encoder.norm_layers_{1,2}.{i}.gamma [192], .beta [192]
  encoder.ffn_layers.{i}.conv_{1,2}.weight, .bias
  proj.weight [384, 192, 1], proj.bias [384]

flow.*:
  flows.{f}.pre.weight [192, 96, 1], .bias [192]        (f=0,2,4,6 — coupling layers)
  flows.{f}.enc.in_layers.{l}.weight [384, 192, 5], .bias [384]  (weight-norm reconstructed)
  flows.{f}.enc.res_skip_layers.{l}.weight, .bias       (weight-norm reconstructed)
  flows.{f}.enc.cond_layer.weight [1152, 256, 1], .bias [1152]   (weight-norm reconstructed)
  flows.{f}.post.weight [96, 192, 1], .bias [96]
"""

from __future__ import annotations

import numpy as np

from models._hifigan_weight_loader import reconstruct_weight_norm, extract_hifigan_weights, parse_hifigan_config


def _to_numpy(v) -> np.ndarray:
    """Convert a tensor or array-like to a numpy array."""
    if hasattr(v, "numpy"):
        return v.numpy()
    if hasattr(v, "detach"):
        return v.detach().numpy()
    return np.asarray(v)


def _collect_prefix(state_dict: dict, prefix: str) -> dict[str, np.ndarray]:
    """Extract all keys starting with *prefix* and strip it.

    Returns a dict of {stripped_key: numpy_float32_array}.
    """
    result: dict[str, np.ndarray] = {}
    n = len(prefix)
    for k, v in state_dict.items():
        if k.startswith(prefix):
            result[k[n:]] = np.asarray(_to_numpy(v), dtype=np.float32)
    return result


def _reconstruct_weight_norms(raw: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
    """Reconstruct weight-norm pairs into plain weights.

    Handles both formats:
      - Old: <prefix>.weight_g + <prefix>.weight_v
      - New: <prefix>.parametrizations.weight.original0 (weight_g)
             <prefix>.parametrizations.weight.original1 (weight_v)

    Removes the raw weight-norm keys and inserts the reconstructed
    <prefix>.weight in their place.  All other keys are passed through.

    Args:
        raw: Dict of float32 numpy arrays (already stripped of top-level prefix).

    Returns:
        New dict with weight-norm pairs reconstructed.
    """
    result: dict[str, np.ndarray] = {}

    # --- Detect old-style: .weight_g / .weight_v ---
    old_prefixes: set[str] = set()
    for k in raw:
        if k.endswith(".weight_v"):
            old_prefixes.add(k[: -len(".weight_v")])
        elif k.endswith(".weight_g"):
            old_prefixes.add(k[: -len(".weight_g")])

    old_consumed: set[str] = set()
    for pfx in sorted(old_prefixes):
        v_key = f"{pfx}.weight_v"
        g_key = f"{pfx}.weight_g"
        if v_key in raw and g_key in raw:
            result[f"{pfx}.weight"] = reconstruct_weight_norm(raw[v_key], raw[g_key])
            old_consumed.add(v_key)
            old_consumed.add(g_key)
        else:
            # Partial pair — keep as-is (shouldn't happen in practice)
            if v_key in raw:
                result[v_key] = raw[v_key]
                old_consumed.add(v_key)
            if g_key in raw:
                result[g_key] = raw[g_key]
                old_consumed.add(g_key)

    # --- Detect new-style: .parametrizations.weight.original0 / .original1 ---
    # original0 = weight_g, original1 = weight_v
    PARAM_SUFFIX_G = ".parametrizations.weight.original0"
    PARAM_SUFFIX_V = ".parametrizations.weight.original1"

    new_prefixes: set[str] = set()
    for k in raw:
        if k.endswith(PARAM_SUFFIX_G):
            new_prefixes.add(k[: -len(PARAM_SUFFIX_G)])
        elif k.endswith(PARAM_SUFFIX_V):
            new_prefixes.add(k[: -len(PARAM_SUFFIX_V)])

    new_consumed: set[str] = set()
    for pfx in sorted(new_prefixes):
        g_key = f"{pfx}{PARAM_SUFFIX_G}"
        v_key = f"{pfx}{PARAM_SUFFIX_V}"
        if g_key in raw and v_key in raw:
            result[f"{pfx}.weight"] = reconstruct_weight_norm(raw[v_key], raw[g_key])
            new_consumed.add(g_key)
            new_consumed.add(v_key)
        else:
            if g_key in raw:
                result[g_key] = raw[g_key]
                new_consumed.add(g_key)
            if v_key in raw:
                result[v_key] = raw[v_key]
                new_consumed.add(v_key)

    consumed = old_consumed | new_consumed

    # Pass through all other keys
    for k, v in raw.items():
        if k not in consumed:
            result[k] = v

    return result


def extract_vits_weights(state_dict: dict) -> dict[str, np.ndarray]:
    """Extract and process VITS TextEncoder + Flow weights from an RVC v2 state dict.

    Combines enc_p.* and flow.* keys into a single flat dict.  Flip layers
    (odd-indexed flow.flows entries) are skipped because they carry no parameters
    and thus produce no keys in the checkpoint.

    Steps:
      1. Filter enc_p.* and flow.* keys, strip prefix, convert to float32
      2. Reconstruct weight-norm pairs (both old and modern formats)
      3. Return merged dict

    Args:
        state_dict: Full RVC checkpoint state dict (values may be tensors or arrays).

    Returns:
        Dict mapping internal weight names to float32 numpy arrays.
        Keys retain sub-prefixes: e.g. "emb_phone.weight", "flows.0.pre.weight".
    """
    # Collect enc_p.* (strip "enc_p." prefix)
    enc_p_raw = _collect_prefix(state_dict, "enc_p.")

    # Collect flow.* (strip "flow." prefix)
    flow_raw = _collect_prefix(state_dict, "flow.")

    # Merge
    raw = {**enc_p_raw, **flow_raw}

    # Reconstruct weight-norm pairs
    result = _reconstruct_weight_norms(raw)

    return result


def extract_speaker_embedding(state_dict: dict, sid: int = 0) -> np.ndarray:
    """Extract a single speaker embedding vector from emb_g.weight.

    Args:
        state_dict: Full RVC checkpoint state dict.
        sid: Speaker ID (row index into emb_g.weight).

    Returns:
        Float32 numpy array of shape [256, 1] — the speaker conditioning vector.
    """
    emb_weight = np.asarray(_to_numpy(state_dict["emb_g.weight"]), dtype=np.float32)
    # emb_weight shape: [n_speakers, 256]
    g = emb_weight[sid].reshape(-1, 1)  # [256, 1]
    return g.astype(np.float32)


def bake_hifigan_cond(
    hifigan_weights: dict[str, np.ndarray],
    g: np.ndarray,
    cond_weight: np.ndarray,
    cond_bias: np.ndarray,
) -> None:
    """Fold the speaker conditioning cond(g) into conv_pre.bias in-place.

    Computes:
        cond_out = cond_weight[:, :, 0] @ g + cond_bias   # shape [C, 1] -> [C]
        conv_pre.bias += cond_out

    This bakes the (constant) speaker embedding conditioning directly into the
    HiFiGAN conv_pre bias so the graph does not need a separate cond branch.

    Args:
        hifigan_weights: Dict of HiFiGAN weights; modified in-place.
        g: Speaker embedding, shape [256, 1], float32.
        cond_weight: dec.cond.weight, shape [upsample_initial_channel, 256, 1], float32.
        cond_bias: dec.cond.bias, shape [upsample_initial_channel], float32.
    """
    # cond_weight: [C, 256, 1] -> squeeze to [C, 256]
    w = np.asarray(cond_weight, dtype=np.float32)[:, :, 0]  # [C, 256]
    g = np.asarray(g, dtype=np.float32)                      # [256, 1]
    b = np.asarray(cond_bias, dtype=np.float32)              # [C]

    # cond_out: [C, 256] @ [256, 1] -> [C, 1] -> squeeze -> [C]
    cond_out = (w @ g).squeeze(-1) + b                       # [C]

    hifigan_weights["conv_pre.bias"] += cond_out


def load_vits_weights(
    checkpoint_path: str,
    sid: int = 0,
) -> tuple[dict[str, np.ndarray], dict[str, np.ndarray], dict]:
    """Load VITS (enc_p + flow) and HiFiGAN decoder weights from an RVC v2 .pth checkpoint.

    Steps:
      1. Load checkpoint with torch
      2. Extract VITS weights (enc_p.* + flow.*) via extract_vits_weights
      3. Extract HiFiGAN weights (dec.*) via extract_hifigan_weights
      4. Extract speaker embedding g via extract_speaker_embedding
      5. Extract dec.cond.weight / dec.cond.bias from the state dict
      6. Bake cond(g) into conv_pre.bias via bake_hifigan_cond
      7. Parse config via parse_hifigan_config

    Args:
        checkpoint_path: Path to RVC .pth file.
        sid: Speaker ID to use (default 0).

    Returns:
        (vits_weights, hifigan_weights, config) where:
          - vits_weights: flat dict for enc_p and flow (keys without enc_p./flow. prefix)
          - hifigan_weights: flat dict for NSFHiFiGAN decoder (keys without dec. prefix),
            with conv_pre.bias already baked with cond(g)
          - config: model architecture dict (sr, upsample_rates, etc.)
    """
    import torch

    ckpt = torch.load(checkpoint_path, map_location="cpu", weights_only=False)

    if "weight" in ckpt:
        sd = ckpt["weight"]
    else:
        sd = {k: v for k, v in ckpt.items() if hasattr(v, "shape")}

    sr = ckpt.get("sr", 48000)
    config_list = ckpt.get("config", [])
    if not config_list:
        raise ValueError("RVC checkpoint missing 'config' key")

    # Extract VITS weights (enc_p + flow)
    vits_weights = extract_vits_weights(sd)

    # Extract HiFiGAN decoder weights
    hifigan_weights = extract_hifigan_weights(sd)

    # Extract speaker embedding
    g = extract_speaker_embedding(sd, sid=sid)

    # Extract dec.cond weights (use raw sd; they have "dec." prefix)
    cond_weight_key = "dec.cond.weight"
    cond_bias_key = "dec.cond.bias"
    if cond_weight_key in sd and cond_bias_key in sd:
        cond_weight = np.asarray(_to_numpy(sd[cond_weight_key]), dtype=np.float32)
        cond_bias = np.asarray(_to_numpy(sd[cond_bias_key]), dtype=np.float32)
        bake_hifigan_cond(hifigan_weights, g, cond_weight, cond_bias)

    # Parse config
    config = parse_hifigan_config(config_list, sr)

    return vits_weights, hifigan_weights, config
