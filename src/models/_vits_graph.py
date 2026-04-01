"""VITS graph building blocks for MAX Engine.

Implements:
  - TextEncoder (enc_p): phone/pitch embedding -> transformer encoder -> prior stats
  - Normalizing flow (reverse pass): WaveNet + ResidualCouplingLayer + Flip

All convolutions use the im2col + matmul workaround from _hifigan_graph.py.
Operations use channel-first [B, C, T] format internally, but the
underlying conv1d expects NHWC [B, T, 1, C], so we transpose at boundaries.
"""

from __future__ import annotations

import math
import numpy as np
from max.graph import Graph, TensorType, ops, DeviceRef, Dim
from max.dtype import DType

from models._hifigan_graph import conv1d, leaky_relu


# ---------------------------------------------------------------------------
# Layout helpers: BCT <-> NHWC
# ---------------------------------------------------------------------------


def _bct_to_nhwc(x):
    """Transpose [B, C, T] -> [B, T, 1, C] for conv1d."""
    # [B, C, T] -> [B, T, C]
    x = ops.transpose(x, 1, 2)
    # [B, T, C] -> [B, T, 1, C]
    x = ops.unsqueeze(x, 2)
    return x


def _nhwc_to_bct(x):
    """Transpose [B, T, 1, C] -> [B, C, T] after conv1d."""
    # [B, T, 1, C] -> [B, T, C]
    x = ops.squeeze(x, 2)
    # [B, T, C] -> [B, C, T]
    x = ops.transpose(x, 1, 2)
    return x


def _conv1d_bct(x, w_np, b_np, dilation, device_ref):
    """Conv1d wrapper for BCT-format tensors.

    1. Transposes x from [B, C, T] -> [B, T, 1, C] (NHWC)
    2. Calls the im2col conv1d from _hifigan_graph
    3. Transposes result back from [B, T, 1, C] -> [B, C, T]

    Args:
        x: Input tensor in [B, C, T] format.
        w_np: Weight numpy array [C_out, C_in, K].
        b_np: Bias numpy array [C_out] or None.
        dilation: Dilation factor.
        device_ref: MAX DeviceRef.

    Returns:
        Output tensor in [B, C, T] format.
    """
    x_nhwc = _bct_to_nhwc(x)
    out_nhwc = conv1d(x_nhwc, w_np, b_np, dilation=dilation, device_ref=device_ref)
    return _nhwc_to_bct(out_nhwc)


# ---------------------------------------------------------------------------
# WaveNet
# ---------------------------------------------------------------------------


def build_wavenet(x, mask, g_const, weights, config, device_ref):
    """Build WaveNet graph in BCT format.

    Args:
        x: Input [B, 192, T] in BCT format.
        mask: Mask [B, 1, T].
        g_const: Speaker conditioning [1, 256, 1] (already a graph constant/value).
        weights: Dict with keys like "cond_layer.weight", "cond_layer.bias",
                 "in_layers.0.weight", "in_layers.0.bias",
                 "res_skip_layers.0.weight", "res_skip_layers.0.bias", etc.
        config: Dict with "hidden_channels", "n_layers".
        device_ref: MAX DeviceRef.

    Returns:
        Output [B, 192, T] in BCT format.
    """
    hidden = config["hidden_channels"]  # 192
    n_layers = config["n_layers"]       # 3

    # Initialize output as zeros (same shape as x)
    zero = ops.constant(np.array(0.0, dtype=np.float32), device=device_ref)
    output = ops.mul(x, zero)  # [B, 192, T] of zeros

    # Compute g_all = cond_layer(g) — Conv1d(256, 2*192*3=1152, k=1)
    g_all = _conv1d_bct(
        g_const,
        weights["cond_layer.weight"],
        weights.get("cond_layer.bias"),
        dilation=1,
        device_ref=device_ref,
    )  # [1, 1152, 1]

    for i in range(n_layers):
        # x_in = in_layers[i](x) — Conv1d(192, 384, k=5, dil=dilation_rate**i)
        # For RVC v2 with dilation_rate=1, all dilations are 1
        dil = config.get("dilation_rate", 1) ** i
        x_in = _conv1d_bct(
            x,
            weights[f"in_layers.{i}.weight"],
            weights.get(f"in_layers.{i}.bias"),
            dilation=dil,
            device_ref=device_ref,
        )  # [B, 384, T]

        # Slice conditioning: g_l = g_all[:, i*384:(i+1)*384, :]
        g_l = ops.slice_tensor(
            g_all,
            [slice(None), slice(i * 2 * hidden, (i + 1) * 2 * hidden), slice(None)],
        )  # [1, 384, 1]

        # Add conditioning to input: in_act = x_in + g_l (broadcast over T)
        in_act = ops.add(x_in, g_l)

        # Gated activation: tanh(first_half) * sigmoid(second_half)
        t_act = ops.tanh(
            ops.slice_tensor(in_act, [slice(None), slice(0, hidden), slice(None)])
        )  # [B, 192, T]
        s_act = ops.sigmoid(
            ops.slice_tensor(in_act, [slice(None), slice(hidden, 2 * hidden), slice(None)])
        )  # [B, 192, T]
        acts = ops.mul(t_act, s_act)  # [B, 192, T]

        # res_skip = res_skip_layers[i](acts) — Conv1d(192, 384, k=1) or Conv1d(192, 192, k=1)
        res_skip = _conv1d_bct(
            acts,
            weights[f"res_skip_layers.{i}.weight"],
            weights.get(f"res_skip_layers.{i}.bias"),
            dilation=1,
            device_ref=device_ref,
        )

        if i < n_layers - 1:
            # res_skip is [B, 384, T]
            # res = res_skip[:, :192, :], skip = res_skip[:, 192:, :]
            res = ops.slice_tensor(
                res_skip, [slice(None), slice(0, hidden), slice(None)]
            )
            skip = ops.slice_tensor(
                res_skip, [slice(None), slice(hidden, 2 * hidden), slice(None)]
            )
            # x = (x + res) * mask
            x = ops.mul(ops.add(x, res), mask)
            # Rebind x to keep symbolic T consistent
            x = ops.rebind(x, x.shape, message=f"wavenet layer {i}: rebind x after res")
            # output += skip
            output = ops.add(output, skip)
        else:
            # Last layer: res_skip is [B, 192, T]
            output = ops.add(output, res_skip)

    # Return output * mask
    return ops.mul(output, mask)


# ---------------------------------------------------------------------------
# ResidualCouplingLayer (reverse)
# ---------------------------------------------------------------------------


def build_coupling_layer_reverse(x, mask, g_const, weights, config, device_ref):
    """Build a single ResidualCouplingLayer reverse pass in BCT format.

    mean_only=True (as used in RVC v2).

    Args:
        x: Input [B, 192, T] in BCT format.
        mask: Mask [B, 1, T].
        g_const: Speaker conditioning [1, 256, 1].
        weights: Dict with keys like "pre.weight", "pre.bias",
                 "enc.cond_layer.weight", "enc.in_layers.0.weight", etc.
                 "post.weight", "post.bias".
        config: Dict with "hidden_channels", "n_layers", "dilation_rate".
        device_ref: MAX DeviceRef.

    Returns:
        Output [B, 192, T] in BCT format.
    """
    half_ch = config["hidden_channels"] // 2  # 96

    # Split x into x0 [B, 96, T] and x1 [B, 96, T] along channel dim
    x0 = ops.slice_tensor(x, [slice(None), slice(0, half_ch), slice(None)])
    x1 = ops.slice_tensor(x, [slice(None), slice(half_ch, 2 * half_ch), slice(None)])

    # h = pre_conv(x0) * mask — Conv1d(96, 192, k=1)
    h = _conv1d_bct(
        x0,
        weights["pre.weight"],
        weights.get("pre.bias"),
        dilation=1,
        device_ref=device_ref,
    )
    h = ops.mul(h, mask)

    # h = wavenet(h, mask, g_const, enc_weights)
    enc_weights = {}
    for k, v in weights.items():
        if k.startswith("enc."):
            enc_weights[k[4:]] = v  # strip "enc." prefix
    h = build_wavenet(h, mask, g_const, enc_weights, config, device_ref)

    # m = post_conv(h) * mask — Conv1d(192, 96, k=1)
    m = _conv1d_bct(
        h,
        weights["post.weight"],
        weights.get("post.bias"),
        dilation=1,
        device_ref=device_ref,
    )
    m = ops.mul(m, mask)

    # Reverse coupling (mean_only=True): x1 = (x1 - m) * mask
    x1 = ops.mul(ops.sub(x1, m), mask)

    # Concat [x0, x1] along channel dim (dim=1)
    return ops.concat([x0, x1], axis=1)


# ---------------------------------------------------------------------------
# Flip
# ---------------------------------------------------------------------------


def flip_channels(x, n_channels, device_ref):
    """Reverse channels along dim=1 for a [B, C, T] tensor.

    Uses a permutation matrix matmul instead of per-channel slicing.
    The old approach created ~209 graph nodes per flip (192 slices + concats),
    causing OOM during MAX compilation. This approach uses 3 ops per flip
    (2 transposes + 1 matmul).

    Args:
        x: Input [B, C, T].
        n_channels: Number of channels (e.g. 192).
        device_ref: MAX DeviceRef.

    Returns:
        [B, C, T] with channels reversed.
    """
    # Identity matrix with reversed rows = channel reversal permutation
    perm = np.eye(n_channels, dtype=np.float32)[::-1].copy()
    perm_const = ops.constant(perm, device=device_ref)

    # [B, C, T] -> [B, T, C] -> matmul with perm -> [B, T, C] -> [B, C, T]
    x_t = ops.transpose(x, 1, 2)
    x_flipped = ops.matmul(x_t, perm_const)
    return ops.transpose(x_flipped, 1, 2)


# ---------------------------------------------------------------------------
# Full flow graph
# ---------------------------------------------------------------------------


def build_flow_graph(weights, g_np, config, device="cpu", batch_size=1):
    """Build the full VITS normalizing flow (reverse pass) as a MAX Graph.

    The flow has 8 modules in forward order:
        [CouplingLayer0, Flip, CouplingLayer1, Flip, CouplingLayer2, Flip, CouplingLayer3, Flip]

    At inference (reverse=True), iterate in REVERSE order:
        [Flip, CouplingLayer3, Flip, CouplingLayer2, Flip, CouplingLayer1, Flip, CouplingLayer0]

    Args:
        weights: Dict of numpy arrays with flow weight keys (e.g. "flows.0.pre.weight").
        g_np: Speaker embedding numpy array, shape [256, 1].
        config: Dict with keys: hidden_channels, n_layers, dilation_rate, n_flows.
        device: Device string, default "cpu".
        batch_size: Batch size, default 1.

    Returns:
        A compiled MAX Graph with inputs z_p [B, 192, T] and mask [B, 1, T],
        output z [B, 192, T].
    """
    n_flows = config.get("n_flows", 4)
    inter_channels = config.get("inter_channels", 192)

    dev = DeviceRef.CPU() if device == "cpu" else DeviceRef(device)
    T = Dim("T")

    with Graph(
        "vits_flow",
        input_types=[
            TensorType(DType.float32, [batch_size, inter_channels, T], dev),  # z_p
            TensorType(DType.float32, [batch_size, 1, T], dev),              # mask
        ],
    ) as g:
        z_p, mask = g.inputs

        # Bake speaker embedding as constant [1, 256, 1]
        # g_np is [256, 1], need to reshape to [1, 256, 1]
        g_const = ops.constant(
            g_np.reshape(1, -1, 1).astype(np.float32), device=dev
        )

        x = z_p

        # Build the flow modules in forward order:
        # [CouplingLayer0, Flip, CouplingLayer1, Flip, ..., CouplingLayer3, Flip]
        # Coupling layers are at flow indices 0, 2, 4, 6
        # Flip layers are at flow indices 1, 3, 5, 7
        #
        # For reverse, iterate in reverse: [Flip, CouplingLayer3, Flip, ..., CouplingLayer0]

        # The forward module list is:
        # modules = [(coupling, 0), (flip,), (coupling, 1), (flip,), ...]
        # In reverse: flip, coupling3, flip, coupling2, flip, coupling1, flip, coupling0

        # Coupling layer indices in the checkpoint: 0, 2, 4, 6
        coupling_flow_indices = [i * 2 for i in range(n_flows)]  # [0, 2, 4, 6]

        # Reverse iteration order
        for i in range(n_flows - 1, -1, -1):
            flow_idx = coupling_flow_indices[i]

            # First: Flip (this is the Flip that comes AFTER coupling layer i)
            x = flip_channels(x, inter_channels, dev)

            # Then: CouplingLayer i (reverse)
            coupling_weights = {}
            for k, v in weights.items():
                prefix = f"flows.{flow_idx}."
                if k.startswith(prefix):
                    coupling_weights[k[len(prefix):]] = v

            wavenet_config = {
                "hidden_channels": config.get("hidden_channels", 192),
                "n_layers": config.get("n_layers", 3),
                "dilation_rate": config.get("dilation_rate", 1),
            }

            x = build_coupling_layer_reverse(
                x, mask, g_const, coupling_weights, wavenet_config, dev
            )

        g.output(x)

    return g


# ---------------------------------------------------------------------------
# TextEncoder (enc_p) — building blocks
# ---------------------------------------------------------------------------


def build_layer_norm(x, gamma_np, beta_np, device_ref, eps=1e-5):
    """Channel-wise LayerNorm for BCT tensors.

    VITS LayerNorm normalises over the channel dimension (dim=1) for each
    (batch, timestep) pair.  Transposes to BTC, applies ops.layer_norm on
    the last dim, and transposes back.

    Args:
        x: Input [B, C, T] in BCT format.
        gamma_np: Scale parameter [C], numpy float32.
        beta_np: Shift parameter [C], numpy float32.
        device_ref: MAX DeviceRef.
        eps: Epsilon for numerical stability.

    Returns:
        Normalised tensor [B, C, T].
    """
    x_t = ops.transpose(x, 1, 2)  # [B, T, C]
    gamma = ops.constant(gamma_np.astype(np.float32), device=device_ref)
    beta = ops.constant(beta_np.astype(np.float32), device=device_ref)
    x_norm = ops.layer_norm(x_t, gamma, beta, eps)
    return ops.transpose(x_norm, 1, 2)  # [B, C, T]


def build_ffn(x, mask, weights, config, device_ref):
    """Feed-forward network: Conv1d -> ReLU -> Conv1d, masked.

    Args:
        x: Input [B, hidden, T] in BCT format.
        mask: Mask [B, 1, T].
        weights: Dict with "conv_1.weight", "conv_1.bias",
                 "conv_2.weight", "conv_2.bias".
        config: Unused (kernel sizes come from weights).
        device_ref: MAX DeviceRef.

    Returns:
        Output [B, hidden, T] in BCT format.
    """
    y = ops.mul(x, mask)
    y = _conv1d_bct(y, weights["conv_1.weight"], weights.get("conv_1.bias"),
                    dilation=1, device_ref=device_ref)
    y = ops.relu(y)
    y = ops.mul(y, mask)
    y = _conv1d_bct(y, weights["conv_2.weight"], weights.get("conv_2.bias"),
                    dilation=1, device_ref=device_ref)
    return ops.mul(y, mask)


def build_attention_precomputed(
    x, mask, attn_bias_k, attn_bias_v, weights, config, device_ref
):
    """Multi-head attention with precomputed relative position biases.

    Biases are passed as graph values (inputs), computed externally in numpy
    by ``compute_rel_attention_biases``.

    Args:
        x: Input [B, hidden, T] in BCT format.
        mask: Mask [B, 1, T].
        attn_bias_k: Relative key bias [1, n_heads, T, T] graph value.
        attn_bias_v: Relative value contribution [1, n_heads, T, head_dim].
        weights: Dict with conv_q/k/v/o weight/bias numpy arrays.
        config: Dict with hidden_channels, n_heads.
        device_ref: MAX DeviceRef.

    Returns:
        Output [B, hidden, T] in BCT format.
    """
    hidden = config["hidden_channels"]
    n_heads = config["n_heads"]
    head_dim = hidden // n_heads

    q = _conv1d_bct(x, weights["conv_q.weight"], weights.get("conv_q.bias"),
                    dilation=1, device_ref=device_ref)
    k = _conv1d_bct(x, weights["conv_k.weight"], weights.get("conv_k.bias"),
                    dilation=1, device_ref=device_ref)
    v = _conv1d_bct(x, weights["conv_v.weight"], weights.get("conv_v.bias"),
                    dilation=1, device_ref=device_ref)

    T = x.shape[2]

    # Reshape to multi-head: [B, hidden, T] -> [B, n_heads, T, head_dim]
    q = ops.transpose(ops.reshape(q, [1, n_heads, head_dim, T]), 2, 3)
    k = ops.transpose(ops.reshape(k, [1, n_heads, head_dim, T]), 2, 3)
    v = ops.transpose(ops.reshape(v, [1, n_heads, head_dim, T]), 2, 3)

    # Scaled dot-product scores
    scale = ops.constant(
        np.array(1.0 / math.sqrt(head_dim), dtype=np.float32), device=device_ref
    )
    scores = ops.matmul(ops.mul(q, scale), ops.transpose(k, 2, 3))

    # Add relative position key bias
    scores = ops.add(scores, attn_bias_k)

    # Build and apply attention mask from sequence mask
    mask_col = ops.unsqueeze(mask, 3)  # [B, 1, T, 1]
    mask_row = ops.unsqueeze(mask, 2)  # [B, 1, 1, T]
    attn_mask = ops.mul(mask_col, mask_row)  # broadcasts to [B, n_heads, T, T]

    large_neg = ops.constant(np.array(-1e4, dtype=np.float32), device=device_ref)
    zero = ops.constant(np.array(0.0, dtype=np.float32), device=device_ref)
    scores = ops.where(ops.greater(attn_mask, zero), scores, large_neg)

    p_attn = ops.softmax(scores, axis=-1)

    # Weighted sum + relative value contribution
    output = ops.add(ops.matmul(p_attn, v), attn_bias_v)

    # Reshape back to [B, hidden, T]
    output = ops.reshape(ops.transpose(output, 2, 3), [1, hidden, T])

    # Output projection
    return _conv1d_bct(output, weights["conv_o.weight"], weights.get("conv_o.bias"),
                       dilation=1, device_ref=device_ref)


def build_encoder_precomputed(x, mask, attn_biases_k, attn_biases_v,
                              weights, config, device_ref):
    """6-layer transformer encoder with precomputed relative position biases.

    Each layer: attention + LN + FFN + LN (pre-norm residual).

    Args:
        x: Input [B, hidden, T] in BCT format.
        mask: Mask [B, 1, T].
        attn_biases_k: List of n_layers graph values, each [1, n_heads, T, T].
        attn_biases_v: List of n_layers graph values, each [1, n_heads, T, head_dim].
        weights: Dict with encoder weight keys (prefix-stripped).
        config: Dict with n_layers, hidden_channels, n_heads, etc.
        device_ref: MAX DeviceRef.

    Returns:
        Output [B, hidden, T] in BCT format.
    """
    n_layers = config["n_layers"]
    x = ops.mul(x, mask)

    for i in range(n_layers):
        attn_weights = {}
        for conv in ["conv_q", "conv_k", "conv_v", "conv_o"]:
            attn_weights[f"{conv}.weight"] = weights[f"attn_layers.{i}.{conv}.weight"]
            bias_key = f"attn_layers.{i}.{conv}.bias"
            if bias_key in weights:
                attn_weights[f"{conv}.bias"] = weights[bias_key]

        y = build_attention_precomputed(
            x, mask, attn_biases_k[i], attn_biases_v[i],
            attn_weights, config, device_ref,
        )

        x = build_layer_norm(
            ops.add(x, y),
            weights[f"norm_layers_1.{i}.gamma"],
            weights[f"norm_layers_1.{i}.beta"],
            device_ref,
        )
        x = ops.rebind(x, x.shape, message=f"encoder layer {i}: after attn+norm")

        ffn_weights = {
            "conv_1.weight": weights[f"ffn_layers.{i}.conv_1.weight"],
            "conv_1.bias": weights.get(f"ffn_layers.{i}.conv_1.bias"),
            "conv_2.weight": weights[f"ffn_layers.{i}.conv_2.weight"],
            "conv_2.bias": weights.get(f"ffn_layers.{i}.conv_2.bias"),
        }
        y = build_ffn(x, mask, ffn_weights, config, device_ref)

        x = build_layer_norm(
            ops.add(x, y),
            weights[f"norm_layers_2.{i}.gamma"],
            weights[f"norm_layers_2.{i}.beta"],
            device_ref,
        )
        x = ops.rebind(x, x.shape, message=f"encoder layer {i}: after ffn+norm")

    return ops.mul(x, mask)


# ---------------------------------------------------------------------------
# Numpy helpers: relative position attention bias computation
# ---------------------------------------------------------------------------


def _rel_to_abs_numpy(x, B, n_heads, T):
    """_relative_position_to_absolute_position in numpy.

    Input: [B, heads, T, 2T-1]  Output: [B, heads, T, T]
    """
    x_padded = np.pad(x, [[0, 0], [0, 0], [0, 0], [0, 1]])
    x_flat = x_padded.reshape(B, n_heads, T * 2 * T)
    x_flat = np.pad(x_flat, [[0, 0], [0, 0], [0, T - 1]])
    x_final = x_flat.reshape(B, n_heads, T + 1, 2 * T - 1)
    return x_final[:, :, :T, T - 1:]


def _abs_to_rel_numpy(x, B, n_heads, T):
    """_absolute_position_to_relative_position in numpy.

    Input: [B, heads, T, T]  Output: [B, heads, T, 2T-1]
    """
    x_padded = np.pad(x, [[0, 0], [0, 0], [0, 0], [0, T - 1]])
    x_flat = x_padded.reshape(B, n_heads, T * T + T * (T - 1))
    x_flat = np.pad(x_flat, [[0, 0], [0, 0], [T, 0]])
    x_final = x_flat.reshape(B, n_heads, T, 2 * T)
    return x_final[:, :, :, 1:]


def _get_rel_embeddings_numpy(embeddings, length, window_size):
    """Get relative embeddings for a given sequence length."""
    pad_length = max(length - (window_size + 1), 0)
    start = max((window_size + 1) - length, 0)
    end = start + 2 * length - 1
    if pad_length > 0:
        embeddings = np.pad(embeddings, [[0, 0], [pad_length, pad_length], [0, 0]])
    return embeddings[:, start:end]


def _layer_norm_numpy(x, gamma, beta, eps=1e-5):
    """Channel-wise LayerNorm in numpy. x: [B, C, T], gamma/beta: [C]."""
    x_t = x.transpose(0, 2, 1)
    mean = x_t.mean(axis=-1, keepdims=True)
    var = x_t.var(axis=-1, keepdims=True)
    x_norm = (x_t - mean) / np.sqrt(var + eps) * gamma + beta
    return x_norm.transpose(0, 2, 1)


def _conv1d_numpy(x_bct, weight, bias, padding=0):
    """Simple Conv1d in numpy. x: [B, C_in, T], weight: [C_out, C_in, K]."""
    B, C_in, T = x_bct.shape
    C_out, _, K = weight.shape
    if padding > 0:
        x_bct = np.pad(x_bct, [[0, 0], [0, 0], [padding, padding]])
    T_out = x_bct.shape[2] - K + 1
    # PyTorch weight layout: [C_out, C_in, K]
    # w.reshape(C_out, C_in*K) has order (c_in, k) in C layout
    # So cols must match: for each (c_in, k) pair, store x[:, c_in, t+k]
    cols = np.zeros((B, C_in * K, T_out), dtype=x_bct.dtype)
    for c in range(C_in):
        for k in range(K):
            cols[:, c * K + k, :] = x_bct[:, c, k:k + T_out]
    w_mat = weight.reshape(C_out, -1)
    out = np.einsum("oi,bit->bot", w_mat, cols)
    if bias is not None:
        out = out + bias.reshape(1, -1, 1)
    return out


def _ffn_numpy(x, mask, c1_w, c1_b, c2_w, c2_b):
    """FFN in numpy. x: [B, hidden, T]."""
    padding = (c1_w.shape[2] - 1) // 2
    y = _conv1d_numpy(x * mask, c1_w, c1_b, padding=padding)
    y = np.maximum(y, 0)
    y = _conv1d_numpy(y * mask, c2_w, c2_b, padding=padding)
    return y * mask


def compute_rel_attention_biases(
    vits_weights: dict[str, np.ndarray],
    x_bct: np.ndarray,
    mask_np: np.ndarray,
    config: dict,
) -> tuple[list[np.ndarray], list[np.ndarray]]:
    """Compute per-layer relative position attention biases in numpy.

    Runs the full encoder forward pass in numpy to produce the Q and p_attn
    values at each layer, then computes the relative position key bias and
    value contribution.  These are passed as graph inputs to the MAX graph.

    Args:
        vits_weights: Dict with encoder.* weight keys.
        x_bct: Input to the encoder [B, hidden, T] float32.
        mask_np: Mask [B, 1, T] float32.
        config: Dict with hidden_channels, n_heads, n_layers, window_size.

    Returns:
        (biases_k, biases_v):
          biases_k: list of n_layers arrays, each [1, n_heads, T, T]
          biases_v: list of n_layers arrays, each [1, n_heads, T, head_dim]
    """
    hidden = config["hidden_channels"]
    n_heads = config["n_heads"]
    head_dim = hidden // n_heads
    n_layers = config["n_layers"]
    window_size = config["window_size"]

    x = x_bct * mask_np
    B, _, T = x.shape

    biases_k, biases_v = [], []

    for i in range(n_layers):
        # Q, K, V projections (Conv1d k=1 = matrix multiply: w[:, :, 0] @ x)
        def _proj(name):
            w = vits_weights[f"encoder.attn_layers.{i}.{name}.weight"]  # [C_out, C_in, 1]
            b = vits_weights.get(f"encoder.attn_layers.{i}.{name}.bias")
            out = np.matmul(w[:, :, 0], x[0])  # [hidden, hidden] @ [hidden, T] -> [hidden, T]
            out = out[np.newaxis]  # [1, hidden, T]
            if b is not None:
                out = out + b.reshape(1, -1, 1)
            return out.reshape(B, n_heads, head_dim, T).transpose(0, 1, 3, 2)

        q, k, v = _proj("conv_q"), _proj("conv_k"), _proj("conv_v")
        scale = 1.0 / math.sqrt(head_dim)
        q_scaled = q * scale

        # Content-based scores
        scores = np.matmul(q_scaled, k.transpose(0, 1, 3, 2))

        # Relative key bias
        emb_k = vits_weights[f"encoder.attn_layers.{i}.emb_rel_k"]
        rel_emb_k = _get_rel_embeddings_numpy(emb_k, T, window_size)
        rel_logits = np.matmul(q_scaled, rel_emb_k.transpose(0, 2, 1))
        rel_k_bias = _rel_to_abs_numpy(rel_logits, B, n_heads, T)
        biases_k.append(rel_k_bias.astype(np.float32))

        # Full attention for p_attn (needed for value bias)
        scores_full = scores + rel_k_bias
        attn_mask = mask_np[:, :, :, np.newaxis] * mask_np[:, :, np.newaxis, :]
        scores_full = np.where(attn_mask > 0, scores_full, -1e4)
        scores_max = np.max(scores_full, axis=-1, keepdims=True)
        exp_scores = np.exp(scores_full - scores_max)
        p_attn = exp_scores / np.sum(exp_scores, axis=-1, keepdims=True)

        # Relative value bias
        emb_v = vits_weights[f"encoder.attn_layers.{i}.emb_rel_v"]
        rel_emb_v = _get_rel_embeddings_numpy(emb_v, T, window_size)
        rel_weights = _abs_to_rel_numpy(p_attn, B, n_heads, T)
        rel_v_output = np.matmul(rel_weights, rel_emb_v)
        biases_v.append(rel_v_output.astype(np.float32))

        # Full layer output (to propagate x to next layer)
        output = np.matmul(p_attn, v) + rel_v_output
        output = output.transpose(0, 1, 3, 2).reshape(B, hidden, T)

        o_w = vits_weights[f"encoder.attn_layers.{i}.conv_o.weight"]  # [hidden, hidden, 1]
        o_b = vits_weights.get(f"encoder.attn_layers.{i}.conv_o.bias")
        y_attn = np.matmul(o_w[:, :, 0], output[0])[np.newaxis]  # [1, hidden, T]
        if o_b is not None:
            y_attn = y_attn + o_b.reshape(1, -1, 1)

        x = _layer_norm_numpy(
            x + y_attn,
            vits_weights[f"encoder.norm_layers_1.{i}.gamma"],
            vits_weights[f"encoder.norm_layers_1.{i}.beta"],
        )

        y_ffn = _ffn_numpy(
            x, mask_np,
            vits_weights[f"encoder.ffn_layers.{i}.conv_1.weight"],
            vits_weights.get(f"encoder.ffn_layers.{i}.conv_1.bias"),
            vits_weights[f"encoder.ffn_layers.{i}.conv_2.weight"],
            vits_weights.get(f"encoder.ffn_layers.{i}.conv_2.bias"),
        )

        x = _layer_norm_numpy(
            x + y_ffn,
            vits_weights[f"encoder.norm_layers_2.{i}.gamma"],
            vits_weights[f"encoder.norm_layers_2.{i}.beta"],
        )

    return biases_k, biases_v


# ---------------------------------------------------------------------------
# Full enc_p graph builder
# ---------------------------------------------------------------------------


def build_enc_p_graph(weights, config, device="cpu", batch_size=1, max_T=2000):
    """Build the VITS TextEncoder (enc_p) as a MAX Graph.

    Relative position attention biases are computed in numpy outside the graph
    (via ``compute_rel_attention_biases``) and passed as additional inputs.
    This avoids complex dynamic-T reshape operations inside the graph.

    Args:
        weights: Dict of numpy arrays with enc_p weight keys.
        config: Dict with hidden_channels, filter_channels, n_heads, n_layers,
                kernel_size, window_size, out_channels.
        device: "cpu" or "gpu".
        batch_size: Batch size.
        max_T: Maximum sequence length for the arange constant.

    Returns:
        MAX Graph with inputs:
          - features [B, T, 768] float32
          - pitch [B, T] int32 (quantised pitch 0-255)
          - lengths [B] int32
          - n_layers pairs of (attn_bias_k [1, n_heads, T, T],
                               attn_bias_v [1, n_heads, T, head_dim])
        Outputs:
          - mean [B, out_channels, T] float32
          - logvar [B, out_channels, T] float32
          - mask [B, 1, T] float32
    """
    hidden = config["hidden_channels"]
    n_heads = config["n_heads"]
    head_dim = hidden // n_heads
    n_layers = config["n_layers"]
    out_channels = config["out_channels"]

    dev = DeviceRef.CPU() if device == "cpu" else DeviceRef(device)
    T = Dim("T")

    input_types = [
        TensorType(DType.float32, [batch_size, T, 768], dev),   # features
        TensorType(DType.int32, [batch_size, T], dev),           # pitch
        TensorType(DType.int32, [batch_size], dev),              # lengths
    ]
    for _ in range(n_layers):
        input_types.append(TensorType(DType.float32, [1, n_heads, T, T], dev))
        input_types.append(TensorType(DType.float32, [1, n_heads, T, head_dim], dev))

    with Graph("vits_enc_p", input_types=input_types) as g:
        inputs = g.inputs
        features, pitch, lengths = inputs[0], inputs[1], inputs[2]

        attn_biases_k = [inputs[3 + 2 * i] for i in range(n_layers)]
        attn_biases_v = [inputs[3 + 2 * i + 1] for i in range(n_layers)]

        # 1. Phone embedding: Linear(768, hidden)
        emb_phone_w = ops.constant(
            weights["emb_phone.weight"].astype(np.float32), device=dev
        )
        x = ops.matmul(features, ops.transpose(emb_phone_w, 0, 1))
        if "emb_phone.bias" in weights:
            x = ops.add(x, ops.constant(
                weights["emb_phone.bias"].reshape(1, 1, -1).astype(np.float32),
                device=dev,
            ))

        # 2. Pitch embedding: Embedding(256, hidden) via gather from identity
        emb_pitch_w = ops.constant(
            weights["emb_pitch.weight"].astype(np.float32), device=dev
        )
        eye_256 = ops.constant(np.eye(256, dtype=np.float32), device=dev)
        pitch_one_hot = ops.gather(eye_256, pitch, axis=0)  # [B, T, 256]
        x = ops.add(x, ops.matmul(pitch_one_hot, emb_pitch_w))

        # 3. Scale by sqrt(hidden) and LeakyReLU(0.1)
        x = ops.mul(x, ops.constant(
            np.array(math.sqrt(hidden), dtype=np.float32), device=dev
        ))
        x = leaky_relu(x, alpha=0.1, device_ref=dev)

        # 4. Transpose to BCT
        x = ops.transpose(x, 1, 2)

        # 5. Compute mask from lengths
        arange_const = ops.constant(
            np.arange(max_T, dtype=np.int32).reshape(1, -1), device=dev
        )
        arange_sliced = ops.slice_tensor(arange_const, [slice(None), slice(0, T)])
        lengths_expanded = ops.unsqueeze(lengths, 1)
        mask_bool = ops.greater(lengths_expanded, arange_sliced)
        one = ops.constant(np.array(1.0, dtype=np.float32), device=dev)
        zero = ops.constant(np.array(0.0, dtype=np.float32), device=dev)
        mask = ops.unsqueeze(ops.where(mask_bool, one, zero), 1)  # [B, 1, T]

        # 6. Encoder (6-layer transformer)
        encoder_weights = {
            k[len("encoder."):]: v
            for k, v in weights.items()
            if k.startswith("encoder.")
        }
        x = build_encoder_precomputed(
            x, mask, attn_biases_k, attn_biases_v,
            encoder_weights, config, dev,
        )

        # 7. Projection: Conv1d(hidden, 2*out_channels, k=1)
        stats = ops.mul(
            _conv1d_bct(x, weights["proj.weight"], weights.get("proj.bias"),
                        dilation=1, device_ref=dev),
            mask,
        )

        # 8. Split into mean and logvar
        mean = ops.slice_tensor(
            stats, [slice(None), slice(0, out_channels), slice(None)]
        )
        logvar = ops.slice_tensor(
            stats, [slice(None), slice(out_channels, 2 * out_channels), slice(None)]
        )

        g.output(mean, logvar, mask)

    return g
