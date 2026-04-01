"""VITS flow graph building blocks for MAX Engine.

Implements the normalizing flow (reverse pass) from the RVC v2 VITS model:
  - WaveNet (gated-activation residual network with speaker conditioning)
  - ResidualCouplingLayer (affine coupling, mean_only=True)
  - Flip (channel reversal)
  - Full flow graph (4 coupling layers + 4 flips, iterated in reverse)

All convolutions use the im2col + matmul workaround from _hifigan_graph.py.
The flow operates in channel-first [B, C, T] format internally, but the
underlying conv1d expects NHWC [B, T, 1, C], so we transpose at boundaries.
"""

from __future__ import annotations

import numpy as np
from max.graph import Graph, TensorType, ops, DeviceRef, Dim
from max.dtype import DType

from models._hifigan_graph import conv1d


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
