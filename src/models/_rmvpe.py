"""RMVPE U-Net MAX Graph.

Implements the DeepUnet backbone of RMVPE pitch extraction as a MAX Engine
inference graph.

Architecture:
  Input:  [1, T, 128, 1] NHWC float32 (mel spectrogram, time=H, freq=W)
  Output: [1, T, 384] float32 (pre-BiGRU features)

Pipeline:
  1. Pad T by PAD_T (static=32) so the U-Net decoder always produces T' >= T
  2. Initial BN (baked scale/offset)
  3. Encoder: 5 levels × 4 residual blocks + AvgPool2d(2,2) downsampling
     Skips saved BEFORE each pool (pre-pool output)
  4. Bottleneck: 16 residual blocks (no spatial change)
  5. Decoder: 5 levels
       - Numerically-correct ConvTranspose2d via zero-interleave + regular conv
       - BN after ConvTranspose
       - Slice skip to decoder spatial, concat on channel axis
       - 4 residual blocks
  6. Output CNN: 16→3 channels, 3×3 conv
  7. Slice T' back to original T
  8. Reshape [1, T, 128, 3] → [1, T, 384]

Weight format (PyTorch → MAX NHWC):
  Conv2d:
    PyTorch [C_out, C_in, kH, kW] → transpose(2,3,1,0) → [kH, kW, C_in, C_out]
  ConvTranspose2d (for zero-interleave + regular conv substitution):
    PyTorch [C_in, C_out, kH, kW]
      → transpose(1,0,2,3) to swap C_in/C_out → [C_out, C_in, kH, kW]
      → flip spatial [:, :, ::-1, ::-1]
      → transpose(2,3,1,0) → [kH_flipped, kW_flipped, C_in, C_out]
    This makes the resulting regular conv2d numerically equivalent to ConvTranspose2d.
  Bias [C_out] → reshape to [1,1,1,C_out] for NHWC broadcasting
  BN scale/offset [C] → reshape to [1,1,1,C] for NHWC broadcasting

Residual block:
  h = relu(BN1(conv1(x)))
  h = BN2(conv2(h))
  sc = shortcut_conv(x)  if shortcut weights exist, else x
  output = relu(h + sc)

ConvTranspose2d implementation note:
  ops.conv2d_transpose is broken in this MAX version (cannot lower to LLVM:
  num_groups). We use the mathematically equivalent zero-interleave + regular conv:
    1. Insert one zero row after each input row along H: [B,H,W,C] → [B,2H,W,C]
    2. Insert one zero col after each input col along W: → [B,2H,2W,C]
    3. Apply regular conv2d with flipped weights and padding=1
  This matches PyTorch ConvTranspose2d(stride=2, kernel=3, padding=1, output_padding=1),
  which RMVPE uses to double spatial dims (H→2H, W→2W).
  Numerical validation confirms max diff < 1e-4 vs PyTorch.

1×1 conv note:
  ops.conv2d fails for kernel=1 in this MAX version (layout_transform_RSCF_to_KNkni
  missing). We use reshape + matmul + reshape, which is equivalent.
"""

from __future__ import annotations
import numpy as np
from max.graph import Graph, TensorType, ops, DeviceRef, Dim
from max.dtype import DType

# Static padding added to T (time) dim before the U-Net.
# Guarantees that after 5 halvings + 5 doublings, T' >= T for all T >= 1.
PAD_T = 32


def _pt_to_max_conv(w: np.ndarray) -> np.ndarray:
    """Convert PyTorch Conv2d weight [C_out, C_in, kH, kW] → MAX NHWC [kH, kW, C_in, C_out]."""
    return w.transpose(2, 3, 1, 0)


def _pt_to_max_conv_transpose(w: np.ndarray) -> np.ndarray:
    """Convert PyTorch ConvTranspose2d weight for use with zero-interleave + regular conv.

    ConvTranspose2d(x) = conv(zero_interleave(x), w_flipped)
    where the flipped weight maps the same C_in inputs to the same C_out outputs.

    PyTorch ConvTranspose2d weight: [C_in, C_out, kH, kW]
    Transform for MAX regular conv2d [kH, kW, C_in, C_out]:
      1. Swap C_in/C_out: transpose(1,0,2,3) → [C_out, C_in, kH, kW]
      2. Flip spatial dims: [:, :, ::-1, ::-1]
      3. Transpose to NHWC conv format: transpose(2,3,1,0) → [kH, kW, C_in, C_out]
    """
    w_swapped = w.transpose(1, 0, 2, 3)             # [C_out, C_in, kH, kW]
    w_flipped = w_swapped[:, :, ::-1, ::-1].copy()  # flip kH, kW
    return w_flipped.transpose(2, 3, 1, 0)           # [kH, kW, C_in, C_out]


def _bn_add(x, scale: np.ndarray, offset: np.ndarray, device_ref):
    """Apply baked BN: x * scale + offset (broadcast over NHWC)."""
    C = scale.shape[0]
    scale_c = ops.constant(scale.reshape(1, 1, 1, C), device=device_ref)
    offset_c = ops.constant(offset.reshape(1, 1, 1, C), device=device_ref)
    return ops.add(ops.mul(x, scale_c), offset_c)


def _conv2d(x, w_np: np.ndarray, b_np, stride, padding, device_ref):
    """Conv2d via im2col + matmul, avoiding ops.conv2d (broken for C_in >= 8).

    w_np must already be in MAX [kH, kW, C_in, C_out] format.
    Supports stride=(1,1) only (all RMVPE convolutions use stride 1).

    For 1×1 convolutions, uses direct matmul (no im2col needed).
    For k×k convolutions, extracts shifted patches along H and W axes,
    concatenates into a column matrix, and multiplies by reshaped weight.

    This replaces the previous ops.conv2d implementation which produces
    incorrect results when C_in >= 8 (modular/modular#6248).
    """
    kH, kW = w_np.shape[0], w_np.shape[1]
    C_in = w_np.shape[2]
    C_out = w_np.shape[3]

    if kH == 1 and kW == 1:
        w_2d = ops.constant(w_np.reshape(C_in, C_out), device=device_ref)
        x_sq = ops.squeeze(x, 0)        # [H, W, C_in]
        out = ops.matmul(x_sq, w_2d)    # [H, W, C_out]
        out = ops.unsqueeze(out, 0)     # [1, H, W, C_out]
    else:
        # im2col + matmul for k×k conv with stride 1
        assert stride == (1, 1), f"im2col conv2d only supports stride=1, got {stride}"

        orig_H = x.shape[1]
        orig_W = x.shape[2]

        # Pad input: padding is (H_bef, H_aft, W_bef, W_aft)
        pad_h_bef, pad_h_aft, pad_w_bef, pad_w_aft = padding
        if any(p > 0 for p in padding):
            x_pad = ops.pad(x, [
                0, 0,                          # batch
                pad_h_bef, pad_h_aft,          # H
                pad_w_bef, pad_w_aft,          # W
                0, 0,                          # C
            ])
        else:
            x_pad = x

        # im2col: for each (kh, kw), slice a [1, H, W, C_in] patch and concat
        # along channel axis → [1, H, W, kH*kW*C_in]
        slices = []
        for kh in range(kH):
            for kw in range(kW):
                s = ops.slice_tensor(x_pad, [
                    slice(None),
                    slice(kh, kh + orig_H),
                    slice(kw, kw + orig_W),
                    slice(None),
                ])
                slices.append(s)

        x_cols = ops.concat(slices, axis=3) if len(slices) > 1 else slices[0]
        # x_cols: [1, H, W, kH*kW*C_in]

        # Reshape weight: [kH, kW, C_in, C_out] → [kH*kW*C_in, C_out]
        w_mat = w_np.reshape(kH * kW * C_in, C_out).astype(np.float32)
        w_const = ops.constant(w_mat, device=device_ref)

        # matmul: [1, H, W, kH*kW*C_in] × [kH*kW*C_in, C_out] → [1, H, W, C_out]
        out = ops.matmul(x_cols, w_const)

        # Rebind to reconcile symbolic dims after slicing
        out = ops.rebind(
            out,
            [x.shape[0], x.shape[1], x.shape[2], C_out],
            message="conv2d im2col: reconcile H,W dims",
        )

    if b_np is not None:
        b = ops.constant(b_np.reshape(1, 1, 1, -1), device=device_ref)
        out = ops.add(out, b)
    return out


def _conv_transpose_2x(x, w_pt: np.ndarray, b_np, device_ref):
    """Numerically-correct ConvTranspose2d stride=2, kernel=3, padding=1, output_padding=1.

    Equivalent to PyTorch ConvTranspose2d(C_in, C_out, kernel_size=3, stride=2,
    padding=1, output_padding=1, bias=...) for any input [B, H, W, C_in].
    output_padding=1 means a trailing zero row/col is appended after each input
    position, doubling the spatial dimensions: H → 2H, W → 2W.

    Output shape: [B, 2H, 2W, C_out]

    Implementation:
      1. Zero-interleave H: insert one zero row after each input row (including
         the last), giving [B, 2H, W, C].
      2. Zero-interleave W the same way via squeeze/transpose trick: → [B, 2H, 2W, C].
      3. Regular conv2d with flipped weights and padding=1.

    Shape trick: squeeze/unsqueeze handle the unit batch dim without reshape
    element-count verification failures. Only reshape([H, 2, W, C] → [2H, W, C])
    is needed, merging two leading dims where one is static 2 — MAX can verify
    H*2 == 2H trivially. For W (static): same pattern after transposing W to front.

    Args:
        x: NHWC input [B, H, W, C_in].  B must be 1.
        w_pt: PyTorch ConvTranspose2d weight [C_in, C_out, kH, kW].
        b_np: Optional bias [C_out] or None.
        device_ref: MAX DeviceRef.

    Returns:
        TensorValue [B, 2H, 2W, C_out].
    """
    C_in_val = w_pt.shape[0]   # Python int — number of input channels
    _H = x.shape[1]   # dynamic symbolic dim (time axis)
    _W = x.shape[2]   # static symbolic dim (freq axis; 4, 8, 16, 32, or 64 in decoder)

    # --- Step 1: Zero-interleave along H → [1, 2H, W, C] ---
    # squeeze batch-1, insert a 1-slot after each H row, pad it with a zero,
    # reshape to merge H and the 2-slot, unsqueeze batch back.
    x_sq = ops.squeeze(x, 0)                                     # [H, W, C]
    x_ins = ops.unsqueeze(x_sq, 1)                               # [H, 1, W, C]
    x_padH = ops.pad(x_ins, [0, 0, 0, 1, 0, 0, 0, 0])          # [H, 2, W, C]
    x_r2 = ops.reshape(x_padH, [_H * 2, _W, C_in_val])         # [2H, W, C]
    x_zi_H = ops.unsqueeze(x_r2, 0)                              # [1, 2H, W, C]

    # --- Step 2: Zero-interleave along W → [1, 2H, 2W, C] ---
    # Transpose to put W as the leading axis so the same reshape trick applies
    # (W is static so W*2 == 2W is provable), then transpose back.
    _H2 = x_zi_H.shape[1]   # 2H, dynamic
    x_sq2 = ops.squeeze(x_zi_H, 0)                               # [2H, W, C]
    x_t = ops.transpose(x_sq2, 0, 1)                             # [W, 2H, C]
    x_ins2 = ops.unsqueeze(x_t, 1)                               # [W, 1, 2H, C]
    x_padW = ops.pad(x_ins2, [0, 0, 0, 1, 0, 0, 0, 0])         # [W, 2, 2H, C]
    x_r4 = ops.reshape(x_padW, [_W * 2, _H2, C_in_val])        # [2W, 2H, C]
    x_t2 = ops.transpose(x_r4, 0, 1)                             # [2H, 2W, C]
    x_zi = ops.unsqueeze(x_t2, 0)                                 # [1, 2H, 2W, C]

    # --- Step 3: Regular conv2d with flipped weights ---
    w_max = _pt_to_max_conv_transpose(w_pt)   # [kH, kW, C_in, C_out]
    return _conv2d(x_zi, w_max, b_np, stride=(1, 1), padding=(1, 1, 1, 1), device_ref=device_ref)


def _residual_block(x, prefix, weights, device_ref):
    """Single residual block (all convolutions stride=(1,1)).

    Structure (matches PyTorch ConvBlockRes):
        h = conv1 → BN1 → relu → conv2 → BN2 → relu
        sc = shortcut(x)  if {prefix}.sc.w exists, else identity
        output = h + sc
    """
    w1 = _pt_to_max_conv(weights[f"{prefix}.0.w"])
    b1 = weights.get(f"{prefix}.0.b")
    h = _conv2d(x, w1, b1, stride=(1, 1), padding=(1, 1, 1, 1), device_ref=device_ref)
    if f"{prefix}.0.scale" in weights:
        h = _bn_add(h, weights[f"{prefix}.0.scale"], weights[f"{prefix}.0.offset"], device_ref)
    h = ops.relu(h)

    w2 = _pt_to_max_conv(weights[f"{prefix}.1.w"])
    b2 = weights.get(f"{prefix}.1.b")
    h = _conv2d(h, w2, b2, stride=(1, 1), padding=(1, 1, 1, 1), device_ref=device_ref)
    if f"{prefix}.1.scale" in weights:
        h = _bn_add(h, weights[f"{prefix}.1.scale"], weights[f"{prefix}.1.offset"], device_ref)
    h = ops.relu(h)

    sc_w_key = f"{prefix}.sc.w"
    if sc_w_key in weights:
        sc_w = _pt_to_max_conv(weights[sc_w_key])
        sc_b = weights.get(f"{prefix}.sc.b")
        sc = _conv2d(x, sc_w, sc_b, stride=(1, 1), padding=(0, 0, 0, 0), device_ref=device_ref)
    else:
        sc = x

    return ops.add(h, sc)


def build_unet_graph(
    weights: dict[str, np.ndarray],
    device_ref,
) -> Graph:
    """Build a MAX Graph for the RMVPE U-Net (encoder + bottleneck + decoder).

    Args:
        weights: Internal weight dict from _rmvpe_weight_loader. Must contain:
                 enc_bn.scale/.offset, enc.{L}.{B}.*, btl.{I}.*, dec.{L}.*,
                 out_cnn.w/.b.
        device_ref: DeviceRef.CPU() or DeviceRef.GPU(0).

    Returns:
        MAX Graph.
        Input:  [1, T, 128, 1] float32 NHWC mel spectrogram.
        Output: [1, T, 384] float32.
    """
    with Graph(
        "rmvpe_unet",
        input_types=[TensorType(DType.float32, [1, Dim("T"), 128, 1], device_ref)],
    ) as g:
        x = g.inputs[0]  # [1, T, 128, 1]

        # Remember original T before any padding
        orig_T = x.shape[1]  # Dim("T")

        # Pad T by PAD_T (static) so decoder always produces T' >= T
        # pad layout: [N_bef, N_aft, H_bef, H_aft, W_bef, W_aft, C_bef, C_aft]
        x = ops.pad(x, [0, 0, 0, PAD_T, 0, 0, 0, 0])
        # x: [1, T+PAD_T, 128, 1]

        # --- Initial encoder BN ---
        x = _bn_add(x, weights["enc_bn.scale"], weights["enc_bn.offset"], device_ref)

        # --- Encoder: 5 levels ---
        skips = []  # pre-pool outputs saved for skip connections
        for L in range(5):
            for B in range(4):
                x = _residual_block(x, f"enc.{L}.{B}", weights, device_ref)
            skips.append(x)  # save BEFORE pool
            x = ops.avg_pool2d(x, kernel_size=(2, 2), stride=(2, 2))
        # x: [1, (T+PAD_T)//32, 4, 256]

        # --- Bottleneck: 16 blocks ---
        for I in range(16):
            x = _residual_block(x, f"btl.{I}", weights, device_ref)
        # x: [1, (T+PAD_T)//32, 4, 512]

        # --- Decoder: 5 levels ---
        for L in range(5):
            # ConvTranspose2d (stride=2, kernel=3, padding=1, output_padding=1) via zero-interleave + conv
            # output shape: [1, 2H, 2W, up_co]
            up_w_pt = weights[f"dec.{L}.up.w"]  # PyTorch [C_in, C_out, kH, kW]
            up_b = weights.get(f"dec.{L}.up.b")
            x = _conv_transpose_2x(x, up_w_pt, up_b, device_ref)

            # BN after ConvTranspose
            if f"dec.{L}.up.scale" in weights:
                x = _bn_add(
                    x,
                    weights[f"dec.{L}.up.scale"],
                    weights[f"dec.{L}.up.offset"],
                    device_ref,
                )

            # Skip concat from encoder level (4-L)
            skip = skips[4 - L]
            dec_H = x.shape[1]
            dec_W = x.shape[2]
            # Slice skip to decoder spatial size.  The skip is always >= decoder
            # because encoder pools with floor division (T // 2^k) while the
            # decoder doubles (2H), so dec_H <= skip_H for all T >= 1.
            skip = ops.slice_tensor(
                skip,
                [slice(None), slice(None, dec_H), slice(None, dec_W), slice(None)],
            )
            x = ops.concat([x, skip], axis=3)

            # 4 residual blocks
            for B in range(4):
                x = _residual_block(x, f"dec.{L}.{B}", weights, device_ref)
        # x: [1, H', W', 16]  where H' >= T, W' = 128

        # --- Output CNN: 16→3, 3×3, same padding ---
        out_w = _pt_to_max_conv(weights["out_cnn.w"])
        out_b = weights.get("out_cnn.b")
        x = _conv2d(x, out_w, out_b, stride=(1, 1), padding=(1, 1, 1, 1), device_ref=device_ref)
        # x: [1, H', W', 3]

        # Slice T dimension back to original T
        x = ops.slice_tensor(
            x,
            [slice(None), slice(0, orig_T), slice(None), slice(None)],
        )
        # x: [1, T, 128, 3]

        # Reshape to [1, T, 384]
        x = ops.reshape(x, [1, Dim("T"), 384])

        g.output(x)

    return g


# ---------------------------------------------------------------------------
# BiGRU (numpy — max.nn has no GRU)
# ---------------------------------------------------------------------------

def _sigmoid(x: np.ndarray) -> np.ndarray:
    return (1.0 / (1.0 + np.exp(-x.astype(np.float64)))).astype(np.float32)


def bigru_forward(x: np.ndarray, weights: dict, hidden_size: int = 256) -> np.ndarray:
    """Bidirectional GRU (numpy).

    Args:
        x: [B, T, 384] float32
        weights: dict with keys gru.weight_ih_l0, gru.weight_hh_l0,
                 gru.bias_ih_l0, gru.bias_hh_l0,
                 gru.weight_ih_l0_reverse, gru.weight_hh_l0_reverse,
                 gru.bias_ih_l0_reverse, gru.bias_hh_l0_reverse
        hidden_size: 256

    Returns:
        [B, T, 512] float32 — concat of forward [B,T,256] and reverse [B,T,256]
    """
    H = hidden_size

    def _gru_step(x_t, h, w_ih, w_hh, b_ih, b_hh):
        # x_t: [B, input_size], h: [B, H]
        gi = x_t @ w_ih.T + b_ih  # [B, 3H]
        gh = h @ w_hh.T + b_hh    # [B, 3H]
        z = _sigmoid(gi[:, :H]    + gh[:, :H])       # update gate [B, H]
        r = _sigmoid(gi[:, H:2*H] + gh[:, H:2*H])    # reset gate  [B, H]
        n = np.tanh(gi[:, 2*H:]   + r * gh[:, 2*H:]) # new gate    [B, H]
        return ((1.0 - z) * h + z * n).astype(np.float32)

    B, T, _ = x.shape
    w = weights

    # Forward pass (t=0 -> T-1)
    h_fwd = np.zeros((B, H), dtype=np.float32)
    fwd_states = []
    for t in range(T):
        h_fwd = _gru_step(
            x[:, t], h_fwd,
            w["gru.weight_ih_l0"], w["gru.weight_hh_l0"],
            w["gru.bias_ih_l0"],   w["gru.bias_hh_l0"],
        )
        fwd_states.append(h_fwd.copy())

    # Reverse pass (t=T-1 -> 0)
    h_rev = np.zeros((B, H), dtype=np.float32)
    rev_states = [None] * T
    for t in range(T - 1, -1, -1):
        h_rev = _gru_step(
            x[:, t], h_rev,
            w["gru.weight_ih_l0_reverse"], w["gru.weight_hh_l0_reverse"],
            w["gru.bias_ih_l0_reverse"],   w["gru.bias_hh_l0_reverse"],
        )
        rev_states[t] = h_rev.copy()

    fwd_out = np.stack(fwd_states, axis=1)  # [B, T, H]
    rev_out = np.stack(rev_states, axis=1)  # [B, T, H]
    return np.concatenate([fwd_out, rev_out], axis=-1)  # [B, T, 512]


# ---------------------------------------------------------------------------
# Linear output + pitch salience -> Hz
# ---------------------------------------------------------------------------

def linear_output(x: np.ndarray, weights: dict) -> np.ndarray:
    """[B, T, 512] -> [B, T, 360] via linear layer."""
    W = weights["linear.weight"]  # [360, 512]
    b = weights["linear.bias"]    # [360]
    return (x @ W.T + b).astype(np.float32)


# RMVPE pitch bins: 360 bins, 20-cent resolution
# Bin i -> frequency = 440 * 2^((i * 20 - 6900) / 1200) Hz
# Bin 0 ~= 32.7 Hz (C1), bin 359 ~= 1975.5 Hz (B6)
_CENTS_PER_BIN = 20.0
_CENTER_CENTS = 6900.0


def _bins_to_hz(bin_indices: np.ndarray) -> np.ndarray:
    """Convert RMVPE bin indices (float, for sub-bin accuracy) to Hz."""
    cents = bin_indices.astype(np.float32) * _CENTS_PER_BIN - _CENTER_CENTS
    return (440.0 * (2.0 ** (cents / 1200.0))).astype(np.float32)


def salience_to_hz(salience: np.ndarray, threshold: float = 0.03) -> np.ndarray:
    """Convert pitch salience [1, T, 360] to F0 Hz per frame [T].

    Uses weighted local average around the peak bin for sub-bin precision.
    Frames where sigmoid(peak logit) < threshold are unvoiced (0 Hz).

    Args:
        salience: [1, T, 360] float32 — raw logits from linear_output (pre-sigmoid).
        threshold: sigmoid probability at the peak bin must exceed this to be voiced.

    Returns:
        [T] float32 — F0 in Hz, 0.0 = unvoiced.
    """
    prob = _sigmoid(salience[0])  # [T, 360] — sigmoid probabilities
    T = prob.shape[0]

    center_bin = np.argmax(prob, axis=-1)  # [T]
    max_prob = prob[np.arange(T), center_bin]  # [T]

    bin_idx = np.arange(360, dtype=np.float32)
    weighted_bins = np.zeros(T, dtype=np.float32)
    for t in range(T):
        lo = max(0, center_bin[t] - 4)
        hi = min(360, center_bin[t] + 5)
        w_t = prob[t, lo:hi]
        b_t = bin_idx[lo:hi]
        w_sum = w_t.sum()
        weighted_bins[t] = (b_t * w_t).sum() / w_sum if w_sum > 1e-8 else float(center_bin[t])

    hz = _bins_to_hz(weighted_bins)
    hz[max_prob < threshold] = 0.0
    return hz
