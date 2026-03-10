"""RMVPE U-Net MAX Graph.

Implements the DeepUnet backbone of RMVPE pitch extraction as a MAX Engine
inference graph.

Architecture:
  Input:  [1, T, 128, 1] NHWC float32 (mel spectrogram, time=H, freq=W)
  Output: [1, T, 384] float32 (pre-BiGRU features)

Pipeline:
  1. Pad T by PAD_T (static=32) so the U-Net always has T' >= T+1
  2. Initial BN (baked scale/offset)
  3. Encoder: 5 levels × 4 residual blocks + AvgPool2d(2,2) downsampling
     Skips saved BEFORE each pool (pre-pool spatial size)
  4. Bottleneck: 16 residual blocks (no spatial change)
  5. Decoder: 5 levels
       - 2× nearest upsample (repeat_interleave on H and W)
       - Conv2d + BN (the "ConvTranspose" weight used as regular conv after upsample)
       - Slice skip to decoder spatial, concat on channel axis
       - 4 residual blocks
  6. Output CNN: 16→3 channels, 3×3 conv
  7. Slice T' back to original T
  8. Reshape [1, T, 128, 3] → [1, T, 384]

Weight format (PyTorch → MAX NHWC):
  Conv2d  [C_out, C_in, kH, kW] → transpose(2,3,1,0) → [kH, kW, C_in, C_out]
  ConvTranspose2d [C_in, C_out, kH, kW] → also transpose(2,3,1,0) → [kH, kW, C_in, C_out]
    (used as regular conv2d after 2× nearest upsample)
  Bias [C_out] → reshape to [1,1,1,C_out] for broadcasting
  BN scale/offset [C] → reshape to [1,1,1,C] for broadcasting

Residual block:
  y = relu(BN2(conv2(relu(BN1(conv1(x)))))) + shortcut(x)

Note on conv2d_transpose:
  ops.conv2d_transpose is broken in this MAX version (cannot lower to LLVM).
  We substitute with 2× nearest upsample + ops.conv2d, which is shape-equivalent.
"""

from __future__ import annotations
import numpy as np
from max.graph import Graph, TensorType, ops, DeviceRef, Dim
from max.dtype import DType

# Static padding added to T (time) dim before the U-Net to handle non-power-of-2 T.
# After 5 halvings and 5 doublings, T' = 2^5 * floor((T+PAD_T) / 2^5) >= T.
PAD_T = 32


def _pt_to_max_conv(w: np.ndarray) -> np.ndarray:
    """Convert PyTorch Conv2d weight [C_out, C_in, kH, kW] → MAX NHWC [kH, kW, C_in, C_out]."""
    return w.transpose(2, 3, 1, 0)


def _pt_to_max_convtranspose(w: np.ndarray) -> np.ndarray:
    """Convert PyTorch ConvTranspose2d weight [C_in, C_out, kH, kW] → MAX NHWC [kH, kW, C_in, C_out].

    We use this weight for a regular conv2d after 2× nearest upsample
    (shape-equivalent substitute for the broken ops.conv2d_transpose).
    PyTorch ConvTranspose2d [C_in, C_out, kH, kW] → transpose (2,3,0,1) → [kH, kW, C_in, C_out].
    """
    return w.transpose(2, 3, 0, 1)


def _bn_add(x, scale, offset, device_ref):
    """Apply baked BN: x * scale + offset (both shape [C], broadcast to NHWC)."""
    C = scale.shape[0]
    scale_c = ops.constant(scale.reshape(1, 1, 1, C), device=device_ref)
    offset_c = ops.constant(offset.reshape(1, 1, 1, C), device=device_ref)
    return ops.add(ops.mul(x, scale_c), offset_c)


def _conv2d(x, w_np, b_np, stride, padding, device_ref):
    """ops.conv2d wrapper; w_np is already in MAX [kH, kW, C_in, C_out] format.

    Falls back to matmul for 1×1 convolutions (ops.conv2d fails for kernel=1 in this
    MAX version with 'layout_transform_RSCF_to_KNkni' missing kernel error).
    """
    kH, kW = w_np.shape[0], w_np.shape[1]
    if kH == 1 and kW == 1:
        # 1×1 conv via matmul: reshape [1, H, W, C_in] → [H*W, C_in], matmul, reshape back
        C_in = w_np.shape[2]
        C_out = w_np.shape[3]
        w_2d = ops.constant(w_np.reshape(C_in, C_out), device=device_ref)
        dyn_H = x.shape[1]
        dyn_W = x.shape[2]
        x_r = ops.reshape(x, [-1, C_in])
        out = ops.matmul(x_r, w_2d)
        out = ops.reshape(out, [1, dyn_H, dyn_W, C_out])
    else:
        w = ops.constant(w_np, device=device_ref)
        out = ops.conv2d(x, w, stride=stride, padding=padding)
    if b_np is not None:
        b = ops.constant(b_np.reshape(1, 1, 1, -1), device=device_ref)
        out = ops.add(out, b)
    return out


def _residual_block(x, prefix, weights, device_ref):
    """Single residual block (stride always (1,1)).

    Structure:
        h = conv1 → BN1 → relu → conv2 → BN2
        sc = shortcut(x)  if {prefix}.sc.w exists, else x
        output = relu(h + sc)
    """
    # Conv 1
    w1 = _pt_to_max_conv(weights[f"{prefix}.0.w"])
    b1 = weights.get(f"{prefix}.0.b")
    h = _conv2d(x, w1, b1, stride=(1, 1), padding=(1, 1, 1, 1), device_ref=device_ref)
    # BN 1
    if f"{prefix}.0.scale" in weights:
        h = _bn_add(h, weights[f"{prefix}.0.scale"], weights[f"{prefix}.0.offset"], device_ref)
    h = ops.relu(h)

    # Conv 2
    w2 = _pt_to_max_conv(weights[f"{prefix}.1.w"])
    b2 = weights.get(f"{prefix}.1.b")
    h = _conv2d(h, w2, b2, stride=(1, 1), padding=(1, 1, 1, 1), device_ref=device_ref)
    # BN 2
    if f"{prefix}.1.scale" in weights:
        h = _bn_add(h, weights[f"{prefix}.1.scale"], weights[f"{prefix}.1.offset"], device_ref)

    # Shortcut
    sc_w_key = f"{prefix}.sc.w"
    if sc_w_key in weights:
        sc_w = _pt_to_max_conv(weights[sc_w_key])
        sc_b = weights.get(f"{prefix}.sc.b")
        sc = _conv2d(x, sc_w, sc_b, stride=(1, 1), padding=(0, 0, 0, 0), device_ref=device_ref)
    else:
        sc = x

    return ops.relu(ops.add(h, sc))


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

        # Remember original T for final slice
        orig_T = x.shape[1]  # Dim("T")

        # Pad T by PAD_T (static) so U-Net always has enough temporal extent
        # ops.pad takes [pad_N_bef, pad_N_aft, pad_H_bef, pad_H_aft, pad_W_bef, pad_W_aft, pad_C_bef, pad_C_aft]
        x = ops.pad(x, [0, 0, 0, PAD_T, 0, 0, 0, 0])
        # x: [1, T+PAD_T, 128, 1]

        # --- Initial encoder BN ---
        x = _bn_add(x, weights["enc_bn.scale"], weights["enc_bn.offset"], device_ref)

        # --- Encoder: 5 levels ---
        enc_channels = [1, 16, 32, 64, 128, 256]
        skips = []  # pre-pool outputs (saved before AvgPool)
        for L in range(5):
            for B in range(4):
                prefix = f"enc.{L}.{B}"
                x = _residual_block(x, prefix, weights, device_ref)
            # Save skip BEFORE pooling
            skips.append(x)
            # AvgPool2d((2,2)) to halve both T and W
            x = ops.avg_pool2d(x, kernel_size=(2, 2), stride=(2, 2))
        # After 5 pools: [1, (T+PAD_T)//32, 128//32=4, 256]

        # --- Bottleneck: 16 blocks ---
        for I in range(16):
            prefix = f"btl.{I}"
            x = _residual_block(x, prefix, weights, device_ref)
        # After bottleneck: [1, (T+PAD_T)//32, 4, 512]

        # --- Decoder: 5 levels ---
        # dec_channels[L] is C_in to this level, dec_channels[L+1] is C_out after CT
        dec_channels = [512, 256, 128, 64, 32, 16]
        for L in range(5):
            up_ci = dec_channels[L]
            up_co = dec_channels[L + 1]

            # 2× nearest-neighbor upsample on both H (time) and W (freq)
            x = ops.repeat_interleave(x, 2, axis=1)  # double T
            x = ops.repeat_interleave(x, 2, axis=2)  # double W (freq)
            # x: [1, H*2, W*2, up_ci]

            # Conv2d (substitutes for ConvTranspose; weight already in NHWC format)
            up_w = _pt_to_max_convtranspose(weights[f"dec.{L}.up.w"])
            up_b = weights.get(f"dec.{L}.up.b")
            x = _conv2d(x, up_w, up_b, stride=(1, 1), padding=(1, 1, 1, 1), device_ref=device_ref)
            # x: [1, H*2, W*2, up_co]

            # BN after upsample conv
            if f"dec.{L}.up.scale" in weights:
                x = _bn_add(
                    x,
                    weights[f"dec.{L}.up.scale"],
                    weights[f"dec.{L}.up.offset"],
                    device_ref,
                )

            # Skip concat: take skip from encoder level (4-L), slice to decoder spatial
            skip = skips[4 - L]  # [1, skip_H, skip_W, enc_ch]
            dec_H = x.shape[1]  # current decoder spatial (might differ from skip_H)
            dec_W = x.shape[2]
            # Slice skip to match decoder spatial (skip might be larger due to odd T)
            skip = ops.slice_tensor(
                skip,
                [slice(None), slice(None, dec_H), slice(None, dec_W), slice(None)],
            )
            # Concat on channel axis: [1, H, W, up_co] + [1, H, W, enc_ch] → [1, H, W, up_co+enc_ch]
            x = ops.concat([x, skip], axis=3)

            # 4 residual blocks
            for B in range(4):
                prefix = f"dec.{L}.{B}"
                x = _residual_block(x, prefix, weights, device_ref)
        # After decoder: [1, H', W', 16]

        # --- Output CNN: 16→3, 3×3, same padding ---
        out_w = _pt_to_max_conv(weights["out_cnn.w"])
        out_b = weights.get("out_cnn.b")
        x = _conv2d(x, out_w, out_b, stride=(1, 1), padding=(1, 1, 1, 1), device_ref=device_ref)
        # x: [1, H', W', 3]

        # Slice back to original T (H' >= T due to PAD_T)
        x = ops.slice_tensor(
            x,
            [slice(None), slice(0, orig_T), slice(None), slice(None)],
        )
        # x: [1, T, 128, 3]

        # Reshape to [1, T, 384]
        x = ops.reshape(x, [1, Dim("T"), 384])

        g.output(x)

    return g
