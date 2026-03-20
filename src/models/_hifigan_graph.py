"""HiFiGAN MAX Graph building blocks.

Implements ConvTranspose1d for arbitrary strides via zero-interleave + regular
conv2d, generalizing the stride-2 pattern from _rmvpe.py.

All Conv1d operations use Conv2d with W=1 in NHWC format: [B, T, 1, C].
"""

from __future__ import annotations
from functools import reduce
import operator
import numpy as np
from max.graph import Graph, TensorType, ops, DeviceRef, Dim
from max.dtype import DType


def _flip_conv_transpose_1d_weights(w_pt: np.ndarray) -> np.ndarray:
    """Convert PyTorch ConvTranspose1d weight for zero-interleave + regular conv2d.

    PyTorch ConvTranspose1d weight: [C_in, C_out, K]
    Steps:
      1. Flip kernel along K: w[:, :, ::-1]
      2. Convert to MAX RSCF format [K, 1, C_in, C_out]:
         transpose(2, 0, 1) gives [K, C_in, C_out], then insert W=1 axis.

    Returns:
        np.ndarray of shape [K, 1, C_in, C_out] in MAX conv2d weight format.
    """
    w_flipped = w_pt[:, :, ::-1].copy()  # [C_in, C_out, K] with K flipped
    # transpose to [K, C_in, C_out]
    w_rearranged = np.transpose(w_flipped, (2, 0, 1))  # [K, C_in, C_out]
    # insert W=1 dim: [K, 1, C_in, C_out]
    return w_rearranged[:, np.newaxis, :, :]


def conv_transpose_1d(x, w_pt: np.ndarray, b_np, *, stride: int, device_ref):
    """Generalized ConvTranspose1d via zero-interleave + regular conv2d.

    Equivalent to PyTorch ConvTranspose1d(C_in, C_out, K, stride=S, padding=(K-S)//2)
    which produces T_out = T_in * S exactly.

    Input format: NHWC [B, T, 1, C_in] where W=1 (all Conv1d uses Conv2d with W=1).
    Output: [B, T*stride, 1, C_out].

    Algorithm:
      1. Zero-interleave: insert (S-1) zeros between each time step along T.
         - Unsqueeze: [B, T, 1, C] -> squeeze batch -> [T, 1, C]
         - Unsqueeze slot: [T, 1, 1, C]
         - Pad with S-1 zeros: [T, S, 1, C]
         - Reshape to merge T*S: [T*S, 1, C]
         - Unsqueeze batch back: [1, T*S, 1, C]
      2. Regular conv2d with flipped kernel and appropriate padding.
         Since the interleave includes S-1 trailing zeros:
           pad_left  = (K + S - 2) // 2
           pad_right = (K - S) // 2
         This ensures conv2d output = T*S + pad_left + pad_right - K + 1 = T*S.

    Args:
        x: NHWC input [B, T, 1, C_in]. B must be 1.
        w_pt: PyTorch ConvTranspose1d weight [C_in, C_out, K].
        b_np: Optional bias [C_out] or None.
        stride: Upsampling stride S.
        device_ref: MAX DeviceRef.

    Returns:
        TensorValue [B, T*stride, 1, C_out].
    """
    S = stride
    C_in_val = w_pt.shape[0]
    K = w_pt.shape[2]

    _T = x.shape[1]  # dynamic symbolic dim

    # --- Step 1: Zero-interleave along T with stride S ---
    # squeeze batch-1 dim
    x_sq = ops.squeeze(x, 0)  # [T, 1, C]

    # Insert a 1-slot after T for zero-interleaving
    x_ins = ops.unsqueeze(x_sq, 1)  # [T, 1, 1, C]

    # Pad that dim with S-1 zeros: [T, 1, 1, C] -> [T, S, 1, C]
    # pad format for 4D: [dim3_before, dim3_after, dim2_before, dim2_after, dim1_before, dim1_after, dim0_before, dim0_after]
    # We want to pad dim1 (the "1" we just inserted) with S-1 after.
    x_pad = ops.pad(x_ins, [0, 0, 0, 0, 0, S - 1, 0, 0])  # [T, S, 1, C]

    # Reshape to merge T and S: [T*S, 1, C]
    x_merged = ops.reshape(x_pad, [_T * S, 1, C_in_val])  # [T*S, 1, C]

    # Unsqueeze batch back
    x_zi = ops.unsqueeze(x_merged, 0)  # [1, T*S, 1, C]

    # --- Step 2: Regular conv2d with flipped weights and padding ---
    # The zero-interleave above produces T*S samples including S-1 trailing
    # zeros after the last real sample.  The mathematically correct padding
    # for the subsequent regular conv (to reproduce PyTorch ConvTranspose1d
    # with padding=(K-S)//2, i.e. T_out = T_in * S exactly) is:
    #
    #   Without trailing zeros the interleaved length would be T*S - S + 1,
    #   needing symmetric padding of (K + S - 2) / 2 on each side.
    #   The S-1 extra trailing zeros shift the right side, so:
    #     pad_left  = (K + S - 2) // 2
    #     pad_right = (K - S) // 2
    #
    # Verify: conv2d output = T*S + pad_left + pad_right - K + 1 = T*S.
    pad_left = (K + S - 2) // 2
    pad_right = (K - S) // 2

    w_max = _flip_conv_transpose_1d_weights(w_pt)  # [K, 1, C_in, C_out]
    w_const = ops.constant(w_max, device=device_ref)

    # conv2d padding: (top, bottom, left, right) for NHWC [B, H, W, C]
    # H=T axis gets the padding, W=1 axis gets no padding
    out = ops.conv2d(x_zi, w_const, stride=(1, 1), padding=(pad_left, pad_right, 0, 0))

    if b_np is not None:
        b = ops.constant(b_np.reshape(1, 1, 1, -1), device=device_ref)
        out = ops.add(out, b)

    return out


def leaky_relu(x, alpha=0.1, device_ref=None):
    """LeakyReLU: where(x > 0, x, alpha*x)."""
    zero = ops.constant(np.array(0.0, dtype=np.float32), device=device_ref)
    alpha_const = ops.constant(
        np.array(alpha, dtype=np.float32), device=device_ref
    )
    mask = ops.greater(x, zero)
    return ops.where(mask, x, ops.mul(x, alpha_const))


def _dilate_kernel(w_np: np.ndarray, dilation: int) -> np.ndarray:
    """Expand a 1D kernel by inserting (dilation-1) zeros between elements.

    Input:  [C_out, C_in, K]
    Output: [C_out, C_in, K_eff] where K_eff = K + (K-1)*(dilation-1)
    """
    if dilation == 1:
        return w_np
    C_out, C_in, K = w_np.shape
    K_eff = K + (K - 1) * (dilation - 1)
    w_dilated = np.zeros((C_out, C_in, K_eff), dtype=w_np.dtype)
    w_dilated[:, :, ::dilation] = w_np
    return w_dilated


def conv1d(x, w_np, b_np, dilation=1, device_ref=None):
    """Conv1d via im2col + matmul, avoiding ops.conv2d (broken for C_in >= 8).

    Input:  [B, T, 1, C_in]  (NHWC with W=1)
    Weight: PyTorch format [C_out, C_in, K]
    Output: [B, T, 1, C_out] (same T due to dilated "same" padding)

    Approach:
      1. Dilate kernel (zero-insertion for dilation > 1)
      2. Pad input along T for "same" output length
      3. im2col: extract K_eff shifted slices, concat → [B, T, 1, K_eff*C_in]
      4. matmul with reshaped weight → [B, T, 1, C_out]
      5. Add bias

    This replaces the previous ops.conv2d implementation which produces
    incorrect results when C_in >= 8 (modular/modular#6248).

    Args:
        x: NHWC input [B, T, 1, C_in].
        w_np: PyTorch Conv1d weight [C_out, C_in, K].
        b_np: Optional bias [C_out] or None.
        dilation: Dilation factor for the convolution.
        device_ref: MAX DeviceRef.

    Returns:
        TensorValue [B, T, 1, C_out].
    """
    C_out, C_in, K = w_np.shape

    # Expand kernel for dilation (no-op when dilation=1)
    w_dilated = _dilate_kernel(w_np, dilation)
    K_eff = w_dilated.shape[2]

    # "same" padding: ensures output T == input T
    pad = (K_eff - 1) // 2

    # Save original symbolic T before padding
    orig_T = x.shape[1]

    # --- Step 1: Pad input along T (dim 1) ---
    # ops.pad format for 4D: [d0_bef, d0_aft, d1_bef, d1_aft, d2_bef, d2_aft, d3_bef, d3_aft]
    # For [B, T, 1, C_in]: d1 = T axis.
    if pad > 0:
        x_pad = ops.pad(x, [0, 0, pad, pad, 0, 0, 0, 0])
    else:
        x_pad = x
    # x_pad: [B, T + 2*pad, 1, C_in]

    # --- Step 2: im2col via shifted slices ---
    # For each kernel position k, slice x_pad[:, k:k+T, :, :].
    # orig_T is symbolic (dynamic), so we use it in slice bounds.
    slices = []
    for k in range(K_eff):
        s = ops.slice_tensor(
            x_pad,
            [slice(None), slice(k, k + orig_T), slice(None), slice(None)],
        )
        slices.append(s)

    # Concat along channel dim: [B, T, 1, K_eff * C_in]
    if K_eff == 1:
        x_cols = slices[0]
    else:
        x_cols = ops.concat(slices, axis=3)

    # --- Step 3: Reshape weight for matmul ---
    # w_dilated: [C_out, C_in, K_eff]
    # Need w_mat[k*C_in + c, c_out] = w_dilated[c_out, c, k]
    # => transpose to [K_eff, C_in, C_out], reshape to [K_eff*C_in, C_out]
    w_mat = np.transpose(w_dilated, (2, 1, 0)).reshape(K_eff * C_in, C_out)
    w_const = ops.constant(w_mat.astype(np.float32), device=device_ref)

    # --- Step 4: matmul ---
    # x_cols: [B, T, 1, K_eff*C_in], squeeze W dim -> [B, T, K_eff*C_in]
    x_cols_sq = ops.squeeze(x_cols, 2)  # [B, T, K_eff*C_in]
    out = ops.matmul(x_cols_sq, w_const)  # [B, T, C_out]

    # --- Step 5: Reshape back to NHWC ---
    out = ops.unsqueeze(out, 2)  # [B, T, 1, C_out]

    if b_np is not None:
        b = ops.constant(b_np.reshape(1, 1, 1, -1), device=device_ref)
        out = ops.add(out, b)

    # Rebind output to match input's symbolic shape — the im2col slicing creates
    # new symbolic dims that MAX can't prove equal to the original Dim("T").
    # Without this, ResBlock residual ops.add(x, residual) fails compilation.
    out = ops.rebind(out, x.shape, message="conv1d: reconcile T dim after im2col")

    return out


def build_resblock(x, weights, dilations, device_ref):
    """HiFiGAN ResBlock type "1": sequential dilated-conv residual blocks.

    For each dilation d in dilations:
        residual = x
        x = leaky_relu(x)
        x = conv1d(x, convs1.{i}.weight, convs1.{i}.bias, dilation=d)
        x = leaky_relu(x)
        x = conv1d(x, convs2.{i}.weight, convs2.{i}.bias, dilation=1)
        x = x + residual

    Args:
        x: NHWC input [B, T, 1, C].
        weights: Dict with keys like "convs1.0.weight", "convs1.0.bias", etc.
        dilations: List of dilation values, e.g. [1, 3, 5].
        device_ref: MAX DeviceRef.

    Returns:
        TensorValue [B, T, 1, C] (same shape as input).
    """
    for i, d in enumerate(dilations):
        residual = x
        x = leaky_relu(x, device_ref=device_ref)
        x = conv1d(
            x,
            weights[f"convs1.{i}.weight"],
            weights.get(f"convs1.{i}.bias"),
            dilation=d,
            device_ref=device_ref,
        )
        x = leaky_relu(x, device_ref=device_ref)
        x = conv1d(
            x,
            weights[f"convs2.{i}.weight"],
            weights.get(f"convs2.{i}.bias"),
            dilation=1,
            device_ref=device_ref,
        )
        x = ops.add(x, residual)

    return x


def _noise_conv(excitation, w_np, b_np, stride, device_ref):
    """Strided Conv1d for downsampling excitation signal to match upsample resolution.

    Converts PyTorch Conv1d weight [C_out, C_in, K] to MAX RSCF [K, 1, C_in, C_out].
    Uses stride on the H (time) axis only.

    Args:
        excitation: NHWC input [B, Ta, 1, 1].
        w_np: PyTorch Conv1d weight [C_out, C_in, K].
        b_np: Optional bias [C_out] or None.
        stride: Downsampling stride along T.
        device_ref: MAX DeviceRef.

    Returns:
        TensorValue [B, Ta//stride, 1, C_out].
    """
    K = w_np.shape[2]
    C_out = w_np.shape[0]
    C_in = w_np.shape[1]

    # MAX Engine doesn't support conv2d with K=1 and C_in=1 (RSCF layout
    # transform bug). Work around by zero-padding the kernel to K=3.
    if K == 1:
        w_padded = np.zeros((C_out, C_in, 3), dtype=w_np.dtype)
        w_padded[:, :, 1] = w_np[:, :, 0]  # center the single tap
        w_np = w_padded
        K = 3

    # Padding to achieve exact downsampling: T_out = T_in / stride.
    # With pad = (K - stride) // 2: output = (T_in + 2*pad - K) / stride + 1
    #   = (T_in + K - stride - K) / stride + 1 = T_in / stride.
    # When K == stride (common case), pad = 0.
    pad = (K - stride) // 2

    # PyTorch [C_out, C_in, K] -> MAX RSCF [K, 1, C_in, C_out]
    w_max = np.transpose(w_np, (2, 1, 0))[:, np.newaxis, :, :]
    w_const = ops.constant(w_max, device=device_ref)

    out = ops.conv2d(
        excitation, w_const,
        stride=(stride, 1),
        dilation=(1, 1),
        padding=(pad, pad, 0, 0),
    )

    if b_np is not None:
        b = ops.constant(b_np.reshape(1, 1, 1, -1), device=device_ref)
        out = ops.add(out, b)

    return out


def build_hifigan_graph(weights, config, device="cpu", batch_size=1):
    """Build the full NSF-HiFiGAN generator as a MAX Graph.

    Architecture:
        latents -> conv_pre(inter_channels -> uic, K=7, pad=3)
        -> for each upsample block i:
            LeakyReLU -> ConvTranspose(ch -> ch//2, stride=rates[i])
            + noise_conv_i(excitation, stride=product(rates[i+1:]))
            -> 3x ResBlock (one per kernel size [3, 7, 11])
        -> LeakyReLU -> conv_post(ch -> 1, K=7, pad=3) -> tanh

    Args:
        weights: Dict of numpy arrays with HiFiGAN weight keys.
        config: Dict with keys: inter_channels, upsample_rates,
                upsample_initial_channel, upsample_kernel_sizes,
                resblock_kernel_sizes, resblock_dilation_sizes.
        device: Device string, default "cpu".
        batch_size: Batch size, default 1.

    Returns:
        A compiled MAX Graph with two inputs (latents, excitation) and one output.
    """
    upsample_rates = config["upsample_rates"]
    uic = config["upsample_initial_channel"]  # 512
    inter_channels = config["inter_channels"]  # 192
    resblock_kernel_sizes = config["resblock_kernel_sizes"]  # [3, 7, 11]
    resblock_dilation_sizes = config["resblock_dilation_sizes"]

    dev = DeviceRef.CPU() if device == "cpu" else DeviceRef(device)

    T = Dim("T")
    Ta = Dim("Ta")

    with Graph(
        "hifigan",
        input_types=[
            TensorType(DType.float32, [batch_size, T, 1, inter_channels], dev),
            TensorType(DType.float32, [batch_size, Ta, 1, 1], dev),
        ],
    ) as g:
        latents, excitation = g.inputs

        # --- conv_pre: [B, T, 1, 192] -> [B, T, 1, 512] ---
        x = conv1d(
            latents,
            weights["conv_pre.weight"],
            weights.get("conv_pre.bias"),
            dilation=1,
            device_ref=dev,
        )

        ch = uic  # current channel count, starts at 512

        for i, (rate, uk) in enumerate(
            zip(upsample_rates, config["upsample_kernel_sizes"])
        ):
            ch_next = ch // 2

            # LeakyReLU
            x = leaky_relu(x, device_ref=dev)

            # ConvTranspose upsample
            x = conv_transpose_1d(
                x,
                weights[f"ups.{i}.weight"],
                weights.get(f"ups.{i}.bias"),
                stride=rate,
                device_ref=dev,
            )

            # noise_conv: downsample excitation to current resolution
            noise_stride = reduce(
                operator.mul, upsample_rates[i + 1 :], 1
            )
            noise = _noise_conv(
                excitation,
                weights[f"noise_convs.{i}.weight"],
                weights.get(f"noise_convs.{i}.bias"),
                stride=noise_stride,
                device_ref=dev,
            )

            # Rebind noise to match x's symbolic time dim (MAX can't prove
            # Ta/noise_stride == T*product(rates[:i+1]) symbolically).
            noise = ops.rebind(
                noise,
                x.shape,
                message=f"noise_conv {i}: excitation time mismatch after downsample",
            )

            # Add noise to upsampled signal
            x = ops.add(x, noise)

            # 3 ResBlocks (one per kernel size) run in PARALLEL, then average.
            # This matches PyTorch RVC: xs = sum(resblock_k(x) for k) / num_kernels
            num_kernels = len(resblock_kernel_sizes)
            rb_outputs = []
            for k_idx, (rk, rd) in enumerate(
                zip(resblock_kernel_sizes, resblock_dilation_sizes)
            ):
                rb_idx = i * num_kernels + k_idx
                rb_weights = {}
                for j in range(len(rd)):
                    rb_weights[f"convs1.{j}.weight"] = weights[
                        f"resblocks.{rb_idx}.convs1.{j}.weight"
                    ]
                    rb_weights[f"convs1.{j}.bias"] = weights.get(
                        f"resblocks.{rb_idx}.convs1.{j}.bias"
                    )
                    rb_weights[f"convs2.{j}.weight"] = weights[
                        f"resblocks.{rb_idx}.convs2.{j}.weight"
                    ]
                    rb_weights[f"convs2.{j}.bias"] = weights.get(
                        f"resblocks.{rb_idx}.convs2.{j}.bias"
                    )
                rb_outputs.append(build_resblock(x, rb_weights, dilations=rd, device_ref=dev))
            # Average the parallel ResBlock outputs
            x = rb_outputs[0]
            for rb_out in rb_outputs[1:]:
                x = ops.add(x, rb_out)
            inv_k = ops.constant(np.array(1.0 / num_kernels, dtype=np.float32), device=dev)
            x = ops.mul(x, inv_k)

            ch = ch_next

        # Final: LeakyReLU -> conv_post -> tanh
        x = leaky_relu(x, device_ref=dev)
        x = conv1d(
            x,
            weights["conv_post.weight"],
            weights.get("conv_post.bias"),
            dilation=1,
            device_ref=dev,
        )

        # tanh: try ops.tanh first, fallback to sigmoid approximation
        try:
            x = ops.tanh(x)
        except (AttributeError, Exception):
            # tanh(x) = 2 * sigmoid(2x) - 1
            two = ops.constant(np.array(2.0, dtype=np.float32), device=dev)
            one = ops.constant(np.array(1.0, dtype=np.float32), device=dev)
            x = ops.sub(ops.mul(two, ops.sigmoid(ops.mul(two, x))), one)

        g.output(x)

    return g
