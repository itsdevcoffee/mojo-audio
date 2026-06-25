"""HiFiGAN MAX Graph building blocks.

All Conv1d and ConvTranspose1d operations delegate to native ops.conv2d
primitives via express-as-plain (src/models/_conv.py): dilation handled by
kernel expansion, transpose by zero-interleave — both reduced to plain conv2d.

All Conv1d operations use Conv2d with W=1 in NHWC format: [B, T, 1, C].
"""

from __future__ import annotations
from functools import reduce
import operator
import numpy as np
from max.graph import Graph, TensorType, ops, DeviceRef, Dim
from max.dtype import DType


def conv_transpose_1d(x, w_pt: np.ndarray, b_np, *, stride: int, device_ref):
    """Native ConvTranspose1d via zero-interleave + plain conv2d (was im2col). Signature unchanged."""
    from ._conv import conv_transpose1d
    return conv_transpose1d(x, w_pt, b_np, stride=stride, device_ref=device_ref)


def leaky_relu(x, alpha=0.1, device_ref=None):
    """LeakyReLU: where(x > 0, x, alpha*x)."""
    zero = ops.constant(np.array(0.0, dtype=np.float32), device=device_ref)
    alpha_const = ops.constant(
        np.array(alpha, dtype=np.float32), device=device_ref
    )
    mask = ops.greater(x, zero)
    return ops.where(mask, x, ops.mul(x, alpha_const))


def conv1d(x, w_np, b_np, dilation=1, device_ref=None):
    """Native Conv1d via express-as-plain (was im2col). Signature unchanged."""
    from ._conv import conv1d as _native_conv1d
    return _native_conv1d(x, w_np, b_np, dilation=dilation, device_ref=device_ref)


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
    """Strided Conv1d for downsampling excitation signal via im2col + matmul.

    Avoids ops.conv2d which triggers MAX compiler errors ("All operation types
    must have the same shape") when multiple conv2d ops with different strides
    coexist in the same graph.

    For C_in=1 and K divisible by stride, exploits reshape to extract
    stride-spaced input samples without slice stepping:
      1. Pad input along T for "same"-style downsampling
      2. For each block b (K/stride blocks), slice a Ta-length region and
         reshape to [B, T_out, stride] — columns give stride-spaced samples
      3. Concatenate blocks → [B, T_out, K]
      4. Matmul with [K, C_out] → [B, T_out, C_out]

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

    # K=1 → pad to K=3 (same workaround as before for single-tap kernels)
    if K == 1:
        w_padded = np.zeros((C_out, 1, 3), dtype=w_np.dtype)
        w_padded[:, :, 1] = w_np[:, :, 0]
        w_np = w_padded
        K = 3

    S = max(stride, 1)
    pad = (K - S) // 2
    orig_Ta = excitation.shape[1]  # symbolic

    # Squeeze to [B, Ta, 1] then [B, Ta]
    x = ops.squeeze(excitation, 3)  # [B, Ta, 1]
    x = ops.squeeze(x, 2)  # [B, Ta]

    # Pad along T: [B, Ta + 2*pad]
    if pad > 0:
        x = ops.pad(x, [0, 0, pad, pad])

    # Build im2col blocks: K must be divisible by S
    n_blocks = K // S
    T_out = orig_Ta // S
    blocks = []
    for b in range(n_blocks):
        # Slice a Ta-length region starting at b*S
        blk = ops.slice_tensor(x, [slice(None), slice(b * S, b * S + orig_Ta)])
        # Rebind to assert Ta is divisible by S, then reshape
        blk = ops.rebind(blk, [excitation.shape[0], T_out * S],
                         message=f"noise_conv: assert Ta divisible by stride {S}")
        blk = ops.reshape(blk, [excitation.shape[0], T_out, S])
        blocks.append(blk)

    # Concatenate blocks: [B, T_out, K]
    if len(blocks) == 1:
        x_cols = blocks[0]
    else:
        x_cols = ops.concat(blocks, axis=2)

    # Weight: [C_out, 1, K] -> [K, C_out]
    w_mat = w_np[: , 0, :].T.astype(np.float32)  # [K, C_out]
    w_const = ops.constant(w_mat, device=device_ref)

    # Matmul: [B, T_out, K] @ [K, C_out] -> [B, T_out, C_out]
    out = ops.matmul(x_cols, w_const)

    # Unsqueeze to NHWC: [B, T_out, 1, C_out]
    out = ops.unsqueeze(out, 2)

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
