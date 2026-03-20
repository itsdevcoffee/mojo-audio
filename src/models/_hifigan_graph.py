"""HiFiGAN MAX Graph building blocks.

Implements ConvTranspose1d for arbitrary strides via zero-interleave + regular
conv2d, generalizing the stride-2 pattern from _rmvpe.py.

All Conv1d operations use Conv2d with W=1 in NHWC format: [B, T, 1, C].
"""

from __future__ import annotations
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
