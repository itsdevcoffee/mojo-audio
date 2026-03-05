"""CNN feature extractor for HuBERT / ContentVec via MAX Graph.

7 Conv1D layers (implemented as Conv2D with H=1, NHWC layout).
Each layer: Conv1D -> GroupNorm (LayerNorm) -> GELU.

HuBERT uses group norm with groups=C_out (equiv to layer norm per channel position).
We approximate this with ops.layer_norm applied to the channel dimension.

Input graph tensor:  [B=1, L, 1, C_in=1] — audio in NHWC format
Output graph tensor: [B=1, T, 512]        — 512-dim features at 50Hz

Weight key convention (internal, from _weight_loader):
  cnn.{i}.weight       — [C_out, C_in, K] PyTorch format
  cnn.{i}.norm.weight  — [C_out]
  cnn.{i}.norm.bias    — [C_out]
"""

from __future__ import annotations
import numpy as np
from max.graph import Graph, TensorType, ops, DeviceRef, Dim
from max.dtype import DType

# HuBERT CNN architecture: (in_ch, out_ch, kernel, stride)
_CNN_CONFIG = [
    (1, 512, 10, 5),
    (512, 512, 3, 2),
    (512, 512, 3, 2),
    (512, 512, 3, 2),
    (512, 512, 3, 2),
    (512, 512, 2, 2),
    (512, 512, 2, 2),
]


def _pt_weight_to_max(w: np.ndarray) -> np.ndarray:
    """Convert PyTorch Conv1D weight [C_out, C_in, K] to MAX RSCF [K, 1, C_in, C_out]."""
    # w shape: [C_out, C_in, K]
    # MAX conv2d RSCF: [R=K, S=1, C=C_in, F=C_out]
    return np.transpose(w, (2, 1, 0))[:, np.newaxis, :, :]


def build_feature_extractor_graph(
    weights: dict[str, np.ndarray],
    device_ref,
) -> Graph:
    """Build a MAX Graph for the 7-layer CNN feature extractor.

    Args:
        weights: Internal weight dict (from _weight_loader). Expects:
                 cnn.{i}.weight [C_out, C_in, K], cnn.{i}.norm.weight/bias [C_out]
        device_ref: DeviceRef.CPU() or DeviceRef.GPU(0).

    Returns:
        MAX Graph. Input: [1, L, 1, 1] float32. Output: [1, T, 512] float32.
    """
    with Graph(
        "feature_extractor",
        input_types=[TensorType(DType.float32, [1, Dim("L"), 1, 1], device_ref)],
    ) as g:
        x = g.inputs[0]  # [1, L, 1, 1]

        for i, (c_in, c_out, kernel, stride) in enumerate(_CNN_CONFIG):
            # Convert weight from PyTorch [C_out, C_in, K] to MAX RSCF [K, 1, C_in, C_out]
            w_max = _pt_weight_to_max(weights[f"cnn.{i}.weight"])
            w_const = ops.constant(w_max, device=device_ref)

            # Conv1D via Conv2D (stride over L-dim, W-dim always 1)
            conv_out = ops.conv2d(x, w_const, stride=(stride, 1))  # [1, L', 1, C_out]

            # Reshape for LayerNorm: [1, L', 1, C_out] -> [1, L', C_out]
            conv_out = ops.reshape(conv_out, [1, -1, c_out])

            # GroupNorm (groups=C_out) == LayerNorm along channel dim
            g_const = ops.constant(weights[f"cnn.{i}.norm.weight"], device=device_ref)
            b_const = ops.constant(weights[f"cnn.{i}.norm.bias"], device=device_ref)
            normed = ops.layer_norm(conv_out, g_const, b_const, 1e-5)

            # GELU activation
            activated = ops.gelu(normed)

            # Reshape back to NHWC for next conv: [1, L', C_out] -> [1, L', 1, C_out]
            x = ops.reshape(activated, [1, -1, 1, c_out])

        # Final output: [1, T, 1, 512] -> [1, T, 512]
        x = ops.reshape(x, [1, -1, 512])
        g.output(x)

    return g
