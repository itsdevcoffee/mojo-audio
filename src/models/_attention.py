"""HuBERT multi-head self-attention via MAX Graph ops.

Standard scaled dot-product attention. No positional encoding (no RoPE, no sinusoidal).
12 heads x 64 head_dim = 768 hidden.

Input:  [1, T, 768]
Output: [1, T, 768]

Weight format: PyTorch [out_features, in_features] -- transposed for ops.matmul.
"""

from __future__ import annotations
import numpy as np
from max.graph import Graph, TensorType, ops, DeviceRef, Dim
from max.dtype import DType


def _perm4(x, p0, p1, p2, p3):
    """Permute a 4-D tensor to order [p0, p1, p2, p3] using pairwise transposes.

    ops.transpose only swaps two axes at a time, so we decompose arbitrary
    permutations into a sequence of two-axis swaps via selection sort.
    """
    # We track where each original axis currently lives.
    # current[i] = which original axis is now at position i.
    current = [0, 1, 2, 3]
    target = [p0, p1, p2, p3]

    # Build inverse: where is original axis v currently sitting?
    def inv(cur):
        pos = [0, 0, 0, 0]
        for i, v in enumerate(cur):
            pos[v] = i
        return pos

    for i in range(4):
        desired = target[i]
        pos = inv(current)
        src = pos[desired]
        if src != i:
            # swap axes src and i in x
            x = ops.transpose(x, src, i)
            # update current
            current[i], current[src] = current[src], current[i]

    return x


def build_attention_graph(
    weights: dict[str, np.ndarray],
    device_ref,
    heads: int = 12,
    hidden: int = 768,
) -> Graph:
    """Build a MAX Graph for HuBERT multi-head self-attention.

    Args:
        weights: Dict with keys: q.weight [hidden,hidden], k.weight, v.weight,
                 out.weight -- all [hidden, hidden] in PyTorch format [out, in].
                 Optional bias keys: q.bias, k.bias, v.bias, out.bias -- [hidden].
        device_ref: DeviceRef.CPU() or DeviceRef.GPU(0).
        heads: Number of attention heads (12 for HuBERT base).
        hidden: Hidden dimension (768 for HuBERT base).

    Returns:
        MAX Graph. Input: [1, T, hidden] float32. Output: [1, T, hidden] float32.
    """
    head_dim = hidden // heads
    scale = float(head_dim) ** -0.5

    def _proj(x, w_key, b_key):
        """Linear projection: x @ W.T + b. W is [out, in] PyTorch format."""
        w = ops.constant(weights[w_key].T, device=device_ref)  # [in, out]
        out = ops.matmul(x, w)
        if b_key in weights:
            b = ops.constant(weights[b_key], device=device_ref)
            out = ops.add(out, b)
        return out

    with Graph(
        "attention",
        input_types=[TensorType(DType.float32, [1, Dim("T"), hidden], device_ref)],
    ) as g:
        x = g.inputs[0]  # [1, T, hidden]

        # Q, K, V projections: [1, T, hidden]
        q = _proj(x, "q.weight", "q.bias")
        k = _proj(x, "k.weight", "k.bias")
        v = _proj(x, "v.weight", "v.bias")

        # Split into heads: [1, T, hidden] -> [1, T, heads, head_dim]
        q = ops.reshape(q, [1, Dim("T"), heads, head_dim])
        k = ops.reshape(k, [1, Dim("T"), heads, head_dim])
        v = ops.reshape(v, [1, Dim("T"), heads, head_dim])

        # Permute [1, T, heads, head_dim] -> [1, heads, T, head_dim]  (perm: 0,2,1,3)
        q = _perm4(q, 0, 2, 1, 3)
        k = _perm4(k, 0, 2, 1, 3)
        v = _perm4(v, 0, 2, 1, 3)

        # Scaled dot-product attention
        # k_t: [1, heads, head_dim, T]  (swap last two dims: perm 0,1,3,2)
        k_t = ops.transpose(k, 2, 3)
        scores = ops.matmul(q, k_t)            # [1, heads, T, T]
        scale_const = ops.constant(
            np.array(scale, dtype=np.float32), device=device_ref
        )
        scores = ops.mul(scores, scale_const)
        attn_weights = ops.softmax(scores, axis=-1)  # [1, heads, T, T]

        # Weighted sum: [1, heads, T, head_dim]
        context = ops.matmul(attn_weights, v)

        # Merge heads: [1, heads, T, head_dim] -> [1, T, heads, head_dim] -> [1, T, hidden]
        context = _perm4(context, 0, 2, 1, 3)         # [1, T, heads, head_dim]
        context = ops.reshape(context, [1, Dim("T"), hidden])  # [1, T, hidden]

        # Output projection
        out = _proj(context, "out.weight", "out.bias")

        g.output(out)

    return g
