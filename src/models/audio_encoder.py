"""AudioEncoder: HuBERT / ContentVec feature extraction via MAX Graph."""

from __future__ import annotations
import numpy as np
from pathlib import Path


def _transformer_block_ops(x, block_weights: dict, device_ref, heads: int = 12, hidden: int = 768):
    """Apply one HuBERT transformer block to a MAX graph tensor.

    Pre-norm architecture: LayerNorm -> Attention -> residual -> LayerNorm -> FFN -> residual.

    This is NOT a sub-graph -- it takes and returns MAX graph TensorValue objects.
    Called 12x inside build_audio_encoder_graph to construct the transformer stack.

    Args:
        x: MAX graph TensorValue, shape [1, T, hidden].
        block_weights: Dict with keys: norm1.*, attn.q/k/v/out.*, norm2.*, ffn.fc1/fc2.*.
                       All weight arrays in PyTorch [out, in] format.
        device_ref: DeviceRef for constant placement.
        heads: Number of attention heads (12).
        hidden: Hidden dimension (768).

    Returns:
        MAX graph TensorValue, shape [1, T, hidden].
    """
    from max.graph import ops, Dim

    head_dim = hidden // heads
    scale = float(head_dim) ** -0.5

    def _const(arr):
        return ops.constant(arr, device=device_ref)

    def _linear(tensor, w_key, b_key=None):
        """Apply linear: x @ W.T + b. W is PyTorch [out, in]."""
        out = ops.matmul(tensor, _const(block_weights[w_key].T))
        if b_key and b_key in block_weights:
            out = ops.add(out, _const(block_weights[b_key]))
        return out

    def _perm4(tensor, target_perm):
        """Apply 4D permutation via sequential 2-axis swaps (MAX constraint)."""
        current = list(range(4))
        for i in range(4):
            desired = target_perm[i]
            src = current.index(desired)
            if src != i:
                tensor = ops.transpose(tensor, i, src)
                current[i], current[src] = current[src], current[i]
        return tensor

    # --- Pre-attention LayerNorm ---
    normed = ops.layer_norm(
        x,
        _const(block_weights["norm1.weight"]),
        _const(block_weights["norm1.bias"]),
        1e-5,
    )

    # --- Multi-head Self-Attention ---
    q = _linear(normed, "attn.q.weight", "attn.q.bias")
    k = _linear(normed, "attn.k.weight", "attn.k.bias")
    v = _linear(normed, "attn.v.weight", "attn.v.bias")

    # Reshape to [1, T, heads, head_dim] then transpose to [1, heads, T, head_dim]
    q = _perm4(ops.reshape(q, [1, -1, heads, head_dim]), [0, 2, 1, 3])
    k = _perm4(ops.reshape(k, [1, -1, heads, head_dim]), [0, 2, 1, 3])
    v = _perm4(ops.reshape(v, [1, -1, heads, head_dim]), [0, 2, 1, 3])

    # Scaled dot-product attention
    scores = ops.mul(
        ops.matmul(q, ops.transpose(k, 2, 3)),  # [1, heads, T, T]
        _const(np.array(scale, dtype=np.float32)),
    )
    attn_weights = ops.softmax(scores, axis=-1)
    context = ops.matmul(attn_weights, v)  # [1, heads, T, head_dim]

    # Merge heads: [1, heads, T, head_dim] -> [1, T, hidden]
    context = _perm4(context, [0, 2, 1, 3])  # [1, T, heads, head_dim]
    context = ops.reshape(context, [1, -1, hidden])

    # Output projection + residual
    attn_out = _linear(context, "attn.out.weight", "attn.out.bias")
    x = ops.add(x, attn_out)

    # --- Pre-FFN LayerNorm ---
    normed2 = ops.layer_norm(
        x,
        _const(block_weights["norm2.weight"]),
        _const(block_weights["norm2.bias"]),
        1e-5,
    )

    # --- Feed-Forward Network ---
    ffn1 = ops.gelu(_linear(normed2, "ffn.fc1.weight", "ffn.fc1.bias"))
    ffn2 = _linear(ffn1, "ffn.fc2.weight", "ffn.fc2.bias")

    # FFN residual
    return ops.add(x, ffn2)


class AudioEncoder:
    """MAX Graph implementation of HuBERT / ContentVec audio encoder.

    Supports facebook/hubert-base-ls960 and lengyue233/content-vec-best.
    Automatically selects GPU if available, falls back to CPU.

    Example:
        model = AudioEncoder.from_pretrained("facebook/hubert-base-ls960")
        features = model.encode(audio_np)  # [1, seq] -> [1, frames, 768]
    """

    def __init__(self, _model, _device):
        self._model = _model
        self._device = _device

    @classmethod
    def _from_weights(cls, weights: dict, device: str = "auto") -> "AudioEncoder":
        """Build MAX Graph from loaded weight dict. Implemented in Task 6."""
        raise NotImplementedError("Implemented in Task 6")

    @classmethod
    def from_pretrained(
        cls,
        model_id: str,
        device: str = "auto",
        cache_dir: str | None = None,
    ) -> "AudioEncoder":
        """Load model from HuggingFace Hub or local path.

        Args:
            model_id: HuggingFace model ID or local path to .safetensors/.pt file.
            device: "auto" (default), "gpu", or "cpu".
            cache_dir: Override default cache (~/.cache/mojo-audio/models/).
        """
        raise NotImplementedError("Implemented in Task 6")

    def encode(self, audio: np.ndarray) -> np.ndarray:
        """Encode raw audio waveform to feature vectors.

        Args:
            audio: Float32 numpy array, shape [1, samples], 16kHz, normalized [-1, 1].

        Returns:
            Float32 numpy array, shape [1, time_frames, 768].
            For 1s audio: [1, 49, 768].
        """
        raise NotImplementedError("Implemented in Task 6")
