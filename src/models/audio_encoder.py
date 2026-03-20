"""AudioEncoder: HuBERT / ContentVec feature extraction via MAX Graph."""

from __future__ import annotations
import numpy as np
from pathlib import Path


def _transformer_block_ops(x, block_weights: dict, device_ref, heads: int = 12, hidden: int = 768, batch_size: int = 1):
    """Apply one HuBERT transformer block to a MAX graph tensor.

    Pre-norm architecture: LayerNorm -> Attention -> residual -> LayerNorm -> FFN -> residual.

    This is NOT a sub-graph -- it takes and returns MAX graph TensorValue objects.
    Called 12x inside build_audio_encoder_graph to construct the transformer stack.

    Args:
        x: MAX graph TensorValue, shape [B, T, hidden].
        block_weights: Dict with keys: norm1.*, attn.q/k/v/out.*, norm2.*, ffn.fc1/fc2.*.
                       All weight arrays in PyTorch [out, in] format.
        device_ref: DeviceRef for constant placement.
        heads: Number of attention heads (12).
        hidden: Hidden dimension (768).

    Returns:
        MAX graph TensorValue, shape [B, T, hidden].
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

    # --- Multi-head Self-Attention (POST-NORM architecture) ---
    # HuBERT uses: attention -> residual -> layer_norm -> FFN -> residual -> layer_norm
    # NOT pre-norm (layer_norm -> attention -> residual -> ...).

    q = _linear(x, "attn.q.weight", "attn.q.bias")
    k = _linear(x, "attn.k.weight", "attn.k.bias")
    v = _linear(x, "attn.v.weight", "attn.v.bias")

    # Reshape to [B, T, heads, head_dim] then transpose to [B, heads, T, head_dim]
    q = _perm4(ops.reshape(q, [batch_size, -1, heads, head_dim]), [0, 2, 1, 3])
    k = _perm4(ops.reshape(k, [batch_size, -1, heads, head_dim]), [0, 2, 1, 3])
    v = _perm4(ops.reshape(v, [batch_size, -1, heads, head_dim]), [0, 2, 1, 3])

    # Scaled dot-product attention
    scores = ops.mul(
        ops.matmul(q, ops.transpose(k, 2, 3)),  # [B, heads, T, T]
        _const(np.array(scale, dtype=np.float32)),
    )
    attn_weights = ops.softmax(scores, axis=-1)
    context = ops.matmul(attn_weights, v)  # [B, heads, T, head_dim]

    # Merge heads: [B, heads, T, head_dim] -> [B, T, hidden]
    context = _perm4(context, [0, 2, 1, 3])  # [B, T, heads, head_dim]
    context = ops.reshape(context, [batch_size, -1, hidden])

    # Output projection + residual
    attn_out = _linear(context, "attn.out.weight", "attn.out.bias")
    x = ops.add(x, attn_out)

    # --- Post-Attention LayerNorm ---
    x = ops.layer_norm(
        x,
        _const(block_weights["norm1.weight"]),
        _const(block_weights["norm1.bias"]),
        1e-5,
    )

    # --- Feed-Forward Network ---
    ffn1 = ops.gelu(_linear(x, "ffn.fc1.weight", "ffn.fc1.bias"))
    ffn2 = _linear(ffn1, "ffn.fc2.weight", "ffn.fc2.bias")

    # FFN residual + post-FFN LayerNorm
    x = ops.add(x, ffn2)
    return ops.layer_norm(
        x,
        _const(block_weights["norm2.weight"]),
        _const(block_weights["norm2.bias"]),
        1e-5,
    )


class AudioEncoder:
    """MAX Graph implementation of HuBERT / ContentVec audio encoder.

    Supports facebook/hubert-base-ls960 and lengyue233/content-vec-best.
    Automatically selects GPU if available, falls back to CPU.

    Example:
        model = AudioEncoder.from_pretrained("facebook/hubert-base-ls960")
        features = model.encode(audio_np)  # [1, seq] -> [1, frames, 768]
    """

    def __init__(self, _model, _device, _model2=None, _pos_conv_weights=None, _enc_norm_weights=None, _batch_size=1):
        self._model = _model   # CNN + feature projection graph
        self._device = _device
        self._model2 = _model2  # Encoder norm + transformer blocks graph
        # Pos conv weights stored for numpy-based computation (avoids MAX dynamic dim bug)
        self._pos_conv_weights = _pos_conv_weights  # dict: weight, bias
        self._enc_norm_weights = _enc_norm_weights  # dict: weight, bias (if separate from model2)
        self._batch_size = _batch_size

    @classmethod
    def _from_weights(cls, weights: dict, device: str = "auto", batch_size: int = 1) -> "AudioEncoder":
        """Build MAX Graph from loaded weight dict.

        Args:
            weights: Internal weight dict (from _weight_loader or _make_full_weights).
            device: "auto", "gpu", or "cpu".

        Returns:
            AudioEncoder instance backed by a compiled MAX Graph model.
        """
        from max import engine
        from max.driver import Accelerator, CPU, accelerator_count
        from max.graph import Graph, TensorType, ops, DeviceRef, Dim
        from max.dtype import DType
        from ._feature_extractor import _pt_weight_to_max

        # Device selection
        use_gpu = accelerator_count() > 0 if device == "auto" else device == "gpu"
        dev = Accelerator() if use_gpu else CPU()
        device_ref = DeviceRef.GPU(0) if use_gpu else DeviceRef.CPU()

        cnn_configs = [
            (1, 512, 10, 5),
            (512, 512, 3, 2),
            (512, 512, 3, 2),
            (512, 512, 3, 2),
            (512, 512, 3, 2),
            (512, 512, 2, 2),
            (512, 512, 2, 2),
        ]

        def _const(arr):
            return ops.constant(np.asarray(arr, dtype=np.float32), device=device_ref)

        # ---------------------------------------------------------------
        # Graph 1: CNN Feature Extractor + Feature Projection
        # Input: [1, L, 1, 1] NHWC audio → Output: [1, T, 768] features
        # ---------------------------------------------------------------
        with Graph(
            "cnn_proj",
            input_types=[TensorType(DType.float32, [batch_size, Dim("L"), 1, 1], device_ref)],
        ) as g1:
            x = g1.inputs[0]  # [1, L, 1, 1]

            # Stage 1: CNN Feature Extractor (7 layers)
            # HuBERT layer 0 uses GroupNorm(num_groups=512, num_channels=512) which
            # normalizes each of the 512 channels independently over the T' time dimension
            # (equivalent to instance norm per channel).
            # Layers 1-6 have no normalization.
            EPS_GN = _const(np.array(1e-5, dtype=np.float32))
            for i, (c_in, c_out, kernel, stride) in enumerate(cnn_configs):
                w_max = _pt_weight_to_max(weights[f"cnn.{i}.weight"])
                conv_out = ops.conv2d(x, _const(w_max), stride=(stride, 1))
                # conv_out: [B, T', 1, c_out] -> reshape to [B, T', c_out]
                conv_out = ops.reshape(conv_out, [batch_size, -1, c_out])
                norm_w_key = f"cnn.{i}.norm.weight"
                if norm_w_key in weights:
                    # GroupNorm(num_groups=C, num_channels=C): normalize each channel c
                    # independently over the T' temporal dimension, per sample.
                    # [B, T', C] → transpose(1,2) → [B, C, T'] → reshape [B*C, T']
                    gn_in = ops.reshape(
                        ops.transpose(conv_out, 1, 2),  # [B, C, T']
                        [batch_size * c_out, -1],        # [B*C, T']
                    )
                    mean_ = ops.mean(gn_in, axis=1)                           # [B*C, 1]
                    diff_ = ops.sub(gn_in, mean_)                             # [B*C, T']
                    var_ = ops.mean(ops.mul(diff_, diff_), axis=1)            # [B*C, 1]
                    std_ = ops.sqrt(ops.add(var_, EPS_GN))                    # [B*C, 1]
                    normed_ = ops.div(diff_, std_)                            # [B*C, T']
                    # gamma/beta: [C] → tile to [B*C, 1] by repeating B times
                    gamma_1d = _const(weights[norm_w_key])                    # [C]
                    beta_1d = _const(weights[f"cnn.{i}.norm.bias"])           # [C]
                    gamma_tiled = ops.reshape(
                        ops.tile(ops.reshape(gamma_1d, [1, c_out]), [batch_size, 1]),
                        [batch_size * c_out, 1],                              # [B*C, 1]
                    )
                    beta_tiled = ops.reshape(
                        ops.tile(ops.reshape(beta_1d, [1, c_out]), [batch_size, 1]),
                        [batch_size * c_out, 1],
                    )
                    gn_out = ops.add(ops.mul(normed_, gamma_tiled), beta_tiled)  # [B*C, T']
                    # Reshape back: [B*C, T'] → [B, C, T'] → transpose(1,2) → [B, T', C]
                    conv_out = ops.transpose(
                        ops.reshape(gn_out, [batch_size, c_out, -1]),        # [B, C, T']
                        1, 2,                                                 # [B, T', C]
                    )
                x = ops.reshape(ops.gelu(conv_out), [batch_size, -1, 1, c_out])

            x = ops.reshape(x, [batch_size, -1, 512])  # [B, T, 512]

            # Stage 2: Feature Projection LayerNorm(512) -> Linear(512->768)
            # HuBERT applies layer norm to the 512-dim CNN output BEFORE projecting to 768.
            x = ops.layer_norm(
                x,
                _const(weights["proj.norm.weight"]),
                _const(weights["proj.norm.bias"]),
                1e-5,
            )  # [1, T, 512]
            x = ops.add(
                ops.matmul(x, _const(weights["proj.weight"].T)),
                _const(weights["proj.bias"]),
            )  # [1, T, 768]

            g1.output(x)

        # ---------------------------------------------------------------
        # Graph 2: Encoder Norm + Transformer Blocks
        # Input: [1, T, 768] (after pos_conv+gelu+residual, done in numpy) → Output: [1, T, 768]
        #
        # MAX engine bug: conv2d with K=128 on dynamic Dim("T") input with C_in >= 48
        # produces incorrect results. The pos_conv is therefore implemented in numpy
        # (outside the MAX graph) to guarantee correctness.
        # ---------------------------------------------------------------
        with Graph(
            "encoder_transformer",
            input_types=[TensorType(DType.float32, [batch_size, Dim("T"), 768], device_ref)],
        ) as g2:
            x = g2.inputs[0]  # [1, T, 768] after pos_conv + enc_norm

            # Stage 4: 12x Transformer Encoder Blocks (post-norm architecture)
            for i in range(12):
                block_w = {
                    k[len(f"blocks.{i}."):]: weights[k]
                    for k in weights
                    if k.startswith(f"blocks.{i}.")
                }
                x = _transformer_block_ops(x, block_w, device_ref, batch_size=batch_size)

            g2.output(x)

        session = engine.InferenceSession(devices=[dev])
        model1 = session.load(g1)
        model2 = session.load(g2)

        # Store pos_conv and enc_norm weights for numpy-based computation
        pos_conv_w = {
            "weight": weights["pos_conv.weight"].copy() if "pos_conv.weight" in weights else None,
            "bias": weights["pos_conv.bias"].copy() if "pos_conv.bias" in weights else None,
        }
        enc_norm_w = {
            "weight": weights["enc_norm.weight"].copy() if "enc_norm.weight" in weights else None,
            "bias": weights["enc_norm.bias"].copy() if "enc_norm.bias" in weights else None,
        }
        return cls(_model=model1, _device=dev, _model2=model2,
                   _pos_conv_weights=pos_conv_w, _enc_norm_weights=enc_norm_w,
                   _batch_size=batch_size)

    @classmethod
    def from_pretrained(
        cls,
        model_id: str,
        device: str = "auto",
        cache_dir: str | None = None,
        batch_size: int = 1,
    ) -> "AudioEncoder":
        """Load model from HuggingFace Hub or local path.

        Args:
            model_id: HuggingFace model ID or local path to .safetensors/.pt file.
            device: "auto" (default), "gpu", or "cpu".
            cache_dir: Override default cache (~/.cache/mojo-audio/models/).
            batch_size: Batch size for inference (default 1).
        """
        from ._weight_loader import load_weights
        weights = load_weights(model_id, cache_dir)
        return cls._from_weights(weights, device=device, batch_size=batch_size)

    def encode(self, audio: np.ndarray) -> np.ndarray:
        """Encode raw audio waveform to feature vectors.

        Args:
            audio: Float32 numpy array, shape [B, samples], 16kHz, normalized [-1, 1].
                   B must match the batch_size used at construction time.

        Returns:
            Float32 numpy array, shape [B, time_frames, 768].
            For 1s audio at batch=1: [1, 49, 768].
        """
        from max.driver import Accelerator, Buffer

        B = audio.shape[0]
        if B != self._batch_size:
            raise ValueError(
                f"Input batch size {B} does not match model batch_size {self._batch_size}. "
                f"Rebuild model with AudioEncoder._from_weights(..., batch_size={B})"
            )

        # Reshape to [B, L, 1, 1] NHWC format required by CNN
        audio_in = audio.reshape(B, -1, 1, 1).astype(np.float32)

        # Transfer to GPU if needed
        if isinstance(self._device, Accelerator):
            inp = Buffer.from_numpy(audio_in).to(self._device)
        else:
            inp = audio_in

        # Graph 1: CNN + Feature Projection → [1, T, 768]
        result1 = self._model.execute(inp)
        features = list(result1.values())[0] if isinstance(result1, dict) else result1[0]
        features_np = features.to_numpy()  # [1, T, 768]

        if self._model2 is not None:
            # Stage 3: Pos Conv (numpy, avoids MAX dynamic-dim conv2d bug with K=128, C>=48)
            # HuBERT: Conv1d(768, 768, K=128, groups=16, padding=64) + HubertSamePadLayer(-1)
            if self._pos_conv_weights is not None and self._pos_conv_weights["weight"] is not None:
                pw = self._pos_conv_weights["weight"]   # [768, 48, 128]
                pb = self._pos_conv_weights["bias"]     # [768] or None
                # features_np: [B, T, 768]
                C_in = 768
                groups = 16
                C_per_g = C_in // groups  # 48
                K = pw.shape[2]           # 128
                pad = K // 2              # 64
                from scipy.special import erf as scipy_erf

                pos_batch = np.zeros_like(features_np)  # [B, T, 768]
                for b in range(B):
                    x_bt = features_np[b]  # [T, 768]
                    T = x_bt.shape[0]
                    out_t = np.zeros((T, C_in), dtype=np.float32)
                    for grp in range(groups):
                        gs = grp * C_per_g
                        ge = gs + C_per_g
                        x_g = x_bt[:, gs:ge]  # [T, C_per_g]
                        x_g_pad = np.pad(x_g, ((pad, pad), (0, 0)))  # [T+128, C_per_g]
                        rows = np.stack([x_g_pad[t:t+K, :] for t in range(T + 1)], axis=0)
                        rows_2d = rows.reshape(T + 1, K * C_per_g)
                        w_g = pw[gs:ge, :, :]
                        w_2d = w_g.transpose(2, 1, 0).reshape(K * C_per_g, C_per_g)
                        out_g = (rows_2d @ w_2d)[:T, :]
                        out_t[:, gs:ge] = out_g
                    if pb is not None:
                        out_t += pb[np.newaxis, :]
                    # GELU activation
                    pos_batch[b] = (0.5 * out_t * (1.0 + scipy_erf(out_t / np.sqrt(2.0)))).astype(np.float32)
                features_np = features_np + pos_batch  # residual add [B, T, 768]

            # Stage 3b: Encoder LayerNorm (numpy)
            if self._enc_norm_weights is not None and self._enc_norm_weights["weight"] is not None:
                gamma = self._enc_norm_weights["weight"]  # [768]
                beta = self._enc_norm_weights["bias"]     # [768]
                mean = features_np.mean(axis=-1, keepdims=True)
                var = ((features_np - mean) ** 2).mean(axis=-1, keepdims=True)
                features_np = ((features_np - mean) / np.sqrt(var + 1e-5)).astype(np.float32)
                features_np = (features_np * gamma + beta).astype(np.float32)

            # Graph 2: Transformer Blocks → [1, T, 768]
            if isinstance(self._device, Accelerator):
                feat_in = Buffer.from_numpy(np.ascontiguousarray(features_np)).to(self._device)
            else:
                feat_in = np.ascontiguousarray(features_np)
            result2 = self._model2.execute(feat_in)
            out = list(result2.values())[0] if isinstance(result2, dict) else result2[0]
            return out.to_numpy()
        else:
            return features_np
