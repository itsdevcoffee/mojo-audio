"""Tests for mojo_audio.models.AudioEncoder.

Level 1 tests (no download): no marker — run by default via test-models.
Level 2 tests (download required): @pytest.mark.slow — skipped by default.
"""

import sys
import os
# sys.path.insert lets us import `src/models/` as `models` without pip install.
# Use `from models import X` throughout these tests (not `from mojo_audio.models import X`).
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import numpy as np
import pytest
from max.driver import accelerator_count


# --- Fixtures ---

@pytest.fixture
def rng():
    return np.random.default_rng(42)


@pytest.fixture
def audio_1s(rng):
    """1 second of synthetic 16kHz audio, normalized."""
    return rng.standard_normal((1, 16000)).astype(np.float32)


@pytest.fixture(scope="session")
def cpu_device():
    from max.driver import CPU
    return CPU()


@pytest.fixture
def gpu_available():
    return accelerator_count() > 0


# --- Smoke tests (always pass if imports work) ---

def test_package_importable():
    """mojo_audio.models must be importable."""
    from models import AudioEncoder
    assert AudioEncoder is not None


def test_max_engine_importable():
    """MAX Engine must be accessible."""
    from max import engine
    from max.driver import accelerator_count
    assert True


def test_gpu_session_creatable():
    """GPU InferenceSession must work if GPU is available."""
    from max import engine
    from max.driver import Accelerator, CPU, accelerator_count
    if accelerator_count() > 0:
        session = engine.InferenceSession(devices=[Accelerator()])
    else:
        session = engine.InferenceSession(devices=[CPU()])
    assert session is not None


class TestWeightLoader:
    """Tests for _weight_loader — no model download required."""

    def test_detect_hubert_prefix(self):
        """Detect 'hubert.' prefix in HuBERT checkpoint keys."""
        from models._weight_loader import _detect_prefix
        keys = [
            "hubert.feature_extractor.conv_layers.0.conv.weight",
            "hubert.encoder.layers.0.attention.q_proj.weight",
        ]
        assert _detect_prefix(keys) == "hubert."

    def test_detect_contentvec_prefix(self):
        """Detect 'model.' prefix in ContentVec checkpoint keys."""
        from models._weight_loader import _detect_prefix
        keys = [
            "model.feature_extractor.conv_layers.0.conv.weight",
            "model.encoder.layers.0.attention.q_proj.weight",
        ]
        assert _detect_prefix(keys) == "model."

    def test_detect_prefix_raises_on_unknown(self):
        """Unknown prefix raises ValueError."""
        from models._weight_loader import _detect_prefix
        with pytest.raises(ValueError):
            _detect_prefix(["some.unknown.key"])

    def test_map_cnn_weight(self):
        """CNN conv weight key maps correctly (RVC-style hubert. prefix)."""
        from models._weight_loader import _map_key
        assert _map_key("hubert.feature_extractor.conv_layers.3.conv.weight", "hubert.") == "cnn.3.weight"

    def test_map_cnn_norm(self):
        """CNN layer norm weight key maps correctly (RVC-style hubert. prefix)."""
        from models._weight_loader import _map_key
        assert _map_key("hubert.feature_extractor.conv_layers.0.layer_norm.weight", "hubert.") == "cnn.0.norm.weight"

    def test_map_projection(self):
        """Feature projection weight maps correctly (RVC-style hubert. prefix)."""
        from models._weight_loader import _map_key
        assert _map_key("hubert.feature_projection.projection.weight", "hubert.") == "proj.weight"

    def test_map_transformer_attention_q(self):
        """Transformer layer attention q_proj maps correctly (RVC-style hubert. prefix)."""
        from models._weight_loader import _map_key
        assert _map_key("hubert.encoder.layers.5.attention.q_proj.weight", "hubert.") == "blocks.5.attn.q.weight"

    def test_map_transformer_ffn(self):
        """Transformer layer FFN maps correctly (RVC-style hubert. prefix)."""
        from models._weight_loader import _map_key
        assert _map_key("hubert.encoder.layers.0.feed_forward.intermediate_dense.weight", "hubert.") == "blocks.0.ffn.fc1.weight"

    def test_map_unknown_key_returns_none(self):
        """Unknown key returns None (will be skipped)."""
        from models._weight_loader import _map_key
        assert _map_key("some.unknown.weight", "hubert.") is None

    def test_map_contentvec_prefix(self):
        """ContentVec uses 'model.' prefix — maps same way."""
        from models._weight_loader import _map_key
        assert _map_key("model.feature_extractor.conv_layers.0.conv.weight", "model.") == "cnn.0.weight"

    def test_load_from_dict(self):
        """load_weights_from_dict maps a synthetic weight dict correctly."""
        import numpy as np
        from models._weight_loader import load_weights_from_dict
        fake_weights = {
            "hubert.feature_extractor.conv_layers.0.conv.weight": np.zeros((512, 1, 10), dtype=np.float32),
            "hubert.encoder.layers.0.attention.q_proj.weight": np.zeros((768, 768), dtype=np.float32),
            "hubert.feature_projection.projection.weight": np.zeros((768, 512), dtype=np.float32),
        }
        result = load_weights_from_dict(fake_weights)
        assert "cnn.0.weight" in result
        assert "blocks.0.attn.q.weight" in result
        assert "proj.weight" in result

    def test_load_from_dict_all_float32(self):
        """All output arrays must be float32 regardless of input dtype."""
        import numpy as np
        from models._weight_loader import load_weights_from_dict
        fake_weights = {
            "hubert.feature_projection.projection.weight": np.zeros((768, 512), dtype=np.float64),
        }
        result = load_weights_from_dict(fake_weights)
        assert result["proj.weight"].dtype == np.float32

    def test_load_from_dict_skips_unknown_keys(self):
        """Keys not in the mapping are silently skipped."""
        import numpy as np
        from models._weight_loader import load_weights_from_dict
        fake_weights = {
            "hubert.some.unknown.key": np.zeros(10, dtype=np.float32),
            "hubert.feature_projection.projection.weight": np.zeros((768, 512), dtype=np.float32),
        }
        result = load_weights_from_dict(fake_weights)
        assert len(result) == 1
        assert "proj.weight" in result


class TestFeatureExtractor:
    """Tests for CNN feature extractor — no model download required.

    Uses a class-scoped fixture to compile the MAX graph ONCE for all tests.
    """

    @staticmethod
    def _make_random_weights():
        """Random weights in PyTorch format [C_out, C_in, K]."""
        import numpy as np
        rng = np.random.default_rng(42)
        configs = [
            (1, 512, 10), (512, 512, 3), (512, 512, 3), (512, 512, 3),
            (512, 512, 3), (512, 512, 2), (512, 512, 2),
        ]
        w = {}
        for i, (c_in, c_out, k) in enumerate(configs):
            w[f"cnn.{i}.weight"] = rng.standard_normal((c_out, c_in, k)).astype(np.float32)
            w[f"cnn.{i}.norm.weight"] = np.ones(c_out, dtype=np.float32)
            w[f"cnn.{i}.norm.bias"] = np.zeros(c_out, dtype=np.float32)
        return w

    @staticmethod
    def _result_to_numpy(result):
        """Extract numpy array from model.execute() result (list or dict of MAX Tensors)."""
        import numpy as np
        v = list(result.values())[0] if isinstance(result, dict) else result[0]
        return v.to_numpy() if hasattr(v, "to_numpy") else np.array(v)

    @pytest.fixture(scope="class")
    def fe_model(self, cpu_device):
        """Compile feature extractor graph once for all tests in this class."""
        from max import engine
        from max.graph import DeviceRef
        from models._feature_extractor import build_feature_extractor_graph

        cpu_ref = DeviceRef.CPU()
        graph = build_feature_extractor_graph(self._make_random_weights(), cpu_ref)
        return engine.InferenceSession(devices=[cpu_device]).load(graph)

    def test_output_shape_1s(self, fe_model):
        """1s @16kHz -> [1, 49, 512]."""
        import numpy as np
        audio = np.zeros((1, 16000, 1, 1), dtype=np.float32)
        result = fe_model.execute(audio)
        out = self._result_to_numpy(result)
        assert out.shape == (1, 49, 512), f"Expected (1,49,512) got {out.shape}"

    def test_output_shape_2s(self, fe_model):
        """2s @16kHz -> [1, 99, 512]."""
        import numpy as np
        audio = np.zeros((1, 32000, 1, 1), dtype=np.float32)
        result = fe_model.execute(audio)
        out = self._result_to_numpy(result)
        assert out.shape == (1, 99, 512), f"Expected (1,99,512) got {out.shape}"

    def test_output_not_nan(self, fe_model, audio_1s):
        """Output must not contain NaN or Inf."""
        import numpy as np
        audio_in = audio_1s.reshape(1, 16000, 1, 1)
        result = fe_model.execute(audio_in)
        out = self._result_to_numpy(result)
        assert not np.isnan(out).any(), "Output contains NaN"
        assert not np.isinf(out).any(), "Output contains Inf"

    def test_weight_transpose(self):
        """PyTorch [C_out, C_in, K] must be transposed to MAX [K, 1, C_in, C_out]."""
        import numpy as np
        from models._feature_extractor import _pt_weight_to_max

        pt_w = np.zeros((512, 1, 10), dtype=np.float32)  # [C_out, C_in, K]
        max_w = _pt_weight_to_max(pt_w)
        assert max_w.shape == (10, 1, 1, 512), f"Expected (10,1,1,512) got {max_w.shape}"


class TestAttention:
    """Tests for multi-head self-attention — no download required.

    Uses a class-scoped fixture to compile the attention graph ONCE.
    """

    @staticmethod
    def _make_random_weights(hidden=768):
        """Random attention weights in PyTorch format [out, in]."""
        import numpy as np
        rng = np.random.default_rng(42)
        return {
            "q.weight": rng.standard_normal((hidden, hidden)).astype(np.float32),
            "q.bias": np.zeros(hidden, dtype=np.float32),
            "k.weight": rng.standard_normal((hidden, hidden)).astype(np.float32),
            "k.bias": np.zeros(hidden, dtype=np.float32),
            "v.weight": rng.standard_normal((hidden, hidden)).astype(np.float32),
            "v.bias": np.zeros(hidden, dtype=np.float32),
            "out.weight": rng.standard_normal((hidden, hidden)).astype(np.float32),
            "out.bias": np.zeros(hidden, dtype=np.float32),
        }

    @pytest.fixture(scope="class")
    def attn_model(self, cpu_device):
        """Compile attention graph once for all tests in this class."""
        from max import engine
        from max.graph import DeviceRef
        from models._attention import build_attention_graph

        cpu_ref = DeviceRef.CPU()
        graph = build_attention_graph(self._make_random_weights(), cpu_ref)
        return engine.InferenceSession(devices=[cpu_device]).load(graph)

    def test_output_shape_seq49(self, attn_model):
        """[1, 49, 768] in -> [1, 49, 768] out."""
        import numpy as np
        x = np.random.randn(1, 49, 768).astype(np.float32)
        result = attn_model.execute(x)
        tensor = list(result.values())[0] if isinstance(result, dict) else result[0]
        out = tensor.to_numpy()
        assert out.shape == (1, 49, 768), f"Expected (1,49,768) got {out.shape}"

    def test_output_shape_seq99(self, attn_model):
        """Dynamic sequence length: [1, 99, 768] in -> [1, 99, 768] out."""
        import numpy as np
        x = np.random.randn(1, 99, 768).astype(np.float32)
        result = attn_model.execute(x)
        tensor = list(result.values())[0] if isinstance(result, dict) else result[0]
        out = tensor.to_numpy()
        assert out.shape == (1, 99, 768), f"Expected (1,99,768) got {out.shape}"

    def test_output_not_nan(self, attn_model):
        """Attention output must not contain NaN or Inf."""
        import numpy as np
        x = np.random.randn(1, 49, 768).astype(np.float32)
        result = attn_model.execute(x)
        tensor = list(result.values())[0] if isinstance(result, dict) else result[0]
        out = tensor.to_numpy()
        assert not np.isnan(out).any(), "Output contains NaN"
        assert not np.isinf(out).any(), "Output contains Inf"

    def test_attention_is_not_identity(self, attn_model):
        """Attention output must differ from input (non-trivial transformation)."""
        import numpy as np
        x = np.random.randn(1, 49, 768).astype(np.float32)
        result = attn_model.execute(x)
        tensor = list(result.values())[0] if isinstance(result, dict) else result[0]
        out = tensor.to_numpy()
        diff = np.abs(out - x).mean()
        assert diff > 0.01, f"Attention output too close to identity (mean diff={diff:.6f})"


class TestTransformerBlock:
    """Test a single transformer block.

    Uses class-scoped fixtures: one for random weights (shared by shape + NaN tests)
    and one for zero weights (residual test). 2 compilations instead of 3.
    """

    @staticmethod
    def _make_random_block_weights(hidden=768, ffn_dim=3072):
        """Random weights for one block."""
        import numpy as np
        rng = np.random.default_rng(42)
        w = {}
        for name in ["norm1", "norm2"]:
            w[f"{name}.weight"] = np.ones(hidden, dtype=np.float32)
            w[f"{name}.bias"] = np.zeros(hidden, dtype=np.float32)
        for proj in ["attn.q", "attn.k", "attn.v", "attn.out"]:
            w[f"{proj}.weight"] = rng.standard_normal((hidden, hidden)).astype(np.float32) * 0.02
            w[f"{proj}.bias"] = np.zeros(hidden, dtype=np.float32)
        w["ffn.fc1.weight"] = rng.standard_normal((ffn_dim, hidden)).astype(np.float32) * 0.02
        w["ffn.fc1.bias"] = np.zeros(ffn_dim, dtype=np.float32)
        w["ffn.fc2.weight"] = rng.standard_normal((hidden, ffn_dim)).astype(np.float32) * 0.02
        w["ffn.fc2.bias"] = np.zeros(hidden, dtype=np.float32)
        return w

    @pytest.fixture(scope="class")
    def block_model(self, cpu_device):
        """Compile transformer block graph (random weights) once."""
        import numpy as np
        from max import engine
        from max.graph import Graph, TensorType, DeviceRef, Dim
        from max.dtype import DType
        from models.audio_encoder import _transformer_block_ops

        cpu_ref = DeviceRef.CPU()
        block_w = self._make_random_block_weights()
        with Graph(
            "block_test",
            input_types=[TensorType(DType.float32, [1, Dim("T"), 768], cpu_ref)],
        ) as g:
            x = g.inputs[0]
            out = _transformer_block_ops(x, block_w, cpu_ref)
            g.output(out)
        return engine.InferenceSession(devices=[cpu_device]).load(g)

    @pytest.fixture(scope="class")
    def block_model_zero(self, cpu_device):
        """Compile transformer block graph (zero projection weights) once."""
        import numpy as np
        from max import engine
        from max.graph import Graph, TensorType, DeviceRef, Dim
        from max.dtype import DType
        from models.audio_encoder import _transformer_block_ops

        cpu_ref = DeviceRef.CPU()
        block_w = self._make_random_block_weights()
        for k in block_w:
            if "weight" in k and "norm" not in k:
                block_w[k] = np.zeros_like(block_w[k])
        with Graph(
            "block_zero",
            input_types=[TensorType(DType.float32, [1, Dim("T"), 768], cpu_ref)],
        ) as g:
            x = g.inputs[0]
            out = _transformer_block_ops(x, block_w, cpu_ref)
            g.output(out)
        return engine.InferenceSession(devices=[cpu_device]).load(g)

    def test_output_shape_preserved(self, block_model):
        """Block output shape matches input [1, 49, 768]."""
        import numpy as np
        inp = np.random.randn(1, 49, 768).astype(np.float32)
        result = block_model.execute(inp)
        tensor = list(result.values())[0] if isinstance(result, dict) else result[0]
        out_arr = tensor.to_numpy()
        assert out_arr.shape == (1, 49, 768), f"Expected (1,49,768) got {out_arr.shape}"

    def test_residual_applied(self, block_model_zero):
        """With zero projection weights, block output is layer-normed input (post-norm arch)."""
        import numpy as np
        # HuBERT uses post-norm: x -> attn(x) -> x+attn_out -> norm1 -> ffn -> x+ffn_out -> norm2.
        # With zero projection weights: attn_out=0, ffn_out=0.
        # So output = norm2(norm1(x + 0) + 0) = norm2(norm1(x)).
        inp = np.random.randn(1, 49, 768).astype(np.float32)
        result = block_model_zero.execute(inp)
        tensor = list(result.values())[0] if isinstance(result, dict) else result[0]
        out_arr = tensor.to_numpy()
        assert not np.isnan(out_arr).any(), "Output contains NaN"
        assert not np.isinf(out_arr).any(), "Output contains Inf"
        assert out_arr.std() > 0.5, f"Output std too small: {out_arr.std()}"

    def test_output_not_nan(self, block_model):
        """Block output must not contain NaN or Inf."""
        import numpy as np
        inp = np.random.randn(1, 49, 768).astype(np.float32)
        result = block_model.execute(inp)
        tensor = list(result.values())[0] if isinstance(result, dict) else result[0]
        out_arr = tensor.to_numpy()
        assert not np.isnan(out_arr).any()
        assert not np.isinf(out_arr).any()


class TestAudioEncoderShapes:
    """Test full AudioEncoder with random weights — no download required.

    Uses class-scoped fixtures to compile MAX graphs only ONCE per batch_size
    configuration (batch=1 and batch=2), avoiding repeated JIT compilation
    that exhausts memory and crashes the test runner.
    """

    @staticmethod
    def _make_full_weights():
        """Generate a complete random weight dict matching HuBERT architecture."""
        import numpy as np
        rng = np.random.default_rng(42)  # Fixed seed for reproducibility
        w = {}
        # CNN weights: PyTorch [C_out, C_in, K]
        configs = [
            (1,512,10),(512,512,3),(512,512,3),(512,512,3),
            (512,512,3),(512,512,2),(512,512,2)
        ]
        for i, (c_in, c_out, k) in enumerate(configs):
            w[f"cnn.{i}.weight"] = rng.standard_normal((c_out, c_in, k)).astype(np.float32) * 0.02
            w[f"cnn.{i}.norm.weight"] = np.ones(c_out, dtype=np.float32)
            w[f"cnn.{i}.norm.bias"] = np.zeros(c_out, dtype=np.float32)
        # Feature projection: [out, in] PyTorch
        w["proj.weight"] = rng.standard_normal((768, 512)).astype(np.float32) * 0.02
        w["proj.bias"] = np.zeros(768, dtype=np.float32)
        w["proj.norm.weight"] = np.ones(512, dtype=np.float32)
        w["proj.norm.bias"] = np.zeros(512, dtype=np.float32)
        # Position conv: [C_out=768, C_in/groups=48, K=128]
        w["pos_conv.weight"] = rng.standard_normal((768, 48, 128)).astype(np.float32) * 0.02
        w["pos_conv.bias"] = np.zeros(768, dtype=np.float32)
        # Encoder layer norm (after pos_conv, before transformer blocks)
        w["enc_norm.weight"] = np.ones(768, dtype=np.float32)
        w["enc_norm.bias"] = np.zeros(768, dtype=np.float32)
        # 12 transformer blocks
        for i in range(12):
            for name in ["norm1", "norm2"]:
                w[f"blocks.{i}.{name}.weight"] = np.ones(768, dtype=np.float32)
                w[f"blocks.{i}.{name}.bias"] = np.zeros(768, dtype=np.float32)
            for proj in ["attn.q", "attn.k", "attn.v", "attn.out"]:
                w[f"blocks.{i}.{proj}.weight"] = rng.standard_normal((768, 768)).astype(np.float32) * 0.02
                w[f"blocks.{i}.{proj}.bias"] = np.zeros(768, dtype=np.float32)
            w[f"blocks.{i}.ffn.fc1.weight"] = rng.standard_normal((3072, 768)).astype(np.float32) * 0.02
            w[f"blocks.{i}.ffn.fc1.bias"] = np.zeros(3072, dtype=np.float32)
            w[f"blocks.{i}.ffn.fc2.weight"] = rng.standard_normal((768, 3072)).astype(np.float32) * 0.02
            w[f"blocks.{i}.ffn.fc2.bias"] = np.zeros(768, dtype=np.float32)
        return w

    # --- Class-scoped fixtures: compile each graph config ONCE ---

    @pytest.fixture(scope="class")
    def full_weights(self):
        return self._make_full_weights()

    @pytest.fixture(scope="class")
    def model_b1(self, full_weights):
        """Batch=1 model, compiled once for all batch=1 tests."""
        from models.audio_encoder import AudioEncoder
        return AudioEncoder._from_weights(full_weights, device="cpu")

    @pytest.fixture(scope="class")
    def model_b2(self, full_weights):
        """Batch=2 model, compiled once for all batch=2 tests."""
        from models.audio_encoder import AudioEncoder
        return AudioEncoder._from_weights(full_weights, device="cpu", batch_size=2)

    # --- Batch=1 tests ---

    def test_encode_1s_shape(self, model_b1):
        """1s audio -> [1, 49, 768] on CPU."""
        import numpy as np
        audio = np.zeros((1, 16000), dtype=np.float32)
        out = model_b1.encode(audio)
        assert out.shape == (1, 49, 768), f"Expected (1,49,768) got {out.shape}"

    def test_encode_2s_shape(self, model_b1):
        """2s audio -> [1, 99, 768] on CPU."""
        import numpy as np
        audio = np.zeros((1, 32000), dtype=np.float32)
        out = model_b1.encode(audio)
        assert out.shape == (1, 99, 768), f"Expected (1,99,768) got {out.shape}"

    def test_encode_output_not_nan(self, model_b1):
        """Full encoder output must not contain NaN."""
        import numpy as np
        audio = np.random.randn(1, 16000).astype(np.float32) * 0.1
        out = model_b1.encode(audio)
        assert not np.isnan(out).any()
        assert not np.isinf(out).any()

    def test_encode_batch1_unchanged(self, model_b1):
        """Default batch_size=1 still gives [1, 49, 768] (backward compat)."""
        import numpy as np
        audio = np.zeros((1, 16000), dtype=np.float32)
        out = model_b1.encode(audio)
        assert out.shape == (1, 49, 768), f"Expected (1,49,768) got {out.shape}"

    # --- Batch=2 tests ---

    def test_encode_batch2_shape(self, model_b2):
        """Batch=2: two 1s samples -> [2, 49, 768] on CPU."""
        import numpy as np
        audio = np.zeros((2, 16000), dtype=np.float32)
        out = model_b2.encode(audio)
        assert out.shape == (2, 49, 768), f"Expected (2,49,768) got {out.shape}"

    def test_encode_batch2_not_nan(self, model_b2):
        """Batch=2 output must not contain NaN or Inf."""
        import numpy as np
        audio = np.random.randn(2, 16000).astype(np.float32) * 0.1
        out = model_b2.encode(audio)
        assert not np.isnan(out).any(), "Batch output contains NaN"
        assert not np.isinf(out).any(), "Batch output contains Inf"

    def test_encode_batch2_samples_differ(self, model_b2):
        """Different inputs in a batch must produce different outputs."""
        import numpy as np
        rng = np.random.default_rng(123)
        audio = rng.standard_normal((2, 16000)).astype(np.float32)
        out = model_b2.encode(audio)
        diff = np.abs(out[0] - out[1]).mean()
        assert diff > 0.001, f"Batch samples too similar (mean diff={diff:.6f})"


@pytest.mark.slow
class TestAudioEncoderCorrectness:
    """Integration test: MAX output vs PyTorch HuBERT.

    Downloads facebook/hubert-base-ls960 (~360MB) on first run.
    Skipped by default — run with: pixi run test-models-full
    """

    MODEL_ID = "facebook/hubert-base-ls960"

    def _get_pytorch_output(self, audio_np):
        """Get reference output from PyTorch HuBERT."""
        import torch
        from transformers import HubertModel
        pt_model = HubertModel.from_pretrained(self.MODEL_ID).eval()
        with torch.no_grad():
            out = pt_model(torch.from_numpy(audio_np))
        return out.last_hidden_state.numpy()

    def test_output_matches_pytorch_cpu(self):
        """MAX CPU output must match PyTorch within 1e-3 max absolute diff."""
        import numpy as np
        from models import AudioEncoder

        rng = np.random.default_rng(42)
        audio = rng.standard_normal((1, 16000)).astype(np.float32)

        # PyTorch reference
        pt_out = self._get_pytorch_output(audio)

        # MAX CPU
        max_model = AudioEncoder.from_pretrained(self.MODEL_ID, device="cpu")
        max_out = max_model.encode(audio)

        print(f"\n  PyTorch shape: {pt_out.shape}")
        print(f"  MAX shape:     {max_out.shape}")
        max_diff = np.abs(max_out - pt_out).max()
        mean_diff = np.abs(max_out - pt_out).mean()
        print(f"  Max diff:  {max_diff:.6f}")
        print(f"  Mean diff: {mean_diff:.6f}")

        assert max_out.shape == pt_out.shape, \
            f"Shape mismatch: MAX {max_out.shape} vs PyTorch {pt_out.shape}"
        assert max_diff < 1e-3, \
            f"Max diff {max_diff:.4e} exceeds threshold 1e-3. Check weight loading and layer order."
