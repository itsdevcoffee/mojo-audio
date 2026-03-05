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


@pytest.fixture
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
        """Detect 'hubert' prefix in HuBERT checkpoint keys."""
        from models._weight_loader import _detect_prefix
        keys = [
            "hubert.feature_extractor.conv_layers.0.conv.weight",
            "hubert.encoder.layers.0.attention.q_proj.weight",
        ]
        assert _detect_prefix(keys) == "hubert"

    def test_detect_contentvec_prefix(self):
        """Detect 'model' prefix in ContentVec checkpoint keys."""
        from models._weight_loader import _detect_prefix
        keys = [
            "model.feature_extractor.conv_layers.0.conv.weight",
            "model.encoder.layers.0.attention.q_proj.weight",
        ]
        assert _detect_prefix(keys) == "model"

    def test_detect_prefix_raises_on_unknown(self):
        """Unknown prefix raises ValueError."""
        from models._weight_loader import _detect_prefix
        with pytest.raises(ValueError):
            _detect_prefix(["some.unknown.key"])

    def test_map_cnn_weight(self):
        """CNN conv weight key maps correctly."""
        from models._weight_loader import _map_key
        assert _map_key("hubert.feature_extractor.conv_layers.3.conv.weight", "hubert") == "cnn.3.weight"

    def test_map_cnn_norm(self):
        """CNN layer norm weight key maps correctly."""
        from models._weight_loader import _map_key
        assert _map_key("hubert.feature_extractor.conv_layers.0.layer_norm.weight", "hubert") == "cnn.0.norm.weight"

    def test_map_projection(self):
        """Feature projection weight maps correctly."""
        from models._weight_loader import _map_key
        assert _map_key("hubert.feature_projection.projection.weight", "hubert") == "proj.weight"

    def test_map_transformer_attention_q(self):
        """Transformer layer attention q_proj maps correctly."""
        from models._weight_loader import _map_key
        assert _map_key("hubert.encoder.layers.5.attention.q_proj.weight", "hubert") == "blocks.5.attn.q.weight"

    def test_map_transformer_ffn(self):
        """Transformer layer FFN maps correctly."""
        from models._weight_loader import _map_key
        assert _map_key("hubert.encoder.layers.0.feed_forward.intermediate_dense.weight", "hubert") == "blocks.0.ffn.fc1.weight"

    def test_map_unknown_key_returns_none(self):
        """Unknown key returns None (will be skipped)."""
        from models._weight_loader import _map_key
        assert _map_key("some.unknown.weight", "hubert") is None

    def test_map_contentvec_prefix(self):
        """ContentVec uses 'model' prefix — maps same way."""
        from models._weight_loader import _map_key
        assert _map_key("model.feature_extractor.conv_layers.0.conv.weight", "model") == "cnn.0.weight"

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
    """Tests for CNN feature extractor — no model download required."""

    def _make_random_weights(self):
        """Random weights in PyTorch format [C_out, C_in, K]."""
        import numpy as np
        configs = [
            (1, 512, 10), (512, 512, 3), (512, 512, 3), (512, 512, 3),
            (512, 512, 3), (512, 512, 2), (512, 512, 2),
        ]
        w = {}
        for i, (c_in, c_out, k) in enumerate(configs):
            w[f"cnn.{i}.weight"] = np.random.randn(c_out, c_in, k).astype(np.float32)
            w[f"cnn.{i}.norm.weight"] = np.ones(c_out, dtype=np.float32)
            w[f"cnn.{i}.norm.bias"] = np.zeros(c_out, dtype=np.float32)
        return w

    def _result_to_numpy(self, result):
        """Extract numpy array from model.execute() result (list or dict of MAX Tensors)."""
        import numpy as np
        v = list(result.values())[0] if isinstance(result, dict) else result[0]
        return v.to_numpy() if hasattr(v, "to_numpy") else np.array(v)

    def test_output_shape_1s(self, cpu_device):
        """1s @16kHz -> [1, 49, 512]."""
        import numpy as np
        from max import engine
        from max.graph import DeviceRef
        from models._feature_extractor import build_feature_extractor_graph

        cpu_ref = DeviceRef.CPU()
        graph = build_feature_extractor_graph(self._make_random_weights(), cpu_ref)
        model = engine.InferenceSession(devices=[cpu_device]).load(graph)

        audio = np.zeros((1, 16000, 1, 1), dtype=np.float32)
        result = model.execute(audio)
        out = self._result_to_numpy(result)
        assert out.shape == (1, 49, 512), f"Expected (1,49,512) got {out.shape}"

    def test_output_shape_2s(self, cpu_device):
        """2s @16kHz -> [1, 99, 512]."""
        import numpy as np
        from max import engine
        from max.graph import DeviceRef
        from models._feature_extractor import build_feature_extractor_graph

        cpu_ref = DeviceRef.CPU()
        graph = build_feature_extractor_graph(self._make_random_weights(), cpu_ref)
        model = engine.InferenceSession(devices=[cpu_device]).load(graph)

        audio = np.zeros((1, 32000, 1, 1), dtype=np.float32)
        result = model.execute(audio)
        out = self._result_to_numpy(result)
        assert out.shape == (1, 99, 512), f"Expected (1,99,512) got {out.shape}"

    def test_output_not_nan(self, cpu_device, audio_1s):
        """Output must not contain NaN or Inf."""
        import numpy as np
        from max import engine
        from max.graph import DeviceRef
        from models._feature_extractor import build_feature_extractor_graph

        cpu_ref = DeviceRef.CPU()
        graph = build_feature_extractor_graph(self._make_random_weights(), cpu_ref)
        model = engine.InferenceSession(devices=[cpu_device]).load(graph)

        audio_in = audio_1s.reshape(1, 16000, 1, 1)
        result = model.execute(audio_in)
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
