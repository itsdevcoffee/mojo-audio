"""Tests for RMVPE-based PitchExtractor.

Level 1 (no download): no marker — run via: pixi run test-pitch-extractor
Level 2 (download required): @pytest.mark.slow — run via: pixi run test-pitch-extractor-full
"""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import numpy as np
import pytest


@pytest.fixture
def rng():
    return np.random.default_rng(42)


class TestRmvpeWeightLoader:
    """Tests for _rmvpe_weight_loader — no checkpoint download required."""

    def _make_fake_raw_weights(self):
        """Minimal synthetic checkpoint mimicking rmvpe.pt key/shape structure."""
        import numpy as np
        w = {}
        # Initial BN
        for k in ["weight", "bias", "running_mean", "running_var"]:
            w[f"unet.encoder.bn.{k}"] = np.ones(1, dtype=np.float32)
        w["unet.encoder.bn.num_batches_tracked"] = np.array(0, dtype=np.int64)
        # Encoder level 0, block 0 (channels 1→16, has shortcut)
        w["unet.encoder.layers.0.conv.0.conv.0.weight"] = np.random.randn(16, 1, 3, 3).astype(np.float32)
        w["unet.encoder.layers.0.conv.0.conv.0.bias"] = np.zeros(16, dtype=np.float32)
        for k in ["weight", "bias", "running_mean", "running_var"]:
            w[f"unet.encoder.layers.0.conv.0.conv.1.{k}"] = np.ones(16, dtype=np.float32)
        w["unet.encoder.layers.0.conv.0.conv.1.num_batches_tracked"] = np.array(0, dtype=np.int64)
        w["unet.encoder.layers.0.conv.0.shortcut.weight"] = np.random.randn(16, 1, 1, 1).astype(np.float32)
        # Output CNN
        w["cnn.weight"] = np.random.randn(3, 16, 3, 3).astype(np.float32)
        w["cnn.bias"] = np.zeros(3, dtype=np.float32)
        # BiGRU
        w["fc.0.gru.weight_ih_l0"] = np.random.randn(768, 384).astype(np.float32)
        w["fc.0.gru.weight_hh_l0"] = np.random.randn(768, 256).astype(np.float32)
        w["fc.0.gru.bias_ih_l0"] = np.zeros(768, dtype=np.float32)
        w["fc.0.gru.bias_hh_l0"] = np.zeros(768, dtype=np.float32)
        w["fc.0.gru.weight_ih_l0_reverse"] = np.random.randn(768, 384).astype(np.float32)
        w["fc.0.gru.weight_hh_l0_reverse"] = np.random.randn(768, 256).astype(np.float32)
        w["fc.0.gru.bias_ih_l0_reverse"] = np.zeros(768, dtype=np.float32)
        w["fc.0.gru.bias_hh_l0_reverse"] = np.zeros(768, dtype=np.float32)
        # Linear output
        w["fc.1.weight"] = np.random.randn(360, 512).astype(np.float32)
        w["fc.1.bias"] = np.zeros(360, dtype=np.float32)
        return w

    def test_load_from_dict_returns_dict(self):
        from models._rmvpe_weight_loader import load_rmvpe_from_dict
        result = load_rmvpe_from_dict(self._make_fake_raw_weights())
        assert isinstance(result, dict)
        assert len(result) > 0

    def test_bn_baked_into_scale_offset(self):
        """BatchNorm running stats must be baked — no raw running_mean in output."""
        from models._rmvpe_weight_loader import load_rmvpe_from_dict
        result = load_rmvpe_from_dict(self._make_fake_raw_weights())
        for key in result:
            assert "running_mean" not in key, f"Unbaked BN key found: {key}"
            assert "running_var" not in key, f"Unbaked BN key found: {key}"
            assert "num_batches_tracked" not in key

    def test_gru_weights_preserved(self):
        """BiGRU weights must be in output dict with correct shapes."""
        from models._rmvpe_weight_loader import load_rmvpe_from_dict
        result = load_rmvpe_from_dict(self._make_fake_raw_weights())
        assert "gru.weight_ih_l0" in result
        assert result["gru.weight_ih_l0"].shape == (768, 384)
        assert "gru.weight_ih_l0_reverse" in result

    def test_linear_weights_preserved(self):
        """Output linear weights must be present."""
        from models._rmvpe_weight_loader import load_rmvpe_from_dict
        result = load_rmvpe_from_dict(self._make_fake_raw_weights())
        assert "linear.weight" in result
        assert result["linear.weight"].shape == (360, 512)

    def test_all_values_float32(self):
        """All output arrays must be float32."""
        from models._rmvpe_weight_loader import load_rmvpe_from_dict
        result = load_rmvpe_from_dict(self._make_fake_raw_weights())
        for key, arr in result.items():
            assert arr.dtype == np.float32, f"Key {key} has dtype {arr.dtype}"

    def test_bake_bn_correctness(self):
        """Baked BN: (x - mean) / sqrt(var + eps) * weight + bias == x * scale + offset."""
        from models._rmvpe_weight_loader import bake_batch_norm
        rng = np.random.default_rng(0)
        weight = rng.standard_normal(16).astype(np.float32)
        bias = rng.standard_normal(16).astype(np.float32)
        running_mean = rng.standard_normal(16).astype(np.float32)
        running_var = np.abs(rng.standard_normal(16)).astype(np.float32)
        scale, offset = bake_batch_norm(weight, bias, running_mean, running_var)
        # Verify against reference formula
        x = rng.standard_normal((4, 16)).astype(np.float32)
        ref = (x - running_mean) / np.sqrt(running_var + 1e-5) * weight + bias
        out = x * scale + offset
        assert np.allclose(ref, out, atol=1e-5), f"Max diff: {np.abs(ref - out).max()}"
