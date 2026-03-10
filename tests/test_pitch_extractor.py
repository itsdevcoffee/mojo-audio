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
        rng = np.random.default_rng(42)
        w = {}
        # Initial BN
        for k in ["weight", "bias", "running_mean", "running_var"]:
            w[f"unet.encoder.bn.{k}"] = np.ones(1, dtype=np.float32)
        w["unet.encoder.bn.num_batches_tracked"] = np.array(0, dtype=np.int64)
        # Encoder level 0, block 0 (channels 1→16, has shortcut)
        w["unet.encoder.layers.0.conv.0.conv.0.weight"] = rng.standard_normal((16, 1, 3, 3)).astype(np.float32)
        w["unet.encoder.layers.0.conv.0.conv.0.bias"] = np.zeros(16, dtype=np.float32)
        for k in ["weight", "bias", "running_mean", "running_var"]:
            w[f"unet.encoder.layers.0.conv.0.conv.1.{k}"] = np.ones(16, dtype=np.float32)
        w["unet.encoder.layers.0.conv.0.conv.1.num_batches_tracked"] = np.array(0, dtype=np.int64)
        w["unet.encoder.layers.0.conv.0.shortcut.weight"] = rng.standard_normal((16, 1, 1, 1)).astype(np.float32)
        # Output CNN
        w["cnn.weight"] = rng.standard_normal((3, 16, 3, 3)).astype(np.float32)
        w["cnn.bias"] = np.zeros(3, dtype=np.float32)
        # BiGRU
        w["fc.0.gru.weight_ih_l0"] = rng.standard_normal((768, 384)).astype(np.float32)
        w["fc.0.gru.weight_hh_l0"] = rng.standard_normal((768, 256)).astype(np.float32)
        w["fc.0.gru.bias_ih_l0"] = np.zeros(768, dtype=np.float32)
        w["fc.0.gru.bias_hh_l0"] = np.zeros(768, dtype=np.float32)
        w["fc.0.gru.weight_ih_l0_reverse"] = rng.standard_normal((768, 384)).astype(np.float32)
        w["fc.0.gru.weight_hh_l0_reverse"] = rng.standard_normal((768, 256)).astype(np.float32)
        w["fc.0.gru.bias_ih_l0_reverse"] = np.zeros(768, dtype=np.float32)
        w["fc.0.gru.bias_hh_l0_reverse"] = np.zeros(768, dtype=np.float32)
        # Linear output
        w["fc.1.weight"] = rng.standard_normal((360, 512)).astype(np.float32)
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


class TestUNetGraph:
    """U-Net MAX Graph shape tests — no download required."""

    def _make_full_random_weights(self):
        """Full random weight dict matching actual RMVPE U-Net architecture."""
        import numpy as np
        rng = np.random.default_rng(2)
        w = {}
        enc_channels = [1, 16, 32, 64, 128, 256]
        SCALE = 0.01

        def _rbn(c):
            return np.ones(c, np.float32), np.zeros(c, np.float32)
        def _rconv(co, ci, k=3):
            return rng.standard_normal((co, ci, k, k)).astype(np.float32) * SCALE

        # Initial enc BN
        w["enc_bn.scale"], w["enc_bn.offset"] = _rbn(1)

        # Encoder: 5 levels, 4 blocks each
        for L in range(5):
            c_in, c_out = enc_channels[L], enc_channels[L+1]
            for B in range(4):
                ci = c_in if B == 0 else c_out
                w[f"enc.{L}.{B}.0.w"] = _rconv(c_out, ci)
                w[f"enc.{L}.{B}.0.b"] = np.zeros(c_out, np.float32)
                w[f"enc.{L}.{B}.0.scale"], w[f"enc.{L}.{B}.0.offset"] = _rbn(c_out)
                w[f"enc.{L}.{B}.1.w"] = _rconv(c_out, c_out)
                w[f"enc.{L}.{B}.1.b"] = np.zeros(c_out, np.float32)
                w[f"enc.{L}.{B}.1.scale"], w[f"enc.{L}.{B}.1.offset"] = _rbn(c_out)
                if B == 0:
                    w[f"enc.{L}.{B}.sc.w"] = _rconv(c_out, ci, k=1)
                    w[f"enc.{L}.{B}.sc.b"] = np.zeros(c_out, np.float32)

        # Bottleneck: 16 blocks (I=0..15)
        btl_in = [256] + [512]*15
        for I in range(16):
            ci, co = btl_in[I], 512
            w[f"btl.{I}.0.w"] = _rconv(co, ci)
            w[f"btl.{I}.0.b"] = np.zeros(co, np.float32)
            w[f"btl.{I}.0.scale"], w[f"btl.{I}.0.offset"] = _rbn(co)
            w[f"btl.{I}.1.w"] = _rconv(co, co)
            w[f"btl.{I}.1.b"] = np.zeros(co, np.float32)
            w[f"btl.{I}.1.scale"], w[f"btl.{I}.1.offset"] = _rbn(co)
            if I == 0:
                w[f"btl.{I}.sc.w"] = _rconv(co, ci, k=1)
                w[f"btl.{I}.sc.b"] = np.zeros(co, np.float32)

        # Decoder: 5 levels, 4 blocks each
        dec_channels = [512, 256, 128, 64, 32, 16]
        for L in range(5):
            up_ci, up_co = dec_channels[L], dec_channels[L+1]
            # ConvTranspose: PyTorch [C_in, C_out, H, W]
            w[f"dec.{L}.up.w"] = rng.standard_normal((up_ci, up_co, 3, 3)).astype(np.float32) * SCALE
            w[f"dec.{L}.up.b"] = np.zeros(up_co, np.float32)
            w[f"dec.{L}.up.scale"], w[f"dec.{L}.up.offset"] = _rbn(up_co)
            # After skip concat: C_in = up_co + enc_channels[4-L]
            enc_skip_ch = enc_channels[5 - L]  # mirror level
            skip_ci = up_co + enc_skip_ch
            for B in range(4):
                ci = skip_ci if B == 0 else up_co
                w[f"dec.{L}.{B}.0.w"] = _rconv(up_co, ci)
                w[f"dec.{L}.{B}.0.b"] = np.zeros(up_co, np.float32)
                w[f"dec.{L}.{B}.0.scale"], w[f"dec.{L}.{B}.0.offset"] = _rbn(up_co)
                w[f"dec.{L}.{B}.1.w"] = _rconv(up_co, up_co)
                w[f"dec.{L}.{B}.1.b"] = np.zeros(up_co, np.float32)
                w[f"dec.{L}.{B}.1.scale"], w[f"dec.{L}.{B}.1.offset"] = _rbn(up_co)
                if B == 0:
                    w[f"dec.{L}.{B}.sc.w"] = _rconv(up_co, ci, k=1)
                    w[f"dec.{L}.{B}.sc.b"] = np.zeros(up_co, np.float32)

        # Output CNN: [C_out=3, C_in=16, 3, 3] PyTorch
        w["out_cnn.w"] = _rconv(3, 16)
        w["out_cnn.b"] = np.zeros(3, np.float32)
        return w

    def test_unet_graph_buildable(self):
        """build_unet_graph must construct without errors."""
        from max import engine
        from max.driver import CPU
        from max.graph import DeviceRef
        from models._rmvpe import build_unet_graph

        weights = self._make_full_random_weights()
        graph = build_unet_graph(weights, DeviceRef.CPU())
        model = engine.InferenceSession(devices=[CPU()]).load(graph)
        assert model is not None

    def test_output_shape_t100(self):
        """Mel [1, T=100, 128, 1] → [1, 100, 384]."""
        import numpy as np
        from max import engine
        from max.driver import CPU
        from max.graph import DeviceRef
        from models._rmvpe import build_unet_graph

        weights = self._make_full_random_weights()
        model = engine.InferenceSession(devices=[CPU()]).load(
            build_unet_graph(weights, DeviceRef.CPU())
        )
        mel = np.random.randn(1, 100, 128, 1).astype(np.float32) * 0.1
        result = model.execute(mel)
        out = (list(result.values())[0] if isinstance(result, dict) else result[0]).to_numpy()
        assert out.shape == (1, 100, 384), f"Expected (1,100,384) got {out.shape}"

    def test_output_shape_t200(self):
        """Dynamic T: [1, 200, 128, 1] → [1, 200, 384]."""
        import numpy as np
        from max import engine
        from max.driver import CPU
        from max.graph import DeviceRef
        from models._rmvpe import build_unet_graph

        weights = self._make_full_random_weights()
        model = engine.InferenceSession(devices=[CPU()]).load(
            build_unet_graph(weights, DeviceRef.CPU())
        )
        mel = np.random.randn(1, 200, 128, 1).astype(np.float32) * 0.1
        result = model.execute(mel)
        out = (list(result.values())[0] if isinstance(result, dict) else result[0]).to_numpy()
        assert out.shape == (1, 200, 384), f"Expected (1,200,384) got {out.shape}"

    def test_output_not_nan(self):
        """U-Net output must not contain NaN or Inf."""
        import numpy as np
        from max import engine
        from max.driver import CPU
        from max.graph import DeviceRef
        from models._rmvpe import build_unet_graph

        weights = self._make_full_random_weights()
        model = engine.InferenceSession(devices=[CPU()]).load(
            build_unet_graph(weights, DeviceRef.CPU())
        )
        mel = np.random.randn(1, 100, 128, 1).astype(np.float32) * 0.1
        result = model.execute(mel)
        out = (list(result.values())[0] if isinstance(result, dict) else result[0]).to_numpy()
        assert not np.isnan(out).any(), "Output contains NaN"
        assert not np.isinf(out).any(), "Output contains Inf"
