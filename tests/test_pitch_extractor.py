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

    def test_conv_transpose_numerically_equivalent(self):
        """MAX ConvTranspose implementation matches PyTorch ConvTranspose2d numerically.

        Both sides use stride=2, kernel=3, padding=1, output_padding=1 (the RMVPE
        configuration).  Input H×W → output 2H×2W.
        """
        import numpy as np
        import torch
        from max import engine
        from max.driver import CPU
        from max.graph import Graph, TensorType, DeviceRef
        from max.dtype import DType
        from models._rmvpe import _conv_transpose_2x

        rng = np.random.default_rng(7)
        B, H, W, C_in, C_out, K = 1, 8, 8, 4, 8, 3
        x_np = rng.standard_normal((B, H, W, C_in)).astype(np.float32)
        w_pt = rng.standard_normal((C_in, C_out, K, K)).astype(np.float32) * 0.1

        # PyTorch reference: output_padding=1 matches _conv_transpose_2x → output 2H×2W
        pt_conv = torch.nn.ConvTranspose2d(
            C_in, C_out, K, stride=2, padding=1, output_padding=1, bias=False
        )
        pt_conv.weight.data = torch.from_numpy(w_pt)
        x_pt = torch.from_numpy(x_np.transpose(0, 3, 1, 2))  # NHWC → NCHW
        with torch.no_grad():
            ref = pt_conv(x_pt).numpy().transpose(0, 2, 3, 1)  # NCHW → NHWC

        # MAX implementation via zero-interleave + conv
        cpu_ref = DeviceRef.CPU()
        with Graph("ct_test", input_types=[TensorType(DType.float32, [B, H, W, C_in], cpu_ref)]) as g:
            x = g.inputs[0]
            out = _conv_transpose_2x(x, w_pt, None, cpu_ref)
            g.output(out)

        model = engine.InferenceSession(devices=[CPU()]).load(g)
        result = model.execute(x_np)
        max_out = (list(result.values())[0] if isinstance(result, dict) else result[0]).to_numpy()

        assert max_out.shape == ref.shape, f"Shape mismatch: MAX {max_out.shape} vs PyTorch {ref.shape}"
        diff = np.abs(max_out - ref).max()
        assert diff < 1e-4, f"ConvTranspose diff {diff:.2e} exceeds tolerance"


class TestBiGRU:
    """BiGRU numpy implementation tests."""

    def _make_gru_weights(self, hidden=256, input_size=384):
        """Random BiGRU weights matching internal naming."""
        rng = np.random.default_rng(3)
        w = {}
        gate_size = 3 * hidden
        for suffix, shape in [
            ("weight_ih_l0", (gate_size, input_size)),
            ("weight_hh_l0", (gate_size, hidden)),
            ("bias_ih_l0", (gate_size,)),
            ("bias_hh_l0", (gate_size,)),
            ("weight_ih_l0_reverse", (gate_size, input_size)),
            ("weight_hh_l0_reverse", (gate_size, hidden)),
            ("bias_ih_l0_reverse", (gate_size,)),
            ("bias_hh_l0_reverse", (gate_size,)),
        ]:
            w[f"gru.{suffix}"] = rng.standard_normal(shape).astype(np.float32) * 0.01
        return w

    def test_output_shape(self):
        """BiGRU: [1, T, 384] -> [1, T, 512]."""
        from models._rmvpe import bigru_forward
        weights = self._make_gru_weights()
        x = np.random.randn(1, 100, 384).astype(np.float32) * 0.1
        out = bigru_forward(x, weights)
        assert out.shape == (1, 100, 512), f"Expected (1,100,512) got {out.shape}"

    def test_output_not_nan(self):
        from models._rmvpe import bigru_forward
        weights = self._make_gru_weights()
        x = np.random.randn(1, 50, 384).astype(np.float32) * 0.1
        out = bigru_forward(x, weights)
        assert not np.isnan(out).any()
        assert not np.isinf(out).any()

    def test_forward_reverse_differ(self):
        """Forward and reverse halves must differ (not identical)."""
        from models._rmvpe import bigru_forward
        weights = self._make_gru_weights()
        x = np.random.randn(1, 20, 384).astype(np.float32)
        out = bigru_forward(x, weights)
        fwd = out[:, :, :256]
        rev = out[:, :, 256:]
        assert not np.allclose(fwd, rev), "Forward and reverse GRU outputs are identical (bug)"

    def test_causal_ordering(self):
        """Forward half must depend on past, reverse half on future."""
        from models._rmvpe import bigru_forward
        rng = np.random.default_rng(5)
        weights = self._make_gru_weights()
        x = rng.standard_normal((1, 10, 384)).astype(np.float32) * 0.1
        x_same_start = x.copy()
        x_diff_end = x.copy()
        x_diff_end[:, 5:, :] = rng.standard_normal((1, 5, 384)).astype(np.float32)
        out_orig = bigru_forward(x_same_start, weights)
        out_diff = bigru_forward(x_diff_end, weights)
        # Forward half at t=0..4 should be IDENTICAL (only depends on past 0..t)
        assert np.allclose(out_orig[:, :5, :256], out_diff[:, :5, :256]), \
            "Forward half changed when only future frames differ — not causal"
        # Reverse half at t=0..4 should DIFFER (depends on future frames 5..9 which changed)
        assert not np.allclose(out_orig[:, :5, 256:], out_diff[:, :5, 256:]), \
            "Reverse half unchanged when future frames differ — bug"


class TestPitchPostProcessing:
    """Tests for pitch salience -> Hz conversion."""

    def test_to_hz_shape(self):
        """Salience [1, T, 360] -> Hz [T] float32."""
        from models._rmvpe import salience_to_hz
        rng = np.random.default_rng(10)
        salience = rng.random((1, 100, 360)).astype(np.float32)
        hz = salience_to_hz(salience, threshold=0.03)
        assert hz.shape == (100,), f"Expected (100,) got {hz.shape}"
        assert hz.dtype == np.float32

    def test_hz_range_voiced(self):
        """Voiced frames must be in reasonable pitch range (20-5000 Hz)."""
        from models._rmvpe import salience_to_hz
        salience = np.zeros((1, 10, 360), dtype=np.float32)
        salience[:, :, 180] = 1.0  # peak at bin 180
        hz = salience_to_hz(salience, threshold=0.03)
        voiced = hz[hz > 0]
        assert len(voiced) == 10
        assert (voiced > 20).all() and (voiced < 5000).all(), \
            f"Voiced Hz out of range: min={voiced.min():.1f}, max={voiced.max():.1f}"

    def test_unvoiced_returns_zero(self):
        """Low-salience frames must return 0 Hz.

        Logit of -5.0 -> sigmoid ~= 0.0067, well below threshold=0.03.
        """
        from models._rmvpe import salience_to_hz
        salience = np.full((1, 10, 360), -5.0, dtype=np.float32)
        hz = salience_to_hz(salience, threshold=0.03)
        assert (hz == 0).all(), f"Expected all zeros, got: {hz}"

    def test_linear_output_shape(self):
        """linear_output: [1, T, 512] -> [1, T, 360]."""
        from models._rmvpe import linear_output
        rng = np.random.default_rng(6)
        weights = {
            "linear.weight": rng.standard_normal((360, 512)).astype(np.float32),
            "linear.bias": np.zeros(360, dtype=np.float32),
        }
        x = rng.standard_normal((1, 50, 512)).astype(np.float32)
        out = linear_output(x, weights)
        assert out.shape == (1, 50, 360), f"Expected (1,50,360) got {out.shape}"


class TestMelSpectrogram:
    """Mel spectrogram preprocessing — no download required."""

    def test_output_shape_1s(self):
        """1s @16kHz → [1, 1, ~100, 128]."""
        from models.pitch_extractor import _mel_spectrogram
        audio = np.zeros((1, 16000), dtype=np.float32)
        mel = _mel_spectrogram(audio)
        assert mel.ndim == 4
        assert mel.shape[0] == 1 and mel.shape[1] == 1 and mel.shape[3] == 128
        assert 95 <= mel.shape[2] <= 105, f"Unexpected T: {mel.shape[2]}"

    def test_output_dtype(self):
        from models.pitch_extractor import _mel_spectrogram
        audio = np.zeros((1, 16000), dtype=np.float32)
        assert _mel_spectrogram(audio).dtype == np.float32

    def test_output_not_nan(self):
        from models.pitch_extractor import _mel_spectrogram
        audio = np.random.default_rng(0).standard_normal((1, 16000)).astype(np.float32) * 0.1
        mel = _mel_spectrogram(audio)
        assert not np.isnan(mel).any() and not np.isinf(mel).any()

    def test_output_shape_2s(self):
        """2s @16kHz → [1, 1, ~200, 128]."""
        from models.pitch_extractor import _mel_spectrogram
        audio = np.zeros((1, 32000), dtype=np.float32)
        mel = _mel_spectrogram(audio)
        assert 195 <= mel.shape[2] <= 205, f"Unexpected T: {mel.shape[2]}"


class TestPitchExtractorShapes:
    """PitchExtractor with random weights — no download required."""

    def _make_weights(self):
        """Full weight dict: U-Net + BiGRU + linear output."""
        rng = np.random.default_rng(99)
        w = TestUNetGraph()._make_full_random_weights()
        # Add BiGRU weights (hidden=256, input_size=384)
        H, I = 256, 384
        gate = 3 * H
        for suffix, shape in [
            ("weight_ih_l0", (gate, I)),
            ("weight_hh_l0", (gate, H)),
            ("bias_ih_l0", (gate,)),
            ("bias_hh_l0", (gate,)),
            ("weight_ih_l0_reverse", (gate, I)),
            ("weight_hh_l0_reverse", (gate, H)),
            ("bias_ih_l0_reverse", (gate,)),
            ("bias_hh_l0_reverse", (gate,)),
        ]:
            w[f"gru.{suffix}"] = rng.standard_normal(shape).astype(np.float32) * 0.01
        # Add linear output weights
        w["linear.weight"] = rng.standard_normal((360, 512)).astype(np.float32) * 0.01
        w["linear.bias"] = np.zeros(360, dtype=np.float32)
        return w

    def test_from_weights_builds(self):
        from models.pitch_extractor import PitchExtractor
        model = PitchExtractor._from_weights(self._make_weights(), device="cpu")
        assert model is not None

    def test_extract_1s_shape(self):
        """1s audio → [T_frames] Hz, T ≈ 100."""
        from models.pitch_extractor import PitchExtractor
        model = PitchExtractor._from_weights(self._make_weights(), device="cpu")
        audio = np.zeros((1, 16000), dtype=np.float32)
        f0 = model.extract(audio)
        assert f0.ndim == 1
        assert 95 <= len(f0) <= 105, f"Expected ~100 frames, got {len(f0)}"
        assert f0.dtype == np.float32

    def test_extract_not_nan(self):
        from models.pitch_extractor import PitchExtractor
        model = PitchExtractor._from_weights(self._make_weights(), device="cpu")
        audio = np.random.default_rng(1).standard_normal((1, 16000)).astype(np.float32) * 0.1
        f0 = model.extract(audio)
        assert not np.isnan(f0).any()

    def test_extract_returns_float32(self):
        from models.pitch_extractor import PitchExtractor
        model = PitchExtractor._from_weights(self._make_weights(), device="cpu")
        f0 = model.extract(np.zeros((1, 16000), dtype=np.float32))
        assert f0.dtype == np.float32


@pytest.mark.slow
class TestPitchExtractorCorrectness:
    """Integration test: compare MAX RMVPE output against PyTorch reference.

    Requires rmvpe.pt download (~181MB). Run with: pixi run test-pitch-extractor-full
    """

    def test_salience_matches_pytorch(self):
        """MAX RMVPE salience output matches PyTorch E2E on same mel input."""
        import sys
        import numpy as np
        import torch

        # --- PyTorch reference ---
        applio_paths = ["/home/visage/repos/Applio", "/home/maskkiller/repos/Applio"]
        applio = next((p for p in applio_paths if __import__("os").path.isdir(p)), None)
        assert applio is not None, "Applio not found — needed for PyTorch RMVPE reference"
        if applio not in sys.path:
            sys.path.insert(0, applio)

        from rvc.lib.predictors.RMVPE import E2E

        rmvpe_paths = [
            f"{applio}/rvc/models/predictors/rmvpe.pt",
            f"{applio}/rvc/models/pretraineds/rmvpe.pt",
        ]
        rmvpe_pt = next((p for p in rmvpe_paths if __import__("os").path.exists(p)), None)
        assert rmvpe_pt is not None, "rmvpe.pt not found in Applio"

        pt_model = E2E(4, 1, (2, 2))
        pt_model.load_state_dict(torch.load(rmvpe_pt, map_location="cpu", weights_only=True))
        pt_model.eval()

        # Create deterministic mel input [1, 128, 64] (64 frames, must be multiple of 32)
        rng = np.random.RandomState(42)
        mel_np = rng.randn(1, 128, 64).astype(np.float32)

        with torch.no_grad():
            pt_hidden = pt_model(torch.from_numpy(mel_np)).numpy()  # [1, 64, 360]

        # --- MAX pipeline ---
        from models.pitch_extractor import PitchExtractor
        from models._rmvpe import bigru_forward, linear_output

        max_model = PitchExtractor.from_pretrained()

        # Run U-Net: expects NHWC [1, T, 128, 1]
        mel_nhwc = mel_np.transpose(0, 2, 1)[:, :, :, np.newaxis]  # [1, 64, 128, 1]
        mel_nhwc = np.ascontiguousarray(mel_nhwc)

        result = max_model._unet_model.execute(mel_nhwc)
        unet_out = (list(result.values())[0] if isinstance(result, dict) else result[0]).to_numpy()

        # BiGRU + linear
        gru_out = bigru_forward(unet_out, max_model._weights)
        max_hidden = linear_output(gru_out, max_model._weights)  # [1, 64, 360]

        # --- Compare ---
        max_diff = np.abs(max_hidden - pt_hidden).max()
        mean_diff = np.abs(max_hidden - pt_hidden).mean()
        correlation = np.corrcoef(max_hidden.flatten(), pt_hidden.flatten())[0, 1]

        print(f"\n  PitchExtractor numerical validation:")
        print(f"  Max diff:     {max_diff:.6f}")
        print(f"  Mean diff:    {mean_diff:.6f}")
        print(f"  Correlation:  {correlation:.6f}")
        print(f"  MAX range:    [{max_hidden.min():.4f}, {max_hidden.max():.4f}]")
        print(f"  PT  range:    [{pt_hidden.min():.4f}, {pt_hidden.max():.4f}]")

        assert max_diff < 0.01, f"Max diff {max_diff} >= 0.01"
        assert correlation > 0.99, f"Correlation {correlation} < 0.99"
