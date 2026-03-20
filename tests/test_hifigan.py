"""Tests for NSF-HiFiGAN weight loader.

Covers config parsing, weight-norm reconstruction, and dec.* key extraction.
"""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import numpy as np
import pytest


class TestHiFiGANWeightLoader:
    """Tests for _hifigan_weight_loader — no checkpoint download required."""

    def _make_config_list(self, sr: int, upsample_rates: list[int]) -> list:
        """Build a 17-element RVC v2 config list.

        Only indices 3, 10-15 matter; the rest are placeholders.
        """
        cfg = [None] * 17
        cfg[3] = 192                             # inter_channels
        cfg[10] = "1"                            # resblock type
        cfg[11] = [3, 7, 11]                     # resblock_kernel_sizes
        cfg[12] = [[1, 3, 5], [1, 3, 5], [1, 3, 5]]  # resblock_dilation_sizes
        cfg[13] = upsample_rates
        cfg[14] = 512                            # upsample_initial_channel
        cfg[15] = [k * 2 for k in upsample_rates]  # upsample_kernel_sizes (common convention)
        return cfg, sr

    # ---- test_parse_config_48k ----
    def test_parse_config_48k(self):
        from models._hifigan_weight_loader import parse_hifigan_config

        cfg_list, sr = self._make_config_list(48000, [12, 10, 2, 2])
        cfg = parse_hifigan_config(cfg_list, sr)

        assert cfg["sr"] == 48000
        assert cfg["inter_channels"] == 192
        assert cfg["resblock"] == "1"
        assert cfg["resblock_kernel_sizes"] == [3, 7, 11]
        assert cfg["resblock_dilation_sizes"] == [[1, 3, 5], [1, 3, 5], [1, 3, 5]]
        assert cfg["upsample_rates"] == [12, 10, 2, 2]
        assert cfg["upsample_initial_channel"] == 512
        assert cfg["upsample_kernel_sizes"] == [24, 20, 4, 4]
        assert cfg["hop_length"] == 480  # 12 * 10 * 2 * 2

    # ---- test_parse_config_40k ----
    def test_parse_config_40k(self):
        from models._hifigan_weight_loader import parse_hifigan_config

        cfg_list, sr = self._make_config_list(40000, [10, 10, 2, 2])
        cfg = parse_hifigan_config(cfg_list, sr)

        assert cfg["sr"] == 40000
        assert cfg["upsample_rates"] == [10, 10, 2, 2]
        assert cfg["hop_length"] == 400  # 10 * 10 * 2 * 2
        assert cfg["upsample_kernel_sizes"] == [20, 20, 4, 4]

    # ---- test_weight_norm_reconstruction ----
    def test_weight_norm_reconstruction(self):
        from models._hifigan_weight_loader import reconstruct_weight_norm

        rng = np.random.default_rng(42)
        weight_v = rng.standard_normal((64, 32, 7)).astype(np.float32)
        weight_g = rng.standard_normal((64, 1, 1)).astype(np.float32)

        result = reconstruct_weight_norm(weight_v, weight_g)

        assert result.dtype == np.float32
        assert result.shape == (64, 32, 7)

        # Manual reference calculation
        norm = np.linalg.norm(weight_v.reshape(64, -1), axis=1).reshape(64, 1, 1)
        expected = weight_v * (weight_g / norm)
        np.testing.assert_allclose(result, expected, rtol=1e-5)

    # ---- test_extract_dec_keys ----
    def test_extract_dec_keys(self):
        from models._hifigan_weight_loader import extract_hifigan_weights

        rng = np.random.default_rng(99)

        fake_sd = {}
        # Non-dec keys (should be ignored)
        fake_sd["enc.embed.weight"] = rng.standard_normal((192, 256)).astype(np.float32)
        fake_sd["flow.conv.weight"] = rng.standard_normal((192, 192, 1)).astype(np.float32)

        # dec.conv_pre (plain weight, no weight norm)
        fake_sd["dec.conv_pre.weight"] = rng.standard_normal((512, 192, 7)).astype(np.float32)
        fake_sd["dec.conv_pre.bias"] = np.zeros(512, dtype=np.float32)

        # dec.ups.0 with weight norm
        weight_v = rng.standard_normal((256, 512, 24)).astype(np.float32)
        weight_g = np.abs(rng.standard_normal((256, 1, 1)).astype(np.float32)) + 0.1
        fake_sd["dec.ups.0.weight_v"] = weight_v
        fake_sd["dec.ups.0.weight_g"] = weight_g

        # dec.conv_post (plain weight)
        fake_sd["dec.conv_post.weight"] = rng.standard_normal((1, 256, 7)).astype(np.float32)
        fake_sd["dec.conv_post.bias"] = np.zeros(1, dtype=np.float32)

        result = extract_hifigan_weights(fake_sd)

        # Non-dec keys should NOT appear
        assert "enc.embed.weight" not in result
        assert "flow.conv.weight" not in result

        # dec. prefix should be stripped
        assert "conv_pre.weight" in result
        assert "conv_pre.bias" in result
        assert "conv_post.weight" in result
        assert "conv_post.bias" in result

        # Weight norm should be reconstructed into plain "weight"
        assert "ups.0.weight" in result
        assert "ups.0.weight_v" not in result
        assert "ups.0.weight_g" not in result

        # Verify reconstruction is correct
        norm = np.linalg.norm(weight_v.reshape(256, -1), axis=1).reshape(256, 1, 1)
        expected = weight_v * (weight_g / norm)
        np.testing.assert_allclose(result["ups.0.weight"], expected, rtol=1e-5)

        # All values should be float32
        for k, v in result.items():
            assert v.dtype == np.float32, f"{k} has dtype {v.dtype}, expected float32"


class TestConvTranspose:
    """Tests for generalized ConvTranspose1d via zero-interleave + regular conv2d."""

    @pytest.fixture(scope="class")
    def cpu_device(self):
        from max.driver import CPU
        return CPU()

    @staticmethod
    def _result_to_numpy(result):
        """Extract numpy array from model.execute() result."""
        v = list(result.values())[0] if isinstance(result, dict) else result[0]
        return v.to_numpy() if hasattr(v, "to_numpy") else np.array(v)

    def _run_conv_transpose(self, cpu_device, C_in, C_out, K, S, T_in):
        """Build a minimal MAX graph with one conv_transpose_1d and execute it."""
        from max import engine
        from max.graph import Graph, TensorType, DeviceRef, Dim
        from max.dtype import DType
        from models._hifigan_graph import conv_transpose_1d

        rng = np.random.default_rng(42)
        # PyTorch ConvTranspose1d weight: [C_in, C_out, K]
        w_pt = rng.standard_normal((C_in, C_out, K)).astype(np.float32) * 0.01
        b_np = np.zeros(C_out, dtype=np.float32)

        T = Dim("T")
        dev = DeviceRef.CPU()
        with Graph(
            "test_conv_transpose",
            input_types=[TensorType(DType.float32, [1, T, 1, C_in], dev)],
        ) as g:
            x = g.inputs[0]
            out = conv_transpose_1d(x, w_pt, b_np, stride=S, device_ref=dev)
            g.output(out)

        model = engine.InferenceSession(devices=[cpu_device]).load(g)
        x_np = rng.standard_normal((1, T_in, 1, C_in)).astype(np.float32) * 0.1
        result = model.execute(x_np)
        return self._result_to_numpy(result)

    def test_stride_2(self, cpu_device):
        """C_in=64, C_out=32, K=4, S=2, T_in=10 -> T_out=20."""
        out = self._run_conv_transpose(cpu_device, C_in=64, C_out=32, K=4, S=2, T_in=10)
        assert out.shape == (1, 20, 1, 32), f"Expected (1,20,1,32) got {out.shape}"

    def test_stride_10(self, cpu_device):
        """C_in=256, C_out=128, K=16, S=10, T_in=10 -> T_out=100."""
        out = self._run_conv_transpose(cpu_device, C_in=256, C_out=128, K=16, S=10, T_in=10)
        assert out.shape == (1, 100, 1, 128), f"Expected (1,100,1,128) got {out.shape}"

    def test_stride_12(self, cpu_device):
        """C_in=512, C_out=256, K=24, S=12, T_in=10 -> T_out=120."""
        out = self._run_conv_transpose(cpu_device, C_in=512, C_out=256, K=24, S=12, T_in=10)
        assert out.shape == (1, 120, 1, 256), f"Expected (1,120,1,256) got {out.shape}"
