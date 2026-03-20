"""Tests for NSF-HiFiGAN weight loader and full graph.

Covers config parsing, weight-norm reconstruction, dec.* key extraction,
and full HiFiGAN graph shape/numerical tests.
"""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from functools import reduce
import operator
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


class TestResBlock:
    """Tests for HiFiGAN ResBlock (dilated conv residual blocks)."""

    @pytest.fixture(scope="class")
    def cpu_device(self):
        from max.driver import CPU
        return CPU()

    @staticmethod
    def _result_to_numpy(result):
        """Extract numpy array from model.execute() result."""
        v = list(result.values())[0] if isinstance(result, dict) else result[0]
        return v.to_numpy() if hasattr(v, "to_numpy") else np.array(v)

    @staticmethod
    def _make_resblock_weights(rng, channels, kernel_size, dilations):
        """Generate random ResBlock weights in PyTorch format."""
        w = {}
        for i in range(len(dilations)):
            w[f"convs1.{i}.weight"] = rng.standard_normal(
                (channels, channels, kernel_size)
            ).astype(np.float32) * 0.01
            w[f"convs1.{i}.bias"] = np.zeros(channels, dtype=np.float32)
            w[f"convs2.{i}.weight"] = rng.standard_normal(
                (channels, channels, kernel_size)
            ).astype(np.float32) * 0.01
            w[f"convs2.{i}.bias"] = np.zeros(channels, dtype=np.float32)
        return w

    def test_resblock_preserves_shape(self, cpu_device):
        """ResBlock with K=3, dilations=[1,3,5] preserves [1,50,1,256]."""
        from max import engine
        from max.graph import Graph, TensorType, DeviceRef, Dim
        from max.dtype import DType
        from models._hifigan_graph import build_resblock

        channels = 256
        kernel_size = 3
        dilations = [1, 3, 5]
        T_in = 50

        rng = np.random.default_rng(42)
        weights = self._make_resblock_weights(rng, channels, kernel_size, dilations)

        T = Dim("T")
        dev = DeviceRef.CPU()
        with Graph(
            "test_resblock",
            input_types=[TensorType(DType.float32, [1, T, 1, channels], dev)],
        ) as g:
            x = g.inputs[0]
            out = build_resblock(x, weights, dilations=dilations, device_ref=dev)
            g.output(out)

        model = engine.InferenceSession(devices=[cpu_device]).load(g)
        x_np = rng.standard_normal((1, T_in, 1, channels)).astype(np.float32) * 0.1
        result = model.execute(x_np)
        out_np = self._result_to_numpy(result)

        assert out_np.shape == (1, T_in, 1, channels), (
            f"Expected (1,{T_in},1,{channels}) got {out_np.shape}"
        )


def _make_full_hifigan_weights(rng, config):
    """Generate all random weights for a full HiFiGAN architecture.

    Returns a dict with keys matching the weight key convention:
        conv_pre.weight, conv_pre.bias
        ups.{i}.weight, ups.{i}.bias
        noise_convs.{i}.weight, noise_convs.{i}.bias
        resblocks.{idx}.convs1.{j}.weight, .bias
        resblocks.{idx}.convs2.{j}.weight, .bias
        conv_post.weight, conv_post.bias
    """
    upsample_rates = config["upsample_rates"]
    uic = config["upsample_initial_channel"]
    inter_channels = config["inter_channels"]
    resblock_kernel_sizes = config["resblock_kernel_sizes"]
    resblock_dilation_sizes = config["resblock_dilation_sizes"]
    upsample_kernel_sizes = config["upsample_kernel_sizes"]

    scale = 0.01
    w = {}

    # conv_pre: Conv1d(inter_channels, uic, K=7)
    # PyTorch format: [C_out, C_in, K]
    w["conv_pre.weight"] = rng.standard_normal(
        (uic, inter_channels, 7)
    ).astype(np.float32) * scale
    w["conv_pre.bias"] = np.zeros(uic, dtype=np.float32)

    ch = uic
    for i, (rate, uk) in enumerate(zip(upsample_rates, upsample_kernel_sizes)):
        ch_next = ch // 2

        # ConvTranspose1d: ups.{i}
        # PyTorch ConvTranspose1d weight: [C_in, C_out, K]
        w[f"ups.{i}.weight"] = rng.standard_normal(
            (ch, ch_next, uk)
        ).astype(np.float32) * scale
        w[f"ups.{i}.bias"] = np.zeros(ch_next, dtype=np.float32)

        # noise_conv: Conv1d(1, ch_next, K=noise_stride or 1)
        noise_stride = reduce(operator.mul, upsample_rates[i + 1:], 1)
        noise_k = max(noise_stride, 1)
        # PyTorch Conv1d: [C_out, C_in, K]
        w[f"noise_convs.{i}.weight"] = rng.standard_normal(
            (ch_next, 1, noise_k)
        ).astype(np.float32) * scale
        w[f"noise_convs.{i}.bias"] = np.zeros(ch_next, dtype=np.float32)

        # ResBlocks: 3 per upsample block (one per kernel size)
        for k_idx, (rk, rd) in enumerate(
            zip(resblock_kernel_sizes, resblock_dilation_sizes)
        ):
            rb_idx = i * len(resblock_kernel_sizes) + k_idx
            for j in range(len(rd)):
                # convs1.{j}: Conv1d(ch_next, ch_next, rk)
                w[f"resblocks.{rb_idx}.convs1.{j}.weight"] = rng.standard_normal(
                    (ch_next, ch_next, rk)
                ).astype(np.float32) * scale
                w[f"resblocks.{rb_idx}.convs1.{j}.bias"] = np.zeros(
                    ch_next, dtype=np.float32
                )
                # convs2.{j}: Conv1d(ch_next, ch_next, rk)
                w[f"resblocks.{rb_idx}.convs2.{j}.weight"] = rng.standard_normal(
                    (ch_next, ch_next, rk)
                ).astype(np.float32) * scale
                w[f"resblocks.{rb_idx}.convs2.{j}.bias"] = np.zeros(
                    ch_next, dtype=np.float32
                )

        ch = ch_next

    # conv_post: Conv1d(ch, 1, K=7)
    w["conv_post.weight"] = rng.standard_normal(
        (1, ch, 7)
    ).astype(np.float32) * scale
    w["conv_post.bias"] = np.zeros(1, dtype=np.float32)

    return w


class TestHiFiGANGraph:
    """Tests for the full HiFiGAN MAX graph."""

    CONFIG_48K = {
        "inter_channels": 192,
        "upsample_rates": [12, 10, 2, 2],
        "upsample_initial_channel": 512,
        "upsample_kernel_sizes": [24, 20, 4, 4],
        "resblock_kernel_sizes": [3, 7, 11],
        "resblock_dilation_sizes": [[1, 3, 5], [1, 3, 5], [1, 3, 5]],
        "sample_rate": 48000,
        "hop_length": 480,
    }

    @pytest.fixture(scope="class")
    def cpu_device(self):
        from max.driver import CPU
        return CPU()

    @pytest.fixture(scope="class")
    def model_48k(self, cpu_device):
        """Build full HiFiGAN graph with random weights, 48kHz. Compiled once."""
        from max import engine
        from models._hifigan_graph import build_hifigan_graph

        rng = np.random.default_rng(42)
        weights = _make_full_hifigan_weights(rng, self.CONFIG_48K)
        graph = build_hifigan_graph(weights, self.CONFIG_48K, device="cpu", batch_size=1)
        model = engine.InferenceSession(devices=[cpu_device]).load(graph)
        return model

    @staticmethod
    def _result_to_numpy(result):
        v = list(result.values())[0] if isinstance(result, dict) else result[0]
        return v.to_numpy() if hasattr(v, "to_numpy") else np.array(v)

    def test_output_shape_48k(self, model_48k):
        """latents [1, 10, 1, 192] + excitation [1, 4800, 1, 1] -> output T=4800."""
        rng = np.random.default_rng(99)
        T_latent = 10
        T_excitation = T_latent * self.CONFIG_48K["hop_length"]  # 4800

        latents = rng.standard_normal(
            (1, T_latent, 1, self.CONFIG_48K["inter_channels"])
        ).astype(np.float32) * 0.1
        excitation = rng.standard_normal(
            (1, T_excitation, 1, 1)
        ).astype(np.float32) * 0.1

        result = model_48k.execute(latents, excitation)
        out_np = self._result_to_numpy(result)

        assert out_np.shape == (1, T_excitation, 1, 1), (
            f"Expected (1, {T_excitation}, 1, 1) got {out_np.shape}"
        )

    def test_output_not_nan(self, model_48k):
        """Random input with small magnitudes produces no NaN/Inf."""
        rng = np.random.default_rng(123)
        T_latent = 10
        T_excitation = T_latent * self.CONFIG_48K["hop_length"]

        latents = rng.standard_normal(
            (1, T_latent, 1, self.CONFIG_48K["inter_channels"])
        ).astype(np.float32) * 0.1
        excitation = rng.standard_normal(
            (1, T_excitation, 1, 1)
        ).astype(np.float32) * 0.1

        result = model_48k.execute(latents, excitation)
        out_np = self._result_to_numpy(result)

        assert not np.any(np.isnan(out_np)), "Output contains NaN values"
        assert not np.any(np.isinf(out_np)), "Output contains Inf values"


def _make_sr_config(sr, upsample_rates, upsample_kernel_sizes):
    """Build a full HiFiGAN config dict for a given sample rate."""
    return {
        "inter_channels": 192,
        "upsample_rates": upsample_rates,
        "upsample_initial_channel": 512,
        "upsample_kernel_sizes": upsample_kernel_sizes,
        "resblock_kernel_sizes": [3, 7, 11],
        "resblock_dilation_sizes": [[1, 3, 5], [1, 3, 5], [1, 3, 5]],
        "sr": sr,
        "hop_length": reduce(operator.mul, upsample_rates, 1),
    }


# Sample-rate configs: (upsample_rates, upsample_kernel_sizes)
SR_CONFIGS = {
    32000: ([10, 8, 2, 2], [20, 16, 4, 4]),
    40000: ([10, 10, 2, 2], [16, 16, 4, 4]),
    48000: ([12, 10, 2, 2], [24, 20, 4, 4]),
}


class TestNSFHiFiGAN:
    """Tests for the public NSFHiFiGAN class (synthesize pipeline)."""

    @pytest.fixture(scope="class")
    def vocoder_48k(self):
        from models.hifigan import NSFHiFiGAN
        rng = np.random.default_rng(42)
        config = _make_sr_config(48000, *SR_CONFIGS[48000])
        weights = _make_full_hifigan_weights(rng, config)
        return NSFHiFiGAN._from_weights(weights, config, device="cpu", batch_size=1)

    @pytest.fixture(scope="class")
    def vocoder_40k(self):
        from models.hifigan import NSFHiFiGAN
        rng = np.random.default_rng(42)
        config = _make_sr_config(40000, *SR_CONFIGS[40000])
        weights = _make_full_hifigan_weights(rng, config)
        return NSFHiFiGAN._from_weights(weights, config, device="cpu", batch_size=1)

    @pytest.fixture(scope="class")
    def vocoder_32k(self):
        from models.hifigan import NSFHiFiGAN
        rng = np.random.default_rng(42)
        config = _make_sr_config(32000, *SR_CONFIGS[32000])
        weights = _make_full_hifigan_weights(rng, config)
        return NSFHiFiGAN._from_weights(weights, config, device="cpu", batch_size=1)

    @pytest.fixture(scope="class")
    def vocoder_48k_batch2(self):
        from models.hifigan import NSFHiFiGAN
        rng = np.random.default_rng(42)
        config = _make_sr_config(48000, *SR_CONFIGS[48000])
        weights = _make_full_hifigan_weights(rng, config)
        return NSFHiFiGAN._from_weights(weights, config, device="cpu", batch_size=2)

    def test_synthesize_shape_48k(self, vocoder_48k):
        """latents [1, 192, 10] + f0 [1, 10] -> output [1, 4800]."""
        rng = np.random.default_rng(99)
        T = 10
        latents = rng.standard_normal((1, 192, T)).astype(np.float32) * 0.1
        f0 = rng.uniform(100, 400, (1, T)).astype(np.float32)

        audio = vocoder_48k.synthesize(latents, f0)
        assert audio.shape == (1, 4800), f"Expected (1, 4800) got {audio.shape}"

    def test_synthesize_shape_40k(self, vocoder_40k):
        """latents [1, 192, 10] + f0 [1, 10] -> output [1, 4000]."""
        rng = np.random.default_rng(99)
        T = 10
        latents = rng.standard_normal((1, 192, T)).astype(np.float32) * 0.1
        f0 = rng.uniform(100, 400, (1, T)).astype(np.float32)

        audio = vocoder_40k.synthesize(latents, f0)
        assert audio.shape == (1, 4000), f"Expected (1, 4000) got {audio.shape}"

    def test_synthesize_shape_32k(self, vocoder_32k):
        """latents [1, 192, 10] + f0 [1, 10] -> output [1, 3200]."""
        rng = np.random.default_rng(99)
        T = 10
        latents = rng.standard_normal((1, 192, T)).astype(np.float32) * 0.1
        f0 = rng.uniform(100, 400, (1, T)).astype(np.float32)

        audio = vocoder_32k.synthesize(latents, f0)
        assert audio.shape == (1, 3200), f"Expected (1, 3200) got {audio.shape}"

    def test_synthesize_not_nan(self, vocoder_48k):
        """Random input with small magnitudes produces no NaN/Inf."""
        rng = np.random.default_rng(123)
        T = 10
        latents = rng.standard_normal((1, 192, T)).astype(np.float32) * 0.1
        f0 = rng.uniform(100, 400, (1, T)).astype(np.float32)

        audio = vocoder_48k.synthesize(latents, f0)
        assert not np.any(np.isnan(audio)), "Output contains NaN values"
        assert not np.any(np.isinf(audio)), "Output contains Inf values"

    def test_synthesize_unvoiced_not_nan(self, vocoder_48k):
        """f0 = zeros (fully unvoiced) produces no NaN/Inf."""
        rng = np.random.default_rng(456)
        T = 10
        latents = rng.standard_normal((1, 192, T)).astype(np.float32) * 0.1
        f0 = np.zeros((1, T), dtype=np.float32)

        audio = vocoder_48k.synthesize(latents, f0)
        assert not np.any(np.isnan(audio)), "Output contains NaN values"
        assert not np.any(np.isinf(audio)), "Output contains Inf values"

    @pytest.mark.xfail(
        reason="conv_transpose_1d uses squeeze(axis=0) which requires B=1; batch>1 needs graph builder update",
        raises=ValueError,
    )
    def test_synthesize_batch2_shape(self, vocoder_48k_batch2):
        """batch_size=2: latents [2, 192, 10] + f0 [2, 10] -> output [2, 4800]."""
        rng = np.random.default_rng(789)
        T = 10
        latents = rng.standard_normal((2, 192, T)).astype(np.float32) * 0.1
        f0 = rng.uniform(100, 400, (2, T)).astype(np.float32)

        audio = vocoder_48k_batch2.synthesize(latents, f0)
        assert audio.shape == (2, 4800), f"Expected (2, 4800) got {audio.shape}"
