"""Tests for VITS weight loader — enc_p, flow, and speaker embedding extraction.

Covers:
  - enc_p.* key extraction (prefix stripped)
  - flow.* key extraction (prefix stripped, Flip layers skipped)
  - emb_g.weight extraction -> baked [256, 1] speaker embedding
  - weight-norm reconstruction (.weight_g/.weight_v -> plain .weight)
  - modern parametrizations format (.parametrizations.weight.original0/1)
  - All values are float32
  - bake_hifigan_cond folds cond(g) into conv_pre.bias in-place
  - Real checkpoint loading (skipped if file absent)
"""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import numpy as np
import pytest


CHECKPOINT_PATH = (
    "/home/maskkiller/Downloads/voice files/extracted/"
    "theweeknd biggest data set/theweekv1.pth"
)


class TestVITSWeightLoader:
    """Unit tests for _vits_weight_loader — no checkpoint download required."""

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _rng():
        return np.random.default_rng(42)

    @staticmethod
    def _to_tensor(arr):
        """Wrap a numpy array as a minimal tensor-like object (has .numpy())."""
        import types
        t = types.SimpleNamespace()
        t.numpy = lambda: arr
        t.shape = arr.shape
        return t

    def _make_fake_sd(self, rng, use_parametrizations: bool = False):
        """Build a minimal fake state_dict with enc_p, flow, emb_g keys.

        If use_parametrizations=True, uses the modern
        .parametrizations.weight.original0 (weight_g) / .original1 (weight_v)
        format instead of the old .weight_g / .weight_v format.
        """
        sd = {}

        # --- enc_p keys (no weight-norm in this checkpoint format) ---
        sd["enc_p.emb_phone.weight"] = rng.standard_normal((192, 768)).astype(np.float32)
        sd["enc_p.emb_phone.bias"] = rng.standard_normal(192).astype(np.float32)
        sd["enc_p.emb_pitch.weight"] = rng.standard_normal((256, 192)).astype(np.float32)
        sd["enc_p.proj.weight"] = rng.standard_normal((384, 192, 1)).astype(np.float32)
        sd["enc_p.proj.bias"] = rng.standard_normal(384).astype(np.float32)

        # --- flow keys: coupling layers at 0 and 2, Flip at 1 and 3 ---
        # Coupling layer 0
        sd["flow.flows.0.pre.weight"] = rng.standard_normal((192, 96, 1)).astype(np.float32)
        sd["flow.flows.0.pre.bias"] = rng.standard_normal(192).astype(np.float32)
        sd["flow.flows.0.post.weight"] = rng.standard_normal((96, 192, 1)).astype(np.float32)
        sd["flow.flows.0.post.bias"] = rng.standard_normal(96).astype(np.float32)

        # Weight-norm layer inside flow.flows.0.enc
        wv = rng.standard_normal((384, 192, 5)).astype(np.float32)
        wg = np.abs(rng.standard_normal((384, 1, 1)).astype(np.float32)) + 0.1
        if use_parametrizations:
            sd["flow.flows.0.enc.in_layers.0.parametrizations.weight.original0"] = wg
            sd["flow.flows.0.enc.in_layers.0.parametrizations.weight.original1"] = wv
        else:
            sd["flow.flows.0.enc.in_layers.0.weight_g"] = wg
            sd["flow.flows.0.enc.in_layers.0.weight_v"] = wv
        sd["flow.flows.0.enc.in_layers.0.bias"] = rng.standard_normal(384).astype(np.float32)
        # Store for later verification
        self._wv = wv
        self._wg = wg

        # Coupling layer 2
        sd["flow.flows.2.pre.weight"] = rng.standard_normal((192, 96, 1)).astype(np.float32)
        sd["flow.flows.2.pre.bias"] = rng.standard_normal(192).astype(np.float32)
        sd["flow.flows.2.post.weight"] = rng.standard_normal((96, 192, 1)).astype(np.float32)
        sd["flow.flows.2.post.bias"] = rng.standard_normal(96).astype(np.float32)

        # Flip layers (indices 1 and 3) — no parameters (should be skipped)
        # They don't produce any keys in the checkpoint.

        # --- emb_g ---
        # 5 speakers, embedding dim 256
        sd["emb_g.weight"] = rng.standard_normal((5, 256)).astype(np.float32)

        # --- dec.cond (for bake test) ---
        sd["dec.cond.weight"] = rng.standard_normal((512, 256, 1)).astype(np.float32)
        sd["dec.cond.bias"] = rng.standard_normal(512).astype(np.float32)

        # --- unrelated key (should be ignored) ---
        sd["some_other.weight"] = rng.standard_normal((10, 10)).astype(np.float32)

        return sd

    # ------------------------------------------------------------------
    # Test 1: enc_p key extraction
    # ------------------------------------------------------------------

    def test_extract_enc_p_keys_present(self):
        """extract_vits_weights returns enc_p.* keys with prefix stripped."""
        from models._vits_weight_loader import extract_vits_weights

        rng = self._rng()
        sd = self._make_fake_sd(rng)
        result = extract_vits_weights(sd)

        assert "emb_phone.weight" in result
        assert "emb_phone.bias" in result
        assert "emb_pitch.weight" in result
        assert "proj.weight" in result
        assert "proj.bias" in result

    def test_extract_enc_p_shapes(self):
        """Extracted enc_p weights have correct shapes."""
        from models._vits_weight_loader import extract_vits_weights

        rng = self._rng()
        sd = self._make_fake_sd(rng)
        result = extract_vits_weights(sd)

        assert result["emb_phone.weight"].shape == (192, 768)
        assert result["emb_phone.bias"].shape == (192,)
        assert result["emb_pitch.weight"].shape == (256, 192)
        assert result["proj.weight"].shape == (384, 192, 1)
        assert result["proj.bias"].shape == (384,)

    def test_extract_enc_p_excludes_other_prefixes(self):
        """extract_vits_weights does NOT include flow.* or emb_g.* keys."""
        from models._vits_weight_loader import extract_vits_weights

        rng = self._rng()
        sd = self._make_fake_sd(rng)
        result = extract_vits_weights(sd)

        for k in result:
            assert not k.startswith("flow."), f"Unexpected flow key: {k}"
            assert not k.startswith("emb_g."), f"Unexpected emb_g key: {k}"
            assert not k.startswith("dec."), f"Unexpected dec key: {k}"
            assert not k.startswith("some_other."), f"Unexpected key: {k}"

    # ------------------------------------------------------------------
    # Test 2: flow key extraction
    # ------------------------------------------------------------------

    def test_extract_flow_keys_present(self):
        """extract_vits_weights returns flow.* keys (stripped) for coupling layers."""
        from models._vits_weight_loader import extract_vits_weights

        rng = self._rng()
        sd = self._make_fake_sd(rng)
        result = extract_vits_weights(sd)

        assert "flows.0.pre.weight" in result
        assert "flows.0.pre.bias" in result
        assert "flows.0.post.weight" in result
        assert "flows.0.post.bias" in result
        assert "flows.0.enc.in_layers.0.weight" in result  # weight-norm reconstructed
        assert "flows.0.enc.in_layers.0.bias" in result
        assert "flows.2.pre.weight" in result
        assert "flows.2.pre.bias" in result

    def test_extract_flow_raw_wn_keys_absent(self):
        """weight_g / weight_v raw keys should NOT appear in the output."""
        from models._vits_weight_loader import extract_vits_weights

        rng = self._rng()
        sd = self._make_fake_sd(rng)
        result = extract_vits_weights(sd)

        for k in result:
            assert not k.endswith(".weight_g"), f"Raw weight_g key leaked: {k}"
            assert not k.endswith(".weight_v"), f"Raw weight_v key leaked: {k}"
            assert "parametrizations" not in k, f"Parametrization key leaked: {k}"

    # ------------------------------------------------------------------
    # Test 3: emb_g / speaker embedding
    # ------------------------------------------------------------------

    def test_extract_speaker_embedding_shape(self):
        """extract_speaker_embedding returns [256, 1] float32 for sid=0."""
        from models._vits_weight_loader import extract_speaker_embedding

        rng = self._rng()
        sd = self._make_fake_sd(rng)
        g = extract_speaker_embedding(sd, sid=0)

        assert g.shape == (256, 1), f"Expected (256, 1), got {g.shape}"
        assert g.dtype == np.float32

    def test_extract_speaker_embedding_correct_row(self):
        """extract_speaker_embedding returns the correct row from emb_g.weight."""
        from models._vits_weight_loader import extract_speaker_embedding

        rng = self._rng()
        sd = self._make_fake_sd(rng)
        emb_weight = np.asarray(sd["emb_g.weight"], dtype=np.float32)

        for sid in [0, 1, 4]:
            g = extract_speaker_embedding(sd, sid=sid)
            expected = emb_weight[sid].reshape(256, 1)
            np.testing.assert_array_equal(g, expected, err_msg=f"sid={sid}")

    # ------------------------------------------------------------------
    # Test 4: weight-norm reconstruction (old .weight_g / .weight_v format)
    # ------------------------------------------------------------------

    def test_weight_norm_reconstructed_correctly(self):
        """flow weight-norm pairs are reconstructed to plain .weight."""
        from models._vits_weight_loader import extract_vits_weights

        rng = self._rng()
        sd = self._make_fake_sd(rng, use_parametrizations=False)
        result = extract_vits_weights(sd)

        assert "flows.0.enc.in_layers.0.weight" in result

        wv = self._wv
        wg = self._wg
        norm = np.linalg.norm(wv.reshape(384, -1), axis=1).reshape(384, 1, 1)
        expected = (wv * (wg / norm)).astype(np.float32)
        np.testing.assert_allclose(
            result["flows.0.enc.in_layers.0.weight"], expected, rtol=1e-5
        )

    # ------------------------------------------------------------------
    # Test 4b: weight-norm reconstruction (modern parametrizations format)
    # ------------------------------------------------------------------

    def test_parametrizations_weight_norm_reconstructed(self):
        """Modern .parametrizations.weight.original0/1 format is handled."""
        from models._vits_weight_loader import extract_vits_weights

        rng = self._rng()
        sd = self._make_fake_sd(rng, use_parametrizations=True)
        result = extract_vits_weights(sd)

        assert "flows.0.enc.in_layers.0.weight" in result

        wv = self._wv
        wg = self._wg
        norm = np.linalg.norm(wv.reshape(384, -1), axis=1).reshape(384, 1, 1)
        expected = (wv * (wg / norm)).astype(np.float32)
        np.testing.assert_allclose(
            result["flows.0.enc.in_layers.0.weight"], expected, rtol=1e-5
        )

    # ------------------------------------------------------------------
    # Test 5: All values are float32
    # ------------------------------------------------------------------

    def test_all_values_float32(self):
        """Every value in extract_vits_weights output is float32."""
        from models._vits_weight_loader import extract_vits_weights

        rng = self._rng()
        sd = self._make_fake_sd(rng)
        result = extract_vits_weights(sd)

        for k, v in result.items():
            assert isinstance(v, np.ndarray), f"{k} is not ndarray"
            assert v.dtype == np.float32, f"{k} has dtype {v.dtype}, expected float32"

    # ------------------------------------------------------------------
    # Test 6: bake_hifigan_cond
    # ------------------------------------------------------------------

    def test_bake_hifigan_cond_modifies_bias(self):
        """bake_hifigan_cond folds cond(g) into conv_pre.bias in-place."""
        from models._vits_weight_loader import bake_hifigan_cond, extract_speaker_embedding

        rng = self._rng()
        sd = self._make_fake_sd(rng)

        # Build a fake hifigan weights dict with conv_pre.bias
        hifigan_weights = {
            "conv_pre.weight": rng.standard_normal((512, 192, 7)).astype(np.float32),
            "conv_pre.bias": np.zeros(512, dtype=np.float32),
        }

        g = extract_speaker_embedding(sd, sid=0)
        cond_weight = np.asarray(sd["dec.cond.weight"], dtype=np.float32)  # [512, 256, 1]
        cond_bias = np.asarray(sd["dec.cond.bias"], dtype=np.float32)      # [512]

        original_bias = hifigan_weights["conv_pre.bias"].copy()

        bake_hifigan_cond(hifigan_weights, g, cond_weight, cond_bias)

        # cond_out = weight.squeeze(-1) @ g + bias[:, None] = cond_weight[:, :, 0] @ g + cond_bias
        cond_out = (cond_weight[:, :, 0] @ g).squeeze(-1) + cond_bias  # [512]
        expected_bias = original_bias + cond_out

        np.testing.assert_allclose(
            hifigan_weights["conv_pre.bias"], expected_bias, rtol=1e-5
        )

    def test_bake_hifigan_cond_modifies_inplace(self):
        """bake_hifigan_cond mutates the hifigan_weights dict in-place."""
        from models._vits_weight_loader import bake_hifigan_cond, extract_speaker_embedding

        rng = self._rng()
        sd = self._make_fake_sd(rng)

        hifigan_weights = {
            "conv_pre.bias": np.zeros(512, dtype=np.float32),
        }

        g = extract_speaker_embedding(sd, sid=0)
        cond_weight = np.asarray(sd["dec.cond.weight"], dtype=np.float32)
        cond_bias = np.asarray(sd["dec.cond.bias"], dtype=np.float32)

        bias_before = hifigan_weights["conv_pre.bias"]
        bake_hifigan_cond(hifigan_weights, g, cond_weight, cond_bias)

        # The array object itself should be the same (in-place modification)
        assert hifigan_weights["conv_pre.bias"] is bias_before


class TestVITSWeightLoaderRealCheckpoint:
    """Integration tests that load the real TheWeeknd checkpoint.

    Skipped automatically if the .pth file is not present.
    """

    @pytest.fixture(scope="class")
    def checkpoint_weights(self):
        """Load the real checkpoint once for the class."""
        import torch
        ckpt = torch.load(CHECKPOINT_PATH, map_location="cpu", weights_only=False)
        sd = ckpt["weight"]
        return sd, ckpt

    @pytest.mark.skipif(
        not os.path.exists(CHECKPOINT_PATH),
        reason=f"Checkpoint not found: {CHECKPOINT_PATH}",
    )
    def test_enc_p_key_shapes(self, checkpoint_weights):
        """enc_p weights from real checkpoint have expected shapes."""
        from models._vits_weight_loader import extract_vits_weights

        sd, _ = checkpoint_weights
        result = extract_vits_weights(sd)

        assert result["emb_phone.weight"].shape == (192, 768), (
            f"emb_phone.weight shape: {result['emb_phone.weight'].shape}"
        )
        assert result["emb_phone.bias"].shape == (192,)
        assert result["emb_pitch.weight"].shape == (256, 192)
        assert result["proj.weight"].shape == (384, 192, 1)
        assert result["proj.bias"].shape == (384,)

        # Check attn_layers conv weights (should be [192, 192, 1])
        for i in range(6):
            for conv in ["conv_q", "conv_k", "conv_v", "conv_o"]:
                key = f"encoder.attn_layers.{i}.{conv}.weight"
                assert key in result, f"Missing key: {key}"
                assert result[key].shape == (192, 192, 1), (
                    f"{key} shape: {result[key].shape}"
                )

        # ffn_layers
        for i in range(6):
            key1 = f"encoder.ffn_layers.{i}.conv_1.weight"
            key2 = f"encoder.ffn_layers.{i}.conv_2.weight"
            assert result[key1].shape == (768, 192, 3), f"{key1}: {result[key1].shape}"
            assert result[key2].shape == (192, 768, 3), f"{key2}: {result[key2].shape}"

    @pytest.mark.skipif(
        not os.path.exists(CHECKPOINT_PATH),
        reason=f"Checkpoint not found: {CHECKPOINT_PATH}",
    )
    def test_flow_has_4_coupling_layers(self, checkpoint_weights):
        """flow weights contain exactly 4 coupling layers at indices 0, 2, 4, 6."""
        from models._vits_weight_loader import extract_vits_weights

        sd, _ = checkpoint_weights
        result = extract_vits_weights(sd)

        coupling_indices = [0, 2, 4, 6]
        for f in coupling_indices:
            assert f"flows.{f}.pre.weight" in result, (
                f"Missing flows.{f}.pre.weight"
            )
            assert f"flows.{f}.post.weight" in result, (
                f"Missing flows.{f}.post.weight"
            )
            assert f"flows.{f}.pre.weight" in result
            assert result[f"flows.{f}.pre.weight"].shape == (192, 96, 1), (
                f"flows.{f}.pre.weight shape: {result[f'flows.{f}.pre.weight'].shape}"
            )
            assert result[f"flows.{f}.post.weight"].shape == (96, 192, 1)

        # Flip layers (1, 3, 5, 7) should have no keys
        flip_indices = [1, 3, 5, 7]
        for f in flip_indices:
            matching = [k for k in result if k.startswith(f"flows.{f}.")]
            assert len(matching) == 0, (
                f"Flip layer {f} should have no keys, found: {matching}"
            )

    @pytest.mark.skipif(
        not os.path.exists(CHECKPOINT_PATH),
        reason=f"Checkpoint not found: {CHECKPOINT_PATH}",
    )
    def test_flow_weight_norm_reconstructed(self, checkpoint_weights):
        """flow weight-norm pairs are reconstructed (no raw _g/_v keys)."""
        from models._vits_weight_loader import extract_vits_weights

        sd, _ = checkpoint_weights
        result = extract_vits_weights(sd)

        for k in result:
            assert not k.endswith(".weight_g"), f"Raw weight_g leaked: {k}"
            assert not k.endswith(".weight_v"), f"Raw weight_v leaked: {k}"

        # in_layers weights should exist and have the right shape
        for f in [0, 2, 4, 6]:
            for l in range(3):
                key = f"flows.{f}.enc.in_layers.{l}.weight"
                assert key in result, f"Missing: {key}"
                assert result[key].shape == (384, 192, 5), (
                    f"{key} shape: {result[key].shape}"
                )

        # cond_layer
        for f in [0, 2, 4, 6]:
            key = f"flows.{f}.enc.cond_layer.weight"
            assert key in result, f"Missing: {key}"
            assert result[key].shape == (1152, 256, 1), (
                f"{key} shape: {result[key].shape}"
            )

    @pytest.mark.skipif(
        not os.path.exists(CHECKPOINT_PATH),
        reason=f"Checkpoint not found: {CHECKPOINT_PATH}",
    )
    def test_speaker_embedding_shape(self, checkpoint_weights):
        """Speaker embedding g has shape [256, 1]."""
        from models._vits_weight_loader import extract_speaker_embedding

        sd, _ = checkpoint_weights
        g = extract_speaker_embedding(sd, sid=0)

        assert g.shape == (256, 1), f"Expected (256, 1), got {g.shape}"
        assert g.dtype == np.float32

    @pytest.mark.skipif(
        not os.path.exists(CHECKPOINT_PATH),
        reason=f"Checkpoint not found: {CHECKPOINT_PATH}",
    )
    def test_bake_hifigan_cond_modifies_conv_pre_bias(self, checkpoint_weights):
        """Baked cond(g) is non-zero and modifies conv_pre.bias correctly."""
        from models._vits_weight_loader import (
            extract_vits_weights,
            extract_speaker_embedding,
            bake_hifigan_cond,
        )
        from models._hifigan_weight_loader import extract_hifigan_weights

        sd, _ = checkpoint_weights
        hifigan_weights = extract_hifigan_weights(sd)
        g = extract_speaker_embedding(sd, sid=0)

        # Extract cond weights from the state dict
        cond_weight = np.asarray(sd["dec.cond.weight"].numpy(), dtype=np.float32)  # [512, 256, 1]
        cond_bias = np.asarray(sd["dec.cond.bias"].numpy(), dtype=np.float32)    # [512]

        original_bias = hifigan_weights["conv_pre.bias"].copy()
        bake_hifigan_cond(hifigan_weights, g, cond_weight, cond_bias)

        # The bias should have changed
        assert not np.allclose(hifigan_weights["conv_pre.bias"], original_bias), (
            "conv_pre.bias was not modified by bake_hifigan_cond"
        )

        # Verify correctness: cond_out = cond_weight[:, :, 0] @ g + cond_bias
        cond_out = (cond_weight[:, :, 0] @ g).squeeze(-1) + cond_bias
        expected = original_bias + cond_out
        np.testing.assert_allclose(
            hifigan_weights["conv_pre.bias"], expected, rtol=1e-5
        )

    @pytest.mark.skipif(
        not os.path.exists(CHECKPOINT_PATH),
        reason=f"Checkpoint not found: {CHECKPOINT_PATH}",
    )
    def test_load_vits_weights_pipeline(self, checkpoint_weights):
        """load_vits_weights returns (vits_weights, hifigan_weights, config)."""
        from models._vits_weight_loader import load_vits_weights

        vits_weights, hifigan_weights, config = load_vits_weights(CHECKPOINT_PATH)

        # VITS weights contain enc_p and flow
        assert "emb_phone.weight" in vits_weights
        assert "flows.0.pre.weight" in vits_weights

        # HiFiGAN weights should be present
        assert "conv_pre.weight" in hifigan_weights

        # Config should have expected keys
        assert "sr" in config
        assert "upsample_rates" in config
        assert "upsample_initial_channel" in config


# ======================================================================
# Flow graph tests (Task 3)
# ======================================================================


def _make_wavenet_weights(rng, hidden=192, kernel_size=5, n_layers=3, gin_channels=256):
    """Generate random WaveNet weights matching RVC v2 config.

    Returns dict with keys: cond_layer.weight, cond_layer.bias,
    in_layers.{i}.weight/bias, res_skip_layers.{i}.weight/bias.
    """
    w = {}
    # cond_layer: Conv1d(256, 2*192*3=1152, k=1)
    w["cond_layer.weight"] = rng.standard_normal(
        (2 * hidden * n_layers, gin_channels, 1)
    ).astype(np.float32) * 0.01
    w["cond_layer.bias"] = np.zeros(2 * hidden * n_layers, dtype=np.float32)

    for i in range(n_layers):
        # in_layers[i]: Conv1d(192, 384, k=5, dilation=dilation_rate**i)
        w[f"in_layers.{i}.weight"] = rng.standard_normal(
            (2 * hidden, hidden, kernel_size)
        ).astype(np.float32) * 0.01
        w[f"in_layers.{i}.bias"] = np.zeros(2 * hidden, dtype=np.float32)

        # res_skip_layers[i]: Conv1d(192, 384, k=1) for i<n-1, Conv1d(192, 192, k=1) for i==n-1
        if i < n_layers - 1:
            out_ch = 2 * hidden
        else:
            out_ch = hidden
        w[f"res_skip_layers.{i}.weight"] = rng.standard_normal(
            (out_ch, hidden, 1)
        ).astype(np.float32) * 0.01
        w[f"res_skip_layers.{i}.bias"] = np.zeros(out_ch, dtype=np.float32)

    return w


def _make_coupling_layer_weights(rng, hidden=192, kernel_size=5, n_layers=3, gin_channels=256):
    """Generate random ResidualCouplingLayer weights.

    Returns dict with keys: pre.weight, pre.bias, post.weight, post.bias,
    enc.cond_layer.weight, enc.in_layers.{i}.weight, etc.
    """
    half_ch = hidden // 2  # 96
    w = {}

    # pre: Conv1d(96, 192, k=1)
    w["pre.weight"] = rng.standard_normal((hidden, half_ch, 1)).astype(np.float32) * 0.01
    w["pre.bias"] = np.zeros(hidden, dtype=np.float32)

    # post: Conv1d(192, 96, k=1) — initialized to zero in PyTorch
    w["post.weight"] = np.zeros((half_ch, hidden, 1), dtype=np.float32)
    w["post.bias"] = np.zeros(half_ch, dtype=np.float32)

    # enc (WaveNet) weights
    enc_w = _make_wavenet_weights(rng, hidden, kernel_size, n_layers, gin_channels)
    for k, v in enc_w.items():
        w[f"enc.{k}"] = v

    return w


def _make_full_flow_weights(rng, n_flows=4, hidden=192, kernel_size=5, n_layers=3, gin_channels=256):
    """Generate random weights for the full flow (4 coupling layers).

    Returns dict with keys like "flows.0.pre.weight", etc.
    Coupling layers at indices 0, 2, 4, 6.
    """
    w = {}
    for f in range(n_flows):
        flow_idx = f * 2  # 0, 2, 4, 6
        coupling_w = _make_coupling_layer_weights(rng, hidden, kernel_size, n_layers, gin_channels)
        for k, v in coupling_w.items():
            w[f"flows.{flow_idx}.{k}"] = v
    return w


class TestFlowGraph:
    """Tests for the full flow graph (WaveNet + coupling layers + flip).

    Compiles ONE full flow graph and tests all properties from it.
    This avoids multiple graph compilations which exhaust memory.
    """

    @pytest.fixture(scope="class")
    def cpu_device(self):
        from max.driver import CPU
        return CPU()

    @pytest.fixture(scope="class")
    def flow_model(self, cpu_device):
        """Build and compile ONE full flow graph with random weights. Shared by all tests."""
        from max import engine
        from models._vits_graph import build_flow_graph

        rng = np.random.default_rng(42)
        hidden = 192

        # Use non-zero post weights so coupling actually transforms
        weights = _make_full_flow_weights(rng, n_flows=4, hidden=hidden)
        for f in range(4):
            flow_idx = f * 2
            weights[f"flows.{flow_idx}.post.weight"] = rng.standard_normal(
                (hidden // 2, hidden, 1)
            ).astype(np.float32) * 0.01
            weights[f"flows.{flow_idx}.post.bias"] = rng.standard_normal(
                hidden // 2
            ).astype(np.float32) * 0.01

        g_np = rng.standard_normal((256, 1)).astype(np.float32) * 0.1

        config = {
            "inter_channels": hidden,
            "hidden_channels": hidden,
            "n_layers": 3,
            "dilation_rate": 1,
            "n_flows": 4,
        }

        graph = build_flow_graph(weights, g_np, config, device="cpu", batch_size=1)
        model = engine.InferenceSession(devices=[cpu_device]).load(graph)
        return model

    @staticmethod
    def _result_to_numpy(result):
        v = list(result.values())[0] if isinstance(result, dict) else result[0]
        return v.to_numpy() if hasattr(v, "to_numpy") else np.array(v)

    def test_flow_output_shape(self, flow_model):
        """Full flow reverse: z_p [1, 192, T] + mask -> z [1, 192, T]."""
        rng = np.random.default_rng(99)
        hidden = 192
        T_val = 20

        z_p = rng.standard_normal((1, hidden, T_val)).astype(np.float32) * 0.1
        mask = np.ones((1, 1, T_val), dtype=np.float32)

        result = flow_model.execute(z_p, mask)
        out_np = self._result_to_numpy(result)

        assert out_np.shape == (1, hidden, T_val), (
            f"Expected (1, {hidden}, {T_val}), got {out_np.shape}"
        )

    def test_flow_output_not_nan(self, flow_model):
        """Flow output contains no NaN or Inf."""
        rng = np.random.default_rng(123)
        hidden = 192
        T_val = 20

        z_p = rng.standard_normal((1, hidden, T_val)).astype(np.float32) * 0.1
        mask = np.ones((1, 1, T_val), dtype=np.float32)

        result = flow_model.execute(z_p, mask)
        out_np = self._result_to_numpy(result)

        assert not np.any(np.isnan(out_np)), "Flow output contains NaN"
        assert not np.any(np.isinf(out_np)), "Flow output contains Inf"

    def test_flow_output_differs_from_input(self, flow_model):
        """Full flow reverse output is different from input z_p."""
        rng = np.random.default_rng(42)
        hidden = 192
        T_val = 20

        z_p = rng.standard_normal((1, hidden, T_val)).astype(np.float32) * 0.1
        mask = np.ones((1, 1, T_val), dtype=np.float32)

        result = flow_model.execute(z_p, mask)
        out_np = self._result_to_numpy(result)

        assert not np.allclose(out_np, z_p, atol=1e-6), (
            "Flow output should differ from input"
        )

    def test_flow_respects_mask(self, flow_model):
        """Flow output is zero where mask is zero."""
        rng = np.random.default_rng(456)
        hidden = 192
        T_val = 20

        z_p = rng.standard_normal((1, hidden, T_val)).astype(np.float32) * 0.1
        mask = np.ones((1, 1, T_val), dtype=np.float32)
        mask[:, :, 15:] = 0.0  # last 5 timesteps masked

        result = flow_model.execute(z_p, mask)
        out_np = self._result_to_numpy(result)

        assert np.allclose(out_np[:, :, 15:], 0.0, atol=1e-6), (
            "Flow output should be zero where mask is zero"
        )


class TestFlowNumericalValidation:
    """Numerical validation against PyTorch reference.

    Compiles ONE flow graph with real checkpoint weights.
    Skipped if checkpoint file is not present.
    """

    @pytest.fixture(scope="class")
    def cpu_device(self):
        from max.driver import CPU
        return CPU()

    @staticmethod
    def _result_to_numpy(result):
        v = list(result.values())[0] if isinstance(result, dict) else result[0]
        return v.to_numpy() if hasattr(v, "to_numpy") else np.array(v)

    @pytest.mark.skipif(
        not os.path.exists(CHECKPOINT_PATH),
        reason=f"Checkpoint not found: {CHECKPOINT_PATH}",
    )
    def test_flow_matches_pytorch(self, cpu_device):
        """MAX flow output matches PyTorch reference: max diff < 1e-3, correlation > 0.999."""
        from max import engine
        from models._vits_weight_loader import load_vits_weights, extract_speaker_embedding
        from models._vits_graph import build_flow_graph
        from _rvc_pytorch_reference import run_flow_reference

        import torch

        # Load checkpoint
        ckpt = torch.load(CHECKPOINT_PATH, map_location="cpu", weights_only=False)
        sd = ckpt["weight"]

        vits_weights, _, config = load_vits_weights(CHECKPOINT_PATH)
        g_np = extract_speaker_embedding(sd, sid=0)  # [256, 1]

        hidden = 192
        T_val = 20

        # Fixed z_p and mask
        rng = np.random.default_rng(42)
        z_p = rng.standard_normal((1, hidden, T_val)).astype(np.float32) * 0.1
        mask = np.ones((1, 1, T_val), dtype=np.float32)

        # --- PyTorch reference ---
        z_pt = run_flow_reference(
            CHECKPOINT_PATH, z_p, mask, sid=0, sr_tag="48k"
        )

        # --- MAX implementation ---
        flow_config = {
            "inter_channels": hidden,
            "hidden_channels": hidden,
            "n_layers": 3,
            "dilation_rate": 1,
            "n_flows": 4,
        }

        graph = build_flow_graph(
            vits_weights, g_np, flow_config, device="cpu", batch_size=1
        )
        model = engine.InferenceSession(devices=[cpu_device]).load(graph)

        result = model.execute(z_p, mask)
        z_max = self._result_to_numpy(result)

        # --- Comparison ---
        max_diff = np.max(np.abs(z_max - z_pt))
        correlation = np.corrcoef(z_max.flatten(), z_pt.flatten())[0, 1]

        print(f"\nFlow numerical validation:")
        print(f"  Max diff:     {max_diff:.6f}")
        print(f"  Correlation:  {correlation:.6f}")
        print(f"  MAX range:    [{z_max.min():.4f}, {z_max.max():.4f}]")
        print(f"  PT  range:    [{z_pt.min():.4f}, {z_pt.max():.4f}]")

        assert max_diff < 1e-3, f"Max diff {max_diff} exceeds tolerance 1e-3"
        assert correlation > 0.999, f"Correlation {correlation} below 0.999"
