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


_CHECKPOINT_CANDIDATES = [
    "/home/maskkiller/Downloads/voice files/extracted/theweeknd biggest data set/theweekv1.pth",
    "/home/visage/Downloads/voice files/extracted/theweeknd biggest data set/theweekv1.pth",
]
CHECKPOINT_PATH = next((p for p in _CHECKPOINT_CANDIDATES if os.path.exists(p)), _CHECKPOINT_CANDIDATES[0])


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


# ======================================================================
# enc_p graph tests (Task 4)
# ======================================================================


def _make_enc_p_weights(rng, config):
    """Generate random weights for the TextEncoder (enc_p).

    Returns a dict with all the keys expected by build_enc_p_graph:
      emb_phone.weight [hidden, 768], emb_phone.bias [hidden]
      emb_pitch.weight [256, hidden]
      encoder.attn_layers.{i}.conv_{q,k,v,o}.weight [hidden, hidden, 1] / .bias [hidden]
      encoder.attn_layers.{i}.emb_rel_k [1, 2*window+1, head_dim]
      encoder.attn_layers.{i}.emb_rel_v [1, 2*window+1, head_dim]
      encoder.norm_layers_{1,2}.{i}.gamma [hidden] / .beta [hidden]
      encoder.ffn_layers.{i}.conv_1.weight [filter, hidden, k] / .bias [filter]
      encoder.ffn_layers.{i}.conv_2.weight [hidden, filter, k] / .bias [hidden]
      proj.weight [2*out, hidden, 1] / .bias [2*out]
    """
    hidden = config["hidden_channels"]
    filter_ch = config["filter_channels"]
    n_heads = config["n_heads"]
    n_layers = config["n_layers"]
    kernel_size = config["kernel_size"]
    window_size = config["window_size"]
    out_ch = config["out_channels"]
    head_dim = hidden // n_heads
    scale = 0.02  # small init for stability

    w = {}

    # Phone embedding: Linear(768, hidden)
    w["emb_phone.weight"] = rng.standard_normal((hidden, 768)).astype(np.float32) * scale
    w["emb_phone.bias"] = np.zeros(hidden, dtype=np.float32)

    # Pitch embedding: Embedding(256, hidden)
    w["emb_pitch.weight"] = rng.standard_normal((256, hidden)).astype(np.float32) * scale

    # Encoder layers
    for i in range(n_layers):
        # Attention Conv1d projections (k=1)
        for conv in ["conv_q", "conv_k", "conv_v", "conv_o"]:
            w[f"encoder.attn_layers.{i}.{conv}.weight"] = (
                rng.standard_normal((hidden, hidden, 1)).astype(np.float32) * scale
            )
            w[f"encoder.attn_layers.{i}.{conv}.bias"] = np.zeros(hidden, dtype=np.float32)

        # Relative position embeddings
        rel_std = head_dim ** -0.5
        n_rel = 2 * window_size + 1
        w[f"encoder.attn_layers.{i}.emb_rel_k"] = (
            rng.standard_normal((1, n_rel, head_dim)).astype(np.float32) * rel_std
        )
        w[f"encoder.attn_layers.{i}.emb_rel_v"] = (
            rng.standard_normal((1, n_rel, head_dim)).astype(np.float32) * rel_std
        )

        # Layer norms
        for ln in ["norm_layers_1", "norm_layers_2"]:
            w[f"encoder.{ln}.{i}.gamma"] = np.ones(hidden, dtype=np.float32)
            w[f"encoder.{ln}.{i}.beta"] = np.zeros(hidden, dtype=np.float32)

        # FFN: Conv1d(hidden, filter, k) -> Conv1d(filter, hidden, k)
        pad = (kernel_size - 1) // 2
        w[f"encoder.ffn_layers.{i}.conv_1.weight"] = (
            rng.standard_normal((filter_ch, hidden, kernel_size)).astype(np.float32) * scale
        )
        w[f"encoder.ffn_layers.{i}.conv_1.bias"] = np.zeros(filter_ch, dtype=np.float32)
        w[f"encoder.ffn_layers.{i}.conv_2.weight"] = (
            rng.standard_normal((hidden, filter_ch, kernel_size)).astype(np.float32) * scale
        )
        w[f"encoder.ffn_layers.{i}.conv_2.bias"] = np.zeros(hidden, dtype=np.float32)

    # Projection: Conv1d(hidden, 2*out_ch, k=1)
    w["proj.weight"] = rng.standard_normal((2 * out_ch, hidden, 1)).astype(np.float32) * scale
    w["proj.bias"] = np.zeros(2 * out_ch, dtype=np.float32)

    return w


def _prepare_enc_p_inputs(rng, weights, config, T_val=20, batch_size=1):
    """Prepare all inputs for the enc_p graph including relative attention biases.

    Returns a list of numpy arrays in the order expected by the graph.
    """
    import math as _math

    hidden = config["hidden_channels"]

    features = rng.standard_normal((batch_size, T_val, 768)).astype(np.float32) * 0.1
    pitch = rng.integers(0, 256, (batch_size, T_val)).astype(np.int32)
    lengths = np.array([T_val] * batch_size, dtype=np.int32)

    # Compute the encoder input in numpy to get relative biases
    # Replicate enc_p front-end: emb_phone + emb_pitch, scale, leaky_relu, transpose
    x = features @ weights["emb_phone.weight"].T  # [B, T, hidden]
    if "emb_phone.bias" in weights:
        x = x + weights["emb_phone.bias"]

    # Pitch embedding via indexing
    pitch_emb = weights["emb_pitch.weight"][pitch]  # [B, T, hidden]
    x = x + pitch_emb
    x = x * _math.sqrt(hidden)

    # LeakyReLU(0.1)
    x = np.where(x > 0, x, 0.1 * x)

    # Transpose to BCT
    x_bct = x.transpose(0, 2, 1)  # [B, hidden, T]

    mask = np.ones((batch_size, 1, T_val), dtype=np.float32)

    from models._vits_graph import compute_rel_attention_biases
    biases_k, biases_v = compute_rel_attention_biases(weights, x_bct, mask, config)

    # Build input list
    inputs = [features, pitch, lengths]
    n_layers = config["n_layers"]
    for i in range(n_layers):
        inputs.append(biases_k[i])
        inputs.append(biases_v[i])

    return inputs


_ENC_P_CONFIG = {
    "hidden_channels": 192,
    "filter_channels": 768,
    "n_heads": 2,
    "n_layers": 6,
    "kernel_size": 3,
    "window_size": 10,
    "out_channels": 192,
}


class TestEncPGraph:
    """Tests for the TextEncoder (enc_p) MAX Graph.

    Compiles ONE enc_p graph with random weights and tests all properties.
    """

    @pytest.fixture(scope="class")
    def cpu_device(self):
        from max.driver import CPU
        return CPU()

    @pytest.fixture(scope="class")
    def enc_p_model_and_inputs(self, cpu_device):
        """Build and compile enc_p graph ONCE. Returns (model, inputs, config)."""
        from max import engine
        from models._vits_graph import build_enc_p_graph

        rng = np.random.default_rng(42)
        config = _ENC_P_CONFIG.copy()
        weights = _make_enc_p_weights(rng, config)

        graph = build_enc_p_graph(weights, config, device="cpu", batch_size=1, max_T=200)
        model = engine.InferenceSession(devices=[cpu_device]).load(graph)

        # Prepare inputs for T=20
        rng_input = np.random.default_rng(99)
        inputs = _prepare_enc_p_inputs(rng_input, weights, config, T_val=20, batch_size=1)

        return model, inputs, config, weights

    @staticmethod
    def _results_to_numpy(result):
        """Extract mean, logvar, mask from model result."""
        if isinstance(result, dict):
            vals = list(result.values())
        else:
            vals = list(result)
        return tuple(
            v.to_numpy() if hasattr(v, "to_numpy") else np.array(v) for v in vals
        )

    def test_enc_p_output_shapes(self, enc_p_model_and_inputs):
        """Verify mean [1, 192, T], logvar [1, 192, T], mask [1, 1, T]."""
        model, inputs, config, _ = enc_p_model_and_inputs
        result = model.execute(*inputs)
        mean, logvar, mask = self._results_to_numpy(result)

        T_val = 20
        hidden = config["out_channels"]
        assert mean.shape == (1, hidden, T_val), f"mean shape: {mean.shape}"
        assert logvar.shape == (1, hidden, T_val), f"logvar shape: {logvar.shape}"
        assert mask.shape == (1, 1, T_val), f"mask shape: {mask.shape}"

    def test_enc_p_output_not_nan(self, enc_p_model_and_inputs):
        """No NaN or Inf in outputs."""
        model, inputs, config, _ = enc_p_model_and_inputs
        result = model.execute(*inputs)
        mean, logvar, mask = self._results_to_numpy(result)

        assert not np.any(np.isnan(mean)), "mean contains NaN"
        assert not np.any(np.isinf(mean)), "mean contains Inf"
        assert not np.any(np.isnan(logvar)), "logvar contains NaN"
        assert not np.any(np.isinf(logvar)), "logvar contains Inf"

    def test_enc_p_mask_applied(self, enc_p_model_and_inputs):
        """Outputs are zero where mask is zero (partial mask test)."""
        model, _, config, weights = enc_p_model_and_inputs

        # Create inputs with partial mask (last 5 timesteps masked out)
        rng = np.random.default_rng(456)
        T_val = 20
        valid_len = 15  # only first 15 timesteps are valid

        features = rng.standard_normal((1, T_val, 768)).astype(np.float32) * 0.1
        pitch = rng.integers(0, 256, (1, T_val)).astype(np.int32)
        lengths = np.array([valid_len], dtype=np.int32)

        # Compute encoder input for bias computation
        import math as _math
        x = features @ weights["emb_phone.weight"].T
        if "emb_phone.bias" in weights:
            x = x + weights["emb_phone.bias"]
        x = x + weights["emb_pitch.weight"][pitch]
        x = x * _math.sqrt(config["hidden_channels"])
        x = np.where(x > 0, x, 0.1 * x)
        x_bct = x.transpose(0, 2, 1)

        mask_np = np.zeros((1, 1, T_val), dtype=np.float32)
        mask_np[:, :, :valid_len] = 1.0

        from models._vits_graph import compute_rel_attention_biases
        biases_k, biases_v = compute_rel_attention_biases(weights, x_bct, mask_np, config)

        inputs = [features, pitch, lengths]
        for i in range(config["n_layers"]):
            inputs.append(biases_k[i])
            inputs.append(biases_v[i])

        result = model.execute(*inputs)
        mean, logvar, mask_out = self._results_to_numpy(result)

        # Mask should be 0 for positions >= valid_len
        assert np.allclose(mask_out[:, :, valid_len:], 0.0, atol=1e-6), (
            "Mask should be zero for masked positions"
        )
        # Mean and logvar should be zero where mask is zero
        assert np.allclose(mean[:, :, valid_len:], 0.0, atol=1e-6), (
            "Mean should be zero where mask is zero"
        )
        assert np.allclose(logvar[:, :, valid_len:], 0.0, atol=1e-6), (
            "Logvar should be zero where mask is zero"
        )


class TestEncPNumericalValidation:
    """Numerical comparison with PyTorch reference.

    Skipped if checkpoint file is not present.
    """

    @pytest.fixture(scope="class")
    def cpu_device(self):
        from max.driver import CPU
        return CPU()

    @staticmethod
    def _results_to_numpy(result):
        if isinstance(result, dict):
            vals = list(result.values())
        else:
            vals = list(result)
        return tuple(
            v.to_numpy() if hasattr(v, "to_numpy") else np.array(v) for v in vals
        )

    @pytest.mark.skipif(
        not os.path.exists(CHECKPOINT_PATH),
        reason=f"Checkpoint not found: {CHECKPOINT_PATH}",
    )
    def test_enc_p_matches_pytorch(self, cpu_device):
        """MAX enc_p output matches PyTorch reference: max diff < 1e-3."""
        import math as _math

        from max import engine
        from models._vits_weight_loader import load_vits_weights
        from models._vits_graph import build_enc_p_graph, compute_rel_attention_biases
        from _rvc_pytorch_reference import run_enc_p_reference

        import torch

        vits_weights, _, _ = load_vits_weights(CHECKPOINT_PATH)

        config = _ENC_P_CONFIG.copy()
        T_val = 20

        rng = np.random.default_rng(42)
        features = rng.standard_normal((1, T_val, 768)).astype(np.float32)
        pitch = rng.integers(0, 256, (1, T_val)).astype(np.int64)

        # --- PyTorch reference ---
        m_pt, logs_pt, mask_pt = run_enc_p_reference(
            CHECKPOINT_PATH, features, pitch, sr_tag="48k"
        )

        # --- MAX implementation ---
        hidden = config["hidden_channels"]

        # Replicate front-end to get encoder input for bias computation
        x = features @ vits_weights["emb_phone.weight"].T
        if "emb_phone.bias" in vits_weights:
            x = x + vits_weights["emb_phone.bias"]
        pitch_i32 = pitch.astype(np.int32)
        x = x + vits_weights["emb_pitch.weight"][pitch_i32]
        x = x * _math.sqrt(hidden)
        x = np.where(x > 0, x, 0.1 * x)
        x_bct = x.transpose(0, 2, 1)

        mask_np = np.ones((1, 1, T_val), dtype=np.float32)
        biases_k, biases_v = compute_rel_attention_biases(
            vits_weights, x_bct, mask_np, config
        )

        graph = build_enc_p_graph(
            vits_weights, config, device="cpu", batch_size=1, max_T=200
        )
        model = engine.InferenceSession(devices=[cpu_device]).load(graph)

        lengths = np.array([T_val], dtype=np.int32)
        inputs = [features, pitch_i32, lengths]
        for i in range(config["n_layers"]):
            inputs.append(biases_k[i])
            inputs.append(biases_v[i])

        result = model.execute(*inputs)
        m_max, logs_max, mask_max = self._results_to_numpy(result)

        # --- Comparison ---
        max_diff_m = np.max(np.abs(m_max - m_pt))
        max_diff_logs = np.max(np.abs(logs_max - logs_pt))
        corr_m = np.corrcoef(m_max.flatten(), m_pt.flatten())[0, 1]
        corr_logs = np.corrcoef(logs_max.flatten(), logs_pt.flatten())[0, 1]

        print(f"\nenc_p numerical validation:")
        print(f"  m_p max diff:     {max_diff_m:.6f}")
        print(f"  logs_p max diff:  {max_diff_logs:.6f}")
        print(f"  m_p correlation:  {corr_m:.6f}")
        print(f"  logs_p correlation: {corr_logs:.6f}")

        assert max_diff_m < 1e-3, f"Mean max diff {max_diff_m} exceeds tolerance 1e-3"
        assert max_diff_logs < 1e-3, f"LogVar max diff {max_diff_logs} exceeds tolerance 1e-3"
        assert corr_m > 0.999, f"Mean correlation {corr_m} below 0.999"
        assert corr_logs > 0.999, f"LogVar correlation {corr_logs} below 0.999"


# ======================================================================
# Speaker conditioning baking tests (Task 5)
# ======================================================================


class TestSpeakerCondBaking:
    """Verify bake_hifigan_cond works correctly with real checkpoint data.

    Tests here rely on the real .pth checkpoint; they are skipped when the
    file is absent.  No HiFiGAN graphs are compiled in this class — only
    numpy weight arithmetic is verified — so there is zero risk of OOM.
    """

    @pytest.fixture(scope="class")
    def checkpoint_state_dict(self):
        """Load the real checkpoint state dict once for the class."""
        import torch

        ckpt = torch.load(CHECKPOINT_PATH, map_location="cpu", weights_only=False)
        sd = ckpt["weight"]
        return sd

    # ------------------------------------------------------------------
    # Helper
    # ------------------------------------------------------------------

    @staticmethod
    def _to_np(t) -> np.ndarray:
        if hasattr(t, "numpy"):
            return t.numpy()
        if hasattr(t, "detach"):
            return t.detach().numpy()
        return np.asarray(t)

    # ------------------------------------------------------------------
    # Test 1: conv_pre.bias changes after baking
    # ------------------------------------------------------------------

    @pytest.mark.skipif(
        not os.path.exists(CHECKPOINT_PATH),
        reason=f"Checkpoint not found: {CHECKPOINT_PATH}",
    )
    def test_bias_changed_after_baking(self, checkpoint_state_dict):
        """conv_pre.bias is different from the original after bake_hifigan_cond."""
        from models._vits_weight_loader import extract_speaker_embedding, bake_hifigan_cond
        from models._hifigan_weight_loader import extract_hifigan_weights

        sd = checkpoint_state_dict

        hifigan_weights = extract_hifigan_weights(sd)
        g = extract_speaker_embedding(sd, sid=0)

        cond_weight = np.asarray(self._to_np(sd["dec.cond.weight"]), dtype=np.float32)
        cond_bias = np.asarray(self._to_np(sd["dec.cond.bias"]), dtype=np.float32)

        original_bias = hifigan_weights["conv_pre.bias"].copy()

        bake_hifigan_cond(hifigan_weights, g, cond_weight, cond_bias)

        assert not np.allclose(hifigan_weights["conv_pre.bias"], original_bias), (
            "conv_pre.bias was not modified by bake_hifigan_cond — "
            "speaker conditioning had no effect"
        )

    # ------------------------------------------------------------------
    # Test 2: mathematical correctness of the baked bias
    # ------------------------------------------------------------------

    @pytest.mark.skipif(
        not os.path.exists(CHECKPOINT_PATH),
        reason=f"Checkpoint not found: {CHECKPOINT_PATH}",
    )
    def test_baked_bias_is_mathematically_correct(self, checkpoint_state_dict):
        """new_bias == old_bias + cond_weight[:,:,0] @ g + cond_bias (within float32 tolerance)."""
        from models._vits_weight_loader import extract_speaker_embedding, bake_hifigan_cond
        from models._hifigan_weight_loader import extract_hifigan_weights

        sd = checkpoint_state_dict

        hifigan_weights = extract_hifigan_weights(sd)
        g = extract_speaker_embedding(sd, sid=0)  # [256, 1]

        cond_weight = np.asarray(self._to_np(sd["dec.cond.weight"]), dtype=np.float32)  # [C, 256, 1]
        cond_bias = np.asarray(self._to_np(sd["dec.cond.bias"]), dtype=np.float32)      # [C]

        original_bias = hifigan_weights["conv_pre.bias"].copy()  # [C]

        bake_hifigan_cond(hifigan_weights, g, cond_weight, cond_bias)

        # Expected: old_bias + cond_weight[:, :, 0] @ g.squeeze(-1) + cond_bias
        # cond_weight[:, :, 0]: [C, 256]   g: [256, 1]  -> matmul -> [C, 1] -> squeeze -> [C]
        cond_out = (cond_weight[:, :, 0] @ g).squeeze(-1) + cond_bias  # [C]
        expected = original_bias + cond_out

        np.testing.assert_allclose(
            hifigan_weights["conv_pre.bias"],
            expected,
            rtol=1e-5,
            err_msg="Baked conv_pre.bias does not match expected value",
        )

    # ------------------------------------------------------------------
    # Test 3: different speakers produce different baked biases
    # ------------------------------------------------------------------

    @pytest.mark.skipif(
        not os.path.exists(CHECKPOINT_PATH),
        reason=f"Checkpoint not found: {CHECKPOINT_PATH}",
    )
    def test_different_speakers_produce_different_biases(self, checkpoint_state_dict):
        """Baking with sid=0 vs sid=1 yields different conv_pre.bias arrays."""
        from models._vits_weight_loader import extract_speaker_embedding, bake_hifigan_cond
        from models._hifigan_weight_loader import extract_hifigan_weights

        sd = checkpoint_state_dict

        # Check the checkpoint has at least 2 speakers
        emb_weight = self._to_np(sd["emb_g.weight"])
        if emb_weight.shape[0] < 2:
            pytest.skip("Checkpoint has only one speaker — cannot compare sid=0 vs sid=1")

        cond_weight = np.asarray(self._to_np(sd["dec.cond.weight"]), dtype=np.float32)
        cond_bias = np.asarray(self._to_np(sd["dec.cond.bias"]), dtype=np.float32)

        # Bake for sid=0
        hw0 = extract_hifigan_weights(sd)
        g0 = extract_speaker_embedding(sd, sid=0)
        bake_hifigan_cond(hw0, g0, cond_weight, cond_bias)
        bias0 = hw0["conv_pre.bias"].copy()

        # Bake for sid=1 (fresh hifigan_weights to start from the same original)
        hw1 = extract_hifigan_weights(sd)
        g1 = extract_speaker_embedding(sd, sid=1)
        bake_hifigan_cond(hw1, g1, cond_weight, cond_bias)
        bias1 = hw1["conv_pre.bias"].copy()

        assert not np.allclose(bias0, bias1), (
            "Speaker 0 and speaker 1 produced identical conv_pre.bias — "
            "speaker embeddings may be degenerate"
        )


# ======================================================================
# VoiceConverter orchestration tests (Task 6)
# ======================================================================


class TestVoiceConverter:
    """Pure-numpy tests for VoiceConverter orchestration helpers.

    These do NOT compile any MAX graphs or load any checkpoints — all
    tests run on numpy alone and complete instantly.
    """

    # ------------------------------------------------------------------
    # Test 1: F0 quantization
    # ------------------------------------------------------------------

    def test_f0_quantization_zero_is_unvoiced(self):
        """F0 = 0.0 (unvoiced) always maps to pitch bin 0."""
        from models.voice_converter import quantize_f0

        f0 = np.array([0.0, 100.0, 200.0, 0.0, 440.0], dtype=np.float32)
        bins = quantize_f0(f0.copy())

        assert bins[0] == 0, f"Unvoiced frame should be bin 0, got {bins[0]}"
        assert bins[3] == 0, f"Unvoiced frame should be bin 0, got {bins[3]}"

    def test_f0_quantization_voiced_in_range(self):
        """Voiced F0 values map to bins in [1, 255]."""
        from models.voice_converter import quantize_f0

        # Typical voiced range 80-800 Hz
        f0 = np.linspace(80.0, 800.0, 50).astype(np.float32)
        bins = quantize_f0(f0.copy())

        assert np.all(bins >= 1), f"All voiced bins should be >= 1, got min {bins.min()}"
        assert np.all(bins <= 255), f"All bins should be <= 255, got max {bins.max()}"

    def test_f0_quantization_output_dtype(self):
        """quantize_f0 returns int32 array."""
        from models.voice_converter import quantize_f0

        f0 = np.array([0.0, 220.0, 440.0, 880.0], dtype=np.float32)
        bins = quantize_f0(f0)

        assert bins.dtype == np.int32, f"Expected int32, got {bins.dtype}"

    def test_f0_quantization_monotonic(self):
        """Higher F0 → higher pitch bin (monotonic mapping for voiced frames)."""
        from models.voice_converter import quantize_f0

        f0 = np.array([100.0, 200.0, 400.0, 800.0], dtype=np.float32)
        bins = quantize_f0(f0.copy())

        for i in range(len(bins) - 1):
            assert bins[i] <= bins[i + 1], (
                f"Non-monotonic: bins[{i}]={bins[i]} > bins[{i+1}]={bins[i+1]}"
            )

    def test_f0_quantization_440hz_near_midpoint(self):
        """440 Hz (A4) should map to a mid-range bin (roughly around 120-150)."""
        from models.voice_converter import quantize_f0

        f0 = np.array([440.0], dtype=np.float32)
        bins = quantize_f0(f0)

        assert 100 <= bins[0] <= 180, (
            f"440 Hz should map to mid-range bin, got {bins[0]}"
        )

    def test_f0_quantization_preserves_shape(self):
        """quantize_f0 preserves the input array shape."""
        from models.voice_converter import quantize_f0

        f0_2d = np.random.rand(3, 50).astype(np.float32) * 500.0
        bins = quantize_f0(f0_2d)

        assert bins.shape == f0_2d.shape, (
            f"Shape mismatch: {bins.shape} != {f0_2d.shape}"
        )

    # ------------------------------------------------------------------
    # Test 2: Feature interpolation (2x nearest-neighbor)
    # ------------------------------------------------------------------

    def test_feature_interpolation_doubles_time(self):
        """interpolate_features_2x doubles the time dimension."""
        from models.voice_converter import interpolate_features_2x

        features = np.random.rand(1, 50, 768).astype(np.float32)
        up = interpolate_features_2x(features)

        assert up.shape == (1, 100, 768), f"Expected (1, 100, 768), got {up.shape}"

    def test_feature_interpolation_batch(self):
        """interpolate_features_2x works correctly with batch size > 1."""
        from models.voice_converter import interpolate_features_2x

        B, T, C = 3, 20, 768
        features = np.random.rand(B, T, C).astype(np.float32)
        up = interpolate_features_2x(features)

        assert up.shape == (B, 2 * T, C), f"Expected ({B}, {2*T}, {C}), got {up.shape}"

    def test_feature_interpolation_is_nearest_neighbor(self):
        """Each frame in input appears exactly twice, consecutively, in output."""
        from models.voice_converter import interpolate_features_2x

        T, C = 5, 4
        features = np.arange(T * C, dtype=np.float32).reshape(1, T, C)
        up = interpolate_features_2x(features)

        for t in range(T):
            np.testing.assert_array_equal(
                up[0, 2 * t], features[0, t],
                err_msg=f"Frame {t}: first copy mismatch",
            )
            np.testing.assert_array_equal(
                up[0, 2 * t + 1], features[0, t],
                err_msg=f"Frame {t}: second copy mismatch",
            )

    def test_feature_interpolation_preserves_channel_dim(self):
        """interpolate_features_2x does not alter the channel dimension."""
        from models.voice_converter import interpolate_features_2x

        features = np.random.rand(2, 30, 256).astype(np.float32)
        up = interpolate_features_2x(features)

        assert up.shape[2] == 256, f"Channel dim changed: {up.shape[2]}"

    # ------------------------------------------------------------------
    # Test 3: Pitch shift
    # ------------------------------------------------------------------

    def test_pitch_shift_zero_is_identity(self):
        """apply_pitch_shift with semitones=0 returns unchanged F0."""
        from models.voice_converter import apply_pitch_shift

        f0 = np.array([0.0, 100.0, 220.0, 440.0], dtype=np.float32)
        out = apply_pitch_shift(f0, semitones=0)

        np.testing.assert_array_equal(out, f0)

    def test_pitch_shift_12_semitones_doubles_f0(self):
        """Shifting up 12 semitones (one octave) doubles the voiced F0."""
        from models.voice_converter import apply_pitch_shift

        f0 = np.array([0.0, 220.0, 440.0], dtype=np.float32)
        out = apply_pitch_shift(f0, semitones=12)

        np.testing.assert_allclose(out[1], 440.0, rtol=1e-5)
        np.testing.assert_allclose(out[2], 880.0, rtol=1e-5)
        assert out[0] == 0.0, "Unvoiced frame should stay 0 after pitch shift"

    def test_pitch_shift_negative_halves_f0(self):
        """Shifting down 12 semitones halves the voiced F0."""
        from models.voice_converter import apply_pitch_shift

        f0 = np.array([440.0, 880.0], dtype=np.float32)
        out = apply_pitch_shift(f0, semitones=-12)

        np.testing.assert_allclose(out[0], 220.0, rtol=1e-5)
        np.testing.assert_allclose(out[1], 440.0, rtol=1e-5)

    def test_pitch_shift_unvoiced_stays_zero(self):
        """Unvoiced frames (f0 == 0) remain zero after any pitch shift."""
        from models.voice_converter import apply_pitch_shift

        f0 = np.array([0.0, 0.0, 440.0, 0.0], dtype=np.float32)
        for semitones in [-12, -5, 3, 7, 12]:
            out = apply_pitch_shift(f0, semitones=semitones)
            assert out[0] == 0.0 and out[1] == 0.0 and out[3] == 0.0, (
                f"Unvoiced frames changed with semitones={semitones}: {out}"
            )

    def test_pitch_shift_does_not_modify_input(self):
        """apply_pitch_shift does not mutate the input array."""
        from models.voice_converter import apply_pitch_shift

        f0 = np.array([220.0, 440.0, 880.0], dtype=np.float32)
        original = f0.copy()
        _ = apply_pitch_shift(f0, semitones=7)

        np.testing.assert_array_equal(f0, original, err_msg="Input was mutated")

    # ------------------------------------------------------------------
    # Test 4: sequence_mask
    # ------------------------------------------------------------------

    def test_sequence_mask_shape(self):
        """sequence_mask returns [B, 1, max_len] float32."""
        from models.voice_converter import sequence_mask

        lengths = np.array([5, 10, 7], dtype=np.int32)
        mask = sequence_mask(lengths, max_len=12)

        assert mask.shape == (3, 1, 12), f"Expected (3, 1, 12), got {mask.shape}"
        assert mask.dtype == np.float32

    def test_sequence_mask_valid_positions(self):
        """sequence_mask is 1.0 for valid positions and 0.0 beyond length."""
        from models.voice_converter import sequence_mask

        lengths = np.array([3, 5], dtype=np.int32)
        mask = sequence_mask(lengths, max_len=8)

        np.testing.assert_array_equal(mask[0, 0, :3], 1.0)
        np.testing.assert_array_equal(mask[0, 0, 3:], 0.0)
        np.testing.assert_array_equal(mask[1, 0, :5], 1.0)
        np.testing.assert_array_equal(mask[1, 0, 5:], 0.0)

    # ------------------------------------------------------------------
    # Test 5: sample_z_p
    # ------------------------------------------------------------------

    def test_sample_z_p_shape(self):
        """sample_z_p returns same shape as mean/logvar."""
        from models.voice_converter import sample_z_p

        rng = np.random.default_rng(42)
        mean = rng.standard_normal((1, 192, 20)).astype(np.float32)
        logvar = rng.standard_normal((1, 192, 20)).astype(np.float32)
        mask = np.ones((1, 1, 20), dtype=np.float32)

        z = sample_z_p(mean, logvar, mask)

        assert z.shape == mean.shape, f"Expected {mean.shape}, got {z.shape}"

    def test_sample_z_p_masked_to_zero(self):
        """sample_z_p output is 0.0 where mask is 0.0."""
        from models.voice_converter import sample_z_p

        rng = np.random.default_rng(7)
        mean = rng.standard_normal((1, 192, 20)).astype(np.float32)
        logvar = np.zeros((1, 192, 20), dtype=np.float32)
        mask = np.ones((1, 1, 20), dtype=np.float32)
        mask[:, :, 15:] = 0.0

        z = sample_z_p(mean, logvar, mask)

        np.testing.assert_allclose(
            z[:, :, 15:], 0.0, atol=1e-6,
            err_msg="Masked positions should be zero",
        )

    @pytest.mark.slow
    @pytest.mark.skipif(
        not os.path.exists(CHECKPOINT_PATH),
        reason=f"Checkpoint not found: {CHECKPOINT_PATH}",
    )
    def test_convert_shape(self):
        """Full pipeline: VoiceConverter.from_pretrained + convert produces audio."""
        from models.voice_converter import VoiceConverter

        vc = VoiceConverter.from_pretrained(
            CHECKPOINT_PATH,
            hubert_path="facebook/hubert-base-ls960",
            rmvpe_path="lj1995/VoiceConversionWebUI",
            device="cpu",
        )

        # 1 second of silence at 16 kHz
        audio = np.zeros((1, 16000), dtype=np.float32)
        out = vc.convert(audio, pitch_shift=0, sr=16000)

        assert out.ndim == 2, f"Expected 2D output, got {out.ndim}D"
        assert out.shape[0] == 1, f"Expected batch=1, got {out.shape[0]}"
        assert out.shape[1] > 0, "Output audio should have non-zero length"
