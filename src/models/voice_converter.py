"""VoiceConverter: public API that orchestrates the full RVC v2 voice conversion pipeline.

Pipeline:
    AudioEncoder → RMVPE PitchExtractor → enc_p → sample z_p → flow (reverse) → HiFiGAN

Usage:
    vc = VoiceConverter.from_pretrained("path/to/model.pth")
    audio_out = vc.convert(audio_in, pitch_shift=0)

For testing without full model load (enc_p + flow + HiFiGAN only):
    vc = VoiceConverter._from_vits_only("path/to/model.pth")
"""

from __future__ import annotations

import math
import numpy as np


# ---------------------------------------------------------------------------
# Orchestration helpers (pure numpy — testable without any MAX graphs)
# ---------------------------------------------------------------------------


def sequence_mask(lengths: np.ndarray, max_len: int) -> np.ndarray:
    """Build a sequence mask from a lengths vector.

    Args:
        lengths: [B] int array of valid lengths.
        max_len: Maximum sequence length.

    Returns:
        [B, 1, max_len] float32 mask — 1.0 for valid positions, 0.0 elsewhere.
    """
    B = len(lengths)
    mask = np.zeros((B, 1, max_len), dtype=np.float32)
    for i, length in enumerate(lengths):
        mask[i, :, :length] = 1.0
    return mask


def interpolate_features_2x(features: np.ndarray) -> np.ndarray:
    """Upsample feature frames 2x by repeating each frame (nearest-neighbor).

    Matches Applio's ``F.interpolate(feats.permute(0, 2, 1), scale_factor=2)``
    which uses nearest-neighbour interpolation along the time axis.

    Args:
        features: [B, T, C] float32 feature array.

    Returns:
        [B, 2*T, C] float32 — each frame duplicated.
    """
    # np.repeat along axis=1 replicates each time step — exactly nearest-neighbor
    return np.repeat(features, 2, axis=1)


def apply_pitch_shift(f0: np.ndarray, semitones: float) -> np.ndarray:
    """Shift F0 by a number of semitones.

    Unvoiced frames (f0 == 0.0) are left at 0.0.

    Args:
        f0: [T] or [B, T] float32 array of F0 in Hz. 0 = unvoiced.
        semitones: Number of semitones to shift (positive = up, negative = down).

    Returns:
        Shifted F0 array, same shape as input.
    """
    if semitones == 0:
        return f0
    factor = 2.0 ** (semitones / 12.0)
    voiced = f0 > 0.0
    shifted = f0.copy()
    shifted[voiced] = f0[voiced] * factor
    return shifted


def quantize_f0(
    f0: np.ndarray,
    f0_mel_min: float | None = None,
    f0_mel_max: float | None = None,
) -> np.ndarray:
    """Convert F0 in Hz to mel-scale pitch bins (0-255, int32).

    Matches the Applio pipeline.py quantization exactly:

        f0_mel = 1127 * log(1 + f0 / 700)
        Scaled to [1, 255] over [f0_mel_min, f0_mel_max].
        Unvoiced frames (f0 == 0) → bin 0.

    Args:
        f0: [T] or [B, T] float32 array in Hz. 0 = unvoiced.
        f0_mel_min: Mel value at minimum voiced F0. Defaults to mel(50 Hz).
        f0_mel_max: Mel value at maximum voiced F0. Defaults to mel(1100 Hz).

    Returns:
        Same shape as f0, dtype int32, values in [0, 255].
    """
    if f0_mel_min is None:
        f0_mel_min = 1127.0 * np.log(1.0 + 50.0 / 700.0)
    if f0_mel_max is None:
        f0_mel_max = 1127.0 * np.log(1.0 + 1100.0 / 700.0)

    f0_mel = 1127.0 * np.log(1.0 + f0 / 700.0)

    # Voiced frames: map to [1, 255]
    voiced = f0_mel > 0
    f0_mel[voiced] = (
        (f0_mel[voiced] - f0_mel_min) * 254.0 / (f0_mel_max - f0_mel_min) + 1.0
    )

    # Clamp and round
    f0_mel = np.clip(f0_mel, 1, 255)
    f0_mel[~voiced] = 0.0  # restore unvoiced to bin 0

    return np.rint(f0_mel).astype(np.int32)


def sample_z_p(
    mean: np.ndarray, logvar: np.ndarray, mask: np.ndarray, noise_scale: float = 0.66666
) -> np.ndarray:
    """Sample z_p from the posterior distribution N(mean, exp(logvar)).

    Args:
        mean: [B, C, T] float32 posterior mean.
        logvar: [B, C, T] float32 posterior log-variance.
        mask: [B, 1, T] float32 sequence mask.
        noise_scale: Scaling factor for the noise (RVC default 0.66666).

    Returns:
        [B, C, T] float32 — sampled latent, zeroed outside the mask.
    """
    noise = np.random.randn(*mean.shape).astype(np.float32)
    z_p = (mean + np.exp(logvar) * noise * noise_scale) * mask
    return z_p


# ---------------------------------------------------------------------------
# VoiceConverter
# ---------------------------------------------------------------------------


class VoiceConverter:
    """Orchestrates the full RVC v2 voice conversion pipeline.

    Load with:
        vc = VoiceConverter.from_pretrained("model.pth")         # full pipeline
        vc = VoiceConverter._from_vits_only("model.pth")         # enc_p+flow+HiFiGAN only

    Then call:
        audio_out = vc.convert(audio_np, pitch_shift=0)
    """

    def __init__(
        self,
        _enc_p_model,
        _flow_model,
        _hifigan,
        _vits_weights: dict,
        _enc_p_config: dict,
        _config: dict,
        _device,
        _audio_encoder=None,
        _pitch_extractor=None,
    ):
        self._enc_p_model = _enc_p_model
        self._flow_model = _flow_model
        self._hifigan = _hifigan
        self._vits_weights = _vits_weights
        self._enc_p_config = _enc_p_config
        self._config = _config
        self._device = _device
        self._audio_encoder = _audio_encoder
        self._pitch_extractor = _pitch_extractor

    # ------------------------------------------------------------------
    # Factory methods
    # ------------------------------------------------------------------

    @classmethod
    def from_pretrained(
        cls,
        checkpoint_path: str,
        hubert_path: str = "facebook/hubert-base-ls960",
        rmvpe_path: str = "lj1995/VoiceConversionWebUI",
        device: str = "auto",
    ) -> "VoiceConverter":
        """Load all pipeline components from a single RVC v2 .pth checkpoint.

        This loads AudioEncoder and PitchExtractor in addition to the VITS
        graphs, which takes several minutes on first call (graph compilation).
        For faster testing, use ``_from_vits_only`` instead.

        Args:
            checkpoint_path: Path to RVC v2 .pth file.
            hubert_path: HuggingFace model ID for HuBERT/ContentVec encoder.
            rmvpe_path: HuggingFace model ID for RMVPE pitch extractor.
            device: "auto", "cpu", or "gpu".

        Returns:
            VoiceConverter with all components loaded and compiled.
        """
        vc = cls._from_vits_only(checkpoint_path, device=device)

        from .audio_encoder import AudioEncoder
        from .pitch_extractor import PitchExtractor

        vc._audio_encoder = AudioEncoder.from_pretrained(hubert_path, device=device)
        vc._pitch_extractor = PitchExtractor.from_pretrained(rmvpe_path, device=device)
        return vc

    @classmethod
    def _from_vits_only(
        cls,
        checkpoint_path: str,
        device: str = "auto",
    ) -> "VoiceConverter":
        """Load enc_p + flow + HiFiGAN only (skips AudioEncoder and RMVPE).

        Useful for testing and experimentation where the heavy audio encoder
        and pitch extractor are not needed (e.g. unit tests, or when features
        and F0 are supplied manually).

        Args:
            checkpoint_path: Path to RVC v2 .pth file.
            device: "auto", "cpu", or "gpu".

        Returns:
            VoiceConverter without _audio_encoder and _pitch_extractor set.
        """
        import torch
        from max import engine
        from max.driver import CPU, Accelerator, accelerator_count

        from ._vits_weight_loader import load_vits_weights, extract_speaker_embedding
        from ._vits_graph import build_enc_p_graph, build_flow_graph
        from .hifigan import NSFHiFiGAN

        # 1. Load weights
        vits_weights, hifigan_weights, config = load_vits_weights(checkpoint_path)

        # 2. Speaker embedding for flow graph
        ckpt = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
        sd = ckpt["weight"]
        g_np = extract_speaker_embedding(sd, sid=0)

        # 3. enc_p config (RVC v2 standard)
        enc_p_config = {
            "hidden_channels": 192,
            "filter_channels": 768,
            "n_heads": 2,
            "n_layers": 6,
            "kernel_size": 3,
            "window_size": 10,
            "out_channels": 192,
        }

        # 4. flow config (RVC v2 standard)
        flow_config = {
            "inter_channels": 192,
            "hidden_channels": 192,
            "n_layers": 3,
            "dilation_rate": 1,
            "n_flows": 4,
        }

        # 5. Determine device
        use_gpu = accelerator_count() > 0 if device == "auto" else device == "gpu"
        dev = Accelerator() if use_gpu else CPU()
        graph_device = "gpu" if use_gpu else "cpu"

        # 6. Build and compile MAX graphs
        enc_p_graph = build_enc_p_graph(vits_weights, enc_p_config, device=graph_device)
        flow_graph = build_flow_graph(vits_weights, g_np, flow_config, device=graph_device)

        session = engine.InferenceSession(devices=[dev])
        enc_p_model = session.load(enc_p_graph)
        flow_model = session.load(flow_graph)

        # 7. Build HiFiGAN from baked weights
        hifigan = NSFHiFiGAN._from_weights(hifigan_weights, config, device=device)

        return cls(
            _enc_p_model=enc_p_model,
            _flow_model=flow_model,
            _hifigan=hifigan,
            _vits_weights=vits_weights,
            _enc_p_config=enc_p_config,
            _config=config,
            _device=dev,
            _audio_encoder=None,
            _pitch_extractor=None,
        )

    # ------------------------------------------------------------------
    # Public inference API
    # ------------------------------------------------------------------

    def convert(
        self,
        audio: np.ndarray,
        pitch_shift: float = 0,
        sr: int = 16000,
    ) -> np.ndarray:
        """Convert audio to the target voice.

        Args:
            audio: [B, N] or [N] float32 input audio at `sr` Hz.
            pitch_shift: Semitones to shift pitch (0 = no shift, positive = up).
            sr: Input sample rate in Hz.

        Returns:
            [B, T_audio] float32 converted audio at the model's target sample rate.

        Raises:
            RuntimeError: If AudioEncoder or PitchExtractor are not loaded
                          (use ``from_pretrained`` instead of ``_from_vits_only``).
        """
        if self._audio_encoder is None or self._pitch_extractor is None:
            raise RuntimeError(
                "AudioEncoder and PitchExtractor are not loaded. "
                "Use VoiceConverter.from_pretrained() for the full pipeline, "
                "or supply features and F0 directly via convert_from_features()."
            )

        # --- Normalise input shape to [B, N] ---
        if audio.ndim == 1:
            audio = audio[np.newaxis, :]
        B = audio.shape[0]

        # 1. Resample to 16 kHz if needed
        if sr != 16000:
            audio = _resample(audio, sr, 16000)

        # 2. AudioEncoder → [B, T_cv, 768]
        features = self._audio_encoder.encode(audio)  # [B, T_cv, 768]

        # 3. RMVPE PitchExtractor → [T_rmvpe] Hz  (per-sample)
        # PitchExtractor expects [1, N]; process each batch element
        f0_list = []
        for b in range(B):
            f0_b = self._pitch_extractor.extract(audio[b : b + 1])  # [T_rmvpe]
            f0_list.append(f0_b)

        # 4. Interpolate features 2x to match RMVPE frame rate (~50fps → ~100fps)
        features_up = interpolate_features_2x(features)  # [B, 2*T_cv, 768]

        # 5-6. Align lengths, apply pitch shift, quantize F0
        T_feat = features_up.shape[1]

        f0_batch = np.zeros((B, T_feat), dtype=np.float32)
        pitch_batch = np.zeros((B, T_feat), dtype=np.int32)

        for b in range(B):
            f0 = f0_list[b]  # [T_rmvpe]
            # Trim/pad to T_feat
            T_f = min(len(f0), T_feat)
            f0_aligned = np.zeros(T_feat, dtype=np.float32)
            f0_aligned[:T_f] = f0[:T_f]

            if pitch_shift != 0:
                f0_aligned = apply_pitch_shift(f0_aligned, pitch_shift)

            f0_batch[b] = f0_aligned
            pitch_batch[b] = quantize_f0(f0_aligned.copy())

        # 7. Build sequence mask from actual feature lengths
        lengths = np.full(B, T_feat, dtype=np.int32)

        return self.convert_from_features(
            features_up, f0_batch, pitch_batch, lengths
        )

    def convert_from_features(
        self,
        features: np.ndarray,
        f0: np.ndarray,
        pitch: np.ndarray,
        lengths: np.ndarray | None = None,
        noise_scale: float = 0.66666,
    ) -> np.ndarray:
        """Run enc_p → sample z_p → flow → HiFiGAN from pre-computed inputs.

        This is the core synthesis path. ``convert()`` calls this internally
        after running the AudioEncoder and PitchExtractor.

        Args:
            features: [B, T, 768] float32 content features from AudioEncoder.
            f0: [B, T] float32 F0 in Hz (0 = unvoiced). Must already be pitch-shifted.
            pitch: [B, T] int32 pitch bins (0-255). Already quantized.
            lengths: [B] int32 valid frame counts. Defaults to T for all.
            noise_scale: Posterior noise scale (RVC default 0.66666).

        Returns:
            [B, T_audio] float32 converted audio.
        """
        from ._vits_graph import compute_rel_attention_biases

        B, T, _ = features.shape

        if lengths is None:
            lengths = np.full(B, T, dtype=np.int32)

        # 7. Sequence mask [B, 1, T]
        mask = sequence_mask(lengths, T)

        # 8. Compute relative position attention biases (numpy pre-pass)
        #    Need encoder input in BCT format for bias computation.
        enc_p_cfg = self._enc_p_config
        hidden = enc_p_cfg["hidden_channels"]

        x = features @ self._vits_weights["emb_phone.weight"].T  # [B, T, hidden]
        if "emb_phone.bias" in self._vits_weights:
            x = x + self._vits_weights["emb_phone.bias"]

        pitch_i32 = pitch.astype(np.int32)
        pitch_emb = self._vits_weights["emb_pitch.weight"][pitch_i32]  # [B, T, hidden]
        x = x + pitch_emb
        x = x * math.sqrt(hidden)
        x = np.where(x > 0, x, 0.1 * x)  # LeakyReLU(0.1)
        x_bct = x.transpose(0, 2, 1)       # [B, hidden, T]

        biases_k, biases_v = compute_rel_attention_biases(
            self._vits_weights, x_bct, mask, enc_p_cfg
        )

        # 9. enc_p graph → mean [B, 192, T], logvar [B, 192, T], mask [B, 1, T]
        inputs = [
            self._to_device_tensor(features),
            self._to_device_tensor(pitch_i32),
            self._to_device_tensor(lengths),
        ]
        for i in range(enc_p_cfg["n_layers"]):
            inputs.append(self._to_device_tensor(biases_k[i]))
            inputs.append(self._to_device_tensor(biases_v[i]))

        enc_result = self._enc_p_model.execute(*inputs)
        mean, logvar, enc_mask = self._unpack_3(enc_result)

        # 10. Sample z_p
        z_p = sample_z_p(mean, logvar, enc_mask, noise_scale=noise_scale)

        # 11. flow (reverse) → z [B, 192, T]
        flow_result = self._flow_model.execute(
            self._to_device_tensor(z_p),
            self._to_device_tensor(enc_mask),
        )
        z = self._unpack_1(flow_result)

        # 12. Apply mask
        z = z * enc_mask

        # 13. HiFiGAN → audio [B, T_audio]
        return self._hifigan.synthesize(z, f0)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _to_device_tensor(self, arr: np.ndarray):
        """Wrap a numpy array as a device Buffer when running on GPU.

        MAX graphs compiled for GPU reject CPU tensors at ``execute()``. On CPU
        the array is passed through unchanged. Mirrors the pattern used by
        AudioEncoder, PitchExtractor, and NSFHiFiGAN.
        """
        from max.driver import Accelerator, Buffer

        if isinstance(self._device, Accelerator):
            return Buffer.from_numpy(np.ascontiguousarray(arr)).to(self._device)
        return arr

    @staticmethod
    def _unpack_1(result) -> np.ndarray:
        """Extract single output tensor from MAX engine result."""
        v = list(result.values())[0] if isinstance(result, dict) else result[0]
        return v.to_numpy() if hasattr(v, "to_numpy") else np.array(v)

    @staticmethod
    def _unpack_3(result) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Extract three output tensors from MAX engine result."""
        vals = list(result.values()) if isinstance(result, dict) else list(result)
        return tuple(
            v.to_numpy() if hasattr(v, "to_numpy") else np.array(v) for v in vals
        )


# ---------------------------------------------------------------------------
# Resampling helper
# ---------------------------------------------------------------------------


def _resample(audio: np.ndarray, orig_sr: int, target_sr: int) -> np.ndarray:
    """Resample audio from orig_sr to target_sr using scipy.

    Args:
        audio: [B, N] float32.
        orig_sr: Original sample rate.
        target_sr: Target sample rate.

    Returns:
        [B, N'] float32 resampled audio.
    """
    from scipy.signal import resample_poly
    from math import gcd

    g = gcd(orig_sr, target_sr)
    up = target_sr // g
    down = orig_sr // g
    return np.stack(
        [resample_poly(audio[b], up, down).astype(np.float32) for b in range(audio.shape[0])]
    )
