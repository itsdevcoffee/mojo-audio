"""PitchExtractor: RMVPE pitch estimation via MAX Graph U-Net + numpy BiGRU.

Replaces PyTorch RMVPE in the voice conversion pipeline.
Works on DGX Spark ARM64 without PyTorch CUDA.

Example:
    model = PitchExtractor.from_pretrained()
    f0 = model.extract(audio_np)  # [1, N] @16kHz → [T_frames] Hz, 0=unvoiced
"""
from __future__ import annotations
import numpy as np

_MEL_SR = 16000
_MEL_N_FFT = 1024
_MEL_WIN = 1024
_MEL_HOP = 160
_MEL_N_MELS = 128
_MEL_FMIN = 50.0
_MEL_FMAX = 2006.0


def _mel_spectrogram(audio: np.ndarray) -> np.ndarray:
    """Compute log mel spectrogram matching RMVPE training preprocessing.

    Matches torchaudio MelSpectrogram with norm="slaney", mel_scale="slaney",
    center=True, hann window — implemented via librosa for portability.

    Args:
        audio: [1, N] float32 @16kHz.

    Returns:
        [1, 1, T, 128] float32 (batch=1, channel=1, time, mel_bins).
    """
    import librosa

    # audio: [1, N] → [N]
    y = audio[0]

    # librosa mel_spectrogram with center=True (default) matches torchaudio center=True
    mel = librosa.feature.melspectrogram(
        y=y,
        sr=_MEL_SR,
        n_fft=_MEL_N_FFT,
        win_length=_MEL_WIN,
        hop_length=_MEL_HOP,
        n_mels=_MEL_N_MELS,
        fmin=_MEL_FMIN,
        fmax=_MEL_FMAX,
        window="hann",
        center=True,
        norm="slaney",
        htk=False,  # htk=False → slaney mel scale
    )
    # mel: [n_mels, T]
    log_mel = np.log(mel + 1e-8)

    # [n_mels, T] → [1, 1, T, n_mels]
    log_mel = log_mel.T[np.newaxis, np.newaxis, :, :]
    return log_mel.astype(np.float32)


class PitchExtractor:
    """RMVPE pitch extractor: MAX Graph U-Net + numpy BiGRU.

    Extracts F0 (fundamental frequency) from raw 16kHz audio.
    Output is Hz per 10ms frame; 0.0 Hz = unvoiced.

    Example:
        model = PitchExtractor.from_pretrained()
        f0 = model.extract(audio)  # [1, N] @16kHz → [T] Hz
    """

    def __init__(self, _unet_model, _device, _weights: dict):
        self._unet_model = _unet_model
        self._device = _device
        self._weights = _weights

    @classmethod
    def _from_weights(cls, weights: dict, device: str = "auto") -> "PitchExtractor":
        """Build MAX Graph model from internal weight dict.

        Args:
            weights: Internal weight dict from _rmvpe_weight_loader.load_rmvpe_weights().
                     Must include U-Net conv/BN weights, GRU weights (gru.*), and
                     linear output weights (linear.*).
            device: "auto" (GPU if available), "gpu", or "cpu".

        Returns:
            PitchExtractor backed by a compiled MAX Graph U-Net.
        """
        from max import engine
        from max.driver import Accelerator, CPU, accelerator_count
        from max.graph import DeviceRef
        from ._rmvpe import build_unet_graph

        use_gpu = accelerator_count() > 0 if device == "auto" else device == "gpu"
        dev = Accelerator() if use_gpu else CPU()
        device_ref = DeviceRef.GPU(0) if use_gpu else DeviceRef.CPU()

        graph = build_unet_graph(weights, device_ref)
        unet_model = engine.InferenceSession(devices=[dev]).load(graph)
        return cls(_unet_model=unet_model, _device=dev, _weights=weights)

    @classmethod
    def from_pretrained(
        cls,
        repo_id: str = "lj1995/VoiceConversionWebUI",
        filename: str = "rmvpe.pt",
        device: str = "auto",
        cache_dir: str | None = None,
    ) -> "PitchExtractor":
        """Load RMVPE from HuggingFace Hub.

        Args:
            repo_id: HuggingFace repo containing rmvpe.pt.
            filename: Weight file within the repo.
            device: "auto" (GPU if available), "gpu", or "cpu".
            cache_dir: Override download cache location.
        """
        from ._rmvpe_weight_loader import load_rmvpe_weights
        weights = load_rmvpe_weights(repo_id, filename, cache_dir)
        return cls._from_weights(weights, device=device)

    def extract(self, audio: np.ndarray, threshold: float = 0.03) -> np.ndarray:
        """Extract F0 from raw audio.

        Args:
            audio: [1, N] float32, 16kHz, normalized to roughly [-1, 1].
            threshold: Voiced/unvoiced threshold on sigmoid probability (default 0.03).

        Returns:
            [T_frames] float32 — F0 in Hz, 0.0 = unvoiced. ~100 frames/second.
        """
        from max.driver import Accelerator, Tensor
        from ._rmvpe import bigru_forward, linear_output, salience_to_hz

        # Step 1: mel [1, N] → [1, 1, T, 128]
        mel = _mel_spectrogram(audio)

        # Step 2: reshape to NHWC [1, T, 128, 1] for MAX Graph
        mel_nhwc = np.ascontiguousarray(mel.transpose(0, 2, 3, 1))  # [1, 1, T, 128] → [1, T, 128, 1]

        # Step 3: U-Net MAX Graph [1, T, 128, 1] → [1, T, 384]
        if isinstance(self._device, Accelerator):
            inp = Tensor.from_numpy(mel_nhwc).to(self._device)
        else:
            inp = mel_nhwc
        result = self._unet_model.execute(inp)
        features = (list(result.values())[0] if isinstance(result, dict) else result[0]).to_numpy()

        # Step 4: BiGRU [1, T, 384] → [1, T, 512]
        gru_out = bigru_forward(features, self._weights)

        # Step 5: Linear [1, T, 512] → [1, T, 360]
        salience = linear_output(gru_out, self._weights)

        # Step 6: Salience → Hz [T]
        return salience_to_hz(salience, threshold=threshold)
