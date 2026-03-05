"""AudioEncoder: HuBERT / ContentVec feature extraction via MAX Graph."""

from __future__ import annotations
import numpy as np
from pathlib import Path


def _transformer_block_ops(x, block_weights: dict, device_ref, heads: int = 12, hidden: int = 768):
    """Placeholder — implemented in Task 5."""
    raise NotImplementedError("Implemented in Task 5")


class AudioEncoder:
    """MAX Graph implementation of HuBERT / ContentVec audio encoder.

    Supports facebook/hubert-base-ls960 and lengyue233/content-vec-best.
    Automatically selects GPU if available, falls back to CPU.

    Example:
        model = AudioEncoder.from_pretrained("facebook/hubert-base-ls960")
        features = model.encode(audio_np)  # [1, seq] -> [1, frames, 768]
    """

    def __init__(self, _model, _device, _device_ref, _session):
        self._model = _model
        self._device = _device
        self._device_ref = _device_ref
        self._session = _session

    @classmethod
    def _from_weights(cls, weights: dict, device: str = "auto") -> "AudioEncoder":
        """Build MAX Graph from loaded weight dict. Implemented in Task 6."""
        raise NotImplementedError("Implemented in Task 6")

    @classmethod
    def from_pretrained(
        cls,
        model_id: str,
        device: str = "auto",
        cache_dir: str | None = None,
    ) -> "AudioEncoder":
        """Load model from HuggingFace Hub or local path.

        Args:
            model_id: HuggingFace model ID or local path to .safetensors/.pt file.
            device: "auto" (default), "gpu", or "cpu".
            cache_dir: Override default cache (~/.cache/mojo-audio/models/).
        """
        raise NotImplementedError("Implemented in Task 6")

    def encode(self, audio: np.ndarray) -> np.ndarray:
        """Encode raw audio waveform to feature vectors.

        Args:
            audio: Float32 numpy array, shape [1, samples], 16kHz, normalized [-1, 1].

        Returns:
            Float32 numpy array, shape [1, time_frames, 768].
            For 1s audio: [1, 49, 768].
        """
        raise NotImplementedError("Implemented in Task 6")
