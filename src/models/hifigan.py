"""NSFHiFiGAN: Neural Source-Filter HiFiGAN vocoder via MAX Graph.

Converts latent features + F0 pitch -> audio waveform for RVC v2 voice conversion.
Supports 32kHz, 40kHz, and 48kHz sample rates (auto-detected from checkpoint).

Example:
    vocoder = NSFHiFiGAN.from_pretrained("path/to/rvc_model.pth")
    audio = vocoder.synthesize(latents, f0)  # [B, 192, T], [B, T] -> [B, T_audio]
"""

from __future__ import annotations

import numpy as np


class NSFHiFiGAN:
    """Public-facing NSF-HiFiGAN vocoder orchestrating harmonic source,
    NHWC layout conversion, MAX graph execution, and output formatting."""

    def __init__(self, _model, _device, _config, _batch_size=1):
        self._model = _model
        self._device = _device
        self._config = _config
        self._batch_size = _batch_size

    @classmethod
    def _from_weights(cls, weights, config, device="auto", batch_size=1):
        """Build MAX Graph from weight dict + config."""
        from max import engine
        from max.driver import CPU, Accelerator, accelerator_count

        from ._hifigan_graph import build_hifigan_graph

        use_gpu = accelerator_count() > 0 if device == "auto" else device == "gpu"
        dev = Accelerator() if use_gpu else CPU()
        graph_device = "gpu" if use_gpu else "cpu"
        graph = build_hifigan_graph(
            weights, config, device=graph_device, batch_size=batch_size
        )
        session = engine.InferenceSession(devices=[dev])
        model = session.load(graph)
        return cls(_model=model, _device=dev, _config=config, _batch_size=batch_size)

    @classmethod
    def from_pretrained(cls, checkpoint_path, device="auto", batch_size=1):
        """Load from RVC v2 .pth checkpoint.

        Args:
            checkpoint_path: Path to an RVC v2 .pth checkpoint file.
            device: "auto", "cpu", or "gpu".
            batch_size: Batch size for the compiled graph.

        Returns:
            An NSFHiFiGAN instance ready for synthesis.
        """
        import torch

        from ._hifigan_weight_loader import extract_hifigan_weights, parse_hifigan_config

        ckpt = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
        state_dict = ckpt.get("weight", ckpt)
        sr = ckpt.get("sr", 48000)
        config_list = ckpt.get("config", [None] * 17)

        config = parse_hifigan_config(config_list, sr)
        weights = extract_hifigan_weights(state_dict)
        return cls._from_weights(weights, config, device=device, batch_size=batch_size)

    def synthesize(self, latents: np.ndarray, f0: np.ndarray) -> np.ndarray:
        """Synthesize audio from latent features and F0.

        Args:
            latents: [B, 192, T] float32 -- from VITS encoder/flow (PyTorch channel-first)
            f0: [B, T] float32 -- F0 in Hz at frame rate, 0 = unvoiced

        Returns:
            [B, T_audio] float32 -- audio waveform [-1, 1], where T_audio = T * hop_length
        """
        from max.driver import Accelerator, Buffer

        B = latents.shape[0]
        T = latents.shape[2]
        hop = self._config["hop_length"]
        sr = self._config["sr"]
        T_audio = T * hop

        # Stage 1: Harmonic source (numpy)
        excitation = self._harmonic_source(f0, T_audio, sr)  # [B, 1, T_audio]

        # Convert latents from [B, C, T] to NHWC [B, T, 1, C]
        latents_nhwc = np.ascontiguousarray(
            latents.transpose(0, 2, 1)[:, :, np.newaxis, :]
        ).astype(np.float32)

        # Convert excitation from [B, 1, T_audio] to NHWC [B, T_audio, 1, 1]
        exc_nhwc = np.ascontiguousarray(
            excitation.transpose(0, 2, 1)[:, :, :, np.newaxis]
        ).astype(np.float32)

        # Execute MAX graph
        if isinstance(self._device, Accelerator):
            inp_lat = Buffer.from_numpy(latents_nhwc).to(self._device)
            inp_exc = Buffer.from_numpy(exc_nhwc).to(self._device)
        else:
            inp_lat = latents_nhwc
            inp_exc = exc_nhwc

        result = self._model.execute(inp_lat, inp_exc)
        out = list(result.values())[0] if isinstance(result, dict) else result[0]
        out_np = out.to_numpy()  # [B, T_audio, 1, 1] NHWC

        # Squeeze to [B, T_audio]
        return out_np.reshape(B, -1)

    @staticmethod
    def _harmonic_source(f0: np.ndarray, T_audio: int, sr: int) -> np.ndarray:
        """Generate excitation signal from F0.

        Args:
            f0: [B, T] F0 in Hz at frame rate, 0 = unvoiced
            T_audio: Target audio length in samples
            sr: Sample rate

        Returns:
            [B, 1, T_audio] excitation signal
        """
        B, T = f0.shape
        hop = T_audio // T

        # Upsample F0 to audio sample rate
        f0_up = np.repeat(f0, hop, axis=-1)[:, :T_audio]  # [B, T_audio]

        # Voiced/unvoiced mask
        uv = (f0_up > 0).astype(np.float32)

        # Phase accumulation + sine
        phase = np.cumsum(2.0 * np.pi * f0_up / sr, axis=-1)
        sine = np.sin(phase) * uv

        # Add noise
        noise = np.random.randn(B, T_audio).astype(np.float32) * 0.003
        excitation = (sine + noise).astype(np.float32)

        return excitation[:, np.newaxis, :]  # [B, 1, T_audio]
