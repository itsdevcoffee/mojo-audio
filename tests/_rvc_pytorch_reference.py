"""RVC v2 NSF-HiFiGAN PyTorch reference implementation.

Self-contained decoder code extracted from the official RVC source for
numerical comparison testing against our MAX implementation.

Source: https://github.com/RVC-Project/Retrieval-based-Voice-Conversion-WebUI
Files:  infer/lib/infer_pack/models.py, modules.py, commons.py
"""

from __future__ import annotations

import math
from typing import Optional

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn
from torch.nn import Conv1d, ConvTranspose1d
from torch.nn.utils import remove_weight_norm, weight_norm

# ---------------------------------------------------------------------------
# Utilities from commons.py
# ---------------------------------------------------------------------------

def get_padding(kernel_size: int, dilation: int = 1) -> int:
    return int((kernel_size * dilation - dilation) / 2)


def init_weights(m, mean: float = 0.0, std: float = 0.01):
    classname = m.__class__.__name__
    if classname.find("Conv") != -1:
        m.weight.data.normal_(mean, std)


# ---------------------------------------------------------------------------
# LRELU_SLOPE from modules.py
# ---------------------------------------------------------------------------

LRELU_SLOPE = 0.1

# ---------------------------------------------------------------------------
# ResBlock1 from modules.py
# ---------------------------------------------------------------------------


class ResBlock1(torch.nn.Module):
    def __init__(self, channels: int, kernel_size: int = 3, dilation: tuple = (1, 3, 5)):
        super(ResBlock1, self).__init__()
        self.convs1 = nn.ModuleList(
            [
                weight_norm(
                    Conv1d(
                        channels,
                        channels,
                        kernel_size,
                        1,
                        dilation=dilation[0],
                        padding=get_padding(kernel_size, dilation[0]),
                    )
                ),
                weight_norm(
                    Conv1d(
                        channels,
                        channels,
                        kernel_size,
                        1,
                        dilation=dilation[1],
                        padding=get_padding(kernel_size, dilation[1]),
                    )
                ),
                weight_norm(
                    Conv1d(
                        channels,
                        channels,
                        kernel_size,
                        1,
                        dilation=dilation[2],
                        padding=get_padding(kernel_size, dilation[2]),
                    )
                ),
            ]
        )
        self.convs1.apply(init_weights)

        self.convs2 = nn.ModuleList(
            [
                weight_norm(
                    Conv1d(
                        channels,
                        channels,
                        kernel_size,
                        1,
                        dilation=1,
                        padding=get_padding(kernel_size, 1),
                    )
                ),
                weight_norm(
                    Conv1d(
                        channels,
                        channels,
                        kernel_size,
                        1,
                        dilation=1,
                        padding=get_padding(kernel_size, 1),
                    )
                ),
                weight_norm(
                    Conv1d(
                        channels,
                        channels,
                        kernel_size,
                        1,
                        dilation=1,
                        padding=get_padding(kernel_size, 1),
                    )
                ),
            ]
        )
        self.convs2.apply(init_weights)
        self.lrelu_slope = LRELU_SLOPE

    def forward(self, x: torch.Tensor, x_mask: Optional[torch.Tensor] = None):
        for c1, c2 in zip(self.convs1, self.convs2):
            xt = F.leaky_relu(x, self.lrelu_slope)
            if x_mask is not None:
                xt = xt * x_mask
            xt = c1(xt)
            xt = F.leaky_relu(xt, self.lrelu_slope)
            if x_mask is not None:
                xt = xt * x_mask
            xt = c2(xt)
            x = xt + x
        if x_mask is not None:
            x = x * x_mask
        return x

    def remove_weight_norm(self):
        for l in self.convs1:
            remove_weight_norm(l)
        for l in self.convs2:
            remove_weight_norm(l)


# ---------------------------------------------------------------------------
# ResBlock2 from modules.py
# ---------------------------------------------------------------------------


class ResBlock2(torch.nn.Module):
    def __init__(self, channels: int, kernel_size: int = 3, dilation: tuple = (1, 3)):
        super(ResBlock2, self).__init__()
        self.convs = nn.ModuleList(
            [
                weight_norm(
                    Conv1d(
                        channels,
                        channels,
                        kernel_size,
                        1,
                        dilation=dilation[0],
                        padding=get_padding(kernel_size, dilation[0]),
                    )
                ),
                weight_norm(
                    Conv1d(
                        channels,
                        channels,
                        kernel_size,
                        1,
                        dilation=dilation[1],
                        padding=get_padding(kernel_size, dilation[1]),
                    )
                ),
            ]
        )
        self.convs.apply(init_weights)
        self.lrelu_slope = LRELU_SLOPE

    def forward(self, x, x_mask: Optional[torch.Tensor] = None):
        for c in self.convs:
            xt = F.leaky_relu(x, self.lrelu_slope)
            if x_mask is not None:
                xt = xt * x_mask
            xt = c(xt)
            x = xt + x
        if x_mask is not None:
            x = x * x_mask
        return x

    def remove_weight_norm(self):
        for l in self.convs:
            remove_weight_norm(l)


# ---------------------------------------------------------------------------
# SineGen from models.py
# ---------------------------------------------------------------------------


class SineGen(torch.nn.Module):
    """Sine waveform generator for NSF source excitation.

    Generates sine waves at the fundamental frequency and harmonics,
    with additive noise for unvoiced regions.
    """

    def __init__(
        self,
        samp_rate: int,
        harmonic_num: int = 0,
        sine_amp: float = 0.1,
        noise_std: float = 0.003,
        voiced_threshold: float = 0,
        flag_for_pulse: bool = False,
    ):
        super(SineGen, self).__init__()
        self.sine_amp = sine_amp
        self.noise_std = noise_std
        self.harmonic_num = harmonic_num
        self.dim = self.harmonic_num + 1
        self.sampling_rate = samp_rate
        self.voiced_threshold = voiced_threshold

    def _f02uv(self, f0):
        # generate uv signal
        uv = torch.ones_like(f0)
        uv = uv * (f0 > self.voiced_threshold)
        if uv.device.type == "privateuseone":  # for DirectML
            uv = uv.float()
        return uv

    def _f02sine(self, f0, upp):
        """f0: (batchsize, length, dim)
        where dim indicates fundamental tone and overtones
        """
        a = torch.arange(1, upp + 1, dtype=f0.dtype, device=f0.device)
        rad = f0 / self.sampling_rate * a
        rad2 = torch.fmod(rad[:, :-1, -1:].float() + 0.5, 1.0) - 0.5
        rad_acc = rad2.cumsum(dim=1).fmod(1.0).to(f0)
        rad += F.pad(rad_acc, (0, 0, 1, 0), mode="constant")
        rad = rad.reshape(f0.shape[0], -1, 1)
        b = torch.arange(
            1, self.dim + 1, dtype=f0.dtype, device=f0.device
        ).reshape(1, 1, -1)
        rad *= b
        rand_ini = torch.rand(1, 1, self.dim, device=f0.device)
        rand_ini[..., 0] = 0
        rad += rand_ini
        sines = torch.sin(2 * np.pi * rad)
        return sines

    def forward(self, f0: torch.Tensor, upp: int):
        """
        input F0: tensor(batchsize=1, length, dim=1)
                  f0 for unvoiced steps should be 0
        output sine_tensor: tensor(batchsize=1, length, dim)
        output uv: tensor(batchsize=1, length, 1)
        """
        with torch.no_grad():
            f0 = f0.unsqueeze(-1)
            sine_waves = self._f02sine(f0, upp) * self.sine_amp
            uv = self._f02uv(f0)
            uv = F.interpolate(
                uv.transpose(2, 1), scale_factor=float(upp), mode="nearest"
            ).transpose(2, 1)
            noise_amp = uv * self.noise_std + (1 - uv) * self.sine_amp / 3
            noise = noise_amp * torch.randn_like(sine_waves)
            sine_waves = sine_waves * uv + noise
        return sine_waves, uv, noise


# ---------------------------------------------------------------------------
# SourceModuleHnNSF from models.py
# ---------------------------------------------------------------------------


class SourceModuleHnNSF(torch.nn.Module):
    """Source module for harmonic-plus-noise NSF.

    Produces a sine-based excitation signal from F0, merged through a
    learned linear layer.
    """

    def __init__(
        self,
        sampling_rate: int,
        harmonic_num: int = 0,
        sine_amp: float = 0.1,
        add_noise_std: float = 0.003,
        voiced_threshod: float = 0,
        is_half: bool = True,
    ):
        super(SourceModuleHnNSF, self).__init__()

        self.sine_amp = sine_amp
        self.noise_std = add_noise_std
        self.is_half = is_half
        # to produce sine waveforms
        self.l_sin_gen = SineGen(
            sampling_rate, harmonic_num, sine_amp, add_noise_std, voiced_threshod
        )

        # to merge source harmonics into a single excitation
        self.l_linear = torch.nn.Linear(harmonic_num + 1, 1)
        self.l_tanh = torch.nn.Tanh()

    def forward(self, x: torch.Tensor, upp: int = 1):
        sine_wavs, uv, _ = self.l_sin_gen(x, upp)
        sine_wavs = sine_wavs.to(dtype=self.l_linear.weight.dtype)
        sine_merge = self.l_tanh(self.l_linear(sine_wavs))
        return sine_merge, None, None


# ---------------------------------------------------------------------------
# GeneratorNSF from models.py
# ---------------------------------------------------------------------------


class GeneratorNSF(torch.nn.Module):
    """RVC v2 NSF-HiFiGAN decoder (vocoder).

    Takes latent features [B, C, T] and frame-level F0 [B, T, 1],
    produces waveform audio [B, 1, T_audio].
    """

    def __init__(
        self,
        initial_channel: int,
        resblock: str,
        resblock_kernel_sizes: list,
        resblock_dilation_sizes: list,
        upsample_rates: list,
        upsample_initial_channel: int,
        upsample_kernel_sizes: list,
        gin_channels: int,
        sr: int,
        is_half: bool = False,
    ):
        super(GeneratorNSF, self).__init__()
        self.num_kernels = len(resblock_kernel_sizes)
        self.num_upsamples = len(upsample_rates)

        self.f0_upsamp = torch.nn.Upsample(scale_factor=math.prod(upsample_rates))
        self.m_source = SourceModuleHnNSF(
            sampling_rate=sr, harmonic_num=0, is_half=is_half
        )
        self.noise_convs = nn.ModuleList()
        self.conv_pre = Conv1d(
            initial_channel, upsample_initial_channel, 7, 1, padding=3
        )
        resblock_cls = ResBlock1 if resblock == "1" else ResBlock2

        self.ups = nn.ModuleList()
        for i, (u, k) in enumerate(zip(upsample_rates, upsample_kernel_sizes)):
            c_cur = upsample_initial_channel // (2 ** (i + 1))
            self.ups.append(
                weight_norm(
                    ConvTranspose1d(
                        upsample_initial_channel // (2**i),
                        upsample_initial_channel // (2 ** (i + 1)),
                        k,
                        u,
                        padding=(k - u) // 2,
                    )
                )
            )
            if i + 1 < len(upsample_rates):
                stride_f0 = math.prod(upsample_rates[i + 1 :])
                self.noise_convs.append(
                    Conv1d(
                        1,
                        c_cur,
                        kernel_size=stride_f0 * 2,
                        stride=stride_f0,
                        padding=stride_f0 // 2,
                    )
                )
            else:
                self.noise_convs.append(Conv1d(1, c_cur, kernel_size=1))

        self.resblocks = nn.ModuleList()
        for i in range(len(self.ups)):
            ch = upsample_initial_channel // (2 ** (i + 1))
            for j, (k, d) in enumerate(
                zip(resblock_kernel_sizes, resblock_dilation_sizes)
            ):
                self.resblocks.append(resblock_cls(ch, k, d))

        self.conv_post = Conv1d(ch, 1, 7, 1, padding=3, bias=False)
        self.ups.apply(init_weights)

        if gin_channels != 0:
            self.cond = nn.Conv1d(gin_channels, upsample_initial_channel, 1)

        self.upp = math.prod(upsample_rates)

        self.lrelu_slope = LRELU_SLOPE

    def forward(
        self,
        x,
        f0,
        g: Optional[torch.Tensor] = None,
        n_res: Optional[torch.Tensor] = None,
    ):
        har_source, noi_source, uv = self.m_source(f0, self.upp)
        har_source = har_source.transpose(1, 2)
        if n_res is not None:
            assert isinstance(n_res, torch.Tensor)
            n = int(n_res.item())
            if n * self.upp != har_source.shape[-1]:
                har_source = F.interpolate(
                    har_source, size=n * self.upp, mode="linear"
                )
            if n != x.shape[-1]:
                x = F.interpolate(x, size=n, mode="linear")
        x = self.conv_pre(x)
        if g is not None:
            x = x + self.cond(g)
        for i, (ups, noise_convs) in enumerate(zip(self.ups, self.noise_convs)):
            if i < self.num_upsamples:
                x = F.leaky_relu(x, self.lrelu_slope)
                x = ups(x)
                x_source = noise_convs(har_source)
                x = x + x_source
                xs: Optional[torch.Tensor] = None
                l = [i * self.num_kernels + j for j in range(self.num_kernels)]
                for j, resblock in enumerate(self.resblocks):
                    if j in l:
                        if xs is None:
                            xs = resblock(x)
                        else:
                            xs += resblock(x)
                assert isinstance(xs, torch.Tensor)
                x = xs / self.num_kernels
        x = F.leaky_relu(x)
        x = self.conv_post(x)
        x = torch.tanh(x)

        return x

    def remove_weight_norm(self):
        for l in self.ups:
            remove_weight_norm(l)
        for l in self.resblocks:
            l.remove_weight_norm()


# ---------------------------------------------------------------------------
# RVC v2 default config (40k / 48k models use the same architecture)
# ---------------------------------------------------------------------------

RVC_V2_GENERATOR_CONFIG = {
    "initial_channel": 192,
    "resblock": "1",
    "resblock_kernel_sizes": [3, 7, 11],
    "resblock_dilation_sizes": [[1, 3, 5], [1, 3, 5], [1, 3, 5]],
    "upsample_rates": [12, 10, 2, 2],
    "upsample_initial_channel": 512,
    "upsample_kernel_sizes": [24, 20, 4, 4],
    "gin_channels": 256,
}

# Sample-rate lookup for common RVC v2 models
_SR_TABLE = {
    "32k": 32000,
    "40k": 40000,
    "48k": 48000,
}


# ---------------------------------------------------------------------------
# Helper: load generator from RVC checkpoint
# ---------------------------------------------------------------------------


def load_rvc_generator(checkpoint_path: str, sr_tag: str = "40k") -> GeneratorNSF:
    """Load the NSF-HiFiGAN decoder from an RVC v2 checkpoint.

    Creates a GeneratorNSF with the standard RVC v2 config, loads only the
    ``dec.*`` weights (the vocoder portion of the full model), and returns
    the model in eval mode.

    Args:
        checkpoint_path: Path to an RVC ``.pth`` file.
        sr_tag: Sample-rate tag, one of ``"32k"``, ``"40k"``, ``"48k"``.
            Used to set the correct ``sr`` parameter on the generator.

    Returns:
        A :class:`GeneratorNSF` ready for inference.
    """
    sr = _SR_TABLE.get(sr_tag, 40000)

    # Build the generator with standard config
    model = GeneratorNSF(
        **RVC_V2_GENERATOR_CONFIG,
        sr=sr,
        is_half=False,
    )

    # Load checkpoint
    ckpt = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    # RVC checkpoints store weights under "weight" key
    state = ckpt.get("weight", ckpt)

    # Extract dec.* keys and strip the "dec." prefix
    dec_state = {}
    for k, v in state.items():
        if k.startswith("dec."):
            dec_state[k[4:]] = v.float()

    model.load_state_dict(dec_state)
    model.eval()
    return model


# ---------------------------------------------------------------------------
# Reference inference function
# ---------------------------------------------------------------------------


def run_pytorch_reference(
    checkpoint_path: str,
    latents: np.ndarray,
    f0: np.ndarray,
    sr_tag: str = "40k",
) -> np.ndarray:
    """Run the PyTorch reference decoder and return audio output.

    Args:
        checkpoint_path: Path to an RVC ``.pth`` file.
        latents: ``[B, 192, T]`` float32 latent features.
        f0: ``[B, T]`` float32 fundamental frequency at frame rate.
            Will be unsqueezed to ``[B, T, 1]`` as expected by
            :class:`GeneratorNSF`.
        sr_tag: Sample-rate tag (``"32k"``, ``"40k"``, ``"48k"``).

    Returns:
        ``[B, T_audio]`` float32 numpy array of generated audio.
    """
    model = load_rvc_generator(checkpoint_path, sr_tag=sr_tag)

    x = torch.from_numpy(latents).float()  # [B, 192, T]
    f0_t = torch.from_numpy(f0).float()  # [B, T]

    # GeneratorNSF.forward expects f0 as [B, T] — SineGen unsqueezes internally
    with torch.no_grad():
        audio = model(x, f0_t, g=None)  # [B, 1, T_audio]

    # Squeeze channel dim -> [B, T_audio]
    return audio.squeeze(1).numpy()
