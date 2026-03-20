# NSF-HiFiGAN Vocoder Design

**Date:** 2026-03-20
**Status:** Approved
**Sprint:** 3

## Summary

Implement the NSF-HiFiGAN neural vocoder in MAX Graph for the RVC voice conversion pipeline. Takes 192-dim latent features + F0 pitch → produces audio waveform. Supports RVC v2 checkpoints at 32kHz, 40kHz, and 48kHz sample rates.

## Architecture

### Two-stage inference

**Stage 1 — Harmonic Source (numpy in `synthesize()`):**

F0 arrives at frame rate `[B, T]` (one value per latent frame, matching the T dimension of latents). The harmonic source module:

1. **Upsample F0** from frame rate to audio sample rate: repeat each frame value `hop_length` times (or `np.interp` for smoother interpolation), producing `[B, T_audio]` where `T_audio = T × hop_length`.
2. **Generate excitation signal:**

```python
# f0_upsampled: [B, T_audio] — F0 in Hz at sample rate, 0 = unvoiced
uv = (f0_upsampled > 0).astype(np.float32)          # [B, T_audio] voiced mask
# Instantaneous phase via cumulative sum
phase = np.cumsum(2 * np.pi * f0_upsampled / sr, axis=-1)  # [B, T_audio]
sine = np.sin(phase) * uv                            # zero sine for unvoiced
noise = np.random.randn(*sine.shape).astype(np.float32) * 0.003
excitation = (sine + noise).astype(np.float32)        # [B, 1, T_audio] after unsqueeze
```

Unvoiced frames (F0=0): sine is zeroed via the `uv` mask; only the noise component persists. This matches the reference NSF-HiFiGAN implementation.

The `m_source` module in the RVC checkpoint has one learned parameter: `m_source.l_linear.weight` (a 1×1 linear layer that mixes sine harmonics). For Sprint 3 we use the simplified single-harmonic source above. The learned mixing can be added later if quality requires it.

Output: `[B, 1, T_audio]` excitation signal passed to the MAX graph.

**Stage 2 — Neural Filter (MAX Graph):** The main computation.

```
Inputs: latents [B, 192, T], excitation [B, 1, T_audio]
  → Conv Pre: Conv1d(192, 512, K=7, pad=3) — [B, 192, T] → [B, 512, T]
  → Upsample Block 0: ConvTranspose(512→256, stride=S0) + noise_conv_0(excitation) + 3× ResBlock
  → Upsample Block 1: ConvTranspose(256→128, stride=S1) + noise_conv_1(excitation) + 3× ResBlock
  → Upsample Block 2: ConvTranspose(128→64, stride=S2) + noise_conv_2(excitation) + 3× ResBlock
  → Upsample Block 3: ConvTranspose(64→32, stride=S3) + noise_conv_3(excitation) + 3× ResBlock
  → Conv Post: Conv1d(32, 1, K=7, pad=3) + tanh
Output: [B, 1, T_audio] → squeeze → [B, T_audio] audio waveform
```

### ResBlock detail

Each ResBlock (type "1") contains two sets of 3 dilated Conv1d layers with a residual connection:

```
x_in → LeakyReLU → Conv1d(K, dilation=d1) → LeakyReLU → Conv1d(K, dilation=1) → + x_in → x_out
```

Repeated for 3 kernel sizes [3, 7, 11] with dilation patterns [1, 3, 5] for the first conv of each pair. The second conv always uses dilation=1. All convolutions use weight normalization (no BatchNorm). LeakyReLU with alpha=0.1.

**BatchNorm clarification:** Standard HiFiGAN and RVC's NSF-HiFiGAN ResBlocks use weight normalization, NOT BatchNorm. The weight loader bakes weight normalization (weight_g/weight_v → plain weight). BatchNorm baking is only needed if non-standard checkpoints include it; the loader should handle both cases defensively.

### Noise convolutions (excitation injection)

Each upsample block has a `noise_conv` that downsamples the excitation signal to match the current feature resolution:

| Block | Input channels | Output channels | Kernel | Stride | Purpose |
|---|---|---|---|---|---|
| 0 | 1 | 256 | S0×S1×S2×S3 | S0×S1×S2×S3 | Downsample excitation to lowest res |
| 1 | 1 | 128 | S1×S2×S3 | S1×S2×S3 | Downsample to block 1 resolution |
| 2 | 1 | 64 | S2×S3 | S2×S3 | Downsample to block 2 resolution |
| 3 | 1 | 32 | S3 | S3 | Downsample to block 3 resolution |

For 48kHz (strides [12,10,2,2]): noise_conv kernels/strides are [480, 40, 4, 2].
For 40kHz (strides [10,10,2,2]): noise_conv kernels/strides are [400, 40, 4, 2].

The noise_conv output is added element-wise to the upsampled features before the residual blocks.

### ConvTranspose1d generalization

The RMVPE zero-interleave workaround (`_rmvpe.py:124-181`) is hardcoded for stride=2. HiFiGAN upsample strides are 10, 8, 12, etc. The workaround must be **generalized to arbitrary stride S**: insert (S-1) zeros between each sample along the time dimension, then apply regular conv2d. This is new work — not a direct copy from RMVPE.

### Speaker conditioning (gin_channels)

RVC's NSF-HiFiGAN supports optional speaker embedding `g: [B, 256, 1]` injected via addition after the Conv Pre output. In the RVC inference path, `g` comes from the speaker encoder. For Sprint 3, speaker conditioning is **not implemented** — the latent features from the VITS flow already encode speaker identity. We store `gin_channels` in config for future use but do not wire it into the graph.

### Multi-sample-rate config

| Param | 32kHz | 40kHz | 48kHz |
|---|---|---|---|
| `upsample_rates` | [10,8,2,2] | [10,10,2,2] | [12,10,2,2] |
| `upsample_kernel_sizes` | [20,16,4,4] | [16,16,4,4] | [24,20,4,4] |
| `n_mel_channels` | 80 | 125 | 128 |
| `hop_length` | 320 | 400 | 480 |
| `sample_rate` | 32000 | 40000 | 48000 |

All other params are shared: `upsample_initial_channel=512`, `resblock_kernel_sizes=[3,7,11]`, `resblock_dilation_sizes=[[1,3,5],[1,3,5],[1,3,5]]`, `inter_channels=192`, `gin_channels=256`.

### RVC config list parsing

The RVC `.pth` checkpoint contains a `config` key with a positional list. The mapping for v2:

```python
config = checkpoint["config"]
# config[0]  = filter_channels (256)
# config[1]  = filter_channels (256, repeated)
# config[2]  = segment_size (not needed)
# config[3]  = inter_channels (192)
# config[4]  = hidden_channels (192)
# config[5]  = filter_channels (256)
# config[6]  = n_heads (not needed)
# config[7]  = n_layers (not needed)
# config[8]  = kernel_size (not needed)
# config[9]  = p_dropout (not needed)
# config[10] = resblock ("1")
# config[11] = resblock_kernel_sizes ([3, 7, 11])
# config[12] = resblock_dilation_sizes ([[1,3,5],[1,3,5],[1,3,5]])
# config[13] = upsample_rates (e.g., [12, 10, 2, 2])
# config[14] = upsample_initial_channel (512)
# config[15] = upsample_kernel_sizes (e.g., [24, 20, 4, 4])
# config[16] = protect (not needed)
#
# Sample rate comes from checkpoint["sr"] (int)
```

The weight loader parses these indices to build the model config dict.

## Files

| File | Purpose |
|---|---|
| `src/models/hifigan.py` | `NSFHiFiGAN` class: `__init__`, `_from_weights`, `from_pretrained`, `synthesize` |
| `src/models/_hifigan_weight_loader.py` | Load from RVC `.pth`, extract `dec.*` keys, bake weight-norm, parse config |
| `src/models/_hifigan_graph.py` | MAX Graph builder: upsampling blocks, residual blocks, conv pre/post |
| `tests/test_hifigan.py` | Shape, NaN, correctness tests |

### `src/models/hifigan.py`

```python
class NSFHiFiGAN:
    def __init__(self, _model, _device, _config, _batch_size=1): ...

    @classmethod
    def _from_weights(cls, weights, config, device="auto", batch_size=1): ...

    @classmethod
    def from_pretrained(cls, checkpoint_path, device="auto", batch_size=1): ...

    def synthesize(self, latents, f0):
        """
        Args:
            latents: [B, 192, T] float32 — from VITS encoder/flow
            f0: [B, T] float32 — F0 in Hz at frame rate, 0 = unvoiced.
                Upsampled to audio sample rate internally.

        Returns:
            [B, T_audio] float32 — audio waveform, [-1, 1]
            where T_audio = T × hop_length
        """
```

### `src/models/_hifigan_weight_loader.py`

- Load checkpoint, handle nested `{"weight": state_dict}` or flat state dict
- Extract `dec.*` keys, strip `dec.` prefix
- Parse `config` list and `sr` to build model config dict
- Reconstruct weight-normed layers: `weight = weight_v * (weight_g / norm(weight_v))`
- Defensively bake BatchNorm if present (unlikely in standard RVC but handle gracefully)
- Internal key convention: `conv_pre.*`, `ups.{i}.*`, `noise_convs.{i}.*`, `resblocks.{i}.convs{1,2}.{j}.*`, `conv_post.*`

### `src/models/_hifigan_graph.py`

- Two graph inputs: `latents [B, inter_channels, T]`, `excitation [B, 1, T_audio]`
- Conv Pre: Conv1d(192, 512, K=7, pad=3) via conv2d NHWC
- 4× Upsample blocks:
  - ConvTranspose1d via **generalized** zero-interleave + conv2d (stride S, insert S-1 zeros)
  - noise_conv: Conv1d on excitation, output added to upsampled features
  - 3× ResBlock1: pairs of dilated Conv1d with LeakyReLU(0.1), residual connection
- Conv Post: Conv1d(32, 1, K=7, pad=3) + tanh
- All Conv1d implemented as Conv2d with H=1, NHWC layout

### MAX Graph ops and workarounds

| Operation | MAX implementation |
|---|---|
| Conv1d | `ops.conv2d` with NHWC, width=1 |
| Conv1d (dilated) | `ops.conv2d` with `dilation` param, NHWC |
| ConvTranspose1d (stride S) | Insert (S-1) zeros between samples + `ops.conv2d` (**generalized from RMVPE**) |
| Weight normalization | Reconstructed at load time → plain weight |
| BatchNorm (if present) | Baked at load time → `y = x * scale + offset` |
| LeakyReLU(0.1) | `ops.maximum(x, ops.mul(x, 0.1))` or `ops.leaky_relu` if available |
| tanh | `ops.tanh` |

## Weight Loading Detail

### RVC checkpoint structure

```python
checkpoint = torch.load("model.pth", map_location="cpu")
# checkpoint keys:
#   "weight": OrderedDict  — full SynthesizerTrn state dict
#   "config": list          — positional config (see parsing section above)
#   "info": str             — training info
#   "sr": int               — sample rate
#   "f0": int               — 1 if F0-conditioned, 0 if not
#   "version": str          — "v2"
```

We extract only `dec.*` keys from the state dict. The `f0` flag confirms NSF conditioning (should be 1).

### Weight normalization reconstruction

HiFiGAN uses `torch.nn.utils.weight_norm` on Conv Pre, upsampling ConvTranspose, residual convolutions, and Conv Post:

```python
# Checkpoint has weight_g (scalar magnitude) and weight_v (direction tensor)
weight = weight_v * (weight_g / torch.norm(weight_v, dim=reduce_dims, keepdim=True))
# Store reconstructed weight directly — no weight_norm in MAX
```

## Tests

### Level 1 (no download, fast)

**Shape tests:**
- `test_synthesize_shape_48k`: random weights, 48kHz config → output is `[1, T*480]`
- `test_synthesize_shape_40k`: random weights, 40kHz config → output is `[1, T*400]`
- `test_synthesize_shape_32k`: random weights, 32kHz config → output is `[1, T*320]`
- `test_synthesize_batch2_shape`: batch=2 → output is `[2, T*hop]`

**NaN tests:**
- `test_synthesize_not_nan`: random weights, random input → no NaN/Inf
- `test_synthesize_unvoiced_not_nan`: random weights, `f0 = zeros` (all unvoiced) → no NaN/Inf

**Config auto-detection:**
- `test_config_detection_48k`: mock checkpoint with 48k config → correct upsample rates
- `test_config_detection_40k`: mock checkpoint with 40k config → correct upsample rates

### Level 2 (@pytest.mark.slow, requires RVC checkpoint)

**Numerical correctness:**
- `test_output_matches_pytorch`: Load real RVC v2 checkpoint, extract intermediate latents + F0 from PyTorch run, feed same inputs through MAX, compare output waveform. Tolerance: absolute max element-wise difference < 0.01.

### Class-scoped fixtures

Follow the Sprint 2 pattern: compile MAX graph once per config, reuse across tests. One fixture per sample rate tested.

## Definition of Done

- `NSFHiFiGAN.from_pretrained("path/to/rvc_v2.pth")` loads 32k/40k/48k checkpoints
- `vocoder.synthesize(latents, f0)` returns correctly shaped `[B, T_audio]` waveform
- All Level 1 tests pass (shape, NaN, config detection)
- Numerical correctness vs PyTorch within tolerance on at least one real checkpoint
- Batch>1 support from day one

## Unchanged

- `AudioEncoder` — no changes
- `PitchExtractor` — no changes
- Mojo DSP layer — no changes
- Weight loading for other models — no changes
