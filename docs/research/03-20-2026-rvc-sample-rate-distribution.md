# RVC Model Sample Rate Distribution in the Wild

**Date:** 2026-03-20
**Status:** Research snapshot

## Executive Summary

**40kHz is the dominant sample rate for RVC models in the wild**, especially for models trained between 2022-2025. This is because:
1. RVC v2 initially shipped with only 40k pretrained models
2. 40k was the default if unchanged in the training UI
3. The most popular community pretrains (Ov2Super, RIN_E3) only support 40k
4. 48k pretrained models for v2 came later and required users to opt in

For a decoder that wants to support "most RVC models," **40kHz is the must-have target**. 48kHz is a nice-to-have for newer models.

## Detailed Findings

### 1. Available Sample Rates

RVC supports three sample rates: **32k, 40k, and 48k**. These are NOT the standard audio rates (44.1k, 48k) -- they are model-internal rates chosen to align with the HiFi-GAN upsampling architecture.

| Version | 32k config | 40k config | 48k config |
|---------|-----------|-----------|-----------|
| v1      | Yes       | Yes       | Yes       |
| v2      | Yes       | Yes (added later) | Yes (added later) |

RVC v2 initially shipped with **only** 40k pretrained weights. 32k and 48k pretrains for v2 were added months later.

### 2. Why 40kHz Instead of 44.1kHz?

The choice of 40k comes from HiFi-GAN's upsampling architecture. The generator upsamples from a latent at hop_length intervals to the target sample rate. The product of `upsample_rates` must equal `hop_length`:

| Rate | hop_length | upsample_rates | Product |
|------|-----------|----------------|---------|
| 32k  | 320       | [10, 8, 2, 2]  | 320     |
| 40k  | 400       | [10, 10, 2, 2] | 400     |
| 48k  | 480       | [12, 10, 2, 2] | 480     |

Using 44.1k would require `hop_length=441` which factors as `3 * 3 * 7 * 7` -- awkward for upsampling layers. 40k with `hop_length=400 = 10*10*2*2` is much cleaner. This is a purely practical architectural choice, not a quality decision.

### 3. Architectural Differences Between Sample Rates

The sample rate is **a config change, not an architectural change**. The model structure (ResBlocks, number of layers, hidden dims) stays identical. Only these parameters differ:

| Parameter | 32k | 40k | 48k |
|-----------|-----|-----|-----|
| sampling_rate | 32000 | 40000 | 48000 |
| hop_length | 320 | 400 | 480 |
| filter_length | 1024 | 2048 | 2048 |
| win_length | 1024 | 2048 | 2048 |
| n_mel_channels | 80 | 125 | 128 |
| upsample_rates | [10,8,2,2] | [10,10,2,2] | [12,10,2,2] |
| upsample_kernel_sizes | [20,16,4,4] | [16,16,4,4] | [24,20,4,4] |
| segment_size | 12800 | 12800 | 17280 |

Key observations:
- `inter_channels`, `hidden_channels`, `filter_channels` are all **identical** (192, 192, 768)
- `resblock_kernel_sizes` identical: [3, 7, 11]
- `upsample_initial_channel` identical: 512
- The number of upsampling layers is always 4
- Only the upsample rates and kernel sizes change (to hit the target sample rate)
- `n_mel_channels` differs: 80 (32k), 125 (40k), 128 (48k) -- this affects the mel spectrogram input shape

**Important:** Because `n_mel_channels` differs, weights are NOT interchangeable between sample rates. The first conv layer expects a different input dimension. But the overall architecture class is the same.

### 4. Distribution of Models in the Wild

#### Community Pretrained Models (Applio ecosystem, 2024-2026)

| Pretrained | 32k | 40k | 48k |
|-----------|-----|-----|-----|
| Ov2 Super | Yes | **Yes** | No (planned) |
| RIN_E3 | No | **Yes** | No |
| TITAN | Yes | **Yes** | Yes |
| SnowieV3.1 | Yes | **Yes** | Yes |
| KLM 4.1 | Yes | No | Yes |
| DMR V1 | Yes | No | No |
| Nanashi V1.7 | Yes | No | No |
| SingerPreTrain | Yes | No | No |

**40k is supported by 5/8 popular pretrains.** The two most popular community pretrains (Ov2Super and RIN_E3) support **only** 40k (not 48k).

#### Estimated Distribution of User-Trained Models

Based on defaults, pretrain availability, and community guidance:

| Sample Rate | Estimated Share | Reasoning |
|-------------|----------------|-----------|
| **40kHz** | **~60-70%** | Default in RVC v2, only option for Ov2Super/RIN_E3, recommended for 44.1k source audio |
| **48kHz** | **~20-30%** | Available in TITAN/Snowie, used by users with 48k source audio, growing share |
| **32kHz** | **~5-10%** | Used for low-quality source audio (below 15kHz bandwidth), niche |

#### Historical Timeline

- **2023 Q1-Q2 (RVC v1):** All three rates available (32k, 40k, 48k). 40k was common default.
- **2023 Q2-Q3 (RVC v2 launch):** v2 shipped with **only 40k pretrains**. This locked the vast majority of v2 early adopters into 40k.
- **2023 Q3+ (v2 updates):** 32k and 48k pretrains added for v2, but 40k remained the default and most-recommended.
- **2024-2025 (Applio era):** Community pretrains (Ov2Super, TITAN) emerged. Ov2Super (very popular) supports only 32k/40k. TITAN supports all three.
- **2025-2026:** 48k gaining ground with newer pretrains, but 40k legacy is massive.

### 5. Answer to Specific Questions

**What percentage of models in the wild are 40kHz vs 48kHz vs 32kHz?**
Roughly 60-70% are 40k, 20-30% are 48k, 5-10% are 32k. No formal census exists, but this is based on: default settings, pretrain availability, community recommendations, and the fact that v2 launched 40k-only.

**Was 40kHz the default in earlier versions of RVC?**
Yes. RVC v2 launched with only 40k pretrained models. The training UI defaulted to 40k. Community guides from 2023 consistently recommend 40k.

**If someone has been training models since 2022, what sample rate would most of their models be?**
Almost certainly 40kHz. RVC was released in early 2023 (not 2022). From v1 through v2, 40k was the path of least resistance -- it was the default, had the most pretrain support, and was explicitly recommended for the common case of 44.1kHz source audio.

**Is sample rate a simple config change or does it require architectural changes?**
It is a **config-level change only**. The model class is identical. The differences are:
- `upsample_rates` (determines the product of upsampling factors)
- `upsample_kernel_sizes` (must be >= 2x the corresponding upsample rate)
- `hop_length` (= product of upsample_rates)
- `n_mel_channels` (80/125/128 -- affects input conv shape)

The weights are not transferable between sample rates because `n_mel_channels` differs, but a single model implementation can handle all three rates if parameterized by config.

## Implications for mojo-audio

1. **Must support 40kHz** -- this covers the majority of models in the wild
2. **Should support 48kHz** -- growing share, easy to add since it is just different upsample rates
3. **32kHz is low priority** -- niche use case
4. The decoder/generator can be a single parameterized implementation that takes `upsample_rates`, `upsample_kernel_sizes`, and `n_mel_channels` from config
5. Sample rate detection can be inferred from the model weights: check the shape of the first conv layer to determine `n_mel_channels`, then map to sample rate

## Sources

- [RVC GitHub Issue #1565 - Why 40k and 48k instead of 44.1k](https://github.com/RVC-Project/Retrieval-based-Voice-Conversion-WebUI/issues/1565)
- [RVC GitHub Issue #514 - Question about sample rates](https://github.com/RVC-Project/Retrieval-based-Voice-Conversion-WebUI/issues/514)
- [RVC GitHub Issue #233 - Standard for most recorded music is 44.1k](https://github.com/RVC-Project/Retrieval-based-Voice-Conversion-WebUI/issues/233)
- [RVC v2/48k.json config](https://github.com/RVC-Project/Retrieval-based-Voice-Conversion-WebUI/blob/main/configs/v2/48k.json)
- [RVC v2/32k.json config](https://github.com/RVC-Project/Retrieval-based-Voice-Conversion-WebUI/blob/main/configs/v2/32k.json)
- [RVC v1/40k.json config](https://huggingface.co/spaces/oItsMineZ/RVC-v2-WebUI/blob/57f428721851d4cd858c893b5cf5e847a25cfab0/configs/40k.json)
- [Applio Pretrained Models Documentation](https://docs.applio.org/getting-started/pretrained/)
- [TITAN pretrained model](https://huggingface.co/blaise-tk/TITAN)
- [RVC GitHub Issue #1626 - v1 vs v2 config structure](https://github.com/RVC-Project/Retrieval-based-Voice-Conversion-WebUI/issues/1626)
- [Applio GitHub Issue #1117 - Config file improvements](https://github.com/IAHispano/Applio/issues/1117)
- [RunPod RVC Deployment Guide](https://www.runpod.io/articles/guides/ai-engineer-guide-rvc-cloud)
- [AI Hub Docs - RVC Mainline](https://ai-hub-docs.vercel.app/rvc/local/mainline/)
