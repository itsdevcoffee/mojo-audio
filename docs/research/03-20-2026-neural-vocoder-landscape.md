# Neural Vocoder Landscape — March 2026

Research snapshot: state of the art in neural vocoders relevant to RVC-style voice conversion pipelines.

## Executive Summary

**NSF-HiFiGAN remains the pragmatic best choice for an RVC voice conversion pipeline in March 2026.** No single vocoder has emerged that is clearly better *and* compatible with existing RVC/Applio pipelines. BigVGAN-v2 produces higher quality audio but lacks F0 conditioning, is 8x larger, and would require a full pipeline redesign. Vocos is dramatically faster but also lacks native F0 conditioning. The RVC/Applio community has explored RefineGAN at 44kHz but it requires new pretrains and is not yet a proven drop-in replacement.

The most interesting near-term upgrade path is **PC-NSF-HiFiGAN** (released Feb 2025 by the OpenVPI/DiffSinger community), which adds pitch-shifting ability while maintaining NSF-HiFiGAN's architecture and quality level.

---

## Ranked Comparison Table

| Rank | Vocoder | Quality (MOS/UTMOS) | Speed (xRT, GPU) | Params | F0/Pitch Cond. | RVC Compatible | Pretrained Avail. | Architecture |
|------|---------|-------------------|------------------|--------|----------------|----------------|-------------------|--------------|
| 1 | **NSF-HiFiGAN** | ~4.0 MOS (RVC context) | ~168x RT (V100) | ~14M | **Yes (NSF)** | **Yes (stock)** | Yes (RVC/Applio) | Conv upsampling + neural source-filter |
| 2 | **PC-NSF-HiFiGAN** | Same as NSF-HiFiGAN | Similar (~168x RT) | ~14M | **Yes (NSF + pitch shift)** | Partial (SVC pipelines) | Yes (openvpi) | NSF-HiFiGAN + MiniNSF module |
| 3 | **RefineGAN** | Claimed > ground truth | Unknown | Unknown | Unknown | Partial (Applio 3.6+, new pretrains) | Yes (Applio) | GAN-based, multi-scale |
| 4 | **BigVGAN-v2** | 3.75 UTMOS | 240x RT (A100, CUDA kernel) | 112M (large) / 14M (base) | **No** | **No** (no F0 conditioning) | Yes (NVIDIA HF) | Conv upsampling + Snake activation + anti-aliasing |
| 5 | **Vocos** | 3.73 UTMOS | 6697x RT (GPU) / 170x (CPU) | 13.5M | **No** | **No** (no F0 conditioning) | Yes (gemelo-ai) | Fourier-based, iSTFT output, no upsampling |
| 6 | **EVA-GAN** | High (no public MOS) | Unknown | 200M (full) / 13.6M (gen) | **No** | **No** | **No** (closed) | Context-aware blocks + upsample parallel resblocks |
| 7 | **WaveFM** | Competitive w/ GAN vocoders | Single-step inference | Unknown | **No** | **No** | Yes (GitHub) | Flow matching (diffusion-family) |
| 8 | **vec2wav 2.0** | High (VC-optimized) | Unknown | Unknown | No (uses discrete tokens) | **No** (different paradigm) | Yes (GitHub) | Discrete token vocoder + adaptive Snake |

### Codec-Based / LM-Based Approaches (Different Paradigm)

| System | Quality | Speed | Params | F0 Cond. | RVC Compat. | Notes |
|--------|---------|-------|--------|----------|-------------|-------|
| **WavTokenizer** | 3.88 UTMOS (12kbps) | Real-time capable | Unknown | No | No | ICLR 2025; 40-75 tokens/sec; VQ-GAN codec |
| **FACodec** | Good (disentangled) | Real-time capable | Unknown | Implicit | No | Decomposes speech into content/prosody/timbre subspaces |
| **SoundStream/EnCodec** | Baseline codec quality | Real-time | ~10-30M | No | No | Foundation for newer codec approaches |

---

## Detailed Notes Per Vocoder

### NSF-HiFiGAN (Current Choice)

- **Architecture**: Standard HiFiGAN generator with a Neural Source-Filter (NSF) module prepended. The NSF module generates a sine-based excitation signal from F0, which is then filtered by the dilated convolution network.
- **Why it works for RVC**: Takes F0 as explicit input, which is critical for voice conversion where pitch must be preserved/controlled. The source-filter decomposition prevents the "sound interruption" artifacts that plain HiFiGAN exhibits in VC.
- **Limitations**: Quality ceiling is lower than BigVGAN-v2 on general audio; high-frequency reconstruction can be blurry.
- **Pretrained**: Widely available via RVC, Applio, openvpi/vocoders, so-vits-svc.

### PC-NSF-HiFiGAN (Feb 2025, OpenVPI)

- **What's new**: Replaces the HN-NSF module with "MiniNSF" — lighter and faster. Gains pitch-shifting ability (-12 to +12 semitones) while preserving formants (like WORLD vocoder).
- **Config**: 44.1kHz, 128 mel bins, hop 512, window 2048.
- **Relevance**: Direct upgrade path from NSF-HiFiGAN for singing voice conversion. Could be adapted for RVC if someone builds the pretrain integration.

### BigVGAN-v2 (July 2024, NVIDIA)

- **Architecture**: HiFiGAN-style upsampling generator with periodic Snake activation functions and anti-aliased representation. Trained on 112M params with diverse 36k-hour dataset.
- **Key strength**: Universal vocoder — generalizes to unseen speakers, languages, music, environmental sounds without fine-tuning.
- **Key weakness for RVC**: No F0/pitch conditioning. It reconstructs from mel spectrogram only. For voice conversion you'd need to handle pitch preservation upstream or add an NSF-style module.
- **Speed**: With custom CUDA kernels, up to 240x real-time on A100. Without kernels, ~45x RT.
- **Verdict**: Best general-purpose vocoder, but not plug-compatible with RVC's F0-conditioned pipeline.

### Vocos (2023, Gemelo AI)

- **Architecture**: Purely frequency-domain. No transposed convolutions or upsampling layers. Generates STFT coefficients directly, then uses iSTFT. ConvNeXt-based backbone.
- **Key strength**: Extremely fast — 13x faster than HiFiGAN, 70x faster than BigVGAN. Only 13.5M params.
- **Key weakness for RVC**: No native F0 conditioning. Designed for mel-to-waveform or codec-to-waveform reconstruction.
- **MOS**: UTMOS 3.734 (vs BigVGAN's 3.749) — essentially tied on quality.
- **Verdict**: If you needed a fast mel-to-wav vocoder without pitch control, Vocos is the best choice. Not suitable as an RVC vocoder drop-in.

### EVA-GAN (Jan 2024, Douyin/TikTok)

- **Architecture**: Scaled-up HiFiGAN with Context Aware Blocks that extend the receptive field to ~3 seconds (vs 32-64 frames in HiFiGAN). 200M parameters total, 13.6M generator.
- **Key strength**: 44.1kHz native, excellent high-frequency and spectral continuity.
- **Key weakness**: No public pretrained weights. No F0 conditioning. Closed-source.
- **Verdict**: Interesting architecture ideas but unusable for our purposes.

### RefineGAN (Interspeech 2022, used in Applio 2025)

- **Claims**: "Generates waveform better than ground truth" with accurate pitch and intensity responses.
- **Status in Applio**: Available as an option in Applio 3.6+, but requires dedicated pretrained models (not compatible with stock RVC pretrains). Listed alongside MRF-HiFiGAN as experimental.
- **Community adoption**: Users are training with RefineGAN 44kHz in Applio, but it hasn't displaced HiFiGAN as the default. Limited benchmark data available.
- **Verdict**: Promising but unproven at scale. The Applio community is still experimenting.

### WaveFM (NAACL 2025)

- **Architecture**: Flow matching (diffusion family). Mel-conditioned prior distribution instead of Gaussian. Can generate in a single inference step via consistency distillation.
- **Relevance**: Represents the trend toward diffusion/flow-based vocoders. Quality is competitive with GAN vocoders but the single-step mode is key for practical speed.
- **Verdict**: Interesting research direction but not ready for RVC integration.

### vec2wav 2.0 (ICASSP 2025)

- **Architecture**: Discrete token vocoder for voice conversion. Uses WavLM features for timbre and an adaptive Snake activation.
- **Key insight**: Treats VC as a "prompted vocoding" task — the vocoder itself does the voice conversion, not a separate model.
- **Relevance**: Represents a different paradigm where the vocoder and VC model are unified. Could be the future but requires a completely different pipeline.

### Codec-Based Approaches (WavTokenizer, FACodec)

- **WavTokenizer**: ICLR 2025, achieves SOTA reconstruction with only 40-75 tokens per second. Designed for audio language models, not traditional VC.
- **FACodec**: Decomposes speech into content/prosody/timbre subspaces. Can do zero-shot VC by swapping speaker embeddings. Part of the Amphion toolkit.
- **Relevance**: These represent the "language model" paradigm for speech processing. They bypass the mel spectrogram entirely. Likely the long-term future but require a complete architectural shift away from RVC's design.

---

## Trends and Observations

1. **GAN vocoders remain dominant for production use.** Flow/diffusion vocoders are catching up but GANs still win on speed and simplicity.

2. **F0 conditioning is rare outside the SVC/RVC niche.** Most general-purpose vocoders (BigVGAN, Vocos, EVA-GAN) do not take pitch as input. The NSF-HiFiGAN family is specifically designed for the voice/singing conversion use case.

3. **The RVC community has not moved away from NSF-HiFiGAN.** RVC v2 still uses it. Applio's experiments with RefineGAN are promising but not yet standard. RVC v3 is anticipated but no confirmed vocoder change.

4. **Codec-based approaches are the long-term disruption.** WavTokenizer, FACodec, and vec2wav 2.0 suggest a future where discrete tokens replace mel spectrograms entirely. This would make traditional vocoders obsolete but requires rebuilding the entire pipeline.

5. **PC-NSF-HiFiGAN is the most relevant incremental upgrade** — same architecture family, adds pitch shifting, faster MiniNSF module.

---

## Recommendation for mojo-audio

**Short-term (current sprint):** Stay with NSF-HiFiGAN. It is the correct choice for an RVC-compatible voice conversion pipeline. No vocoder has clearly displaced it for this specific use case.

**Medium-term (if adding 44kHz support):** Evaluate PC-NSF-HiFiGAN from the OpenVPI project. Same architecture family, adds pitch-shifting, and the MiniNSF module is lighter weight (good for Mojo port efficiency).

**Long-term (future architecture):** Monitor vec2wav 2.0 and codec-based approaches. If the project ever moves beyond RVC compatibility toward a custom pipeline, these represent the next generation.

**What NOT to do:** Do not switch to BigVGAN or Vocos for the RVC pipeline. They lack F0 conditioning, which is non-negotiable for voice conversion quality.

---

## Sources

- [BigVGAN — NVIDIA Research](https://research.nvidia.com/labs/adlr/projects/bigvgan/)
- [BigVGAN GitHub (NVIDIA)](https://github.com/NVIDIA/BigVGAN)
- [BigVGAN-v2 on HuggingFace](https://huggingface.co/nvidia/bigvgan_v2_44khz_128band_512x)
- [Vocos GitHub (gemelo-ai)](https://github.com/gemelo-ai/vocos)
- [Vocos paper (arXiv)](https://arxiv.org/abs/2306.00814)
- [EVA-GAN paper (arXiv)](https://arxiv.org/abs/2402.00892)
- [RefineGAN paper (INTERSPEECH 2022)](https://www.isca-archive.org/interspeech_2022/xu22d_interspeech.pdf)
- [PC-NSF-HiFiGAN release (openvpi/vocoders)](https://github.com/openvpi/vocoders/releases/tag/pc-nsf-hifigan-44.1k-hop512-128bin-2025.02)
- [NSF-HiFiGAN GitHub](https://github.com/vtuber-plan/NSF-HiFiGAN)
- [WaveFM (NAACL 2025)](https://github.com/luotianze666/wavefm)
- [vec2wav 2.0 (arXiv)](https://arxiv.org/abs/2409.01995)
- [WavTokenizer GitHub (ICLR 2025)](https://github.com/jishengpeng/WavTokenizer)
- [FACodec / Amphion toolkit](https://github.com/open-mmlab/Amphion)
- [Applio releases](https://github.com/IAHispano/Applio/releases)
- [RVC-Project WebUI](https://github.com/RVC-Project/Retrieval-based-Voice-Conversion-WebUI)
- [Applio vocoder documentation (DeepWiki)](https://deepwiki.com/IAHispano/Applio/5.1-downloading-pre-trained-models)
- [Applio training resources (aihub.gg)](https://docs.aihub.gg/rvc/resources/training/)
- [Source-Filter HiFi-GAN paper (arXiv)](https://arxiv.org/abs/2210.15533)
- [Spiking Vocos (arXiv)](https://arxiv.org/html/2509.13049v1)
