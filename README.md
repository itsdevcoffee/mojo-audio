# mojo-audio

High-performance audio DSP and ML inference library for voice conversion — built in Mojo and Python, runs on NVIDIA DGX Spark ARM64 with zero PyTorch CUDA dependency.

[![Mojo](https://img.shields.io/badge/Mojo-0.26.1-orange?logo=fire)](https://docs.modular.com/mojo/)
[![License](https://img.shields.io/badge/license-MIT-blue.svg)](LICENSE)
[![Performance](https://img.shields.io/badge/vs_librosa-20--40%25_faster-brightgreen)](benchmarks/)

---

## What it is

mojo-audio has two layers:

**DSP layer (Mojo)** — low-level audio processing: FFT, mel spectrogram, resampling, VAD, pitch shifting, iSTFT. 20–40% faster than librosa through SIMD vectorization and multi-core parallelization.

**ML inference layer (Python + MAX Graph)** — GPU-accelerated neural network inference without PyTorch CUDA. Runs natively on DGX Spark SM_121 ARM64 via [MAX Engine](https://www.modular.com/max):

| Model | Purpose | Backend |
|-------|---------|---------|
| `AudioEncoder` | HuBERT / ContentVec content features | MAX Graph GPU |
| `PitchExtractor` | RMVPE pitch (F0) estimation | MAX Graph GPU + numpy BiGRU |

Together these form the core of a voice conversion pipeline (content extraction → pitch extraction → synthesis) that runs fully on Spark without cloud or PyTorch.

---

## ML Inference

### AudioEncoder — HuBERT / ContentVec

Extracts content feature vectors from raw audio. Supports `facebook/hubert-base-ls960` and `lengyue233/content-vec-best`. Automatically uses GPU if available.

```python
from models import AudioEncoder

model = AudioEncoder.from_pretrained("facebook/hubert-base-ls960")
features = model.encode(audio_np)  # [1, N] float32 @16kHz → [1, T, 768]
```

GPU pipeline: CNN feature extractor + positional conv (numpy, avoids MAX conv2d groups bug) + 12× transformer blocks.

### PitchExtractor — RMVPE

Extracts F0 (fundamental frequency) per 10ms frame. No PyTorch CUDA needed — runs on DGX Spark ARM64.

```python
from models import PitchExtractor

model = PitchExtractor.from_pretrained()  # downloads lj1995/VoiceConversionWebUI/rmvpe.pt
f0_hz = model.extract(audio_np)  # [1, N] float32 @16kHz → [T] float32 Hz, 0=unvoiced
```

Architecture: U-Net MAX Graph (5-level encoder + bottleneck + 5-level decoder) → numpy BiGRU → pitch salience bins → Hz per frame.

### Running the models

```bash
# Fast tests (no download)
pixi run test-models
pixi run test-pitch-extractor

# Full correctness tests (downloads model weights ~180–360MB)
pixi run test-models-full
pixi run test-pitch-extractor-full

# GPU benchmark
pixi run bench-models
```

---

## DSP Layer

### Mel Spectrogram (Mojo)

Whisper-compatible mel spectrogram preprocessing — 20–40% faster than librosa.

```mojo
from audio import mel_spectrogram

var mel = mel_spectrogram(audio)  // (80, 2998) for 30s @16kHz, ~12ms with -O3
```

**Performance:**
```
30-second audio @16kHz:

librosa (Python):   15ms  (1993x realtime)
mojo-audio (-O3):   12ms  (2457x realtime)  ← 20–40% faster
```

Optimization journey: 476ms (naive) → 12ms (-O3) = **40x total speedup** through iterative FFT, RFFT, twiddle caching, sparse mel filterbank, SIMD float32, radix-4 butterflies, and multi-core parallelization.

### Other DSP components

| Module | What it does |
|--------|-------------|
| `resample.mojo` | Lanczos resampler (48kHz → 16kHz) |
| `vad.mojo` | Voice activity detection / silence trimming |
| `pitch.mojo` | Phase vocoder pitch shifting |
| `wav_io.mojo` | WAV file I/O |
| `ffi/` | C-compatible shared library (`libmojo_audio.so`) |

### Running DSP tests

```bash
# All Mojo DSP tests
pixi run test

# Individual
pixi run test-pitch
pixi run bench-optimized   # mel spectrogram benchmark
pixi run bench-python      # librosa baseline comparison
```

---

## Installation

**Requirements:** [pixi](https://pixi.sh), Mojo 0.26+, Linux x86_64 or aarch64

```bash
git clone https://github.com/itsdevcoffee/mojo-audio.git
cd mojo-audio
pixi install
```

**Build FFI shared library** (for C/Rust/Python DSP integration):
```bash
pixi run build-ffi-optimized   # → libmojo_audio.so (Linux) or .dylib (macOS)
```

See [macOS Build Guide](docs/guides/02-04-2026-macos-build-guide.md) for macOS-specific setup.

---

## Project Structure

```
mojo-audio/
├── src/
│   ├── audio.mojo              # Mel spectrogram, FFT, STFT, windowing
│   ├── pitch.mojo              # Phase vocoder pitch shifting
│   ├── resample.mojo           # Lanczos resampler
│   ├── vad.mojo                # Voice activity detection
│   ├── wav_io.mojo             # WAV I/O
│   ├── ffi/                    # C-compatible shared library exports
│   └── models/                 # MAX Graph ML inference (Python)
│       ├── audio_encoder.py    # HuBERT / ContentVec via MAX Graph
│       ├── pitch_extractor.py  # RMVPE pitch extraction via MAX Graph
│       ├── _rmvpe.py           # U-Net graph + numpy BiGRU
│       ├── _rmvpe_weight_loader.py
│       └── _weight_loader.py   # HuBERT/ContentVec weight loader
├── tests/
│   ├── test_audio_encoder.py   # AudioEncoder tests (pytest)
│   ├── test_pitch_extractor.py # PitchExtractor tests (pytest)
│   ├── test_fft.mojo           # FFT correctness
│   ├── test_mel.mojo           # Mel spectrogram
│   └── ...                     # Other Mojo DSP tests
├── experiments/
│   ├── hubert-max/             # HuBERT MAX Graph experiments
│   ├── contentvec-max/         # ContentVec benchmarks
│   └── max-bug-repro/          # MAX Engine bug reproductions
├── docs/
│   ├── plans/                  # Implementation plans
│   ├── context/                # Architecture reference
│   └── project/                # Roadmap
└── pixi.toml
```

---

## Platform Support

| Platform | DSP | ML Inference |
|----------|-----|-------------|
| Linux x86_64 (NVIDIA RTX) | ✅ | ✅ GPU |
| Linux aarch64 (DGX Spark SM_121) | ✅ | ✅ GPU |
| macOS Apple Silicon | ✅ | ✅ CPU |
| macOS Intel | ✅ | ✅ CPU |

---

## Roadmap

The next steps are tracked in [docs/project/03-06-2026-roadmap.md](docs/project/03-06-2026-roadmap.md):

- **Sprint 2:** Full GPU AudioEncoder (remove numpy bridge once MAX conv2d groups bug is fixed), phase-locked phase vocoder
- **Sprint 3:** HiFiGAN vocoder in MAX Graph
- **Sprint 4:** Full VITS synthesis — end-to-end voice conversion on Spark
- **Sprint 5:** Shade integration and demo

---

## Comparison

| Feature | mojo-audio | librosa | torchaudio | RVC / Applio | pyworld |
|---------|:----------:|:-------:|:----------:|:------------:|:-------:|
| **DSP** | | | | | |
| Mel spectrogram | ✅ | ✅ | ✅ | via librosa | ❌ |
| FFT / STFT | ✅ | ✅ | ✅ | via librosa | partial |
| Resampling | ✅ | ✅ | ✅ | via librosa | ❌ |
| Voice activity detection | ✅ | ❌ | ❌ | via silero | ❌ |
| Phase vocoder pitch shift | ✅ | ✅ | ❌ | ✅ | ❌ |
| iSTFT / Griffin-Lim | ✅ | ✅ | ✅ | ❌ | ❌ |
| WAV I/O | ✅ | ✅ | ✅ | ✅ | ❌ |
| C FFI / shared library | ✅ | ❌ | ❌ | ❌ | ❌ |
| **ML Inference** | | | | | |
| HuBERT content features | ✅ MAX Graph | ❌ | ❌ | ✅ PyTorch | ❌ |
| ContentVec content features | ✅ MAX Graph | ❌ | ❌ | ✅ PyTorch | ❌ |
| RMVPE pitch extraction | ✅ MAX Graph | ❌ | ❌ | ✅ PyTorch | ❌ |
| WORLD pitch extraction | ❌ | ❌ | ❌ | via pyworld | ✅ |
| GPU inference | ✅ MAX Engine | ❌ | ✅ CUDA | ✅ CUDA | ❌ |
| **Platform** | | | | | |
| Linux x86_64 | ✅ | ✅ | ✅ | ✅ | ✅ |
| DGX Spark ARM64 | ✅ | ✅ | ❌ | ❌ | ❌ |
| macOS Apple Silicon | ✅ | ✅ | ✅ | partial | ✅ |
| PyTorch CUDA required | ❌ | ❌ | ✅ | ✅ | ❌ |
| **Performance** | | | | | |
| Mel spec vs librosa | **+20–40%** | baseline | ~parity | baseline | — |
| GPU inference without CUDA | ✅ | ❌ | ❌ | ❌ | ❌ |

---

## Known Issues

**MAX Engine conv2d groups bug (v26.1):** `ops.conv2d` returns incorrect results when `groups > 1` and kernel size is large (K≥128). Filed as [modular/modular#6129](https://github.com/modular/modular/issues/6129). Workaround: HuBERT's `pos_conv` layer runs outside the MAX Graph via numpy.

---

## Citation

```bibtex
@software{mojo_audio_2026,
  author = {Dev Coffee},
  title = {mojo-audio: Audio DSP and ML inference for voice conversion},
  year = {2026},
  url = {https://github.com/itsdevcoffee/mojo-audio}
}
```

---

**[GitHub](https://github.com/itsdevcoffee/mojo-audio)** | **[Issues](https://github.com/itsdevcoffee/mojo-audio/issues)** | **[Roadmap](docs/project/03-06-2026-roadmap.md)**
