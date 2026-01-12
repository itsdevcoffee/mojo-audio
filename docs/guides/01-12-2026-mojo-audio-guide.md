# mojo-audio: Complete Guide

**High-performance audio DSP library in Mojo** — 20-40% faster than Python's librosa.

**Status:** Production-ready | **Version:** 0.1.0 | **Mojo:** 0.26.1+

---

## What is mojo-audio?

A library for transforming raw audio into mathematical representations that:
- Machines can process (ML models, neural networks)
- Preserve perceptual qualities humans care about
- Run fast enough for real-time applications

**Primary output:** Mel spectrograms — the standard input format for speech models like OpenAI's Whisper.

---

## Audio Fundamentals for Beginners

### What is Audio Data?

Raw audio is a sequence of **samples** — numbers representing air pressure at tiny time intervals.

```
Sample rate: 16,000 Hz = 16,000 measurements per second
30 seconds of audio = 480,000 numbers (Float32 values between -1.0 and 1.0)
```

**Problem:** Raw samples are useless for ML. A 30-second clip is 480,000 numbers with no obvious patterns. We need to transform this into something meaningful.

### The Audio → ML Pipeline

```
Raw Audio (time domain)
    ↓ Window + FFT
Power Spectrum (frequency domain)
    ↓ Repeat for overlapping frames
Spectrogram (time × frequency)
    ↓ Apply mel filterbank
Mel Spectrogram (time × perceptual frequency)
    ↓ Log scale
Final Features (80 × 3000 matrix for 30s audio)
```

---

## Core Concepts Explained

### 1. Fourier Transform (FFT)

**What it does:** Converts a signal from *time domain* to *frequency domain*.

```
Time domain:  "The sound wave wiggles like this over time"
Frequency domain:  "This sound contains 440 Hz (A note) + 880 Hz (octave above)"
```

**Why it matters:** Frequency content reveals *what* sounds are present, not just *when* they happen.

**In code:**
```mojo
from audio import fft, rfft

# Full FFT: N samples → N complex frequency bins
spectrum = fft(audio_samples)

# Real FFT: Optimized for real signals (1.4x faster)
spectrum = rfft(audio_samples)  # Returns N/2+1 bins (positive frequencies only)
```

### 2. Short-Time Fourier Transform (STFT)

**Problem:** A single FFT loses all timing information. We know *what* frequencies exist but not *when* they occur.

**Solution:** Divide audio into overlapping **frames**, compute FFT for each.

```
30s audio → 3000 overlapping frames (10ms apart)
Each frame → 201 frequency bins
Result: 201 × 3000 spectrogram matrix
```

**Parameters:**
| Name | Value | Meaning |
|------|-------|---------|
| n_fft | 400 | Samples per frame (25ms at 16kHz) |
| hop_length | 160 | Samples between frame starts (10ms) |
| window | Hann | Smooth tapering to reduce artifacts |

**In code:**
```mojo
from audio import stft

# Returns power spectrogram: (n_fft/2+1, n_frames)
spectrogram = stft(audio, n_fft=400, hop_length=160)
# Shape: (201, ~3000) for 30s audio
```

### 3. Window Functions

**Problem:** Chopping audio into frames creates harsh edges that produce false frequencies (spectral leakage).

**Solution:** Multiply each frame by a **window function** that tapers smoothly to zero at the edges.

```
Hann window:  Tapers to exactly 0 at edges
              Best for general spectral analysis

Hamming window:  Minimum ~0.08 (never reaches 0)
                 Better frequency selectivity
```

**Visual:**
```
Hann:     ___/‾‾‾‾‾‾‾\___    (smooth bell, touches zero)
Hamming:  __/‾‾‾‾‾‾‾‾‾\__    (flatter top, doesn't touch zero)
```

**In code:**
```mojo
from audio import hann_window, hamming_window, apply_window_simd

var window = hann_window(400)
var windowed_frame = apply_window_simd(frame, window)
```

### 4. Mel Scale

**Problem:** Humans don't hear frequencies linearly. The difference between 100 Hz and 200 Hz sounds huge; 8000 Hz and 8100 Hz sounds tiny.

**Solution:** The **mel scale** — a perceptual frequency scale matching human hearing.

```
Linear Hz:  100  200  400  800  1600  3200  6400
Mel scale:  More resolution here → ← Less resolution here
```

**Formula:**
```
mel = 2595 × log₁₀(1 + hz/700)
hz = 700 × (10^(mel/2595) - 1)
```

**In code:**
```mojo
from audio import hz_to_mel, mel_to_hz

var mel = hz_to_mel(1000.0)   # ≈ 1903.96
var hz = mel_to_hz(1903.96)   # ≈ 1000.0
```

### 5. Mel Filterbank

A set of **triangular filters** spaced evenly on the mel scale, used to compress the frequency axis.

```
Standard spectrogram: 201 frequency bins (linear Hz spacing)
After mel filterbank: 80 mel bands (perceptual spacing)
```

**How it works:**
1. Create 80 triangular filters spanning 0–8000 Hz
2. Each filter peaks at one mel frequency, overlaps neighbors
3. Multiply spectrogram by filterbank → weighted sum per band

**In code:**
```mojo
from audio import create_mel_filterbank, apply_mel_filterbank

# Shape: (80, 201) — 80 filters × 201 frequency bins
var filterbank = create_mel_filterbank(n_mels=80, n_fft=400, sample_rate=16000)

# Apply to spectrogram: (201, 3000) → (80, 3000)
var mel_spec = apply_mel_filterbank(spectrogram, filterbank)
```

### 6. Log Scaling

**Problem:** Audio energy varies by orders of magnitude. Quiet sounds get lost.

**Solution:** Take logarithm to compress dynamic range (like decibels).

```
Before log:  [0.0001, 0.001, 0.01, 0.1, 1.0, 10.0]
After log:   [-4, -3, -2, -1, 0, 1]  (much more uniform)
```

This is applied in the final `mel_spectrogram()` function automatically.

---

## Complete Pipeline: mel_spectrogram()

The main function combining all steps:

```mojo
from audio import mel_spectrogram

fn main() raises:
    # Load audio: 30s at 16kHz = 480,000 samples
    var audio = List[Float32]()
    # ... populate with audio samples ...

    # Compute mel spectrogram (all defaults are Whisper-compatible)
    var mel_spec = mel_spectrogram(
        audio,
        sample_rate=16000,
        n_fft=400,
        hop_length=160,
        n_mels=80
    )

    # Result: (80, ~3000) matrix ready for ML models
```

**What happens inside:**
1. **STFT** → Power spectrogram (201 × 3000)
2. **Mel filterbank** → Compress to (80 × 3000)
3. **Log scale** → Compress dynamic range

**Performance:** 12ms for 30s audio (2457× realtime)

---

## FFI: Using from C/Python/Rust

The library exports a C API for cross-language integration.

### Build the Shared Library

```bash
pixi run build-ffi-optimized  # Creates libmojo_audio.so
```

### C Example

```c
#include "mojo_audio.h"

int main() {
    MojoMelConfig config;
    mojo_mel_config_default(&config);

    float audio[480000] = { /* your samples */ };

    int64_t handle = mojo_mel_spectrogram_compute(audio, 480000, &config);
    if (handle <= 0) return 1;  // Error

    size_t n_mels, n_frames;
    mojo_mel_spectrogram_get_shape(handle, &n_mels, &n_frames);

    float* output = malloc(n_mels * n_frames * sizeof(float));
    mojo_mel_spectrogram_get_data(handle, output, n_mels * n_frames);

    // Use output...

    free(output);
    mojo_mel_spectrogram_free(handle);
}
```

### Python Example (ctypes)

```python
import ctypes
import numpy as np

lib = ctypes.CDLL('./libmojo_audio.so')

class MojoMelConfig(ctypes.Structure):
    _fields_ = [
        ('sample_rate', ctypes.c_int32),
        ('n_fft', ctypes.c_int32),
        ('hop_length', ctypes.c_int32),
        ('n_mels', ctypes.c_int32)
    ]

config = MojoMelConfig()
lib.mojo_mel_config_default(ctypes.byref(config))

audio = np.random.randn(480000).astype(np.float32)
handle = lib.mojo_mel_spectrogram_compute(
    audio.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
    len(audio),
    ctypes.byref(config)
)

n_mels, n_frames = ctypes.c_uint64(), ctypes.c_uint64()
lib.mojo_mel_spectrogram_get_shape(handle, ctypes.byref(n_mels), ctypes.byref(n_frames))

output = np.zeros(n_mels.value * n_frames.value, dtype=np.float32)
lib.mojo_mel_spectrogram_get_data(
    handle,
    output.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
    len(output)
)

mel_spec = output.reshape(n_mels.value, n_frames.value)
lib.mojo_mel_spectrogram_free(handle)
```

---

## API Reference

### Window Functions

| Function | Description |
|----------|-------------|
| `hann_window(size)` | Creates Hann window (tapers to 0) |
| `hamming_window(size)` | Creates Hamming window (min ~0.08) |
| `apply_window_simd(signal, window)` | SIMD-optimized element-wise multiply |

### FFT Operations

| Function | Description |
|----------|-------------|
| `fft(signal)` | Full complex FFT, auto-pads to power of 2 |
| `rfft(signal)` | Real FFT (1.4× faster), returns N/2+1 bins |
| `power_spectrum(fft_output)` | Convert complex → power (real² + imag²) |
| `stft(signal, n_fft, hop_length, window_fn)` | Short-time Fourier transform |

### Mel Operations

| Function | Description |
|----------|-------------|
| `hz_to_mel(freq_hz)` | Convert Hz → mel scale |
| `mel_to_hz(freq_mel)` | Convert mel → Hz |
| `create_mel_filterbank(n_mels, n_fft, sample_rate)` | Create triangular filterbank |
| `apply_mel_filterbank(spectrogram, filterbank)` | Apply filterbank to spectrogram |
| `mel_spectrogram(audio, ...)` | Complete pipeline (STFT → mel → log) |

### Utilities

| Function | Description |
|----------|-------------|
| `pad_to_length(signal, target)` | Zero-pad signal |
| `rms_energy(signal)` | Root mean square energy |
| `normalize_audio(signal)` | Normalize to [-1.0, 1.0] |

### Normalization

| Function | Description |
|----------|-------------|
| `normalize_whisper(mel_spec)` | Whisper normalization: max-8 clamp + (x+4)/4 |
| `normalize_minmax(mel_spec)` | Scale to [0, 1] |
| `normalize_zscore(mel_spec)` | Z-score: (x - mean) / std |
| `apply_normalization(mel_spec, norm_type)` | Apply normalization by constant |

| Constant | Value | Description |
|----------|-------|-------------|
| `NORM_NONE` | 0 | Raw log mels [-10, 0] |
| `NORM_WHISPER` | 1 | Whisper normalization ~[-1, 1] |
| `NORM_MINMAX` | 2 | Min-max to [0, 1] |
| `NORM_ZSCORE` | 3 | Z-score ~[-3, 3] |

---

## Whisper Compatibility

All defaults match OpenAI Whisper's preprocessing:

| Parameter | Value | Notes |
|-----------|-------|-------|
| sample_rate | 16000 | 16 kHz mono |
| n_fft | 400 | 25ms frames |
| hop_length | 160 | 10ms hops |
| n_mels | 80 or 128 | Depends on model version |
| window | Hann | Smooth tapering |
| output | log mel | Log-scaled energy |

### Mel Bins by Model Version

| Whisper Model | n_mels | Constant |
|---------------|--------|----------|
| tiny, base, small, medium, large, large-v2 | 80 | `WHISPER_N_MELS` |
| large-v3 | 128 | `WHISPER_N_MELS_V3` |

```mojo
// Default: 80 mel bins (v2 and earlier)
var mel = mel_spectrogram(audio)

// For large-v3: 128 mel bins
var mel = mel_spectrogram(audio, n_mels=128)
```

**30-second audio produces (n_mels, ~3000) output** — directly usable as Whisper input.

---

## Normalization

Different ML models expect different input value ranges. Raw log mel values are in [-10, 0], but models like Whisper expect normalized values.

### Available Methods

| Method | Constant | Output Range | Formula |
|--------|----------|--------------|---------|
| None (raw) | `NORM_NONE` (0) | [-10, 0] | `log10(max(x, 1e-10))` |
| Whisper | `NORM_WHISPER` (1) | ~[-1, 1] | clamp to max-8, then `(x+4)/4` |
| Min-Max | `NORM_MINMAX` (2) | [0, 1] | `(x - min) / (max - min)` |
| Z-Score | `NORM_ZSCORE` (3) | ~[-3, 3] | `(x - mean) / std` |

### Why Whisper Normalization?

OpenAI Whisper applies specific normalization:
1. **80dB dynamic range clamp**: `max(log_spec, max - 8.0)` — limits the range to 80dB below peak
2. **Scale to [-1, 1]**: `(x + 4.0) / 4.0` — centers and scales for neural network input

Without this, Whisper may output EOT immediately (thinks there's no speech).

### Usage in Pure Mojo

```mojo
from audio import mel_spectrogram, normalize_whisper, apply_normalization, NORM_WHISPER

// Option 1: Compute then normalize separately
var raw_mel = mel_spectrogram(audio)
var whisper_mel = normalize_whisper(raw_mel)

// Option 2: Use apply_normalization with constant
var normalized = apply_normalization(raw_mel, NORM_WHISPER)
```

### Usage via FFI

Set the `normalization` field in config:

```c
MojoMelConfig config;
mojo_mel_config_default(&config);
config.normalization = MOJO_NORM_WHISPER;  // Apply Whisper normalization

MojoMelHandle handle = mojo_mel_spectrogram_compute(audio, num_samples, &config);
// Result is already normalized, ready for Whisper
```

```rust
let config = MojoMelConfig {
    sample_rate: 16000,
    n_fft: 400,
    hop_length: 160,
    n_mels: 128,           // Whisper large-v3
    normalization: 1,      // MOJO_NORM_WHISPER
};
```

### Choosing a Normalization

| Model | Recommended |
|-------|-------------|
| Whisper (any version) | `NORM_WHISPER` |
| Wav2Vec2 | `NORM_ZSCORE` |
| Custom/Research | `NORM_NONE` (normalize yourself) |
| General classification | `NORM_MINMAX` or `NORM_ZSCORE` |

---

## Performance

### Benchmark Results

| Implementation | Time (30s audio) | Realtime Factor |
|----------------|------------------|-----------------|
| mojo-audio (-O3) | 12ms | 2457× |
| Python librosa | 15ms | 1993× |
| **Speedup** | **20-40%** | |

### Key Optimizations

1. **TRUE Real FFT** — Exploits conjugate symmetry (1.4× faster)
2. **Radix-4 FFT** — 4-point butterflies for power-of-4 sizes
3. **Twiddle caching** — Pre-compute once, reuse 3000× (2× speedup)
4. **Sparse filterbank** — Skip zero weights (~1.24× speedup)
5. **SIMD vectorization** — 16 Float32 elements per operation
6. **Multi-core parallelization** — Independent frame processing

---

## Running Examples

```bash
# Full mel spectrogram demo
pixi run demo

# Individual demonstrations
pixi run demo-window   # Window functions
pixi run demo-fft      # FFT and STFT
pixi run demo-mel      # Mel filterbank
```

---

## Running Tests

```bash
pixi run test          # All tests
pixi run test-window   # Window function tests
pixi run test-fft      # FFT operation tests
pixi run test-mel      # Mel filterbank tests
```

---

## When to Use mojo-audio

**Use when you need:**
- Mel spectrograms for speech/audio ML models
- Whisper-compatible preprocessing
- High-throughput audio processing
- C/Python/Rust integration via FFI
- Educational DSP examples in Mojo

**Don't use when you need:**
- Audio file I/O (use librosa/soundfile for loading)
- Audio playback/recording
- Advanced audio effects (reverb, compression)
- Music information retrieval (tempo, beat tracking)

---

## Glossary

| Term | Definition |
|------|------------|
| **Sample** | Single measurement of audio amplitude |
| **Sample rate** | Measurements per second (Hz) |
| **FFT** | Fast Fourier Transform — time → frequency |
| **STFT** | Short-Time FFT — spectrogram over time |
| **Spectrogram** | 2D time × frequency representation |
| **Mel scale** | Perceptual frequency scale |
| **Mel filterbank** | Triangular filters on mel scale |
| **Mel spectrogram** | Spectrogram compressed to mel bands |
| **Hop length** | Samples between STFT frame starts |
| **n_fft** | FFT window size in samples |
| **Window function** | Tapering function to reduce spectral leakage |
