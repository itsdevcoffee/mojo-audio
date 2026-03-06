# mojo-audio: Full Capabilities Map

> Reference doc for Nick and Drew. Shows everything the library currently does,
> how the pieces connect, and where it fits in the voice conversion pipeline.

---

## What mojo-audio is

Two layers in one repo:

- **Layer 1 — DSP Core (Mojo):** Compiled, SIMD-optimized audio processing primitives. Outputs a `.so` shared library callable from Python, Rust, or C.
- **Layer 2 — ML Inference (Python + MAX Graph):** HuBERT/ContentVec audio feature extraction running via Modular's MAX Engine. Runs on x86 CUDA and DGX Spark SM_121 ARM64.

The DSP layer handles everything before and after the neural network. The inference layer runs the neural network itself.

---

## Full Capability Tree

```
mojo-audio
│
├── DSP Core (Mojo → libmojo_audio.so)
│   │
│   ├── Audio I/O
│   │   └── WAV read/write (16-bit PCM, 32-bit, stereo→mono mixdown)
│   │
│   ├── Sample Rate Conversion
│   │   └── Lanczos sinc resampler
│   │       ├── 48kHz → 16kHz  (studio → HuBERT)
│   │       ├── 44.1kHz → 16kHz (CD → HuBERT)
│   │       ├── 16kHz → 48kHz  (HuBERT → DAW)
│   │       └── any rate → any rate (anti-aliased)
│   │
│   ├── Spectral Analysis
│   │   ├── Window functions (Hann, Hamming)
│   │   ├── FFT (Radix-2, Radix-4, Split-Radix, SIMD AVX-512)
│   │   ├── RFFT (Real FFT, half-spectrum)
│   │   ├── STFT (Short-Time Fourier Transform)
│   │   ├── Power spectrum
│   │   ├── Mel filterbank (configurable bands, Whisper-compatible)
│   │   └── Mel spectrogram (80 or 128 mel bands, used by Whisper + mojovoice)
│   │
│   ├── Waveform Reconstruction
│   │   ├── Inverse STFT (overlap-add synthesis)
│   │   └── Griffin-Lim (iterative phase estimation from magnitude spectrogram)
│   │
│   ├── Voice Activity Detection (VAD)
│   │   ├── RMS energy frame analysis
│   │   ├── Silence trimming (leading/trailing, with padding)
│   │   └── Voice segment detection (returns [start, end] pairs)
│   │
│   ├── Pitch Processing
│   │   └── Phase vocoder pitch shifter
│   │       ├── Semitone-accurate shift (±N semitones)
│   │       ├── Duration-preserving (output same length as input)
│   │       └── Based on OLA time-stretch + Lanczos resample
│   │
│   └── FFI Exports (C ABI, callable from Python/Rust/C)
│       ├── mojo_mel_spectrogram_compute  (used by mojovoice today)
│       ├── mojo_resample
│       └── mojo_trim_silence
│
└── ML Inference (Python + MAX Graph)
    │
    └── AudioEncoder (HuBERT / ContentVec)
        ├── Supported checkpoints
        │   ├── facebook/hubert-base-ls960  (general community use)
        │   └── lengyue233/content-vec-best (RVC/Shade voice conversion)
        │
        ├── Architecture (4 stages, all in MAX Graph)
        │   ├── Stage 1: CNN Feature Extractor  (7 Conv1D layers, GroupNorm, GELU)
        │   ├── Stage 2: Feature Projection     (Linear 512→768 + LayerNorm)
        │   ├── Stage 3: Position Embeddings    (Conv1D depthwise, GELU, residual)
        │   └── Stage 4: Transformer Encoder    (12 blocks, post-norm, MHA + FFN)
        │
        ├── Hardware targets
        │   ├── x86_64 + NVIDIA CUDA (RTX series) — GPU via MAX Engine
        │   ├── DGX Spark GB10 SM_121 ARM64      — GPU via MAX Engine *
        │   └── CPU-only fallback (any platform)
        │
        └── API
            ├── AudioEncoder.from_pretrained(model_id, device="auto")
            └── model.encode(audio_np)  →  [1, frames, 768]

* pos_conv currently runs on CPU bridge (MAX conv2d groups bug)
  — transformer and CNN stages run on GPU
```

---

## Full Pipeline Flow: Voice Conversion (Shade)

```
Raw Audio Input (WAV, 44.1kHz or 48kHz)
          │
          ▼
┌─────────────────────────────┐
│  mojo-audio DSP (Layer 1)   │
│                             │
│  1. WAV read                │
│  2. Resample → 16kHz        │
│  3. VAD / trim silence      │
└────────────┬────────────────┘
             │ float32 [1, 16000] per second
             ▼
┌─────────────────────────────┐
│  mojo-audio ML (Layer 2)    │
│                             │
│  4. AudioEncoder.encode()   │
│     ├── CNN feature extract │
│     ├── Feature projection  │
│     ├── Position embeddings │
│     └── 12x Transformer     │
└────────────┬────────────────┘
             │ float32 [1, 49, 768] content features
             ▼
┌─────────────────────────────┐  ◄── NOT YET IN mojo-audio
│  Python (PyTorch / Applio)  │
│                             │
│  5. RMVPE pitch extraction  │
│  6. FAISS index retrieval   │
│  7. VITS synthesis          │
└────────────┬────────────────┘
             │ converted audio waveform
             ▼
┌─────────────────────────────┐
│  mojo-audio DSP (Layer 1)   │
│                             │
│  8. Resample → 48kHz        │
│  9. Post-EQ / normalize     │
│ 10. WAV write               │
└─────────────────────────────┘
             │
             ▼
     Output WAV (48kHz, ready to mix)
```

---

## What Each Layer Owns

| Stage | Owner | Status |
|-------|-------|--------|
| Audio I/O (read/write WAV) | mojo-audio DSP | Done |
| Sample rate conversion | mojo-audio DSP | Done |
| Silence detection / trimming | mojo-audio DSP | Done |
| Pitch shifting (traditional) | mojo-audio DSP | Done |
| Mel spectrogram (for Whisper) | mojo-audio DSP | Done, used in mojovoice |
| Waveform reconstruction | mojo-audio DSP | Done |
| FFI shared library | mojo-audio DSP | Done |
| HuBERT / ContentVec encoding | mojo-audio ML | Done (with one workaround) |
| RMVPE pitch extraction | **Not yet** | Next target after HuBERT |
| FAISS nearest-neighbor retrieval | **Not yet** | Stays Python/C++ |
| VITS synthesis (voice model) | **Not yet** | Long-term MAX Graph target |
| Mel-Band RoFormer (stem sep.) | **Not yet** | Python/ORT for now |

---

## Hardware Support

| Platform | DSP Layer | AudioEncoder CPU | AudioEncoder GPU |
|----------|-----------|-----------------|-----------------|
| x86_64 + RTX (CUDA 12.8) | Full | Yes, 154ms/s | Partial* |
| DGX Spark GB10 SM_121 ARM64 | Full | Yes, 100ms/s | Partial* |
| Apple Silicon (M-series) | Full | Yes | Not yet (MAX partial) |
| CPU-only (any) | Full | Yes | — |

\* GPU runs CNN + 12 transformer blocks on GPU. pos_conv bridge runs on CPU.
Once the MAX conv2d groups bug is fixed, full GPU pipeline activates.

### Benchmark: AudioEncoder — `facebook/hubert-base-ls960`, 1s @16kHz

**Local machine (x86_64, RTX 4060 Ti):**

| Backend | Mean (ms) | vs PyTorch CPU |
|---------|-----------|----------------|
| PyTorch CPU | 82.9 | 1.00x |
| PyTorch GPU | 4.8 | 17.1x |
| MAX Engine CPU | 154.3 | 0.54x |
| MAX Engine GPU | 221.9 | 0.37x |

**DGX Spark (aarch64, GB10 SM_121, CUDA 13.0):**

| Backend | Mean (ms) | vs PyTorch CPU |
|---------|-----------|----------------|
| PyTorch CPU | 179.2 | 1.00x |
| PyTorch GPU | N/A (ARM64 CUDA wheel broken) | — |
| MAX Engine CPU | 100.1 | **1.79x** |
| MAX Engine GPU | 197.1 | 0.91x |

**Key insight:** On Spark, MAX Engine CPU is the only option that actually delivers and it's 1.79x faster than PyTorch CPU. PyTorch GPU doesn't work on Spark due to ARM64/CUDA wheel issues that MAX sidesteps entirely.

---

## Integrations

```
mojovoice (Rust)                    Shade (Python FastAPI)
      │                                      │
      │ calls FFI                            │ imports Python
      ▼                                      ▼
libmojo_audio.so                    mojo_audio.models.AudioEncoder
      │                                      │
      │ mel spectrogram                      │ HuBERT / ContentVec
      │ (Whisper preprocessing)              │ (voice conversion)
      ▼                                      ▼
   Whisper STT                         RVC v2 / VITS
```

---

## Test Coverage

| Module | Test file | Count |
|--------|-----------|-------|
| Window functions | test_window.mojo | 6 tests |
| FFT / RFFT / STFT | test_fft.mojo | ~15 tests |
| Mel spectrogram | test_mel.mojo | 5 tests |
| WAV I/O | test_wav_io.mojo | 5 tests |
| Resampler | test_resample.mojo | 8 tests |
| VAD | test_vad.mojo | 7 tests |
| iSTFT + Griffin-Lim | test_istft.mojo | 5 tests |
| Pitch shifter | test_pitch.mojo | 5 tests |
| AudioEncoder (fast) | test_audio_encoder.py | 30 tests |
| AudioEncoder (integration) | test_audio_encoder.py | 1 slow test |
| **Total** | | **~87 tests** |
