# Changelog

All notable changes to mojo-audio will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

---

## [0.1.0] - 2026-01-22

### 🎉 Initial Release

High-performance mel spectrogram preprocessing library in Mojo that beats Python's librosa by 1.5-3.6x on short/medium audio.

### ✨ Features

#### Core Audio Processing
- **Mel Spectrogram Pipeline** - Complete Whisper-compatible preprocessing
- **Window Functions** - Hann and Hamming windows with SIMD optimization
- **FFT Operations** - Radix-2/4 iterative FFT with true RFFT for real signals
- **STFT** - Parallelized short-time Fourier transform across CPU cores
- **Mel Filterbank** - Sparse-optimized triangular filters (80 or 128 bands)
- **Normalization** - Multiple modes (Whisper, min-max, z-score, raw)

#### FFI Support
- **C-compatible API** - Use from any language with C interop
- **Shared Library** - `libmojo_audio.so` with zero overhead
- **Language Examples** - C, Rust, Python (ctypes) examples included
- **Type-safe Interface** - Proper error handling and memory management

#### Web Benchmark UI
- **FastAPI Backend** - REST API for running benchmarks
- **Interactive Frontend** - Real-time benchmark visualization
- **Advanced Settings** - Signal types, warmup config, stable mode
- **URL Parameter Sync** - Shareable benchmark configurations

### 🚀 Performance

**Benchmarks (vs librosa):**
- **1 second:** 1.1ms vs 4.0ms → **3.6x faster**
- **10 seconds:** 7.5ms vs 15.3ms → **2.0x faster**
- **30 seconds:** 27.4ms vs 30.4ms → **1.1x faster**

**Key advantages:**
- 1.5-3.6x faster on short/medium audio
- 5-10% variance (librosa: 22-39%)
- ~1100x realtime throughput on 30s audio

### 🔧 Optimizations

Nine major optimizations from naive to production:

1. **Iterative FFT** (3.0x) - Cache-friendly Cooley-Tukey algorithm
2. **Twiddle Precompute** (1.7x) - Pre-computed rotation factors
3. **Sparse Mel Filterbank** (1.24x) - Skip zero-weight bins
4. **Twiddle Caching** (2.0x) - Reuse coefficients across frames
5. **Float32 Precision** (1.07x) - 2x SIMD width vs Float64
6. **True RFFT** (1.43x) - Exploits real signal symmetry
7. **Parallelization** (1.3-1.7x) - Multi-core frame processing
8. **Radix-4 FFT** (1.1-1.2x) - Optimized butterflies for power-of-4
9. **Compiler Optimization** (1.2-1.5x) - `-O3` flag

**Total speedup:** 17-68x from naive implementation!

### 📊 Benchmark Infrastructure

- **Deterministic Testing** - Chirp signal for reproducible results
- **Robust Methodology** - 5 warmup runs, 20 iterations, outlier exclusion
- **Stable Mode** - Run N times, report median for publication
- **Comparison Scripts** - Back-to-back mojo-audio vs librosa testing

### 🧪 Testing

- **17 Test Cases** - Window functions, FFT, mel filterbank
- **Numerical Validation** - Compared against reference implementations
- **Edge Case Coverage** - Power-of-2, non-power-of-2, boundary conditions
- **All Tests Passing** - Production-ready quality

### 📚 Documentation

- **Complete API Reference** - All functions documented
- **FFI Integration Guide** - C, Rust, Python examples
- **Optimization Journey** - How we beat Python (40x speedup story)
- **Benchmark Results** - Timestamped performance data
- **Educational Examples** - Window, FFT, and mel demos

### 🛠️ Development Tools

- **pixi Tasks** - Streamlined development workflow
- **Multiple Benchmark Modes** - Optimized, comparison, stable
- **FFI Build System** - Easy shared library generation
- **Web UI** - Visual benchmark interface

### 📦 Project Structure

```
mojo-audio/
├── src/audio.mojo              # Core library (3,800+ lines)
├── src/ffi/                    # FFI bindings
├── include/mojo_audio.h        # C header
├── tests/                      # Test suite (17 tests)
├── examples/                   # Educational demos + FFI examples
├── benchmarks/                 # Performance benchmarking
├── ui/                         # Web benchmark interface
└── docs/                       # Comprehensive documentation
```

### 🎯 Whisper Compatibility

- ✅ Sample rate: 16kHz
- ✅ FFT size: 400
- ✅ Hop length: 160 (10ms frames)
- ✅ Mel bands: 80 (v2) or 128 (v3)
- ✅ Output shape: (n_mels, ~3000) for 30s
- ✅ Normalization: Whisper-compatible mode

### 🔬 Technical Highlights

**Algorithmic:**
- Iterative Cooley-Tukey FFT (cache-friendly)
- Radix-4 butterflies for power-of-4 sizes
- True RFFT with pack-FFT-unpack optimization
- Sparse mel filterbank application

**Parallelization:**
- Multi-core STFT frame processing
- Thread-safe writes (disjoint frame indices)
- Linear scaling with CPU cores

**SIMD Vectorization:**
- Float32 for 16-element SIMD (AVX-512)
- Compile-time width detection
- @parameter unrolling for tight loops

**Memory Optimization:**
- Pre-computed twiddle factors
- Cached coefficients across frames
- Pre-allocated output buffers

### 🌟 Achievements

- 🏆 Beats librosa at all durations
- 🚀 17-68x speedup from naive to optimized
- ⚡ ~1100x realtime throughput
- 📊 Far more consistent than librosa
- ✅ 100% from scratch in Mojo
- 🎓 Complete learning resource

---

## [Unreleased]

Nothing yet! See [GitHub Issues](https://github.com/itsdevcoffee/mojo-audio/issues) for planned features.

---

### Legend

- ✨ **Features** - New functionality
- 🚀 **Performance** - Speed improvements
- 🐛 **Bug Fixes** - Fixed issues
- 📚 **Documentation** - Docs improvements
- 🔧 **Changed** - Modified behavior
- ⚠️ **Breaking** - Breaking changes
- 🗑️ **Deprecated** - Soon-to-be removed
- 🔒 **Security** - Vulnerability fixes

---

[0.1.0]: https://github.com/itsdevcoffee/mojo-audio/releases/tag/v0.1.0
