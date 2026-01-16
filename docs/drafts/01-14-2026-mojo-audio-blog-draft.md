# Building a Fast Mel Spectrogram Library in Mojo

*By [Dev Coffee](https://github.com/itsdevcoffee) — building ML infrastructure in Mojo and Rust*

**TL;DR:** We built an audio DSP library from scratch in Mojo. On 1-second audio, it's 3x faster than librosa. On 30-second audio, librosa's MKL-optimized FFT pulls ahead. Nine optimization stages took us from 476ms to 22ms—a 22x internal speedup. Here's exactly what worked, what failed, and what we learned.

**Who this is for:**
- **Skimmers:** See the [benchmark table](#benchmarks) and [optimization summary](#the-9-optimization-stages)
- **Implementers:** Jump to [getting started](#try-it)
- **Evaluators:** Read the [what failed section](#what-failed)
- **Whisper users:** This is drop-in preprocessing for speech-to-text pipelines

---

## The Result

```
Whisper audio preprocessing (random audio, fair comparison):

              1 second    10 seconds    30 seconds
librosa:        3ms          10ms          14ms
mojo-audio:     1ms           9ms          22ms

Result: Mojo 3x faster on short audio
        librosa ~1.5x faster on long audio
        (Performance depends on audio length)
```

We started 31.7x *slower* than librosa. After 9 optimization stages, we're competitive—and significantly faster on short audio where startup overhead dominates.

<!-- IMAGE: mojo-audio-diagram-3.jpeg
     Alt: "Performance comparison bar chart showing mojo-audio vs librosa at 1s, 10s, and 30s audio durations"
     Caption: "mojo-audio wins on short audio (3x), librosa wins on long audio (~1.5x)"
     Width: full-width
-->

[→ Try the interactive benchmark demo](#try-it)

---

## Why We Built This

We're building [Mojo Voice](https://mojovoice.ai)—a developer-focused voice-to-text app. Under the hood, it uses a Whisper speech-to-text pipeline built in Rust with [Candle](https://github.com/huggingface/candle) (Hugging Face's ML framework).

Whisper requires mel spectrogram preprocessing. We ran into issues with Candle's mel spectrogram implementation—output inconsistencies that were hard to debug through their abstractions.

We had two options:
1. Debug Candle's internals and hope our fixes get merged
2. Build our own from scratch and own the code

We chose option 2.

**Why Mojo instead of Rust?** Our Whisper pipeline is in Rust, but Mojo offered a compelling experiment: could we get C-level performance with Python-like iteration speed? DSP algorithms benefit from rapid prototyping—we tried 9+ optimization approaches, and Mojo let us iterate quickly while still compiling to fast native code. The FFI bridge to Rust is straightforward.

**Why "from scratch" matters:**
- Full control over correctness (we validate against Whisper's expected output)
- Full control over performance (we choose the algorithms)
- No upstream abstractions hiding bugs
- A genuine technical differentiator for Mojo Voice

**The question:** Could Mojo actually compete with librosa, which delegates to decades of C/Fortran optimization via BLAS/MKL?

**The answer:** Yes—on some workloads. But it took 9 optimization stages and some humbling failures.

---

## Architecture

Before diving into optimizations, here's what we're building:

<!-- IMAGE: mojo-audio-diagram-1.jpeg
     Alt: "Mel spectrogram pipeline flowchart showing Audio Input → Window Function → STFT → Mel Filterbank → Log + Normalize → Output"
     Caption: "The mel spectrogram pipeline: 6 stages from raw audio to Whisper-compatible output"
     Width: full-width
-->

**Why these parameters?** They match OpenAI Whisper's expected input: 16kHz sample rate, 400-sample FFT (25ms window), 160-sample hop (10ms), 80 mel bands. Different choices would require retraining the model.

**Key design decisions:**
- **No external dependencies:** Full control over memory layout and algorithms
- **SoA layout:** Store real/imaginary components separately for SIMD efficiency
- **64-byte alignment:** Match cache line size for optimal memory access
- **Handle-based FFI:** C-compatible API for Rust/Python/Go integration

---

## The 9 Optimization Stages

*Quick terminology: "Twiddle factors" are pre-computed sine/cosine values used in FFT butterfly operations. Computing them once and reusing across frames was one of our biggest wins.*

| Stage | Technique | Speedup | Time | What We Did |
|-------|-----------|---------|------|-------------|
| 0 | Naive implementation | — | 476ms | Recursive FFT, allocations everywhere |
| 1 | Iterative FFT | 3.0x | 165ms | Cooley-Tukey, cache-friendly access |
| 2 | Twiddle precomputation | 1.7x | 97ms | Pre-compute sin/cos, reuse across calls |
| 3 | Sparse filterbank | 1.24x | 78ms | Store only non-zero mel coefficients |
| 4 | Twiddle caching | 2.0x | 38ms | Cache twiddles across all STFT frames |
| 5 | @always_inline | 1.05x | 36ms | Force inline hot functions |
| 6 | Float32 precision | 1.07x | 34ms | 2x SIMD width (16 vs 8 elements) |
| 7 | True RFFT | 1.43x | 24ms | Real-to-complex FFT, half the work |
| 8 | RFFTCache + Radix-4 | 1.1x | 22ms | Zero-allocation RFFT, 4-point butterflies |

**Total: 22x faster than where we started.**

*Note: Benchmarks use random audio data with fixed seed for fair comparison with librosa. Constant/synthetic audio can show artificially better results.*

<!-- IMAGE: mojo-audio-diagram-4.jpeg
     Alt: "Horizontal bar chart showing 22x optimization journey from 476ms naive implementation down to 22ms final, with librosa baseline marked"
     Caption: "Each optimization stage shrinks the bar—we match librosa's ballpark at stage 8"
     Width: full-width
-->

---

## Deep Dives: What Actually Moved the Needle

### 1. Memory Layout: Structure-of-Arrays (SoA)

Our first implementation stored complex numbers as `List[Complex]`—an array of structs. Each complex number sat in its own memory location, fragmenting cache access.

**The fix:** Structure-of-Arrays. Store all real components contiguously, all imaginary components contiguously.

<!-- IMAGE: mojo-audio-diagram-2.jpeg
     Alt: "Memory layout comparison showing AoS (fragmented, red) vs SoA (contiguous, green) with SIMD loading visualization"
     Caption: "SoA enables SIMD to load 8 values in one instruction instead of scattered cache access"
     Width: full-width
-->

SIMD can now load 8+ real values in one instruction. Cache lines fetch useful data instead of interleaved noise.

**Impact:** SoA is a foundational change that enables later SIMD optimizations. It's not a single stage in our table—the benefits are distributed across stages 6-9 where SIMD and compiler optimizations compound on the better memory layout.

### 2. True RFFT: Don't Compute What You Don't Need

Audio is real-valued (no imaginary component). A standard complex FFT computes N complex outputs, but for real input, the output has a symmetry property: the second half mirrors the first half.

Mathematically: `X[k] = conjugate(X[N-k])` — if you know `X[3]`, you automatically know `X[N-3]`.

This means we only need to compute the first `N/2 + 1` frequency bins. The rest is redundant.

**The algorithm (pack-FFT-unpack):**
1. **Pack:** Treat N real samples as N/2 complex numbers by pairing adjacent samples: `z[k] = x[2k] + i·x[2k+1]`
2. **FFT:** Run a standard N/2-point complex FFT on these packed values
3. **Unpack:** Recover the N/2+1 real-FFT bins using twiddle factors (pre-computed sin/cos values) to "unmix" the packed result

The key insight: we do an FFT of half the size, then some cheap arithmetic to recover the full result.

**Impact:** ~1.4x faster for real audio signals.

### 3. Parallelization: Obvious but Effective

STFT processes ~3000 independent frames. Each frame's FFT doesn't depend on others.

```mojo
parallelize[process_frame](num_frames, num_cores)
```

On a 16-core i7-1360P: 1.3-1.7x speedup. Not linear (overhead from thread coordination), but meaningful.

**Gotcha:** Only parallelize for N > 4096. Smaller transforms lose more to thread overhead than they gain.

---

## What Failed

Honest accounting of approaches that didn't work.

### Naive SIMD: 18% Slower

Our first SIMD attempt made performance *worse*.

```mojo
# What we tried (wrong)
for j in range(simd_width):
    simd_vec[j] = data[i + j]  # Scalar loop inside "SIMD" code!
```

**Why it failed:**
- Manual load/store loops negate SIMD benefits
- `List[Float64]` has no alignment guarantees
- 400 samples per frame—too small to amortize SIMD setup

**The lesson:** SIMD requires pointer-based access with aligned memory. Naive vectorization is often slower than scalar code.

**What worked instead:**

```mojo
# Correct approach: direct pointer loads
fn apply_window_simd(
    signal: UnsafePointer[Float32],
    window: UnsafePointer[Float32],
    result: UnsafePointer[Float32],
    length: Int
):
    alias width = 8  # Process 8 Float32s at once
    for i in range(0, length, width):
        var sig = signal.load[width=width](i)  # Single instruction
        var win = window.load[width=width](i)  # Single instruction
        result.store(i, sig * win)             # Vectorized multiply + store
```

The difference: `load[width=8]()` compiles to a single SIMD instruction. The naive loop compiles to 8 scalar loads.

### Four-Step FFT: Cache Blocking That Didn't Help

Theory: For large FFTs, cache misses dominate. The four-step algorithm restructures computation as matrix operations to stay cache-resident.

We implemented it. It worked correctly. And it was 40-60% *slower* for all practical audio sizes.

| Size | Memory | Direct FFT | Four-Step | Result |
|------|--------|------------|-----------|--------|
| 4096 | 32KB | 0.12ms | 0.21ms | 0.57x slower |
| 65536 | 512KB | 2.8ms | 4.6ms | 0.61x slower |

**Why it failed:**
- 512 allocations per FFT (column extraction)
- Transpose overhead adds ~15-20%
- At N ≤ 65536, working set barely exceeds L2 cache

**The lesson:** Cache blocking helps when N > 1M and memory bandwidth is the bottleneck. Audio FFTs (N ≤ 65536) don't hit that threshold. We archived the code and moved on.

### SIMD Pack/Unpack: Strided Access Doesn't Vectorize

RFFT requires "packing" real samples into complex pairs (reading every 2nd element) and "unpacking" with mirror indexing (accessing both k and N-k simultaneously).

We tried SIMD-ifying these loops. Result: **30% slower**.

**Why it failed:**
- Strided access (every 2nd element) requires scalar gather operations
- Mirror indexing (k and N-k) can't be vectorized—data flows in opposite directions
- The overhead of building SIMD vectors from scattered elements exceeds any benefit

**The lesson:** SIMD only helps with contiguous, forward-only memory access. Complex access patterns are better left to the compiler's auto-vectorizer.

### Split-Radix: Complexity vs. Reality

Split-radix FFT promises 33% fewer multiplications than radix-2. We implemented it.

**The catch:** True split-radix uses Decimation-In-Frequency (DIF), requiring bit-reversal at the *end*, not the beginning. Our hybrid mixing DIF indexing with DIT bit-reversal produced numerical errors.

**Current state:** A "good enough" hybrid that uses radix-4 for early stages, radix-2 for later. True split-radix is documented but deprioritized—the 10-15% theoretical gain wouldn't close the gap with librosa's MKL-optimized FFT, and the complexity isn't justified.

---

## Benchmarks

### mojo-audio vs. librosa

| Duration | mojo-audio | librosa | Result |
|----------|------------|---------|--------|
| 1s | ~1ms | ~3ms | **Mojo 3x faster** |
| 10s | ~9ms | ~10ms | Roughly tied |
| 30s | ~22ms | ~14ms | librosa ~1.5x faster |

**Honest assessment:** We win decisively on short audio where Python/NumPy startup overhead hurts librosa. On longer audio, librosa's MKL-optimized BLAS kernels pull ahead—decades of hand-tuned C/Fortran are hard to beat. Our advantage on short audio comes from zero Python overhead and algorithm choices (true RFFT, sparse filterbank).

**Why this matters for real use:**
- Whisper inference takes 500ms-5s depending on model size
- The 8ms preprocessing difference on 30s audio is <2% of total pipeline time
- For real-time streaming (short chunks), Mojo's 3x advantage matters more

**Methodology:**
- 5+ iterations, average reported (variance: ~15-20% due to CPU frequency scaling)
- Hardware: Intel i7-1360P, 12 cores, 32GB RAM
- Audio: Random data with fixed seed (same for both implementations)
- Mojo: 0.26.1, compiled with `-O3`
- librosa: 0.10.1, using OpenBLAS backend
- Benchmark scripts: [`benchmarks/bench_mel_spectrogram.mojo`](https://github.com/itsdevcoffee/mojo-audio/blob/main/benchmarks/bench_mel_spectrogram.mojo), [`benchmarks/compare_librosa.py`](https://github.com/itsdevcoffee/mojo-audio/blob/main/benchmarks/compare_librosa.py)

**Correctness validation:** Our output matches librosa to within 1e-4 max absolute error per frequency bin. We also validated against Whisper's expected input shape (80 × ~3000) and value ranges ([-1, 1] after normalization).

---

## Try It

<!-- IMAGE: mojo-audio-blog-2.png
     Alt: "Interactive benchmark UI showing mojo-audio vs librosa comparison with results display"
     Caption: "Try the interactive benchmark demo yourself"
     Width: full-width
     Note: This is a screenshot of the benchmark web UI - can be replaced with embedded iframe if supported
-->

### Installation

```bash
git clone https://github.com/itsdevcoffee/mojo-audio.git
cd mojo-audio
pixi install
pixi run bench-optimized
```

### Basic Usage

```mojo
from audio import mel_spectrogram

fn main() raises:
    # 30s audio @ 16kHz
    var audio = List[Float32]()
    for i in range(480000):
        audio.append(sin(2.0 * 3.14159 * 440.0 * Float32(i) / 16000.0))

    # Whisper-compatible mel spectrogram
    var mel = mel_spectrogram(audio)
    # Output: (80, ~3000) in ~22ms
```

### Cross-Language Integration (FFI)

```c
// C
MojoMelConfig config;
mojo_mel_config_default(&config);
MojoMelHandle handle = mojo_mel_spectrogram_compute(audio, num_samples, &config);
```

```rust
// Rust
let config = MojoMelConfig::default();
let mel = mojo_mel_spectrogram_compute(&audio, config);
```

---

## What's Next

mojo-audio now powers the preprocessing pipeline in [Mojo Voice](https://mojovoice.ai), our developer-focused voice-to-text app.

**Planned improvements:**
- Whisper v3 support (128 mel bins)
- ARM/Apple Silicon profiling
- Streaming API for real-time processing

**Contributions welcome:** [github.com/itsdevcoffee/mojo-audio](https://github.com/itsdevcoffee/mojo-audio)

---

## Takeaways

1. **Algorithms beat brute force:** Iterative FFT (3x) outperformed naive SIMD (-18%). Understand the problem before reaching for low-level tools.
2. **Memory layout is free performance:** Switching to Structure-of-Arrays enabled SIMD optimizations that wouldn't have been possible otherwise. How you store data matters as much as how you process it.
3. **Know your scale:** Cache blocking helps at N > 1M, not audio sizes (N ≤ 65536). Optimization advice is context-dependent.
4. **Document failures:** Four-step FFT didn't help, but writing it down helped us understand why—and saved future us from trying again.
5. **Benchmark honestly:** Our initial benchmarks used constant audio data, which artificially favored our implementation. Random data with fixed seeds gives fair comparisons.

**Final result:** 476ms → 22ms. 22x internal speedup. 3x faster than librosa on short audio, competitive on medium, ~1.5x slower on long audio.

We set out to own our preprocessing stack. We ended up with a useful library, honest benchmarks, and a lot of lessons about FFT optimization. Sometimes "competitive with decades of C/Fortran optimization" is a win worth celebrating.

---

*Built with [Mojo](https://www.modular.com/mojo) | [Source Code](https://github.com/itsdevcoffee/mojo-audio) | [Benchmarks](https://github.com/itsdevcoffee/mojo-audio/tree/main/benchmarks) | [Mojo Voice](https://mojovoice.ai)*
