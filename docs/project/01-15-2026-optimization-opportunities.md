# Optimization Opportunities

**Status:** Reference Document
**Last Updated:** 2026-01-15

Current performance: ~22ms for 30s audio (22x faster than naive, ~1.5x slower than librosa on long audio)

---

## High Priority

### 1. Sparse Mel Filterbank Application

| | |
|---|---|
| **Status** | Structure exists, unused |
| **Location** | `src/audio.mojo:3219-3249` |
| **Issue** | `SparseFilter` struct defined but apply code uses dense triple-nested loops |
| **Impact** | 10-20% overall (mel filterbank is 25-35% of frame time) |
| **Effort** | Low - data structure exists, just needs wiring |

**What exists:**
```mojo
struct SparseFilter:
    var start_bin: Int      # First non-zero bin
    var end_bin: Int        # Last non-zero bin
    var weights: List[Float32]  # Only non-zero coefficients
```

**What's missing:** Application function that iterates only over non-zero bins instead of full spectrum.

---

### 2. RFFT Unpack SIMD

| | |
|---|---|
| **Status** | Scalar only |
| **Location** | `src/audio.mojo:2794-2843` |
| **Issue** | Mirror indexing (k and N-k simultaneously) prevents vectorization |
| **Impact** | 5-15% on RFFT stage |
| **Effort** | Medium |

**Current pattern:**
```mojo
for k in range(1, quarter):
    # Access both X[k] and X[N/2-k] - opposite directions
    var mirror_k = half_size - k
    # ... scalar operations
```

**Potential fix:** Temporary buffer rearrangement to enable contiguous SIMD access, then scatter back.

---

### 3. Bit-Reversal Lookup Table

| | |
|---|---|
| **Status** | Scalar bit-shifting loop |
| **Location** | `src/audio.mojo:415-427` |
| **Issue** | Computes bit-reversal per-element instead of table lookup |
| **Impact** | 5-10% on small FFTs (256-512 points) |
| **Effort** | Low |

**Current:**
```mojo
fn bit_reverse(n: Int, bits: Int) -> Int:
    var result = 0
    var x = n
    for _ in range(bits):
        result = (result << 1) | (x & 1)
        x >>= 1
    return result
```

**Fix:** Pre-compute 256 or 512-element lookup table at compile time or cache initialization.

---

## Medium Priority

### 4. FFT-Internal Parallelization

| | |
|---|---|
| **Status** | Not implemented |
| **Location** | FFT butterfly stages |
| **Issue** | Parallelization only at frame level (STFT), not within FFT |
| **Impact** | 2-8x on large FFTs (16384+), minimal on speech FFTs (256-512) |
| **Effort** | High |

**Opportunity:** Butterfly stages operate on disjoint data - could parallelize within stages for large N.

**Challenge:** Overhead exceeds benefit for typical Whisper FFT sizes (400 samples).

---

### 5. Cache Blocking for Mid-Size FFTs (4096-8192)

| | |
|---|---|
| **Status** | Benchmarked, not optimized |
| **Location** | `src/audio.mojo:2418-2646` (four-step FFT) |
| **Issue** | Four-step implemented but only for very large sizes |
| **Impact** | 10-30% at 4096-8192 range |
| **Effort** | High |

**Current behavior:** Falls back to standard FFT without cache optimization for mid-sizes.

**Note:** Previous four-step attempt was 40-60% slower due to allocation overhead. Would need zero-allocation version.

---

### 6. True Split-Radix DIF

| | |
|---|---|
| **Status** | Hybrid compromise |
| **Location** | `src/audio.mojo:1994-2390` |
| **Issue** | Mixes DIF indexing with DIT bit-reversal, causes numerical errors |
| **Impact** | 10-15% theoretical (33% fewer multiplications) |
| **Effort** | High |

**Current state:** Working hybrid radix-4/radix-2. True split-radix documented but deprioritized.

**Trade-off:** Complexity vs. gain - wouldn't close gap with librosa's MKL FFT.

---

## Low Priority

### 7. AVX-512 Specialization

| | |
|---|---|
| **Status** | Generic SIMD width=8 |
| **Issue** | Could use width=16 for Float32 on AVX-512 systems |
| **Impact** | 20-30% on supported hardware |
| **Effort** | Medium |

**Limitation:** Not all target hardware supports AVX-512.

---

### 8. Adaptive Precision (Float16)

| | |
|---|---|
| **Status** | Float32 hardcoded |
| **Issue** | Window operations don't need full precision |
| **Impact** | 20-40% with 2x SIMD throughput |
| **Effort** | High |

**Risk:** Numerical stability in long FFT chains.

---

### 9. GPU Acceleration

| | |
|---|---|
| **Status** | Not implemented |
| **Issue** | Audio frames processed serially, limited batch parallelism |
| **Impact** | 50-200x for batch processing only |
| **Effort** | Very High |

**Reality:** Individual frames too small to amortize GPU transfer overhead.

---

## Current Bottleneck Breakdown

```
Component          % of Frame Time
─────────────────────────────────
FFT + RFFT         35-45%
Mel filterbank     25-35%  ← Sparse filters target this
Window + misc      15-25%
Power spectrum      5-10%
```

---

## Recommended Implementation Order

1. **Sparse Mel Filter Application** - Highest ROI, structure exists
2. **Bit-Reversal LUT** - Trivial change, measurable gain
3. **RFFT Unpack optimization** - Medium effort, targets biggest bottleneck

**Theoretical combined impact:** 20-40% overall speedup

---

## What We've Already Tried (Failed)

| Approach | Result | Why |
|----------|--------|-----|
| SIMD Pack/Unpack | 30% slower | Strided/mirror access requires scalar gather |
| Four-Step FFT | 40-60% slower | 512 allocations per FFT, transpose overhead |
| Naive SIMD | 18% slower | Manual load loops, no alignment guarantees |

See `docs/drafts/01-14-2026-mojo-audio-blog-draft.md` → "What Failed" section for details.
