# Optimization Opportunities

**Status:** Reference Document
**Last Updated:** 2026-01-16

**Current Performance (benchmarked 2026-01-16):**
- **1s audio:** 1.4ms (1.9x **FASTER** than librosa's 2.7ms)
- **10s audio:** 8.4ms (1.23x slower than librosa's 6.8ms)
- **30s audio:** 19.6ms (1.29x slower than librosa's 15.2ms)

**Key insight:** Frame-level parallelization gives us an edge on short audio! We beat librosa on 1s audio because our parallel STFT overhead is minimal. On longer audio (10s+), librosa's Intel MKL FFT backend pulls ahead.

**Reality check:** The 1.2-1.3x gap on long audio may be irreducible without FFI to MKL/FFTW. Librosa uses Intel MKL—decades of hand-tuned assembly. Pure Mojo optimizations have closed the gap significantly.

---

## High Priority

### 1. Sparse Mel Filterbank Application

| | |
|---|---|
| **Status** | ✅ COMPLETED - Already implemented |
| **Location** | `src/audio.mojo:3292-3352` (apply_mel_filterbank) |
| **Issue** | N/A - Sparse optimization already in place |
| **Impact** | Already delivering 2.5x speedup vs Python/librosa |
| **Effort** | N/A - Already done |

**Implementation details:**
- `SparseFilter` struct fully implemented (lines 3219-3249)
- `apply_mel_filterbank` uses sparse representation (lines 3292-3352)
- Iterates only over non-zero bins (start to end, ~10-30 bins per filter)
- Conditional check handles edge case of all-zero filters
- Performance: ~30ms for 30s audio (vs 75ms for librosa)

**Note:** This was incorrectly listed as "unused" in the initial document. The sparse optimization has been in place since the initial release.

---

### 2. RFFT Unpack SIMD

| | |
|---|---|
| **Status** | Scalar only |
| **Location** | `src/audio.mojo:2794-2843` |
| **Issue** | Mirror indexing (k and N-k simultaneously) prevents vectorization |
| **Impact** | 1-3% overall (5-15% on RFFT stage, which is ~20% of total) |
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

### 3. Bit-Reversal Lookup Table ❌ TESTED - NO BENEFIT

| | |
|---|---|
| **Status** | ❌ Attempted, reverted (16% slower) |
| **Location** | `src/audio.mojo:415-427` |
| **Issue** | Computes bit-reversal per-element instead of table lookup |
| **Impact** | **-16% regression** (worse than original) |
| **Effort** | Low |

**Attempted fix (2026-01-16):**
- Pre-computed 512-element SIMD lookup table
- Result: 37.7ms vs 32.5ms baseline (16% slower)
- Root cause: SIMD indexing + Int conversion + bit-shift overhead exceeded loop cost
- Original loop is highly optimized by compiler (8-9 iterations, excellent branch prediction)

**Conclusion:** Keep original bit-shifting loop. Modern CPU optimization makes simple loops competitive with lookups for small iteration counts.

---

## Medium Priority

### 4. Frame-Level Parallelization (STFT)

| | |
|---|---|
| **Status** | ✅ COMPLETED - Implemented in initial release (v0.1.0) |
| **Location** | `src/audio.mojo:3052-3086` (stft function) |
| **Issue** | N/A - Already parallelized using Mojo's `parallelize()` |
| **Impact** | 1.3-1.7x speedup on multi-core systems (already delivering) |
| **Effort** | N/A - Already done |

**Implementation details:**
- Pre-allocated spectrogram buffer (thread-safe: each thread writes to its own frame_idx)
- `process_frame` closure handles independent frame processing (lines 3054-3082)
- Uses `num_physical_cores()` to determine worker count (line 3085)
- `parallelize[process_frame](num_frames, workers)` distributes frames across cores (line 3086)
- Each frame: extract → window → RFFT → power spectrum (all independent operations)

**Performance:** Contributes 1.3-1.7x speedup to overall performance, scaling with core count and audio duration. Minimal overhead on short audio due to Mojo's efficient task scheduling.

**Note:** This was implemented in the initial release (commit 2dc6b17, Jan 8 2026) as optimization #8 in the release notes.

---

### 5. AVX-512 Specialization

| | |
|---|---|
| **Status** | Generic SIMD width=8 |
| **Location** | All SIMD operations |
| **Issue** | Could use width=16 for Float32 on AVX-512 systems |
| **Impact** | 20-30% on supported hardware (modern server/desktop) |
| **Effort** | Medium - Mojo's SIMD abstraction should make this testable |

**Why medium priority:** If targeting modern server/desktop, this is significant and relatively low-risk.

**Limitation:** Not all target hardware supports AVX-512 (disable for mobile/embedded).

---

### 6. FFT-Internal Parallelization

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

### 7. Cache Blocking for Mid-Size FFTs (4096-8192)

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

### 8. True Split-Radix DIF

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

### 9. Adaptive Precision (Float16)

| | |
|---|---|
| **Status** | Float32 hardcoded |
| **Issue** | Window operations don't need full precision |
| **Impact** | 20-40% with 2x SIMD throughput |
| **Effort** | High |

**Risk:** Numerical stability in long FFT chains.

---

### 10. GPU Acceleration

| | |
|---|---|
| **Status** | Not implemented |
| **Issue** | Audio frames processed serially, limited batch parallelism |
| **Impact** | 50-200x for batch processing only |
| **Effort** | Very High |

**Reality:** Individual frames too small to amortize GPU transfer overhead.

---

### 11. FFI to MKL/FFTW

| | |
|---|---|
| **Status** | Not implemented |
| **Issue** | Pure Mojo FFT unlikely to match MKL performance |
| **Impact** | 1.5-2x (closes gap to librosa) |
| **Effort** | Medium - Requires C FFI bindings |

**Option:** Bind to Intel MKL or FFTW3 for FFT operations while keeping Mojo for STFT orchestration and mel filterbanks.

**Trade-off:** External dependency vs. performance. May be the only way to fully close the librosa gap.

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

**Phase 1 - Quick Wins (attempted):**
1. ~~**Sparse Mel Filter Application** (#1)~~ - ✅ Already completed!
2. ~~**Bit-Reversal LUT** (#3)~~ - ❌ Tested, 16% slower (reverted)
3. ~~**Frame-Level Parallelization** (#4)~~ - ✅ Already completed!

**Phase 2 - Remaining Medium-Impact Options:**
1. **RFFT Unpack SIMD** (#2) - 1-3% overall, medium effort, mirror indexing challenge
2. **AVX-512 Specialization** (#5) - 20-30% on modern hardware, testable in Mojo

**Phase 3 - Benchmark current state:**
- Current: 19.6ms for 30s audio (1.29x slower than librosa's 15.2ms)
- **WIN:** 1.9x FASTER than librosa on 1s audio (1.4ms vs 2.7ms)
- Gap is very competitive - parallelization is working!
- If gap is acceptable (<1.3x on long audio, FASTER on short): Consider complete
- If parity needed: Consider Phase 4 (FFI to MKL)

**Skip these unless special requirements:**
- FFT-Internal Parallelization (#6) - Overhead exceeds benefit for Whisper FFT sizes (400 samples)
- Cache Blocking (#7) - Previous attempts failed (40-60% slower), high complexity
- Split-Radix (#8) - High complexity, won't close librosa gap alone
- Adaptive Precision (#9) - Risk to numerical stability
- GPU Acceleration (#10) - Transfer overhead exceeds benefit for individual frames

**Already delivered optimizations:** Sparse filterbank + Frame parallelization = 1.6-2.1x combined speedup (already in v0.1.0)

---

## What We've Already Tried (Failed)

| Approach | Result | Why |
|----------|--------|-----|
| SIMD Pack/Unpack | 30% slower | Strided/mirror access requires scalar gather |
| Four-Step FFT | 40-60% slower | 512 allocations per FFT, transpose overhead |
| Naive SIMD | 18% slower | Manual load loops, no alignment guarantees |
| Bit-Reversal LUT | 16% slower | SIMD indexing overhead exceeds 8-9 iteration loop cost |

See `docs/drafts/01-14-2026-mojo-audio-blog-draft.md` → "What Failed" section for details.
