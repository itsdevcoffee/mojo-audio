# FFT Optimization Stages

**Goal:** Match or exceed librosa/MKL performance on all audio lengths
**Current gap:** ~1.8x slower on 30s audio

---

## Stage Overview

| Stage | Optimization | Expected Gain | Cumulative |
|-------|--------------|---------------|------------|
| 1 | SoA Memory Layout | 1.2-1.5x | 1.2-1.5x |
| 2 | SIMD Butterflies | 3-4x | 4-6x |
| 3 | Cache Blocking | 1.3-1.5x | 5-9x |
| 4 | Memory Alignment | 1.1-1.2x | 6-10x |
| 5 | Split-Radix Algorithm | 1.2x | 7-12x |
| 6 | Parallelization | 2-4x (large N) | 14-48x |

---

## Stage 1: Structure of Arrays (SoA) Memory Layout

**Why first:** Enables all subsequent SIMD optimizations

**Current (Array of Structures):**
```
[Complex(r0,i0), Complex(r1,i1), Complex(r2,i2), ...]
Memory: r0 i0 r1 i1 r2 i2 r3 i3 ...
```

**Target (Structure of Arrays):**
```
real: [r0, r1, r2, r3, ...]
imag: [i0, i1, i2, i3, ...]
```

**Benefits:**
- Contiguous memory for SIMD loads
- Better cache line utilization
- Enables vectorized operations

**Validation:**
- [x] All existing FFT tests pass
- [x] Numerical accuracy within 1e-5
- [x] Benchmark shows no regression

**Status (2026-01-13):** Complete. `ComplexArray` and `TwiddleFactorsSoA` structs implemented.

---

## Stage 2: SIMD Butterfly Operations

**Why second:** Biggest single performance gain

**Current (scalar):**
```mojo
# 1 complex operation at a time
var t = twiddle * x[j + half]
x[j + half] = x[j] - t
x[j] = x[j] + t
```

**Target (SIMD):**
```mojo
# 4-8 complex operations at a time
var re = SIMD[DType.float32, 8].load(real, j)
var im = SIMD[DType.float32, 8].load(imag, j)
# Vectorized twiddle multiply
# Vectorized butterfly add/sub
```

**Key operations to vectorize:**
1. Twiddle factor multiplication (complex mul)
2. Butterfly add/subtract
3. Bit-reversal permutation (use shuffle intrinsics)

**Validation:**
- [x] All FFT tests pass
- [x] RFFT tests pass
- [~] Benchmark shows 1.2-1.7x improvement (radix-2 path)
- [x] Test with N=256, 512, 1024, 2048, 4096, 8192

**Status (2026-01-13):**
- Radix-2 SIMD: Complete, 1.2-1.7x speedup
- Radix-4 SIMD: Implemented with stage-specific twiddle precomputation, but allocation overhead negates benefits
- **Decision:** Using radix-2 for all sizes (consistent performance)
- See `docs/benchmarks/01-13-2026-stage2-simd-fft-results.md`

---

## Stage 3: Cache Blocking

**Why third:** Reduces memory bandwidth bottleneck

**Problem:** Large FFTs exceed L1/L2 cache
- L1: 32KB → ~4K complex Float32
- L2: 256KB → ~32K complex Float32
- 30s audio @ 16kHz = 480K samples

**Solution:** Process in cache-sized blocks
```
For each block that fits in L2:
    Perform partial FFT stages
    Write back to memory
Combine blocks with twiddle factors
```

**Approaches:**
1. **Four-step FFT** - Classic cache-oblivious
2. **Six-step FFT** - Better for very large N
3. **Recursive decomposition** - Auto-tunes to cache

**Validation:**
- [ ] FFT tests pass for N > 4096
- [ ] Benchmark 30s audio shows improvement
- [ ] Memory bandwidth profiling (perf stat)

---

## Stage 4: Memory Alignment

**Why fourth:** Enables faster aligned SIMD loads

**Current:** Default heap allocation (8-byte aligned)
**Target:** 64-byte aligned (cache line, AVX-512)

**Changes:**
```mojo
# Aligned allocation
var real = UnsafePointer[Float32].alloc(N, alignment=64)
var imag = UnsafePointer[Float32].alloc(N, alignment=64)

# Aligned loads (no penalty)
var v = SIMD[DType.float32, 16].load[alignment=64](real, i)
```

**Validation:**
- [ ] All tests pass
- [ ] Benchmark shows 10-20% improvement
- [ ] Verify alignment with debug assertions

---

## Stage 5: Split-Radix Algorithm

**Why fifth:** Algorithmic improvement (fewer operations)

**Multiplication counts for N-point FFT:**
| Algorithm | Complex Muls |
|-----------|--------------|
| Radix-2 | N log₂N |
| Radix-4 | 0.75 N log₂N |
| Split-radix | 0.67 N log₂N |

**Split-radix idea:**
- Use radix-2 for even indices
- Use radix-4 for odd indices
- ~20% fewer multiplications than radix-4

**Validation:**
- [ ] FFT tests pass
- [ ] Numerical accuracy maintained
- [ ] Benchmark shows ~20% improvement

---

## Stage 6: Parallelization

**Why last:** Only benefits large transforms, adds complexity

**Parallel opportunities:**
1. Independent butterfly groups within a stage
2. Independent blocks in cache-blocked FFT
3. Batch processing multiple FFTs

**Implementation:**
```mojo
from algorithm import parallelize

# Parallel butterfly groups
parallelize[process_group](num_groups, num_workers)
```

**Threshold:** Only parallelize for N > 4096 (overhead)

**Validation:**
- [ ] Tests pass with 1, 2, 4, 8 workers
- [ ] No race conditions
- [ ] Benchmark shows scaling on multi-core

---

## Validation Protocol

After each stage:

1. **Correctness tests:**
   ```bash
   pixi run test-fft
   ```

2. **Benchmark comparison:**
   ```bash
   python ui/backend/run_benchmark.py mojo 30 5
   python ui/backend/run_benchmark.py librosa 30 5
   ```

3. **Regression check:** No stage should make performance worse

4. **Document results:** Update benchmark markdown

---

## Success Criteria

| Audio Length | Current | Target | Status |
|--------------|---------|--------|--------|
| 1s | 0.92ms (2.2x faster) | maintain | |
| 10s | 7.58ms (tie) | 2x faster | |
| 30s | 21.77ms (1.8x slower) | 2x faster | |

**Final goal:** Beat librosa across all audio lengths
