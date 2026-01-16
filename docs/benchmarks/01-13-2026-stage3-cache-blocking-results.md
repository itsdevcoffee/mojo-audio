# Stage 3: Cache Blocking FFT Results

**Date:** 2026-01-13
**Status:** Complete (algorithm works with radix-4, but overhead dominates)

## Summary

| Algorithm | Performance vs R4-SIMD | Notes |
|-----------|------------------------|-------|
| Four-Step FFT (with radix-4) | 0.4-0.6x (slower) | Structural overhead dominates at N ≤ 65536 |

---

## Implementation Details

### Four-Step FFT Algorithm

```
1. View N-point DFT as N1 × N2 matrix (N1 = N2 = sqrt(N))
2. Compute N2 column FFTs of size N1
3. Multiply by twiddle factors W_N^(k1 * n2)
4. Transpose to N2 × N1
5. Compute N1 column FFTs of size N2 (complex-to-complex)
6. Reorder output to match standard FFT ordering
```

### Key Implementation Notes

- Step 4 requires **complex-to-complex FFT** (not real-to-complex)
- Output permutation: element at index `k2*N1 + j` contains `X[j + k2*N1]`
- For N1 = N2, output is already in correct order

### API

```mojo
var cache = FourStepCache(N)  # Pre-compute twiddles
var result = fft_four_step(signal, cache)  # N must be perfect square
```

---

## Benchmark Results (with radix-4 inner FFTs)

| Size | Memory | R4-SIMD (ms) | 4-Step (ms) | Ratio |
|------|--------|--------------|-------------|-------|
| 1024 | 8KB | 0.023 | 0.050 | 0.46x |
| 4096 | 32KB | 0.117 | 0.205 | 0.57x |
| 16384 | 128KB | 0.720 | 1.316 | 0.55x |
| 65536 | 512KB | 2.811 | 4.601 | 0.61x |

---

## Analysis

### Why Four-Step is Still Slower (Despite Radix-4)

1. **Per-column allocation**: 512 ComplexArray allocations per FFT (256 columns × 2 steps)
2. **Data copying**: Column extraction and write-back for each sub-FFT
3. **Transpose cost**: Block transpose adds ~15-20% overhead
4. **Sub-FFT size**: At N=65536, sub-FFTs are 256-point, already cache-friendly

### When Four-Step Would Help

Cache blocking typically benefits sizes where:
- Working set >> L2 cache (e.g., N > 1M for 256KB L2)
- Cache miss penalty > algorithm overhead
- Memory bandwidth is the bottleneck

At N=65536 (512KB), the working set barely exceeds L2, and the cache miss penalty doesn't yet outweigh the four-step overhead.

---

## Conclusions

- **Correctness**: ✓ Algorithm produces identical results to original FFT
- **Performance**: Not beneficial for N ≤ 65536
- **Recommendation**: Use R2-SIMD for all practical audio sizes

### Potential Optimizations (Future)

| Optimization | Expected Gain | Complexity |
|--------------|---------------|------------|
| Zero-allocation (reuse column buffer) | 1.5-2x | Medium |
| SIMD transpose with prefetch | 1.2-1.3x | Medium |
| Fuse twiddle multiply with FFT | 1.1-1.2x | High |
| Threshold-based hybrid (direct for N ≤ 65536) | Best of both | Low |

### Recommendation

For audio applications (N ≤ 65536), **use direct radix-4 SIMD** (`fft_radix4_cached_simd`).

Four-step cache blocking would only benefit at N > 1M where cache misses dominate,
but audio rarely needs FFTs that large.

---

## System Configuration

| Spec | Value |
|------|-------|
| CPU | 13th Gen Intel Core i7-1360P |
| L2 Cache | ~256KB |
| Mojo | 0.26.1 |
| Build | mojo -O3 |
