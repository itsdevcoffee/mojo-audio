# Mel Spectrogram Benchmark Results

**Date:** 2026-01-12
**Status:** Complete

## System Configuration

| Spec | Value |
|------|-------|
| CPU | 13th Gen Intel Core i7-1360P |
| Cores | 12 |
| RAM | 32 GB |
| OS | Fedora 43 (Kernel 6.18.4) |
| Mojo | 0.26.1.0.dev2026010718 |

## Benchmark Parameters

| Parameter | Value |
|-----------|-------|
| Sample Rate | 16000 Hz |
| FFT Size | 400 |
| Hop Length | 160 |
| Mel Bins | 80 |
| Audio Lengths | 1s, 10s, 30s |

---

## Controlled Benchmark Results

Alternating runs (Mojo then Librosa) to ensure fair thermal/CPU state comparison.

### 1 Second Audio (10 iterations per measurement)

| Round | Mojo (ms) | Librosa (ms) |
|-------|-----------|--------------|
| 1 | 1.032 | 2.392 |
| 2 | 0.809 | 1.836 |
| 3 | 0.833 | 2.506 |
| 4 | 0.925 | 1.901 |
| 5 | 1.011 | 1.491 |

| Stat | Mojo | Librosa |
|------|------|---------|
| **Mean** | 0.92 ms | 2.03 ms |
| **Min** | 0.81 ms | 1.49 ms |
| **Max** | 1.03 ms | 2.51 ms |
| **Winner** | **Mojo 2.2x faster** | |

### 10 Second Audio (5 iterations per measurement)

| Round | Mojo (ms) | Librosa (ms) |
|-------|-----------|--------------|
| 1 | 8.922 | 9.052 |
| 2 | 6.887 | 6.275 |
| 3 | 7.101 | 5.854 |
| 4 | 7.992 | 7.974 |
| 5 | 6.974 | 8.093 |

| Stat | Mojo | Librosa |
|------|------|---------|
| **Mean** | 7.58 ms | 7.45 ms |
| **Min** | 6.89 ms | 5.85 ms |
| **Max** | 8.92 ms | 9.05 ms |
| **Winner** | ~Equal (within margin) | |

### 30 Second Audio (3 iterations per measurement)

| Round | Mojo (ms) | Librosa (ms) |
|-------|-----------|--------------|
| 1 | 23.816 | 14.614 |
| 2 | 20.552 | 11.801 |
| 3 | 25.349 | 12.710 |
| 4 | 19.883 | 10.078 |
| 5 | 19.248 | 10.984 |

| Stat | Mojo | Librosa |
|------|------|---------|
| **Mean** | 21.77 ms | 12.04 ms |
| **Min** | 19.25 ms | 10.08 ms |
| **Max** | 25.35 ms | 14.61 ms |
| **Winner** | | **Librosa 1.8x faster** |

---

## Summary

| Audio Length | Mojo -O3 | Librosa | Ratio | Winner |
|--------------|----------|---------|-------|--------|
| 1s | 0.92 ms | 2.03 ms | 0.45x | **Mojo** |
| 10s | 7.58 ms | 7.45 ms | 1.02x | Tie |
| 30s | 21.77 ms | 12.04 ms | 1.81x | **Librosa** |

### Throughput (realtime multiplier)

| Audio Length | Mojo | Librosa |
|--------------|------|---------|
| 1s | 1087x | 493x |
| 10s | 1319x | 1342x |
| 30s | 1378x | 2492x |

---

## Analysis

### Why Librosa Wins on Long Audio

1. **Optimized BLAS/MKL**: Librosa uses numpy/scipy with Intel MKL or OpenBLAS
2. **Cache efficiency**: MKL FFT algorithms optimize for CPU cache hierarchy
3. **Vectorization**: MKL uses AVX-512 SIMD on supported CPUs
4. **Numba JIT**: Some librosa code paths use Numba for JIT compilation

### Where Mojo Wins

1. **Short audio**: Lower startup overhead, no Python interpreter
2. **Consistent performance**: Less variance between runs
3. **No dependencies**: Self-contained, no numpy/scipy required

### Scaling Behavior

```
Audio Length vs Time:
                Mojo        Librosa     Scaling
1s  →  10s:    8.2x        3.7x        Librosa scales better
10s →  30s:    2.9x        1.6x        Librosa scales better
```

Librosa's FFT implementation scales ~O(n log n) more efficiently for larger inputs.

---

## Next Steps

To improve Mojo performance on longer audio:

- [ ] Use SIMD intrinsics for FFT butterflies
- [ ] Implement cache-oblivious FFT algorithm
- [ ] Add split-radix FFT for better constant factors
- [ ] Consider calling into optimized C FFT library (FFTW)
- [ ] Profile memory access patterns

## Methodology Notes

- Used `ui/backend/run_benchmark.py` for controlled comparison
- Both implementations use same random seed (42) for identical input data
- Alternating runs prevent thermal/turbo bias
- Results vary ~20% between runs due to CPU frequency scaling
