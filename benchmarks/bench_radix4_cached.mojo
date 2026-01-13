"""
Benchmark: Radix-4 Cached vs Radix-2 SIMD FFT.

Tests the zero-allocation radix-4 implementation against radix-2.
Expected: Radix-4 should be ~1.3x faster due to 75% operation count.
"""

from audio import (
    fft, fft_simd, TwiddleFactorsSoA,
    Radix4TwiddleCache, fft_radix4_cached, fft_radix4_cached_simd,
    ComplexArray, next_power_of_2
)
from time import perf_counter_ns


fn benchmark_fft_original(signal: List[Float32], iterations: Int) raises -> Float64:
    """Benchmark original AoS FFT (baseline)."""
    var start = perf_counter_ns()

    for _ in range(iterations):
        var result = fft(signal)
        var _ = result[0].real

    var end = perf_counter_ns()
    return Float64(end - start) / Float64(iterations) / 1_000_000.0


fn benchmark_radix2_simd(signal: List[Float32], twiddles: TwiddleFactorsSoA, iterations: Int) raises -> Float64:
    """Benchmark radix-2 SIMD FFT."""
    var start = perf_counter_ns()

    for _ in range(iterations):
        var result = fft_simd(signal, twiddles)
        var _ = result.real[0]

    var end = perf_counter_ns()
    return Float64(end - start) / Float64(iterations) / 1_000_000.0


fn benchmark_radix4_scalar(signal: List[Float32], cache: Radix4TwiddleCache, iterations: Int) raises -> Float64:
    """Benchmark radix-4 cached scalar FFT."""
    var start = perf_counter_ns()

    for _ in range(iterations):
        var result = fft_radix4_cached(signal, cache)
        var _ = result.real[0]

    var end = perf_counter_ns()
    return Float64(end - start) / Float64(iterations) / 1_000_000.0


fn benchmark_radix4_simd(signal: List[Float32], cache: Radix4TwiddleCache, iterations: Int) raises -> Float64:
    """Benchmark radix-4 cached SIMD FFT."""
    var start = perf_counter_ns()

    for _ in range(iterations):
        var result = fft_radix4_cached_simd(signal, cache)
        var _ = result.real[0]

    var end = perf_counter_ns()
    return Float64(end - start) / Float64(iterations) / 1_000_000.0


fn main() raises:
    print("\n" + "="*70)
    print("Radix-4 Cached vs Radix-2 SIMD FFT Benchmark")
    print("="*70 + "\n")

    # Test sizes (powers of 4 only for fair comparison)
    var test_sizes = List[Int]()
    test_sizes.append(256)   # 4^4
    test_sizes.append(1024)  # 4^5
    test_sizes.append(4096)  # 4^6

    print("Phase 1: Radix-4 Scalar vs Original")
    print("-"*60)
    print("    Size      Original    R4-Scalar   Speedup")
    print("-"*60)

    for idx in range(len(test_sizes)):
        var N = test_sizes[idx]
        var iterations = 100

        # Create test signal
        var signal = List[Float32]()
        for i in range(N):
            signal.append(Float32(i % 100) / 100.0)

        # Pre-compute caches
        var cache = Radix4TwiddleCache(N)

        # Warmup
        _ = fft(signal)
        _ = fft_radix4_cached(signal, cache)

        # Benchmark
        var orig_time = benchmark_fft_original(signal, iterations)
        var r4_scalar_time = benchmark_radix4_scalar(signal, cache, iterations)

        var speedup = orig_time / r4_scalar_time

        print("    " + String(N) + "       " +
              String(orig_time)[:6] + "      " +
              String(r4_scalar_time)[:6] + "      " +
              String(speedup)[:4] + "x")

    print("\n\nPhase 2: Radix-4 SIMD vs Radix-2 SIMD")
    print("-"*60)
    print("    Size      R2-SIMD     R4-SIMD     Speedup")
    print("-"*60)

    for idx in range(len(test_sizes)):
        var N = test_sizes[idx]
        var iterations = 100

        var signal = List[Float32]()
        for i in range(N):
            signal.append(Float32(i % 100) / 100.0)

        # Pre-compute caches
        var twiddles = TwiddleFactorsSoA(N)
        var cache = Radix4TwiddleCache(N)

        # Warmup
        _ = fft_simd(signal, twiddles)
        _ = fft_radix4_cached_simd(signal, cache)

        # Benchmark
        var r2_time = benchmark_radix2_simd(signal, twiddles, iterations)
        var r4_time = benchmark_radix4_simd(signal, cache, iterations)

        var speedup = r2_time / r4_time

        print("    " + String(N) + "       " +
              String(r2_time)[:6] + "      " +
              String(r4_time)[:6] + "      " +
              String(speedup)[:4] + "x")

    print("\n\nFull Comparison (all methods)")
    print("-"*70)
    print("    Size      Original    R2-SIMD     R4-Scalar   R4-SIMD")
    print("-"*70)

    for idx in range(len(test_sizes)):
        var N = test_sizes[idx]
        var iterations = 100

        var signal = List[Float32]()
        for i in range(N):
            signal.append(Float32(i % 100) / 100.0)

        var twiddles = TwiddleFactorsSoA(N)
        var cache = Radix4TwiddleCache(N)

        # Warmup
        _ = fft(signal)
        _ = fft_simd(signal, twiddles)
        _ = fft_radix4_cached(signal, cache)
        _ = fft_radix4_cached_simd(signal, cache)

        var orig_time = benchmark_fft_original(signal, iterations)
        var r2_time = benchmark_radix2_simd(signal, twiddles, iterations)
        var r4_scalar_time = benchmark_radix4_scalar(signal, cache, iterations)
        var r4_simd_time = benchmark_radix4_simd(signal, cache, iterations)

        print("    " + String(N) + "       " +
              String(orig_time)[:6] + "      " +
              String(r2_time)[:6] + "      " +
              String(r4_scalar_time)[:6] + "      " +
              String(r4_simd_time)[:6])

    print("\n" + "="*70)
    print("Benchmark complete!")
    print("="*70 + "\n")
