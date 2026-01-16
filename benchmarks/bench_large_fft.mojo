"""
Benchmark: Large FFT Performance (Stage 3 - Cache Blocking).

Tests FFT performance on sizes that exceed L2 cache to identify
where cache blocking would help.

L2 cache: ~256KB = 32K complex Float32
Target: Improve performance for N > 8192
"""

from audio import (
    fft, fft_simd, TwiddleFactorsSoA,
    Radix4TwiddleCache, fft_radix4_cached_simd,
    FourStepCache, fft_four_step,
    ComplexArray, next_power_of_2
)
from time import perf_counter_ns


fn benchmark_original_fft(signal: List[Float32], iterations: Int) raises -> Float64:
    """Benchmark original AoS FFT."""
    var start = perf_counter_ns()

    for _ in range(iterations):
        var result = fft(signal)
        var _ = result[0].real

    var end = perf_counter_ns()
    return Float64(end - start) / Float64(iterations) / 1_000_000.0


fn benchmark_simd_fft(signal: List[Float32], twiddles: TwiddleFactorsSoA, iterations: Int) raises -> Float64:
    """Benchmark SIMD radix-2 FFT."""
    var start = perf_counter_ns()

    for _ in range(iterations):
        var result = fft_simd(signal, twiddles)
        var _ = result.real[0]

    var end = perf_counter_ns()
    return Float64(end - start) / Float64(iterations) / 1_000_000.0


fn benchmark_radix4_simd(signal: List[Float32], cache: Radix4TwiddleCache, iterations: Int) raises -> Float64:
    """Benchmark radix-4 SIMD FFT with aligned cache."""
    var start = perf_counter_ns()

    for _ in range(iterations):
        var result = fft_radix4_cached_simd(signal, cache)
        var _ = result.real[0]

    var end = perf_counter_ns()
    return Float64(end - start) / Float64(iterations) / 1_000_000.0


fn benchmark_four_step(signal: List[Float32], cache: FourStepCache, iterations: Int) raises -> Float64:
    """Benchmark four-step cache-blocked FFT."""
    var start = perf_counter_ns()

    for _ in range(iterations):
        var result = fft_four_step(signal, cache)
        var _ = result.real[0]

    var end = perf_counter_ns()
    return Float64(end - start) / Float64(iterations) / 1_000_000.0


fn main() raises:
    print("\n" + "="*70)
    print("Large FFT Benchmark (Cache Blocking Analysis)")
    print("="*70)
    print("\nL2 Cache: ~256KB = 32K complex Float32")
    print("Sizes > 8192 may benefit from cache blocking\n")

    # Test sizes from small (cache-friendly) to large (cache-unfriendly)
    var test_sizes = List[Int]()
    test_sizes.append(1024)   # 8KB - fits in L1
    test_sizes.append(4096)   # 32KB - fits in L1/L2
    test_sizes.append(8192)   # 64KB - exceeds L1, fits L2
    test_sizes.append(16384)  # 128KB - fits L2
    test_sizes.append(32768)  # 256KB - at L2 limit
    test_sizes.append(65536)  # 512KB - exceeds L2

    print("-"*80)
    print("    Size     Memory      Original    R2-SIMD     R4-SIMD     4-Step")
    print("-"*80)

    for idx in range(len(test_sizes)):
        var N = test_sizes[idx]
        var memory_kb = N * 8 // 1024  # Complex Float32 = 8 bytes

        # Adjust iterations based on size
        var iterations = 100
        if N >= 32768:
            iterations = 20
        elif N >= 16384:
            iterations = 50

        # Create test signal
        var signal = List[Float32]()
        for i in range(N):
            signal.append(Float32(i % 100) / 100.0)

        # Pre-compute caches (only for power-of-4 sizes for radix-4)
        var twiddles = TwiddleFactorsSoA(N)

        # Check if power of 4 for radix-4
        var log2_n = 0
        var temp = N
        while temp > 1:
            temp //= 2
            log2_n += 1
        var is_power_of_4 = (log2_n % 2 == 0)

        # Warmup
        _ = fft(signal)
        _ = fft_simd(signal, twiddles)

        # Benchmark
        var orig_time = benchmark_original_fft(signal, iterations)
        var simd_time = benchmark_simd_fft(signal, twiddles, iterations)

        var r4_time_str = String("   N/A   ")
        if is_power_of_4:
            var cache = Radix4TwiddleCache(N)
            _ = fft_radix4_cached_simd(signal, cache)
            var r4_time = benchmark_radix4_simd(signal, cache, iterations)
            var r4_str = String(r4_time)
            r4_time_str = String(r4_str[:8])

        # Four-step FFT (works on perfect squares = power of 4)
        var fs_time_str = String("   N/A   ")
        if is_power_of_4:
            var fs_cache = FourStepCache(N)
            _ = fft_four_step(signal, fs_cache)
            var fs_time = benchmark_four_step(signal, fs_cache, iterations)
            var fs_str = String(fs_time)
            fs_time_str = String(fs_str[:8])

        var orig_str = String(orig_time)
        var simd_str = String(simd_time)
        print("    " + String(N) + "     " + String(memory_kb) + "KB" +
              "       " + String(orig_str[:8]) +
              "    " + String(simd_str[:8]) +
              "    " + r4_time_str +
              "    " + fs_time_str)

    print("-"*80)
    print("\nAnalysis:")
    print("- If time grows faster than O(N log N), cache misses are hurting")
    print("- Compare time/N*log(N) ratio across sizes")
    print("- Cache blocking should help sizes > L2 cache")
    print("\n" + "="*70 + "\n")
