"""
Benchmark comparing AoS vs SIMD FFT implementations.

Stage 2 optimization validation - SIMD butterflies with SoA layout.
"""

from audio import (
    fft, rfft, power_spectrum,
    fft_simd, rfft_simd, power_spectrum_simd,
    TwiddleFactorsSoA, ComplexArray, next_power_of_2
)
from time import perf_counter_ns


fn benchmark_fft_aos(signal: List[Float32], iterations: Int) raises -> Float64:
    """Benchmark original AoS FFT."""
    var start = perf_counter_ns()

    for _ in range(iterations):
        var result = fft(signal)
        # Use result to prevent dead code elimination
        var _ = result[0].real

    var end = perf_counter_ns()
    return Float64(end - start) / Float64(iterations) / 1_000_000.0


fn benchmark_fft_simd(signal: List[Float32], twiddles: TwiddleFactorsSoA, iterations: Int) raises -> Float64:
    """Benchmark SIMD FFT with SoA layout."""
    var start = perf_counter_ns()

    for _ in range(iterations):
        var result = fft_simd(signal, twiddles)
        var _ = result.real[0]

    var end = perf_counter_ns()
    return Float64(end - start) / Float64(iterations) / 1_000_000.0


fn benchmark_rfft_aos(signal: List[Float32], iterations: Int) raises -> Float64:
    """Benchmark original AoS RFFT."""
    var start = perf_counter_ns()

    for _ in range(iterations):
        var result = rfft(signal)
        var _ = result[0].real

    var end = perf_counter_ns()
    return Float64(end - start) / Float64(iterations) / 1_000_000.0


fn benchmark_rfft_simd(signal: List[Float32], twiddles: TwiddleFactorsSoA, iterations: Int) raises -> Float64:
    """Benchmark SIMD RFFT with SoA layout."""
    var start = perf_counter_ns()

    for _ in range(iterations):
        var result = rfft_simd(signal, twiddles)
        var _ = result.real[0]

    var end = perf_counter_ns()
    return Float64(end - start) / Float64(iterations) / 1_000_000.0


fn main() raises:
    print("\n" + "="*70)
    print("AoS vs SIMD FFT Performance Comparison (Stage 2)")
    print("="*70 + "\n")

    # Test sizes
    var test_sizes = List[Int]()
    test_sizes.append(256)
    test_sizes.append(512)
    test_sizes.append(1024)
    test_sizes.append(2048)
    test_sizes.append(4096)
    test_sizes.append(8192)

    print("FFT Comparison:")
    print("-"*60)
    print("    Size      AoS (ms)    SIMD (ms)   Speedup")
    print("-"*60)

    for idx in range(len(test_sizes)):
        var N = test_sizes[idx]
        var iterations = 100

        # Create test signal
        var signal = List[Float32]()
        for i in range(N):
            signal.append(Float32(i % 100) / 100.0)

        # Pre-compute twiddles for SIMD
        var twiddles = TwiddleFactorsSoA(N)

        # Warmup
        _ = fft(signal)
        _ = fft_simd(signal, twiddles)

        # Benchmark
        var aos_time = benchmark_fft_aos(signal, iterations)
        var simd_time = benchmark_fft_simd(signal, twiddles, iterations)

        var speedup = aos_time / simd_time

        print("    " + String(N) + "       " +
              String(aos_time)[:6] + "      " +
              String(simd_time)[:6] + "      " +
              String(speedup)[:4] + "x")

    print("\nRFFT Comparison:")
    print("-"*60)
    print("    Size      AoS (ms)    SIMD (ms)   Speedup")
    print("-"*60)

    for idx in range(len(test_sizes)):
        var N = test_sizes[idx]
        var iterations = 100

        var signal = List[Float32]()
        for i in range(N):
            signal.append(Float32(i % 100) / 100.0)

        var fft_size = next_power_of_2(N)
        var twiddles = TwiddleFactorsSoA(fft_size)

        # Warmup
        _ = rfft(signal)
        _ = rfft_simd(signal, twiddles)

        # Benchmark
        var aos_time = benchmark_rfft_aos(signal, iterations)
        var simd_time = benchmark_rfft_simd(signal, twiddles, iterations)

        var speedup = aos_time / simd_time

        print("    " + String(N) + "       " +
              String(aos_time)[:6] + "      " +
              String(simd_time)[:6] + "      " +
              String(speedup)[:4] + "x")

    print("\n" + "="*70)
    print("Benchmark complete!")
    print("="*70 + "\n")
