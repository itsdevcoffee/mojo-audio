"""
Benchmark mel spectrogram performance.

Measures time to compute mel spectrogram for various audio lengths.
Uses deterministic chirp signal for reproducible, fair comparison with librosa.
"""

from audio import mel_spectrogram
from time import perf_counter_ns
from math import sin, pi, sqrt


fn generate_chirp(num_samples: Int, sample_rate: Int = 16000) -> List[Float32]:
    """Generate a linear chirp signal from 20Hz to 8000Hz.

    Deterministic, reproducible, and exercises full frequency range
    relevant to mel spectrogram computation.
    """
    var audio = List[Float32](capacity=num_samples)
    var f0: Float64 = 20.0      # Start frequency (Hz)
    var f1: Float64 = 8000.0    # End frequency (Hz)
    var duration = Float64(num_samples) / Float64(sample_rate)
    var k = (f1 - f0) / duration  # Chirp rate

    for i in range(num_samples):
        var t = Float64(i) / Float64(sample_rate)
        var phase = 2.0 * pi * (f0 * t + 0.5 * k * t * t)
        audio.append(Float32(0.1 * sin(phase)))

    return audio^


struct BenchStats:
    """Statistics from benchmark runs."""
    var mean: Float64
    var std_dev: Float64
    var min_val: Float64
    var max_val: Float64

    fn __init__(out self, mean: Float64, std_dev: Float64, min_val: Float64, max_val: Float64):
        self.mean = mean
        self.std_dev = std_dev
        self.min_val = min_val
        self.max_val = max_val


struct TrimmedStats:
    """Statistics after outlier removal."""
    var mean: Float64
    var std_dev: Float64
    var count: Int

    fn __init__(out self, mean: Float64, std_dev: Float64, count: Int):
        self.mean = mean
        self.std_dev = std_dev
        self.count = count


fn calculate_stats(times: List[Float64]) -> BenchStats:
    """Calculate mean, std dev, min, max from timing data."""
    var n = len(times)
    if n == 0:
        return BenchStats(0.0, 0.0, 0.0, 0.0)

    # Calculate mean
    var sum: Float64 = 0.0
    var min_val = times[0]
    var max_val = times[0]
    for i in range(n):
        sum += times[i]
        if times[i] < min_val:
            min_val = times[i]
        if times[i] > max_val:
            max_val = times[i]
    var mean = sum / Float64(n)

    # Calculate std dev
    var sum_sq: Float64 = 0.0
    for i in range(n):
        var diff = times[i] - mean
        sum_sq += diff * diff
    var std_dev = sqrt(sum_sq / Float64(n))

    return BenchStats(mean, std_dev, min_val, max_val)


fn calculate_stats_excluding_outliers(times: List[Float64]) -> TrimmedStats:
    """Calculate mean and std dev after removing highest and lowest values.

    Returns TrimmedStats with mean, std_dev, and count.
    """
    var n = len(times)
    if n <= 2:
        var stats = calculate_stats(times)
        return TrimmedStats(stats.mean, stats.std_dev, n)

    # Find min and max indices
    var min_idx = 0
    var max_idx = 0
    for i in range(n):
        if times[i] < times[min_idx]:
            min_idx = i
        if times[i] > times[max_idx]:
            max_idx = i

    # Calculate mean excluding outliers
    var sum: Float64 = 0.0
    var count = 0
    for i in range(n):
        if i != min_idx and i != max_idx:
            sum += times[i]
            count += 1

    if count == 0:
        return TrimmedStats(0.0, 0.0, 0)

    var mean = sum / Float64(count)

    # Calculate std dev excluding outliers
    var sum_sq: Float64 = 0.0
    for i in range(n):
        if i != min_idx and i != max_idx:
            var diff = times[i] - mean
            sum_sq += diff * diff
    var std_dev = sqrt(sum_sq / Float64(count))

    return TrimmedStats(mean, std_dev, count)


fn benchmark_mel_spec(audio_seconds: Int, iterations: Int = 20, warmup_runs: Int = 5) raises:
    """Benchmark mel spectrogram for given audio length."""
    print("Benchmarking", audio_seconds, "seconds of audio...")
    print("  Generating chirp signal...")

    # Generate deterministic chirp signal
    var audio = generate_chirp(audio_seconds * 16000)

    # Warmup runs (not timed)
    print("  Warmup:", warmup_runs, "runs...")
    for _ in range(warmup_runs):
        _ = mel_spectrogram(audio)

    # Benchmark with per-iteration timing
    print("  Benchmarking:", iterations, "iterations...")
    var times = List[Float64](capacity=iterations)

    for _ in range(iterations):
        var start = perf_counter_ns()
        _ = mel_spectrogram(audio)
        var end = perf_counter_ns()
        times.append(Float64(end - start) / 1_000_000.0)  # Convert to ms

    # Calculate statistics
    var all_stats = calculate_stats(times)
    var trimmed = calculate_stats_excluding_outliers(times)

    var throughput = Float64(audio_seconds) / (trimmed.mean / 1000.0)

    print()
    print("  Results (all runs):")
    print("    Mean:    ", all_stats.mean, "ms")
    print("    Std dev: ", all_stats.std_dev, "ms")
    print("    Min:     ", all_stats.min_val, "ms")
    print("    Max:     ", all_stats.max_val, "ms")
    print()
    print("  Results (excluding outliers, n=" + String(trimmed.count) + "):")
    print("    Mean:      ", trimmed.mean, "ms ±", trimmed.std_dev, "ms")
    print("    Throughput:", throughput, "x realtime")
    print()


fn main() raises:
    print()
    print("=" * 70)
    print("Mel Spectrogram Performance Benchmark (mojo-audio)")
    print("=" * 70)
    print()
    print("Signal: Linear chirp 20Hz-8000Hz (deterministic)")
    print("Sample rate: 16000 Hz")
    print("Parameters: n_fft=400, hop_length=160, n_mels=80")
    print()

    # Benchmark different lengths with consistent methodology
    benchmark_mel_spec(1, iterations=20, warmup_runs=5)
    benchmark_mel_spec(10, iterations=20, warmup_runs=5)
    benchmark_mel_spec(30, iterations=20, warmup_runs=5)

    print("=" * 70)
    print("Benchmark complete!")
    print()
    print("Compare with librosa:")
    print("  pixi run bench-python")
    print("=" * 70)
    print()
