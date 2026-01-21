"""
Compare mojo-audio performance against librosa (Python standard).

Uses identical methodology to Mojo benchmark:
- Deterministic chirp signal (20Hz-8000Hz)
- 5 warmup runs
- 20 iterations with per-iteration timing
- Outlier rejection (drop min/max)
"""

import numpy as np
import time

try:
    import librosa
    LIBROSA_AVAILABLE = True
except ImportError:
    LIBROSA_AVAILABLE = False
    print("librosa not installed - install with: pip install librosa")


def generate_chirp(num_samples: int, sample_rate: int = 16000) -> np.ndarray:
    """Generate a linear chirp signal from 20Hz to 8000Hz.

    Deterministic, reproducible, and exercises full frequency range
    relevant to mel spectrogram computation.
    """
    f0 = 20.0      # Start frequency (Hz)
    f1 = 8000.0    # End frequency (Hz)
    duration = num_samples / sample_rate
    k = (f1 - f0) / duration  # Chirp rate

    t = np.arange(num_samples) / sample_rate
    phase = 2.0 * np.pi * (f0 * t + 0.5 * k * t * t)
    return (0.1 * np.sin(phase)).astype(np.float32)


def calculate_stats(times: list) -> tuple:
    """Calculate mean, std dev, min, max from timing data."""
    times = np.array(times)
    return (
        np.mean(times),
        np.std(times),
        np.min(times),
        np.max(times)
    )


def calculate_stats_excluding_outliers(times: list) -> tuple:
    """Calculate mean and std dev after removing highest and lowest values.

    Returns (mean, std_dev, num_used).
    """
    times = np.array(times)
    if len(times) <= 2:
        return (np.mean(times), np.std(times), len(times))

    # Remove min and max
    min_idx = np.argmin(times)
    max_idx = np.argmax(times)
    mask = np.ones(len(times), dtype=bool)
    mask[min_idx] = False
    mask[max_idx] = False
    trimmed = times[mask]

    return (np.mean(trimmed), np.std(trimmed), len(trimmed))


def benchmark_librosa_mel(duration_seconds: int, iterations: int = 20, warmup_runs: int = 5):
    """Benchmark librosa mel spectrogram."""
    if not LIBROSA_AVAILABLE:
        return None

    print(f"Benchmarking {duration_seconds} seconds of audio...")
    print(f"  Generating chirp signal...")

    # Generate deterministic chirp signal
    sr = 16000
    audio = generate_chirp(duration_seconds * sr, sr)

    # Whisper parameters
    n_fft = 400
    hop_length = 160
    n_mels = 80

    # Warmup runs (not timed)
    print(f"  Warmup: {warmup_runs} runs...")
    for _ in range(warmup_runs):
        _ = librosa.feature.melspectrogram(
            y=audio,
            sr=sr,
            n_fft=n_fft,
            hop_length=hop_length,
            n_mels=n_mels
        )

    # Benchmark with per-iteration timing
    print(f"  Benchmarking: {iterations} iterations...")
    times = []
    mel_spec = None

    for _ in range(iterations):
        start = time.perf_counter()
        mel_spec = librosa.feature.melspectrogram(
            y=audio,
            sr=sr,
            n_fft=n_fft,
            hop_length=hop_length,
            n_mels=n_mels
        )
        end = time.perf_counter()
        times.append((end - start) * 1000)  # Convert to ms

    # Calculate statistics
    all_stats = calculate_stats(times)
    trimmed = calculate_stats_excluding_outliers(times)

    throughput = duration_seconds / (trimmed[0] / 1000.0)

    print()
    print("  Results (all runs):")
    print(f"    Mean:    {all_stats[0]:.4f} ms")
    print(f"    Std dev: {all_stats[1]:.4f} ms")
    print(f"    Min:     {all_stats[2]:.4f} ms")
    print(f"    Max:     {all_stats[3]:.4f} ms")
    print()
    print(f"  Results (excluding outliers, n={trimmed[2]}):")
    print(f"    Mean:      {trimmed[0]:.4f} ms \u00b1 {trimmed[1]:.4f} ms")
    print(f"    Throughput: {throughput:.1f}x realtime")
    print(f"    Output shape: {mel_spec.shape}")
    print()

    return trimmed[0]  # Return mean for comparison


def main():
    print()
    print("=" * 70)
    print("Mel Spectrogram Performance Benchmark (librosa)")
    print("=" * 70)
    print()

    if not LIBROSA_AVAILABLE:
        print("Please install librosa to run benchmarks:")
        print("  pip install librosa")
        return

    print("Signal: Linear chirp 20Hz-8000Hz (deterministic)")
    print("Sample rate: 16000 Hz")
    print("Parameters: n_fft=400, hop_length=160, n_mels=80")
    print()

    # Benchmark different lengths with consistent methodology
    benchmark_librosa_mel(1, iterations=20, warmup_runs=5)
    benchmark_librosa_mel(10, iterations=20, warmup_runs=5)
    benchmark_librosa_mel(30, iterations=20, warmup_runs=5)

    print("=" * 70)
    print("Benchmark complete!")
    print()
    print("Compare with mojo-audio:")
    print("  pixi run bench-optimized")
    print("=" * 70)
    print()


if __name__ == "__main__":
    main()
