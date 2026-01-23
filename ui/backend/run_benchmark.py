"""
Simple wrapper to run mojo/librosa benchmarks with specific parameters.

This is called by the FastAPI backend with user-selected configuration.
"""

import subprocess
import sys
import tempfile
import time
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).parent.parent.parent
SAMPLE_RATE = 16000
RANDOM_SEED = 42

def benchmark_mojo_single(
    duration_s: int,
    iterations: int = 20,
    n_fft: int = 400,
    hop_length: int = 160,
    n_mels: int = 80
) -> tuple[float | None, float | None]:
    """
    Run mojo mel_spectrogram benchmark for specific duration and parameters.

    Uses random seed for reproducibility (same as librosa).
    Returns: (average time in ms, std dev in ms) or (None, None) on failure.
    """
    num_samples = duration_s * SAMPLE_RATE

    mojo_code = f"""
from audio import mel_spectrogram
from time import perf_counter_ns
from random import seed, random_float64
from math import sqrt

fn main() raises:
    seed({RANDOM_SEED})
    var audio = List[Float32]()
    for _ in range({num_samples}):
        audio.append(Float32(random_float64(0.0, 0.1)))

    # Warmup
    _ = mel_spectrogram(audio, n_fft={n_fft}, hop_length={hop_length}, n_mels={n_mels})

    # Benchmark - collect per-iteration times
    var times = List[Float64]()
    for _ in range({iterations}):
        var iter_start = perf_counter_ns()
        var result = mel_spectrogram(audio, n_fft={n_fft}, hop_length={hop_length}, n_mels={n_mels})
        var iter_end = perf_counter_ns()
        times.append(Float64(iter_end - iter_start) / 1_000_000.0)
        # Prevent dead code elimination
        var checksum = result[0][0]

    # Calculate mean
    var sum_times: Float64 = 0.0
    for i in range(len(times)):
        sum_times += times[i]
    var avg_ms = sum_times / Float64(len(times))

    # Calculate std dev
    var sum_sq_diff: Float64 = 0.0
    for i in range(len(times)):
        var diff = times[i] - avg_ms
        sum_sq_diff += diff * diff
    var std_ms = sqrt(sum_sq_diff / Float64(len(times)))

    print(String(avg_ms) + "," + String(std_ms))
"""

    with tempfile.NamedTemporaryFile(mode='w', suffix='.mojo', delete=False) as f:
        temp_path = Path(f.name)
        f.write(mojo_code)

    try:
        result = subprocess.run(
            ['pixi', 'run', '--', 'mojo', '-O3', '-I', 'src', str(temp_path)],
            capture_output=True,
            text=True,
            cwd=REPO_ROOT
        )

        if result.returncode != 0:
            print(f"Error: {result.stderr}", file=sys.stderr)
            return None, None

        return parse_benchmark_output(result.stdout)
    finally:
        if temp_path.exists():
            temp_path.unlink()


def parse_benchmark_output(stdout: str) -> tuple[float | None, float | None]:
    """Parse benchmark output in avg,std format from last line of stdout."""
    try:
        line = stdout.strip().split('\n')[-1]
        parts = line.split(',')
        avg = float(parts[0])
        std = float(parts[1]) if len(parts) > 1 else 0.0
        return avg, std
    except (ValueError, IndexError):
        return None, None


def benchmark_librosa_single(
    duration_s: int,
    iterations: int = 20,
    n_fft: int = 400,
    hop_length: int = 160,
    n_mels: int = 80
) -> tuple[float | None, float | None]:
    """
    Run librosa mel_spectrogram benchmark for specific duration and parameters.

    Uses random seed for reproducibility (same as mojo).
    Returns: (average time in ms, std dev in ms) or (None, None) on failure.
    """
    try:
        import librosa
    except ImportError:
        print("librosa not available", file=sys.stderr)
        return None, None

    np.random.seed(RANDOM_SEED)
    audio = np.random.rand(duration_s * SAMPLE_RATE).astype(np.float32) * 0.1

    # Warmup
    _ = librosa.feature.melspectrogram(
        y=audio, sr=SAMPLE_RATE, n_fft=n_fft,
        hop_length=hop_length, n_mels=n_mels
    )

    # Benchmark
    times = []
    for _ in range(iterations):
        start = time.perf_counter()
        _ = librosa.feature.melspectrogram(
            y=audio, sr=SAMPLE_RATE, n_fft=n_fft,
            hop_length=hop_length, n_mels=n_mels
        )
        elapsed_ms = (time.perf_counter() - start) * 1000
        times.append(elapsed_ms)

    return float(np.mean(times)), float(np.std(times))


def get_arg(index: int, default: int) -> int:
    """Get command line argument at index, or return default."""
    return int(sys.argv[index]) if len(sys.argv) > index else default


def main() -> int:
    """Run benchmark based on command line arguments. Returns exit code."""
    if len(sys.argv) < 3:
        print("Usage: run_benchmark.py <implementation> <duration> [iterations] [n_fft] [hop_length] [n_mels]")
        return 1

    implementation = sys.argv[1]
    duration = int(sys.argv[2])
    iterations = get_arg(3, 3)
    n_fft = get_arg(4, 400)
    hop_length = get_arg(5, 160)
    n_mels = get_arg(6, 80)

    benchmark_fn = {
        "mojo": benchmark_mojo_single,
        "librosa": benchmark_librosa_single
    }.get(implementation)

    if benchmark_fn is None:
        print(f"Unknown implementation: {implementation}", file=sys.stderr)
        return 1

    avg, std = benchmark_fn(duration, iterations, n_fft, hop_length, n_mels)

    if avg is None:
        return 1

    print(f"{avg:.3f},{std:.3f}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
