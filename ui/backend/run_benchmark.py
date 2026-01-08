"""
Simple wrapper to run mojo/librosa benchmarks with specific parameters.

This is called by the FastAPI backend with user-selected configuration.
"""

import sys
import numpy as np
import time

def benchmark_mojo_single(duration_s: int, iterations: int = 3, n_fft: int = 400, hop_length: int = 160, n_mels: int = 80):
    """
    Run mojo mel_spectrogram benchmark for specific duration and parameters.

    Returns: average time in milliseconds
    """
    import subprocess

    # Create simple Mojo script that benchmarks with specified parameters
    mojo_code = f"""
from audio import mel_spectrogram
from time import perf_counter_ns

fn main() raises:
    # Create {duration_s}s audio
    var audio = List[Float32]()
    for _ in range({duration_s * 16000}):
        audio.append(0.1)

    # Warmup with specified parameters
    _ = mel_spectrogram(audio, n_fft={n_fft}, hop_length={hop_length}, n_mels={n_mels})

    # Benchmark with specified parameters
    var start = perf_counter_ns()
    for _ in range({iterations}):
        _ = mel_spectrogram(audio, n_fft={n_fft}, hop_length={hop_length}, n_mels={n_mels})
    var end = perf_counter_ns()

    var avg_ns = (end - start) / {iterations}
    var avg_ms = Float64(avg_ns) / 1_000_000.0

    print(avg_ms)
"""

    # Write temp file
    with open('/tmp/mojo_bench_temp.mojo', 'w') as f:
        f.write(mojo_code)

    # Run with -O3 using pixi (mojo is in pixi environment)
    result = subprocess.run(
        ['pixi', 'run', '-e', 'default', 'mojo', '-O3', '-I', 'src', '/tmp/mojo_bench_temp.mojo'],
        capture_output=True,
        text=True,
        cwd='/home/maskkiller/dev-coffee/repos/mojo-audio'
    )

    if result.returncode != 0:
        print(f"Error: {result.stderr}", file=sys.stderr)
        return None

    # Parse output (just the number)
    try:
        return float(result.stdout.strip().split('\n')[-1])
    except:
        return None


def benchmark_librosa_single(duration_s: int, iterations: int = 3, n_fft: int = 400, hop_length: int = 160, n_mels: int = 80):
    """
    Run librosa mel_spectrogram benchmark for specific duration and parameters.

    Returns: average time in milliseconds
    """
    try:
        import librosa
    except ImportError:
        print("librosa not available", file=sys.stderr)
        return None

    # Create test audio
    sr = 16000
    audio = np.random.rand(duration_s * sr).astype(np.float32) * 0.1

    # Use specified parameters
    # (n_fft, hop_length, n_mels passed as arguments)

    # Warmup with specified parameters
    _ = librosa.feature.melspectrogram(
        y=audio, sr=sr, n_fft=n_fft,
        hop_length=hop_length, n_mels=n_mels
    )

    # Benchmark with specified parameters
    times = []
    for _ in range(iterations):
        start = time.perf_counter()
        _ = librosa.feature.melspectrogram(
            y=audio, sr=sr, n_fft=n_fft,
            hop_length=hop_length, n_mels=n_mels
        )
        end = time.perf_counter()
        times.append((end - start) * 1000)  # Convert to ms

    return np.mean(times)


if __name__ == "__main__":
    if len(sys.argv) < 3:
        print("Usage: run_benchmark.py <implementation> <duration> [iterations] [n_fft] [hop_length] [n_mels]")
        sys.exit(1)

    implementation = sys.argv[1]  # "mojo" or "librosa"
    duration = int(sys.argv[2])
    iterations = int(sys.argv[3]) if len(sys.argv) > 3 else 3
    n_fft = int(sys.argv[4]) if len(sys.argv) > 4 else 400
    hop_length = int(sys.argv[5]) if len(sys.argv) > 5 else 160
    n_mels = int(sys.argv[6]) if len(sys.argv) > 6 else 80

    if implementation == "mojo":
        result = benchmark_mojo_single(duration, iterations, n_fft, hop_length, n_mels)
    elif implementation == "librosa":
        result = benchmark_librosa_single(duration, iterations, n_fft, hop_length, n_mels)
    else:
        print(f"Unknown implementation: {implementation}", file=sys.stderr)
        sys.exit(1)

    if result:
        print(f"{result:.3f}")
    else:
        sys.exit(1)
