"""
Benchmark runner with configurable audio signals and methodology.

Supports multiple signal types for fair, reproducible benchmarking.
"""

import sys
import numpy as np
import time

# Signal type constants
SIGNAL_RANDOM = "random"
SIGNAL_CHIRP = "chirp"
SIGNAL_SINE = "sine"
SIGNAL_WHITE_NOISE = "white_noise"
SIGNAL_MULTI_TONE = "multi_tone"


def generate_signal(signal_type: str, num_samples: int, sample_rate: int = 16000, seed: int = 42) -> np.ndarray:
    """
    Generate audio signal of specified type.

    Args:
        signal_type: One of 'random', 'chirp', 'sine', 'white_noise', 'multi_tone'
        num_samples: Number of samples to generate
        sample_rate: Sample rate in Hz
        seed: Random seed (only used for 'random' type)

    Returns:
        numpy array of float32 audio samples
    """
    if signal_type == SIGNAL_RANDOM:
        # Seeded random - reproducible
        np.random.seed(seed)
        return (np.random.rand(num_samples).astype(np.float32) * 0.1)

    elif signal_type == SIGNAL_CHIRP:
        # Linear chirp 20Hz to 8000Hz - exercises full frequency range
        f0, f1 = 20.0, 8000.0
        duration = num_samples / sample_rate
        k = (f1 - f0) / duration
        t = np.arange(num_samples) / sample_rate
        phase = 2.0 * np.pi * (f0 * t + 0.5 * k * t * t)
        return (0.1 * np.sin(phase)).astype(np.float32)

    elif signal_type == SIGNAL_SINE:
        # Pure 440Hz tone - minimal spectral content, predictable
        t = np.arange(num_samples) / sample_rate
        return (0.1 * np.sin(2.0 * np.pi * 440.0 * t)).astype(np.float32)

    elif signal_type == SIGNAL_WHITE_NOISE:
        # Unseeded white noise - different each run, realistic
        return (np.random.rand(num_samples).astype(np.float32) * 0.1)

    elif signal_type == SIGNAL_MULTI_TONE:
        # Multiple frequencies like speech formants (F1=500Hz, F2=1500Hz, F3=2500Hz)
        t = np.arange(num_samples) / sample_rate
        signal = (
            0.04 * np.sin(2.0 * np.pi * 150.0 * t) +   # Fundamental
            0.03 * np.sin(2.0 * np.pi * 500.0 * t) +   # F1
            0.02 * np.sin(2.0 * np.pi * 1500.0 * t) +  # F2
            0.01 * np.sin(2.0 * np.pi * 2500.0 * t)    # F3
        )
        return signal.astype(np.float32)

    else:
        raise ValueError(f"Unknown signal type: {signal_type}")


def generate_mojo_signal_code(signal_type: str, num_samples: int, sample_rate: int = 16000, seed: int = 42) -> str:
    """Generate Mojo code to create the same signal."""
    if signal_type == SIGNAL_RANDOM:
        return f"""
    seed({seed})
    var audio = List[Float32]()
    for _ in range({num_samples}):
        audio.append(Float32(random_float64(0.0, 0.1)))
"""
    elif signal_type == SIGNAL_CHIRP:
        return f"""
    var audio = List[Float32](capacity={num_samples})
    var f0: Float64 = 20.0
    var f1: Float64 = 8000.0
    var duration = Float64({num_samples}) / Float64({sample_rate})
    var k = (f1 - f0) / duration
    for i in range({num_samples}):
        var t = Float64(i) / Float64({sample_rate})
        var phase = 2.0 * pi * (f0 * t + 0.5 * k * t * t)
        audio.append(Float32(0.1 * sin(phase)))
"""
    elif signal_type == SIGNAL_SINE:
        return f"""
    var audio = List[Float32](capacity={num_samples})
    for i in range({num_samples}):
        var t = Float64(i) / Float64({sample_rate})
        var phase = 2.0 * pi * 440.0 * t
        audio.append(Float32(0.1 * sin(phase)))
"""
    elif signal_type == SIGNAL_WHITE_NOISE:
        # Use time-based seed for different data each run
        return f"""
    seed(perf_counter_ns())
    var audio = List[Float32]()
    for _ in range({num_samples}):
        audio.append(Float32(random_float64(0.0, 0.1)))
"""
    elif signal_type == SIGNAL_MULTI_TONE:
        return f"""
    var audio = List[Float32](capacity={num_samples})
    for i in range({num_samples}):
        var t = Float64(i) / Float64({sample_rate})
        var sample = (
            0.04 * sin(2.0 * pi * 150.0 * t) +
            0.03 * sin(2.0 * pi * 500.0 * t) +
            0.02 * sin(2.0 * pi * 1500.0 * t) +
            0.01 * sin(2.0 * pi * 2500.0 * t)
        )
        audio.append(Float32(sample))
"""
    else:
        raise ValueError(f"Unknown signal type: {signal_type}")


def benchmark_mojo_single(
    duration_s: int,
    iterations: int = 20,
    warmup_runs: int = 5,
    n_fft: int = 400,
    hop_length: int = 160,
    n_mels: int = 80,
    signal_type: str = SIGNAL_CHIRP,
    seed: int = 42
):
    """
    Run mojo mel_spectrogram benchmark.

    Returns: (average time in ms, std dev in ms)
    """
    import subprocess

    num_samples = duration_s * 16000
    signal_code = generate_mojo_signal_code(signal_type, num_samples, 16000, seed)

    # Determine imports needed
    imports = "from audio import mel_spectrogram\nfrom time import perf_counter_ns\nfrom math import sqrt, sin, pi"
    if signal_type in [SIGNAL_RANDOM, SIGNAL_WHITE_NOISE]:
        imports += "\nfrom random import seed, random_float64"

    mojo_code = f"""
{imports}

fn main() raises:
    # Generate audio signal
{signal_code}

    # Warmup runs
    for _ in range({warmup_runs}):
        _ = mel_spectrogram(audio, n_fft={n_fft}, hop_length={hop_length}, n_mels={n_mels})

    # Benchmark - collect per-iteration times
    var times = List[Float64]()
    for _ in range({iterations}):
        var iter_start = perf_counter_ns()
        var result = mel_spectrogram(audio, n_fft={n_fft}, hop_length={hop_length}, n_mels={n_mels})
        var iter_end = perf_counter_ns()
        times.append(Float64(iter_end - iter_start) / 1_000_000.0)
        var checksum = result[0][0]  # Prevent dead code elimination

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

    with open('/tmp/mojo_bench_temp.mojo', 'w') as f:
        f.write(mojo_code)

    result = subprocess.run(
        ['pixi', 'run', '--', 'mojo', '-O3', '-I', 'src', '/tmp/mojo_bench_temp.mojo'],
        capture_output=True,
        text=True,
        cwd='/home/maskkiller/dev-coffee/repos/mojo-audio'
    )

    if result.returncode != 0:
        print(f"Error: {result.stderr}", file=sys.stderr)
        return None, None

    try:
        line = result.stdout.strip().split('\n')[-1]
        parts = line.split(',')
        avg = float(parts[0])
        std = float(parts[1]) if len(parts) > 1 else 0.0
        return avg, std
    except:
        return None, None


def benchmark_librosa_single(
    duration_s: int,
    iterations: int = 20,
    warmup_runs: int = 5,
    n_fft: int = 400,
    hop_length: int = 160,
    n_mels: int = 80,
    signal_type: str = SIGNAL_CHIRP,
    seed: int = 42
):
    """
    Run librosa mel_spectrogram benchmark.

    Returns: (average time in ms, std dev in ms)
    """
    try:
        import librosa
    except ImportError:
        print("librosa not available", file=sys.stderr)
        return None, None

    sr = 16000
    num_samples = duration_s * sr

    # Generate signal
    audio = generate_signal(signal_type, num_samples, sr, seed)

    # Warmup runs
    for _ in range(warmup_runs):
        _ = librosa.feature.melspectrogram(
            y=audio, sr=sr, n_fft=n_fft,
            hop_length=hop_length, n_mels=n_mels
        )

    # Benchmark
    times = []
    for _ in range(iterations):
        start = time.perf_counter()
        _ = librosa.feature.melspectrogram(
            y=audio, sr=sr, n_fft=n_fft,
            hop_length=hop_length, n_mels=n_mels
        )
        end = time.perf_counter()
        times.append((end - start) * 1000)

    return np.mean(times), np.std(times)


def run_stable_benchmark(
    duration_s: int,
    num_runs: int = 5,
    iterations: int = 20,
    warmup_runs: int = 5,
    n_fft: int = 400,
    hop_length: int = 160,
    n_mels: int = 80,
    signal_type: str = SIGNAL_CHIRP,
    seed: int = 42
):
    """
    Run benchmark num_runs times and return median results.

    Returns: dict with mojo and librosa median results
    """
    import statistics

    mojo_times = []
    librosa_times = []

    for _ in range(num_runs):
        mojo_avg, _ = benchmark_mojo_single(
            duration_s, iterations, warmup_runs,
            n_fft, hop_length, n_mels, signal_type, seed
        )
        if mojo_avg:
            mojo_times.append(mojo_avg)

        lib_avg, _ = benchmark_librosa_single(
            duration_s, iterations, warmup_runs,
            n_fft, hop_length, n_mels, signal_type, seed
        )
        if lib_avg:
            librosa_times.append(lib_avg)

    result = {}
    if mojo_times:
        result['mojo'] = {
            'median_ms': statistics.median(mojo_times),
            'stdev_ms': statistics.stdev(mojo_times) if len(mojo_times) > 1 else 0,
            'all_runs': mojo_times
        }
    if librosa_times:
        result['librosa'] = {
            'median_ms': statistics.median(librosa_times),
            'stdev_ms': statistics.stdev(librosa_times) if len(librosa_times) > 1 else 0,
            'all_runs': librosa_times
        }

    return result


if __name__ == "__main__":
    if len(sys.argv) < 3:
        print("Usage: run_benchmark.py <implementation> <duration> [iterations] [n_fft] [hop_length] [n_mels] [warmup] [signal_type] [seed]")
        sys.exit(1)

    implementation = sys.argv[1]
    duration = int(sys.argv[2])
    iterations = int(sys.argv[3]) if len(sys.argv) > 3 else 20
    n_fft = int(sys.argv[4]) if len(sys.argv) > 4 else 400
    hop_length = int(sys.argv[5]) if len(sys.argv) > 5 else 160
    n_mels = int(sys.argv[6]) if len(sys.argv) > 6 else 80
    warmup = int(sys.argv[7]) if len(sys.argv) > 7 else 5
    signal_type = sys.argv[8] if len(sys.argv) > 8 else SIGNAL_CHIRP
    seed = int(sys.argv[9]) if len(sys.argv) > 9 else 42

    if implementation == "mojo":
        avg, std = benchmark_mojo_single(duration, iterations, warmup, n_fft, hop_length, n_mels, signal_type, seed)
    elif implementation == "librosa":
        avg, std = benchmark_librosa_single(duration, iterations, warmup, n_fft, hop_length, n_mels, signal_type, seed)
    else:
        print(f"Unknown implementation: {implementation}", file=sys.stderr)
        sys.exit(1)

    if avg is not None:
        print(f"{avg:.3f},{std:.3f}")
    else:
        sys.exit(1)
