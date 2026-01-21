#!/usr/bin/env python3
"""
Stable benchmark: Run full comparison N times and report median.

This wrapper runs the benchmarks multiple times to reduce system noise
and reports the median result for more reliable comparisons.

Usage:
    python benchmarks/bench_stable.py [num_runs]
    pixi run bench-stable          # Default: 5 runs
    pixi run bench-stable 10       # Custom: 10 runs
"""

import subprocess
import re
import sys
import statistics
from pathlib import Path

# Import shared functions from compare_librosa
from compare_librosa import (
    generate_chirp,
    calculate_stats_excluding_outliers,
    LIBROSA_AVAILABLE
)

if LIBROSA_AVAILABLE:
    import librosa
import numpy as np
import time


def run_mojo_benchmark(duration: int, iterations: int = 20, warmup: int = 5) -> float:
    """Run Mojo benchmark and return trimmed mean time in ms."""
    result = subprocess.run(
        ["pixi", "run", "bench-optimized"],
        capture_output=True,
        text=True,
        cwd=Path(__file__).parent.parent
    )

    # Parse the output for the specific duration
    pattern = rf"Benchmarking {duration} seconds.*?Results \(excluding outliers.*?Mean:\s+([\d.]+)\s*ms"
    match = re.search(pattern, result.stdout, re.DOTALL)
    if match:
        return float(match.group(1))
    return None


def run_librosa_benchmark(duration: int, iterations: int = 20, warmup: int = 5) -> float:
    """Run librosa benchmark directly and return trimmed mean time in ms."""
    if not LIBROSA_AVAILABLE:
        return None

    sr = 16000
    n_fft = 400
    hop_length = 160
    n_mels = 80

    audio = generate_chirp(duration * sr, sr)

    # Warmup
    for _ in range(warmup):
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

    trimmed = calculate_stats_excluding_outliers(times)
    return trimmed[0]


def run_single_pass() -> dict:
    """Run a single benchmark pass for all durations."""
    results = {}

    # Run Mojo benchmark once (it does all durations)
    mojo_output = subprocess.run(
        ["pixi", "run", "bench-optimized"],
        capture_output=True,
        text=True,
        cwd=Path(__file__).parent.parent
    )

    # Parse Mojo results for each duration
    for duration in [1, 10, 30]:
        pattern = rf"Benchmarking {duration} seconds.*?Results \(excluding outliers.*?Mean:\s+([\d.]+)\s*ms"
        match = re.search(pattern, mojo_output.stdout, re.DOTALL)
        if match:
            results[f"mojo_{duration}"] = float(match.group(1))

    # Run librosa benchmarks
    for duration in [1, 10, 30]:
        lib_time = run_librosa_benchmark(duration)
        if lib_time:
            results[f"librosa_{duration}"] = lib_time

    return results


def run_stable_benchmark(num_runs: int = 5):
    """Run the full benchmark num_runs times and report median."""
    print()
    print("#" * 70)
    print(f"#  STABLE BENCHMARK: {num_runs} complete runs, reporting median")
    print("#" * 70)
    print()

    durations = [1, 10, 30]
    all_results = {d: {'mojo': [], 'librosa': []} for d in durations}

    for run in range(1, num_runs + 1):
        print(f"Run {run}/{num_runs}...")

        results = run_single_pass()

        for duration in durations:
            mojo_key = f"mojo_{duration}"
            lib_key = f"librosa_{duration}"

            if mojo_key in results:
                all_results[duration]['mojo'].append(results[mojo_key])
            if lib_key in results:
                all_results[duration]['librosa'].append(results[lib_key])

        # Print this run's results
        for duration in durations:
            mojo_val = all_results[duration]['mojo'][-1] if all_results[duration]['mojo'] else 0
            lib_val = all_results[duration]['librosa'][-1] if all_results[duration]['librosa'] else 0
            print(f"  {duration}s: Mojo {mojo_val:.2f}ms, librosa {lib_val:.2f}ms")
        print()

    # Calculate and report medians
    print("=" * 70)
    print("STABLE RESULTS (median of all runs)")
    print("=" * 70)
    print()
    print(f"{'Duration':<10} {'mojo-audio':<20} {'librosa':<20} {'Comparison':<20}")
    print("-" * 70)

    for duration in durations:
        mojo_times = all_results[duration]['mojo']
        lib_times = all_results[duration]['librosa']

        if mojo_times and lib_times:
            mojo_median = statistics.median(mojo_times)
            lib_median = statistics.median(lib_times)
            mojo_stdev = statistics.stdev(mojo_times) if len(mojo_times) > 1 else 0
            lib_stdev = statistics.stdev(lib_times) if len(lib_times) > 1 else 0

            ratio = lib_median / mojo_median
            if ratio > 1:
                comparison = f"Mojo {ratio:.2f}x faster"
            else:
                comparison = f"librosa {1/ratio:.2f}x faster"

            print(f"{duration}s{'':<9} {mojo_median:.2f} ± {mojo_stdev:.2f} ms{'':<5} {lib_median:.2f} ± {lib_stdev:.2f} ms{'':<5} {comparison}")

    print("-" * 70)
    print()

    # Show all individual runs for transparency
    print("Individual runs:")
    for duration in durations:
        mojo_runs = ", ".join(f"{t:.2f}" for t in all_results[duration]['mojo'])
        lib_runs = ", ".join(f"{t:.2f}" for t in all_results[duration]['librosa'])
        print(f"  {duration}s Mojo:    [{mojo_runs}] ms")
        print(f"  {duration}s librosa: [{lib_runs}] ms")
    print()

    # Variance analysis
    print("Run-to-run variance (coefficient of variation):")
    for duration in durations:
        mojo_times = all_results[duration]['mojo']
        lib_times = all_results[duration]['librosa']
        if len(mojo_times) > 1 and len(lib_times) > 1:
            mojo_cv = (statistics.stdev(mojo_times) / statistics.mean(mojo_times)) * 100
            lib_cv = (statistics.stdev(lib_times) / statistics.mean(lib_times)) * 100
            print(f"  {duration}s: Mojo {mojo_cv:.1f}%, librosa {lib_cv:.1f}%")
    print()

    print("=" * 70)
    print("Stable benchmark complete!")
    print("=" * 70)


def main():
    num_runs = 5
    if len(sys.argv) > 1:
        try:
            num_runs = int(sys.argv[1])
        except ValueError:
            print(f"Invalid number of runs: {sys.argv[1]}")
            sys.exit(1)

    if num_runs < 2:
        print("Need at least 2 runs for meaningful statistics")
        sys.exit(1)

    run_stable_benchmark(num_runs)


if __name__ == "__main__":
    main()
