#!/usr/bin/env python3
"""
Combined benchmark: mojo-audio vs librosa

Runs both benchmarks back-to-back under identical conditions
and produces a comparison table.
"""

import subprocess
import re
import sys
from pathlib import Path

# Import librosa benchmark functions
from compare_librosa import (
    generate_chirp,
    calculate_stats,
    calculate_stats_excluding_outliers,
    LIBROSA_AVAILABLE
)

if LIBROSA_AVAILABLE:
    import librosa
import numpy as np
import time


def run_mojo_benchmark() -> dict:
    """Run Mojo benchmark and parse results."""
    print("=" * 70)
    print("Running mojo-audio benchmark...")
    print("=" * 70)
    print()

    # Run the Mojo benchmark
    result = subprocess.run(
        ["pixi", "run", "bench-optimized"],
        capture_output=True,
        text=True,
        cwd=Path(__file__).parent.parent
    )

    print(result.stdout)
    if result.stderr:
        print(result.stderr, file=sys.stderr)

    # Parse results - look for "Results (excluding outliers" lines
    results = {}
    current_duration = None

    for line in result.stdout.split('\n'):
        # Match "Benchmarking X seconds of audio..."
        match = re.search(r'Benchmarking (\d+) seconds', line)
        if match:
            current_duration = int(match.group(1))

        # Match "Mean: X.XXX ms ± Y.YYY ms"
        match = re.search(r'Mean:\s+([\d.]+)\s*ms\s*±\s*([\d.]+)\s*ms', line)
        if match and current_duration:
            results[current_duration] = {
                'mean': float(match.group(1)),
                'std': float(match.group(2))
            }

    return results


def run_librosa_benchmark() -> dict:
    """Run librosa benchmark directly (more reliable than parsing)."""
    print()
    print("=" * 70)
    print("Running librosa benchmark...")
    print("=" * 70)
    print()

    if not LIBROSA_AVAILABLE:
        print("librosa not available!")
        return {}

    results = {}
    sr = 16000
    n_fft = 400
    hop_length = 160
    n_mels = 80
    iterations = 20
    warmup_runs = 5

    for duration in [1, 10, 30]:
        print(f"Benchmarking {duration} seconds of audio...")
        print(f"  Generating chirp signal...")

        audio = generate_chirp(duration * sr, sr)

        # Warmup
        print(f"  Warmup: {warmup_runs} runs...")
        for _ in range(warmup_runs):
            _ = librosa.feature.melspectrogram(
                y=audio, sr=sr, n_fft=n_fft,
                hop_length=hop_length, n_mels=n_mels
            )

        # Benchmark
        print(f"  Benchmarking: {iterations} iterations...")
        times = []
        for _ in range(iterations):
            start = time.perf_counter()
            mel_spec = librosa.feature.melspectrogram(
                y=audio, sr=sr, n_fft=n_fft,
                hop_length=hop_length, n_mels=n_mels
            )
            end = time.perf_counter()
            times.append((end - start) * 1000)

        all_stats = calculate_stats(times)
        trimmed = calculate_stats_excluding_outliers(times)
        throughput = duration / (trimmed[0] / 1000.0)

        print()
        print("  Results (all runs):")
        print(f"    Mean:    {all_stats[0]:.4f} ms")
        print(f"    Std dev: {all_stats[1]:.4f} ms")
        print(f"    Min:     {all_stats[2]:.4f} ms")
        print(f"    Max:     {all_stats[3]:.4f} ms")
        print()
        print(f"  Results (excluding outliers, n={trimmed[2]}):")
        print(f"    Mean:      {trimmed[0]:.4f} ms ± {trimmed[1]:.4f} ms")
        print(f"    Throughput: {throughput:.1f}x realtime")
        print(f"    Output shape: {mel_spec.shape}")
        print()

        results[duration] = {
            'mean': trimmed[0],
            'std': trimmed[1]
        }

    return results


def print_comparison(mojo_results: dict, librosa_results: dict):
    """Print a comparison table."""
    print()
    print("=" * 70)
    print("COMPARISON: mojo-audio vs librosa")
    print("=" * 70)
    print()
    print("Methodology: 5 warmup runs, 20 iterations, outliers excluded")
    print("Signal: Linear chirp 20Hz-8000Hz (deterministic)")
    print()
    print("-" * 70)
    print(f"{'Duration':<10} {'mojo-audio':<20} {'librosa':<20} {'Comparison':<20}")
    print("-" * 70)

    for duration in [1, 10, 30]:
        mojo = mojo_results.get(duration, {})
        lib = librosa_results.get(duration, {})

        mojo_str = f"{mojo.get('mean', 0):.2f} ± {mojo.get('std', 0):.2f} ms" if mojo else "N/A"
        lib_str = f"{lib.get('mean', 0):.2f} ± {lib.get('std', 0):.2f} ms" if lib else "N/A"

        if mojo and lib and mojo['mean'] > 0 and lib['mean'] > 0:
            ratio = lib['mean'] / mojo['mean']
            if ratio > 1:
                comparison = f"Mojo {ratio:.2f}x faster"
            else:
                comparison = f"librosa {1/ratio:.2f}x faster"
        else:
            comparison = "N/A"

        print(f"{duration}s{'':<9} {mojo_str:<20} {lib_str:<20} {comparison:<20}")

    print("-" * 70)
    print()

    # Summary
    if mojo_results and librosa_results:
        print("Summary:")
        for duration in [1, 10, 30]:
            mojo = mojo_results.get(duration, {})
            lib = librosa_results.get(duration, {})
            if mojo and lib and mojo['mean'] > 0 and lib['mean'] > 0:
                mojo_rt = duration / (mojo['mean'] / 1000.0)
                lib_rt = duration / (lib['mean'] / 1000.0)
                print(f"  {duration}s: mojo-audio {mojo_rt:.0f}x RT, librosa {lib_rt:.0f}x RT")
    print()


def main():
    print()
    print("#" * 70)
    print("#  COMBINED BENCHMARK: mojo-audio vs librosa")
    print("#  Running back-to-back for fair comparison")
    print("#" * 70)
    print()

    # Run both benchmarks
    mojo_results = run_mojo_benchmark()
    librosa_results = run_librosa_benchmark()

    # Print comparison
    print_comparison(mojo_results, librosa_results)

    print("=" * 70)
    print("Benchmark complete!")
    print("=" * 70)


if __name__ == "__main__":
    main()
