#!/usr/bin/env python3
"""
Generate pre-computed benchmark data for demo mode.

Runs all 24 configuration combinations and saves results to JSON.
This data is used for the static demo deployment.

Usage:
    python generate_demo_data.py [--machine MACHINE_ID] [--cpu CPU_NAME] [--cores CORES]

Examples:
    python generate_demo_data.py --machine intel-i7-1360p --cpu "Intel i7-1360P" --cores 12
    python generate_demo_data.py --machine nvidia-dgx-spark --cpu "ARM Neoverse" --cores 20
"""

import argparse
import json
import subprocess
import sys
from pathlib import Path
from typing import Dict, List

# Configuration matrix
DURATIONS = [1, 10, 30]
FFT_SIZES = [256, 400, 512, 1024]
BLAS_BACKENDS = ["mkl", "openblas"]
ITERATIONS = 20  # Fixed for pre-generation

REPO_ROOT = Path(__file__).parent.parent.parent


def run_benchmark(impl: str, duration: int, fft_size: int, blas: str = "mkl") -> Dict:
    """Run a single benchmark and return results."""
    print(f"  Running {impl} benchmark: {duration}s, FFT={fft_size}, BLAS={blas}...")

    if impl == "mojo":
        cmd = [
            "python", "ui/backend/run_benchmark.py",
            "mojo", str(duration), str(ITERATIONS),
            str(fft_size), "160", "80"  # hop_length=160, n_mels=80
        ]
    else:  # librosa
        cmd = [
            "pixi", "run", "-e", blas,
            "python", "ui/backend/run_benchmark.py",
            "librosa", str(duration), str(ITERATIONS),
            str(fft_size), "160", "80"
        ]

    try:
        result = subprocess.run(
            cmd,
            cwd=REPO_ROOT,
            capture_output=True,
            text=True,
            timeout=180
        )

        if result.returncode != 0:
            print(f"    ❌ Failed: {result.stderr}")
            return None

        # Parse output (format: "avg_ms,std_ms")
        parts = result.stdout.strip().split(',')
        avg_time = float(parts[0])
        std_time = float(parts[1]) if len(parts) > 1 else 0.0

        throughput = duration / (avg_time / 1000.0)

        print(f"    ✅ {avg_time:.2f}ms ± {std_time:.2f}ms ({throughput:.0f}x realtime)")

        return {
            "avg_time_ms": round(avg_time, 2),
            "std_time_ms": round(std_time, 2),
            "throughput_realtime": round(throughput, 1)
        }

    except subprocess.TimeoutExpired:
        print(f"    ⏱️  Timeout!")
        return None
    except Exception as e:
        print(f"    ❌ Error: {e}")
        return None


def generate_all_benchmarks() -> Dict:
    """Generate all 24 benchmark configurations."""
    results = {}
    total = len(DURATIONS) * len(FFT_SIZES) * len(BLAS_BACKENDS)
    current = 0

    print(f"🚀 Generating {total} benchmark configurations...\n")

    for duration in DURATIONS:
        for fft_size in FFT_SIZES:
            for blas in BLAS_BACKENDS:
                current += 1
                config_key = f"{duration}s_{fft_size}fft_{blas}"

                print(f"[{current}/{total}] {config_key}")

                # Run Mojo benchmark (BLAS doesn't affect Mojo, but we track it)
                mojo_result = run_benchmark("mojo", duration, fft_size, blas)

                # Run librosa benchmark
                librosa_result = run_benchmark("librosa", duration, fft_size, blas)

                if mojo_result and librosa_result:
                    # Calculate speedup
                    speedup = librosa_result["avg_time_ms"] / mojo_result["avg_time_ms"]
                    faster_pct = ((librosa_result["avg_time_ms"] - mojo_result["avg_time_ms"])
                                  / librosa_result["avg_time_ms"]) * 100

                    results[config_key] = {
                        "config": {
                            "duration": duration,
                            "fft_size": fft_size,
                            "blas": blas,
                            "iterations": ITERATIONS
                        },
                        "mojo": mojo_result,
                        "librosa": librosa_result,
                        "speedup_factor": round(speedup, 2),
                        "faster_percentage": round(faster_pct, 1),
                        "mojo_is_faster": mojo_result["avg_time_ms"] < librosa_result["avg_time_ms"]
                    }

                    print(f"  📊 Speedup: {speedup:.2f}x ({faster_pct:+.1f}%)")
                else:
                    print(f"  ⚠️  Skipping {config_key} due to errors")

                print()

    return results


def save_results(results: Dict, output_file: Path):
    """Save results to JSON file."""
    output_file.parent.mkdir(parents=True, exist_ok=True)

    with open(output_file, 'w') as f:
        json.dump(results, f, indent=2)

    print(f"\n✅ Saved {len(results)} benchmark results to: {output_file}")
    print(f"📦 File size: {output_file.stat().st_size / 1024:.1f} KB")


def print_summary(results: Dict):
    """Print summary statistics."""
    if not results:
        return

    print("\n" + "="*60)
    print("📊 SUMMARY")
    print("="*60)

    speedups = [r["speedup_factor"] for r in results.values()]
    faster_pcts = [r["faster_percentage"] for r in results.values()]

    print(f"Total configurations: {len(results)}")
    print(f"Average speedup: {sum(speedups)/len(speedups):.2f}x")
    print(f"Min speedup: {min(speedups):.2f}x")
    print(f"Max speedup: {max(speedups):.2f}x")
    print(f"Average faster: {sum(faster_pcts)/len(faster_pcts):.1f}%")
    print(f"Mojo wins: {sum(1 for r in results.values() if r['mojo_is_faster'])}/{len(results)}")
    print()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate benchmark data for demo mode")
    parser.add_argument("--machine", default="default", help="Machine identifier (e.g., intel-i7-1360p)")
    parser.add_argument("--cpu", default="Unknown CPU", help="CPU name for metadata")
    parser.add_argument("--cores", type=int, default=0, help="Number of CPU cores")
    parser.add_argument("--platform", default="linux-x64", help="Platform (e.g., linux-x64, linux-arm64)")
    args = parser.parse_args()

    # Determine output filename
    if args.machine == "default":
        output_file = REPO_ROOT / "ui" / "static" / "data" / "benchmark_results.json"
    else:
        output_file = REPO_ROOT / "ui" / "static" / "data" / f"benchmark_results_{args.machine}.json"

    print("mojo-audio Benchmark Data Generator")
    print("=" * 60)
    print(f"Machine: {args.machine}")
    print(f"CPU: {args.cpu}")
    print(f"Cores: {args.cores}")
    print(f"Platform: {args.platform}")
    print()

    # Generate all benchmarks
    results = generate_all_benchmarks()

    # Add metadata
    metadata = {
        "machine_id": args.machine,
        "cpu_name": args.cpu,
        "cores": args.cores,
        "platform": args.platform,
        "generated_at": "2026-02-04",
        "benchmarks": results
    }

    # Save to file
    save_results(metadata, output_file)

    # Print summary
    print_summary(results)

    print("🎉 Done! Use this data for static demo deployment.")
    print(f"   File: {output_file}")
    print(f"   Import: fetch('/static/data/{output_file.name}')")
