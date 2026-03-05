#!/usr/bin/env python3
"""
Benchmark AudioEncoder (MAX Graph) vs PyTorch HuBERT.

Tests: PyTorch CPU, PyTorch GPU (if available), MAX Engine CPU, MAX Engine GPU (if available).
Writes results to experiments/contentvec-max/benchmark_results.md.

Run with: pixi run bench-models
"""

import sys
import os
import time
import platform
import subprocess

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../../src"))

import numpy as np
import torch
from transformers import HubertModel
from models import AudioEncoder

MODEL_ID = "facebook/hubert-base-ls960"
N_WARMUP = 3
N_ITERS = 20
AUDIO = np.random.default_rng(42).standard_normal((1, 16000)).astype(np.float32)


def bench(fn, n_warmup=N_WARMUP, n_iters=N_ITERS):
    """Warm up then time n_iters runs. Returns latency array in ms."""
    for _ in range(n_warmup):
        fn()
    times = []
    for _ in range(n_iters):
        t0 = time.perf_counter()
        fn()
        times.append((time.perf_counter() - t0) * 1000)
    return np.array(times)


def benchmark_pytorch_cpu():
    print("Benchmarking PyTorch CPU...")
    model = HubertModel.from_pretrained(MODEL_ID).eval()
    inp = torch.from_numpy(AUDIO)
    lats = bench(lambda: model(inp))
    print(f"  mean: {lats.mean():.1f} ms")
    return lats


def benchmark_pytorch_gpu():
    if not torch.cuda.is_available():
        print("PyTorch GPU: CUDA not available, skipping")
        return None
    print("Benchmarking PyTorch GPU...")
    model = HubertModel.from_pretrained(MODEL_ID).eval().cuda()
    inp = torch.from_numpy(AUDIO).cuda()
    lats = bench(lambda: (model(inp), torch.cuda.synchronize()))
    print(f"  mean: {lats.mean():.1f} ms")
    return lats


def benchmark_max_cpu():
    print("Benchmarking MAX Engine CPU...")
    model = AudioEncoder.from_pretrained(MODEL_ID, device="cpu")
    lats = bench(lambda: model.encode(AUDIO))
    print(f"  mean: {lats.mean():.1f} ms")
    return lats


def benchmark_max_gpu():
    from max.driver import accelerator_count
    if accelerator_count() == 0:
        print("MAX Engine GPU: no accelerator found, skipping")
        return None
    print("Benchmarking MAX Engine GPU...")
    model = AudioEncoder.from_pretrained(MODEL_ID, device="gpu")
    lats = bench(lambda: model.encode(AUDIO))
    print(f"  mean: {lats.mean():.1f} ms")
    return lats


def get_gpu_info():
    try:
        result = subprocess.run(
            ["nvidia-smi", "--query-gpu=name,compute_cap", "--format=csv,noheader"],
            capture_output=True, text=True, timeout=5,
        )
        return result.stdout.strip() or "N/A"
    except Exception:
        return "N/A"


def print_and_write_table(results: dict, output_path: str):
    baseline = None
    for lats in results.values():
        if lats is not None:
            baseline = lats.mean()
            break

    # Terminal output
    print()
    print("=" * 68)
    print(f"AudioEncoder Benchmark — {MODEL_ID}")
    print(f"Platform: {platform.machine()}  Python {platform.python_version()}")
    print(f"GPU: {get_gpu_info()}")
    print(f"Input: {AUDIO.shape[1]} samples (1s @16kHz), batch=1")
    print(f"Iters: {N_ITERS} (after {N_WARMUP} warmup)")
    print()
    print(f"{'Backend':<26} {'Mean ms':>9} {'Std ms':>8} {'P95 ms':>8} {'vs PT-CPU':>10}")
    print("-" * 65)
    for name, lats in results.items():
        if lats is None:
            print(f"{name:<26} {'N/A':>9}")
            continue
        mean, std, p95 = lats.mean(), lats.std(), np.percentile(lats, 95)
        speedup = baseline / mean if baseline else 1.0
        print(f"{name:<26} {mean:>9.1f} {std:>8.1f} {p95:>8.1f} {speedup:>9.2f}x")
    print("=" * 68)

    # Write markdown
    with open(output_path, "w") as f:
        f.write(f"# AudioEncoder Benchmark Results\n\n")
        f.write(f"**Model:** `{MODEL_ID}`  \n")
        f.write(f"**Date:** {time.strftime('%Y-%m-%d')}  \n")
        f.write(f"**Platform:** `{platform.machine()}`  Python {platform.python_version()}  \n")
        f.write(f"**GPU:** {get_gpu_info()}  \n")
        f.write(f"**Input:** {AUDIO.shape[1]} samples (1s @16kHz), batch=1  \n")
        f.write(f"**Iterations:** {N_ITERS} (after {N_WARMUP} warmup)  \n\n")
        f.write("| Backend | Mean (ms) | Std (ms) | P95 (ms) | vs PyTorch CPU |\n")
        f.write("|---------|-----------|----------|----------|----------------|\n")
        for name, lats in results.items():
            if lats is None:
                f.write(f"| {name} | N/A | — | — | — |\n")
                continue
            mean, std, p95 = lats.mean(), lats.std(), np.percentile(lats, 95)
            speedup = baseline / mean if baseline else 1.0
            f.write(f"| {name} | {mean:.1f} | {std:.1f} | {p95:.1f} | {speedup:.2f}x |\n")

        f.write("\n## Notes\n\n")
        f.write("- MAX Engine CPU includes numpy bridge for pos_conv ")
        f.write("(MAX conv2d groups bug workaround)\n")
        f.write("- MAX Engine GPU uses GPU for CNN and transformer blocks; ")
        f.write("pos_conv runs on CPU\n")


def main():
    os.makedirs(os.path.dirname(os.path.abspath(__file__)), exist_ok=True)

    results = {}
    results["PyTorch CPU"] = benchmark_pytorch_cpu()
    results["PyTorch GPU"] = benchmark_pytorch_gpu()
    results["MAX Engine CPU"] = benchmark_max_cpu()
    results["MAX Engine GPU"] = benchmark_max_gpu()

    output_path = os.path.join(os.path.dirname(__file__), "benchmark_results.md")
    print_and_write_table(results, output_path)
    print(f"\nResults written to {output_path}")


if __name__ == "__main__":
    main()
