#!/usr/bin/env python3
"""
Benchmark HuBERT inference across available backends.

MAX Engine excluded — v26.1 has no ONNX importer (see 03_max_inference.py).

Backends tested:
  - PyTorch CPU
  - PyTorch GPU (RTX 4060 Ti)
  - ONNXRuntime CPU

Outputs a comparison table and saves raw latency data to benchmark_raw.npz.
"""

import time
import numpy as np
import torch
import onnxruntime as ort
from pathlib import Path
from transformers import HubertModel

ONNX_PATH = Path(__file__).parent / "hubert_base.onnx"
MODEL_ID = "facebook/hubert-base-ls960"
N_WARMUP = 3
N_ITERS = 20
INPUT_LEN = 16000  # 1 second at 16kHz

rng = np.random.default_rng(42)
INPUT_NP = rng.standard_normal((1, INPUT_LEN)).astype(np.float32)
INPUT_PT = torch.from_numpy(INPUT_NP)


def benchmark_fn(fn, n_warmup=N_WARMUP, n_iters=N_ITERS):
    """Run fn() n_warmup times (ignored), then n_iters times, return latencies in ms."""
    for _ in range(n_warmup):
        fn()

    latencies = []
    for _ in range(n_iters):
        t0 = time.perf_counter()
        fn()
        t1 = time.perf_counter()
        latencies.append((t1 - t0) * 1000)

    return np.array(latencies)


def benchmark_pytorch_cpu():
    print("Loading model for PyTorch CPU ...")
    model = HubertModel.from_pretrained(MODEL_ID).eval()

    def fn():
        with torch.no_grad():
            model(INPUT_PT)

    print("Warming up ...")
    latencies = benchmark_fn(fn)
    print(f"  mean: {latencies.mean():.1f} ms")
    return latencies


def benchmark_pytorch_gpu():
    if not torch.cuda.is_available():
        print("  CUDA not available, skipping")
        return None
    print("Loading model for PyTorch GPU ...")
    model = HubertModel.from_pretrained(MODEL_ID).eval().cuda()
    input_gpu = INPUT_PT.cuda()

    def fn():
        with torch.no_grad():
            model(input_gpu)
        torch.cuda.synchronize()

    print("Warming up ...")
    latencies = benchmark_fn(fn)
    print(f"  mean: {latencies.mean():.1f} ms")
    return latencies


def benchmark_ort_cpu():
    assert ONNX_PATH.exists(), f"Run 01_export_onnx.py first"
    print("Loading ONNX for ONNXRuntime CPU ...")
    session = ort.InferenceSession(str(ONNX_PATH), providers=["CPUExecutionProvider"])

    def fn():
        session.run(["last_hidden_state"], {"input_values": INPUT_NP})

    print("Warming up ...")
    latencies = benchmark_fn(fn)
    print(f"  mean: {latencies.mean():.1f} ms")
    return latencies


def print_table(results):
    baseline_mean = None
    for name, latencies in results.items():
        if latencies is not None:
            baseline_mean = latencies.mean()
            break

    print("\n" + "=" * 70)
    print("HuBERT Inference Benchmark")
    print("=" * 70)
    print(f"Model:      {MODEL_ID}")
    print(f"Input:      {INPUT_LEN} samples (1s @16kHz) — batch size 1")
    print(f"Iterations: {N_ITERS} (after {N_WARMUP} warmup)")
    if torch.cuda.is_available():
        print(f"GPU:        {torch.cuda.get_device_name(0)}")
    print()
    print(f"{'Backend':<22} {'Mean (ms)':>10} {'Std (ms)':>9} {'P95 (ms)':>9} {'vs CPU':>10}")
    print("-" * 65)

    for name, latencies in results.items():
        if latencies is None:
            print(f"{name:<22} {'N/A':>10}")
            continue
        mean = latencies.mean()
        std = latencies.std()
        p95 = np.percentile(latencies, 95)
        speedup = baseline_mean / mean if baseline_mean else 1.0
        print(f"{name:<22} {mean:>10.1f} {std:>9.1f} {p95:>9.1f} {speedup:>9.2f}x")

    print()
    print("Note: MAX Engine v26.1 excluded — no ONNX importer available.")
    print("      MAX would require HuBERT rewritten in MAX Graph API.")
    print("=" * 70)


def main():
    results = {}

    print("\n--- PyTorch CPU ---")
    results["PyTorch CPU"] = benchmark_pytorch_cpu()

    print("\n--- PyTorch GPU ---")
    results["PyTorch GPU (RTX 4060Ti)"] = benchmark_pytorch_gpu()

    print("\n--- ONNXRuntime CPU ---")
    results["ONNXRuntime CPU"] = benchmark_ort_cpu()

    print_table(results)

    # Save raw data
    output_file = Path(__file__).parent / "benchmark_raw.npz"
    save_dict = {k.replace(" ", "_").replace("(", "").replace(")", ""): v
                 for k, v in results.items() if v is not None}
    np.savez(str(output_file), **save_dict)
    print(f"\nRaw latency data saved to {output_file}")


if __name__ == "__main__":
    main()
