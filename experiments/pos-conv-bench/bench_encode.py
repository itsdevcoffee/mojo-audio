"""Benchmark AudioEncoder.encode() — before/after pos_conv migration to MAX graph.

Run: pixi run python experiments/pos-conv-bench/bench_encode.py
"""
import sys, os, time
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "src"))

import numpy as np

def bench_encode(duration_s=1.0, batch_size=1, n_runs=5, warmup=1):
    from models.audio_encoder import AudioEncoder

    # Generate random weights (no download needed)
    rng = np.random.default_rng(42)
    w = _make_full_weights(rng)

    print(f"Building AudioEncoder (batch_size={batch_size}, device=cpu)...")
    t0 = time.time()
    model = AudioEncoder._from_weights(w, device="cpu", batch_size=batch_size)
    build_time = time.time() - t0
    print(f"  Build time: {build_time:.2f}s")

    samples = int(16000 * duration_s)
    audio = rng.standard_normal((batch_size, samples)).astype(np.float32) * 0.1

    # Warmup
    for _ in range(warmup):
        model.encode(audio)

    # Timed runs
    times = []
    for i in range(n_runs):
        t0 = time.time()
        out = model.encode(audio)
        elapsed = time.time() - t0
        times.append(elapsed)
        print(f"  Run {i+1}: {elapsed*1000:.1f}ms  shape={out.shape}")

    avg = np.mean(times) * 1000
    std = np.std(times) * 1000
    print(f"\n  Average: {avg:.1f}ms ± {std:.1f}ms  ({n_runs} runs, {warmup} warmup)")
    print(f"  Audio: {duration_s}s @ 16kHz, batch={batch_size}")
    return avg


def _make_full_weights(rng):
    w = {}
    configs = [
        (1,512,10),(512,512,3),(512,512,3),(512,512,3),
        (512,512,3),(512,512,2),(512,512,2)
    ]
    for i, (c_in, c_out, k) in enumerate(configs):
        w[f"cnn.{i}.weight"] = rng.standard_normal((c_out, c_in, k)).astype(np.float32) * 0.02
        w[f"cnn.{i}.norm.weight"] = np.ones(c_out, dtype=np.float32)
        w[f"cnn.{i}.norm.bias"] = np.zeros(c_out, dtype=np.float32)
    w["proj.weight"] = rng.standard_normal((768, 512)).astype(np.float32) * 0.02
    w["proj.bias"] = np.zeros(768, dtype=np.float32)
    w["proj.norm.weight"] = np.ones(512, dtype=np.float32)
    w["proj.norm.bias"] = np.zeros(512, dtype=np.float32)
    w["pos_conv.weight"] = rng.standard_normal((768, 48, 128)).astype(np.float32) * 0.02
    w["pos_conv.bias"] = np.zeros(768, dtype=np.float32)
    w["enc_norm.weight"] = np.ones(768, dtype=np.float32)
    w["enc_norm.bias"] = np.zeros(768, dtype=np.float32)
    for i in range(12):
        for name in ["norm1", "norm2"]:
            w[f"blocks.{i}.{name}.weight"] = np.ones(768, dtype=np.float32)
            w[f"blocks.{i}.{name}.bias"] = np.zeros(768, dtype=np.float32)
        for proj in ["attn.q", "attn.k", "attn.v", "attn.out"]:
            w[f"blocks.{i}.{proj}.weight"] = rng.standard_normal((768, 768)).astype(np.float32) * 0.02
            w[f"blocks.{i}.{proj}.bias"] = np.zeros(768, dtype=np.float32)
        w[f"blocks.{i}.ffn.fc1.weight"] = rng.standard_normal((3072, 768)).astype(np.float32) * 0.02
        w[f"blocks.{i}.ffn.fc1.bias"] = np.zeros(3072, dtype=np.float32)
        w[f"blocks.{i}.ffn.fc2.weight"] = rng.standard_normal((768, 3072)).astype(np.float32) * 0.02
        w[f"blocks.{i}.ffn.fc2.bias"] = np.zeros(768, dtype=np.float32)
    return w


if __name__ == "__main__":
    print("=" * 60)
    print("AudioEncoder.encode() Benchmark")
    print("=" * 60)

    print("\n--- Batch=1, 1s audio ---")
    bench_encode(duration_s=1.0, batch_size=1)

    print("\n--- Batch=1, 3s audio ---")
    bench_encode(duration_s=3.0, batch_size=1)

    print("\n--- Batch=2, 1s audio ---")
    bench_encode(duration_s=1.0, batch_size=2)
