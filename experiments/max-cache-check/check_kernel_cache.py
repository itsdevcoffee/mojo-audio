"""Check whether MAX Engine caches compiled graphs across Python sessions.

Run this script TWICE in separate processes:
  pixi run python experiments/max-cache-check/check_kernel_cache.py

First run: expect ~slow (graph JIT compilation).
Second run: if <5s, cache works. If still slow, cache is broken.
"""
import sys, os, time
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "src"))

import numpy as np

def main():
    print("=" * 60)
    print("MAX Kernel Cache Check")
    print("=" * 60)

    # Check for known cache locations
    cache_dirs = [
        os.path.expanduser("~/.cache/modular/"),
        os.path.expanduser("~/.local/share/modular/"),
        os.path.expanduser("~/.cache/max/"),
    ]
    env_cache = os.environ.get("MODULAR_CACHE_DIR")
    if env_cache:
        cache_dirs.insert(0, env_cache)
        print(f"  MODULAR_CACHE_DIR = {env_cache}")

    for d in cache_dirs:
        exists = os.path.isdir(d)
        print(f"  {d} {'EXISTS' if exists else '(not found)'}")

    # Time _from_weights with random weights (tests graph compilation caching)
    print("\n--- Test 1: _from_weights (random weights, CPU) ---")
    from models.audio_encoder import AudioEncoder

    rng = np.random.default_rng(42)
    w = _make_weights(rng)

    t0 = time.time()
    model = AudioEncoder._from_weights(w, device="cpu")
    t1 = time.time()
    print(f"  _from_weights time: {t1-t0:.2f}s")

    # Quick encode to verify it works
    audio = np.zeros((1, 16000), dtype=np.float32)
    out = model.encode(audio)
    print(f"  encode shape: {out.shape}")

    # Time from_pretrained (tests full pipeline: weight load + graph compile)
    print("\n--- Test 2: from_pretrained (real HuBERT weights, CPU) ---")
    t0 = time.time()
    model2 = AudioEncoder.from_pretrained("facebook/hubert-base-ls960", device="cpu")
    t1 = time.time()
    print(f"  from_pretrained time: {t1-t0:.2f}s")

    out2 = model2.encode(np.random.randn(1, 16000).astype(np.float32))
    print(f"  encode shape: {out2.shape}")

    # Check cache dirs again after compilation
    print("\n--- Cache directories after compilation ---")
    for d in cache_dirs:
        if os.path.isdir(d):
            total = sum(
                os.path.getsize(os.path.join(dp, f))
                for dp, _, fnames in os.walk(d)
                for f in fnames
            )
            print(f"  {d} — {total / 1024 / 1024:.1f} MB")

    print(f"\n{'=' * 60}")
    print("Run this script again in a NEW Python process.")
    print("If second run is <5s, MAX kernel cache is working.")
    print(f"{'=' * 60}")


def _make_weights(rng):
    w = {}
    configs = [(1,512,10),(512,512,3),(512,512,3),(512,512,3),
               (512,512,3),(512,512,2),(512,512,2)]
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
    main()
