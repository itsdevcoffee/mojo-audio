"""Targeted GPU smoke test for PitchExtractor after the _conv2d fix.

Verifies real-weight PitchExtractor.from_pretrained(device="gpu") + extract().

Run on Spark:
    cd /home/visage/repos/mojo-audio
    ~/.pixi/bin/pixi run python scripts/gpu_pitch_probe_2026_04_16.py
"""
from __future__ import annotations

import os
import sys
import time

import numpy as np

sys.path.insert(0, "/home/visage/repos/mojo-audio/src")

RMVPE = "lj1995/VoiceConversionWebUI"

print(f"Python: {sys.version.split()[0]}")
from max.driver import accelerator_count
print(f"MAX accelerator_count(): {accelerator_count()}")

from models.pitch_extractor import PitchExtractor

print("\n=== Loading PitchExtractor on GPU ===")
t0 = time.time()
pe_gpu = PitchExtractor.from_pretrained(RMVPE, device="gpu")
t_compile = time.time() - t0
print(f"Compile + load: {t_compile:.2f}s")

print("\n=== Loading PitchExtractor on CPU (for diff) ===")
t0 = time.time()
pe_cpu = PitchExtractor.from_pretrained(RMVPE, device="cpu")
print(f"Compile + load: {time.time() - t0:.2f}s")

print("\n=== Extracting F0 on GPU and CPU (2s of audio) ===")
np.random.seed(42)
audio_2s = (np.random.randn(1, 32000).astype(np.float32) * 0.05)

t0 = time.time()
f0_gpu = pe_gpu.extract(audio_2s)
t_gpu = time.time() - t0
print(f"GPU extract: {t_gpu:.3f}s  shape={f0_gpu.shape}  RTF={t_gpu/2.0:.3f}")

t0 = time.time()
f0_cpu = pe_cpu.extract(audio_2s)
t_cpu = time.time() - t0
print(f"CPU extract: {t_cpu:.3f}s  shape={f0_cpu.shape}  RTF={t_cpu/2.0:.3f}")

print(f"\nGPU speedup: {t_cpu / t_gpu:.2f}x")

print("\n=== Numerical check ===")
print(f"Output shape match:  {f0_gpu.shape == f0_cpu.shape}")
print(f"Output finite (GPU): {np.isfinite(f0_gpu).all()}")
print(f"Output finite (CPU): {np.isfinite(f0_cpu).all()}")
print(f"Max abs diff:        {np.abs(f0_gpu - f0_cpu).max():.6e}")
print(f"Mean abs diff:       {np.abs(f0_gpu - f0_cpu).mean():.6e}")
voiced_mask = (f0_gpu > 0) & (f0_cpu > 0)
if voiced_mask.any():
    cents_diff = 1200.0 * np.abs(np.log2(f0_gpu[voiced_mask] / f0_cpu[voiced_mask]))
    print(f"Voiced-frame mean cent error: {cents_diff.mean():.4f}")
    print(f"Voiced-frame max  cent error: {cents_diff.max():.4f}")
print(f"Voiced frames (GPU/total):  {(f0_gpu > 0).sum()} / {f0_gpu.size}")
print(f"Voiced frames (CPU/total):  {(f0_cpu > 0).sum()} / {f0_cpu.size}")

print("\n=== DONE — PitchExtractor works on GPU ===")
