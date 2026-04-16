"""GPU recon — 2026-04-16

Force device="gpu" on each pipeline component in isolation on Spark.
Capture: did it compile? did it run? what shape came out? how long?

Run on Spark:
    cd /home/visage/repos/mojo-audio
    ~/.pixi/bin/pixi run python scripts/gpu_recon_2026_04_16.py 2>&1 | tee /tmp/gpu_recon.log
"""
from __future__ import annotations

import json
import os
import sys
import time
import traceback

import numpy as np

REPO = "/home/visage/repos/mojo-audio"
sys.path.insert(0, os.path.join(REPO, "src"))

CKPT = "/home/visage/Downloads/voice files/extracted/theweeknd biggest data set/theweekv1.pth"
HUBERT = "facebook/hubert-base-ls960"
RMVPE = "lj1995/VoiceConversionWebUI"

OUT = "/tmp/gpu_recon_results.json"

results: dict = {}


def banner(name: str) -> None:
    print("\n" + "=" * 78)
    print(f"=== {name}")
    print("=" * 78)


def run(name: str, fn) -> None:
    banner(name)
    res: dict = {
        "compile_ok": False,
        "run_ok": False,
        "compile_time_s": None,
        "run_time_s": None,
        "output_shape": None,
        "error_type": None,
        "error_msg": None,
        "traceback_tail": None,
    }
    try:
        fn(res)
    except Exception as e:
        res["error_type"] = type(e).__name__
        res["error_msg"] = str(e)
        tb = traceback.format_exc()
        # Keep last 30 lines of traceback for the report
        res["traceback_tail"] = "\n".join(tb.splitlines()[-30:])
        traceback.print_exc()
    results[name] = res
    print(f"\n[{name}] result: compile={res['compile_ok']} run={res['run_ok']} "
          f"shape={res['output_shape']} err={res['error_type']}")


# ---------------------------------------------------------------------------
# Reusable test inputs (deterministic)
# ---------------------------------------------------------------------------
np.random.seed(42)
AUDIO_2S = (np.random.randn(1, 32000).astype(np.float32) * 0.05)  # 2s @ 16 kHz


# ---------------------------------------------------------------------------
# 1. AudioEncoder (HuBERT) — known-working on GPU per Sprint 2
# ---------------------------------------------------------------------------
def probe_audio_encoder(res: dict) -> None:
    from models.audio_encoder import AudioEncoder

    t0 = time.time()
    enc = AudioEncoder.from_pretrained(HUBERT, device="gpu")
    res["compile_ok"] = True
    res["compile_time_s"] = round(time.time() - t0, 2)

    t0 = time.time()
    feats = enc.encode(AUDIO_2S)
    res["run_time_s"] = round(time.time() - t0, 3)
    res["run_ok"] = True
    res["output_shape"] = list(feats.shape)
    res["output_dtype"] = str(feats.dtype)
    res["output_finite"] = bool(np.isfinite(feats).all())
    res["output_stats"] = {
        "min": float(feats.min()),
        "max": float(feats.max()),
        "mean": float(feats.mean()),
        "std": float(feats.std()),
    }


# ---------------------------------------------------------------------------
# 2. PitchExtractor (RMVPE) — KNOWN-BROKEN on GPU (audit §11 item 1)
# ---------------------------------------------------------------------------
def probe_pitch_extractor(res: dict) -> None:
    from models.pitch_extractor import PitchExtractor

    t0 = time.time()
    pe = PitchExtractor.from_pretrained(RMVPE, device="gpu")
    res["compile_ok"] = True
    res["compile_time_s"] = round(time.time() - t0, 2)

    t0 = time.time()
    f0 = pe.extract(AUDIO_2S)
    res["run_time_s"] = round(time.time() - t0, 3)
    res["run_ok"] = True
    res["output_shape"] = list(f0.shape)
    res["output_dtype"] = str(f0.dtype)
    res["output_finite"] = bool(np.isfinite(f0).all())
    voiced = f0[f0 > 0]
    res["output_stats"] = {
        "voiced_frames": int((f0 > 0).sum()),
        "total_frames": int(f0.size),
        "voiced_min_hz": float(voiced.min()) if voiced.size else None,
        "voiced_max_hz": float(voiced.max()) if voiced.size else None,
    }


# ---------------------------------------------------------------------------
# 3. HiFiGAN (NSF vocoder) — UNTESTED on GPU
# ---------------------------------------------------------------------------
def probe_hifigan(res: dict) -> None:
    from models.hifigan import NSFHiFiGAN

    t0 = time.time()
    voc = NSFHiFiGAN.from_pretrained(CKPT, device="gpu")
    res["compile_ok"] = True
    res["compile_time_s"] = round(time.time() - t0, 2)

    T = 200  # ~2s of latent frames at HiFiGAN's input rate
    latents = (np.random.randn(1, 192, T).astype(np.float32) * 0.1)
    f0 = (200.0 * np.ones((1, T), dtype=np.float32))

    t0 = time.time()
    audio = voc.synthesize(latents, f0)
    res["run_time_s"] = round(time.time() - t0, 3)
    res["run_ok"] = True
    res["output_shape"] = list(audio.shape)
    res["output_dtype"] = str(audio.dtype)
    res["output_finite"] = bool(np.isfinite(audio).all())


# ---------------------------------------------------------------------------
# 4. VITS (enc_p + flow) + HiFiGAN via _from_vits_only
# ---------------------------------------------------------------------------
def probe_vits_only(res: dict) -> None:
    from models.voice_converter import VoiceConverter

    t0 = time.time()
    vc = VoiceConverter._from_vits_only(CKPT, device="gpu")
    res["compile_ok"] = True
    res["compile_time_s"] = round(time.time() - t0, 2)

    T = 200
    features = (np.random.randn(1, T, 768).astype(np.float32) * 0.1)
    f0 = (200.0 * np.ones((1, T), dtype=np.float32))
    pitch = (np.ones((1, T), dtype=np.int32) * 100)
    lengths = np.array([T], dtype=np.int32)

    t0 = time.time()
    audio = vc.convert_from_features(features, f0, pitch, lengths)
    res["run_time_s"] = round(time.time() - t0, 3)
    res["run_ok"] = True
    res["output_shape"] = list(audio.shape)
    res["output_dtype"] = str(audio.dtype)
    res["output_finite"] = bool(np.isfinite(audio).all())


# ---------------------------------------------------------------------------
# 5. Full VoiceConverter (all 5 graphs on GPU, real audio in)
# ---------------------------------------------------------------------------
def probe_full_pipeline(res: dict) -> None:
    from models.voice_converter import VoiceConverter

    t0 = time.time()
    vc = VoiceConverter.from_pretrained(
        CKPT, hubert_path=HUBERT, rmvpe_path=RMVPE, device="gpu"
    )
    res["compile_ok"] = True
    res["compile_time_s"] = round(time.time() - t0, 2)

    t0 = time.time()
    audio = vc.convert(AUDIO_2S)
    res["run_time_s"] = round(time.time() - t0, 3)
    res["run_ok"] = True
    res["output_shape"] = list(audio.shape)
    res["output_dtype"] = str(audio.dtype)
    res["output_finite"] = bool(np.isfinite(audio).all())
    res["rtf"] = round(res["run_time_s"] / 2.0, 3)


# ---------------------------------------------------------------------------
# Drive
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    print(f"Python: {sys.version.split()[0]}")
    print(f"NumPy:  {np.__version__}")
    try:
        from max.driver import accelerator_count
        print(f"MAX accelerator_count(): {accelerator_count()}")
    except Exception as e:
        print(f"MAX accelerator probe failed: {e}")

    run("AudioEncoder", probe_audio_encoder)
    run("PitchExtractor", probe_pitch_extractor)
    run("HiFiGAN", probe_hifigan)
    run("VITS_only", probe_vits_only)
    run("FullVoiceConverter", probe_full_pipeline)

    banner("SUMMARY")
    for name, r in results.items():
        flag = "OK " if (r["compile_ok"] and r["run_ok"]) else (
            "C-X" if not r["compile_ok"] else "R-X"
        )
        print(f"  {flag}  {name:24s}  shape={r['output_shape']}  "
              f"compile={r['compile_time_s']}s  run={r['run_time_s']}s  "
              f"err={r['error_type']}")

    with open(OUT, "w") as f:
        json.dump(results, f, indent=2, default=str)
    print(f"\nWrote {OUT}")
