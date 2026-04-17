"""Head-to-head Applio vs mojo-audio comparison on identical input.

Runs BOTH engines on the same audio with matched parameters, then reports:
  - Per-stage wall-clock timing + total RTF for each engine
  - Waveform correlation, RMS level difference, max abs difference
  - Mel-spectrogram correlation + max/mean dB delta
  - F0 trajectory: voicing agreement, mean/median cent error, % within tolerances

Both output WAVs are written to /tmp/ so you can A/B them by ear.

Usage (on Spark):

    cd /home/visage/repos/mojo-audio
    ~/.pixi/bin/pixi run python scripts/compare_vs_applio.py \\
        --model "/home/visage/Downloads/voice files/extracted/theweeknd biggest data set/theweekv1.pth" \\
        --audio /path/to/real_vocal.wav

Matching parameters (per handoff doc table): pitch=0, f0_method=rmvpe,
index_rate=0 (disabled — mojo-audio has no FAISS), protect=0.5 (no-op with no
FAISS), volume_envelope=1.0, embedder=contentvec, noise_scale=0.66666.
"""

from __future__ import annotations

import argparse
import os
import sys
import time

import numpy as np
import soundfile as sf
import torch


APPLIO_CANDIDATE_PATHS = [
    "/home/visage/repos/Applio",
    "/home/maskkiller/repos/Applio",
]

MOJO_CANDIDATE_PATHS = [
    "/home/visage/repos/mojo-audio",
    "/home/maskkiller/dev-coffee/repos/mojo-audio",
]


def find_dir(candidates: list[str], marker: str) -> str:
    for p in candidates:
        if os.path.isdir(p) and os.path.isfile(os.path.join(p, marker)):
            return p
    raise FileNotFoundError(f"None of {candidates} contain {marker}")


# ---------------------------------------------------------------------------
# Applio runner
# ---------------------------------------------------------------------------


def run_applio(model_path: str, audio_16k: np.ndarray, device: str | None):
    """Run Applio once and return (audio_out, tgt_sr, timings_dict)."""
    from rvc.lib.utils import load_embedding
    from rvc.lib.algorithm.synthesizers import Synthesizer
    from rvc.configs.config import Config
    from rvc.infer.pipeline import Pipeline

    config = Config()
    if device is not None:
        config.device = device

    t0 = time.time()
    hubert = load_embedding("contentvec").to(config.device).float().eval()
    cpt = torch.load(model_path, map_location="cpu", weights_only=True)
    tgt_sr = cpt["config"][-1]
    cpt["config"][-3] = cpt["weight"]["emb_g.weight"].shape[0]
    use_f0 = cpt.get("f0", 1)
    version = cpt.get("version", "v1")
    vocoder = cpt.get("vocoder", "HiFi-GAN")
    text_enc_hidden_dim = 768 if version == "v2" else 256

    net_g = Synthesizer(
        *cpt["config"],
        use_f0=use_f0,
        text_enc_hidden_dim=text_enc_hidden_dim,
        vocoder=vocoder,
    )
    if hasattr(net_g, "enc_q"):
        del net_g.enc_q
    net_g.load_state_dict(cpt["weight"], strict=False)
    net_g = net_g.to(config.device).float().eval()
    load_s = time.time() - t0

    # Stage timers — monkey-patch instance methods / class method.
    times = {"hubert_s": 0.0, "rmvpe_s": 0.0, "vits_s": 0.0}

    orig_hubert_forward = hubert.forward

    def timed_hubert_forward(*a, **kw):
        t = time.time()
        r = orig_hubert_forward(*a, **kw)
        times["hubert_s"] += time.time() - t
        return r

    hubert.forward = timed_hubert_forward

    from rvc.lib.predictors.f0 import RMVPE as ApplioRMVPE

    orig_rmvpe_get_f0 = ApplioRMVPE.get_f0

    def timed_rmvpe_get_f0(self, *a, **kw):
        t = time.time()
        r = orig_rmvpe_get_f0(self, *a, **kw)
        times["rmvpe_s"] += time.time() - t
        return r

    ApplioRMVPE.get_f0 = timed_rmvpe_get_f0

    orig_infer = net_g.infer

    def timed_infer(*a, **kw):
        t = time.time()
        r = orig_infer(*a, **kw)
        times["vits_s"] += time.time() - t
        return r

    net_g.infer = timed_infer

    vc = Pipeline(tgt_sr, config)

    # Warm-up run (discarded — accounts for CUDA kernel compilation etc.)
    _ = vc.pipeline(
        model=hubert, net_g=net_g, sid=0, audio=audio_16k,
        pitch=0, f0_method="rmvpe", file_index="", index_rate=0.0,
        pitch_guidance=use_f0, volume_envelope=1.0, version=version,
        protect=0.5, f0_autotune=False, f0_autotune_strength=1.0,
        proposed_pitch=False, proposed_pitch_threshold=155.0,
    )

    # Timed run
    for k in times:
        times[k] = 0.0
    t0 = time.time()
    audio_out = vc.pipeline(
        model=hubert, net_g=net_g, sid=0, audio=audio_16k,
        pitch=0, f0_method="rmvpe", file_index="", index_rate=0.0,
        pitch_guidance=use_f0, volume_envelope=1.0, version=version,
        protect=0.5, f0_autotune=False, f0_autotune_strength=1.0,
        proposed_pitch=False, proposed_pitch_threshold=155.0,
    )
    total_s = time.time() - t0
    times["total_s"] = total_s
    times["load_s"] = load_s
    times["overhead_s"] = max(0.0, total_s - times["hubert_s"] - times["rmvpe_s"] - times["vits_s"])
    times["device"] = str(config.device)

    return audio_out.astype(np.float32), tgt_sr, times


# ---------------------------------------------------------------------------
# mojo-audio runner
# ---------------------------------------------------------------------------


def run_mojo(model_path: str, audio_16k: np.ndarray, device: str):
    """Run mojo-audio once with per-stage timing, return (audio_out, tgt_sr, f0, timings)."""
    import math
    from models.voice_converter import (
        VoiceConverter,
        interpolate_features_2x,
        quantize_f0,
        sequence_mask,
    )
    from models._vits_graph import compute_rel_attention_biases

    times = {"hubert_s": 0.0, "rmvpe_s": 0.0, "vits_s": 0.0}

    t0 = time.time()
    vc = VoiceConverter.from_pretrained(model_path, device=device)
    load_s = time.time() - t0

    tgt_sr = vc._config["sr"]

    # --- Warm-up run (discard) ---
    _ = vc.convert(audio_16k, pitch_shift=0, sr=16000)

    # --- Timed run, broken into stages ---
    audio_batched = audio_16k[np.newaxis, :] if audio_16k.ndim == 1 else audio_16k
    B = audio_batched.shape[0]

    t0 = time.time()

    t = time.time()
    features = vc._audio_encoder.encode(audio_batched)
    times["hubert_s"] = time.time() - t

    t = time.time()
    f0_list = [vc._pitch_extractor.extract(audio_batched[b:b+1]) for b in range(B)]
    times["rmvpe_s"] = time.time() - t

    # Feature/F0 alignment (orchestration — counts as overhead)
    features_up = interpolate_features_2x(features)
    T_feat = features_up.shape[1]
    f0_batch = np.zeros((B, T_feat), dtype=np.float32)
    pitch_batch = np.zeros((B, T_feat), dtype=np.int32)
    for b in range(B):
        f0 = f0_list[b]
        T_f = min(len(f0), T_feat)
        f0_aligned = np.zeros(T_feat, dtype=np.float32)
        f0_aligned[:T_f] = f0[:T_f]
        f0_batch[b] = f0_aligned
        pitch_batch[b] = quantize_f0(f0_aligned.copy())
    lengths = np.full(B, T_feat, dtype=np.int32)

    t = time.time()
    audio_out = vc.convert_from_features(features_up, f0_batch, pitch_batch, lengths)
    times["vits_s"] = time.time() - t

    total_s = time.time() - t0
    times["total_s"] = total_s
    times["load_s"] = load_s
    times["overhead_s"] = max(0.0, total_s - times["hubert_s"] - times["rmvpe_s"] - times["vits_s"])
    times["device"] = str(vc._device)

    audio_np = audio_out[0] if audio_out.ndim == 2 else audio_out
    return audio_np.astype(np.float32), tgt_sr, f0_list[0], times


# ---------------------------------------------------------------------------
# Quality metrics
# ---------------------------------------------------------------------------


def compare_waveforms(ref: np.ndarray, test: np.ndarray) -> dict:
    L = min(len(ref), len(test))
    ref, test = ref[:L], test[:L]
    # Correlation handles near-constant signals gracefully
    if ref.std() < 1e-8 or test.std() < 1e-8:
        corr = float("nan")
    else:
        corr = float(np.corrcoef(ref, test)[0, 1])
    rms_ref = float(np.sqrt(np.mean(ref ** 2)))
    rms_test = float(np.sqrt(np.mean(test ** 2)))
    rms_diff_db = 20 * np.log10((rms_test + 1e-10) / (rms_ref + 1e-10))
    return {
        "length_ref": len(ref),
        "length_test": len(test),
        "corr": corr,
        "rms_ref": rms_ref,
        "rms_test": rms_test,
        "rms_diff_db": rms_diff_db,
        "max_abs_diff": float(np.abs(ref - test).max()),
    }


def compare_spectrograms(ref: np.ndarray, test: np.ndarray, sr: int) -> dict:
    import librosa

    L = min(len(ref), len(test))
    ref, test = ref[:L], test[:L]
    S_ref = librosa.feature.melspectrogram(y=ref, sr=sr, n_mels=128)
    S_test = librosa.feature.melspectrogram(y=test, sr=sr, n_mels=128)
    T = min(S_ref.shape[1], S_test.shape[1])
    S_ref, S_test = S_ref[:, :T], S_test[:, :T]
    log_ref = librosa.power_to_db(S_ref)
    log_test = librosa.power_to_db(S_test)
    return {
        "spec_corr": float(np.corrcoef(log_ref.flatten(), log_test.flatten())[0, 1]),
        "spec_max_diff_db": float(np.abs(log_ref - log_test).max()),
        "spec_mean_diff_db": float(np.abs(log_ref - log_test).mean()),
    }


def compare_f0(f0_ref: np.ndarray, f0_test: np.ndarray) -> dict:
    L = min(len(f0_ref), len(f0_test))
    f0_ref, f0_test = f0_ref[:L], f0_test[:L]
    ref_voiced = f0_ref > 0
    test_voiced = f0_test > 0
    both_voiced = ref_voiced & test_voiced
    out = {
        "frames": int(L),
        "ref_voiced_pct": float(ref_voiced.mean() * 100),
        "test_voiced_pct": float(test_voiced.mean() * 100),
        "voicing_agreement_pct": float((ref_voiced == test_voiced).mean() * 100),
    }
    if both_voiced.any():
        cents = 1200 * np.abs(np.log2(f0_test[both_voiced] / f0_ref[both_voiced]))
        out.update({
            "mean_cent_error": float(cents.mean()),
            "median_cent_error": float(np.median(cents)),
            "pct_within_5_cents": float((cents < 5).mean() * 100),
            "pct_within_50_cents": float((cents < 50).mean() * 100),
        })
    return out


def extract_applio_f0(audio_16k: np.ndarray, device: str | None) -> np.ndarray:
    """Independently run Applio's RMVPE on the input — used so Applio and
    mojo-audio F0 trajectories are compared on the same pitch model."""
    from rvc.lib.predictors.f0 import RMVPE as ApplioRMVPE
    dev = device or ("cuda:0" if torch.cuda.is_available() else "cpu")
    model = ApplioRMVPE(device=dev, sample_rate=16000, hop_size=160)
    return model.get_f0(audio_16k, filter_radius=0.03)


# ---------------------------------------------------------------------------
# Reporting
# ---------------------------------------------------------------------------


def print_speed_table(duration_s: float, applio: dict, mojo: dict) -> None:
    def ratio(a_val: float, m_val: float) -> str:
        # speedup of mojo relative to applio: applio_time / mojo_time
        if m_val <= 0:
            return "—"
        return f"{a_val / m_val:.2f}x"

    print(f"{'':<20}{'Applio':>14}{'mojo-audio':>16}{'Speedup':>12}")
    for key, label in [
        ("load_s", "Cold load"),
        ("hubert_s", "HuBERT/ContentVec"),
        ("rmvpe_s", "RMVPE (F0)"),
        ("vits_s", "VITS synth"),
        ("overhead_s", "Overhead"),
        ("total_s", "Total inference"),
    ]:
        print(f"  {label:<18}{applio[key]:>10.3f}s  {mojo[key]:>12.3f}s  {ratio(applio[key], mojo[key]):>10}")
    applio_rtf = applio["total_s"] / duration_s
    mojo_rtf = mojo["total_s"] / duration_s
    print(f"  {'RTF (warm)':<18}{applio_rtf:>11.3f}x  {mojo_rtf:>13.3f}x")
    print(f"  {'Device':<18}{applio['device']:>14}{mojo['device']:>16}")


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--model", required=True, help="RVC v2 .pth voice model (used by both)")
    p.add_argument("--audio", required=True, help="Input WAV (will be resampled to 16 kHz mono)")
    p.add_argument("--applio-device", default=None, help="Applio torch device (default: auto)")
    p.add_argument("--mojo-device", default="auto", help="mojo-audio device: auto, cpu, gpu")
    p.add_argument("--applio-out", default="/tmp/compare_applio_out.wav")
    p.add_argument("--mojo-out", default="/tmp/compare_mojo_out.wav")
    args = p.parse_args()

    applio_path = find_dir(APPLIO_CANDIDATE_PATHS, "rvc/infer/pipeline.py")
    mojo_path = find_dir(MOJO_CANDIDATE_PATHS, "src/models/voice_converter.py")

    # Applio assumes CWD contains rvc/models/predictors/... — chdir before import.
    sys.path.insert(0, applio_path)
    sys.path.insert(0, os.path.join(mojo_path, "src"))
    os.chdir(applio_path)

    from rvc.lib.utils import load_audio_infer

    print("=== mojo-audio vs Applio Comparison ===")
    print(f"Applio:     {applio_path}")
    print(f"mojo-audio: {mojo_path}")
    print(f"Model:      {args.model}")
    print(f"Input:      {args.audio}")

    audio_16k = load_audio_infer(args.audio, 16000).astype(np.float32)
    duration_s = len(audio_16k) / 16000
    print(f"Duration:   {duration_s:.2f}s ({len(audio_16k)} samples @ 16 kHz)")

    print("\n--- Running Applio ---")
    applio_out, applio_sr, applio_times = run_applio(args.model, audio_16k, args.applio_device)
    print(f"Applio output: {len(applio_out)} samples @ {applio_sr} Hz ({len(applio_out)/applio_sr:.2f}s)")

    print("\n--- Running mojo-audio ---")
    mojo_out, mojo_sr, mojo_f0, mojo_times = run_mojo(args.model, audio_16k, args.mojo_device)
    print(f"mojo-audio output: {len(mojo_out)} samples @ {mojo_sr} Hz ({len(mojo_out)/mojo_sr:.2f}s)")

    # Save both outputs
    sf.write(args.applio_out, applio_out, applio_sr)
    sf.write(args.mojo_out, mojo_out, mojo_sr)

    # --- Speed ---
    print("\n--- Speed ---")
    if applio_sr != mojo_sr:
        print(f"(note: SR differs — Applio {applio_sr} / mojo {mojo_sr} — quality metrics resample to {min(applio_sr, mojo_sr)})")
    print_speed_table(duration_s, applio_times, mojo_times)

    # --- Quality: resample both to the lower SR for fair comparison ---
    cmp_sr = min(applio_sr, mojo_sr)
    if applio_sr != cmp_sr or mojo_sr != cmp_sr:
        from scipy.signal import resample_poly
        from math import gcd
        def rs(x: np.ndarray, src: int, dst: int) -> np.ndarray:
            if src == dst:
                return x
            g = gcd(src, dst)
            return resample_poly(x, dst // g, src // g).astype(np.float32)
        applio_cmp = rs(applio_out, applio_sr, cmp_sr)
        mojo_cmp = rs(mojo_out, mojo_sr, cmp_sr)
    else:
        applio_cmp = applio_out
        mojo_cmp = mojo_out

    print("\n--- Quality ---")
    wf = compare_waveforms(applio_cmp, mojo_cmp)
    print(f"Length:                  applio={wf['length_ref']}  mojo={wf['length_test']}  diff={wf['length_ref']-wf['length_test']} samples")
    print(f"Waveform correlation:    {wf['corr']:.4f}")
    print(f"RMS level (applio/mojo): {wf['rms_ref']:.4f} / {wf['rms_test']:.4f}  (diff {wf['rms_diff_db']:+.2f} dB)")
    print(f"Max abs sample diff:     {wf['max_abs_diff']:.4f}")

    sp = compare_spectrograms(applio_cmp, mojo_cmp, cmp_sr)
    print(f"Spectrogram corr:        {sp['spec_corr']:.4f}")
    print(f"Spectrogram diff (dB):   max={sp['spec_max_diff_db']:.2f}  mean={sp['spec_mean_diff_db']:.2f}")

    # F0 comparison: rerun Applio's RMVPE on the raw input, compare to mojo-audio's F0.
    # This checks whether mojo-audio's RMVPE port matches Applio's RMVPE on the
    # same input — which is the relevant pitch-quality question.
    print("\n--- F0 (RMVPE on raw input, 16 kHz) ---")
    try:
        applio_f0 = extract_applio_f0(audio_16k, args.applio_device)
        f0m = compare_f0(applio_f0, mojo_f0)
        print(f"Frames (applio/mojo):    {len(applio_f0)} / {len(mojo_f0)}  (compared {f0m['frames']})")
        print(f"Voicing (applio/mojo):   {f0m['ref_voiced_pct']:.1f}% / {f0m['test_voiced_pct']:.1f}%")
        print(f"Voicing agreement:       {f0m['voicing_agreement_pct']:.1f}%")
        if "mean_cent_error" in f0m:
            print(f"Cent error (mean/median): {f0m['mean_cent_error']:.2f} / {f0m['median_cent_error']:.2f}")
            print(f"Within 5 cents:          {f0m['pct_within_5_cents']:.1f}%")
            print(f"Within 50 cents:         {f0m['pct_within_50_cents']:.1f}%")
        else:
            print("No frames voiced in both — skipping cent-error stats.")
    except Exception as e:
        print(f"(F0 comparison skipped: {type(e).__name__}: {e})")

    print("\n--- Files ---")
    print(f"Applio output:    {args.applio_out}")
    print(f"mojo-audio output:{args.mojo_out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
