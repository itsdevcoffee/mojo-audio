"""Applio-only RTF baseline with per-stage timing.

Measures Applio's voice-conversion pipeline end-to-end and per stage
(HuBERT, RMVPE, VITS synth = enc_p + flow + vocoder). This is the denominator
for every future speedup claim.

Run on Spark (where Applio + voice models live):

    cd /home/visage/repos/mojo-audio
    ~/.pixi/bin/pixi run python scripts/benchmark_applio_baseline.py \\
        --model "/home/visage/Downloads/voice files/extracted/theweeknd biggest data set/theweekv1.pth" \\
        --audio /path/to/vocal.wav

Local also works if you have Applio + a voice model locally.

Stages are timed by monkey-patching Applio's internal callables — we do NOT
modify Applio's source. `time.time()` is used throughout (not `timeit`) per
the handoff requirements.
"""

from __future__ import annotations

import argparse
import os
import statistics
import sys
import time

import numpy as np
import soundfile as sf
import torch


APPLIO_CANDIDATE_PATHS = [
    "/home/visage/repos/Applio",
    "/home/maskkiller/repos/Applio",
]


def find_applio() -> str:
    for p in APPLIO_CANDIDATE_PATHS:
        if os.path.isdir(p) and os.path.isfile(os.path.join(p, "rvc/infer/pipeline.py")):
            return p
    raise FileNotFoundError(
        f"Could not find Applio in any of: {APPLIO_CANDIDATE_PATHS}. "
        "Edit APPLIO_CANDIDATE_PATHS at the top of this script if installed elsewhere."
    )


def install_stage_timers(hubert_model, net_g) -> dict:
    """Patch the three stage callables to accumulate wall-clock time.

    Returns a dict that the patched functions mutate in place. Safe to call
    once per script run; the patches are not reversed (the script exits
    after reporting).
    """
    times = {"hubert_s": 0.0, "rmvpe_s": 0.0, "vits_s": 0.0}

    # --- HuBERT / ContentVec forward pass ---
    # Applio's voice_conversion calls `model(feats)` — an nn.Module, so patch
    # the instance's forward method. `__call__` dispatches through forward.
    orig_hubert_forward = hubert_model.forward

    def timed_hubert_forward(*args, **kwargs):
        t0 = time.time()
        out = orig_hubert_forward(*args, **kwargs)
        times["hubert_s"] += time.time() - t0
        return out

    hubert_model.forward = timed_hubert_forward

    # --- RMVPE.get_f0 (class-level patch) ---
    # Applio constructs a fresh RMVPE each call in pipeline.get_f0, so we must
    # patch at the class level rather than the instance.
    from rvc.lib.predictors.f0 import RMVPE as ApplioRMVPE

    orig_rmvpe_get_f0 = ApplioRMVPE.get_f0

    def timed_rmvpe_get_f0(self, *args, **kwargs):
        t0 = time.time()
        out = orig_rmvpe_get_f0(self, *args, **kwargs)
        times["rmvpe_s"] += time.time() - t0
        return out

    ApplioRMVPE.get_f0 = timed_rmvpe_get_f0

    # --- net_g.infer (enc_p + flow + vocoder) ---
    orig_infer = net_g.infer

    def timed_infer(*args, **kwargs):
        t0 = time.time()
        out = orig_infer(*args, **kwargs)
        times["vits_s"] += time.time() - t0
        return out

    net_g.infer = timed_infer

    return times


def load_applio_models(model_path: str, device: str):
    """Load HuBERT, RVC checkpoint, and Synthesizer. Returns (hubert, net_g, vc, tgt_sr, version, use_f0)."""
    from rvc.lib.utils import load_embedding
    from rvc.lib.algorithm.synthesizers import Synthesizer
    from rvc.configs.config import Config
    from rvc.infer.pipeline import Pipeline

    config = Config()
    # Config auto-picks device; caller may override.
    if device is not None:
        config.device = device

    t0 = time.time()
    hubert = load_embedding("contentvec").to(config.device).float().eval()
    hubert_load_s = time.time() - t0

    t0 = time.time()
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
    # enc_q is the posterior encoder; only needed at training time
    if hasattr(net_g, "enc_q"):
        del net_g.enc_q
    net_g.load_state_dict(cpt["weight"], strict=False)
    net_g = net_g.to(config.device).float().eval()
    model_load_s = time.time() - t0

    vc = Pipeline(tgt_sr, config)

    return {
        "hubert": hubert,
        "net_g": net_g,
        "vc": vc,
        "tgt_sr": tgt_sr,
        "version": version,
        "use_f0": use_f0,
        "device": config.device,
        "hubert_load_s": hubert_load_s,
        "model_load_s": model_load_s,
    }


def run_once(models: dict, audio: np.ndarray, stage_times: dict) -> tuple[np.ndarray, dict]:
    """Run Applio's pipeline once and return (audio_out, per-run timings)."""
    # Reset stage accumulators so callers get per-call timings.
    for k in stage_times:
        stage_times[k] = 0.0

    t0 = time.time()
    audio_out = models["vc"].pipeline(
        model=models["hubert"],
        net_g=models["net_g"],
        sid=0,
        audio=audio,
        pitch=0,
        f0_method="rmvpe",
        file_index="",
        index_rate=0.0,
        pitch_guidance=models["use_f0"],
        volume_envelope=1.0,
        version=models["version"],
        protect=0.5,
        f0_autotune=False,
        f0_autotune_strength=1.0,
        proposed_pitch=False,
        proposed_pitch_threshold=155.0,
    )
    total_s = time.time() - t0

    stage = dict(stage_times)
    stage["total_s"] = total_s
    stage["overhead_s"] = max(0.0, total_s - sum(stage_times.values()))
    return audio_out, stage


def format_row(label: str, *cells: str) -> str:
    return f"  {label:<20}" + "".join(f"{c:>14}" for c in cells)


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--model", required=True, help="Path to RVC v2 .pth voice model")
    p.add_argument("--audio", required=True, help="Path to input WAV")
    p.add_argument("--device", default=None, help="'cuda:0', 'cpu', or leave blank for Applio default")
    p.add_argument("--warm-runs", type=int, default=3, help="Number of warm runs to average (default 3)")
    p.add_argument("--output", default="/tmp/applio_baseline_out.wav", help="Where to save the last warm output")
    args = p.parse_args()

    applio_path = find_applio()
    sys.path.insert(0, applio_path)
    # Applio's code references relative paths like rvc/models/predictors/rmvpe.pt
    os.chdir(applio_path)

    from rvc.lib.utils import load_audio_infer

    print(f"Applio:     {applio_path}")
    print(f"Model:      {args.model}")
    print(f"Audio:      {args.audio}")

    audio = load_audio_infer(args.audio, 16000)
    duration_s = len(audio) / 16000
    print(f"Duration:   {duration_s:.2f}s @ 16kHz mono ({len(audio)} samples)")

    print("\n--- Loading ---")
    t_total_load = time.time()
    models = load_applio_models(args.model, device=args.device)
    total_load_s = time.time() - t_total_load
    print(f"Device:         {models['device']}")
    print(f"HuBERT load:    {models['hubert_load_s']:.2f}s")
    print(f"Model load:     {models['model_load_s']:.2f}s")
    print(f"Total load:     {total_load_s:.2f}s")
    print(f"Target SR:      {models['tgt_sr']} Hz")
    print(f"Model version:  {models['version']}, f0={models['use_f0']}")

    stage_times = install_stage_timers(models["hubert"], models["net_g"])

    print("\n--- Cold run (1st inference) ---")
    audio_out, cold = run_once(models, audio, stage_times)
    cold_rtf = cold["total_s"] / duration_s
    print(format_row("", "time_s", "frac_of_total"))
    for k in ("hubert_s", "rmvpe_s", "vits_s", "overhead_s"):
        frac = cold[k] / cold["total_s"] if cold["total_s"] > 0 else 0.0
        print(format_row(k, f"{cold[k]:.3f}", f"{frac*100:.1f}%"))
    print(format_row("total_s", f"{cold['total_s']:.3f}"))
    print(f"Cold RTF:   {cold_rtf:.3f}x  ({duration_s:.2f}s audio / {cold['total_s']:.2f}s wall)")

    print(f"\n--- Warm runs ({args.warm_runs}x) ---")
    warm_runs: list[dict] = []
    for i in range(args.warm_runs):
        audio_out, w = run_once(models, audio, stage_times)
        warm_runs.append(w)
        print(f"  run {i+1}: total={w['total_s']:.3f}s  "
              f"hubert={w['hubert_s']:.3f}  rmvpe={w['rmvpe_s']:.3f}  "
              f"vits={w['vits_s']:.3f}  overhead={w['overhead_s']:.3f}")

    def mean(key: str) -> float:
        return statistics.mean(r[key] for r in warm_runs)

    def stdev(key: str) -> float:
        return statistics.stdev(r[key] for r in warm_runs) if len(warm_runs) > 1 else 0.0

    mean_total = mean("total_s")
    warm_rtf = mean_total / duration_s

    print("\n--- Warm average ---")
    print(format_row("", "mean_s", "stdev_s", "frac"))
    for k in ("hubert_s", "rmvpe_s", "vits_s", "overhead_s"):
        frac = mean(k) / mean_total if mean_total > 0 else 0.0
        print(format_row(k, f"{mean(k):.3f}", f"{stdev(k):.3f}", f"{frac*100:.1f}%"))
    print(format_row("total_s", f"{mean_total:.3f}", f"{stdev('total_s'):.3f}"))
    print(f"Warm RTF:   {warm_rtf:.3f}x  ({duration_s:.2f}s audio / {mean_total:.2f}s wall)")

    sf.write(args.output, audio_out, models["tgt_sr"])
    print(f"\nLast warm output written to: {args.output}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
