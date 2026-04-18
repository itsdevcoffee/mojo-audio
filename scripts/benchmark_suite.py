"""Batch benchmark suite for mojo-audio voice conversion.

Wraps compare_vs_applio.py to run across multiple models and audio files,
saves structured JSON results for historical tracking, and displays trends.

Usage:

    # Run full suite (all discovered models, auto-selected audio)
    pixi run python scripts/benchmark_suite.py run

    # Run specific models
    pixi run python scripts/benchmark_suite.py run --models the-weeknd,drake

    # Run with a specific audio file
    pixi run python scripts/benchmark_suite.py run --audio /path/to/vocal.wav

    # Skip Applio (mojo-audio only, faster)
    pixi run python scripts/benchmark_suite.py run --mojo-only

    # Specify mojo device
    pixi run python scripts/benchmark_suite.py run --mojo-device gpu

    # View historical results
    pixi run python scripts/benchmark_suite.py history

    # Show trend for a specific model
    pixi run python scripts/benchmark_suite.py history --model the-weeknd

    # Compare two runs
    pixi run python scripts/benchmark_suite.py history --compare latest~1 latest
"""

from __future__ import annotations

import argparse
import glob
import json
import os
import platform
import socket
import subprocess
import sys
import time
import traceback
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

# ---------------------------------------------------------------------------
# Path constants
# ---------------------------------------------------------------------------

REPO_ROOT = Path(__file__).resolve().parent.parent
RESULTS_DIR = REPO_ROOT / "benchmarks" / "results"

SHADE_MODELS_DIRS = [
    "/home/visage/repos/shade/models",
]

APPLIO_AUDIO_DIRS = [
    "/home/visage/repos/Applio/logs",
]

APPLIO_CANDIDATE_PATHS = [
    "/home/visage/repos/Applio",
    "/home/maskkiller/repos/Applio",
]

MOJO_CANDIDATE_PATHS = [
    "/home/visage/repos/mojo-audio",
    "/home/maskkiller/dev-coffee/repos/mojo-audio",
]

# Cache file so repeated runs use the same default audio clip.
AUDIO_CACHE_FILE = RESULTS_DIR / ".default_audio_path"


# ---------------------------------------------------------------------------
# Discovery helpers
# ---------------------------------------------------------------------------


def discover_models(model_dirs: list[str]) -> list[dict]:
    """Scan model directories for subdirs containing model.pth.

    Returns a sorted list of dicts with 'name' and 'path' keys.
    """
    models = []
    for base in model_dirs:
        if not os.path.isdir(base):
            continue
        for entry in sorted(os.listdir(base)):
            model_pth = os.path.join(base, entry, "model.pth")
            if os.path.isfile(model_pth):
                models.append({"name": entry, "path": model_pth})
    return models


def discover_audio(audio_dirs: list[str], min_dur: float = 2.0, max_dur: float = 5.0) -> str | None:
    """Find a suitable test clip from Applio's sliced_audios directories.

    Returns the path to a WAV file between min_dur and max_dur seconds,
    or None if nothing suitable is found.
    """
    import soundfile as sf

    for base in audio_dirs:
        if not os.path.isdir(base):
            continue
        # Pattern: /home/visage/repos/Applio/logs/*/sliced_audios/*.wav
        pattern = os.path.join(base, "*", "sliced_audios", "*.wav")
        candidates = sorted(glob.glob(pattern))
        for path in candidates:
            try:
                info = sf.info(path)
                dur = info.frames / info.samplerate
                if min_dur <= dur <= max_dur:
                    return path
            except Exception:
                continue
    return None


def get_default_audio() -> str | None:
    """Return the cached default audio path, or discover and cache one."""
    if AUDIO_CACHE_FILE.is_file():
        cached = AUDIO_CACHE_FILE.read_text().strip()
        if os.path.isfile(cached):
            return cached

    path = discover_audio(APPLIO_AUDIO_DIRS)
    if path:
        AUDIO_CACHE_FILE.parent.mkdir(parents=True, exist_ok=True)
        AUDIO_CACHE_FILE.write_text(path)
    return path


# ---------------------------------------------------------------------------
# Metadata collection
# ---------------------------------------------------------------------------


def collect_metadata(
    mojo_device: str,
    applio_device: str | None,
    run_type: str,
) -> dict:
    """Gather environment metadata for the run."""
    meta = {
        "timestamp": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "git_commit": _shell("git rev-parse --short HEAD"),
        "git_branch": _shell("git rev-parse --abbrev-ref HEAD"),
        "max_version": _get_max_version(),
        "platform": platform.machine(),
        "hostname": socket.gethostname(),
        "mojo_device": mojo_device,
        "applio_device": applio_device or "auto",
        "python_version": platform.python_version(),
        "run_type": run_type,
    }
    return meta


def _shell(cmd: str) -> str:
    try:
        return subprocess.check_output(
            cmd, shell=True, text=True, stderr=subprocess.DEVNULL, cwd=str(REPO_ROOT)
        ).strip()
    except Exception:
        return "unknown"


def _get_max_version() -> str:
    raw = _shell("pixi list max 2>/dev/null")
    for line in raw.splitlines():
        parts = line.split()
        if len(parts) >= 2 and parts[0] == "max":
            return parts[1]
    return "unknown"


# ---------------------------------------------------------------------------
# Import bridge to compare_vs_applio.py
# ---------------------------------------------------------------------------


def _setup_imports(skip_applio: bool) -> dict:
    """Set up sys.path and imports, return dict of available callables.

    Keys: run_applio, run_mojo, compare_waveforms, compare_spectrograms,
          compare_f0, extract_applio_f0, load_audio, applio_available.
    """
    scripts_dir = REPO_ROOT / "scripts"
    if str(scripts_dir) not in sys.path:
        sys.path.insert(0, str(scripts_dir))

    # compare_vs_applio uses find_dir to locate mojo-audio src
    mojo_path = None
    for p in MOJO_CANDIDATE_PATHS:
        if os.path.isdir(p) and os.path.isfile(os.path.join(p, "src", "models", "voice_converter.py")):
            mojo_path = p
            break
    if mojo_path is None:
        raise FileNotFoundError("Cannot find mojo-audio source tree with src/models/voice_converter.py")

    src_path = os.path.join(mojo_path, "src")
    if src_path not in sys.path:
        sys.path.insert(0, src_path)

    # Try to set up Applio
    applio_available = False
    applio_path = None
    if not skip_applio:
        for p in APPLIO_CANDIDATE_PATHS:
            if os.path.isdir(p) and os.path.isfile(os.path.join(p, "rvc", "infer", "pipeline.py")):
                applio_path = p
                break
        if applio_path:
            if applio_path not in sys.path:
                sys.path.insert(0, applio_path)
            os.chdir(applio_path)  # Applio expects CWD to be its root
            applio_available = True
        else:
            print("WARNING: Applio not found. Running mojo-audio only.")

    from compare_vs_applio import (
        run_applio,
        run_mojo,
        compare_waveforms,
        compare_spectrograms,
        compare_f0,
        extract_applio_f0,
    )

    # Audio loading: try Applio's loader first, fall back to soundfile
    load_audio = None
    if applio_available:
        try:
            from rvc.lib.utils import load_audio_infer
            load_audio = lambda path: load_audio_infer(path, 16000).astype(np.float32)
        except ImportError:
            pass

    if load_audio is None:
        import soundfile as sf
        from scipy.signal import resample_poly
        from math import gcd

        def _load_audio_sf(path: str) -> np.ndarray:
            audio, sr = sf.read(path, dtype="float32")
            if audio.ndim > 1:
                audio = audio.mean(axis=1)
            if sr != 16000:
                g = gcd(sr, 16000)
                audio = resample_poly(audio, 16000 // g, sr // g).astype(np.float32)
            return audio

        load_audio = _load_audio_sf

    return {
        "run_applio": run_applio,
        "run_mojo": run_mojo,
        "compare_waveforms": compare_waveforms,
        "compare_spectrograms": compare_spectrograms,
        "compare_f0": compare_f0,
        "extract_applio_f0": extract_applio_f0,
        "load_audio": load_audio,
        "applio_available": applio_available,
        "applio_path": applio_path,
    }


# ---------------------------------------------------------------------------
# Single-model benchmark
# ---------------------------------------------------------------------------


def benchmark_one_model(
    model_name: str,
    model_path: str,
    audio_16k: np.ndarray,
    audio_file: str,
    funcs: dict,
    mojo_device: str,
    applio_device: str | None,
    mojo_only: bool,
) -> dict:
    """Run benchmark for a single model. Returns a result dict."""
    duration_s = len(audio_16k) / 16000
    result = {
        "model_name": model_name,
        "model_path": model_path,
        "audio_file": os.path.basename(audio_file),
        "audio_duration_s": round(duration_s, 3),
        "status": "ok",
    }

    # ---- Applio ----
    applio_out = None
    applio_sr = None
    if not mojo_only and funcs["applio_available"]:
        try:
            applio_out, applio_sr, applio_times = funcs["run_applio"](
                model_path, audio_16k, applio_device
            )
            result["applio"] = {
                "load_s": round(applio_times["load_s"], 3),
                "hubert_s": round(applio_times["hubert_s"], 3),
                "rmvpe_s": round(applio_times["rmvpe_s"], 3),
                "vits_s": round(applio_times["vits_s"], 3),
                "overhead_s": round(applio_times["overhead_s"], 3),
                "total_s": round(applio_times["total_s"], 3),
                "rtf": round(applio_times["total_s"] / duration_s, 3),
            }
        except Exception as e:
            result["applio"] = {"error": f"{type(e).__name__}: {e}"}
            traceback.print_exc()
    elif mojo_only:
        result["applio"] = {"skipped": "mojo-only mode"}
    else:
        result["applio"] = {"skipped": "Applio not available"}

    # ---- mojo-audio ----
    mojo_out = None
    mojo_sr = None
    mojo_f0 = None
    try:
        mojo_out, mojo_sr, mojo_f0, mojo_times = funcs["run_mojo"](
            model_path, audio_16k, mojo_device
        )
        result["mojo"] = {
            "load_s": round(mojo_times["load_s"], 3),
            "hubert_s": round(mojo_times["hubert_s"], 3),
            "rmvpe_s": round(mojo_times["rmvpe_s"], 3),
            "vits_s": round(mojo_times["vits_s"], 3),
            "overhead_s": round(mojo_times["overhead_s"], 3),
            "total_s": round(mojo_times["total_s"], 3),
            "rtf": round(mojo_times["total_s"] / duration_s, 3),
        }
    except Exception as e:
        result["mojo"] = {"error": f"{type(e).__name__}: {e}"}
        result["status"] = "mojo_failed"
        traceback.print_exc()
        return result

    # ---- Speedup ----
    if (
        applio_out is not None
        and "total_s" in result.get("applio", {})
        and "total_s" in result.get("mojo", {})
    ):
        result["speedup"] = round(
            result["applio"]["total_s"] / result["mojo"]["total_s"], 3
        )

    # ---- Quality metrics (only when both outputs exist) ----
    if applio_out is not None and mojo_out is not None:
        try:
            from scipy.signal import resample_poly
            from math import gcd

            cmp_sr = min(applio_sr, mojo_sr)

            def _resample(x: np.ndarray, src: int, dst: int) -> np.ndarray:
                if src == dst:
                    return x
                g = gcd(src, dst)
                return resample_poly(x, dst // g, src // g).astype(np.float32)

            a_cmp = _resample(applio_out, applio_sr, cmp_sr)
            m_cmp = _resample(mojo_out, mojo_sr, cmp_sr)

            wf = funcs["compare_waveforms"](a_cmp, m_cmp)
            sp = funcs["compare_spectrograms"](a_cmp, m_cmp, cmp_sr)

            quality = {
                "waveform_corr": round(wf["corr"], 4) if not np.isnan(wf["corr"]) else None,
                "rms_diff_db": round(wf["rms_diff_db"], 2),
                "spec_corr": round(sp["spec_corr"], 4),
                "spec_max_diff_db": round(sp["spec_max_diff_db"], 2),
                "spec_mean_diff_db": round(sp["spec_mean_diff_db"], 2),
            }

            # F0 comparison
            if mojo_f0 is not None:
                try:
                    applio_f0 = funcs["extract_applio_f0"](audio_16k, applio_device)
                    f0m = funcs["compare_f0"](applio_f0, mojo_f0)
                    quality["f0_voicing_agreement"] = round(
                        f0m["voicing_agreement_pct"] / 100.0, 4
                    )
                    if "mean_cent_error" in f0m:
                        quality["f0_mean_cents"] = round(f0m["mean_cent_error"], 2)
                        quality["f0_median_cents"] = round(f0m["median_cent_error"], 2)
                        quality["f0_within_5_cents"] = round(
                            f0m["pct_within_5_cents"] / 100.0, 4
                        )
                        quality["f0_within_50_cents"] = round(
                            f0m["pct_within_50_cents"] / 100.0, 4
                        )
                except Exception as e:
                    quality["f0_error"] = f"{type(e).__name__}: {e}"

            result["quality"] = quality
        except Exception as e:
            result["quality"] = {"error": f"{type(e).__name__}: {e}"}
            traceback.print_exc()

    return result


# ---------------------------------------------------------------------------
# Run subcommand
# ---------------------------------------------------------------------------


def cmd_run(args: argparse.Namespace) -> int:
    """Execute benchmarks across models."""
    mojo_only = args.mojo_only

    # Set up imports
    print("Setting up imports...")
    funcs = _setup_imports(skip_applio=mojo_only)

    # Discover or filter models
    all_models = discover_models(SHADE_MODELS_DIRS)
    if not all_models:
        print("ERROR: No voice models found in any of:", SHADE_MODELS_DIRS)
        print("Provide a model directory that contains subdirs with model.pth files.")
        return 1

    if args.models:
        requested = [m.strip() for m in args.models.split(",")]
        model_lookup = {m["name"]: m for m in all_models}
        models = []
        for name in requested:
            if name in model_lookup:
                models.append(model_lookup[name])
            else:
                print(f"WARNING: Model '{name}' not found. Available: {sorted(model_lookup.keys())}")
        if not models:
            print("ERROR: None of the requested models were found.")
            return 1
    else:
        models = all_models

    # Resolve audio
    audio_file = args.audio
    if audio_file is None:
        audio_file = get_default_audio()
    if audio_file is None:
        print("ERROR: No test audio file found and none provided via --audio.")
        print("Provide an audio file with: --audio /path/to/vocal.wav")
        return 1
    if not os.path.isfile(audio_file):
        print(f"ERROR: Audio file not found: {audio_file}")
        return 1

    print(f"Loading audio: {audio_file}")
    audio_16k = funcs["load_audio"](audio_file)
    duration_s = len(audio_16k) / 16000
    print(f"Audio duration: {duration_s:.2f}s ({len(audio_16k)} samples @ 16 kHz)")

    # Determine run type
    run_type = "full"
    if mojo_only:
        run_type = "mojo-only"
    elif not funcs["applio_available"]:
        run_type = "mojo-only (applio unavailable)"

    applio_device = args.applio_device
    mojo_device = args.mojo_device

    # Collect metadata
    metadata = collect_metadata(mojo_device, applio_device, run_type)

    print()
    print("=" * 60)
    print("  mojo-audio Benchmark Suite")
    print("=" * 60)
    print(f"  Models:    {len(models)}")
    print(f"  Audio:     {os.path.basename(audio_file)} ({duration_s:.1f}s)")
    print(f"  Run type:  {run_type}")
    print(f"  Device:    mojo={mojo_device}  applio={applio_device or 'auto'}")
    print(f"  Commit:    {metadata['git_commit']} ({metadata['git_branch']})")
    print(f"  MAX:       {metadata['max_version']}")
    print(f"  Host:      {metadata['hostname']} ({metadata['platform']})")
    print("=" * 60)
    print()

    # Run benchmarks
    results = []
    for i, model in enumerate(models, 1):
        print(f"[{i}/{len(models)}] {model['name']}...", end="", flush=True)
        t0 = time.time()
        try:
            result = benchmark_one_model(
                model_name=model["name"],
                model_path=model["path"],
                audio_16k=audio_16k,
                audio_file=audio_file,
                funcs=funcs,
                mojo_device=mojo_device,
                applio_device=applio_device,
                mojo_only=mojo_only,
            )
            elapsed = time.time() - t0

            # Print one-line summary
            status_icon = "OK" if result["status"] == "ok" else "FAIL"
            mojo_rtf = result.get("mojo", {}).get("rtf", None)
            speedup = result.get("speedup", None)
            parts = [f" {status_icon} ({elapsed:.1f}s)"]
            if mojo_rtf is not None:
                parts.append(f"RTF={mojo_rtf:.3f}")
            if speedup is not None:
                parts.append(f"speedup={speedup:.2f}x")
            print("  ".join(parts))

        except Exception as e:
            elapsed = time.time() - t0
            print(f"  ERROR ({elapsed:.1f}s): {type(e).__name__}: {e}")
            traceback.print_exc()
            result = {
                "model_name": model["name"],
                "model_path": model["path"],
                "audio_file": os.path.basename(audio_file),
                "status": "error",
                "error": f"{type(e).__name__}: {e}",
            }

        results.append(result)

    # Compute summary
    ok_results = [r for r in results if r.get("status") == "ok"]
    summary = {
        "models_tested": len(results),
        "models_ok": len(ok_results),
        "models_failed": len(results) - len(ok_results),
    }

    mojo_rtfs = [r["mojo"]["rtf"] for r in ok_results if "rtf" in r.get("mojo", {})]
    speedups = [r["speedup"] for r in ok_results if "speedup" in r]
    f0_cents = [
        r["quality"]["f0_mean_cents"]
        for r in ok_results
        if "quality" in r and "f0_mean_cents" in r.get("quality", {})
    ]

    if mojo_rtfs:
        summary["mean_rtf_mojo"] = round(sum(mojo_rtfs) / len(mojo_rtfs), 3)
        summary["min_rtf_mojo"] = round(min(mojo_rtfs), 3)
        summary["max_rtf_mojo"] = round(max(mojo_rtfs), 3)
    if speedups:
        summary["mean_speedup"] = round(sum(speedups) / len(speedups), 3)
    if f0_cents:
        summary["mean_f0_cents"] = round(sum(f0_cents) / len(f0_cents), 2)

    # Build output document
    run_data = {
        "metadata": metadata,
        "results": results,
        "summary": summary,
    }

    # Save
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    ts = datetime.now().strftime("%Y%m%dT%H%M%S")
    out_path = RESULTS_DIR / f"{ts}.json"
    with open(out_path, "w") as f:
        json.dump(run_data, f, indent=2)

    # Print summary
    print()
    print("=" * 60)
    print("  Summary")
    print("=" * 60)
    print(f"  Models OK / total:  {summary['models_ok']} / {summary['models_tested']}")
    if "mean_rtf_mojo" in summary:
        print(f"  Mean RTF (mojo):    {summary['mean_rtf_mojo']:.3f}")
        print(f"  RTF range:          {summary['min_rtf_mojo']:.3f} - {summary['max_rtf_mojo']:.3f}")
    if "mean_speedup" in summary:
        print(f"  Mean speedup:       {summary['mean_speedup']:.2f}x")
    if "mean_f0_cents" in summary:
        print(f"  Mean F0 error:      {summary['mean_f0_cents']:.2f} cents")
    print(f"  Results saved to:   {out_path}")
    print("=" * 60)

    return 0


# ---------------------------------------------------------------------------
# History subcommand
# ---------------------------------------------------------------------------


def _load_all_runs() -> list[dict]:
    """Load all JSON result files from benchmarks/results/, sorted by timestamp."""
    runs = []
    if not RESULTS_DIR.is_dir():
        return runs
    for path in sorted(RESULTS_DIR.glob("*.json")):
        try:
            with open(path) as f:
                data = json.load(f)
            data["_path"] = str(path)
            data["_filename"] = path.stem
            runs.append(data)
        except (json.JSONDecodeError, KeyError):
            continue
    return runs


def _resolve_run_ref(runs: list[dict], ref: str) -> dict | None:
    """Resolve a run reference like 'latest', 'latest~1', or a filename prefix."""
    if ref.startswith("latest"):
        offset = 0
        if "~" in ref:
            try:
                offset = int(ref.split("~")[1])
            except (IndexError, ValueError):
                return None
        idx = len(runs) - 1 - offset
        if 0 <= idx < len(runs):
            return runs[idx]
        return None
    # Try matching by filename prefix
    for run in runs:
        if run["_filename"].startswith(ref):
            return run
    return None


def _format_table(headers: list[str], rows: list[list[str]], col_widths: list[int] | None = None) -> str:
    """Simple ASCII table formatter."""
    if col_widths is None:
        col_widths = [max(len(h), max((len(str(r)) for r in col), default=0)) for h, col in zip(headers, zip(*rows))] if rows else [len(h) for h in headers]

    lines = []
    header_line = "  ".join(h.ljust(w) for h, w in zip(headers, col_widths))
    lines.append(header_line)
    lines.append("  ".join("-" * w for w in col_widths))
    for row in rows:
        lines.append("  ".join(str(v).ljust(w) for v, w in zip(row, col_widths)))
    return "\n".join(lines)


def cmd_history(args: argparse.Namespace) -> int:
    """Display historical benchmark results."""
    runs = _load_all_runs()
    if not runs:
        print("No benchmark results found in", RESULTS_DIR)
        print("Run benchmarks first: pixi run python scripts/benchmark_suite.py run")
        return 1

    # --compare mode
    if args.compare:
        if len(args.compare) != 2:
            print("ERROR: --compare requires exactly two run references (e.g., latest~1 latest)")
            return 1
        ref_a, ref_b = args.compare
        run_a = _resolve_run_ref(runs, ref_a)
        run_b = _resolve_run_ref(runs, ref_b)
        if run_a is None:
            print(f"ERROR: Cannot resolve run reference '{ref_a}'")
            return 1
        if run_b is None:
            print(f"ERROR: Cannot resolve run reference '{ref_b}'")
            return 1
        return _print_comparison(run_a, run_b, ref_a, ref_b)

    # --model mode: show trend for one model
    if args.model:
        return _print_model_trend(runs, args.model)

    # Default: overview table of all runs
    return _print_overview(runs)


def _print_overview(runs: list[dict]) -> int:
    """Print a summary table of all historical runs."""
    headers = ["Date", "Commit", "Branch", "Models", "OK", "RTF", "Speedup", "F0 cents", "Type"]
    rows = []
    for run in runs:
        meta = run.get("metadata", {})
        summary = run.get("summary", {})
        ts = meta.get("timestamp", "?")
        # Parse ISO timestamp to a shorter display format
        try:
            dt = datetime.fromisoformat(ts)
            date_str = dt.strftime("%Y-%m-%d %H:%M")
        except (ValueError, TypeError):
            date_str = ts[:16] if len(ts) >= 16 else ts

        rows.append([
            date_str,
            meta.get("git_commit", "?"),
            meta.get("git_branch", "?"),
            str(summary.get("models_tested", "?")),
            str(summary.get("models_ok", "?")),
            f"{summary['mean_rtf_mojo']:.3f}" if "mean_rtf_mojo" in summary else "-",
            f"{summary['mean_speedup']:.2f}x" if "mean_speedup" in summary else "-",
            f"{summary['mean_f0_cents']:.1f}" if "mean_f0_cents" in summary else "-",
            meta.get("run_type", "?"),
        ])

    print()
    print("Benchmark History")
    print("=" * 100)
    print(_format_table(headers, rows))
    print()
    print(f"{len(runs)} run(s) found in {RESULTS_DIR}")
    return 0


def _print_model_trend(runs: list[dict], model_name: str) -> int:
    """Show a specific model's metrics across all runs."""
    headers = ["Date", "Commit", "RTF", "Speedup", "Spec corr", "F0 cents", "Status"]
    rows = []
    found_any = False

    for run in runs:
        meta = run.get("metadata", {})
        ts = meta.get("timestamp", "?")
        try:
            dt = datetime.fromisoformat(ts)
            date_str = dt.strftime("%Y-%m-%d %H:%M")
        except (ValueError, TypeError):
            date_str = ts[:16] if len(ts) >= 16 else ts

        for result in run.get("results", []):
            if result.get("model_name") != model_name:
                continue
            found_any = True
            mojo = result.get("mojo", {})
            quality = result.get("quality", {})
            rows.append([
                date_str,
                meta.get("git_commit", "?"),
                f"{mojo['rtf']:.3f}" if "rtf" in mojo else "-",
                f"{result['speedup']:.2f}x" if "speedup" in result else "-",
                f"{quality['spec_corr']:.4f}" if "spec_corr" in quality else "-",
                f"{quality['f0_mean_cents']:.2f}" if "f0_mean_cents" in quality else "-",
                result.get("status", "?"),
            ])

    if not found_any:
        print(f"No results found for model '{model_name}'.")
        # List available models
        all_model_names = set()
        for run in runs:
            for result in run.get("results", []):
                name = result.get("model_name")
                if name:
                    all_model_names.add(name)
        if all_model_names:
            print(f"Available models: {', '.join(sorted(all_model_names))}")
        return 1

    print()
    print(f"Trend for model: {model_name}")
    print("=" * 90)
    print(_format_table(headers, rows))
    print()
    return 0


def _print_comparison(run_a: dict, run_b: dict, label_a: str, label_b: str) -> int:
    """Compare two benchmark runs side by side."""
    meta_a = run_a.get("metadata", {})
    meta_b = run_b.get("metadata", {})

    print()
    print(f"Comparing: {label_a} vs {label_b}")
    print("=" * 80)
    print(f"  {'':20} {'[A] ' + label_a:>25}  {'[B] ' + label_b:>25}")
    print(f"  {'Date':20} {meta_a.get('timestamp', '?')[:19]:>25}  {meta_b.get('timestamp', '?')[:19]:>25}")
    print(f"  {'Commit':20} {meta_a.get('git_commit', '?'):>25}  {meta_b.get('git_commit', '?'):>25}")
    print(f"  {'Branch':20} {meta_a.get('git_branch', '?'):>25}  {meta_b.get('git_branch', '?'):>25}")
    print(f"  {'MAX version':20} {meta_a.get('max_version', '?'):>25}  {meta_b.get('max_version', '?'):>25}")
    print()

    # Build model lookup for both runs
    results_a = {r["model_name"]: r for r in run_a.get("results", [])}
    results_b = {r["model_name"]: r for r in run_b.get("results", [])}
    all_models = sorted(set(results_a.keys()) | set(results_b.keys()))

    if not all_models:
        print("No model results to compare.")
        return 0

    headers = ["Model", "RTF [A]", "RTF [B]", "Delta", "Speedup [A]", "Speedup [B]", "Status"]
    rows = []
    for model in all_models:
        ra = results_a.get(model, {})
        rb = results_b.get(model, {})
        rtf_a = ra.get("mojo", {}).get("rtf")
        rtf_b = rb.get("mojo", {}).get("rtf")

        rtf_a_str = f"{rtf_a:.3f}" if rtf_a is not None else "-"
        rtf_b_str = f"{rtf_b:.3f}" if rtf_b is not None else "-"

        if rtf_a is not None and rtf_b is not None:
            delta = rtf_b - rtf_a
            pct = (delta / rtf_a) * 100
            if abs(pct) < 1:
                delta_str = f"{delta:+.3f} (~)"
            elif delta < 0:
                delta_str = f"{delta:+.3f} ({pct:+.1f}%)"  # negative = faster
            else:
                delta_str = f"{delta:+.3f} ({pct:+.1f}%)"
        else:
            delta_str = "-"

        speedup_a = ra.get("speedup")
        speedup_b = rb.get("speedup")
        speedup_a_str = f"{speedup_a:.2f}x" if speedup_a is not None else "-"
        speedup_b_str = f"{speedup_b:.2f}x" if speedup_b is not None else "-"

        status_a = ra.get("status", "missing")
        status_b = rb.get("status", "missing")
        if status_a == status_b == "ok":
            status = "ok"
        else:
            status = f"{status_a}/{status_b}"

        rows.append([model, rtf_a_str, rtf_b_str, delta_str, speedup_a_str, speedup_b_str, status])

    print(_format_table(headers, rows))

    # Summary comparison
    sum_a = run_a.get("summary", {})
    sum_b = run_b.get("summary", {})
    print()
    print("  Summary:")
    for key, label in [
        ("mean_rtf_mojo", "Mean RTF (mojo)"),
        ("mean_speedup", "Mean speedup"),
        ("mean_f0_cents", "Mean F0 error (cents)"),
    ]:
        va = sum_a.get(key)
        vb = sum_b.get(key)
        va_str = f"{va:.3f}" if va is not None else "-"
        vb_str = f"{vb:.3f}" if vb is not None else "-"
        if va is not None and vb is not None:
            diff = vb - va
            diff_str = f"{diff:+.3f}"
        else:
            diff_str = "-"
        print(f"    {label:25} {va_str:>10}  {vb_str:>10}  {diff_str:>10}")

    print()
    return 0


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="benchmark_suite",
        description="Batch benchmark suite for mojo-audio voice conversion.",
    )
    sub = parser.add_subparsers(dest="command", required=True)

    # --- run ---
    p_run = sub.add_parser("run", help="Execute benchmarks across models")
    p_run.add_argument(
        "--models",
        default=None,
        help="Comma-separated list of model names to test (default: all discovered)",
    )
    p_run.add_argument(
        "--audio",
        default=None,
        help="Path to input WAV file (default: auto-discover from Applio logs)",
    )
    p_run.add_argument(
        "--mojo-only",
        action="store_true",
        help="Skip Applio, benchmark mojo-audio only",
    )
    p_run.add_argument(
        "--mojo-device",
        default="auto",
        help="mojo-audio device: auto, cpu, gpu (default: auto)",
    )
    p_run.add_argument(
        "--applio-device",
        default=None,
        help="Applio torch device (default: auto-detect)",
    )

    # --- history ---
    p_hist = sub.add_parser("history", help="View historical benchmark results")
    p_hist.add_argument(
        "--model",
        default=None,
        help="Show trend for a specific model name",
    )
    p_hist.add_argument(
        "--compare",
        nargs=2,
        metavar="REF",
        help="Compare two runs (e.g., latest~1 latest)",
    )

    return parser


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()

    if args.command == "run":
        return cmd_run(args)
    elif args.command == "history":
        return cmd_history(args)
    else:
        parser.print_help()
        return 1


if __name__ == "__main__":
    sys.exit(main())
