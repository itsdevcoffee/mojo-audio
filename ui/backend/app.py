"""
FastAPI backend for mojo-audio benchmark UI.

Provides endpoints for running Mojo and librosa benchmarks.
"""

import subprocess
import tempfile
from pathlib import Path

import numpy as np
from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

app = FastAPI(title="mojo-audio Benchmark API")

# CORS for local development only
# WARNING: For production, restrict allow_origins to specific domains
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Configuration
REPO_ROOT = Path(__file__).parent.parent.parent
UI_ROOT = Path(__file__).parent.parent

ALLOWED_BLAS_BACKENDS = frozenset(["mkl", "openblas"])
BENCHMARK_TIMEOUT_SECONDS = 120
MAX_UPLOAD_BYTES = 100 * 1024 * 1024  # 100 MB

# Serve static files (relative to ui directory)
app.mount("/static", StaticFiles(directory=str(UI_ROOT / "static")), name="static")


class BenchmarkConfig(BaseModel):
    """Benchmark configuration with validated parameter ranges."""
    duration: int = Field(default=30, ge=1, le=300, description="Audio duration in seconds")
    n_fft: int = Field(default=400, ge=64, le=8192, description="FFT window size")
    hop_length: int = Field(default=160, ge=1, le=4096, description="Hop length between frames")
    n_mels: int = Field(default=80, ge=1, le=256, description="Number of mel bands")
    iterations: int = Field(default=20, ge=1, le=100, description="Benchmark iterations")
    # BLAS backend for librosa/scipy - does NOT affect mojo-audio (pure Mojo FFT)
    blas_backend: str = Field(default="mkl", pattern="^(mkl|openblas)$", description="BLAS backend")


class BenchmarkResult(BaseModel):
    """Benchmark result."""
    implementation: str
    duration: int
    avg_time_ms: float
    std_time_ms: float = 0.0
    throughput_realtime: float
    iterations: int
    success: bool
    error: str = ""


@app.get("/")
async def root():
    """Serve the main UI."""
    return FileResponse(str(UI_ROOT / "frontend" / "index.html"))


def parse_benchmark_output(stdout: str) -> tuple[float, float]:
    """Parse benchmark output in avg,std format. Returns (avg_time_ms, std_time_ms)."""
    parts = stdout.strip().split(',')
    avg_time = float(parts[0])
    std_time = float(parts[1]) if len(parts) > 1 else 0.0
    return avg_time, std_time


def calculate_throughput(duration_seconds: int, avg_time_ms: float) -> float:
    """Calculate realtime throughput ratio (how many times faster than realtime)."""
    return duration_seconds / (avg_time_ms / 1000.0)


def run_benchmark_subprocess(cmd: list[str], error_prefix: str) -> str:
    """Run a benchmark subprocess and return stdout on success."""
    result = subprocess.run(
        cmd,
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
        timeout=BENCHMARK_TIMEOUT_SECONDS
    )

    if result.returncode != 0:
        raise HTTPException(
            status_code=500,
            detail=f"{error_prefix}: {result.stderr}"
        )

    return result.stdout


@app.post("/api/benchmark/mojo")
async def benchmark_mojo(config: BenchmarkConfig) -> BenchmarkResult:
    """
    Run mojo-audio benchmark.

    Returns performance metrics for the optimized Mojo implementation.
    Note: mojo-audio uses pure Mojo FFT - BLAS backend setting is ignored.
    """
    try:
        cmd = [
            "python", "ui/backend/run_benchmark.py",
            "mojo",
            str(config.duration),
            str(config.iterations),
            str(config.n_fft),
            str(config.hop_length),
            str(config.n_mels)
        ]

        stdout = run_benchmark_subprocess(cmd, "Mojo benchmark failed")
        avg_time, std_time = parse_benchmark_output(stdout)

        return BenchmarkResult(
            implementation="mojo-audio",
            duration=config.duration,
            avg_time_ms=avg_time,
            std_time_ms=std_time,
            throughput_realtime=calculate_throughput(config.duration, avg_time),
            iterations=config.iterations,
            success=True
        )

    except subprocess.TimeoutExpired:
        raise HTTPException(status_code=408, detail="Benchmark timeout")
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/benchmark/librosa")
async def benchmark_librosa(config: BenchmarkConfig) -> BenchmarkResult:
    """
    Run librosa benchmark.

    Returns performance metrics for Python's librosa.
    Supports switching between MKL and OpenBLAS backends via blas_backend config.
    Note: BLAS backend only affects librosa - mojo-audio uses pure Mojo FFT.
    """
    try:
        if config.blas_backend not in ALLOWED_BLAS_BACKENDS:
            raise HTTPException(
                status_code=400,
                detail="Invalid blas_backend: must be 'mkl' or 'openblas'"
            )

        cmd = [
            "pixi", "run", "-e", config.blas_backend,
            "python", "ui/backend/run_benchmark.py",
            "librosa",
            str(config.duration),
            str(config.iterations),
            str(config.n_fft),
            str(config.hop_length),
            str(config.n_mels)
        ]

        stdout = run_benchmark_subprocess(cmd, "librosa benchmark failed")
        avg_time, std_time = parse_benchmark_output(stdout)

        return BenchmarkResult(
            implementation=f"librosa ({config.blas_backend.upper()})",
            duration=config.duration,
            avg_time_ms=avg_time,
            std_time_ms=std_time,
            throughput_realtime=calculate_throughput(config.duration, avg_time),
            iterations=config.iterations,
            success=True
        )

    except subprocess.TimeoutExpired:
        raise HTTPException(status_code=408, detail="Benchmark timeout")
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/benchmark/both")
async def benchmark_both(config: BenchmarkConfig):
    """
    Run both benchmarks and return comparison.

    Returns side-by-side results with speedup calculation.
    """
    try:
        mojo_result = await benchmark_mojo(config)
        librosa_result = await benchmark_librosa(config)

        speedup = librosa_result.avg_time_ms / mojo_result.avg_time_ms
        faster_pct = ((librosa_result.avg_time_ms - mojo_result.avg_time_ms) / librosa_result.avg_time_ms) * 100

        return {
            "mojo": mojo_result,
            "librosa": librosa_result,
            "speedup_factor": round(speedup, 2),
            "faster_percentage": round(faster_pct, 1),
            "mojo_is_faster": mojo_result.avg_time_ms < librosa_result.avg_time_ms
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/health")
async def health():
    """Health check endpoint."""
    return {"status": "healthy", "service": "mojo-audio benchmark API"}


@app.post("/analyze")
async def analyze_audio(file: UploadFile = File(...)):
    """
    Accept a WAV file, process via mojo-audio DSP, return visualization data.
    """
    if not file.filename or not file.filename.lower().endswith(".wav"):
        raise HTTPException(status_code=400, detail="Only WAV files are accepted")

    content = await file.read()
    if len(content) > MAX_UPLOAD_BYTES:
        raise HTTPException(status_code=413, detail="File too large (max 100 MB)")

    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
        tmp.write(content)
        tmp_path = tmp.name

    try:
        import asyncio
        return await asyncio.to_thread(_process_audio, tmp_path)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Audio processing failed: {e}")
    finally:
        Path(tmp_path).unlink(missing_ok=True)


def _process_audio(wav_path: str) -> dict:
    """Process audio through mojo-audio DSP via subprocess and return visualization data.

    Mojo modules (.mojo) aren't Python-importable — they must be run via the
    Mojo compiler.  We generate a temp .mojo script that does all DSP and
    prints raw numeric data to stdout, then post-process in Python/numpy.
    """
    import json as _json

    # Mojo script: read WAV, resample, mel spec, VAD — output as JSON lines
    mojo_code = f"""
from wav_io import read_wav
from resample import resample_to_16k
from vad import get_voice_segments
from audio import mel_spectrogram, stft
from math import sqrt

fn main() raises:
    # 1. Load audio
    var tup = read_wav("{wav_path}")
    var samples = tup[0].copy()
    var sample_rate = tup[1]
    var n_samples = len(samples)
    var duration_s = Float64(n_samples) / Float64(sample_rate)

    # Print header: duration_s, sample_rate, n_samples
    print(String(duration_s) + "," + String(sample_rate) + "," + String(n_samples))

    # 2. Waveform RMS envelope (~4000 points)
    var display_points = 4000
    var hop = n_samples // display_points
    if hop < 1:
        hop = 1
    var wf_parts = List[String]()
    var i = 0
    while i < n_samples:
        var end = i + hop
        if end > n_samples:
            end = n_samples
        var sum_sq: Float64 = 0.0
        for j in range(i, end):
            var v = Float64(samples[j])
            sum_sq += v * v
        var rms = sqrt(sum_sq / Float64(end - i))
        wf_parts.append(String(Float32(rms)))
        i = end
    # Join with commas
    var wf_line = String("")
    for idx in range(len(wf_parts)):
        if idx > 0:
            wf_line += ","
        wf_line += wf_parts[idx]
    print(wf_line)

    # 3. Resample to 16kHz
    var samples_16k = resample_to_16k(samples, sample_rate)

    # 4. Mel spectrogram (80 bands)
    var mel = mel_spectrogram(samples_16k, n_fft=400, hop_length=160, n_mels=80)
    var n_mels = len(mel)
    var n_frames = len(mel[0]) if n_mels > 0 else 0
    print(String(n_mels) + "," + String(n_frames))

    # Print mel data row by row (each row = one mel band, comma-separated)
    for m in range(n_mels):
        var row_parts = List[String]()
        for f in range(n_frames):
            row_parts.append(String(mel[m][f]))
        var row_line = String("")
        for idx in range(len(row_parts)):
            if idx > 0:
                row_line += ","
            row_line += row_parts[idx]
        print(row_line)

    # 5. VAD segments
    var segs = get_voice_segments(samples, sample_rate)
    var n_segs = len(segs)
    print(String(n_segs))
    for s in range(n_segs):
        print(String(segs[s][0]) + "," + String(segs[s][1]))

    # 6. STFT magnitude (full resolution)
    var n_fft = 400
    var hop_length = 160
    var stft_result = stft(samples_16k, n_fft, hop_length)
    var stft_n_frames = len(stft_result)
    var stft_n_freq_bins = len(stft_result[0]) if stft_n_frames > 0 else 0
    print("STFT_SHAPE:" + String(stft_n_frames) + "," + String(stft_n_freq_bins))

    # Print STFT data flattened (frame by frame, comma-separated per frame)
    for frame_i in range(stft_n_frames):
        var frame_parts = List[String]()
        for bin_i in range(stft_n_freq_bins):
            frame_parts.append(String(stft_result[frame_i][bin_i]))
        var frame_line = String("")
        for idx in range(len(frame_parts)):
            if idx > 0:
                frame_line += ","
            frame_line += frame_parts[idx]
        print(frame_line)

    # 7. Per-frame RMS energy (same hop as STFT)
    var rms_n_frames_16k = (len(samples_16k) - n_fft) // hop_length + 1
    var rms_parts = List[String]()
    for ri in range(rms_n_frames_16k):
        var rms_start = ri * hop_length
        var rms_sum_sq: Float64 = 0.0
        for rj in range(n_fft):
            if rms_start + rj < len(samples_16k):
                var rv = Float64(samples_16k[rms_start + rj])
                rms_sum_sq += rv * rv
        var rms_val = sqrt(rms_sum_sq / Float64(n_fft))
        rms_parts.append(String(Float32(rms_val)))
    var rms_line = String("")
    for idx in range(len(rms_parts)):
        if idx > 0:
            rms_line += ","
        rms_line += rms_parts[idx]
    print("RMS_DATA:" + rms_line)

    # 8. Averaged power spectrum (average STFT frames)
    var avg_ps = List[Float64]()
    for bi in range(stft_n_freq_bins):
        avg_ps.append(0.0)
    for fi in range(stft_n_frames):
        for bi in range(stft_n_freq_bins):
            avg_ps[bi] = avg_ps[bi] + Float64(stft_result[fi][bi])
    var ps_parts = List[String]()
    for bi in range(stft_n_freq_bins):
        ps_parts.append(String(Float32(avg_ps[bi] / Float64(stft_n_frames))))
    var ps_line = String("")
    for idx in range(len(ps_parts)):
        if idx > 0:
            ps_line += ","
        ps_line += ps_parts[idx]
    print("POWER_SPECTRUM:" + ps_line)
"""

    # Write temp Mojo script
    with tempfile.NamedTemporaryFile(mode='w', suffix='.mojo', delete=False) as f:
        mojo_script_path = Path(f.name)
        f.write(mojo_code)

    try:
        result = subprocess.run(
            ['pixi', 'run', '--', 'mojo', 'run', '-I', 'src', str(mojo_script_path)],
            capture_output=True,
            text=True,
            cwd=REPO_ROOT,
            timeout=60,
        )

        if result.returncode != 0:
            raise RuntimeError(f"Mojo DSP failed: {result.stderr.strip()}")

        # Parse stdout
        lines = result.stdout.strip().split('\n')
        line_idx = 0

        # Line 0: duration_s, sample_rate, n_samples
        header = lines[line_idx].split(',')
        line_idx += 1
        duration_s = float(header[0])
        sample_rate = int(header[1])

        # Line 1: waveform RMS values
        waveform = [float(v) for v in lines[line_idx].split(',')]
        line_idx += 1

        # Line 2: n_mels, n_frames
        mel_shape = lines[line_idx].split(',')
        line_idx += 1
        n_mels = int(mel_shape[0])
        n_frames = int(mel_shape[1])

        # Next n_mels lines: mel spectrogram rows
        mel_np = np.zeros((n_mels, n_frames), dtype=np.float32)
        for m in range(n_mels):
            mel_np[m] = [float(v) for v in lines[line_idx].split(',')]
            line_idx += 1

        # Normalize to [0, 1] — Mojo mel_spectrogram already returns log-scale
        # values (mostly negative), so just min-max normalize directly
        np.nan_to_num(mel_np, copy=False, nan=0.0, posinf=0.0, neginf=0.0)
        denom = mel_np.max() - mel_np.min()
        if denom < 1e-8:
            mel_norm = np.zeros_like(mel_np)
        else:
            mel_norm = (mel_np - mel_np.min()) / denom

        # VAD regions
        n_segs = int(lines[line_idx])
        line_idx += 1
        vad_regions = []
        for _ in range(n_segs):
            parts = lines[line_idx].split(',')
            line_idx += 1
            vad_regions.append({
                "start": float(int(parts[0]) / sample_rate),
                "end": float(int(parts[1]) / sample_rate),
            })

        # STFT magnitude
        stft_shape = lines[line_idx].replace("STFT_SHAPE:", "").split(',')
        line_idx += 1
        stft_n_frames = int(stft_shape[0])
        stft_n_freq_bins = int(stft_shape[1])

        stft_np = np.zeros((stft_n_frames, stft_n_freq_bins), dtype=np.float32)
        for fi in range(stft_n_frames):
            stft_np[fi] = [float(v) for v in lines[line_idx].split(',')]
            line_idx += 1
        np.nan_to_num(stft_np, copy=False, nan=0.0, posinf=0.0, neginf=0.0)

        # RMS energy (per-frame)
        rms_line = ""
        power_spectrum_line = ""
        for remaining_line in lines[line_idx:]:
            if remaining_line.startswith("RMS_DATA:"):
                rms_line = remaining_line.replace("RMS_DATA:", "")
            elif remaining_line.startswith("POWER_SPECTRUM:"):
                power_spectrum_line = remaining_line.replace("POWER_SPECTRUM:", "")

        rms_energy = [float(x) for x in rms_line.split(',') if x]
        rms_np = np.array(rms_energy, dtype=np.float32)
        np.nan_to_num(rms_np, copy=False, nan=0.0, posinf=0.0, neginf=0.0)

        power_spectrum = [float(x) for x in power_spectrum_line.split(',') if x]
        ps_np = np.array(power_spectrum, dtype=np.float32)
        np.nan_to_num(ps_np, copy=False, nan=0.0, posinf=0.0, neginf=0.0)

        return {
            "duration_s": duration_s,
            "sample_rate": sample_rate,
            "waveform": waveform,
            "mel_spectrogram": mel_norm.flatten().tolist(),
            "mel_n_mels": n_mels,
            "mel_n_frames": n_frames,
            "vad_regions": vad_regions,
            "stft_magnitude": stft_np.flatten().tolist(),
            "stft_n_frames": stft_n_frames,
            "stft_n_freq_bins": stft_n_freq_bins,
            "rms_energy": rms_np.tolist(),
            "power_spectrum": ps_np.tolist(),
        }

    finally:
        mojo_script_path.unlink(missing_ok=True)


@app.get("/visualizer")
async def visualizer_page():
    """Serve the audio visualizer page."""
    return FileResponse(str(UI_ROOT / "frontend" / "visualizer.html"))


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
