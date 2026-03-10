"""
FastAPI backend for mojo-audio benchmark UI.

Provides endpoints for running Mojo and librosa benchmarks.
"""

import subprocess
import sys
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

# Add src/ to path for mojo_audio modules
if str(REPO_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(REPO_ROOT / "src"))

ALLOWED_BLAS_BACKENDS = frozenset(["mkl", "openblas"])
BENCHMARK_TIMEOUT_SECONDS = 120

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
    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
        content = await file.read()
        tmp.write(content)
        tmp_path = tmp.name

    try:
        return _process_audio(tmp_path)
    finally:
        Path(tmp_path).unlink(missing_ok=True)


def _process_audio(wav_path: str) -> dict:
    """Process audio through mojo-audio DSP and return visualization data."""
    from wav_io import read_wav
    from resample import resample_to_16k
    from vad import get_voice_segments
    from audio import mel_spectrogram, hann_window

    # 1. Load audio
    samples, sample_rate = read_wav(wav_path)
    samples_np = np.array(samples, dtype=np.float32)
    duration_s = len(samples_np) / sample_rate

    # 2. Waveform — downsample to ~4000 RMS envelope points
    display_points = 4000
    hop = max(1, len(samples_np) // display_points)
    chunks = [samples_np[i:i+hop] for i in range(0, len(samples_np), hop)]
    waveform_display = [float(np.sqrt(np.mean(c**2))) if len(c) > 0 else 0.0
                        for c in chunks]

    # 3. Resample to 16kHz for mel spectrogram
    samples_16k_list = resample_to_16k(list(samples), sample_rate)
    samples_16k = np.array(samples_16k_list, dtype=np.float32)

    # 4. Mel spectrogram — 80 mel bands
    n_fft = 400
    hop_length = 160  # 10ms at 16kHz
    mel_spec = mel_spectrogram(
        list(samples_16k),
        n_mels=80,
        n_fft=n_fft,
        hop_length=hop_length,
        sample_rate=16000,
    )
    mel_np = np.array([[v for v in row] for row in mel_spec], dtype=np.float32)
    n_mels, n_frames = mel_np.shape

    # Normalize to [0, 1] log scale
    mel_log = np.log1p(mel_np)
    mel_norm = (mel_log - mel_log.min()) / (mel_log.max() - mel_log.min() + 1e-8)

    # 5. VAD regions
    voice_segs = get_voice_segments(list(samples), sample_rate)
    vad_regions = [
        {"start": float(seg[0] / sample_rate), "end": float(seg[1] / sample_rate)}
        for seg in voice_segs
    ]

    return {
        "duration_s": float(duration_s),
        "sample_rate": int(sample_rate),
        "waveform": waveform_display,
        "mel_spectrogram": mel_norm.flatten().tolist(),
        "mel_n_mels": int(n_mels),
        "mel_n_frames": int(n_frames),
        "vad_regions": vad_regions,
    }


@app.get("/visualizer")
async def visualizer_page():
    """Serve the audio visualizer page."""
    return FileResponse(str(UI_ROOT / "frontend" / "visualizer.html"))


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
