"""
FastAPI backend for mojo-audio benchmark UI.

Provides endpoints for running Mojo and librosa benchmarks.
"""

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from pydantic import BaseModel
import subprocess
from pathlib import Path

app = FastAPI(title="mojo-audio Benchmark API")

# CORS for local development
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

# Serve static files (relative to ui directory)
app.mount("/static", StaticFiles(directory=str(UI_ROOT / "static")), name="static")


class BenchmarkConfig(BaseModel):
    """Benchmark configuration."""
    duration: int = 30  # seconds
    n_fft: int = 400
    hop_length: int = 160
    n_mels: int = 80
    iterations: int = 20  # Increased to 20 for excellent statistical confidence
    # BLAS backend for librosa/scipy - does NOT affect mojo-audio (pure Mojo FFT)
    blas_backend: str = "mkl"  # "mkl" or "openblas"


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
    """Parse benchmark output in avg,std format. Returns (avg_time, std_time)."""
    parts = stdout.strip().split(',')
    avg_time = float(parts[0])
    std_time = float(parts[1]) if len(parts) > 1 else 0.0
    return avg_time, std_time


@app.post("/api/benchmark/mojo")
async def benchmark_mojo(config: BenchmarkConfig) -> BenchmarkResult:
    """
    Run mojo-audio benchmark.

    Returns performance metrics for the optimized Mojo implementation.
    Note: mojo-audio uses pure Mojo FFT - BLAS backend setting is ignored.
    """
    try:
        # Use simple wrapper script with ALL user parameters
        cmd = [
            "python", "ui/backend/run_benchmark.py",
            "mojo",
            str(config.duration),
            str(config.iterations),
            str(config.n_fft),
            str(config.hop_length),
            str(config.n_mels)
        ]

        result = subprocess.run(
            cmd,
            cwd=REPO_ROOT,
            capture_output=True,
            text=True,
            timeout=120
        )

        if result.returncode != 0:
            raise HTTPException(
                status_code=500,
                detail=f"Mojo benchmark failed: {result.stderr}"
            )

        avg_time, std_time = parse_benchmark_output(result.stdout)
        throughput = config.duration / (avg_time / 1000.0)

        return BenchmarkResult(
            implementation="mojo-audio",
            duration=config.duration,
            avg_time_ms=avg_time,
            std_time_ms=std_time,
            throughput_realtime=throughput,
            iterations=config.iterations,
            success=True
        )

    except subprocess.TimeoutExpired:
        raise HTTPException(status_code=408, detail="Benchmark timeout")
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


ALLOWED_BLAS_BACKENDS = frozenset(["mkl", "openblas"])

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

        result = subprocess.run(
            cmd,
            cwd=REPO_ROOT,
            capture_output=True,
            text=True,
            timeout=120
        )

        if result.returncode != 0:
            raise HTTPException(
                status_code=500,
                detail=f"librosa benchmark failed: {result.stderr}"
            )

        avg_time, std_time = parse_benchmark_output(result.stdout)
        throughput = config.duration / (avg_time / 1000.0)

        return BenchmarkResult(
            implementation=f"librosa ({config.blas_backend.upper()})",
            duration=config.duration,
            avg_time_ms=avg_time,
            std_time_ms=std_time,
            throughput_realtime=throughput,
            iterations=config.iterations,
            success=True
        )

    except subprocess.TimeoutExpired:
        raise HTTPException(status_code=408, detail="Benchmark timeout")
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


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
