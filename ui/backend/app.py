"""
FastAPI backend for mojo-audio benchmark UI.

Provides endpoints for running Mojo and librosa benchmarks.
"""

import subprocess
from pathlib import Path

from fastapi import FastAPI, HTTPException
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


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
