"""
FastAPI backend for mojo-audio benchmark UI.

Provides endpoints for running Mojo and librosa benchmarks with
configurable signal types, warmup, and stable mode.
"""

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from pydantic import BaseModel
from typing import Optional, List
import subprocess
import json
import time
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

# Valid signal types
SIGNAL_TYPES = ["random", "chirp", "sine", "white_noise", "multi_tone"]


class BenchmarkConfig(BaseModel):
    """Benchmark configuration."""
    duration: int = 30  # seconds
    n_fft: int = 400
    hop_length: int = 160
    n_mels: int = 80
    iterations: int = 20
    warmup_runs: int = 5
    signal_type: str = "chirp"
    seed: int = 42
    stable_mode: bool = False
    stable_runs: int = 5


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


class StableResult(BaseModel):
    """Stable benchmark result with median."""
    implementation: str
    duration: int
    median_time_ms: float
    stdev_time_ms: float
    all_runs_ms: List[float]
    throughput_realtime: float
    num_runs: int
    success: bool


@app.get("/")
async def root():
    """Serve the main UI."""
    return FileResponse(str(UI_ROOT / "frontend" / "index.html"))


@app.get("/api/signal-types")
async def get_signal_types():
    """Return available signal types with descriptions."""
    return {
        "types": [
            {
                "id": "chirp",
                "name": "Chirp",
                "description": "Linear frequency sweep 20Hz-8000Hz. Deterministic, exercises full spectrum.",
                "icon": "📈",
                "recommended": True
            },
            {
                "id": "random",
                "name": "Random (Seeded)",
                "description": "Reproducible random noise with configurable seed.",
                "icon": "🎲",
                "has_seed": True
            },
            {
                "id": "sine",
                "name": "Sine 440Hz",
                "description": "Pure tone at A4. Minimal spectral content, predictable.",
                "icon": "〰️"
            },
            {
                "id": "white_noise",
                "name": "White Noise",
                "description": "Random each run. Realistic but non-reproducible.",
                "icon": "📻"
            },
            {
                "id": "multi_tone",
                "name": "Multi-Tone",
                "description": "Speech-like formant frequencies (F0, F1, F2, F3).",
                "icon": "🗣️"
            }
        ]
    }


def run_benchmark_subprocess(implementation: str, config: BenchmarkConfig) -> tuple:
    """Run benchmark via subprocess and return (avg, std)."""
    cmd = [
        "python", "ui/backend/run_benchmark.py",
        implementation,
        str(config.duration),
        str(config.iterations),
        str(config.n_fft),
        str(config.hop_length),
        str(config.n_mels),
        str(config.warmup_runs),
        config.signal_type,
        str(config.seed)
    ]

    result = subprocess.run(
        cmd,
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
        timeout=180  # Increased for stable mode
    )

    if result.returncode != 0:
        raise Exception(f"Benchmark failed: {result.stderr}")

    output = result.stdout.strip()
    parts = output.split(',')
    avg_time = float(parts[0])
    std_time = float(parts[1]) if len(parts) > 1 else 0.0

    return avg_time, std_time


@app.post("/api/benchmark/mojo")
async def benchmark_mojo(config: BenchmarkConfig) -> BenchmarkResult:
    """Run mojo-audio benchmark."""
    try:
        if config.signal_type not in SIGNAL_TYPES:
            raise HTTPException(status_code=400, detail=f"Invalid signal type: {config.signal_type}")

        avg_time, std_time = run_benchmark_subprocess("mojo", config)
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


@app.post("/api/benchmark/librosa")
async def benchmark_librosa(config: BenchmarkConfig) -> BenchmarkResult:
    """Run librosa benchmark."""
    try:
        if config.signal_type not in SIGNAL_TYPES:
            raise HTTPException(status_code=400, detail=f"Invalid signal type: {config.signal_type}")

        avg_time, std_time = run_benchmark_subprocess("librosa", config)
        throughput = config.duration / (avg_time / 1000.0)

        return BenchmarkResult(
            implementation="librosa",
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
    """Run both benchmarks and return comparison."""
    try:
        if config.stable_mode:
            return await benchmark_stable(config)

        mojo_result = await benchmark_mojo(config)
        librosa_result = await benchmark_librosa(config)

        speedup = librosa_result.avg_time_ms / mojo_result.avg_time_ms
        faster_pct = ((librosa_result.avg_time_ms - mojo_result.avg_time_ms) / librosa_result.avg_time_ms) * 100

        return {
            "mojo": mojo_result,
            "librosa": librosa_result,
            "speedup_factor": round(speedup, 2),
            "faster_percentage": round(faster_pct, 1),
            "mojo_is_faster": mojo_result.avg_time_ms < librosa_result.avg_time_ms,
            "stable_mode": False
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/benchmark/stable")
async def benchmark_stable(config: BenchmarkConfig):
    """
    Run stable benchmark: multiple complete runs, report median.

    This reduces system noise by running the full benchmark multiple times
    and reporting the median result.
    """
    try:
        import statistics

        if config.signal_type not in SIGNAL_TYPES:
            raise HTTPException(status_code=400, detail=f"Invalid signal type: {config.signal_type}")

        mojo_runs = []
        librosa_runs = []

        for run_num in range(config.stable_runs):
            # Run mojo
            try:
                avg, _ = run_benchmark_subprocess("mojo", config)
                mojo_runs.append(avg)
            except:
                pass

            # Run librosa
            try:
                avg, _ = run_benchmark_subprocess("librosa", config)
                librosa_runs.append(avg)
            except:
                pass

        if not mojo_runs or not librosa_runs:
            raise HTTPException(status_code=500, detail="Benchmarks failed to produce results")

        mojo_median = statistics.median(mojo_runs)
        mojo_stdev = statistics.stdev(mojo_runs) if len(mojo_runs) > 1 else 0
        librosa_median = statistics.median(librosa_runs)
        librosa_stdev = statistics.stdev(librosa_runs) if len(librosa_runs) > 1 else 0

        mojo_throughput = config.duration / (mojo_median / 1000.0)
        librosa_throughput = config.duration / (librosa_median / 1000.0)

        speedup = librosa_median / mojo_median
        faster_pct = ((librosa_median - mojo_median) / librosa_median) * 100

        return {
            "mojo": {
                "implementation": "mojo-audio",
                "duration": config.duration,
                "avg_time_ms": mojo_median,  # Using median as "avg" for compatibility
                "std_time_ms": mojo_stdev,
                "throughput_realtime": mojo_throughput,
                "iterations": config.iterations,
                "success": True,
                "all_runs_ms": mojo_runs
            },
            "librosa": {
                "implementation": "librosa",
                "duration": config.duration,
                "avg_time_ms": librosa_median,
                "std_time_ms": librosa_stdev,
                "throughput_realtime": librosa_throughput,
                "iterations": config.iterations,
                "success": True,
                "all_runs_ms": librosa_runs
            },
            "speedup_factor": round(speedup, 2),
            "faster_percentage": round(faster_pct, 1),
            "mojo_is_faster": mojo_median < librosa_median,
            "stable_mode": True,
            "num_runs": config.stable_runs
        }

    except subprocess.TimeoutExpired:
        raise HTTPException(status_code=408, detail="Stable benchmark timeout")
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/health")
async def health():
    """Health check endpoint."""
    return {"status": "healthy", "service": "mojo-audio benchmark API"}


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
