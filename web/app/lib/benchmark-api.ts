export interface BenchmarkConfig {
  duration: number;
  n_fft: number;
  hop_length: number;
  n_mels: number;
  iterations: number;
  blas_backend: "mkl" | "openblas";
}

export interface BenchmarkResult {
  implementation: string;
  duration: number;
  avg_time_ms: number;
  std_time_ms: number;
  throughput_realtime: number;
  iterations: number;
  success: boolean;
  error: string;
}

export interface ComparisonResult {
  mojo: BenchmarkResult;
  librosa: BenchmarkResult;
  speedup_factor: number;
  faster_percentage: number;
  mojo_is_faster: boolean;
}

export const DEFAULT_CONFIG: BenchmarkConfig = {
  duration: 30,
  n_fft: 400,
  hop_length: 160,
  n_mels: 80,
  iterations: 20,
  blas_backend: "mkl",
};

async function runSingle(endpoint: string, config: BenchmarkConfig): Promise<BenchmarkResult> {
  const res = await fetch(endpoint, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(config),
  });
  if (!res.ok) {
    const err = await res.json().catch(() => ({ detail: res.statusText }));
    throw new Error(err.detail || `HTTP ${res.status}`);
  }
  return res.json();
}

export type BenchmarkPhase = "idle" | "mojo" | "librosa" | "done";

export async function runComparison(
  config: BenchmarkConfig,
  onPhase: (phase: BenchmarkPhase) => void,
): Promise<ComparisonResult> {
  onPhase("mojo");
  const mojo = await runSingle("/api/benchmark/mojo", config);

  onPhase("librosa");
  const librosa = await runSingle("/api/benchmark/librosa", config);

  const speedup_factor = librosa.avg_time_ms / mojo.avg_time_ms;
  const faster_percentage =
    ((librosa.avg_time_ms - mojo.avg_time_ms) / librosa.avg_time_ms) * 100;

  onPhase("done");
  return {
    mojo,
    librosa,
    speedup_factor: Math.round(speedup_factor * 100) / 100,
    faster_percentage: Math.round(faster_percentage * 10) / 10,
    mojo_is_faster: mojo.avg_time_ms < librosa.avg_time_ms,
  };
}
