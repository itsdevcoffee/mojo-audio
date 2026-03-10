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

export async function runComparison(config: BenchmarkConfig): Promise<ComparisonResult> {
  const res = await fetch("/api/benchmark/both", {
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
