import { useState } from "react";
import { ConfigCard } from "../components/benchmark/ConfigCard";
import { ResultCard } from "../components/benchmark/ResultCard";
import {
  DEFAULT_CONFIG,
  runComparison,
  type BenchmarkConfig,
  type ComparisonResult,
} from "../lib/benchmark-api";
import "../styles/benchmarks.css";

export default function Benchmarks() {
  const [config, setConfig] = useState<BenchmarkConfig>(DEFAULT_CONFIG);
  const [result, setResult] = useState<ComparisonResult | null>(null);
  const [isRunning, setIsRunning] = useState(false);
  const [error, setError] = useState<string | null>(null);

  async function handleRun() {
    setIsRunning(true);
    setError(null);
    try {
      const res = await runComparison(config);
      setResult(res);
    } catch (e) {
      setError(e instanceof Error ? e.message : String(e));
    } finally {
      setIsRunning(false);
    }
  }

  return (
    <div className="benchmarks">
      <div className="benchmarks__header">
        <h1 className="gradient-text" style={{ fontSize: 20, fontWeight: 300 }}>
          Benchmarks
        </h1>
        <p className="panel-meta" style={{ marginTop: 4 }}>
          Compare mojo-audio mel spectrogram against librosa
        </p>
      </div>

      <ConfigCard config={config} onChange={setConfig} onRun={handleRun} isRunning={isRunning} />

      {error && (
        <div style={{ padding: 16, color: "#ff6060", textAlign: "center" }}>
          Error: {error}
        </div>
      )}

      {result && <ResultCard result={result} />}
    </div>
  );
}
