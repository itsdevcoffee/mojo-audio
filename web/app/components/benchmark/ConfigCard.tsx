import type { BenchmarkConfig } from "../../lib/benchmark-api";

interface ConfigCardProps {
  config: BenchmarkConfig;
  onChange: (config: BenchmarkConfig) => void;
  onRun: () => void;
  isRunning: boolean;
}

export function ConfigCard({ config, onChange, onRun, isRunning }: ConfigCardProps) {
  return (
    <div className="glass config-card">
      <div className="config-card__header">
        <span className="panel-label">Configuration</span>
      </div>
      <div className="config-card__body">
        <div className="config-card__row">
          <label className="config-card__label">Duration</label>
          <div className="config-card__options">
            {[1, 10, 30].map((d) => (
              <button
                key={d}
                className={`config-card__option ${config.duration === d ? "config-card__option--active" : ""}`}
                onClick={() => onChange({ ...config, duration: d })}
              >
                {d}s
              </button>
            ))}
          </div>
        </div>
        <div className="config-card__row">
          <label className="config-card__label">FFT Size</label>
          <div className="config-card__options">
            {[256, 400, 512, 1024].map((n) => (
              <button
                key={n}
                className={`config-card__option ${config.n_fft === n ? "config-card__option--active" : ""}`}
                onClick={() => onChange({ ...config, n_fft: n, hop_length: Math.floor(n / 2.5) })}
              >
                {n}
              </button>
            ))}
          </div>
        </div>
        <div className="config-card__row">
          <label className="config-card__label">Iterations</label>
          <div className="config-card__options">
            <button
              className="config-card__option"
              onClick={() => onChange({ ...config, iterations: Math.max(1, config.iterations - 5) })}
            >
              -
            </button>
            <span style={{ minWidth: 32, textAlign: "center" }}>{config.iterations}</span>
            <button
              className="config-card__option"
              onClick={() => onChange({ ...config, iterations: Math.min(100, config.iterations + 5) })}
            >
              +
            </button>
          </div>
        </div>
        <div className="config-card__row">
          <label className="config-card__label">BLAS</label>
          <div className="config-card__options">
            {(["mkl", "openblas"] as const).map((b) => (
              <button
                key={b}
                className={`config-card__option ${config.blas_backend === b ? "config-card__option--active" : ""}`}
                onClick={() => onChange({ ...config, blas_backend: b })}
              >
                {b.toUpperCase()}
              </button>
            ))}
          </div>
        </div>
        <button
          className="config-card__run"
          onClick={onRun}
          disabled={isRunning}
        >
          {isRunning ? "Running..." : "Run Benchmark"}
        </button>
      </div>
    </div>
  );
}
