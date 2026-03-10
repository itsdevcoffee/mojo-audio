import type { ComparisonResult } from "../../lib/benchmark-api";

interface ResultCardProps {
  result: ComparisonResult;
}

export function ResultCard({ result }: ResultCardProps) {
  const { mojo, librosa, speedup_factor } = result;

  return (
    <div className="glass result-card">
      <div className="result-card__hero">
        <span className="gradient-text" style={{ fontSize: 48, fontWeight: 300 }}>
          {speedup_factor.toFixed(1)}x
        </span>
        <span className="panel-meta" style={{ marginTop: 4 }}>faster</span>
      </div>
      <div className="result-card__comparison">
        <div className="result-card__impl">
          <span className="panel-label">librosa ({librosa.implementation.split("(")[1]?.replace(")", "") || ""})</span>
          <span className="result-card__time">{librosa.avg_time_ms.toFixed(1)}ms</span>
          <div className="result-card__bar">
            <div className="result-card__bar-fill result-card__bar-fill--librosa" style={{ width: "100%" }} />
          </div>
        </div>
        <div className="result-card__impl">
          <span className="panel-label">mojo-audio</span>
          <span className="result-card__time gradient-text">{mojo.avg_time_ms.toFixed(1)}ms</span>
          <div className="result-card__bar">
            <div
              className="result-card__bar-fill result-card__bar-fill--mojo"
              style={{ width: `${(mojo.avg_time_ms / librosa.avg_time_ms) * 100}%` }}
            />
          </div>
        </div>
      </div>
    </div>
  );
}
