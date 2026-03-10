import { Link } from "@remix-run/react";

export default function Landing() {
  return (
    <div className="landing">
      {/* Glow blobs */}
      <div className="landing__glow landing__glow--primary" />
      <div className="landing__glow landing__glow--secondary" />

      <div className="landing__hero">
        <h1 className="gradient-text" style={{ fontSize: 36, fontWeight: 300, letterSpacing: "-0.5px" }}>
          mojo-audio
        </h1>
        <p style={{ color: "var(--text-muted)", fontSize: 13, marginTop: 8 }}>
          High-performance audio DSP in Mojo
        </p>
        <div className="landing__stat">
          <span className="gradient-text" style={{ fontWeight: 600 }}>20-40%</span>
          <span style={{ color: "var(--text)" }}> faster than librosa</span>
        </div>
        <div className="landing__ctas">
          <Link to="/analyzer" className="landing__cta landing__cta--primary">
            Try Analyzer
          </Link>
          <Link to="/docs" className="landing__cta landing__cta--secondary">
            View Docs
          </Link>
        </div>
      </div>
    </div>
  );
}
