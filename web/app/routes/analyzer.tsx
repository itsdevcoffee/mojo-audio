import { useState, useCallback } from "react";
import { useAnalyzer } from "../lib/use-analyzer";
import { GlassPanel } from "../components/ui/GlassPanel";
import { TabBar } from "../components/ui/TabBar";
import { Waveform } from "../components/viz/Waveform";
import { MelSpectrogram } from "../components/viz/MelSpectrogram";
import { Waterfall3D } from "../components/viz/Waterfall3D";
import "../styles/analyzer.css";

const TABS = [
  { id: "mel", label: "Mel Spectrogram" },
  { id: "stft", label: "Linear STFT" },
  { id: "chroma", label: "Chromagram" },
  { id: "waterfall", label: "3D Waterfall" },
  { id: "rms", label: "RMS Energy" },
  { id: "centroid", label: "Spectral Centroid" },
  { id: "freq", label: "Freq Dist" },
];

export default function Analyzer() {
  const { status, data, error, fileName, analyze, reset } = useAnalyzer();
  const [activeTab, setActiveTab] = useState("mel");

  const handleDrop = useCallback(
    (e: React.DragEvent) => {
      e.preventDefault();
      const file = e.dataTransfer.files[0];
      if (file && file.name.toLowerCase().endsWith(".wav")) {
        analyze(file);
      }
    },
    [analyze]
  );

  const handleFileInput = useCallback(
    (e: React.ChangeEvent<HTMLInputElement>) => {
      const file = e.target.files?.[0];
      if (file) analyze(file);
    },
    [analyze]
  );

  const renderVizPanel = () => {
    if (!data) return null;
    switch (activeTab) {
      case "mel":
        return (
          <MelSpectrogram
            melData={data.mel_spectrogram}
            nMels={data.mel_n_mels}
            nFrames={data.mel_n_frames}
          />
        );
      case "waterfall":
        return (
          <Waterfall3D
            melData={data.mel_spectrogram}
            nMels={data.mel_n_mels}
            nFrames={data.mel_n_frames}
          />
        );
      case "stft":
      case "chroma":
      case "rms":
      case "centroid":
      case "freq":
        return (
          <div className="viz-content" style={{ padding: "40px 14px", textAlign: "center" }}>
            <span className="panel-meta">Coming soon — backend extension needed</span>
          </div>
        );
      default:
        return null;
    }
  };

  return (
    <div className="analyzer">
      {/* Glow blobs */}
      <div className="analyzer__glow analyzer__glow--primary" />
      <div className="analyzer__glow analyzer__glow--secondary" />

      {/* Drop zone / file info */}
      {status === "idle" ? (
        <div
          className="drop-zone"
          onDragOver={(e) => e.preventDefault()}
          onDrop={handleDrop}
        >
          <p className="gradient-text" style={{ fontSize: 14, fontWeight: 600 }}>
            Drop a WAV file here
          </p>
          <p className="panel-meta" style={{ marginTop: 6 }}>
            or{" "}
            <label className="drop-zone__browse">
              browse
              <input
                type="file"
                accept=".wav"
                onChange={handleFileInput}
                style={{ display: "none" }}
              />
            </label>
          </p>
        </div>
      ) : (
        <div className="file-info">
          <div>
            <span className="gradient-text">{fileName}</span>
            {data && (
              <span className="panel-meta">
                {" "}· {data.sample_rate}Hz · {data.duration_s.toFixed(1)}s
              </span>
            )}
          </div>
          <button className="file-info__new" onClick={reset}>
            Drop new file
          </button>
        </div>
      )}

      {/* Loading state */}
      {status === "loading" && (
        <div style={{ textAlign: "center", padding: 40 }}>
          <span className="gradient-text">Analyzing...</span>
        </div>
      )}

      {/* Error state */}
      {status === "error" && (
        <div style={{ textAlign: "center", padding: 40, color: "#ff6060" }}>
          <p>Error: {error}</p>
          <button onClick={reset} style={{ marginTop: 12, color: "var(--text-muted)", background: "none", border: "1px solid var(--panel-border)", borderRadius: 4, padding: "4px 12px", cursor: "pointer", fontFamily: "var(--font-mono)" }}>
            Try again
          </button>
        </div>
      )}

      {/* Visualization panels */}
      {status === "ready" && data && (
        <>
          <Waveform
            waveform={data.waveform}
            vadRegions={data.vad_regions}
            duration={data.duration_s}
            sampleRate={data.sample_rate}
          />

          <TabBar tabs={TABS} activeTab={activeTab} onTabChange={setActiveTab} />

          <GlassPanel
            className="viz-panel"
            style={{ borderTop: "none", borderRadius: "0 0 8px 8px" }}
          >
            {renderVizPanel()}
          </GlassPanel>
        </>
      )}
    </div>
  );
}
