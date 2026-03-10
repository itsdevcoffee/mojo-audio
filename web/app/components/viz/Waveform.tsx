import { useRef, useEffect } from "react";
import type { VadRegion } from "../../lib/types";

interface WaveformProps {
  waveform: number[];
  vadRegions: VadRegion[];
  duration: number;
  sampleRate: number;
}

export function Waveform({ waveform, vadRegions, duration, sampleRate }: WaveformProps) {
  const canvasRef = useRef<HTMLCanvasElement>(null);

  useEffect(() => {
    const canvas = canvasRef.current;
    if (!canvas || waveform.length === 0) return;

    const ctx = canvas.getContext("2d");
    if (!ctx) return;

    const dpr = Math.min(window.devicePixelRatio, 2);
    const rect = canvas.getBoundingClientRect();
    canvas.width = rect.width * dpr;
    canvas.height = rect.height * dpr;
    ctx.scale(dpr, dpr);

    const W = rect.width;
    const H = rect.height;
    const midY = H / 2;

    ctx.clearRect(0, 0, W, H);

    // VAD regions
    for (const region of vadRegions) {
      const x0 = (region.start / duration) * W;
      const x1 = (region.end / duration) * W;
      ctx.fillStyle = "rgba(0, 220, 120, 0.04)";
      ctx.fillRect(x0, 0, x1 - x0, H);
      ctx.strokeStyle = "rgba(0, 220, 120, 0.12)";
      ctx.lineWidth = 1;
      ctx.beginPath();
      ctx.moveTo(x0, 0);
      ctx.lineTo(x1, 0);
      ctx.stroke();
    }

    // Zero line
    ctx.strokeStyle = "rgba(80, 200, 150, 0.08)";
    ctx.lineWidth = 1;
    ctx.beginPath();
    ctx.moveTo(0, midY);
    ctx.lineTo(W, midY);
    ctx.stroke();

    // Waveform — emerald-to-teal gradient
    const grad = ctx.createLinearGradient(0, 0, W, 0);
    grad.addColorStop(0, "rgba(80, 232, 168, 0.7)");
    grad.addColorStop(1, "rgba(56, 200, 180, 0.7)");

    const maxVal = Math.max(...waveform.map(Math.abs), 0.001);

    ctx.fillStyle = grad;
    ctx.beginPath();
    ctx.moveTo(0, midY);

    for (let i = 0; i < waveform.length; i++) {
      const x = (i / waveform.length) * W;
      const y = midY - (waveform[i] / maxVal) * midY * 0.9;
      if (i === 0) ctx.moveTo(x, y);
      else ctx.lineTo(x, y);
    }

    for (let i = waveform.length - 1; i >= 0; i--) {
      const x = (i / waveform.length) * W;
      const y = midY + (waveform[i] / maxVal) * midY * 0.9;
      ctx.lineTo(x, y);
    }

    ctx.closePath();
    ctx.fill();
  }, [waveform, vadRegions, duration, sampleRate]);

  return (
    <div className="glass waveform-panel">
      <div className="waveform-panel__header">
        <span className="panel-label">Waveform</span>
        <div className="waveform-panel__legend">
          <span className="waveform-panel__vad-dot" />
          <span className="panel-meta">Voice</span>
          <span className="panel-meta" style={{ marginLeft: 8 }}>
            0:00 — {Math.floor(duration / 60)}:{String(Math.floor(duration % 60)).padStart(2, "0")}
          </span>
        </div>
      </div>
      <canvas
        ref={canvasRef}
        className="waveform-panel__canvas"
        style={{ width: "100%", height: 80 }}
      />
    </div>
  );
}
