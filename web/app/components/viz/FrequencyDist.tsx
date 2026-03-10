import { useRef, useEffect } from "react";

interface FrequencyDistProps {
  powerSpectrum: number[];
  sampleRate: number;
}

export function FrequencyDist({ powerSpectrum, sampleRate }: FrequencyDistProps) {
  const canvasRef = useRef<HTMLCanvasElement>(null);

  useEffect(() => {
    const canvas = canvasRef.current;
    if (!canvas || powerSpectrum.length === 0) return;

    const dpr = Math.min(window.devicePixelRatio, 2);
    const rect = canvas.getBoundingClientRect();
    canvas.width = rect.width * dpr;
    canvas.height = rect.height * dpr;

    const ctx = canvas.getContext("2d");
    if (!ctx) return;
    ctx.scale(dpr, dpr);

    const W = rect.width;
    const H = rect.height;
    const padLeft = 36;
    const padBottom = 20;
    const plotW = W - padLeft - 8;
    const plotH = H - padBottom - 8;

    ctx.clearRect(0, 0, W, H);

    // Convert to dB
    const maxPS = Math.max(...powerSpectrum, 1e-10);
    const dbData = powerSpectrum.map((v) =>
      20 * Math.log10(Math.max(v, 1e-10) / maxPS)
    );
    const minDb = -80;

    // Y-axis labels
    ctx.font = "8px 'Fira Code', monospace";
    ctx.fillStyle = "#6aaa90";
    ctx.textAlign = "right";
    ctx.textBaseline = "middle";
    for (let db = 0; db >= minDb; db -= 20) {
      const y = 4 + ((0 - db) / (0 - minDb)) * plotH;
      ctx.fillText(`${db}`, padLeft - 4, y);
      ctx.strokeStyle = "rgba(80, 200, 150, 0.06)";
      ctx.lineWidth = 1;
      ctx.beginPath();
      ctx.moveTo(padLeft, y);
      ctx.lineTo(W - 8, y);
      ctx.stroke();
    }

    // X-axis labels (frequency)
    const maxFreq = sampleRate / 2;
    ctx.textAlign = "center";
    ctx.textBaseline = "top";
    const freqLabels = [0, 1000, 2000, 4000, 8000];
    for (const freq of freqLabels) {
      if (freq > maxFreq) break;
      const x = padLeft + (freq / maxFreq) * plotW;
      const label = freq >= 1000 ? `${freq / 1000}k` : `${freq}`;
      ctx.fillText(label, x, H - padBottom + 4);
    }

    // Bars
    const grad = ctx.createLinearGradient(padLeft, 4, padLeft + plotW, 4);
    grad.addColorStop(0, "rgba(80, 232, 168, 0.7)");
    grad.addColorStop(1, "rgba(56, 200, 180, 0.7)");
    ctx.fillStyle = grad;

    for (let i = 0; i < dbData.length; i++) {
      const normDb = Math.max(0, (dbData[i] - minDb) / (0 - minDb));
      const barH = normDb * plotH;
      const x = padLeft + (i / dbData.length) * plotW;
      const barW = plotW / powerSpectrum.length;
      const y = 4 + plotH - barH;
      ctx.fillRect(x, y, Math.max(barW, 1), barH);
    }
  }, [powerSpectrum, sampleRate]);

  return (
    <div className="viz-content">
      <div className="viz-content__header">
        <span className="panel-label">Frequency Distribution · dB</span>
        <span className="panel-meta">averaged power spectrum</span>
      </div>
      <canvas ref={canvasRef} className="viz-content__canvas" style={{ width: "100%", height: 280 }} />
    </div>
  );
}
