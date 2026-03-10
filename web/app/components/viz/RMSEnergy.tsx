import { useRef, useEffect } from "react";

interface RMSEnergyProps {
  rmsData: number[];
  duration: number;
}

export function RMSEnergy({ rmsData, duration }: RMSEnergyProps) {
  const canvasRef = useRef<HTMLCanvasElement>(null);

  useEffect(() => {
    const canvas = canvasRef.current;
    if (!canvas || rmsData.length === 0) return;

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
    const maxRMS = Math.max(...rmsData, 1e-10);
    const dbData = rmsData.map((v) => 20 * Math.log10(Math.max(v, 1e-10) / maxRMS));
    const minDb = -60;

    // Y-axis labels (dB)
    ctx.font = "8px 'Fira Code', monospace";
    ctx.fillStyle = "#6aaa90";
    ctx.textAlign = "right";
    ctx.textBaseline = "middle";
    for (let db = 0; db >= minDb; db -= 20) {
      const y = 4 + ((0 - db) / (0 - minDb)) * plotH;
      ctx.fillText(`${db}dB`, padLeft - 4, y);
      ctx.strokeStyle = "rgba(80, 200, 150, 0.06)";
      ctx.lineWidth = 1;
      ctx.beginPath();
      ctx.moveTo(padLeft, y);
      ctx.lineTo(W - 8, y);
      ctx.stroke();
    }

    // X-axis labels (time)
    ctx.textAlign = "center";
    ctx.textBaseline = "top";
    const nTimeLabels = 5;
    for (let i = 0; i <= nTimeLabels; i++) {
      const t = (i / nTimeLabels) * duration;
      const x = padLeft + (i / nTimeLabels) * plotW;
      ctx.fillText(`${t.toFixed(1)}s`, x, H - padBottom + 4);
    }

    // RMS curve — emerald gradient fill
    const grad = ctx.createLinearGradient(padLeft, 4, padLeft, 4 + plotH);
    grad.addColorStop(0, "rgba(80, 232, 168, 0.6)");
    grad.addColorStop(1, "rgba(56, 200, 180, 0.05)");

    ctx.beginPath();
    ctx.moveTo(padLeft, 4 + plotH); // bottom-left
    for (let i = 0; i < dbData.length; i++) {
      const x = padLeft + (i / dbData.length) * plotW;
      const normDb = Math.max(0, (dbData[i] - minDb) / (0 - minDb));
      const y = 4 + (1 - normDb) * plotH;
      ctx.lineTo(x, y);
    }
    ctx.lineTo(padLeft + plotW, 4 + plotH); // bottom-right
    ctx.closePath();
    ctx.fillStyle = grad;
    ctx.fill();

    // RMS line on top
    ctx.beginPath();
    for (let i = 0; i < dbData.length; i++) {
      const x = padLeft + (i / dbData.length) * plotW;
      const normDb = Math.max(0, (dbData[i] - minDb) / (0 - minDb));
      const y = 4 + (1 - normDb) * plotH;
      if (i === 0) ctx.moveTo(x, y);
      else ctx.lineTo(x, y);
    }
    ctx.strokeStyle = "#50e8a8";
    ctx.lineWidth = 1.5;
    ctx.stroke();
  }, [rmsData, duration]);

  return (
    <div className="viz-content">
      <div className="viz-content__header">
        <span className="panel-label">RMS Energy · dB scale</span>
        <span className="panel-meta">{rmsData.length} frames</span>
      </div>
      <canvas ref={canvasRef} className="viz-content__canvas" style={{ width: "100%", height: 280 }} />
    </div>
  );
}
