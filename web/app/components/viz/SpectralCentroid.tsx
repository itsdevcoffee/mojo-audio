import { useRef, useEffect, useMemo } from "react";

interface SpectralCentroidProps {
  stftData: number[];
  nFrames: number;
  nFreqBins: number;
  sampleRate: number;
  duration: number;
}

function computeCentroid(
  stftData: number[],
  nFrames: number,
  nFreqBins: number,
  sampleRate: number
): Float32Array {
  const centroids = new Float32Array(nFrames);
  const freqPerBin = sampleRate / ((nFreqBins - 1) * 2);

  for (let f = 0; f < nFrames; f++) {
    let weightedSum = 0;
    let totalEnergy = 0;
    for (let bin = 0; bin < nFreqBins; bin++) {
      const mag = stftData[f * nFreqBins + bin];
      weightedSum += mag * bin * freqPerBin;
      totalEnergy += mag;
    }
    centroids[f] = totalEnergy > 0 ? weightedSum / totalEnergy : 0;
  }
  return centroids;
}

export function SpectralCentroid({ stftData, nFrames, nFreqBins, sampleRate, duration }: SpectralCentroidProps) {
  const canvasRef = useRef<HTMLCanvasElement>(null);

  const centroids = useMemo(
    () => computeCentroid(stftData, nFrames, nFreqBins, sampleRate),
    [stftData, nFrames, nFreqBins, sampleRate]
  );

  useEffect(() => {
    const canvas = canvasRef.current;
    if (!canvas || centroids.length === 0) return;

    const dpr = Math.min(window.devicePixelRatio, 2);
    const rect = canvas.getBoundingClientRect();
    canvas.width = rect.width * dpr;
    canvas.height = rect.height * dpr;

    const ctx = canvas.getContext("2d");
    if (!ctx) return;
    ctx.scale(dpr, dpr);

    const W = rect.width;
    const H = rect.height;
    const padLeft = 40;
    const padBottom = 20;
    const plotW = W - padLeft - 8;
    const plotH = H - padBottom - 8;

    ctx.clearRect(0, 0, W, H);

    const maxFreq = Math.max(...centroids, 100);
    const minFreq = 0;

    // Y-axis labels (Hz)
    ctx.font = "8px 'Fira Code', monospace";
    ctx.fillStyle = "#6aaa90";
    ctx.textAlign = "right";
    ctx.textBaseline = "middle";
    const nYLabels = 5;
    for (let i = 0; i <= nYLabels; i++) {
      const freq = (i / nYLabels) * maxFreq;
      const y = 4 + (1 - i / nYLabels) * plotH;
      const label = freq >= 1000 ? `${(freq / 1000).toFixed(1)}k` : `${Math.round(freq)}`;
      ctx.fillText(label, padLeft - 4, y);
      ctx.strokeStyle = "rgba(80, 200, 150, 0.06)";
      ctx.lineWidth = 1;
      ctx.beginPath();
      ctx.moveTo(padLeft, y);
      ctx.lineTo(W - 8, y);
      ctx.stroke();
    }

    // X-axis labels
    ctx.textAlign = "center";
    ctx.textBaseline = "top";
    const nTimeLabels = 5;
    for (let i = 0; i <= nTimeLabels; i++) {
      const t = (i / nTimeLabels) * duration;
      const x = padLeft + (i / nTimeLabels) * plotW;
      ctx.fillText(`${t.toFixed(1)}s`, x, H - padBottom + 4);
    }

    // Centroid line — gradient stroke
    ctx.beginPath();
    for (let i = 0; i < centroids.length; i++) {
      const x = padLeft + (i / centroids.length) * plotW;
      const y = 4 + (1 - (centroids[i] - minFreq) / (maxFreq - minFreq)) * plotH;
      if (i === 0) ctx.moveTo(x, y);
      else ctx.lineTo(x, y);
    }
    ctx.strokeStyle = "#50e8a8";
    ctx.lineWidth = 1.5;
    ctx.stroke();

    // Fill below line
    ctx.lineTo(padLeft + plotW, 4 + plotH);
    ctx.lineTo(padLeft, 4 + plotH);
    ctx.closePath();
    const grad = ctx.createLinearGradient(0, 4, 0, 4 + plotH);
    grad.addColorStop(0, "rgba(80, 232, 168, 0.15)");
    grad.addColorStop(1, "rgba(56, 200, 180, 0.02)");
    ctx.fillStyle = grad;
    ctx.fill();
  }, [centroids, duration]);

  return (
    <div className="viz-content">
      <div className="viz-content__header">
        <span className="panel-label">Spectral Centroid · Hz</span>
        <span className="panel-meta">"brightness" over time</span>
      </div>
      <canvas ref={canvasRef} className="viz-content__canvas" style={{ width: "100%", height: 280 }} />
    </div>
  );
}
