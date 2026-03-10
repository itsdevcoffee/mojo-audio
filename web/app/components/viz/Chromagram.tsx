import { useRef, useEffect, useMemo } from "react";
import { infernoRGB } from "./shared/inferno";

interface ChromagramProps {
  stftData: number[];
  nFrames: number;
  nFreqBins: number;
  sampleRate: number;
}

const PITCH_CLASSES = ["C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B"];

function computeChromagram(
  stftData: number[],
  nFrames: number,
  nFreqBins: number,
  sampleRate: number
): Float32Array {
  const chroma = new Float32Array(nFrames * 12);
  const freqPerBin = sampleRate / ((nFreqBins - 1) * 2);

  for (let f = 0; f < nFrames; f++) {
    for (let bin = 1; bin < nFreqBins; bin++) {
      const freq = bin * freqPerBin;
      if (freq < 20) continue; // skip sub-bass
      // Map frequency to pitch class
      const midiNote = 12 * Math.log2(freq / 440) + 69;
      const pitchClass = Math.round(midiNote) % 12;
      const pc = ((pitchClass % 12) + 12) % 12;
      chroma[f * 12 + pc] += stftData[f * nFreqBins + bin];
    }
  }
  return chroma;
}

export function Chromagram({ stftData, nFrames, nFreqBins, sampleRate }: ChromagramProps) {
  const canvasRef = useRef<HTMLCanvasElement>(null);

  const chroma = useMemo(
    () => computeChromagram(stftData, nFrames, nFreqBins, sampleRate),
    [stftData, nFrames, nFreqBins, sampleRate]
  );

  useEffect(() => {
    const canvas = canvasRef.current;
    if (!canvas || chroma.length === 0) return;

    const dpr = Math.min(window.devicePixelRatio, 2);
    const rect = canvas.getBoundingClientRect();
    canvas.width = rect.width * dpr;
    canvas.height = rect.height * dpr;

    const ctx = canvas.getContext("2d");
    if (!ctx) return;

    ctx.scale(dpr, dpr);
    const W = rect.width;
    const H = rect.height;

    // Normalize
    let max = 0;
    for (let i = 0; i < chroma.length; i++) {
      if (chroma[i] > max) max = chroma[i];
    }
    if (max === 0) max = 1;

    const cellH = (H - 16) / 12; // leave room for labels
    const labelW = 24;

    ctx.clearRect(0, 0, W, H);

    // Draw pitch class labels
    ctx.font = "8px 'Fira Code', monospace";
    ctx.textAlign = "right";
    ctx.textBaseline = "middle";
    for (let pc = 0; pc < 12; pc++) {
      const y = (11 - pc) * cellH + cellH / 2;
      ctx.fillStyle = "#6aaa90";
      ctx.fillText(PITCH_CLASSES[pc], labelW - 4, y);
    }

    // Draw grid
    const gridW = (W - labelW) / nFrames;
    for (let f = 0; f < nFrames; f++) {
      for (let pc = 0; pc < 12; pc++) {
        const val = chroma[f * 12 + pc] / max;
        const [r, g, b] = infernoRGB(val);
        ctx.fillStyle = `rgb(${r},${g},${b})`;
        ctx.fillRect(
          labelW + f * gridW,
          (11 - pc) * cellH,
          gridW + 0.5,
          cellH + 0.5
        );
      }
    }
  }, [chroma, nFrames]);

  return (
    <div className="viz-content">
      <div className="viz-content__header">
        <span className="panel-label">Chromagram · 12 pitch classes</span>
        <span className="panel-meta">C — B, folded from STFT</span>
      </div>
      <canvas ref={canvasRef} className="viz-content__canvas" style={{ width: "100%", height: 280 }} />
    </div>
  );
}
