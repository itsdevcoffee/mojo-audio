import { useRef, useEffect } from "react";
import { createProgram, createFullscreenQuad } from "./shared/webgl-utils";
import { infernoRGB } from "./shared/inferno";

interface MelSpectrogramProps {
  melData: number[];
  nMels: number;
  nFrames: number;
}

const VERT_SRC = `
  attribute vec2 a_pos;
  varying vec2 v_uv;
  void main() {
    v_uv = a_pos * 0.5 + 0.5;
    gl_Position = vec4(a_pos, 0.0, 1.0);
  }
`;

const FRAG_SRC = `
  precision mediump float;
  varying vec2 v_uv;
  uniform sampler2D u_tex;

  vec3 inferno(float t) {
    float r = clamp(-0.0155 + t*(5.3711 + t*(-14.099 + t*(13.457 - t*4.716))), 0.0, 1.0);
    float g = clamp(0.0109 + t*(-0.670 + t*(3.448 + t*(-5.691 + t*3.889))), 0.0, 1.0);
    float b = clamp(0.178 + t*(3.298 + t*(-12.425 + t*(17.326 - t*8.376))), 0.0, 1.0);
    return vec3(r, g, b);
  }

  void main() {
    float val = texture2D(u_tex, vec2(v_uv.x, 1.0 - v_uv.y)).r / 255.0;
    gl_FragColor = vec4(inferno(val), 1.0);
  }
`;

export function MelSpectrogram({ melData, nMels, nFrames }: MelSpectrogramProps) {
  const canvasRef = useRef<HTMLCanvasElement>(null);

  useEffect(() => {
    const canvas = canvasRef.current;
    if (!canvas || melData.length === 0) return;

    const dpr = Math.min(window.devicePixelRatio, 2);
    const rect = canvas.getBoundingClientRect();
    canvas.width = rect.width * dpr;
    canvas.height = rect.height * dpr;

    // Normalize mel data to [0, 255] Uint8
    let min = Infinity, max = -Infinity;
    for (const v of melData) {
      if (v < min) min = v;
      if (v > max) max = v;
    }
    const range = max - min || 1;
    const mel8 = new Uint8Array(melData.length);
    for (let i = 0; i < melData.length; i++) {
      mel8[i] = Math.round(((melData[i] - min) / range) * 255);
    }

    // Try WebGL
    const gl = canvas.getContext("webgl");
    if (gl) {
      const prog = createProgram(gl, VERT_SRC, FRAG_SRC);
      if (prog) {
        gl.useProgram(prog);
        gl.viewport(0, 0, canvas.width, canvas.height);

        const quad = createFullscreenQuad(gl);
        const aPos = gl.getAttribLocation(prog, "a_pos");
        gl.enableVertexAttribArray(aPos);
        gl.bindBuffer(gl.ARRAY_BUFFER, quad);
        gl.vertexAttribPointer(aPos, 2, gl.FLOAT, false, 0, 0);

        const tex = gl.createTexture();
        gl.bindTexture(gl.TEXTURE_2D, tex);
        gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MIN_FILTER, gl.NEAREST);
        gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MAG_FILTER, gl.NEAREST);
        gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_S, gl.CLAMP_TO_EDGE);
        gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_T, gl.CLAMP_TO_EDGE);
        gl.texImage2D(
          gl.TEXTURE_2D, 0, gl.LUMINANCE, nFrames, nMels, 0,
          gl.LUMINANCE, gl.UNSIGNED_BYTE, mel8
        );

        gl.drawArrays(gl.TRIANGLES, 0, 6);
        return;
      }
    }

    // Canvas 2D fallback
    const ctx = canvas.getContext("2d");
    if (!ctx) return;
    const imgData = ctx.createImageData(nFrames, nMels);
    for (let i = 0; i < mel8.length; i++) {
      const [r, g, b] = infernoRGB(mel8[i] / 255);
      const pi = i * 4;
      imgData.data[pi] = r;
      imgData.data[pi + 1] = g;
      imgData.data[pi + 2] = b;
      imgData.data[pi + 3] = 255;
    }
    ctx.putImageData(imgData, 0, 0);
    ctx.drawImage(canvas, 0, 0, nFrames, nMels, 0, 0, canvas.width, canvas.height);
  }, [melData, nMels, nFrames]);

  return (
    <div className="viz-content">
      <div className="viz-content__header">
        <span className="panel-label">Mel Spectrogram · {nMels} bands</span>
        <span className="panel-meta">n_fft=400 · hop=160</span>
      </div>
      <canvas
        ref={canvasRef}
        className="viz-content__canvas"
        style={{ width: "100%", height: 280 }}
      />
    </div>
  );
}
