# Audio Visualizer — Handoff Context

> Build a polished audio analysis visualizer using Three.js / WebGL / Canvas 2D.
> This is a demo tool for music producers — the visual language of a DAW, powered
> by mojo-audio's DSP layer under the hood.
>
> Target audience: Nick and Drew (music producers). They understand waveforms,
> spectrograms, and frequency content intuitively. Make it look like a professional
> audio tool, not a Python script output.

---

## What to Build

A single-page audio analyzer. User drags a WAV file onto the page. Three panels
appear simultaneously, all rendered with GPU-accelerated graphics:

```
┌────────────────────────────────────────────────────────┐
│  WAVEFORM  (Canvas 2D — full width)                    │
│  Envelope + VAD regions highlighted in color           │
├─────────────────────────────┬──────────────────────────┤
│  MEL SPECTROGRAM            │  3D WATERFALL            │
│  (WebGL shader — 2D view)   │  (Three.js — 3D view)    │
│  time × freq × intensity    │  same data, rotated 3D   │
└─────────────────────────────┴──────────────────────────┘
```

---

## Existing Code to Build On

The repo already has a FastAPI backend and frontend for benchmarks. **Add to this,
don't replace it.** The visualizer is a new page alongside the existing benchmark UI.

```
ui/
  backend/
    app.py                ← ADD /analyze endpoint here
    generate_demo_data.py
  frontend/
    index.html            ← existing benchmark page, don't touch
    visualizer.html       ← CREATE THIS (new page)
  static/
    css/style.css         ← can reference for shared variables/reset
    js/main.js            ← existing benchmark JS, don't touch
```

Run the server: `pixi run ui` (from repo root)
URL: `http://localhost:8000`
New page: `http://localhost:8000/visualizer`

---

## Step 1: FastAPI `/analyze` Endpoint

Add to `ui/backend/app.py`:

```python
import sys
import tempfile
import json
import numpy as np
from pathlib import Path
from fastapi import UploadFile, File

# Add src/ to path for mojo_audio models
sys.path.insert(0, str(REPO_ROOT / "src"))

@app.post("/analyze")
async def analyze_audio(file: UploadFile = File(...)):
    """
    Accept a WAV file, process via mojo-audio DSP, return visualization data.

    Returns JSON with waveform, mel spectrogram, and VAD regions.
    """
    # Write upload to temp file
    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
        content = await file.read()
        tmp.write(content)
        tmp_path = tmp.name

    try:
        return _process_audio(tmp_path)
    finally:
        Path(tmp_path).unlink(missing_ok=True)


def _process_audio(wav_path: str) -> dict:
    """Process audio through mojo-audio DSP and return visualization data."""
    from models._weight_loader import load_weights  # noqa - confirms models importable
    from wav_io import read_wav
    from resample import resample_to_16k
    from vad import get_voice_segments
    from audio import mel_spectrogram, hann_window

    # 1. Load audio
    samples, sample_rate = read_wav(wav_path)
    # samples is List[Float32], convert to numpy
    samples_np = np.array(samples, dtype=np.float32)
    duration_s = len(samples_np) / sample_rate

    # 2. Waveform for display — downsample to ~4000 points (enough for smooth render)
    display_points = 4000
    hop = max(1, len(samples_np) // display_points)
    # RMS envelope per chunk (looks better than raw samples for long files)
    chunks = [samples_np[i:i+hop] for i in range(0, len(samples_np), hop)]
    waveform_display = [float(np.sqrt(np.mean(c**2))) if len(c) > 0 else 0.0
                        for c in chunks]

    # 3. Resample to 16kHz for mel spectrogram
    samples_16k_list = resample_to_16k(list(samples), sample_rate)
    samples_16k = np.array(samples_16k_list, dtype=np.float32)

    # 4. Mel spectrogram — 80 mel bands, matching Whisper params
    # mel_spectrogram returns List[List[Float32]] shape (n_mels, n_frames)
    n_fft = 400
    hop_length = 160  # 10ms at 16kHz
    mel_spec = mel_spectrogram(
        list(samples_16k),
        n_mels=80,
        n_fft=n_fft,
        hop_length=hop_length,
        sample_rate=16000,
    )
    # Convert to numpy [n_mels, n_frames]
    mel_np = np.array([[v for v in row] for row in mel_spec], dtype=np.float32)
    n_mels, n_frames = mel_np.shape

    # Normalize to [0, 1] for the shader (log scale looks better)
    mel_log = np.log1p(mel_np)
    mel_norm = (mel_log - mel_log.min()) / (mel_log.max() - mel_log.min() + 1e-8)

    # 5. VAD regions
    # get_voice_segments returns List[List[Int]] [[start_sample, end_sample], ...]
    voice_segs = get_voice_segments(list(samples), sample_rate)
    vad_regions = [
        {"start": float(seg[0] / sample_rate), "end": float(seg[1] / sample_rate)}
        for seg in voice_segs
    ]

    return {
        "duration_s": float(duration_s),
        "sample_rate": int(sample_rate),
        "waveform": waveform_display,           # list of floats, ~4000 points, RMS envelope
        "mel_spectrogram": mel_norm.flatten().tolist(),  # n_mels * n_frames floats, row-major
        "mel_n_mels": int(n_mels),
        "mel_n_frames": int(n_frames),
        "vad_regions": vad_regions,             # [{start, end}, ...] in seconds
    }
```

Also add a route to serve the new page:

```python
@app.get("/visualizer")
async def visualizer_page():
    return FileResponse(str(UI_ROOT / "frontend" / "visualizer.html"))
```

**Test the endpoint before building the frontend:**
```bash
curl -X POST http://localhost:8000/analyze \
  -F "file=@/path/to/vocal.wav" | python -m json.tool | head -20
```

---

## Step 2: `ui/frontend/visualizer.html`

Single self-contained HTML file. Three.js and other deps from CDN — no npm, no build step.

### Page structure and dark theme

```html
<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8">
  <title>mojo-audio | Audio Analyzer</title>
  <script src="https://cdnjs.cloudflare.com/ajax/libs/three.js/r128/three.min.js"></script>
  <style>
    * { margin: 0; padding: 0; box-sizing: border-box; }
    body {
      background: #080810;
      color: #e0e0ff;
      font-family: 'SF Mono', 'Fira Code', monospace;
      height: 100vh;
      overflow: hidden;
      display: flex;
      flex-direction: column;
    }

    /* Drop zone — shown before file upload */
    #drop-zone {
      position: absolute;
      inset: 0;
      display: flex;
      flex-direction: column;
      align-items: center;
      justify-content: center;
      gap: 16px;
      border: 2px dashed rgba(100, 200, 255, 0.3);
      margin: 40px;
      border-radius: 16px;
      transition: border-color 0.2s, background 0.2s;
      cursor: pointer;
      z-index: 10;
    }
    #drop-zone.drag-over {
      border-color: rgba(100, 200, 255, 0.8);
      background: rgba(100, 200, 255, 0.05);
    }
    #drop-zone h2 { font-size: 1.4rem; color: #a0c4ff; font-weight: 400; }
    #drop-zone p { font-size: 0.85rem; color: rgba(160, 180, 255, 0.5); }

    /* Panels — shown after analysis */
    #panels {
      display: none;
      flex-direction: column;
      height: 100vh;
      padding: 12px;
      gap: 10px;
    }

    /* Top panel: waveform */
    #waveform-panel {
      flex: 0 0 140px;
      position: relative;
      border-radius: 8px;
      overflow: hidden;
      background: #0d0d1a;
      border: 1px solid rgba(100, 120, 200, 0.15);
    }
    #waveform-canvas { width: 100%; height: 100%; }

    /* Bottom row: mel spec + 3D waterfall */
    #bottom-row {
      flex: 1;
      display: grid;
      grid-template-columns: 1fr 1fr;
      gap: 10px;
      min-height: 0;
    }
    .bottom-panel {
      border-radius: 8px;
      overflow: hidden;
      background: #0d0d1a;
      border: 1px solid rgba(100, 120, 200, 0.15);
      position: relative;
    }
    canvas { display: block; }

    /* Labels */
    .panel-label {
      position: absolute;
      top: 8px;
      left: 12px;
      font-size: 0.65rem;
      letter-spacing: 0.1em;
      text-transform: uppercase;
      color: rgba(160, 180, 255, 0.4);
      z-index: 1;
      pointer-events: none;
    }

    /* Loading indicator */
    #loading {
      display: none;
      position: absolute;
      inset: 0;
      align-items: center;
      justify-content: center;
      background: rgba(8, 8, 16, 0.9);
      z-index: 20;
      font-size: 0.9rem;
      color: rgba(160, 180, 255, 0.7);
      letter-spacing: 0.1em;
    }
  </style>
</head>
<body>
  <div id="loading">ANALYZING...</div>

  <div id="drop-zone">
    <h2>Drop a vocal stem here</h2>
    <p>WAV · 44.1kHz or 48kHz · mono or stereo</p>
    <p style="margin-top:8px; font-size:0.75rem;">or click to browse</p>
    <input type="file" id="file-input" accept=".wav" style="display:none">
  </div>

  <div id="panels">
    <div id="waveform-panel">
      <span class="panel-label">Waveform</span>
      <canvas id="waveform-canvas"></canvas>
    </div>
    <div id="bottom-row">
      <div class="bottom-panel" id="mel-panel">
        <span class="panel-label">Mel Spectrogram · 80 bands · 16kHz</span>
        <canvas id="mel-canvas"></canvas>
      </div>
      <div class="bottom-panel" id="waterfall-panel">
        <span class="panel-label">3D Waterfall</span>
        <div id="three-container" style="width:100%;height:100%;"></div>
      </div>
    </div>
  </div>

  <script>
  // ─────────────────────────────────────────────────────────
  // DROP ZONE + FILE UPLOAD
  // ─────────────────────────────────────────────────────────
  const dropZone = document.getElementById('drop-zone');
  const fileInput = document.getElementById('file-input');
  const loading = document.getElementById('loading');
  const panels = document.getElementById('panels');

  dropZone.addEventListener('click', () => fileInput.click());
  fileInput.addEventListener('change', e => handleFile(e.target.files[0]));

  dropZone.addEventListener('dragover', e => {
    e.preventDefault();
    dropZone.classList.add('drag-over');
  });
  dropZone.addEventListener('dragleave', () => dropZone.classList.remove('drag-over'));
  dropZone.addEventListener('drop', e => {
    e.preventDefault();
    dropZone.classList.remove('drag-over');
    handleFile(e.dataTransfer.files[0]);
  });

  async function handleFile(file) {
    if (!file || !file.name.endsWith('.wav')) return;

    dropZone.style.display = 'none';
    loading.style.display = 'flex';

    const form = new FormData();
    form.append('file', file);

    const res = await fetch('/analyze', { method: 'POST', body: form });
    const data = await res.json();

    loading.style.display = 'none';
    panels.style.display = 'flex';

    // Slight delay to let panels layout before measuring
    requestAnimationFrame(() => {
      drawWaveform(data);
      drawMelSpec(data);
      drawWaterfall(data);
    });
  }


  // ─────────────────────────────────────────────────────────
  // PANEL 1: WAVEFORM (Canvas 2D)
  // ─────────────────────────────────────────────────────────
  function drawWaveform(data) {
    const canvas = document.getElementById('waveform-canvas');
    const panel = document.getElementById('waveform-panel');
    canvas.width = panel.clientWidth;
    canvas.height = panel.clientHeight;
    const ctx = canvas.getContext('2d');
    const W = canvas.width, H = canvas.height;
    const mid = H / 2;

    // Background
    ctx.fillStyle = '#0d0d1a';
    ctx.fillRect(0, 0, W, H);

    // VAD regions — draw first (behind waveform)
    data.vad_regions.forEach(region => {
      const x1 = (region.start / data.duration_s) * W;
      const x2 = (region.end / data.duration_s) * W;
      ctx.fillStyle = 'rgba(0, 220, 120, 0.08)';
      ctx.fillRect(x1, 0, x2 - x1, H);
      // Top border accent
      ctx.fillStyle = 'rgba(0, 220, 120, 0.3)';
      ctx.fillRect(x1, 0, x2 - x1, 1);
    });

    // Zero line
    ctx.strokeStyle = 'rgba(100, 120, 200, 0.2)';
    ctx.lineWidth = 1;
    ctx.beginPath(); ctx.moveTo(0, mid); ctx.lineTo(W, mid); ctx.stroke();

    // Waveform envelope — filled mirrored shape
    const wf = data.waveform;
    const step = W / wf.length;
    const maxVal = Math.max(...wf) || 1;

    // Gradient
    const grad = ctx.createLinearGradient(0, 0, 0, H);
    grad.addColorStop(0, 'rgba(80, 160, 255, 0.9)');
    grad.addColorStop(0.5, 'rgba(80, 160, 255, 0.4)');
    grad.addColorStop(1, 'rgba(80, 160, 255, 0.9)');

    ctx.fillStyle = grad;
    ctx.beginPath();
    ctx.moveTo(0, mid);
    // Top half
    for (let i = 0; i < wf.length; i++) {
      const x = i * step;
      const amplitude = (wf[i] / maxVal) * (mid - 8);
      ctx.lineTo(x, mid - amplitude);
    }
    // Bottom half (mirror)
    for (let i = wf.length - 1; i >= 0; i--) {
      const x = i * step;
      const amplitude = (wf[i] / maxVal) * (mid - 8);
      ctx.lineTo(x, mid + amplitude);
    }
    ctx.closePath();
    ctx.fill();
  }


  // ─────────────────────────────────────────────────────────
  // PANEL 2: MEL SPECTROGRAM (WebGL shader)
  // ─────────────────────────────────────────────────────────
  function drawMelSpec(data) {
    const canvas = document.getElementById('mel-canvas');
    const panel = document.getElementById('mel-panel');
    canvas.width = panel.clientWidth;
    canvas.height = panel.clientHeight;

    const gl = canvas.getContext('webgl');
    if (!gl) {
      // Fallback to canvas 2D if WebGL unavailable
      drawMelSpecCanvas2D(data, canvas);
      return;
    }

    // Vertex shader — full-screen quad
    const vsrc = `
      attribute vec2 a_pos;
      varying vec2 v_uv;
      void main() {
        v_uv = a_pos * 0.5 + 0.5;
        gl_Position = vec4(a_pos, 0.0, 1.0);
      }
    `;

    // Fragment shader — inferno colormap on mel spectrogram texture
    const fsrc = `
      precision mediump float;
      uniform sampler2D u_spec;
      uniform float u_n_mels;
      uniform float u_n_frames;
      varying vec2 v_uv;

      // Inferno colormap (polynomial approximation)
      vec3 inferno(float t) {
        t = clamp(t, 0.0, 1.0);
        vec3 c0 = vec3(0.0002, 0.0016, 0.0138);
        vec3 c1 = vec3(0.1061, 0.0544, 0.3942);
        vec3 c2 = vec3(0.4974, 0.1053, 0.4976);
        vec3 c3 = vec3(0.8787, 0.3980, 0.1726);
        vec3 c4 = vec3(0.9882, 0.8126, 0.1451);
        vec3 c = c0;
        c = mix(c, c1, smoothstep(0.0, 0.25, t));
        c = mix(c, c2, smoothstep(0.25, 0.5, t));
        c = mix(c, c3, smoothstep(0.5, 0.75, t));
        c = mix(c, c4, smoothstep(0.75, 1.0, t));
        return c;
      }

      void main() {
        // v_uv.x = time (0=left, 1=right), v_uv.y = freq (0=low, 1=high)
        // Texture is stored row-major [n_mels, n_frames]
        // mel index = (1 - v_uv.y) * n_mels (flip Y so low freq at bottom)
        float mel_idx = (1.0 - v_uv.y) / u_n_mels;
        float frame_idx = v_uv.x / u_n_frames;
        // Sample: texture is [n_mels rows × n_frames cols] stored as 1D → 2D texture
        float energy = texture2D(u_spec, vec2(v_uv.x, 1.0 - v_uv.y)).r;
        gl_FragColor = vec4(inferno(energy), 1.0);
      }
    `;

    // Compile shaders
    function compile(type, src) {
      const s = gl.createShader(type);
      gl.shaderSource(s, src); gl.compileShader(s);
      return s;
    }
    const prog = gl.createProgram();
    gl.attachShader(prog, compile(gl.VERTEX_SHADER, vsrc));
    gl.attachShader(prog, compile(gl.FRAGMENT_SHADER, fsrc));
    gl.linkProgram(prog);
    gl.useProgram(prog);

    // Full-screen quad
    const buf = gl.createBuffer();
    gl.bindBuffer(gl.ARRAY_BUFFER, buf);
    gl.bufferData(gl.ARRAY_BUFFER,
      new Float32Array([-1,-1, 1,-1, -1,1, 1,1]), gl.STATIC_DRAW);
    const pos = gl.getAttribLocation(prog, 'a_pos');
    gl.enableVertexAttribArray(pos);
    gl.vertexAttribPointer(pos, 2, gl.FLOAT, false, 0, 0);

    // Upload mel spectrogram as texture
    // Texture size: n_frames wide × n_mels tall
    const { mel_n_mels: nMels, mel_n_frames: nFrames } = data;
    const mel = new Float32Array(data.mel_spectrogram);

    // Rearrange: texture expects [height=nMels, width=nFrames]
    // Data is row-major [nMels, nFrames] — already correct
    const tex = gl.createTexture();
    gl.bindTexture(gl.TEXTURE_2D, tex);
    gl.texImage2D(gl.TEXTURE_2D, 0, gl.LUMINANCE, nFrames, nMels, 0,
                  gl.LUMINANCE, gl.FLOAT, mel);
    // For WebGL1 non-power-of-two textures, must use CLAMP and NEAREST/LINEAR
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_S, gl.CLAMP_TO_EDGE);
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_T, gl.CLAMP_TO_EDGE);
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MIN_FILTER, gl.LINEAR);
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MAG_FILTER, gl.LINEAR);

    // Note: gl.FLOAT textures require OES_texture_float extension in WebGL1
    const ext = gl.getExtension('OES_texture_float');
    if (!ext) {
      // Fallback: quantize to Uint8 (0-255)
      const mel8 = new Uint8Array(mel.map(v => Math.round(v * 255)));
      gl.texImage2D(gl.TEXTURE_2D, 0, gl.LUMINANCE, nFrames, nMels, 0,
                    gl.LUMINANCE, gl.UNSIGNED_BYTE, mel8);
    }

    gl.uniform1i(gl.getUniformLocation(prog, 'u_spec'), 0);
    gl.uniform1f(gl.getUniformLocation(prog, 'u_n_mels'), nMels);
    gl.uniform1f(gl.getUniformLocation(prog, 'u_n_frames'), nFrames);

    gl.viewport(0, 0, canvas.width, canvas.height);
    gl.drawArrays(gl.TRIANGLE_STRIP, 0, 4);
  }

  // Canvas 2D fallback for mel spec (if WebGL unavailable)
  function drawMelSpecCanvas2D(data, canvas) {
    const ctx = canvas.getContext('2d');
    const { mel_n_mels: nM, mel_n_frames: nF } = data;
    const mel = data.mel_spectrogram;
    const W = canvas.width, H = canvas.height;
    const cellW = W / nF, cellH = H / nM;
    for (let m = 0; m < nM; m++) {
      for (let f = 0; f < nF; f++) {
        const v = mel[m * nF + f];
        const r = Math.round(v * 255);
        ctx.fillStyle = `rgb(${r},${Math.round(r*0.3)},${Math.round((1-v)*80)})`;
        ctx.fillRect(f * cellW, (nM - 1 - m) * cellH, cellW + 1, cellH + 1);
      }
    }
  }


  // ─────────────────────────────────────────────────────────
  // PANEL 3: 3D WATERFALL (Three.js)
  // ─────────────────────────────────────────────────────────
  function drawWaterfall(data) {
    const container = document.getElementById('three-container');
    const W = container.clientWidth, H = container.clientHeight;

    const renderer = new THREE.WebGLRenderer({ antialias: true, alpha: true });
    renderer.setSize(W, H);
    renderer.setClearColor(0x0d0d1a, 1);
    container.appendChild(renderer.domElement);

    const scene = new THREE.Scene();

    // Camera: slightly above and in front, looking down at the waterfall
    const camera = new THREE.PerspectiveCamera(50, W / H, 0.1, 100);
    camera.position.set(0, 4, 7);
    camera.lookAt(0, 0, 0);

    const { mel_n_mels: nMels, mel_n_frames: nFrames } = data;
    const mel = data.mel_spectrogram;

    // Build geometry: width = time (nFrames), depth = frequency (nMels)
    // Each vertex height = mel spectrogram value at that (time, freq) point
    const planeW = 8, planeD = 4;
    const geometry = new THREE.PlaneGeometry(planeW, planeD, nFrames - 1, nMels - 1);
    geometry.rotateX(-Math.PI / 2); // lay flat

    const positions = geometry.attributes.position;
    const colors = [];
    const maxH = 1.5; // max vertex height

    // Inferno colormap in JS (matches the shader)
    function inferno(t) {
      t = Math.max(0, Math.min(1, t));
      const stops = [
        [0.000, 0.002, 0.014],
        [0.106, 0.054, 0.394],
        [0.497, 0.105, 0.498],
        [0.879, 0.398, 0.173],
        [0.988, 0.813, 0.145],
      ];
      const i = t * (stops.length - 1);
      const lo = Math.floor(i), hi = Math.ceil(i);
      const frac = i - lo;
      const a = stops[Math.min(lo, stops.length-1)];
      const b = stops[Math.min(hi, stops.length-1)];
      return [a[0]+(b[0]-a[0])*frac, a[1]+(b[1]-a[1])*frac, a[2]+(b[2]-a[2])*frac];
    }

    // Map mel spectrogram values to vertex heights and colors
    // PlaneGeometry vertex order: row by row, X = time (left→right), Z = freq (back→front)
    for (let i = 0; i < positions.count; i++) {
      const col = i % nFrames;           // time index
      const row = Math.floor(i / nFrames); // freq index (0=low)
      const melIdx = (nMels - 1 - row) * nFrames + col; // flip freq axis
      const val = mel[Math.min(melIdx, mel.length - 1)] || 0;
      positions.setY(i, val * maxH);
      const [r, g, b] = inferno(val);
      colors.push(r, g, b);
    }

    positions.needsUpdate = true;
    geometry.setAttribute('color', new THREE.Float32BufferAttribute(colors, 3));
    geometry.computeVertexNormals();

    const material = new THREE.MeshPhongMaterial({
      vertexColors: true,
      side: THREE.DoubleSide,
      shininess: 20,
      wireframe: false,
    });

    const mesh = new THREE.Mesh(geometry, material);
    scene.add(mesh);

    // Lighting
    scene.add(new THREE.AmbientLight(0x334466, 0.8));
    const dirLight = new THREE.DirectionalLight(0xaabbff, 1.2);
    dirLight.position.set(3, 8, 5);
    scene.add(dirLight);

    // Subtle fog for depth
    scene.fog = new THREE.Fog(0x0d0d1a, 12, 25);

    // Slow auto-rotation to show the 3D shape
    let angle = 0;
    function animate() {
      requestAnimationFrame(animate);
      angle += 0.003;
      camera.position.x = Math.sin(angle) * 7;
      camera.position.z = Math.cos(angle) * 7;
      camera.lookAt(0, 0.5, 0);
      renderer.render(scene, camera);
    }
    animate();
  }
  </script>
</body>
</html>
```

---

## Step 3: Wire Up the Route in `app.py`

Add to the existing `app.py` after the existing static mount:

```python
@app.get("/visualizer")
async def visualizer_page():
    """Serve the audio visualizer page."""
    return FileResponse(str(UI_ROOT / "frontend" / "visualizer.html"))
```

---

## Importing mojo-audio from the FastAPI backend

The backend runs from `ui/backend/` but mojo-audio's Python modules are in `src/`.
Add to the top of `app.py` (after existing imports):

```python
import sys
REPO_ROOT = Path(__file__).parent.parent.parent  # already defined in app.py
if str(REPO_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(REPO_ROOT / "src"))
```

The mojo-audio functions to use:
- `from wav_io import read_wav` — WAV file loading
- `from resample import resample_to_16k` — downsampling to 16kHz
- `from vad import get_voice_segments` — voice activity detection
- `from audio import mel_spectrogram` — mel spectrogram computation

All of these are pure Mojo compiled into the pixi environment and importable directly.

---

## Known Issues to Handle

**OES_texture_float extension:** WebGL1 requires `gl.getExtension('OES_texture_float')`
for float textures. If unavailable (some mobile browsers), fall back to `Uint8Array`
by quantizing `mel_norm * 255`. The code above includes this fallback already.

**Large files:** The mel spectrogram computation scales linearly with audio length.
For a 3-minute song, `n_frames` will be ~1800. The shader handles any size; the
Three.js geometry should be decimated if `n_frames > 500` to avoid vertex count
issues. Add: `const displayFrames = Math.min(nFrames, 300)` and sample accordingly.

**CORS:** The backend already has `allow_origins=["*"]` for local dev.

---

## What Done Looks Like

1. `pixi run ui` starts the server
2. Navigate to `http://localhost:8000/visualizer`
3. Drag a WAV vocal stem onto the page
4. Within 2-3 seconds:
   - **Top panel:** waveform fills the width, green-tinted VAD regions show where voice is active
   - **Bottom left:** mel spectrogram in inferno colormap (black/purple/red/yellow) — harmonics visible as horizontal bands, words visible as vertical patterns
   - **Bottom right:** 3D waterfall slowly rotating, showing the time-frequency landscape as a 3D terrain
5. The backend logs confirm mojo-audio functions were called

---

## Pixi task to add

In `pixi.toml`, the existing `ui` task already serves this. No new task needed.

---

## Files Changed

```
ui/backend/app.py          ← add /analyze endpoint + /visualizer route + sys.path
ui/frontend/visualizer.html ← CREATE (all JS inline, no dependencies to install)
```

That's it. Two files.
