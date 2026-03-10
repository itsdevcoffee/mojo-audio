# Web App Design Brainstorm — Active Context

> This doc captures the in-progress brainstorming session for the mojo-audio web app redesign.
> It exists so context can be compacted without losing decisions made.

## Status

**Phase:** Iterating on visual design (DAW Studio look & feel) before continuing to implementation planning.

**Brainstorm server:** Run this to restart the visual companion:
```bash
/home/maskkiller/.claude/plugins/cache/claude-plugins-official/superpowers/5.0.0/lib/brainstorm-server/start-server.sh --project-dir /home/maskkiller/dev-coffee/repos/mojo-audio --host 0.0.0.0 --url-host fedora
```
Previous mockups are in `.superpowers/brainstorm/` — latest session dir has all HTML files.

**Brainstorm skill tasks still pending:**
- Task 8: Present design and get approval (IN PROGRESS — iterating on visual design)
- Task 9: Write design doc and transition to implementation planning

## Decisions Made

### 1. Audience & Model
- **Essentia-style library showcase** — developer-facing with interactive demos
- NOT a tool UI like RVC/Applio (Gradio-style)
- Visualizations serve as impressive demos that also happen to be useful tools

### 2. Deployment
- **Both local and deployed** — works locally with full backend (pixi run dev), also deployable as static site
- Public site: benchmarks from static JSON, analyzer shows "run locally" prompt
- Local: Remix proxies API calls to FastAPI seamlessly

### 3. Architecture (Approach C: Remix + FastAPI Proxied)
```
web/                          ← Remix app (NEW, replaces ui/frontend/)
  app/
    root.tsx                  ← Shell: nav, layout, theme, font loading
    routes/
      _index.tsx              ← Landing page
      benchmarks.tsx          ← Benchmark UI (rewrite from current index.html)
      analyzer.tsx            ← Audio analyzer (8 visualization panels)
      docs.tsx                ← Docs hub
      docs.$slug.tsx          ← Individual doc pages (markdown)
      playground.tsx          ← Interactive API explorer
    components/
      viz/                    ← Visualization components (reusable)
        Waveform.tsx, MelSpectrogram.tsx, Waterfall3D.tsx,
        Chromagram.tsx, LinearSTFT.tsx, SpectralCentroid.tsx,
        RMSEnergy.tsx, FrequencyDist.tsx
      layout/Nav.tsx, Footer.tsx
      ui/                     ← Shared UI primitives
    lib/
      api.server.ts           ← Proxy to FastAPI in dev
    styles/global.css         ← Theme

ui/                           ← FastAPI backend (KEEP, API only)
  backend/app.py              ← /analyze, /api/benchmark/* endpoints
```

- `pixi run dev` starts both Remix (port 3000) and FastAPI (port 8000)
- Remix proxies `/api/*` and `/analyze` to FastAPI in dev mode

### 4. Navigation
- **Top nav bar** (not sidebar) — maximizes content width for visualizations
- Links: Benchmarks, Analyzer, Docs, Playground, GitHub ↗

### 5. Analyzer Layout
- **Waveform pinned at top** (always visible, shared timeline)
- **7 other panels in tabs** below — one visible at a time
- Tabs: Mel Spectrogram, Linear STFT, Chromagram, 3D Waterfall, RMS Energy, Spectral Centroid, Frequency Dist

### 6. Design Language — "Jade Depths" (Glassmorphic Mint)
- **Style:** Gradient + Glassmorphic — frosted panels with backdrop-blur and radial glow
- **Background:** #060d0a (deep dark green-black)
- **Glass panels:** rgba(10,20,16,0.7) with backdrop-filter:blur(10px), border: 1px solid rgba(80,200,150,0.08)
- **Accent gradient:** linear-gradient(135deg, #50e8a8, #38c8b4) — emerald → teal
- **Subtle fills:** rgba(80,200,150,0.07)
- **Text:** #c8e8d8 (muted mint-white), brighter #e0f0e8 for active
- **Inactive/muted text:** #6aaa90
- **Panel borders:** rgba(80,200,150,0.06) — subtle green tint
- **Radial glow blobs:** rgba(60,200,140,0.06) and rgba(40,180,160,0.04) behind content
- **VAD regions:** green tint (rgba(0,220,120,...)) — kept from original, blends naturally
- **Waveform gradient:** emerald-to-teal matching accent gradient
- **Inferno colormap** for spectrograms (pops against dark background)
- **Typography:** SF Mono / Fira Code monospace, lighter weight
- Pro audio tool aesthetic with jewel-toned depth — richer and more organic than flat blue
- Evolved from "DAW Studio" through Mint Frost (option C) into glassmorphic variant

### What Still Needs Iteration
- Landing page design in Jade Depths theme
- Benchmark page layout
- Docs page layout
- Specific typography choices (weights, sizes)
- Button/interactive element styling
- How the "run locally" prompt looks on deployed analyzer page

## The 8 Visualization Panels

All computed from existing Mojo DSP functions (stft, power_spectrum, rms_energy, fft, mel_spectrogram, get_voice_segments):

| Panel | Source DSP | Rendering | Description |
|-------|-----------|-----------|-------------|
| Waveform + VAD | read_wav, get_voice_segments | Canvas 2D | RMS envelope, green VAD overlay |
| Mel Spectrogram | mel_spectrogram | WebGL shader | 80 bands, inferno colormap |
| Linear STFT | stft | WebGL shader | Full-res linear frequency, harmonics visible |
| Chromagram | stft → pitch class folding | WebGL/Canvas | 12 pitch classes over time |
| 3D Waterfall | mel_spectrogram | Three.js | Terrain mesh, auto-rotation |
| RMS Energy | rms_energy / chunked RMS | Canvas 2D | Peak vs RMS curves, dB scale |
| Spectral Centroid | stft → centroid calc | Canvas 2D | Brightness line over time |
| Frequency Dist | power_spectrum averaged | Canvas 2D | Energy per frequency band |

## Existing Implementation (Already Merged to main)

The current visualizer is already working with 3 panels:
- `ui/backend/app.py` — POST /analyze endpoint (subprocess to Mojo DSP)
- `ui/frontend/visualizer.html` — self-contained HTML with inline JS
- Backend runs Mojo via subprocess: `pixi run -- mojo run -I src <temp.mojo>`
- Known fix: mel_spectrogram returns log-scale values (some negative), normalize with min-max directly (no log1p)
- Known fix: Use Uint8 WebGL textures (OES_texture_float_linear missing on some GPUs)
