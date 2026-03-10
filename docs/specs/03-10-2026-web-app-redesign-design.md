# mojo-audio Web App Redesign — Design Spec

> Validated design for the mojo-audio web app, replacing the current static HTML frontend with a Remix-based app featuring 8 GPU-accelerated audio visualization panels.

## Audience & Purpose

**Essentia-style library showcase** — developer-facing with interactive demos. NOT a tool UI like RVC/Applio. Visualizations serve as impressive demos that also happen to be useful tools.

## Architecture: Remix + FastAPI Proxied

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

### Proxy Mechanism

Remix Vite dev server proxies `/api/*` and `/analyze` to FastAPI using Vite's built-in `server.proxy` config in `vite.config.ts`. No custom proxy code needed — Vite handles it natively.

### Deployment Model

- **Local:** `pixi run dev` starts both Remix (port 3000) and FastAPI (port 8000) via a new pixi task that runs both processes concurrently.
- **Deployed (static):** Benchmarks render from static JSON bundled at build time. Analyzer page detects missing backend via failed fetch to `/analyze` and shows a "run locally" banner with install instructions (`pixi run dev`). All viz panels are backend-dependent — no partial static mode.

## Navigation

**Top nav bar** (not sidebar) — maximizes content width for visualizations.

Links: Benchmarks | Analyzer | Docs | Playground | GitHub ↗

## Analyzer Layout

- **Waveform pinned at top** — always visible, shared timeline, VAD overlay
- **7 other panels in tabs** below — one visible at a time
- Tabs: Mel Spectrogram, Linear STFT, Chromagram, 3D Waterfall, RMS Energy, Spectral Centroid, Frequency Dist

## Design Language: "Jade Depths" (Glassmorphic Mint)

### Colors

| Token | Value | Usage |
|-------|-------|-------|
| `--bg` | `#060d0a` | Page background (deep green-black) |
| `--panel-glass` | `rgba(10,20,16,0.7)` | Glass panel fill + `backdrop-filter:blur(10px)` |
| `--panel-border` | `rgba(80,200,150,0.08)` | Panel borders |
| `--accent-start` | `#50e8a8` | Gradient start (emerald) |
| `--accent-end` | `#38c8b4` | Gradient end (teal) |
| `--accent-gradient` | `linear-gradient(135deg, #50e8a8, #38c8b4)` | Brand, active tabs, buttons |
| `--subtle` | `rgba(80,200,150,0.07)` | Hover states, subtle fills |
| `--text` | `#c8e8d8` | Default body text |
| `--text-bright` | `#e0f0e8` | Active/highlighted text |
| `--text-muted` | `#6aaa90` | Inactive tabs, secondary labels |
| `--glow-primary` | `rgba(60,200,140,0.06)` | Radial glow blob behind content |
| `--glow-secondary` | `rgba(40,180,160,0.04)` | Secondary glow blob |
| `--vad-fill` | `rgba(0,220,120,0.04)` | VAD region background fill |
| `--vad-border` | `rgba(0,220,120,0.12)` | VAD region top border |

### Glass Effect

All panels use glassmorphism:
```css
background: var(--panel-glass);
backdrop-filter: blur(10px);
border: 1px solid var(--panel-border);
box-shadow: inset 0 1px 0 rgba(80,200,150,0.04);
```

Radial glow blobs (`--glow-primary`, `--glow-secondary`) positioned behind content areas for depth.

### Typography

- **Font stack:** `'SF Mono', 'Fira Code', monospace`
- **Lighter weight** — 300-400 for body, 600 for brand/emphasis
- **Panel labels:** 8-9px uppercase, letter-spacing 0.08-0.1em, `--text-muted`
- **Body:** 11-12px
- **Nav brand:** 12px, weight 600, accent gradient text

### Spectrogram Colormap

**Inferno** — pops against the dark green-black background. Already implemented in current visualizer (GLSL shader + JS fallback).

## The 8 Visualization Panels

All computed from existing Mojo DSP functions:

| Panel | Source DSP | Rendering | Description |
|-------|-----------|-----------|-------------|
| Waveform + VAD | read_wav, get_voice_segments | Canvas 2D | RMS envelope, emerald-teal gradient fill, green VAD overlay |
| Mel Spectrogram | mel_spectrogram | WebGL shader | 80 bands, inferno colormap, Uint8 textures |
| Linear STFT | stft | WebGL shader | Full-res linear frequency, harmonics visible |
| Chromagram | stft → pitch class folding | Canvas 2D | 12 pitch classes over time, color-mapped grid |
| 3D Waterfall | mel_spectrogram | Three.js | Terrain mesh, auto-rotation, frame decimation |
| RMS Energy | rms_energy / chunked RMS | Canvas 2D | Peak vs RMS curves, dB scale |
| Spectral Centroid | stft → centroid calc | Canvas 2D | Brightness line over time |
| Frequency Dist | power_spectrum averaged | Canvas 2D | Energy per frequency band |

## Existing Implementation

Already working on main with 3 panels (Waveform, Mel Spectrogram, 3D Waterfall):
- `ui/backend/app.py` — POST /analyze endpoint (subprocess to Mojo DSP)
- `ui/frontend/visualizer.html` — self-contained HTML with inline JS

### Known Technical Constraints

- **Mojo subprocess pattern:** `pixi run -- mojo run -I src <temp.mojo>` — Mojo modules are NOT Python-importable
- **Mel normalization:** mel_spectrogram returns log-scale values, min-max normalize directly (no log1p)
- **WebGL textures:** Always use Uint8 (OES_texture_float_linear missing on some GPUs)
- **Three.js:** Install as npm dependency (`three@^0.128.0`) for proper Vite/Remix bundling. Current CDN approach won't work in a Vite build.
- **HiDPI:** `renderer.setPixelRatio(Math.min(window.devicePixelRatio, 2))`

### `/analyze` Response Contract

The FastAPI `/analyze` endpoint returns JSON with this shape (extending current implementation):

```json
{
  "waveform": [float],           // Raw samples, downsampled to ~2000 points
  "sample_rate": int,            // e.g. 44100
  "duration": float,             // seconds
  "vad_segments": [[start, end], ...],  // seconds
  "mel_spectrogram": [[float]],  // 2D: [n_frames x n_mels], log-scale
  "stft_magnitude": [[float]],   // 2D: [n_frames x n_freq_bins], linear
  "rms_energy": [float],         // Per-frame RMS values
  "power_spectrum": [float]      // Averaged power spectrum (1D)
}
```

Chromagram and spectral centroid are computed client-side from `stft_magnitude` to avoid bloating the response. The backend computes the expensive DSP; the frontend does lightweight derivations.

### CSS Approach

Plain CSS with CSS custom properties (variables) in `global.css`. No Tailwind, no CSS modules — keeps it simple and matches the monospace/minimal aesthetic. Components use class names.

### Font Strategy

SF Mono is macOS-only and serves as a bonus. Fira Code (loaded via Google Fonts or self-hosted) is the primary cross-platform font. The `monospace` generic fallback covers edge cases.

## Pages

1. **Landing** — Hero with brand, performance claim ("20-40% faster than librosa"), mini viz previews, CTAs to Analyzer and Docs
2. **Benchmarks** — Rewrite of current index.html in Jade Depths theme, static JSON data bundled at build
3. **Analyzer** — The star: drop zone, waveform + 7 tabbed panels
4. **Docs** — Hub page lists available docs. Individual pages render markdown from `docs/` directory, loaded at build time via Remix loaders. Syntax highlighting with Shiki.
5. **Playground** — Phase 2 (not in initial implementation). Will allow users to call individual DSP functions with custom parameters and see results. Deferred until core app is stable.
