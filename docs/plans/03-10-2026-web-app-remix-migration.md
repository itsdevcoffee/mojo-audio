# Web App Remix Migration — Implementation Plan

> **For agentic workers:** REQUIRED: Use superpowers:subagent-driven-development (if subagents available) or superpowers:executing-plans to implement this plan. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Replace the current static HTML frontend (`ui/frontend/`) with a Remix app (`web/`) featuring 8 GPU-accelerated audio visualization panels in the "Jade Depths" glassmorphic theme, while keeping the FastAPI backend (`ui/backend/app.py`) as the API layer.

**Architecture:** Remix (Vite) frontend at `web/` proxies `/api/*` and `/analyze` to FastAPI at port 8000 during development. The backend's `/analyze` endpoint is extended to return additional DSP data (STFT magnitude, RMS energy, power spectrum). Client-side code computes chromagram and spectral centroid from STFT data.

**Tech Stack:** Remix 2 (Vite), React 18, Three.js (npm), WebGL, Canvas 2D, CSS custom properties (no Tailwind), Fira Code font, FastAPI (existing)

**Spec:** `docs/specs/03-10-2026-web-app-redesign-design.md`

---

## File Structure

```
web/                                    ← NEW: Remix app
  package.json                          ← Dependencies: remix, react, three, etc.
  tsconfig.json                         ← TypeScript config
  vite.config.ts                        ← Vite config with FastAPI proxy
  app/
    root.tsx                            ← Shell: html, head, nav, theme, Fira Code
    entry.client.tsx                    ← Client hydration entry
    entry.server.tsx                    ← Server rendering entry
    styles/
      global.css                        ← Jade Depths theme tokens + base styles
      analyzer.css                      ← Analyzer page styles
      benchmarks.css                    ← Benchmark page styles
    routes/
      _index.tsx                        ← Landing page
      benchmarks.tsx                    ← Benchmark UI (port from index.html)
      analyzer.tsx                      ← Audio analyzer: drop zone, waveform, tabs
      docs.tsx                          ← Docs hub (stub for now)
    components/
      layout/
        Nav.tsx                         ← Top nav bar (glassmorphic)
      ui/
        GlassPanel.tsx                  ← Reusable glassmorphic panel wrapper
        TabBar.tsx                      ← Reusable tab bar component
      viz/
        Waveform.tsx                    ← Canvas 2D waveform + VAD (port existing)
        MelSpectrogram.tsx              ← WebGL mel spectrogram (port existing)
        Waterfall3D.tsx                 ← Three.js 3D terrain (port existing)
        LinearSTFT.tsx                  ← WebGL linear STFT (NEW)
        Chromagram.tsx                  ← Canvas 2D chromagram (NEW)
        RMSEnergy.tsx                   ← Canvas 2D RMS energy (NEW)
        SpectralCentroid.tsx            ← Canvas 2D spectral centroid (NEW)
        FrequencyDist.tsx               ← Canvas 2D frequency distribution (NEW)
        shared/
          inferno.ts                    ← Inferno colormap (shared by WebGL panels)
          webgl-utils.ts                ← WebGL boilerplate (shared)
      benchmark/
        ConfigCard.tsx                  ← Benchmark configuration form
        ResultCard.tsx                  ← Benchmark result display
    lib/
      types.ts                          ← TypeScript types for API responses
      use-analyzer.ts                   ← React hook: file upload → /analyze → state
      benchmark-api.ts                  ← Benchmark API client (port from JS)

ui/backend/app.py                       ← MODIFY: extend /analyze response
pixi.toml                               ← MODIFY: add `dev` task
```

---

## Chunk 1: Foundation — Remix Scaffold + Theme + Nav

### Task 1: Scaffold Remix App

**Files:**
- Create: `web/package.json`
- Create: `web/tsconfig.json`
- Create: `web/vite.config.ts`
- Create: `web/app/entry.client.tsx`
- Create: `web/app/entry.server.tsx`
- Create: `web/app/root.tsx`

- [ ] **Step 1: Initialize Remix project**

```bash
cd /home/maskkiller/dev-coffee/repos/mojo-audio
npx create-remix@latest web --template remix-run/remix/templates/vite --no-install --no-git-init
```

If the interactive template flag doesn't work, use:
```bash
mkdir -p web && cd web
npm init -y
npm install @remix-run/node @remix-run/react @remix-run/serve react react-dom isbot
npm install -D @remix-run/dev vite typescript @types/react @types/react-dom
```

- [ ] **Step 2: Configure Vite with FastAPI proxy**

Replace `web/vite.config.ts` with:

```typescript
import { vitePlugin as remix } from "@remix-run/dev";
import { defineConfig } from "vite";

export default defineConfig({
  plugins: [remix()],
  server: {
    port: 3000,
    proxy: {
      "/api": {
        target: "http://localhost:8000",
        changeOrigin: true,
      },
      "/analyze": {
        target: "http://localhost:8000",
        changeOrigin: true,
      },
    },
  },
});
```

- [ ] **Step 3: Create minimal root.tsx with Fira Code font**

Replace `web/app/root.tsx` with:

```tsx
import {
  Links,
  Meta,
  Outlet,
  Scripts,
  ScrollRestoration,
} from "@remix-run/react";
import "./styles/global.css";

export default function App() {
  return (
    <html lang="en">
      <head>
        <meta charSet="utf-8" />
        <meta name="viewport" content="width=device-width, initial-scale=1" />
        <link
          href="https://fonts.googleapis.com/css2?family=Fira+Code:wght@300;400;500;600&display=swap"
          rel="stylesheet"
        />
        <Meta />
        <Links />
      </head>
      <body>
        <Outlet />
        <ScrollRestoration />
        <Scripts />
      </body>
    </html>
  );
}
```

- [ ] **Step 4: Verify it starts**

```bash
cd web && npm run dev
```

Expected: Remix dev server on port 3000, blank page loads.

- [ ] **Step 5: Commit**

```bash
git add web/
git commit -m "feat(web): scaffold Remix app with Vite + FastAPI proxy config"
```

---

### Task 2: Jade Depths Theme (CSS Custom Properties)

**Files:**
- Create: `web/app/styles/global.css`

- [ ] **Step 1: Create global.css with all Jade Depths tokens**

```css
/* Jade Depths — Glassmorphic Mint Theme */

:root {
  /* Backgrounds */
  --bg: #060d0a;
  --panel-glass: rgba(10, 20, 16, 0.7);
  --panel-glass-nav: rgba(10, 20, 16, 0.5);
  --panel-border: rgba(80, 200, 150, 0.08);

  /* Accent gradient */
  --accent-start: #50e8a8;
  --accent-end: #38c8b4;
  --accent-gradient: linear-gradient(135deg, var(--accent-start), var(--accent-end));

  /* Subtle fills */
  --subtle: rgba(80, 200, 150, 0.07);
  --subtle-hover: rgba(80, 200, 150, 0.12);

  /* Text */
  --text: #c8e8d8;
  --text-bright: #e0f0e8;
  --text-muted: #6aaa90;

  /* Glow */
  --glow-primary: rgba(60, 200, 140, 0.06);
  --glow-secondary: rgba(40, 180, 160, 0.04);

  /* VAD */
  --vad-fill: rgba(0, 220, 120, 0.04);
  --vad-border: rgba(0, 220, 120, 0.12);

  /* Typography */
  --font-mono: 'Fira Code', 'SF Mono', monospace;
}

* {
  margin: 0;
  padding: 0;
  box-sizing: border-box;
}

html, body {
  background: var(--bg);
  color: var(--text);
  font-family: var(--font-mono);
  font-weight: 400;
  font-size: 14px;
  line-height: 1.5;
  -webkit-font-smoothing: antialiased;
}

/* Glass panel base */
.glass {
  background: var(--panel-glass);
  backdrop-filter: blur(10px);
  -webkit-backdrop-filter: blur(10px);
  border: 1px solid var(--panel-border);
  border-radius: 8px;
  box-shadow: inset 0 1px 0 rgba(80, 200, 150, 0.04);
}

/* Accent gradient text */
.gradient-text {
  background: var(--accent-gradient);
  -webkit-background-clip: text;
  -webkit-text-fill-color: transparent;
  background-clip: text;
}

/* Panel label style */
.panel-label {
  font-size: 9px;
  color: var(--text-muted);
  text-transform: uppercase;
  letter-spacing: 0.1em;
}

/* Panel meta style */
.panel-meta {
  font-size: 9px;
  color: var(--text-muted);
  opacity: 0.7;
}
```

- [ ] **Step 2: Verify theme loads**

```bash
cd web && npm run dev
```

Open http://localhost:3000 — should see dark green-black background with no errors.

- [ ] **Step 3: Commit**

```bash
git add web/app/styles/global.css
git commit -m "feat(web): add Jade Depths glassmorphic theme tokens"
```

---

### Task 3: Nav Component + Layout

**Files:**
- Create: `web/app/components/layout/Nav.tsx`
- Modify: `web/app/root.tsx`

- [ ] **Step 1: Create Nav.tsx**

```tsx
import { NavLink } from "@remix-run/react";

const links = [
  { to: "/benchmarks", label: "Benchmarks" },
  { to: "/analyzer", label: "Analyzer" },
  { to: "/docs", label: "Docs" },
];

export function Nav() {
  return (
    <nav className="nav">
      <NavLink to="/" className="nav-brand gradient-text">
        mojo-audio
      </NavLink>
      {links.map((link) => (
        <NavLink
          key={link.to}
          to={link.to}
          className={({ isActive }) =>
            `nav-link ${isActive ? "nav-link--active" : ""}`
          }
        >
          {link.label}
        </NavLink>
      ))}
      <a
        href="https://github.com/dev-coffee/mojo-audio"
        target="_blank"
        rel="noopener noreferrer"
        className="nav-link nav-link--github"
      >
        GitHub ↗
      </a>
    </nav>
  );
}
```

- [ ] **Step 2: Add nav styles to global.css**

Append to `web/app/styles/global.css`:

```css
/* Navigation */
.nav {
  height: 48px;
  background: var(--panel-glass-nav);
  backdrop-filter: blur(14px);
  -webkit-backdrop-filter: blur(14px);
  border-bottom: 1px solid var(--panel-border);
  padding: 0 28px;
  display: flex;
  align-items: center;
  gap: 24px;
  font-size: 11px;
  box-shadow: 0 2px 16px rgba(60, 180, 140, 0.03);
}

.nav-brand {
  font-weight: 600;
  font-size: 13px;
  text-decoration: none;
}

.nav-link {
  color: var(--text-muted);
  text-decoration: none;
  transition: color 0.15s;
  padding-bottom: 14px;
  margin-bottom: -14px;
}

.nav-link:hover {
  color: var(--text);
}

.nav-link--active {
  color: var(--text-bright);
  border-bottom: 2px solid;
  border-image: var(--accent-gradient) 1;
}

.nav-link--github {
  margin-left: auto;
  opacity: 0.5;
  font-size: 10px;
}
```

- [ ] **Step 3: Update root.tsx to include Nav**

Add import and render Nav above Outlet:

```tsx
import { Nav } from "./components/layout/Nav";

// Inside App():
<body>
  <Nav />
  <Outlet />
  <ScrollRestoration />
  <Scripts />
</body>
```

- [ ] **Step 4: Create placeholder route for landing page**

Create `web/app/routes/_index.tsx`:

```tsx
export default function Landing() {
  return (
    <div style={{ padding: "40px 28px", textAlign: "center" }}>
      <h1 className="gradient-text" style={{ fontSize: "28px", fontWeight: 300 }}>
        mojo-audio
      </h1>
      <p style={{ color: "var(--text-muted)", marginTop: "8px", fontSize: "12px" }}>
        High-performance audio DSP in Mojo
      </p>
    </div>
  );
}
```

- [ ] **Step 5: Verify nav renders and links work**

```bash
cd web && npm run dev
```

Expected: Nav bar with gradient "mojo-audio" brand, links, glassmorphic background.

- [ ] **Step 6: Commit**

```bash
git add web/app/components/layout/Nav.tsx web/app/styles/global.css web/app/root.tsx web/app/routes/_index.tsx
git commit -m "feat(web): add glassmorphic Nav component + landing placeholder"
```

---

### Task 4: GlassPanel + TabBar UI Primitives

**Files:**
- Create: `web/app/components/ui/GlassPanel.tsx`
- Create: `web/app/components/ui/TabBar.tsx`

- [ ] **Step 1: Create GlassPanel.tsx**

```tsx
import type { ReactNode, CSSProperties } from "react";

interface GlassPanelProps {
  children: ReactNode;
  className?: string;
  style?: CSSProperties;
}

export function GlassPanel({ children, className = "", style }: GlassPanelProps) {
  return (
    <div className={`glass ${className}`} style={style}>
      {children}
    </div>
  );
}
```

- [ ] **Step 2: Create TabBar.tsx**

```tsx
interface Tab {
  id: string;
  label: string;
}

interface TabBarProps {
  tabs: Tab[];
  activeTab: string;
  onTabChange: (id: string) => void;
}

export function TabBar({ tabs, activeTab, onTabChange }: TabBarProps) {
  return (
    <div className="tab-bar">
      {tabs.map((tab) => (
        <button
          key={tab.id}
          className={`tab-bar__tab ${tab.id === activeTab ? "tab-bar__tab--active" : ""}`}
          onClick={() => onTabChange(tab.id)}
        >
          {tab.label}
        </button>
      ))}
    </div>
  );
}
```

- [ ] **Step 3: Add tab-bar styles to global.css**

Append to `web/app/styles/global.css`:

```css
/* Tab bar */
.tab-bar {
  display: flex;
  gap: 0;
  border-bottom: 1px solid var(--panel-border);
}

.tab-bar__tab {
  padding: 9px 14px;
  color: var(--text-muted);
  font-size: 10px;
  font-family: var(--font-mono);
  cursor: pointer;
  transition: color 0.15s;
  letter-spacing: 0.02em;
  background: none;
  border: none;
  border-bottom: 2px solid transparent;
  margin-bottom: -1px;
}

.tab-bar__tab:hover {
  color: var(--text);
}

.tab-bar__tab--active {
  border-bottom: 2px solid;
  border-image: var(--accent-gradient) 1;
}

.tab-bar__tab--active {
  background: var(--accent-gradient);
  -webkit-background-clip: text;
  -webkit-text-fill-color: transparent;
  background-clip: text;
}
```

- [ ] **Step 4: Commit**

```bash
git add web/app/components/ui/
git commit -m "feat(web): add GlassPanel and TabBar UI primitives"
```

---

### Task 5: Add pixi `dev` Task

**Files:**
- Modify: `pixi.toml`

- [ ] **Step 1: Add dev task to pixi.toml**

Add a new task that starts both Remix and FastAPI concurrently. Find the `[tasks]` section in `pixi.toml` and add:

```toml
dev = "cd web && npm run dev & cd ui/backend && python app.py & wait"
```

Note: This uses shell backgrounding to run both processes. `wait` keeps the parent alive until both exit. Ctrl+C will kill both.

- [ ] **Step 2: Test the dev task**

```bash
pixi run dev
```

Expected: Both Remix (port 3000) and FastAPI (port 8000) start. http://localhost:3000 shows the app with nav. http://localhost:3000/analyze (proxied) returns 405 method not allowed (correct — it's POST only).

- [ ] **Step 3: Commit**

```bash
git add pixi.toml
git commit -m "feat: add pixi dev task — starts Remix + FastAPI concurrently"
```

---

## Chunk 2: Analyzer Page — Port Existing 3 Panels

### Task 6: Types + useAnalyzer Hook

**Files:**
- Create: `web/app/lib/types.ts`
- Create: `web/app/lib/use-analyzer.ts`

- [ ] **Step 1: Create types.ts**

```typescript
export interface VadRegion {
  start: number;
  end: number;
}

/** Current /analyze response shape (existing backend) */
export interface AnalyzeResponseLegacy {
  duration_s: number;
  sample_rate: number;
  waveform: number[];
  mel_spectrogram: number[];
  mel_n_mels: number;
  mel_n_frames: number;
  vad_regions: VadRegion[];
}

/** Extended /analyze response (after backend update) */
export interface AnalyzeResponse extends AnalyzeResponseLegacy {
  stft_magnitude: number[];
  stft_n_frames: number;
  stft_n_freq_bins: number;
  rms_energy: number[];
  power_spectrum: number[];
}

export interface AnalyzerState {
  status: "idle" | "loading" | "ready" | "error";
  data: AnalyzeResponse | null;
  error: string | null;
  fileName: string | null;
}
```

- [ ] **Step 2: Create use-analyzer.ts hook**

```typescript
import { useState, useCallback } from "react";
import type { AnalyzerState, AnalyzeResponse } from "./types";

export function useAnalyzer() {
  const [state, setState] = useState<AnalyzerState>({
    status: "idle",
    data: null,
    error: null,
    fileName: null,
  });

  const analyze = useCallback(async (file: File) => {
    setState({ status: "loading", data: null, error: null, fileName: file.name });

    try {
      const formData = new FormData();
      formData.append("file", file);

      const res = await fetch("/analyze", { method: "POST", body: formData });

      if (!res.ok) {
        const err = await res.json().catch(() => ({ detail: res.statusText }));
        throw new Error(err.detail || `HTTP ${res.status}`);
      }

      const data: AnalyzeResponse = await res.json();
      setState({ status: "ready", data, error: null, fileName: file.name });
    } catch (e) {
      setState({
        status: "error",
        data: null,
        error: e instanceof Error ? e.message : String(e),
        fileName: file.name,
      });
    }
  }, []);

  const reset = useCallback(() => {
    setState({ status: "idle", data: null, error: null, fileName: null });
  }, []);

  return { ...state, analyze, reset };
}
```

- [ ] **Step 3: Commit**

```bash
git add web/app/lib/
git commit -m "feat(web): add API types and useAnalyzer hook"
```

---

### Task 7: Shared Viz Utilities (Inferno Colormap + WebGL Helpers)

**Files:**
- Create: `web/app/components/viz/shared/inferno.ts`
- Create: `web/app/components/viz/shared/webgl-utils.ts`

- [ ] **Step 1: Create inferno.ts**

Port the inferno colormap from `visualizer.html` (lines 395-410). This is a 256-entry RGB lookup table:

```typescript
/** Inferno colormap — 256 RGB triplets, values 0-255 */
export const INFERNO_LUT: [number, number, number][] = [];

// Generate inferno colormap using smoothstep approximation
// (matches the GLSL shader in the existing visualizer)
for (let i = 0; i < 256; i++) {
  const t = i / 255;
  const r = Math.round(255 * Math.min(1, Math.max(0,
    -0.0155 + t * (5.3711 + t * (-14.099 + t * (13.457 - t * 4.716))))));
  const g = Math.round(255 * Math.min(1, Math.max(0,
    0.0109 + t * (-0.670 + t * (3.448 + t * (-5.691 + t * 3.889))))));
  const b = Math.round(255 * Math.min(1, Math.max(0,
    0.178 + t * (3.298 + t * (-12.425 + t * (17.326 - t * 8.376))))));
  INFERNO_LUT.push([r, g, b]);
}

/** Map a [0,1] value to inferno RGB */
export function infernoRGB(t: number): [number, number, number] {
  const idx = Math.round(Math.min(1, Math.max(0, t)) * 255);
  return INFERNO_LUT[idx];
}
```

- [ ] **Step 2: Create webgl-utils.ts**

```typescript
/** Compile a WebGL shader, returning null on failure */
export function compileShader(
  gl: WebGLRenderingContext,
  type: number,
  source: string
): WebGLShader | null {
  const shader = gl.createShader(type);
  if (!shader) return null;
  gl.shaderSource(shader, source);
  gl.compileShader(shader);
  if (!gl.getShaderParameter(shader, gl.COMPILE_STATUS)) {
    console.error("Shader compile error:", gl.getShaderInfoLog(shader));
    gl.deleteShader(shader);
    return null;
  }
  return shader;
}

/** Create a WebGL program from vertex + fragment source */
export function createProgram(
  gl: WebGLRenderingContext,
  vertSrc: string,
  fragSrc: string
): WebGLProgram | null {
  const vert = compileShader(gl, gl.VERTEX_SHADER, vertSrc);
  const frag = compileShader(gl, gl.FRAGMENT_SHADER, fragSrc);
  if (!vert || !frag) return null;

  const program = gl.createProgram();
  if (!program) return null;
  gl.attachShader(program, vert);
  gl.attachShader(program, frag);
  gl.linkProgram(program);

  if (!gl.getProgramParameter(program, gl.LINK_STATUS)) {
    console.error("Program link error:", gl.getProgramInfoLog(program));
    gl.deleteProgram(program);
    return null;
  }
  return program;
}

/** Create a fullscreen quad (2 triangles) for texture rendering */
export function createFullscreenQuad(gl: WebGLRenderingContext): WebGLBuffer | null {
  const buf = gl.createBuffer();
  gl.bindBuffer(gl.ARRAY_BUFFER, buf);
  gl.bufferData(
    gl.ARRAY_BUFFER,
    new Float32Array([-1, -1, 1, -1, -1, 1, -1, 1, 1, -1, 1, 1]),
    gl.STATIC_DRAW
  );
  return buf;
}
```

- [ ] **Step 3: Commit**

```bash
git add web/app/components/viz/shared/
git commit -m "feat(web): add inferno colormap and WebGL utilities"
```

---

### Task 8: Waveform Component (Port from visualizer.html)

**Files:**
- Create: `web/app/components/viz/Waveform.tsx`

- [ ] **Step 1: Create Waveform.tsx**

Port the Canvas 2D waveform from `visualizer.html` lines 151-200, adapted to React:

```tsx
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

    // Clear
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

    // Waveform (RMS envelope) — emerald-to-teal gradient
    const grad = ctx.createLinearGradient(0, 0, W, 0);
    grad.addColorStop(0, "rgba(80, 232, 168, 0.7)");
    grad.addColorStop(1, "rgba(56, 200, 180, 0.7)");

    const maxVal = Math.max(...waveform.map(Math.abs), 0.001);

    ctx.fillStyle = grad;
    ctx.beginPath();
    ctx.moveTo(0, midY);

    // Top half
    for (let i = 0; i < waveform.length; i++) {
      const x = (i / waveform.length) * W;
      const y = midY - (waveform[i] / maxVal) * midY * 0.9;
      if (i === 0) ctx.moveTo(x, y);
      else ctx.lineTo(x, y);
    }

    // Bottom half (mirror)
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
```

- [ ] **Step 2: Add waveform styles to analyzer.css**

Create `web/app/styles/analyzer.css`:

```css
/* Waveform panel */
.waveform-panel {
  margin-bottom: 12px;
  overflow: hidden;
}

.waveform-panel__header {
  padding: 7px 14px;
  display: flex;
  align-items: center;
  justify-content: space-between;
  border-bottom: 1px solid var(--panel-border);
}

.waveform-panel__legend {
  display: flex;
  gap: 6px;
  align-items: center;
}

.waveform-panel__vad-dot {
  width: 6px;
  height: 6px;
  border-radius: 50%;
  background: rgba(0, 220, 120, 0.5);
}

.waveform-panel__canvas {
  display: block;
  padding: 4px 14px;
}
```

- [ ] **Step 3: Commit**

```bash
git add web/app/components/viz/Waveform.tsx web/app/styles/analyzer.css
git commit -m "feat(web): add Waveform component — Canvas 2D with VAD overlay"
```

---

### Task 9: MelSpectrogram Component (Port WebGL from visualizer.html)

**Files:**
- Create: `web/app/components/viz/MelSpectrogram.tsx`

- [ ] **Step 1: Create MelSpectrogram.tsx**

Port the WebGL mel spectrogram from `visualizer.html` lines 201-343. Key details:
- Always use Uint8 textures (no OES_texture_float_linear dependency)
- Inferno colormap in GLSL fragment shader
- Canvas 2D fallback if WebGL fails

```tsx
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

        // Uint8 texture — always works
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
```

- [ ] **Step 2: Add viz-content styles to analyzer.css**

Append to `web/app/styles/analyzer.css`:

```css
/* Visualization content area (inside glass panel below tabs) */
.viz-content__header {
  padding: 6px 14px;
  display: flex;
  align-items: center;
  justify-content: space-between;
}

.viz-content__canvas {
  display: block;
  margin: 0 14px 14px;
  border-radius: 4px;
}
```

- [ ] **Step 3: Commit**

```bash
git add web/app/components/viz/MelSpectrogram.tsx web/app/styles/analyzer.css
git commit -m "feat(web): add MelSpectrogram — WebGL with Uint8 texture + Canvas 2D fallback"
```

---

### Task 10: Waterfall3D Component (Port Three.js from visualizer.html)

**Files:**
- Create: `web/app/components/viz/Waterfall3D.tsx`

- [ ] **Step 1: Install Three.js**

```bash
cd web && npm install three && npm install -D @types/three
```

- [ ] **Step 2: Create Waterfall3D.tsx**

Port from `visualizer.html` lines 363-454. Three.js must be imported client-side only in Remix:

```tsx
import { useRef, useEffect } from "react";
import { infernoRGB } from "./shared/inferno";

interface Waterfall3DProps {
  melData: number[];
  nMels: number;
  nFrames: number;
}

export function Waterfall3D({ melData, nMels, nFrames }: Waterfall3DProps) {
  const containerRef = useRef<HTMLDivElement>(null);
  const cleanupRef = useRef<(() => void) | null>(null);

  useEffect(() => {
    if (!containerRef.current || melData.length === 0) return;

    // Dynamic import to avoid SSR issues
    import("three").then((THREE) => {
      const container = containerRef.current;
      if (!container) return;

      // Cleanup previous
      if (cleanupRef.current) cleanupRef.current();

      const W = container.clientWidth;
      const H = 280;

      const scene = new THREE.Scene();
      scene.background = new THREE.Color(0x060d0a);
      scene.fog = new THREE.Fog(0x060d0a, 8, 18);

      const camera = new THREE.PerspectiveCamera(50, W / H, 0.1, 100);
      camera.position.set(0, 4, 7);
      camera.lookAt(0, 0, 0);

      const renderer = new THREE.WebGLRenderer({ antialias: true });
      renderer.setSize(W, H);
      renderer.setPixelRatio(Math.min(window.devicePixelRatio, 2));
      container.appendChild(renderer.domElement);

      // Decimate frames
      const maxFrames = 300;
      const step = Math.max(1, Math.floor(nFrames / maxFrames));
      const usedFrames = Math.floor(nFrames / step);

      // Normalize
      let min = Infinity, max = -Infinity;
      for (const v of melData) {
        if (v < min) min = v;
        if (v > max) max = v;
      }
      const range = max - min || 1;

      // Geometry
      const geo = new THREE.PlaneGeometry(6, 4, usedFrames - 1, nMels - 1);
      const positions = geo.attributes.position;
      const colors = new Float32Array(positions.count * 3);
      const maxH = 1.5;

      for (let j = 0; j < nMels; j++) {
        for (let fi = 0; fi < usedFrames; fi++) {
          const srcFrame = fi * step;
          const idx = srcFrame * nMels + j;
          const val = (melData[idx] - min) / range;
          const vertIdx = j * usedFrames + fi;
          (positions.array as Float32Array)[vertIdx * 3 + 2] = val * maxH;

          const [r, g, b] = infernoRGB(val);
          colors[vertIdx * 3] = r / 255;
          colors[vertIdx * 3 + 1] = g / 255;
          colors[vertIdx * 3 + 2] = b / 255;
        }
      }

      geo.setAttribute("color", new THREE.BufferAttribute(colors, 3));
      geo.computeVertexNormals();
      geo.rotateX(-Math.PI / 2);

      const mat = new THREE.MeshPhongMaterial({
        vertexColors: true,
        side: THREE.DoubleSide,
        shininess: 30,
      });

      scene.add(new THREE.Mesh(geo, mat));
      scene.add(new THREE.AmbientLight(0xffffff, 0.4));
      const dir = new THREE.DirectionalLight(0xffffff, 0.8);
      dir.position.set(2, 5, 3);
      scene.add(dir);

      let angle = 0;
      let animId: number;

      function animate() {
        animId = requestAnimationFrame(animate);
        angle += 0.003;
        camera.position.x = 7 * Math.sin(angle);
        camera.position.z = 7 * Math.cos(angle);
        camera.lookAt(0, 0, 0);
        renderer.render(scene, camera);
      }
      animate();

      cleanupRef.current = () => {
        cancelAnimationFrame(animId);
        renderer.dispose();
        geo.dispose();
        mat.dispose();
        if (container.contains(renderer.domElement)) {
          container.removeChild(renderer.domElement);
        }
      };
    });

    return () => {
      if (cleanupRef.current) cleanupRef.current();
    };
  }, [melData, nMels, nFrames]);

  return (
    <div className="viz-content">
      <div className="viz-content__header">
        <span className="panel-label">3D Waterfall · {nMels} bands</span>
        <span className="panel-meta">auto-rotate</span>
      </div>
      <div ref={containerRef} style={{ width: "100%", height: 280, borderRadius: 4, overflow: "hidden" }} />
    </div>
  );
}
```

- [ ] **Step 3: Commit**

```bash
git add web/app/components/viz/Waterfall3D.tsx
git commit -m "feat(web): add Waterfall3D — Three.js terrain with inferno colors"
```

---

### Task 11: Analyzer Page — Wire Everything Together

**Files:**
- Create: `web/app/routes/analyzer.tsx`
- Modify: `web/app/styles/analyzer.css`

- [ ] **Step 1: Create analyzer.tsx**

```tsx
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
```

- [ ] **Step 2: Add analyzer page styles to analyzer.css**

Append to `web/app/styles/analyzer.css`:

```css
/* Analyzer page */
.analyzer {
  padding: 14px 20px;
  position: relative;
  max-width: 1200px;
  margin: 0 auto;
}

.analyzer__glow--primary {
  position: fixed;
  top: -20%;
  left: -5%;
  width: 45%;
  height: 140%;
  background: radial-gradient(ellipse, var(--glow-primary), transparent 70%);
  pointer-events: none;
  z-index: 0;
}

.analyzer__glow--secondary {
  position: fixed;
  bottom: -20%;
  right: -5%;
  width: 35%;
  height: 120%;
  background: radial-gradient(ellipse, var(--glow-secondary), transparent 70%);
  pointer-events: none;
  z-index: 0;
}

.analyzer > *:not(.analyzer__glow--primary):not(.analyzer__glow--secondary) {
  position: relative;
  z-index: 1;
}

/* Drop zone */
.drop-zone {
  border: 2px dashed var(--panel-border);
  border-radius: 12px;
  padding: 48px;
  text-align: center;
  transition: border-color 0.2s, background 0.2s;
  cursor: pointer;
}

.drop-zone:hover {
  border-color: rgba(80, 200, 150, 0.2);
  background: var(--subtle);
}

.drop-zone__browse {
  color: var(--accent-start);
  cursor: pointer;
  text-decoration: underline;
}

/* File info bar */
.file-info {
  display: flex;
  align-items: center;
  justify-content: space-between;
  padding: 8px 14px;
  background: var(--subtle);
  border: 1px solid var(--panel-border);
  border-radius: 6px;
  margin-bottom: 14px;
  font-size: 11px;
}

.file-info__new {
  padding: 4px 10px;
  border: 1px solid var(--panel-border);
  border-radius: 4px;
  color: var(--text-muted);
  font-size: 10px;
  cursor: pointer;
  background: none;
  font-family: var(--font-mono);
  transition: color 0.15s;
}

.file-info__new:hover {
  color: var(--text);
}

/* Viz panel (below tabs) */
.viz-panel {
  border-top: none;
  border-radius: 0 0 8px 8px;
}
```

- [ ] **Step 3: Test the full analyzer flow**

```bash
cd web && npm run dev
```

1. Open http://localhost:3000/analyzer
2. Verify drop zone appears with glassmorphic styling
3. Start FastAPI backend: `pixi run ui` (in another terminal)
4. Drop a WAV file — should see waveform + mel spectrogram
5. Click "3D Waterfall" tab — should see rotating terrain
6. Other tabs show "Coming soon" placeholder

- [ ] **Step 4: Commit**

```bash
git add web/app/routes/analyzer.tsx web/app/styles/analyzer.css
git commit -m "feat(web): add Analyzer page — drop zone, waveform, tabbed panels"
```

---

## Chunk 3: Extend Backend + 5 New Visualization Panels

### Task 12: Extend /analyze Endpoint

**Files:**
- Modify: `ui/backend/app.py`

- [ ] **Step 1: Extend the Mojo script in _process_audio to compute additional DSP**

In `ui/backend/app.py`, find the `_process_audio` function (around line 258). The current Mojo script computes waveform, mel spectrogram, and VAD. Extend it to also compute:
- STFT magnitude (full resolution)
- RMS energy (per-frame)
- Power spectrum (averaged)

Modify the Mojo code string generated in `_process_audio` to add after the existing mel spectrogram computation:

```python
# After existing mel_spectrogram computation, add STFT and RMS:
mojo_code += """
# --- STFT magnitude ---
var stft_result = stft(samples_16k, n_fft, hop_length)
var n_freq_bins = n_fft // 2 + 1
var stft_n_frames = len(stft_result) // n_freq_bins
print("STFT_SHAPE:", stft_n_frames, n_freq_bins)
# Print magnitude (already computed as part of stft)
var stft_str = String("")
for i in range(len(stft_result)):
    if i > 0:
        stft_str += ","
    stft_str += String(stft_result[i])
print("STFT_DATA:", stft_str)

# --- RMS energy (per-frame) ---
var rms_vals = rms_energy(samples_16k, hop_length)
var rms_str = String("")
for i in range(len(rms_vals)):
    if i > 0:
        rms_str += ","
    rms_str += String(rms_vals[i])
print("RMS_DATA:", rms_str)

# --- Power spectrum (averaged over all frames) ---
var ps = power_spectrum(samples_16k, n_fft)
var ps_str = String("")
for i in range(len(ps)):
    if i > 0:
        ps_str += ","
    ps_str += String(ps[i])
print("POWER_SPECTRUM:", ps_str)
"""
```

Then parse the new output fields from stdout and include them in the JSON response:

```python
# Parse STFT
stft_magnitude = []
stft_n_frames = 0
stft_n_freq_bins = 0
for line in stdout_lines:
    if line.startswith("STFT_SHAPE:"):
        parts = line.split(":")[1].strip().split()
        stft_n_frames = int(parts[0])
        stft_n_freq_bins = int(parts[1])
    elif line.startswith("STFT_DATA:"):
        stft_magnitude = [float(x) for x in line.split(":")[1].strip().split(",") if x]

# Parse RMS
rms_energy_data = []
for line in stdout_lines:
    if line.startswith("RMS_DATA:"):
        rms_energy_data = [float(x) for x in line.split(":")[1].strip().split(",") if x]

# Parse power spectrum
power_spectrum_data = []
for line in stdout_lines:
    if line.startswith("POWER_SPECTRUM:"):
        power_spectrum_data = [float(x) for x in line.split(":")[1].strip().split(",") if x]
```

Add to the return dict:

```python
return {
    # ... existing fields ...
    "stft_magnitude": stft_magnitude,
    "stft_n_frames": stft_n_frames,
    "stft_n_freq_bins": stft_n_freq_bins,
    "rms_energy": rms_energy_data,
    "power_spectrum": power_spectrum_data,
}
```

**Important:** Check `src/audio.mojo` first to confirm the exact function signatures for `stft`, `rms_energy`, and `power_spectrum`. The function names and parameters may differ. Read the file and adjust accordingly.

- [ ] **Step 2: Test the extended endpoint**

```bash
pixi run ui &
curl -X POST http://localhost:8000/analyze -F "file=@test.wav" | python3 -m json.tool | head -20
```

Verify the response includes the new fields: `stft_magnitude`, `stft_n_frames`, `stft_n_freq_bins`, `rms_energy`, `power_spectrum`.

- [ ] **Step 3: Handle NaN/Inf in new fields**

Apply the same `np.nan_to_num` treatment to the new fields as the existing mel data:

```python
import numpy as np

stft_np = np.array(stft_magnitude, dtype=np.float32)
np.nan_to_num(stft_np, copy=False, nan=0.0, posinf=0.0, neginf=0.0)
stft_magnitude = stft_np.tolist()

rms_np = np.array(rms_energy_data, dtype=np.float32)
np.nan_to_num(rms_np, copy=False, nan=0.0, posinf=0.0, neginf=0.0)
rms_energy_data = rms_np.tolist()

ps_np = np.array(power_spectrum_data, dtype=np.float32)
np.nan_to_num(ps_np, copy=False, nan=0.0, posinf=0.0, neginf=0.0)
power_spectrum_data = ps_np.tolist()
```

- [ ] **Step 4: Commit**

```bash
git add ui/backend/app.py
git commit -m "feat(api): extend /analyze — add STFT magnitude, RMS energy, power spectrum"
```

---

### Task 13: LinearSTFT Component (WebGL)

**Files:**
- Create: `web/app/components/viz/LinearSTFT.tsx`

- [ ] **Step 1: Create LinearSTFT.tsx**

Nearly identical to MelSpectrogram but with linear frequency axis and different dimensions:

```tsx
import { useRef, useEffect } from "react";
import { createProgram, createFullscreenQuad } from "./shared/webgl-utils";
import { infernoRGB } from "./shared/inferno";

interface LinearSTFTProps {
  stftData: number[];
  nFrames: number;
  nFreqBins: number;
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

export function LinearSTFT({ stftData, nFrames, nFreqBins }: LinearSTFTProps) {
  const canvasRef = useRef<HTMLCanvasElement>(null);

  useEffect(() => {
    const canvas = canvasRef.current;
    if (!canvas || stftData.length === 0) return;

    const dpr = Math.min(window.devicePixelRatio, 2);
    const rect = canvas.getBoundingClientRect();
    canvas.width = rect.width * dpr;
    canvas.height = rect.height * dpr;

    // Log-scale the magnitude for better visibility
    let min = Infinity, max = -Infinity;
    const logData = stftData.map((v) => {
      const lv = Math.log1p(Math.max(0, v));
      if (lv < min) min = lv;
      if (lv > max) max = lv;
      return lv;
    });

    const range = max - min || 1;
    const data8 = new Uint8Array(logData.length);
    for (let i = 0; i < logData.length; i++) {
      data8[i] = Math.round(((logData[i] - min) / range) * 255);
    }

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
          gl.TEXTURE_2D, 0, gl.LUMINANCE, nFrames, nFreqBins, 0,
          gl.LUMINANCE, gl.UNSIGNED_BYTE, data8
        );

        gl.drawArrays(gl.TRIANGLES, 0, 6);
        return;
      }
    }

    // Canvas 2D fallback
    const ctx = canvas.getContext("2d");
    if (!ctx) return;
    const imgData = ctx.createImageData(nFrames, nFreqBins);
    for (let i = 0; i < data8.length; i++) {
      const [r, g, b] = infernoRGB(data8[i] / 255);
      imgData.data[i * 4] = r;
      imgData.data[i * 4 + 1] = g;
      imgData.data[i * 4 + 2] = b;
      imgData.data[i * 4 + 3] = 255;
    }
    ctx.putImageData(imgData, 0, 0);
  }, [stftData, nFrames, nFreqBins]);

  return (
    <div className="viz-content">
      <div className="viz-content__header">
        <span className="panel-label">Linear STFT · {nFreqBins} bins</span>
        <span className="panel-meta">n_fft=400 · hop=160</span>
      </div>
      <canvas ref={canvasRef} className="viz-content__canvas" style={{ width: "100%", height: 280 }} />
    </div>
  );
}
```

- [ ] **Step 2: Commit**

```bash
git add web/app/components/viz/LinearSTFT.tsx
git commit -m "feat(web): add LinearSTFT — WebGL linear frequency spectrogram"
```

---

### Task 14: Chromagram Component (Canvas 2D)

**Files:**
- Create: `web/app/components/viz/Chromagram.tsx`

- [ ] **Step 1: Create Chromagram.tsx**

Chromagram is computed client-side from STFT magnitude by folding into 12 pitch classes:

```tsx
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

    const cellW = W / nFrames;
    const cellH = (H - 16) / 12; // leave room for labels
    const labelW = 24;

    ctx.clearRect(0, 0, W, H);

    // Draw pitch class labels
    ctx.font = "8px 'Fira Code', monospace";
    ctx.textAlign = "right";
    ctx.textBaseline = "middle";
    for (let pc = 0; pc < 12; pc++) {
      const y = (11 - pc) * cellH + cellH / 2;
      ctx.fillStyle = "var(--text-muted)";
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
```

- [ ] **Step 2: Commit**

```bash
git add web/app/components/viz/Chromagram.tsx
git commit -m "feat(web): add Chromagram — client-side pitch class folding from STFT"
```

---

### Task 15: RMSEnergy Component (Canvas 2D)

**Files:**
- Create: `web/app/components/viz/RMSEnergy.tsx`

- [ ] **Step 1: Create RMSEnergy.tsx**

```tsx
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
```

- [ ] **Step 2: Commit**

```bash
git add web/app/components/viz/RMSEnergy.tsx
git commit -m "feat(web): add RMSEnergy — Canvas 2D with dB scale and gradient fill"
```

---

### Task 16: SpectralCentroid Component (Canvas 2D)

**Files:**
- Create: `web/app/components/viz/SpectralCentroid.tsx`

- [ ] **Step 1: Create SpectralCentroid.tsx**

Spectral centroid is computed client-side from STFT magnitude:

```tsx
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
```

- [ ] **Step 2: Commit**

```bash
git add web/app/components/viz/SpectralCentroid.tsx
git commit -m "feat(web): add SpectralCentroid — brightness line over time"
```

---

### Task 17: FrequencyDist Component (Canvas 2D)

**Files:**
- Create: `web/app/components/viz/FrequencyDist.tsx`

- [ ] **Step 1: Create FrequencyDist.tsx**

```tsx
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
    const barW = plotW / powerSpectrum.length;
    const grad = ctx.createLinearGradient(padLeft, 4, padLeft + plotW, 4);
    grad.addColorStop(0, "rgba(80, 232, 168, 0.7)");
    grad.addColorStop(1, "rgba(56, 200, 180, 0.7)");
    ctx.fillStyle = grad;

    for (let i = 0; i < dbData.length; i++) {
      const normDb = Math.max(0, (dbData[i] - minDb) / (0 - minDb));
      const barH = normDb * plotH;
      const x = padLeft + (i / dbData.length) * plotW;
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
```

- [ ] **Step 2: Commit**

```bash
git add web/app/components/viz/FrequencyDist.tsx
git commit -m "feat(web): add FrequencyDist — averaged power spectrum bar chart"
```

---

### Task 18: Wire All 8 Panels in Analyzer Page

**Files:**
- Modify: `web/app/routes/analyzer.tsx`
- Modify: `web/app/lib/types.ts`

- [ ] **Step 1: Update types.ts if needed**

Ensure `AnalyzeResponse` includes all new fields (should already be there from Task 6).

- [ ] **Step 2: Update analyzer.tsx to render all 8 panels**

Replace the `renderVizPanel` function to use the real components:

```tsx
// Add imports at top
import { LinearSTFT } from "../components/viz/LinearSTFT";
import { Chromagram } from "../components/viz/Chromagram";
import { RMSEnergy } from "../components/viz/RMSEnergy";
import { SpectralCentroid } from "../components/viz/SpectralCentroid";
import { FrequencyDist } from "../components/viz/FrequencyDist";

// Replace renderVizPanel:
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
    case "stft":
      return (
        <LinearSTFT
          stftData={data.stft_magnitude}
          nFrames={data.stft_n_frames}
          nFreqBins={data.stft_n_freq_bins}
        />
      );
    case "chroma":
      return (
        <Chromagram
          stftData={data.stft_magnitude}
          nFrames={data.stft_n_frames}
          nFreqBins={data.stft_n_freq_bins}
          sampleRate={data.sample_rate}
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
    case "rms":
      return <RMSEnergy rmsData={data.rms_energy} duration={data.duration_s} />;
    case "centroid":
      return (
        <SpectralCentroid
          stftData={data.stft_magnitude}
          nFrames={data.stft_n_frames}
          nFreqBins={data.stft_n_freq_bins}
          sampleRate={data.sample_rate}
          duration={data.duration_s}
        />
      );
    case "freq":
      return (
        <FrequencyDist
          powerSpectrum={data.power_spectrum}
          sampleRate={data.sample_rate}
        />
      );
    default:
      return null;
  }
};
```

- [ ] **Step 3: Test all 8 panels end-to-end**

```bash
pixi run dev
```

1. Open http://localhost:3000/analyzer
2. Drop a WAV file
3. Click through each tab and verify rendering
4. Check browser console for errors

- [ ] **Step 4: Commit**

```bash
git add web/app/routes/analyzer.tsx web/app/lib/types.ts
git commit -m "feat(web): wire all 8 visualization panels in analyzer page"
```

---

## Chunk 4: Benchmarks Page + Landing Page

### Task 19: Benchmark API Client

**Files:**
- Create: `web/app/lib/benchmark-api.ts`

- [ ] **Step 1: Create benchmark-api.ts**

Port from `ui/static/js/benchmark-api.js` and `ui/static/js/config.js`:

```typescript
export interface BenchmarkConfig {
  duration: number;
  n_fft: number;
  hop_length: number;
  n_mels: number;
  iterations: number;
  blas_backend: "mkl" | "openblas";
}

export interface BenchmarkResult {
  implementation: string;
  duration: number;
  avg_time_ms: number;
  std_time_ms: number;
  throughput_realtime: number;
  iterations: number;
  success: boolean;
  error: string;
}

export interface ComparisonResult {
  mojo: BenchmarkResult;
  librosa: BenchmarkResult;
  speedup_factor: number;
  faster_percentage: number;
  mojo_is_faster: boolean;
}

export const DEFAULT_CONFIG: BenchmarkConfig = {
  duration: 30,
  n_fft: 400,
  hop_length: 160,
  n_mels: 80,
  iterations: 20,
  blas_backend: "mkl",
};

export async function runComparison(config: BenchmarkConfig): Promise<ComparisonResult> {
  const res = await fetch("/api/benchmark/both", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(config),
  });
  if (!res.ok) {
    const err = await res.json().catch(() => ({ detail: res.statusText }));
    throw new Error(err.detail || `HTTP ${res.status}`);
  }
  return res.json();
}
```

- [ ] **Step 2: Commit**

```bash
git add web/app/lib/benchmark-api.ts
git commit -m "feat(web): add benchmark API client"
```

---

### Task 20: Benchmarks Page

**Files:**
- Create: `web/app/routes/benchmarks.tsx`
- Create: `web/app/components/benchmark/ConfigCard.tsx`
- Create: `web/app/components/benchmark/ResultCard.tsx`
- Create: `web/app/styles/benchmarks.css`

- [ ] **Step 1: Create ConfigCard.tsx**

Port the configuration form from `ui/frontend/index.html` lines 1-94:

```tsx
import type { BenchmarkConfig } from "../../lib/benchmark-api";

interface ConfigCardProps {
  config: BenchmarkConfig;
  onChange: (config: BenchmarkConfig) => void;
  onRun: () => void;
  isRunning: boolean;
}

export function ConfigCard({ config, onChange, onRun, isRunning }: ConfigCardProps) {
  return (
    <div className="glass config-card">
      <div className="config-card__header">
        <span className="panel-label">Configuration</span>
      </div>
      <div className="config-card__body">
        <div className="config-card__row">
          <label className="config-card__label">Duration</label>
          <div className="config-card__options">
            {[1, 10, 30].map((d) => (
              <button
                key={d}
                className={`config-card__option ${config.duration === d ? "config-card__option--active" : ""}`}
                onClick={() => onChange({ ...config, duration: d })}
              >
                {d}s
              </button>
            ))}
          </div>
        </div>
        <div className="config-card__row">
          <label className="config-card__label">FFT Size</label>
          <div className="config-card__options">
            {[256, 400, 512, 1024].map((n) => (
              <button
                key={n}
                className={`config-card__option ${config.n_fft === n ? "config-card__option--active" : ""}`}
                onClick={() => onChange({ ...config, n_fft: n, hop_length: Math.floor(n / 2.5) })}
              >
                {n}
              </button>
            ))}
          </div>
        </div>
        <div className="config-card__row">
          <label className="config-card__label">Iterations</label>
          <div className="config-card__options">
            <button
              className="config-card__option"
              onClick={() => onChange({ ...config, iterations: Math.max(1, config.iterations - 5) })}
            >
              −
            </button>
            <span style={{ minWidth: 32, textAlign: "center" }}>{config.iterations}</span>
            <button
              className="config-card__option"
              onClick={() => onChange({ ...config, iterations: Math.min(100, config.iterations + 5) })}
            >
              +
            </button>
          </div>
        </div>
        <div className="config-card__row">
          <label className="config-card__label">BLAS</label>
          <div className="config-card__options">
            {(["mkl", "openblas"] as const).map((b) => (
              <button
                key={b}
                className={`config-card__option ${config.blas_backend === b ? "config-card__option--active" : ""}`}
                onClick={() => onChange({ ...config, blas_backend: b })}
              >
                {b.toUpperCase()}
              </button>
            ))}
          </div>
        </div>
        <button
          className="config-card__run"
          onClick={onRun}
          disabled={isRunning}
        >
          {isRunning ? "Running..." : "Run Benchmark"}
        </button>
      </div>
    </div>
  );
}
```

- [ ] **Step 2: Create ResultCard.tsx**

```tsx
import type { ComparisonResult } from "../../lib/benchmark-api";

interface ResultCardProps {
  result: ComparisonResult;
}

export function ResultCard({ result }: ResultCardProps) {
  const { mojo, librosa, speedup_factor } = result;

  return (
    <div className="glass result-card">
      <div className="result-card__hero">
        <span className="gradient-text" style={{ fontSize: 48, fontWeight: 300 }}>
          {speedup_factor.toFixed(1)}×
        </span>
        <span className="panel-meta" style={{ marginTop: 4 }}>faster</span>
      </div>
      <div className="result-card__comparison">
        <div className="result-card__impl">
          <span className="panel-label">librosa ({librosa.implementation.split("(")[1]?.replace(")", "") || ""})</span>
          <span className="result-card__time">{librosa.avg_time_ms.toFixed(1)}ms</span>
          <div className="result-card__bar">
            <div className="result-card__bar-fill result-card__bar-fill--librosa" style={{ width: "100%" }} />
          </div>
        </div>
        <div className="result-card__impl">
          <span className="panel-label">mojo-audio</span>
          <span className="result-card__time gradient-text">{mojo.avg_time_ms.toFixed(1)}ms</span>
          <div className="result-card__bar">
            <div
              className="result-card__bar-fill result-card__bar-fill--mojo"
              style={{ width: `${(mojo.avg_time_ms / librosa.avg_time_ms) * 100}%` }}
            />
          </div>
        </div>
      </div>
    </div>
  );
}
```

- [ ] **Step 3: Create benchmarks.tsx route**

```tsx
import { useState } from "react";
import { ConfigCard } from "../components/benchmark/ConfigCard";
import { ResultCard } from "../components/benchmark/ResultCard";
import {
  DEFAULT_CONFIG,
  runComparison,
  type BenchmarkConfig,
  type ComparisonResult,
} from "../lib/benchmark-api";
import "../styles/benchmarks.css";

export default function Benchmarks() {
  const [config, setConfig] = useState<BenchmarkConfig>(DEFAULT_CONFIG);
  const [result, setResult] = useState<ComparisonResult | null>(null);
  const [isRunning, setIsRunning] = useState(false);
  const [error, setError] = useState<string | null>(null);

  async function handleRun() {
    setIsRunning(true);
    setError(null);
    try {
      const res = await runComparison(config);
      setResult(res);
    } catch (e) {
      setError(e instanceof Error ? e.message : String(e));
    } finally {
      setIsRunning(false);
    }
  }

  return (
    <div className="benchmarks">
      <div className="benchmarks__header">
        <h1 className="gradient-text" style={{ fontSize: 20, fontWeight: 300 }}>
          Benchmarks
        </h1>
        <p className="panel-meta" style={{ marginTop: 4 }}>
          Compare mojo-audio mel spectrogram against librosa
        </p>
      </div>

      <ConfigCard config={config} onChange={setConfig} onRun={handleRun} isRunning={isRunning} />

      {error && (
        <div style={{ padding: 16, color: "#ff6060", textAlign: "center" }}>
          Error: {error}
        </div>
      )}

      {result && <ResultCard result={result} />}
    </div>
  );
}
```

- [ ] **Step 4: Create benchmarks.css**

```css
.benchmarks {
  padding: 24px 28px;
  max-width: 800px;
  margin: 0 auto;
}

.benchmarks__header {
  text-align: center;
  margin-bottom: 24px;
}

/* Config card */
.config-card__header {
  padding: 10px 16px;
  border-bottom: 1px solid var(--panel-border);
}

.config-card__body {
  padding: 16px;
  display: flex;
  flex-direction: column;
  gap: 12px;
}

.config-card__row {
  display: flex;
  align-items: center;
  justify-content: space-between;
}

.config-card__label {
  font-size: 10px;
  color: var(--text-muted);
  text-transform: uppercase;
  letter-spacing: 0.08em;
}

.config-card__options {
  display: flex;
  gap: 6px;
  align-items: center;
}

.config-card__option {
  padding: 4px 12px;
  border: 1px solid var(--panel-border);
  border-radius: 4px;
  background: none;
  color: var(--text-muted);
  font-family: var(--font-mono);
  font-size: 10px;
  cursor: pointer;
  transition: all 0.15s;
}

.config-card__option:hover {
  color: var(--text);
  border-color: rgba(80, 200, 150, 0.15);
}

.config-card__option--active {
  color: var(--text-bright);
  border-color: var(--accent-start);
  background: var(--subtle);
}

.config-card__run {
  margin-top: 8px;
  padding: 10px 20px;
  background: var(--accent-gradient);
  color: #060d0a;
  border: none;
  border-radius: 6px;
  font-family: var(--font-mono);
  font-weight: 600;
  font-size: 12px;
  cursor: pointer;
  transition: opacity 0.15s;
}

.config-card__run:hover {
  opacity: 0.9;
}

.config-card__run:disabled {
  opacity: 0.5;
  cursor: not-allowed;
}

/* Result card */
.result-card {
  margin-top: 20px;
  overflow: hidden;
}

.result-card__hero {
  display: flex;
  flex-direction: column;
  align-items: center;
  padding: 24px;
  border-bottom: 1px solid var(--panel-border);
}

.result-card__comparison {
  padding: 16px;
  display: flex;
  flex-direction: column;
  gap: 16px;
}

.result-card__impl {
  display: flex;
  flex-direction: column;
  gap: 4px;
}

.result-card__time {
  font-size: 18px;
  font-weight: 300;
}

.result-card__bar {
  height: 6px;
  background: rgba(80, 200, 150, 0.06);
  border-radius: 3px;
  overflow: hidden;
}

.result-card__bar-fill {
  height: 100%;
  border-radius: 3px;
  transition: width 0.5s;
}

.result-card__bar-fill--librosa {
  background: var(--text-muted);
}

.result-card__bar-fill--mojo {
  background: var(--accent-gradient);
}
```

- [ ] **Step 5: Test benchmarks page**

```bash
pixi run dev
```

Open http://localhost:3000/benchmarks — verify config card renders with Jade Depths theme. Click "Run Benchmark" (requires backend running) to verify results display.

- [ ] **Step 6: Commit**

```bash
git add web/app/routes/benchmarks.tsx web/app/components/benchmark/ web/app/styles/benchmarks.css web/app/lib/benchmark-api.ts
git commit -m "feat(web): add Benchmarks page — config card, results display, Jade Depths theme"
```

---

### Task 21: Landing Page

**Files:**
- Modify: `web/app/routes/_index.tsx`

- [ ] **Step 1: Replace placeholder with full landing page**

```tsx
import { Link } from "@remix-run/react";

export default function Landing() {
  return (
    <div className="landing">
      {/* Glow blobs */}
      <div className="landing__glow landing__glow--primary" />
      <div className="landing__glow landing__glow--secondary" />

      <div className="landing__hero">
        <h1 className="gradient-text" style={{ fontSize: 36, fontWeight: 300, letterSpacing: "-0.5px" }}>
          mojo-audio
        </h1>
        <p style={{ color: "var(--text-muted)", fontSize: 13, marginTop: 8 }}>
          High-performance audio DSP in Mojo
        </p>
        <div className="landing__stat">
          <span className="gradient-text" style={{ fontWeight: 600 }}>20-40%</span>
          <span style={{ color: "var(--text)" }}> faster than librosa</span>
        </div>
        <div className="landing__ctas">
          <Link to="/analyzer" className="landing__cta landing__cta--primary">
            Try Analyzer →
          </Link>
          <Link to="/docs" className="landing__cta landing__cta--secondary">
            View Docs
          </Link>
        </div>
      </div>
    </div>
  );
}
```

- [ ] **Step 2: Add landing styles to global.css**

Append to `web/app/styles/global.css`:

```css
/* Landing page */
.landing {
  position: relative;
  min-height: calc(100vh - 48px);
  display: flex;
  align-items: center;
  justify-content: center;
}

.landing__glow--primary {
  position: fixed;
  top: 10%;
  left: 20%;
  width: 40%;
  height: 60%;
  background: radial-gradient(ellipse, var(--glow-primary), transparent 70%);
  pointer-events: none;
}

.landing__glow--secondary {
  position: fixed;
  bottom: 10%;
  right: 15%;
  width: 30%;
  height: 50%;
  background: radial-gradient(ellipse, var(--glow-secondary), transparent 70%);
  pointer-events: none;
}

.landing__hero {
  text-align: center;
  position: relative;
  z-index: 1;
}

.landing__stat {
  margin-top: 16px;
  font-size: 13px;
  padding: 6px 16px;
  background: var(--subtle);
  border: 1px solid var(--panel-border);
  border-radius: 6px;
  display: inline-block;
}

.landing__ctas {
  margin-top: 24px;
  display: flex;
  gap: 12px;
  justify-content: center;
}

.landing__cta {
  padding: 8px 20px;
  border-radius: 6px;
  font-family: var(--font-mono);
  font-size: 11px;
  font-weight: 600;
  text-decoration: none;
  transition: opacity 0.15s;
}

.landing__cta:hover {
  opacity: 0.85;
}

.landing__cta--primary {
  background: var(--accent-gradient);
  color: #060d0a;
}

.landing__cta--secondary {
  border: 1px solid var(--panel-border);
  color: var(--accent-start);
}
```

- [ ] **Step 3: Commit**

```bash
git add web/app/routes/_index.tsx web/app/styles/global.css
git commit -m "feat(web): add Landing page — hero, stat badge, CTAs"
```

---

### Task 22: Docs Page (Stub)

**Files:**
- Create: `web/app/routes/docs.tsx`

- [ ] **Step 1: Create docs stub**

```tsx
export default function Docs() {
  return (
    <div style={{ padding: "40px 28px", maxWidth: 800, margin: "0 auto" }}>
      <h1 className="gradient-text" style={{ fontSize: 20, fontWeight: 300 }}>
        Documentation
      </h1>
      <p style={{ color: "var(--text-muted)", marginTop: 8, fontSize: 12 }}>
        Coming soon — guides, API reference, and examples.
      </p>
    </div>
  );
}
```

- [ ] **Step 2: Commit**

```bash
git add web/app/routes/docs.tsx
git commit -m "feat(web): add Docs page stub"
```

---

### Task 23: Final Integration Test + Cleanup

**Files:**
- All files in `web/`

- [ ] **Step 1: Run full dev stack**

```bash
pixi run dev
```

- [ ] **Step 2: Verify all pages**

| Page | URL | Expected |
|------|-----|----------|
| Landing | http://localhost:3000/ | Hero, gradient text, CTAs, glow blobs |
| Benchmarks | http://localhost:3000/benchmarks | Config card, run button, results |
| Analyzer | http://localhost:3000/analyzer | Drop zone, all 8 panels work |
| Docs | http://localhost:3000/docs | Stub page |

- [ ] **Step 3: Verify nav links work on every page**

Click through all nav links. Active link should have gradient underline.

- [ ] **Step 4: Check for console errors**

Open DevTools console on each page. Fix any errors.

- [ ] **Step 5: Commit any fixes**

```bash
git add -A web/
git commit -m "fix(web): integration test cleanup"
```
