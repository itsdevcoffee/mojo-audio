# Sprint 5 Handoff — Shade Integration Complete

> For a coding agent picking up post-Sprint 5 work on mojo-audio.
> Read this fully before starting.

**Date:** 2026-04-09
**Branch:** `main`
**Working directory:** `/home/maskkiller/dev-coffee/repos/mojo-audio`

---

## What's Done — Sprints 1-5 Complete

The full RVC v2 voice conversion pipeline runs end-to-end on DGX Spark via MAX Engine, zero PyTorch CUDA at inference time. Shade (`app.tryshade.io`) is live with mojo-audio as the engine.

### Sprint 1: AudioEncoder (HuBERT/ContentVec) + PitchExtractor (RMVPE)
### Sprint 2: Phase vocoder, batch>1, pos_conv GPU, kernel cache
### Sprint 3: NSF-HiFiGAN vocoder (im2col workaround for all conv2d bugs)
### Sprint 4: VITS synthesis (enc_p TextEncoder + normalizing flow + VoiceConverter)
### Sprint 5: Shade integration (services/rvc.py swap, live on tryshade.io)

---

## Current Architecture

```
Audio in → AudioEncoder (HuBERT) → ContentVec features [B, T, 768]
Audio in → PitchExtractor (RMVPE) → F0 [B, T] Hz
         ↓
    enc_p (TextEncoder, 6-layer transformer with relative positional attention)
         → prior mean/logvar [B, 192, T]
         ↓
    Sample z_p (reparameterization)
         ↓
    flow (4-stage reverse normalizing flow with WaveNet + speaker conditioning)
         → latents z [B, 192, T]
         ↓
    HiFiGAN (NSF vocoder) + F0 → audio waveform
```

All MAX Graph ops use im2col + matmul (no ops.conv2d — it's buggy for C_in >= 8).

---

## Key Files

| File | Purpose |
|------|---------|
| `src/models/voice_converter.py` | Public API: `VoiceConverter.from_pretrained(ckpt).convert(audio)` |
| `src/models/_vits_graph.py` | MAX graphs: enc_p + flow (WaveNet, coupling layers, flip via permutation matrix) |
| `src/models/_vits_weight_loader.py` | Extract enc_p/flow/emb_g weights, bake speaker cond into HiFiGAN |
| `src/models/hifigan.py` | NSFHiFiGAN: latents + F0 → audio |
| `src/models/_hifigan_graph.py` | HiFiGAN MAX graph (conv1d im2col, conv_transpose, noise_conv) |
| `src/models/audio_encoder.py` | AudioEncoder (HuBERT/ContentVec) |
| `src/models/pitch_extractor.py` | PitchExtractor (RMVPE) |
| `tests/test_vits.py` | 50 tests (weight loader + flow + enc_p + speaker cond + orchestration) |
| `tests/test_hifigan.py` | 16 tests (15 pass, 1 xfail batch>1) |
| `tests/_rvc_pytorch_reference.py` | PyTorch reference for numerical comparison |

---

## Shade Deployment (on Spark)

Shade is at `/home/visage/repos/shade/` on `visage@visage-spark`.

**Current deployment (mojo-audio, non-Docker):**
```bash
# API server (port 8000) — run via pixi for MAX Engine env
cd /home/visage/repos/mojo-audio
MODELS_DIR=/home/visage/repos/shade/models AUDIO_UPLOAD_DIR=/tmp/shade-uploads \
  ~/.pixi/bin/pixi run python -c '
import sys
sys.path.insert(0, "/home/visage/repos/shade/api")
sys.path.insert(0, "/home/visage/repos/mojo-audio/src")
import os; os.chdir("/home/visage/repos/shade/api")
import uvicorn
uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=False)
'

# Frontend (port 3100) — SvelteKit
cd /home/visage/repos/shade/web
PORT=3100 node build/index.js
```

**To roll back to Applio:**
```bash
kill $(lsof -ti:8000) $(lsof -ti:3100)
cd /home/visage/repos/shade && docker compose up -d
```

**Cloudflare tunnel:** `app.tryshade.io` → `localhost:3100` (frontend). Frontend calls API at `localhost:8000`.

**Known issue:** `nohup` processes don't survive reboots. Needs systemd services or similar for persistence.

---

## Key Findings / MAX Engine Workarounds

| Issue | Workaround |
|-------|-----------|
| `ops.conv2d` wrong for C_in >= 8 | im2col + matmul (filed: modular/modular#6248, no response) |
| Multiple conv2d with different strides crashes compiler | im2col (unfiled) |
| `ops.rebind` overwrites static dimensions | Explicit shape: `[x.shape[0], x.shape[1], 1, C_out]` |
| No `ops.conv_transpose` | Zero-interleave + im2col |
| flip_channels OOM (192 slice ops per flip) | Permutation matrix matmul (3 ops per flip) |
| Relative position attention with symbolic T | Precompute biases in numpy, pass as graph inputs |
| Graph compilation OOM on 32GB machine | Class-scoped fixtures, minimize compilations, test on Spark |

---

## Benchmarks (DGX Spark CPU, 2s input)

| Metric | Value |
|--------|-------|
| Cold load (first request) | ~380s (JIT compilation of 5 MAX graphs) |
| Warm inference | 1.26s |
| RTF (warm) | 0.63x |
| Output SR | 48kHz |

---

## Voice Models on Spark

23 models in `/home/visage/repos/shade/models/` including: The Weeknd, Ariana Grande, Travis Scott, Bad Bunny, SZA, Frank Sinatra, Freddie Mercury, Patrick Star, SpongeBob, etc.

Test checkpoint: `/home/visage/Downloads/voice files/extracted/theweeknd biggest data set/theweekv1.pth`

---

## What's NOT Done

### Sprint 4 remaining items
- [ ] Full end-to-end integration test loading all 5 models simultaneously (works manually, no automated test — needs Spark's 121GB RAM)
- [ ] GPU inference (CPU verified, GPU not tested)

### Sprint 5 remaining items
- [ ] Systemd services for persistent deployment (currently nohup)
- [ ] Docker image with MAX Engine (for proper containerized deployment)
- [ ] `/clean` and `/separate` endpoints need `noisereduce` and `audio-separator` installed in pixi env (pip installed them but also upgraded torch to 2.11.0 — may cause issues)

### Future roadmap items
- [ ] FAISS index retrieval (quality improvement, CPU-side bolt-on)
- [ ] Pitch protection blending
- [ ] GPU inference optimization
- [ ] Relative position attention computed in-graph (currently numpy pre-pass duplicates encoder work)
- [ ] BigVGAN vocoder swap (potential quality upgrade)
- [ ] Seed-VC exploration (zero-shot voice conversion)
- [ ] Sprint 6: Multi-Spark NVLink parallelism
- [ ] File 2 unfiled MAX bugs (multi-stride conv2d, rebind behavior)
- [ ] Blog: "Every conv2d Bug We Hit in MAX Engine"

---

## Environment

```bash
# Local machine
cd /home/maskkiller/dev-coffee/repos/mojo-audio
pixi run test-vits           # 50 tests (skip real-checkpoint tests locally — OOM risk)
pixi run test-hifigan        # 16 tests
pixi run test-models         # 34 tests

# DGX Spark (121GB RAM — safe for all tests)
ssh visage@visage-spark
cd /home/visage/repos/mojo-audio
git pull && ~/.pixi/bin/pixi run python -m pytest tests/test_vits.py -v
```

MAX version: `26.3.0.dev2026032005`
Applio: `/home/visage/repos/Applio` (Spark), `/home/maskkiller/repos/Applio` (local)

---

## What NOT to Do

- Do not use `ops.conv2d` — it's broken for C_in >= 8. Use im2col + matmul from `_hifigan_graph.py`
- Do not create graph nodes in loops over channels (OOM). Use matmul-based approaches
- Do not run graph compilation tests locally if they compile multiple graphs (32GB RAM limit). Test on Spark
- Do not change the VoiceConverter.convert() public API
- Do not change the Shade `/convert` endpoint signature (frontend depends on it)
- Do not touch the training pipeline (stays on Applio/Celery)
