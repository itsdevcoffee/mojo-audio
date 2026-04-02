# Shade Integration — Sprint 5 Design Spec

**Date:** 2026-04-02
**Sprint:** 5 — "Shade on Spark"
**Goal:** Replace Applio-based voice conversion in Shade's FastAPI server with mojo-audio's VoiceConverter. Same API, different engine.

---

## 1. Current State

Shade is a FastAPI server at `/home/visage/repos/shade/api/` on the DGX Spark. It has:

- `POST /convert` — voice conversion (currently wraps Applio)
- `POST /separate` — vocal isolation (audio-separator, unchanged)
- `POST /clean` — recording cleanup (unchanged)
- `POST /train` — voice model training (Celery + Applio, unchanged)
- `GET /models` — list available voice models
- `GET /health` — server status

The conversion endpoint calls `services/rvc.py`, which is a thin wrapper around Applio's `VoiceConverter`. Applio requires PyTorch CUDA, a full Applio installation, and an `os.chdir()` hack at import time.

## 2. What Changes

### 2.1 `services/rvc.py` — Engine Swap

Replace the Applio wrapper with a mojo-audio wrapper. The `convert()` function signature stays identical.

**Lazy loading with caching:**
- On first `/convert` request, load `VoiceConverter.from_pretrained()` (~6 min JIT compilation)
- Cache the instance in a module-level dict keyed by model path
- Subsequent requests with the same model are instant (~8s inference)
- Different models trigger a new load (but previous models stay cached)

**Applio fallback:**
- `USE_APPLIO=1` env var switches back to old Applio path
- Default: mojo-audio
- This allows A/B quality comparison during transition

**Parameter mapping:**

| Shade API param | mojo-audio mapping |
|---|---|
| `model_path` | `VoiceConverter.from_pretrained(model_path)` |
| `pitch` | `converter.convert(audio, pitch_shift=pitch)` |
| `pitch_extraction` | Ignored (always RMVPE). Log warning if not "rmvpe". |
| `index_path` | Ignored (FAISS not implemented yet). Accepted for API compat. |
| `index_ratio` | Ignored. |

### 2.2 mojo-audio Import Path

On Spark, mojo-audio is at `/home/visage/repos/mojo-audio/src/`. The service adds this to `sys.path`:

```python
MOJO_AUDIO_PATH = os.environ.get("MOJO_AUDIO_PATH", "/home/visage/repos/mojo-audio/src")
```

No package installation needed — just a path addition, matching the pattern already used for Applio.

### 2.3 Audio I/O

The Shade endpoint receives an uploaded WAV file and expects a WAV file back.

- **Input:** Read WAV from temp file path → numpy array via `soundfile` or `scipy.io.wavfile`
- **Output:** `VoiceConverter.convert()` returns `[B, T_audio]` numpy → write to WAV via `soundfile`

The current Applio wrapper handles file I/O internally. Our wrapper does the same: read WAV → convert → write WAV → return path.

### 2.4 Health Endpoint Update

Update `GET /health` to report mojo-audio status instead of (or in addition to) PyTorch CUDA:

```json
{
    "status": "ok",
    "engine": "mojo-audio",
    "model_cached": true,
    "cached_models": ["weeknd-48k"]
}
```

## 3. What Does NOT Change

| Component | Why |
|---|---|
| `POST /separate` | Uses `audio-separator` (Python), not RVC. Unchanged. |
| `POST /clean` | CPU-only audio cleanup. Unchanged. |
| `POST /train` | Celery + Applio training pipeline. Unchanged (training stays on Applio). |
| `GET /models` | Reads model dirs from disk. Unchanged. |
| Frontend | Same API contract, no changes needed. |
| Model storage | Models stay in `MODELS_DIR` as `.pth` files. VoiceConverter reads them directly. |

## 4. File Changes

All changes are in the Shade repo (`/home/visage/repos/shade/`), NOT in mojo-audio:

| File | Change |
|---|---|
| `api/services/rvc.py` | Rewrite: swap Applio for mojo-audio with lazy caching + fallback |
| `api/main.py` | Minor: update `/health` to report engine info |

## 5. Testing Strategy

### 5.1 Smoke Test

1. Start Shade server on Spark
2. Upload a test WAV via `curl POST /convert` with a voice model
3. Verify: valid WAV returned, correct sample rate, no silence/NaN

### 5.2 A/B Quality Comparison

1. Run same input WAV through both engines:
   - `USE_APPLIO=1` → Applio output
   - Default → mojo-audio output
2. Compare spectrograms side-by-side
3. Listen test: does the mojo-audio output sound like the same voice conversion?

Expected: neural filter matches closely (0.9998 correlation proven in Sprint 3). Harmonic source will differ slightly (simplified sine vs multi-harmonic SineGen). Overall quality should be comparable.

### 5.3 Latency Benchmark

Run `/convert` with a 5s vocal clip on Spark:
- Measure: first-request time (cold load) vs subsequent requests (warm)
- Measure: CPU vs GPU inference time
- Record RTF (real-time factor) for the roadmap
- Do NOT block on hitting < 0.5 RTF target — just record the baseline

## 6. Deployment

All on the DGX Spark (`visage@visage-spark`):

```bash
# Start the server
cd /home/visage/repos/shade/api
python main.py  # or uvicorn main:app --host 0.0.0.0 --port 8000

# Test
curl -X POST http://localhost:8000/convert \
  -F "audio=@test.wav" \
  -F "model_path=/home/visage/repos/shade/models/weeknd/model.pth" \
  -F "pitch=0" \
  --output converted.wav
```

## 7. Risk Assessment

| Risk | Likelihood | Impact | Mitigation |
|---|---|---|---|
| Quality difference vs Applio noticeable | Medium | Medium | A/B comparison + Applio fallback env var |
| First-request 6 min load unacceptable | Low | Low | Documented behavior, warm-up script available |
| mojo-audio import fails on Spark | Low | High | Already tested — Sprint 4 tests pass on Spark |
| Memory pressure from caching multiple models | Low | Medium | Each model ~2-4GB, Spark has 121GB RAM |
