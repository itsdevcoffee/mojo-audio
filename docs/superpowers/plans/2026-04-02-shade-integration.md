# Shade Integration Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Replace Applio voice conversion in Shade's FastAPI server with mojo-audio's VoiceConverter — same API, different engine, with Applio fallback.

**Architecture:** Rewrite `services/rvc.py` to lazy-load mojo-audio's VoiceConverter on first request, cache per model path. `USE_APPLIO=1` env var falls back to old Applio path. Update `/health` endpoint. All changes in the Shade repo on Spark.

**Tech Stack:** Python, FastAPI, mojo-audio (VoiceConverter), soundfile, numpy

**Spec:** `docs/superpowers/specs/2026-04-02-shade-integration-design.md`

---

## File Map

All changes are in the Shade repo at `/home/visage/repos/shade/` on the DGX Spark.

| File | Responsibility |
|---|---|
| `api/services/rvc.py` | Voice conversion service — rewrite to use mojo-audio with Applio fallback |
| `api/main.py` | FastAPI server — update `/health` endpoint |

**Reference files** (read, don't modify):
- `/home/visage/repos/mojo-audio/src/models/voice_converter.py` — VoiceConverter API
- `/home/visage/repos/shade/api/services/separator.py` — Shade service pattern to follow

**Important:** This work happens on the Spark machine (`ssh visage@visage-spark`). The Shade repo is NOT in mojo-audio — it's a separate repo.

---

### Task 1: Rewrite services/rvc.py

**Files:**
- Modify: `/home/visage/repos/shade/api/services/rvc.py`

- [ ] **Step 1: Back up the current file**

```bash
ssh visage@visage-spark "cp /home/visage/repos/shade/api/services/rvc.py /home/visage/repos/shade/api/services/rvc.py.bak"
```

- [ ] **Step 2: Write the new rvc.py**

Replace the entire file with:

```python
"""
RVC voice conversion service — mojo-audio engine with Applio fallback.

Default: uses mojo-audio VoiceConverter (MAX Engine, no PyTorch CUDA needed).
Fallback: set USE_APPLIO=1 env var to use Applio's VoiceConverter instead.
"""

import logging
import os
import sys
import tempfile
from pathlib import Path

import numpy as np
import soundfile as sf

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Engine selection
# ---------------------------------------------------------------------------

USE_APPLIO = os.environ.get("USE_APPLIO", "").strip() in ("1", "true", "yes")

if USE_APPLIO:
    # Legacy Applio path
    APPLIO_PATH = os.environ.get("APPLIO_PATH", "/home/visage/repos/Applio")
    if not os.path.isdir(APPLIO_PATH):
        raise RuntimeError(f"USE_APPLIO=1 but Applio not found at {APPLIO_PATH}")
    if APPLIO_PATH not in sys.path:
        sys.path.insert(0, APPLIO_PATH)
    os.chdir(APPLIO_PATH)
    logger.info("Voice conversion engine: Applio (%s)", APPLIO_PATH)
else:
    # mojo-audio path
    MOJO_AUDIO_PATH = os.environ.get(
        "MOJO_AUDIO_PATH", "/home/visage/repos/mojo-audio/src"
    )
    if MOJO_AUDIO_PATH not in sys.path:
        sys.path.insert(0, MOJO_AUDIO_PATH)
    logger.info("Voice conversion engine: mojo-audio (%s)", MOJO_AUDIO_PATH)

# ---------------------------------------------------------------------------
# Model cache (lazy loading)
# ---------------------------------------------------------------------------

_model_cache: dict = {}


def _get_converter(model_path: str):
    """Get or create a cached VoiceConverter for the given model path."""
    if model_path not in _model_cache:
        from models.voice_converter import VoiceConverter

        logger.info("Loading VoiceConverter for %s (first request, may take several minutes)...", model_path)
        _model_cache[model_path] = VoiceConverter.from_pretrained(
            model_path, device="cpu"
        )
        logger.info("VoiceConverter loaded and cached for %s", model_path)
    return _model_cache[model_path]


def get_cached_models() -> list[str]:
    """Return list of model paths currently cached."""
    return list(_model_cache.keys())


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def convert(
    input_path: str,
    model_path: str,
    index_path: str | None = None,
    pitch: int = 0,
    pitch_extraction: str = "rmvpe",
    index_ratio: float = 0.75,
    output_dir: str | None = None,
) -> str:
    """
    Convert audio to a target voice using an RVC v2 model.

    Args:
        input_path:       Path to the input vocal WAV file.
        model_path:       Path to the .pth voice model file.
        index_path:       Ignored (FAISS not yet implemented in mojo-audio).
        pitch:            Semitone shift. 0 = no shift.
        pitch_extraction: Ignored (always RMVPE). Accepted for API compat.
        index_ratio:      Ignored (FAISS not yet implemented).
        output_dir:       Directory to write the converted file.

    Returns:
        Path to the converted output WAV file.
    """
    if USE_APPLIO:
        return _convert_applio(
            input_path, model_path, index_path, pitch,
            pitch_extraction, index_ratio, output_dir,
        )

    # Log ignored parameters
    if pitch_extraction != "rmvpe":
        logger.warning(
            "pitch_extraction='%s' ignored — mojo-audio always uses RMVPE",
            pitch_extraction,
        )
    if index_path:
        logger.debug("index_path ignored — FAISS not implemented in mojo-audio")

    # Read input audio
    audio, sr = sf.read(input_path, dtype="float32")
    if audio.ndim == 2:
        audio = audio.mean(axis=1)  # stereo to mono
    audio = audio.reshape(1, -1)  # [1, N]

    # Convert
    converter = _get_converter(model_path)
    converted = converter.convert(audio, pitch_shift=pitch, sr=sr)  # [1, T_audio]

    # Write output
    out_dir = output_dir or tempfile.mkdtemp(prefix="shade-rvc-")
    os.makedirs(out_dir, exist_ok=True)
    out_name = Path(input_path).stem + "_converted.wav"
    output_path = str(Path(out_dir) / out_name)

    target_sr = converter._config.get("sr", 48000)
    sf.write(output_path, converted.squeeze(), target_sr)

    return output_path


def _convert_applio(
    input_path, model_path, index_path, pitch,
    pitch_extraction, index_ratio, output_dir,
):
    """Legacy Applio conversion path."""
    from rvc.infer.infer import VoiceConverter as ApplioConverter

    out_dir = output_dir or tempfile.mkdtemp(prefix="shade-rvc-")
    os.makedirs(out_dir, exist_ok=True)
    out_name = Path(input_path).stem + "_converted.wav"
    output_path = str(Path(out_dir) / out_name)

    converter = ApplioConverter()
    converter.convert_audio(
        audio_input_path=input_path,
        audio_output_path=output_path,
        model_path=model_path,
        index_path=index_path or "",
        pitch=pitch,
        f0_method=pitch_extraction,
        index_rate=index_ratio,
        volume_envelope=1,
        protect=0.33,
        hop_length=128,
        split_audio=False,
        f0_autotune=False,
        f0_autotune_strength=1.0,
        filter_radius=3,
        embedder_model="contentvec",
        embedder_model_custom=None,
        clean_audio=True,
        clean_strength=0.7,
        export_format="WAV",
        resample_sr=0,
        sid=0,
    )
    return output_path
```

- [ ] **Step 3: Verify the file is syntactically correct**

```bash
ssh visage@visage-spark "cd /home/visage/repos/shade/api && python -c 'import ast; ast.parse(open(\"services/rvc.py\").read()); print(\"Syntax OK\")'"
```

Expected: `Syntax OK`

- [ ] **Step 4: Verify mojo-audio imports work**

```bash
ssh visage@visage-spark "cd /home/visage/repos/shade/api && python -c '
import sys; sys.path.insert(0, \"/home/visage/repos/mojo-audio/src\")
from models.voice_converter import VoiceConverter
print(\"Import OK:\", VoiceConverter)
'"
```

Expected: `Import OK: <class 'models.voice_converter.VoiceConverter'>`

- [ ] **Step 5: Commit**

```bash
ssh visage@visage-spark "cd /home/visage/repos/shade && git add api/services/rvc.py && git commit -m 'feat: swap Applio for mojo-audio VoiceConverter with fallback'"
```

---

### Task 2: Update /health endpoint

**Files:**
- Modify: `/home/visage/repos/shade/api/main.py`

- [ ] **Step 1: Update the health endpoint**

In `main.py`, replace the existing `health()` function:

```python
@app.get("/health")
def health():
    import torch
    return {
        "status": "ok",
        "cuda": torch.cuda.is_available(),
        "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
    }
```

With:

```python
@app.get("/health")
def health():
    from services.rvc import USE_APPLIO, get_cached_models
    result = {"status": "ok", "engine": "applio" if USE_APPLIO else "mojo-audio"}
    if not USE_APPLIO:
        result["cached_models"] = get_cached_models()
    else:
        import torch
        result["cuda"] = torch.cuda.is_available()
        result["gpu"] = torch.cuda.get_device_name(0) if torch.cuda.is_available() else None
    return result
```

- [ ] **Step 2: Verify syntax**

```bash
ssh visage@visage-spark "cd /home/visage/repos/shade/api && python -c 'import ast; ast.parse(open(\"main.py\").read()); print(\"Syntax OK\")'"
```

- [ ] **Step 3: Commit**

```bash
ssh visage@visage-spark "cd /home/visage/repos/shade && git add api/main.py && git commit -m 'feat: update /health to report mojo-audio engine status'"
```

---

### Task 3: Copy voice model to Shade models directory

**Files:** None (file management only)

The Shade `/models` endpoint expects models in `shade/models/{name}/model.pth`. We need at least one model there for testing.

- [ ] **Step 1: Create model directory and copy checkpoint**

```bash
ssh visage@visage-spark "
mkdir -p /home/visage/repos/shade/models/weeknd
cp '/home/visage/Downloads/voice files/extracted/theweeknd biggest data set/theweekv1.pth' /home/visage/repos/shade/models/weeknd/model.pth
ls -lh /home/visage/repos/shade/models/weeknd/model.pth
"
```

Expected: file exists, ~55MB

- [ ] **Step 2: Verify /models endpoint would find it**

```bash
ssh visage@visage-spark "cd /home/visage/repos/shade/api && python -c '
from pathlib import Path
models_dir = Path(\"../models\")
for d in sorted(models_dir.iterdir()):
    pth = d / \"model.pth\"
    if pth.exists():
        print(f\"{d.name}: {pth}\")
'"
```

Expected: `weeknd: ../models/weeknd/model.pth`

---

### Task 4: Smoke test — full /convert endpoint

**Files:** None (manual testing)

Start the Shade server and test the conversion endpoint end-to-end.

- [ ] **Step 1: Copy test audio to Spark**

```bash
scp /tmp/test_tone.wav visage@visage-spark:/tmp/test_tone.wav
```

- [ ] **Step 2: Start the Shade server**

```bash
ssh visage@visage-spark "cd /home/visage/repos/shade/api && python main.py &"
```

Wait for the server to start (should see `Uvicorn running on http://0.0.0.0:8000`).

- [ ] **Step 3: Test /health endpoint**

```bash
ssh visage@visage-spark "curl -s http://localhost:8000/health | python -m json.tool"
```

Expected: `{"status": "ok", "engine": "mojo-audio", "cached_models": []}`

- [ ] **Step 4: Test /models endpoint**

```bash
ssh visage@visage-spark "curl -s http://localhost:8000/models | python -m json.tool"
```

Expected: JSON with weeknd model listed.

- [ ] **Step 5: Test /convert endpoint (first request — cold load)**

```bash
ssh visage@visage-spark "time curl -X POST http://localhost:8000/convert \
  -F 'audio=@/tmp/test_tone.wav' \
  -F 'model_path=/home/visage/repos/shade/models/weeknd/model.pth' \
  -F 'pitch=0' \
  --output /tmp/shade_converted.wav"
```

Expected: WAV file written, takes ~6-8 minutes (first-time model load + inference).

- [ ] **Step 6: Verify output is valid audio**

```bash
ssh visage@visage-spark "python -c '
import soundfile as sf
import numpy as np
data, sr = sf.read(\"/tmp/shade_converted.wav\")
print(f\"Shape: {data.shape}, SR: {sr}\")
print(f\"Range: [{data.min():.4f}, {data.max():.4f}]\")
print(f\"NaN: {np.any(np.isnan(data))}\")
print(f\"Silent: {np.abs(data).max() < 0.001}\")
'"
```

Expected: Valid shape, 48000 Hz, no NaN, not silent.

- [ ] **Step 7: Test /convert (second request — warm, should be fast)**

```bash
ssh visage@visage-spark "time curl -X POST http://localhost:8000/convert \
  -F 'audio=@/tmp/test_tone.wav' \
  -F 'model_path=/home/visage/repos/shade/models/weeknd/model.pth' \
  -F 'pitch=0' \
  --output /tmp/shade_converted_warm.wav"
```

Expected: Completes in ~10-15 seconds (model already cached).

- [ ] **Step 8: Test /health shows cached model**

```bash
ssh visage@visage-spark "curl -s http://localhost:8000/health | python -m json.tool"
```

Expected: `cached_models` now includes the weeknd model path.

- [ ] **Step 9: Stop the server**

```bash
ssh visage@visage-spark "pkill -f 'python main.py' || true"
```

---

### Task 5: Benchmark + update roadmap

**Files:**
- Modify: `/home/maskkiller/dev-coffee/repos/mojo-audio/docs/project/03-06-2026-roadmap.md`

- [ ] **Step 1: Record benchmark results**

From the smoke test above, record:
- Cold load time (first request)
- Warm inference time (second request)
- Output sample rate
- RTF (real-time factor = inference_time / audio_duration)

- [ ] **Step 2: Update roadmap**

Mark Sprint 5 as complete in the roadmap. Update the benchmark table with actual numbers.

- [ ] **Step 3: Commit roadmap update**

```bash
cd /home/maskkiller/dev-coffee/repos/mojo-audio
git add docs/project/03-06-2026-roadmap.md
git commit -m "docs: update roadmap — Sprint 5 complete, Shade integration done"
git push origin main
```

---

## Task Dependencies

```
Task 1 (rvc.py rewrite) ──→ Task 4 (smoke test)
Task 2 (health endpoint) ──→ Task 4
Task 3 (copy model) ──→ Task 4
Task 4 (smoke test) ──→ Task 5 (benchmark + roadmap)
```

Tasks 1, 2, and 3 are independent and can be done in parallel.
Task 4 depends on all three.
Task 5 depends on Task 4 (needs benchmark numbers).
