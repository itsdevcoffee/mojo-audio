# Handoff: Build mojo-audio vs Applio Comparison Infrastructure

**Date:** 2026-04-17
**For:** An agent building the head-to-head comparison tooling
**Working directory:** `/home/maskkiller/dev-coffee/repos/mojo-audio`
**Spark:** `ssh visage@visage-spark`, repo at `/home/visage/repos/mojo-audio`

---

## Why this matters

mojo-audio replaces Applio (PyTorch RVC) as the voice conversion engine in
Shade. Sprints 1–5 shipped the replacement. But we have **no formal proof**
that mojo-audio matches or beats Applio on speed or quality. We can't walk
into a meeting and say "1.5x faster, lossless quality" because we never
measured the baseline. This doc scopes the tooling to fix that.

---

## What exists today

### mojo-audio performance data (documented)

| Metric | Value | Source |
|---|---|---|
| Full pipeline RTF (Spark CPU, warm, 2s) | 0.63x | `docs/handoff/04-09-2026-sprint-5-complete.md` |
| Full pipeline cold start | ~380s (JIT) | same |
| AudioEncoder GPU RTF | 0.24x | `docs/handoff/04-16-2026-gpu-failure-matrix.md` |
| PitchExtractor GPU RTF | 0.40x | same |
| HiFiGAN GPU RTF | 2.15x (slower than CPU) | same |

### mojo-audio quality data (component-level)

| Component | vs PyTorch | Metric | Source |
|---|---|---|---|
| Mel spectrogram | Applio's `MelSpectrogram` | corr > 0.999, max_diff < 0.1 | `tests/test_pitch_extractor.py:459` |
| RMVPE salience | PyTorch `E2E` model | corr 1.0, max_diff ~1e-5 | `tests/test_pitch_extractor.py:561` |
| F0 (post-fix) | Applio RMVPE | median 0.13 cents, 96% within 5 cents | `docs/handoff/04-11-2026-audit-results.md §11` |
| enc_p (TextEncoder) | PyTorch reference | corr > 0.999 | `tests/test_vits.py:731` |
| Flow (normalizing flow) | PyTorch reference | corr > 0.999 | `tests/test_vits.py:1066` |
| HiFiGAN neural filter | PyTorch reference | corr 0.9998 | `docs/project/03-06-2026-roadmap.md` |

### Applio performance data

**None.** We never measured Applio's RTF on Spark. We don't know per-component
timing. We can't make any relative claims.

### End-to-end output comparison

**None.** No script runs identical audio through both engines and diffs the
output waveform, spectrogram, or F0 trajectory.

---

## What to build

### 1. `scripts/compare_vs_applio.py` — Head-to-head comparison script

Takes a voice model + input WAV, runs both pipelines, reports speed and
quality metrics side by side.

**Required output:**

```
=== mojo-audio vs Applio Comparison ===
Model:     theweekv1.pth (48kHz)
Input:     test_vocal.wav (3.2s @ 16kHz)
Device:    DGX Spark aarch64, GB10 GPU

--- Speed ---
                    Applio (PyTorch)    mojo-audio (MAX)    Speedup
Cold load:          12.3s               380.0s              0.03x ← JIT
Warm inference:     0.95s               1.26s               0.75x
RTF (warm):         0.30x               0.63x               —
Per-stage:
  ContentVec:       0.15s               0.12s               1.25x
  RMVPE:            0.30s               0.22s               1.36x
  enc_p + flow:     0.10s               0.08s               1.25x
  HiFiGAN:          0.40s               0.84s               0.48x

--- Quality ---
Output waveform correlation:   0.9987
Output spectrogram max diff:   0.0234
F0 trajectory correlation:     0.9999
Voiced frame agreement:        98.7%
RMS level difference:          0.3 dB
Output duration match:         exact (153600 samples)

--- Files ---
Applio output:     /tmp/compare_applio_out.wav
mojo-audio output: /tmp/compare_mojo_out.wav
```

### 2. `scripts/benchmark_applio_baseline.py` — Applio-only RTF baseline

Measures Applio's per-component and end-to-end timing on Spark. This gives
us the denominator for all speedup claims.

---

## How to run Applio programmatically

Applio lives at:
- **Local:** `/home/maskkiller/repos/Applio`
- **Spark:** `/home/visage/repos/Applio`

### High-level API (file-based, easiest)

```python
import sys
sys.path.insert(0, "/home/maskkiller/repos/Applio")  # or /home/visage/repos/Applio

from rvc.infer.infer import VoiceConverter

converter = VoiceConverter()
converter.convert_audio(
    audio_input_path="input.wav",
    audio_output_path="output.wav",
    model_path="path/to/model.pth",
    index_path="",              # no FAISS index
    pitch=0,                    # no pitch shift
    f0_method="rmvpe",
    index_rate=0.0,             # disable index blending (mojo-audio doesn't have it)
    volume_envelope=1.0,
    protect=0.5,
    hop_length=128,
    split_audio=False,
    f0_autotune=False,
    embedder_model="contentvec",
    clean_audio=False,
    export_format="WAV",
    sid=0,
)
```

**Problem:** this is file-based and doesn't expose per-stage timing. For
benchmarking, use the low-level pipeline.

### Low-level API (for per-stage timing)

```python
import sys, time, torch
import numpy as np
sys.path.insert(0, "/home/maskkiller/repos/Applio")

from rvc.infer.pipeline import Pipeline
from rvc.lib.utils import load_audio_infer, load_embedding
from rvc.lib.algorithm.synthesizers import Synthesizer
from rvc.configs.config import Config

config = Config()
device = config.device  # "cuda:0" or "cpu"

# 1. Load audio
audio = load_audio_infer("input.wav", 16000)

# 2. Load HuBERT
t0 = time.time()
hubert = load_embedding("contentvec")
hubert = hubert.to(device).float().eval()
print(f"HuBERT load: {time.time()-t0:.2f}s")

# 3. Load RVC checkpoint
t0 = time.time()
model_path = "path/to/model.pth"
cpt = torch.load(model_path, map_location="cpu", weights_only=True)
tgt_sr = cpt["config"][-1]
cpt["config"][-3] = cpt["weight"]["emb_g.weight"].shape[0]
use_f0 = cpt.get("f0", 1)
version = cpt.get("version", "v1")
vocoder = cpt.get("vocoder", "HiFi-GAN")
text_enc_hidden_dim = 768 if version == "v2" else 256

net_g = Synthesizer(
    *cpt["config"],
    use_f0=use_f0,
    text_enc_hidden_dim=text_enc_hidden_dim,
    vocoder=vocoder,
)
del net_g.enc_q
net_g.load_state_dict(cpt["weight"], strict=False)
net_g = net_g.to(device).float().eval()
print(f"Model load: {time.time()-t0:.2f}s")

# 4. Run pipeline
vc = Pipeline(tgt_sr, config)
t0 = time.time()
audio_out = vc.pipeline(
    model=hubert,
    net_g=net_g,
    sid=0,
    audio=audio,
    pitch=0,
    f0_method="rmvpe",
    file_index="",
    index_rate=0.0,
    pitch_guidance=use_f0,
    volume_envelope=1.0,
    version=version,
    protect=0.5,
    f0_autotune=False,
    f0_autotune_strength=1.0,
    proposed_pitch=False,
    proposed_pitch_threshold=155.0,
)
print(f"Inference: {time.time()-t0:.2f}s, RTF: {(time.time()-t0)/(len(audio)/16000):.3f}")

# 5. Save
import soundfile as sf
sf.write("applio_out.wav", audio_out, tgt_sr)
```

**Note on per-stage timing:** Applio's `Pipeline.pipeline()` runs all stages
internally. To get per-stage timing, you'll need to either:
- Monkey-patch timing hooks into the pipeline method
- Or replicate the stages manually using the imports below

### Key Applio component imports for manual staging

```python
# Pitch extraction
from rvc.lib.predictors.RMVPE import RMVPE0Predictor
predictor = RMVPE0Predictor("rvc/models/predictors/rmvpe.pt", device=device)
f0 = predictor.infer_from_audio(audio, thred=0.03)  # → [T] Hz

# Mel spectrogram (for comparison)
from rvc.lib.predictors.RMVPE import MelSpectrogram as ApplioMel
mel_fn = ApplioMel(128, 16000, 1024, 160, mel_fmin=30, mel_fmax=8000)
mel = mel_fn(torch.from_numpy(audio).unsqueeze(0))

# Full RMVPE model
from rvc.lib.predictors.RMVPE import E2E
```

---

## How to run mojo-audio programmatically

mojo-audio lives at:
- **Local:** `/home/maskkiller/dev-coffee/repos/mojo-audio` (src in `src/`)
- **Spark:** `/home/visage/repos/mojo-audio`

```python
import sys, time
import numpy as np
sys.path.insert(0, "/home/maskkiller/dev-coffee/repos/mojo-audio/src")
# or on Spark: sys.path.insert(0, "/home/visage/repos/mojo-audio/src")

from models.voice_converter import VoiceConverter

CKPT = "path/to/model.pth"

# Load (cold start includes JIT compilation of 5 MAX graphs)
t0 = time.time()
vc = VoiceConverter.from_pretrained(CKPT, device="cpu")  # or "gpu" / "auto"
print(f"Load: {time.time()-t0:.2f}s")

# Convert
audio_in = np.random.randn(1, 32000).astype(np.float32)  # or load real audio
t0 = time.time()
audio_out = vc.convert(audio_in, pitch_shift=0, sr=16000)
print(f"Inference: {time.time()-t0:.2f}s")

# Per-stage timing requires calling components individually:
from models.audio_encoder import AudioEncoder
from models.pitch_extractor import PitchExtractor
from models.hifigan import NSFHiFiGAN

enc = AudioEncoder.from_pretrained("facebook/hubert-base-ls960", device="cpu")
pe = PitchExtractor.from_pretrained("lj1995/VoiceConversionWebUI", device="cpu")
# Then time enc.encode(audio), pe.extract(audio), vc.convert_from_features(...)
```

---

## Test checkpoints and audio

### On Spark

- **Voice model:** `/home/visage/Downloads/voice files/extracted/theweeknd biggest data set/theweekv1.pth` (48kHz)
- **All Shade models:** `/home/visage/repos/shade/models/` (23 models: Weeknd, Ariana Grande, Drake, etc.)
- **Applio sliced audios:** `/home/visage/repos/Applio/` (look for `sliced_audios/` or test WAVs)
- **RMVPE weights:** auto-downloaded by both pipelines from `lj1995/VoiceConversionWebUI`
- **HuBERT weights:** auto-downloaded from `facebook/hubert-base-ls960`

### On local machine

- **Voice model:** Same checkpoint if synced, or use any RVC v2 `.pth`
- **Applio:** `/home/maskkiller/repos/Applio`

### Test audio recommendation

Use a real vocal clip (not random noise — noise produces all-unvoiced frames
which trivially match). Ideal: 3–10s of clean isolated vocals at 16kHz mono.
Check `/home/visage/repos/Applio/` for sliced training data, or extract from
any song using Shade's `/separate` endpoint.

---

## Quality metrics to compute

### Waveform-level

```python
import numpy as np

def compare_waveforms(ref: np.ndarray, test: np.ndarray):
    # Trim to same length
    L = min(len(ref), len(test))
    ref, test = ref[:L], test[:L]

    # Correlation
    corr = np.corrcoef(ref, test)[0, 1]

    # RMS difference
    rms_ref = np.sqrt(np.mean(ref**2))
    rms_test = np.sqrt(np.mean(test**2))
    rms_diff_db = 20 * np.log10(rms_test / rms_ref + 1e-10)

    # Max absolute difference
    max_diff = np.abs(ref - test).max()

    return {"corr": corr, "rms_diff_db": rms_diff_db, "max_diff": max_diff}
```

### Spectrogram-level

```python
import librosa

def compare_spectrograms(ref: np.ndarray, test: np.ndarray, sr: int):
    S_ref = librosa.feature.melspectrogram(y=ref, sr=sr, n_mels=128)
    S_test = librosa.feature.melspectrogram(y=test, sr=sr, n_mels=128)
    L = min(S_ref.shape[1], S_test.shape[1])
    S_ref, S_test = S_ref[:, :L], S_test[:, :L]

    log_ref = librosa.power_to_db(S_ref)
    log_test = librosa.power_to_db(S_test)

    return {
        "spec_corr": np.corrcoef(log_ref.flatten(), log_test.flatten())[0, 1],
        "spec_max_diff_db": np.abs(log_ref - log_test).max(),
        "spec_mean_diff_db": np.abs(log_ref - log_test).mean(),
    }
```

### F0-level

```python
def compare_f0(f0_ref: np.ndarray, f0_test: np.ndarray):
    L = min(len(f0_ref), len(f0_test))
    f0_ref, f0_test = f0_ref[:L], f0_test[:L]

    both_voiced = (f0_ref > 0) & (f0_test > 0)
    voicing_agreement = np.mean(
        (f0_ref > 0) == (f0_test > 0)
    )

    if both_voiced.any():
        cents = 1200 * np.abs(np.log2(f0_test[both_voiced] / f0_ref[both_voiced]))
        return {
            "voicing_agreement": voicing_agreement,
            "mean_cent_error": cents.mean(),
            "median_cent_error": np.median(cents),
            "pct_within_5_cents": (cents < 5).mean(),
            "pct_within_50_cents": (cents < 50).mean(),
        }
    return {"voicing_agreement": voicing_agreement}
```

---

## Parameters to match between pipelines

When comparing, use identical parameters so differences reflect the engine,
not config:

| Parameter | Applio value | mojo-audio value | Notes |
|---|---|---|---|
| `pitch_shift` | 0 | 0 | No shift |
| `f0_method` | `"rmvpe"` | (built-in RMVPE) | Same model |
| `index_rate` | 0.0 | N/A | mojo-audio doesn't have FAISS yet — disable in Applio |
| `protect` | 0.5 | N/A | mojo-audio doesn't have pitch protection yet |
| `volume_envelope` | 1.0 | N/A | mojo-audio doesn't adjust RMS |
| `embedder_model` | `"contentvec"` | (built-in HuBERT) | Same model |
| `noise_scale` | 0.66666 | 0.66666 | RVC v2 default, hardcoded in both |

**Expected quality differences from parameter gaps:**
- **FAISS index blending** — Applio uses it (when index_rate > 0) to blend
  content features with the training speaker's voice. Setting index_rate=0
  disables it, making the comparison fair. But in production Shade uses
  index_rate=0.75, so mojo-audio is missing this quality feature.
- **Pitch protection** — Applio's `protect=0.5` prevents consonant pitch
  distortion. mojo-audio doesn't implement this. May cause subtle quality
  diffs on sibilants/plosives.
- **Volume envelope matching** — Applio matches output RMS to input. mojo-audio
  doesn't. Output levels may differ.

---

## Where to run

**Spark is the canonical comparison environment** because:
- Both Applio and mojo-audio are installed
- PyTorch CUDA is available for Applio's GPU path
- DGX Spark GB10 is the production hardware
- Real voice models are available in `/home/visage/repos/shade/models/`

**Local machine works for development** but:
- Different GPU (RTX 4060 Ti vs GB10)
- May not have all voice models
- PyTorch CUDA uses a different driver stack

---

## Deliverables

1. **`scripts/compare_vs_applio.py`** — side-by-side comparison on same input
2. **`scripts/benchmark_applio_baseline.py`** — Applio-only RTF baseline
3. **`docs/benchmarks/04-17-2026-mojo-vs-applio-baseline.md`** — first run
   results with at least one voice model + one real vocal clip
4. Update `docs/project/04-17-2026-backlog-radar.md` with the baseline numbers

---

## What NOT to do

- Don't modify Applio's code — treat it as a black box
- Don't modify mojo-audio's inference code — the comparison must reflect
  current production state
- Don't use random noise as test audio — use real vocals
- Don't compare GPU Applio vs CPU mojo-audio (or vice versa) — compare
  apples to apples (CPU vs CPU, GPU vs GPU when available)
- Don't commit Applio outputs or large WAV files to the repo
