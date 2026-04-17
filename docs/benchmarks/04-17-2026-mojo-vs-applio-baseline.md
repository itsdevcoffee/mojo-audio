# mojo-audio vs Applio — Head-to-Head Baseline

**Date:** 2026-04-17
**Status:** Template — not yet run. Fill the blanks on Spark with real numbers.
**Related:** `docs/handoff/04-17-2026-comparison-infrastructure.md`
**Scripts:** `scripts/benchmark_applio_baseline.py`, `scripts/compare_vs_applio.py`

## Why

We shipped mojo-audio as Shade's RVC replacement (Sprints 1–5) but never
formally measured Applio on Spark. Without a baseline we cannot claim
"faster" or "lossless." This file captures the first side-by-side run.

---

## System Configuration

| Spec | Value |
|------|-------|
| Host | `TODO` (e.g. visage-spark, DGX Spark GB10) |
| CPU | `TODO` |
| GPU | `TODO` (e.g. NVIDIA GB10, driver X) |
| RAM | `TODO` |
| OS / Kernel | `TODO` |
| Mojo / MAX | `TODO` (see `pixi.toml`) |
| PyTorch | `TODO` |
| Applio revision | `TODO` (`git -C ~/repos/Applio rev-parse HEAD`) |
| mojo-audio revision | `TODO` (`git rev-parse HEAD`) |

---

## Test Inputs

| Parameter | Value |
|-----------|-------|
| Voice model | `theweekv1.pth` (48 kHz RVC v2) |
| Model path (Spark) | `/home/visage/Downloads/voice files/extracted/theweeknd biggest data set/theweekv1.pth` |
| Input WAV | `TODO` — prefer real isolated vocals, ~5–10 s, 16 kHz mono |
| Input duration | `TODO`s |
| Target SR | 48000 Hz |

## Matched Parameters

Both engines run with the parameters below so differences reflect the engine
rather than config. See the handoff doc for the full parameter table.

| Parameter | Value |
|-----------|-------|
| `pitch_shift` | 0 (no shift) |
| `f0_method` | RMVPE |
| `index_rate` | 0.0 (no FAISS) |
| `protect` | 0.5 (no-op without FAISS) |
| `volume_envelope` | 1.0 |
| `embedder_model` | contentvec |
| `noise_scale` | 0.66666 |

---

## Speed

Times are wall-clock seconds from `time.time()`; warm = second run after a
discarded warm-up. Cold load includes weight load + graph compilation (MAX
graphs on mojo-audio; CUDA kernel compilation on Applio).

### Applio baseline (`benchmark_applio_baseline.py`)

| Stage | Cold (s) | Warm mean (s) | Warm stdev (s) | Warm % of total |
|-------|----------|---------------|----------------|------------------|
| HuBERT / ContentVec | `TODO` | `TODO` | `TODO` | `TODO` |
| RMVPE (F0) | `TODO` | `TODO` | `TODO` | `TODO` |
| VITS (enc_p + flow + vocoder) | `TODO` | `TODO` | `TODO` | `TODO` |
| Overhead (split, filter, pad, paste) | `TODO` | `TODO` | `TODO` | `TODO` |
| **Total inference** | `TODO` | `TODO` | `TODO` | 100% |
| **RTF** | `TODO`x | `TODO`x | — | — |
| Cold load (weights + CUDA init) | `TODO` | — | — | — |

### Head-to-head (`compare_vs_applio.py`)

| Stage | Applio (s) | mojo-audio (s) | mojo speedup |
|-------|------------|----------------|--------------|
| Cold load | `TODO` | `TODO` | `TODO`x |
| HuBERT / ContentVec | `TODO` | `TODO` | `TODO`x |
| RMVPE (F0) | `TODO` | `TODO` | `TODO`x |
| VITS synth | `TODO` | `TODO` | `TODO`x |
| Overhead | `TODO` | `TODO` | `TODO`x |
| **Total warm** | `TODO` | `TODO` | `TODO`x |
| **RTF (warm)** | `TODO`x | `TODO`x | — |
| Device | `TODO` | `TODO` | — |

---

## Quality

Both output WAVs are compared at `min(applio_sr, mojo_sr)` after trimming to
equal length. F0 trajectory uses Applio's RMVPE on the raw 16 kHz input vs
mojo-audio's RMVPE on the same audio — this isolates the RMVPE port.

| Metric | Value | Notes |
|--------|-------|-------|
| Output length (applio / mojo) | `TODO` / `TODO` samples | Difference in samples |
| Waveform correlation | `TODO` | Expect >0.95 if mojo-audio is a faithful port |
| Max abs sample diff | `TODO` | |
| RMS level diff | `TODO` dB | Applio has `volume_envelope`; mojo-audio doesn't |
| Mel-spectrogram correlation | `TODO` | |
| Mel diff (max dB) | `TODO` | |
| Mel diff (mean dB) | `TODO` | |
| F0 voicing agreement | `TODO`% | |
| F0 mean cent error (both voiced) | `TODO` | Reference: 0.13 median cents from audit (`docs/handoff/04-11-2026-audit-results.md §11`) |
| F0 median cent error | `TODO` | |
| F0 % within 5 cents | `TODO`% | |
| F0 % within 50 cents | `TODO`% | |

---

## Interpretation

Fill this in after the run. Structure to answer:

1. **Speed verdict** — which engine is faster overall, and in which stage?
   Is any stage notably slower in mojo-audio than expected?
2. **Quality verdict** — is waveform correlation high enough to call
   mojo-audio a faithful port of Applio? If not, which stage likely
   introduces the drift (spectrogram vs F0 vs sample-level noise)?
3. **Known parameter gaps** — Applio has FAISS index blending,
   `protect<0.5` pitch protection, and `volume_envelope` RMS matching that
   mojo-audio doesn't. Estimate how much each gap contributes to the
   observed quality differences if visible.
4. **Next actions** — what to fix or investigate before claiming parity
   publicly?

---

## Output Files (not committed)

- `/tmp/compare_applio_out.wav` — last Applio output from the comparison run
- `/tmp/compare_mojo_out.wav` — matching mojo-audio output
- `/tmp/applio_baseline_out.wav` — last warm Applio output from the baseline run

Keep these out of the repo — they're large and not reproducible without the
exact input.

---

## How This Was Generated

```bash
# Cold/warm baseline
pixi run python scripts/benchmark_applio_baseline.py \
    --model "/home/visage/Downloads/voice files/extracted/theweeknd biggest data set/theweekv1.pth" \
    --audio /path/to/vocal.wav

# Head-to-head
pixi run python scripts/compare_vs_applio.py \
    --model "/home/visage/Downloads/voice files/extracted/theweeknd biggest data set/theweekv1.pth" \
    --audio /path/to/vocal.wav
```
