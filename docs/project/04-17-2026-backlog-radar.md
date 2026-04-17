# mojo-audio Backlog Radar — 2026-04-17

**Context:** Sprints 1–5 complete. Shade live on Spark CPU at RTF 0.63x.
PitchExtractor GPU compile fix landed 04-16 (`f6eb128`). Dual-Spark NCCL
bandwidth issue resolved by Chris. This doc captures everything outstanding.

---

## Active — GPU Campaign

These are the next things to land, in order. Each unblocks the next.

| # | Task | Est. | Blocker? | Files |
|---|---|---|---|---|
| 1 | **VITS tensor-placement fix** — `convert_from_features()` passes numpy arrays to GPU-compiled `enc_p_model.execute()` and `flow_model.execute()`. Wrap with `Tensor.from_numpy().to(device)`. Same for `flow_model`. | 30 min | Yes — blocks full pipeline on GPU | `src/models/voice_converter.py` |
| 2 | **AudioEncoder `ops.tile` → baked numpy constant** — lines 195/199 in `audio_encoder.py` use `ops.tile` which silently transfers to CPU on GPU (flagged by MAX dev2026041520 as TODO(GEX-2056)). Pre-tile gamma/beta in numpy. | 10 min | No — perf only | `src/models/audio_encoder.py` |
| 3 | **HiFiGAN + PitchExtractor GPU perf** — both compile and run on GPU but are slower than CPU (HiFiGAN RTF 2.15 vs CPU ~0.2; PitchExtractor RTF 0.40 vs CPU 0.11). Root cause: im2col workaround creates large intermediates that thrash GPU memory. Fix path: swap to native `ops.conv2d` on aarch64 (04-11 audit verified the C_in≥8 bug is fixed there). Profile before/after. | ~1 week | No — perf only | `src/models/_rmvpe.py`, `src/models/_hifigan_graph.py` |
| 4 | **Full pipeline end-to-end GPU RTF** — run real Shade input (Weeknd 30s vocal) through `VoiceConverter.convert()` on GPU. Report warm RTF. Compare vs CPU 0.63x baseline. This number tells the demo story. | 1 hr | After 1–3 | — |

### GPU matrix (04-16 recon, post-fix)

| Stage | Compile | Run | RTF (GPU) | RTF (CPU) | Status |
|---|---|---|---|---|---|
| AudioEncoder | ✅ | ✅ | 0.24 | — | Working |
| PitchExtractor | ✅ | ✅ | 0.40 | 0.11 | Fixed 04-16 — needs perf |
| HiFiGAN | ✅ | ✅ | 2.15 | ~0.2 | Working — needs perf |
| VITS enc_p + flow | ✅ | ❌ | — | — | Placement bug (#1 above) |
| Full VoiceConverter | ⏳ | — | — | — | Blocked on #1 |

---

## Deferred — Shade / Deployment

| Task | Est. | Notes |
|---|---|---|
| Systemd services for Shade API + frontend | 2 hrs | Currently nohup — doesn't survive reboots. Plan in `docs/plans/04-09-2026-shade-redeployment.md`. |
| Docker image with MAX Engine | Days | Proper containerized deployment. Low priority while iterating. |
| `/clean` and `/separate` endpoints | 1 hr | Need `noisereduce` + `audio-separator` in pixi env. pip install upgraded torch to 2.11.0 — may conflict. |

---

## Deferred — MAX Engine Bugs

| Task | Status | Notes |
|---|---|---|
| File `ops.rebind` GPU rank mismatch | Ready to file | Minimal repro: `build_unet_graph` on GPU triggers; isolated `_conv2d` doesn't. Persisted 26+ nightlies (dev2026032005 → dev2026041520). We worked around it (`f6eb128`) but should file upstream. |
| Comment on `modular/modular#6248` (conv2d C_in≥8) | Draft pending review | aarch64 fix confirmed, x64 still broken (byte-identical). K=7 slightly worse (0.191 vs 0.165). Multi-stride crash fixed on both. See `docs/handoff/04-11-2026-audit-results.md §4–5`. |
| Close multi-stride conv2d crash | Done upstream | Fixed on both x64 and aarch64 as of dev2026041020. Just needs a "confirmed fixed" comment. |
| Blog: "Every conv2d Bug We Hit in MAX" | Not started | Research doc ready from Sprint 3. Strong content opportunity. |
| Blog post pitch to Modular | Not started | "Voice conversion on DGX Spark, zero PyTorch CUDA, 100% MAX Engine." |

---

## Deferred — Quality / Model Upgrades

| Task | Priority | Notes |
|---|---|---|
| FAISS index retrieval (speaker similarity blending) | Medium | CPU-side bolt-on. Quality improvement for voice matching. |
| Pitch protection blending | Low | Smoother pitch transitions. |
| Relative position attention in-graph | Low | Currently numpy pre-pass duplicates encoder work. Functional but wasteful. |
| BigVGAN vocoder swap | Exploratory | Potential quality upgrade over NSF-HiFiGAN. |
| Seed-VC (zero-shot voice conversion) | Exploratory | No fine-tuning needed — single reference audio. |
| RVC v3 / shallow diffusion | Exploratory | Next-gen model architectures. |

---

## Deferred — Technical Debt

| Issue | Severity | Notes |
|---|---|---|
| HiFiGAN `batch>1` blocked (ConvTranspose1d zero-interleave requires B=1) | Medium | Blocks concurrent request processing (Sprint 6). |
| GroupNorm approximated as LayerNorm in CNN | Low | — |
| ContentVec `weight_g/weight_v` detection robustness | Medium | Before shipping to external users. |
| NSF-HiFiGAN harmonic source: simplified numpy sine (0.66 corr vs PyTorch SineGen) | Low | Neural filter itself is 0.9998 corr. |
| Stale docstring at `_rmvpe.py:37-42` | Low | Pre-fix residual block structure still documented. Audit finding. |
| Griffin-Lim lacks true complex phase | Low | — |
| FFI exports for AudioEncoder | Low | Only if mojovoice needs VC. |

---

## Deferred — Sprint 6: Multi-Spark NVLink

**Prerequisite:** GPU pipeline working on single Spark (items 1–4 above).

| Task | Status | Notes |
|---|---|---|
| NCCL bandwidth issue | ✅ Resolved | Chris fixed this 04-15. visage-maximus-tag-team repo at `/home/visage/repos/visage-maximus-tag-team`. |
| Shard VITS synthesis across 2 GPUs | Not started | `max.nn.DistributedTransformer`. 256GB unified memory pool. |
| Batch processing concurrent voice conversions | Blocked | Needs HiFiGAN batch>1 fix first. |
| Configure IPs on lowercase interface (`enp1s0f1np1`) | Not done | Prerequisite for dual-NIC NCCL. See `docs/handoff/04-11-2026-dual-spark-nccl-investigation.md`. |
| Align Spark 2 driver to `580.142` | Not done | Next convenient reboot. |

---

## Community / Visibility

| Milestone | Status |
|---|---|
| MAX bug filed (conv2d groups) — `modular/modular#6129` | ✅ |
| MAX bug filed (conv2d C_in≥8) — `modular/modular#6248` | ✅ Filed, pending comment update |
| RMVPE working on Spark GPU | ✅ (04-16) |
| Full pipeline on Spark GPU | ⏳ VITS placement fix away |
| Shade demo (music industry, private beta) | ⏳ Live on CPU, GPU RTF needed for real-time |
| Blog: conv2d bugs in MAX | Not started |
| Blog pitch to Modular | Not started |
