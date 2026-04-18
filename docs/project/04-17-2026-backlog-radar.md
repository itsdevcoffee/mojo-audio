# mojo-audio Backlog Radar — 2026-04-18

**Context:** Sprints 1–5 complete. Full mojo-audio pipeline runs on Spark
GPU at RTF 0.36 (mean across 4 models). However, Applio GPU runs at RTF
0.15 — **2.4x faster** — because mojo-audio's im2col conv workaround tanks
GPU throughput vs PyTorch's native cuDNN conv kernels. **Decision (04-18):
Shade stays on Applio for production. mojo-audio continues development;
switch when benchmarks match or beat Applio.** Dual-Spark NCCL resolved.

---

## Priority 1 — Shade Production on Applio + Dual-Spark Job Routing

Shade runs Applio (PyTorch CUDA) for voice conversion in production. The
immediate work is making this production-ready across both Sparks with
proper job routing, supervision, retry logic, and monitoring.

| # | Task | Est. | Status |
|---|---|---|---|
| 1 | **Revert Shade to Applio** — change `services/rvc.py` back to `USE_APPLIO=1` or Applio-direct. Confirm Shade API returns to Applio GPU perf (RTF ~0.15). | 15 min | Ready |
| 2 | **Add Shade to endpoints.yaml** — register Shade API as a supervised endpoint in `visage-maximus-tag-team`. pm2 on visage, systemd on maximus. Replaces nohup. Survives reboots. | 1 hr | Ready |
| 3 | **Dual-Spark job routing** — dispatch `/convert` requests across both Sparks (visage + maximus). Load balancing, health checks, retry on failure. Options: Ray, custom FastAPI proxy, or Caddy/nginx reverse proxy with health. | Design needed | See §Discussion below |
| 4 | **Job tracking + retry** — track conversion jobs (queued, running, completed, failed). Retry failed jobs. Expose status via API for frontend. | Design needed | See §Discussion below |
| 5 | **Shade frontend deployment** — fix `BODY_SIZE_LIMIT=Infinity` persistence, systemd/pm2 for the Node frontend. | 30 min | Ready |

### Shade Applio GPU benchmark (04-18, 3s real vocal)

| Model | Applio GPU RTF | mojo-audio GPU RTF | Applio advantage |
|---|---|---|---|
| the-weeknd | 0.154 | 0.367 | 2.4x |
| ariana-grande | 0.151 | 0.314 | 2.1x |
| adele | 0.144 | 0.353 | 2.5x |
| frank-sinatra | 0.160 | 0.408 | 2.6x |
| **Mean** | **0.152** | **0.360** | **2.4x** |

---

## Priority 2 — mojo-audio Development (Switch When Ready)

mojo-audio is architecturally complete and numerically correct. The gap is
GPU perf — im2col workaround vs native conv. Continue development; re-run
benchmarks after each improvement or MAX nightly update. Switch Shade when
`benchmark_suite.py` shows mojo-audio matching or beating Applio.

**Switch criteria:** mojo-audio GPU RTF ≤ Applio GPU RTF on the same models
(currently need ~2.4x improvement).

| # | Task | Impact | Notes |
|---|---|---|---|
| 1 | **Swap im2col → native `ops.conv2d` on aarch64** — 04-11 audit verified conv2d is fixed on aarch64 for C_in≥8. This is the single biggest RTF unlock. Profile per-layer. | High — closes most of 2.4x gap | `_rmvpe.py`, `_hifigan_graph.py` |
| 2 | **Track MAX nightly fixes** — periodically bump MAX pin and re-run `benchmark_suite.py`. The bmm rebind bug, conv2d improvements, and new GPU kernel paths may close the gap for free. | Medium | `pixi.toml` |
| 3 | **Output RMS normalization** — match output volume to input (Applio's volume_envelope). Biggest quality gap (+13 dB). | Quality | `voice_converter.py` |
| 4 | **FAISS index retrieval** — speaker similarity blending. Applio uses index_rate=0.75 in production. | Quality | New file |
| 5 | **RVC v1 support** — `sza`, `brent-faiyaz`, `giveon` fail (256 hidden channels). | Compatibility | `_vits_graph.py` |
| 6 | **File MAX bugs** — ops.rebind GPU rank mismatch, update #6248 comment. | Community | — |

### Benchmark scripts (ready to use)

```bash
# mojo-audio GPU only (fast, no Applio needed)
pixi run python scripts/benchmark_suite.py run --mojo-only --mojo-device gpu

# Applio GPU baseline
pixi run python scripts/benchmark_applio_baseline.py --model X --audio Y --device cuda:0

# Head-to-head comparison (speed + quality)
pixi run python scripts/compare_vs_applio.py --model X --audio Y --mojo-device gpu

# View historical trends
pixi run python scripts/benchmark_suite.py history
```

### Benchmark results

First GPU run: `benchmarks/results/20260418T012214.json` (4 models, mean RTF 0.360).

---

## Deferred — Technical Debt

| Issue | Severity | Notes |
|---|---|---|
| HiFiGAN `batch>1` blocked (ConvTranspose1d zero-interleave requires B=1) | Medium | Blocks concurrent request processing. |
| GroupNorm approximated as LayerNorm in CNN | Low | — |
| ContentVec `weight_g/weight_v` detection robustness | Medium | Before shipping to external users. |
| NSF-HiFiGAN harmonic source: simplified numpy sine (0.66 corr vs PyTorch SineGen) | Low | Neural filter itself is 0.9998 corr. |
| Stale docstring at `_rmvpe.py:37-42` | Low | Pre-fix residual block structure still documented. |
| Griffin-Lim lacks true complex phase | Low | — |

---

## Deferred — Multi-Spark Model Sharding (Sprint 6)

Separate from job routing (Priority 1). This is about sharding a single
model across both GPUs for lower per-request latency.

| Task | Status | Notes |
|---|---|---|
| NCCL bandwidth issue | ✅ Resolved | Chris fixed 04-15. |
| Shard VITS via `max.nn.DistributedTransformer` | Not started | 256GB unified memory pool. |
| Blocked by NCCL bandwidth for TP | Open | PP=2/TP=1 is the safe config. |

---

## Community / Visibility

| Milestone | Status |
|---|---|
| MAX bug filed (conv2d groups) — `modular/modular#6129` | ✅ |
| MAX bug filed (conv2d C_in≥8) — `modular/modular#6248` | ✅ Filed, pending comment update |
| Full mojo-audio pipeline on Spark GPU | ✅ (04-17) — RTF 0.36, 3.49x faster than CPU |
| mojo-audio vs Applio formal GPU comparison | ✅ (04-18) — Applio 2.4x faster, im2col gap identified |
| Shade production decision | ✅ (04-18) — stay on Applio until mojo-audio matches perf |
| Blog: conv2d bugs in MAX | Not started |
| Blog pitch to Modular | Not started |
