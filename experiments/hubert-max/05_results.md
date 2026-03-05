# HuBERT x MAX Engine: Experiment Results

**Date:** 2026-03-05
**System:** NVIDIA RTX 4060 Ti (8GB VRAM), Fedora Linux 43, CUDA 12.8
**Stack:** MAX 26.1.0 · PyTorch 2.10.0+cu128 · ONNXRuntime 1.24.2 · transformers 4.57.6

---

## 1. ONNX Export

**Model:** `facebook/hubert-base-ls960` (HuBERT Base, trained on LibriSpeech 960h)
This is the same architecture used by RVC v2 for content encoding. The HuggingFace version is architecturally identical to `lj1995/hubert_base.pt`.

**Export method:** `torch.onnx.export` with `dynamo=False` (legacy TorchScript exporter). PyTorch 2.10 defaults to `dynamo=True`, which requires `onnxscript` and is not needed here — the TorchScript path is more stable for this model.

**Output:** `experiments/hubert-max/hubert_base.onnx`
**Size:** 377.7 MB
**Opset:** 17
**Dynamic shapes:**
- Input: `input_values` — `[batch_size, sequence_length]` (float32)
- Output: `last_hidden_state` — `[batch_size, time_frames, 768]` (float32)

**Validation:** `onnx.checker.check_model()` passed with no errors.

For a 1-second audio clip at 16kHz (1×16000 input), the output shape is `(1, 49, 768)` — 49 time frames of 768-dimensional hidden states.

---

## 2. ONNX Correctness (PyTorch vs ONNXRuntime)

Input: 1 second of synthetic noise at 16kHz (seed=42), batch size 1.

| Metric              | Value           |
|---------------------|-----------------|
| Output shape (ORT)  | (1, 49, 768)    |
| Output shape (PT)   | (1, 49, 768)    |
| Max absolute diff   | 7.08e-05        |
| Mean absolute diff  | ~1e-06          |
| Pass (threshold 1e-4) | YES           |

The ONNX export is numerically correct. The 7.08e-05 max diff is well within the acceptable tolerance for float32 ONNX export — this is standard floating-point rounding from graph reordering and constant folding, not a correctness bug.

---

## 3. MAX Engine Inference

**Result: FAILED**

MAX Engine v26.1 does not support loading ONNX files. Both the full HuBERT ONNX (377.7 MB) and a minimal hand-crafted test ONNX produced the same error:

```
ValueError: cannot compile input with format input has unknown contents
```

**Root cause:** `engine.InferenceSession.load(path)` only accepts MAX's own compiled graph format — not ONNX, not TorchScript, not any other standard ML interchange format. The path-based API compiles MAX Graph IR (MLIR), and anything else is rejected at the format-detection stage.

**What MAX Engine v26.1 actually supports:**

1. **MAX Graph API** (`max.graph.Graph` + `max.graph.ops` + `max.nn`) — Build the model from scratch using MAX's own layer primitives. This is the primary intended workflow. It does NOT accept external model files.

2. **`max.torch.graph_op` decorator** — Wraps a specific PyTorch sub-operation to be compiled and dispatched as a MAX custom kernel. This is for individual ops (e.g., a custom attention kernel), not for running a whole model.

3. **`torch.compile` with Inductor + MAX custom ops** — MAX provides some custom kernels that the Inductor backend can call, but again this is for specific operations, not a full model inference path.

**There is no ONNX importer in MAX Engine v26.1.** The `engine.InferenceSession` API accepts only models that have been compiled through MAX's own toolchain.

---

## 4. Benchmark Results

**Conditions:** 20 iterations after 3 warmup iterations. Input: 1×16000 float32 (1 second at 16kHz), batch size 1. GPU timings use `torch.cuda.synchronize()`.

```
======================================================================
HuBERT Inference Benchmark
======================================================================
Model:      facebook/hubert-base-ls960
Input:      16000 samples (1s @16kHz) -- batch size 1
Iterations: 20 (after 3 warmup)
GPU:        NVIDIA GeForce RTX 4060 Ti

Backend                Mean (ms)   Std (ms)   P95 (ms)     vs CPU
-----------------------------------------------------------------
PyTorch CPU                82.3        3.4        89.3      1.00x
PyTorch GPU (RTX 4060Ti)    4.3        0.2         4.7     19.02x
ONNXRuntime CPU            51.0        2.4        55.4      1.61x

Note: MAX Engine v26.1 excluded -- no ONNX importer available.
      MAX would require HuBERT rewritten in MAX Graph API.
======================================================================
```

**Observations:**

- **PyTorch CPU (82.3 ms):** Baseline. Standard autograd-disabled inference with `torch.no_grad()`. Dominated by sequential CPU execution of transformer attention across 12 layers.
- **PyTorch GPU (4.3 ms):** 19x speedup over CPU. GPU execution time is well under 10ms — well within real-time requirements for a voice conversion pipeline targeting <50ms total latency.
- **ONNXRuntime CPU (51.0 ms):** 1.6x faster than PyTorch CPU. ORT applies kernel fusion at the ONNX graph level and uses its own thread pool tuned for inference. Notably, ORT CPU is still 12x slower than PyTorch GPU — the CPU vs GPU gap dominates.

---

## 5. Analysis

### Why is GPU 19x faster than CPU?

HuBERT's 12-layer transformer stack is dominated by multi-head self-attention. Self-attention requires `O(n^2)` operations over sequence length, implemented as large batched matrix multiplications (QKV projections, attention scores, output projections). These map directly onto GPU tensor cores — the RTX 4060 Ti executes them in massively parallel warps. On CPU, the same matmuls execute sequentially across a small thread pool, with cache misses from large weight matrices stalling the pipeline. The 768-dimensional hidden size across 49 time steps and 12 layers gives the GPU enough work to saturate its compute units, while the CPU cannot parallelize the same compute graph meaningfully.

### Why is ONNXRuntime CPU 1.6x faster than PyTorch CPU?

Three factors:

1. **No autograd overhead.** PyTorch with `torch.no_grad()` still carries per-tensor dispatch overhead from the autograd engine's bookkeeping. ORT has no autograd system — it compiles the computation graph once and executes it directly.

2. **Kernel fusion.** ORT's graph optimizer fuses adjacent operations (LayerNorm + GeLU, attention + softmax, etc.) into single kernels, reducing memory bandwidth and kernel launch overhead.

3. **Thread pool tuning.** ORT uses its own OpenMP/MKL-backed thread pool sized and pinned for inference workloads. PyTorch's CPU threading is more general-purpose.

### What would it take to run HuBERT via MAX Engine?

A complete reimplementation of HuBERT in MAX's Python Graph API. This means:

- Implement all HuBERT components using `max.graph.Graph`, `max.graph.ops`, and `max.nn`: feature extractor (CNN layers), feature projection, positional embedding, 12× transformer encoder blocks (multi-head attention + FFN + LayerNorm), final projection.
- Port all 377.7 MB of weights from the HuggingFace checkpoint into MAX tensor format.
- Validate outputs against PyTorch at each layer.

This is a substantial engineering effort — weeks of work, not a day task. There is no automated ONNX-to-MAX-Graph conversion path.

### Is there a "quick win" via MAX?

No path exists for running the full model. `max.torch.graph_op` only accelerates individual sub-operations (e.g., a single fused attention kernel). You could potentially replace specific attention heads or FFN layers with MAX custom ops and benchmark the delta, but this would not accelerate the majority of the model and introduces integration complexity with no guaranteed speedup over `torch.compile`.

For the Shade pipeline, this is not worth pursuing in MAX today.

---

## 6. Recommendation for Shade Pipeline

**Use PyTorch GPU (`cuda:0`) for HuBERT inference.**

The RTX 4060 Ti delivers 4.3 ms mean latency for a 1-second audio chunk. This is 19x faster than CPU and leaves substantial headroom for the rest of the voice conversion pipeline (pitch estimation, decoder, vocoder). Real-time inference at 16kHz chunked processing is comfortably achievable.

**Concrete setup:**

```python
model = HubertModel.from_pretrained("facebook/hubert-base-ls960").eval().cuda()

with torch.no_grad():
    hidden = model(waveform_chunk.cuda()).last_hidden_state
```

**Do not use ONNXRuntime for the GPU case.** ORT CPU (51 ms) is faster than PyTorch CPU but still 12x slower than PyTorch GPU. ORT does support GPU execution via CUDA execution provider, but there is no evidence it would close the gap meaningfully — and it adds a dependency and conversion maintenance cost without a hardware advantage.

**Do not invest in MAX Engine** for HuBERT at this time. MAX v26.1 has no ONNX importer, and reimplementing HuBERT in MAX Graph API is a multi-week effort with uncertain performance gains given the already-excellent GPU baseline.

**If GPU is unavailable** (CI machines, CPU-only environments): use ONNXRuntime CPU (`CPUExecutionProvider`) at 51 ms. It is significantly faster than PyTorch CPU and requires no code changes beyond swapping the inference call.

---

## 7. Next Steps

1. **Integrate GPU HuBERT into Shade's feature extraction pipeline.** Profile end-to-end with the full voice conversion chain (HuBERT → pitch → decoder → vocoder) to measure total latency.

2. **Evaluate batch processing.** This benchmark used batch size 1. For offline batch conversion, running larger batches through HuBERT on GPU may further amortize kernel launch costs. Profile batch sizes 4, 8, 16.

3. **Try `torch.compile` on the GPU model.** PyTorch's Inductor backend with `mode="reduce-overhead"` may reduce dispatch overhead for repeated calls, potentially shaving another 1–2 ms.

4. **Monitor MAX Engine releases.** Modular is actively developing MAX. If a future release adds an ONNX importer or automated model conversion, revisit this decision — MAX's custom attention kernels could potentially beat ORT CPU and may approach GPU-level performance on targeted hardware.

5. **Evaluate float16 on GPU.** The model currently runs in float32. Half-precision inference (`model.half()`) is common in production voice pipelines and could roughly halve memory bandwidth requirements, potentially reducing the 4.3 ms baseline further.
