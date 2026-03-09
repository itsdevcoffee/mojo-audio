# HuBERT × MAX Engine Experiment Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Determine whether MAX Engine's `InferenceSession` can run HuBERT inference on ONNX-exported weights, benchmark against PyTorch, and document the speedup (or failure modes).

**Architecture:** Export `facebook/hubert-base-ls960` from HuggingFace to ONNX with dynamic input shapes; load via `max.engine.InferenceSession` (CPU first, then GPU); compare output correctness and wall-clock latency against a PyTorch CPU and GPU baseline. All experiment scripts live in `experiments/hubert-max/` in the mojo-audio repo.

**Tech Stack:** Python 3.12, PyTorch 2.9.1+cu128, `transformers`, `onnx`, `onnxruntime` (validation only), `max.engine.InferenceSession` (26.1.0), pixi task runner. GPU: NVIDIA RTX 4060 Ti (CUDA 12.8).

---

## Environment Context

- **Working directory:** `/home/maskkiller/dev-coffee/repos/mojo-audio`
- **Python env:** `pixi run python` — this env already has MAX Engine + PyTorch + CUDA
- **MAX is already a pixi dependency** — no separate install needed
- **ONNX is NOT installed** — must be added to `pixi.toml`

---

## Task 1: Install Dependencies + Create Experiment Directory

**Files:**
- Modify: `pixi.toml` — add `onnx`, `onnxruntime`, `transformers` dependencies
- Create: `experiments/hubert-max/` directory
- Create: `experiments/hubert-max/README.md`

---

**Step 1: Add dependencies to `pixi.toml`**

Under `[dependencies]` in `pixi.toml`, add:

```toml
onnx = ">=1.16.0,<2"
onnxruntime = ">=1.20.0,<2"
transformers = ">=4.45.0,<5"
huggingface_hub = ">=0.25.0,<1"
```

**Step 2: Install**

```bash
pixi install
```

Expected: resolves without conflict (onnx/onnxruntime are CPU-only, no CUDA clash with PyTorch).

**Step 3: Verify**

```bash
pixi run python -c "import onnx, onnxruntime, transformers; print('onnx:', onnx.__version__); print('ort:', onnxruntime.__version__); print('transformers:', transformers.__version__)"
```

Expected: all three print versions, no errors.

**Step 4: Create directory + README**

```bash
mkdir -p experiments/hubert-max
```

Create `experiments/hubert-max/README.md`:

```markdown
# HuBERT × MAX Engine Experiment

Validates whether MAX Engine's InferenceSession can run HuBERT inference
from an ONNX-exported model and benchmarks it against PyTorch.

## Scripts (run in order)

1. `01_export_onnx.py` — Download HuBERT, export to ONNX
2. `02_validate_onnx.py` — Validate ONNX output matches PyTorch (onnxruntime)
3. `03_max_inference.py` — Load ONNX via MAX Engine, verify output
4. `04_benchmark.py` — Benchmark all backends, print comparison table
5. `05_results.md` — Findings (written after running benchmark)

## Run all

```bash
pixi run python experiments/hubert-max/01_export_onnx.py
pixi run python experiments/hubert-max/02_validate_onnx.py
pixi run python experiments/hubert-max/03_max_inference.py
pixi run python experiments/hubert-max/04_benchmark.py
```
```

**Step 5: Commit**

```bash
git add pixi.toml experiments/hubert-max/README.md
git commit -m "chore: add onnx/transformers deps and hubert-max experiment scaffold"
```

---

## Task 2: Export HuBERT to ONNX

**Files:**
- Create: `experiments/hubert-max/01_export_onnx.py`

HuBERT input: raw audio waveform `[batch, sequence_length]` at 16kHz.
HuBERT output: last hidden states `[batch, time_frames, 768]`.

The feature extractor (CNN) + transformer layers all use standard ops — no custom CUDA kernels — so ONNX export should work cleanly.

---

**Step 1: Create `experiments/hubert-max/01_export_onnx.py`**

```python
#!/usr/bin/env python3
"""
Export HuBERT base to ONNX with dynamic input shapes.

Downloads facebook/hubert-base-ls960 from HuggingFace on first run (~360MB).
Outputs: experiments/hubert-max/hubert_base.onnx

This is the model used in RVC v2 for content encoding.
The HuggingFace version is architecturally identical to lj1995/hubert_base.pt.
"""

import torch
import numpy as np
from pathlib import Path
from transformers import HubertModel

OUTPUT_PATH = Path(__file__).parent / "hubert_base.onnx"
MODEL_ID = "facebook/hubert-base-ls960"

# Synthetic 1-second audio at 16kHz (batch=1)
DUMMY_INPUT = torch.zeros(1, 16000)


def export():
    print(f"Loading {MODEL_ID} ...")
    model = HubertModel.from_pretrained(MODEL_ID)
    model.eval()

    print("Exporting to ONNX ...")
    with torch.no_grad():
        torch.onnx.export(
            model,
            (DUMMY_INPUT,),
            str(OUTPUT_PATH),
            input_names=["input_values"],
            output_names=["last_hidden_state"],
            dynamic_axes={
                "input_values": {0: "batch_size", 1: "sequence_length"},
                "last_hidden_state": {0: "batch_size", 1: "time_frames"},
            },
            opset_version=17,
            do_constant_folding=True,
        )

    size_mb = OUTPUT_PATH.stat().st_size / 1_000_000
    print(f"✓ Exported: {OUTPUT_PATH} ({size_mb:.1f} MB)")

    # Quick shape check
    import onnx
    model_proto = onnx.load(str(OUTPUT_PATH))
    onnx.checker.check_model(model_proto)
    print("✓ ONNX model validated (checker passed)")

    # Print input/output info
    print("\nModel inputs:")
    for inp in model_proto.graph.input:
        print(f"  {inp.name}: {[d.dim_param or d.dim_value for d in inp.type.tensor_type.shape.dim]}")
    print("Model outputs:")
    for out in model_proto.graph.output:
        print(f"  {out.name}: {[d.dim_param or d.dim_value for d in out.type.tensor_type.shape.dim]}")


if __name__ == "__main__":
    if OUTPUT_PATH.exists():
        print(f"ONNX already exists at {OUTPUT_PATH}, skipping export.")
        print("Delete it to re-export.")
    else:
        export()
```

**Step 2: Run it**

```bash
pixi run python experiments/hubert-max/01_export_onnx.py
```

Expected output (first run downloads ~360MB, takes ~30s):
```
Loading facebook/hubert-base-ls960 ...
Exporting to ONNX ...
✓ Exported: experiments/hubert-max/hubert_base.onnx (360.X MB)
✓ ONNX model validated (checker passed)

Model inputs:
  input_values: ['batch_size', 'sequence_length']
Model outputs:
  last_hidden_state: ['batch_size', 'time_frames', 768]
```

If ONNX export fails with unsupported ops: note which op, document it in `05_results.md`, and stop — MAX cannot run this model.

**Step 3: Commit**

```bash
git add experiments/hubert-max/01_export_onnx.py
git commit -m "feat: export HuBERT to ONNX (dynamic shapes, opset 17)"
```

---

## Task 3: Validate ONNX Output vs PyTorch

**Files:**
- Create: `experiments/hubert-max/02_validate_onnx.py`

Before testing MAX, confirm the ONNX export is correct by comparing outputs between PyTorch and ONNXRuntime. This is the ground truth comparison.

---

**Step 1: Create `experiments/hubert-max/02_validate_onnx.py`**

```python
#!/usr/bin/env python3
"""
Validate ONNX output matches PyTorch output.

Uses ONNXRuntime (not MAX) as an independent validator.
If this passes, the ONNX export is correct.
If MAX later differs, the issue is in MAX, not the export.
"""

import numpy as np
import torch
import onnxruntime as ort
from pathlib import Path
from transformers import HubertModel

ONNX_PATH = Path(__file__).parent / "hubert_base.onnx"
MODEL_ID = "facebook/hubert-base-ls960"

# Use a fixed seed for reproducibility
rng = np.random.default_rng(42)
INPUT_NP = rng.standard_normal((1, 16000)).astype(np.float32)
INPUT_PT = torch.from_numpy(INPUT_NP)


def run_pytorch():
    model = HubertModel.from_pretrained(MODEL_ID)
    model.eval()
    with torch.no_grad():
        out = model(INPUT_PT)
    return out.last_hidden_state.numpy()


def run_onnxruntime():
    assert ONNX_PATH.exists(), f"Run 01_export_onnx.py first: {ONNX_PATH}"
    session = ort.InferenceSession(str(ONNX_PATH), providers=["CPUExecutionProvider"])
    outputs = session.run(["last_hidden_state"], {"input_values": INPUT_NP})
    return outputs[0]


def main():
    print("Running PyTorch inference ...")
    pt_out = run_pytorch()
    print(f"  PyTorch output shape: {pt_out.shape}")
    print(f"  PyTorch output mean:  {pt_out.mean():.6f}")

    print("\nRunning ONNXRuntime inference ...")
    ort_out = run_onnxruntime()
    print(f"  ORT output shape: {ort_out.shape}")
    print(f"  ORT output mean:  {ort_out.mean():.6f}")

    # Compare
    max_diff = np.abs(pt_out - ort_out).max()
    mean_diff = np.abs(pt_out - ort_out).mean()
    print(f"\nMax absolute difference:  {max_diff:.8f}")
    print(f"Mean absolute difference: {mean_diff:.8f}")

    # Threshold: 1e-4 is acceptable for float32 ONNX export
    THRESHOLD = 1e-4
    if max_diff < THRESHOLD:
        print(f"✓ ONNX export is correct (max diff {max_diff:.2e} < {THRESHOLD:.0e})")
    else:
        print(f"✗ ONNX export has significant error (max diff {max_diff:.2e} >= {THRESHOLD:.0e})")
        raise SystemExit(1)


if __name__ == "__main__":
    main()
```

**Step 2: Run it**

```bash
pixi run python experiments/hubert-max/02_validate_onnx.py
```

Expected:
```
Running PyTorch inference ...
  PyTorch output shape: (1, 49, 768)
  PyTorch output mean:  0.XXXXXX
Running ONNXRuntime inference ...
  ORT output shape: (1, 49, 768)
  ORT output mean:  0.XXXXXX
Max absolute difference:  0.00001234
Mean absolute difference: 0.00000456
✓ ONNX export is correct (max diff 1.23e-05 < 1e-04)
```

If it fails: document the error in `05_results.md`, do not proceed to MAX testing.

**Step 3: Commit**

```bash
git add experiments/hubert-max/02_validate_onnx.py
git commit -m "feat: validate HuBERT ONNX output vs PyTorch (onnxruntime)"
```

---

## Task 4: MAX Engine Inference

**Files:**
- Create: `experiments/hubert-max/03_max_inference.py`

Load the ONNX model via `max.engine.InferenceSession`. Verify output shape and correctness against PyTorch ground truth.

**MAX API reference (v26.1):**
```python
from max import engine
session = engine.InferenceSession()           # defaults to CPU
model = session.load("path/to/model.onnx")   # loads ONNX
result = model.execute(input_values=array)   # keyword = ONNX input name
```

---

**Step 1: Create `experiments/hubert-max/03_max_inference.py`**

```python
#!/usr/bin/env python3
"""
Load HuBERT ONNX via MAX Engine InferenceSession.

Tests:
1. Model loads without error
2. Output shape matches PyTorch
3. Output values match PyTorch within 1e-3
4. Single inference latency

If MAX cannot load the model (unsupported ops), documents the error.
"""

import time
import numpy as np
import torch
from pathlib import Path
from transformers import HubertModel

ONNX_PATH = Path(__file__).parent / "hubert_base.onnx"
MODEL_ID = "facebook/hubert-base-ls960"

rng = np.random.default_rng(42)
INPUT_NP = rng.standard_normal((1, 16000)).astype(np.float32)
INPUT_PT = torch.from_numpy(INPUT_NP)


def get_pytorch_reference():
    model = HubertModel.from_pretrained(MODEL_ID)
    model.eval()
    with torch.no_grad():
        out = model(INPUT_PT)
    return out.last_hidden_state.numpy()


def run_max():
    assert ONNX_PATH.exists(), f"Run 01_export_onnx.py first: {ONNX_PATH}"

    print("Loading ONNX model via MAX Engine ...")
    try:
        from max import engine
        session = engine.InferenceSession()
        model = session.load(str(ONNX_PATH))
        print("✓ Model loaded successfully")
    except Exception as e:
        print(f"✗ MAX Engine failed to load model: {e}")
        print("\nThis may mean MAX doesn't support one or more ops in HuBERT.")
        print("Check the error above for unsupported op names.")
        raise

    print("\nRunning MAX inference ...")
    t0 = time.perf_counter()
    try:
        result = model.execute(input_values=INPUT_NP)
        t1 = time.perf_counter()
    except Exception as e:
        print(f"✗ MAX inference failed: {e}")
        raise

    # result is a dict of output_name -> numpy array
    output_key = list(result.keys())[0]
    max_out = result[output_key]

    print(f"  MAX output shape: {max_out.shape}")
    print(f"  MAX output mean:  {max_out.mean():.6f}")
    print(f"  MAX inference time: {(t1 - t0) * 1000:.1f} ms")

    return max_out, (t1 - t0)


def main():
    print("Getting PyTorch reference output ...")
    pt_out = get_pytorch_reference()
    print(f"  PyTorch output shape: {pt_out.shape}")

    print()
    max_out, max_latency = run_max()

    print("\n--- Correctness Check ---")
    if max_out.shape != pt_out.shape:
        print(f"✗ Shape mismatch: MAX {max_out.shape} vs PyTorch {pt_out.shape}")
        raise SystemExit(1)

    max_diff = np.abs(max_out - pt_out).max()
    mean_diff = np.abs(max_out - pt_out).mean()
    print(f"Max absolute difference:  {max_diff:.8f}")
    print(f"Mean absolute difference: {mean_diff:.8f}")

    # MAX may have slightly more fp32 rounding than ORT, allow 1e-3
    THRESHOLD = 1e-3
    if max_diff < THRESHOLD:
        print(f"✓ MAX Engine output correct (max diff {max_diff:.2e} < {THRESHOLD:.0e})")
    else:
        print(f"⚠ MAX Engine output differs (max diff {max_diff:.2e}) — may be acceptable")
        print("  Document in 05_results.md: check if diff is systematic or random")

    print(f"\n✓ MAX Engine inference: {max_latency * 1000:.1f} ms for 1s audio @16kHz")


if __name__ == "__main__":
    main()
```

**Step 2: Run it**

```bash
pixi run python experiments/hubert-max/03_max_inference.py
```

**Expected success output:**
```
Getting PyTorch reference output ...
  PyTorch output shape: (1, 49, 768)

Loading ONNX model via MAX Engine ...
✓ Model loaded successfully

Running MAX inference ...
  MAX output shape: (1, 49, 768)
  MAX output mean:  0.XXXXXX
  MAX inference time: XXX ms

--- Correctness Check ---
Max absolute difference:  0.0000XXXX
Mean absolute difference: 0.0000XXXX
✓ MAX Engine output correct
✓ MAX Engine inference: XXX ms for 1s audio @16kHz
```

**If MAX fails to load:** Copy the full error message. Look for "unsupported op" messages. Document in `05_results.md` with exact op names. This is valuable data even if it fails.

**If MAX loads but output differs significantly (> 1e-3):** Run a few more inputs to check if the error is consistent or random. Document in `05_results.md`.

**Step 3: Commit**

```bash
git add experiments/hubert-max/03_max_inference.py
git commit -m "feat: MAX Engine inference test for HuBERT ONNX"
```

---

## Task 5: Benchmark All Backends

**Files:**
- Create: `experiments/hubert-max/04_benchmark.py`

Run HuBERT inference across all available backends with warm-up and multiple iterations. Print a comparison table.

Backends to test:
- PyTorch CPU
- PyTorch GPU (RTX 4060 Ti)
- MAX Engine CPU
- ONNXRuntime CPU (reference)

---

**Step 1: Create `experiments/hubert-max/04_benchmark.py`**

```python
#!/usr/bin/env python3
"""
Benchmark HuBERT inference across all backends.

Outputs a comparison table of latency (mean, std, p95) and speedup vs PyTorch CPU.
Saves results to 05_results.md.

Run after 01, 02, 03 all pass.
"""

import time
import numpy as np
import torch
import onnxruntime as ort
from pathlib import Path
from transformers import HubertModel

ONNX_PATH = Path(__file__).parent / "hubert_base.onnx"
MODEL_ID = "facebook/hubert-base-ls960"
N_WARMUP = 3
N_ITERS = 20
INPUT_LEN = 16000  # 1 second at 16kHz

rng = np.random.default_rng(42)
INPUT_NP = rng.standard_normal((1, INPUT_LEN)).astype(np.float32)
INPUT_PT = torch.from_numpy(INPUT_NP)


def benchmark_fn(fn, n_warmup=N_WARMUP, n_iters=N_ITERS):
    """Run fn() n_warmup times (ignored), then n_iters times, return latencies in ms."""
    for _ in range(n_warmup):
        fn()

    latencies = []
    for _ in range(n_iters):
        t0 = time.perf_counter()
        fn()
        t1 = time.perf_counter()
        latencies.append((t1 - t0) * 1000)

    return np.array(latencies)


def benchmark_pytorch_cpu(model_id):
    model = HubertModel.from_pretrained(model_id).eval()

    def fn():
        with torch.no_grad():
            model(INPUT_PT)

    return benchmark_fn(fn)


def benchmark_pytorch_gpu(model_id):
    if not torch.cuda.is_available():
        return None
    model = HubertModel.from_pretrained(model_id).eval().cuda()
    input_gpu = INPUT_PT.cuda()

    def fn():
        with torch.no_grad():
            model(input_gpu)
        torch.cuda.synchronize()

    return benchmark_fn(fn)


def benchmark_ort_cpu():
    session = ort.InferenceSession(str(ONNX_PATH), providers=["CPUExecutionProvider"])

    def fn():
        session.run(["last_hidden_state"], {"input_values": INPUT_NP})

    return benchmark_fn(fn)


def benchmark_max_cpu():
    try:
        from max import engine
        session = engine.InferenceSession()
        model = session.load(str(ONNX_PATH))

        def fn():
            model.execute(input_values=INPUT_NP)

        return benchmark_fn(fn)
    except Exception as e:
        print(f"  MAX Engine failed: {e}")
        return None


def print_table(results):
    """Print results as a markdown table."""
    print("\n## Benchmark Results\n")
    print(f"Model: {MODEL_ID}")
    print(f"Input: {INPUT_LEN} samples (1s @16kHz)")
    print(f"Iterations: {N_ITERS} (after {N_WARMUP} warmup)")
    print(f"GPU: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'N/A'}")
    print()
    print(f"{'Backend':<22} {'Mean (ms)':>10} {'Std (ms)':>10} {'P95 (ms)':>10} {'Speedup vs PT-CPU':>18}")
    print("-" * 75)

    baseline_mean = None
    for name, latencies in results.items():
        if latencies is None:
            print(f"{name:<22} {'FAILED':>10}")
            continue
        mean = latencies.mean()
        std = latencies.std()
        p95 = np.percentile(latencies, 95)
        if baseline_mean is None:
            baseline_mean = mean
        speedup = baseline_mean / mean
        print(f"{name:<22} {mean:>10.1f} {std:>10.1f} {p95:>10.1f} {speedup:>17.2f}x")


def main():
    assert ONNX_PATH.exists(), f"Run 01_export_onnx.py first: {ONNX_PATH}"

    results = {}

    print("Benchmarking PyTorch CPU ...")
    results["PyTorch CPU"] = benchmark_pytorch_cpu(MODEL_ID)
    print(f"  mean: {results['PyTorch CPU'].mean():.1f} ms")

    print("Benchmarking PyTorch GPU ...")
    results["PyTorch GPU (RTX 4060Ti)"] = benchmark_pytorch_gpu(MODEL_ID)
    if results["PyTorch GPU (RTX 4060Ti)"] is not None:
        print(f"  mean: {results['PyTorch GPU (RTX 4060Ti)'].mean():.1f} ms")
    else:
        print("  CUDA not available, skipping")

    print("Benchmarking ONNXRuntime CPU ...")
    results["ONNXRuntime CPU"] = benchmark_ort_cpu()
    print(f"  mean: {results['ONNXRuntime CPU'].mean():.1f} ms")

    print("Benchmarking MAX Engine CPU ...")
    results["MAX Engine CPU"] = benchmark_max_cpu()
    if results["MAX Engine CPU"] is not None:
        print(f"  mean: {results['MAX Engine CPU'].mean():.1f} ms")

    print_table(results)

    # Save raw data for documentation
    output_file = Path(__file__).parent / "benchmark_raw.npz"
    save_dict = {k: v for k, v in results.items() if v is not None}
    np.savez(str(output_file), **{k.replace(" ", "_"): v for k, v in save_dict.items()})
    print(f"\nRaw data saved to {output_file}")
    print("\nNext: write findings to 05_results.md")


if __name__ == "__main__":
    main()
```

**Step 2: Run it**

```bash
pixi run python experiments/hubert-max/04_benchmark.py
```

Expected runtime: ~2-5 minutes (20 iters × 4 backends, PyTorch model loading included).

Expected output format:
```
Benchmarking PyTorch CPU ...
  mean: XXXX ms
Benchmarking PyTorch GPU ...
  mean: XX ms
Benchmarking ONNXRuntime CPU ...
  mean: XXX ms
Benchmarking MAX Engine CPU ...
  mean: XXX ms

## Benchmark Results

Model: facebook/hubert-base-ls960
Input: 16000 samples (1s @16kHz)
Iterations: 20 (after 3 warmup)
GPU: NVIDIA GeForce RTX 4060 Ti

Backend                Mean (ms)   Std (ms)   P95 (ms)  Speedup vs PT-CPU
---------------------------------------------------------------------------
PyTorch CPU               XXXX.X        X.X       XXXX.X              1.00x
PyTorch GPU (RTX 4060Ti)    XX.X        X.X         XX.X             XX.00x
ONNXRuntime CPU            XXX.X        X.X        XXX.X              X.XXx
MAX Engine CPU             XXX.X        X.X        XXX.X              X.XXx
```

**Step 3: Commit**

```bash
git add experiments/hubert-max/04_benchmark.py
git commit -m "feat: multi-backend HuBERT benchmark (PyTorch CPU/GPU, ORT, MAX)"
```

---

## Task 6: Document Findings

**Files:**
- Create: `experiments/hubert-max/05_results.md`

---

**Step 1: Run all scripts in order (if not already done)**

```bash
pixi run python experiments/hubert-max/01_export_onnx.py
pixi run python experiments/hubert-max/02_validate_onnx.py
pixi run python experiments/hubert-max/03_max_inference.py
pixi run python experiments/hubert-max/04_benchmark.py
```

**Step 2: Create `experiments/hubert-max/05_results.md`** with actual numbers from the benchmark. Template:

```markdown
# HuBERT × MAX Engine: Experiment Results

**Date:** 2026-03-05
**System:** RTX 4060 Ti (8GB), Fedora 43, CUDA 12.8, MAX 26.1.0, PyTorch 2.9.1

## ONNX Export

- [ ] Export succeeded / failed
- Model size: X MB
- Opset: 17
- Dynamic shapes: ✓/✗
- ONNX checker: ✓/✗
- Unsupported ops (if any): [list them]

## Correctness (MAX vs PyTorch)

- MAX Engine loaded successfully: ✓/✗
- Output shape match: ✓/✗
- Max absolute difference: X.XXe-XX
- Assessment: ✓ Correct / ⚠ Minor diff / ✗ Wrong

## Benchmark Results

[Paste the table from 04_benchmark.py output here]

## MAX Engine vs ORT

MAX speedup over ONNXRuntime CPU: X.XXx
MAX speedup over PyTorch CPU: X.XXx

## Conclusion

[One paragraph: can MAX run HuBERT? Is it faster than ORT? What's the recommendation
for integrating into the Shade VC pipeline?]

## Recommendation for Shade

[Specific recommendation: use MAX / use ORT / use PyTorch GPU / investigate further]
If MAX works:
- Integration path: [describe]
- Next step: test RMVPE (pitch extractor) via same ONNX path

If MAX fails:
- Which op failed: [name]
- Is a custom Mojo op feasible: [yes/no + reasoning]
- Fallback: [PyTorch GPU / ORT + mojo-audio DSP bookends]
```

**Step 3: Also copy findings to the shade research docs**

```bash
cp experiments/hubert-max/05_results.md /home/maskkiller/repos/shade/docs/research/05-max-engine-hubert-experiment.md
```

**Step 4: Final commit**

```bash
git add experiments/hubert-max/
git commit -m "docs: HuBERT × MAX Engine experiment results"
```

---

## Summary

After completing all 6 tasks, you will know:

1. **Whether MAX Engine can load HuBERT ONNX** (go/no-go for Option B)
2. **Whether outputs are correct** (numerical validation)
3. **Exact speedup vs PyTorch CPU and GPU**
4. **Whether MAX is faster than ONNXRuntime** (the fair comparison since both use ONNX)
5. **A concrete recommendation** for how to integrate HuBERT into the Shade VC pipeline

If MAX works: the next experiment is RMVPE (pitch extractor) — same ONNX path.
If MAX fails on an op: assess whether a custom Mojo op is worth implementing or whether PyTorch GPU (already available) is the right call.
