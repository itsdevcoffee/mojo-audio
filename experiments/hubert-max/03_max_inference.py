#!/usr/bin/env python3
"""
Load HuBERT ONNX via MAX Engine InferenceSession.

Tests:
1. Model loads without error
2. Output shape matches PyTorch (1, 49, 768) for 1s audio
3. Output values match PyTorch within 1e-3
4. Single inference latency

If MAX cannot load the model (unsupported ops or format), documents the error.

--- MAX Engine v26.1 Investigation Notes ---

MAX Engine v26.1 does NOT support loading ONNX or TorchScript files directly
via session.load(path). The `session.load()` path-based API only accepts
MAX's own compiled model format (MAX MLIR).

The supported workflows in MAX v26.1 are:
1. Build a MAX Graph using max.graph.Graph + max.graph.ops + max.nn
2. Use max.torch.graph_op to wrap specific PyTorch ops with MAX kernels
3. Use torch.compile with inductor backend and MAX custom ops

For ONNX inference, the alternatives are:
- ONNX Runtime (onnxruntime) - what task 05_benchmark.py will use
- Re-implement the model using MAX Graph API manually

Error observed: ValueError: cannot compile input with format input has unknown contents
This error occurs for both ONNX and TorchScript formats.
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
        from max.driver import CPU
        session = engine.InferenceSession(devices=[CPU()])
        model = session.load(str(ONNX_PATH))
        print("Model loaded successfully")
    except Exception as e:
        print(f"MAX Engine failed to load model:")
        print(f"  {type(e).__name__}: {e}")
        print()
        print("ROOT CAUSE: MAX Engine v26.1 does not support loading ONNX files")
        print("via session.load(path). It only accepts MAX Graph objects (max.graph.Graph)")
        print("or MAX's own compiled model format.")
        print()
        print("See module docstring for details and workarounds.")
        raise

    print("\nRunning MAX inference ...")
    t0 = time.perf_counter()
    try:
        result = model.execute(input_values=INPUT_NP)
        t1 = time.perf_counter()
    except Exception as e:
        print(f"MAX inference failed:")
        print(f"  {type(e).__name__}: {e}")
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
    try:
        max_out, max_latency = run_max()
    except Exception as e:
        print(f"\n=== MAX ENGINE RESULT: FAILED ===")
        print(f"Error: {type(e).__name__}: {e}")
        print()
        print("FINDING: MAX Engine v26.1 cannot load ONNX models directly.")
        print("It uses its own Graph IR (max.graph.Graph) and compiles models")
        print("from source using MAX's nn/graph APIs, not from ONNX.")
        print()
        print("This is documented in 05_results.md -- not a blocker for the experiment.")
        print("ONNX Runtime will be used for the multi-backend benchmark (Task 5).")
        return

    print("\n--- Correctness Check ---")
    if max_out.shape != pt_out.shape:
        print(f"Shape mismatch: MAX {max_out.shape} vs PyTorch {pt_out.shape}")
        return

    max_diff = np.abs(max_out - pt_out).max()
    mean_diff = np.abs(max_out - pt_out).mean()
    print(f"Max absolute difference:  {max_diff:.8f}")
    print(f"Mean absolute difference: {mean_diff:.8f}")

    THRESHOLD = 1e-3
    if max_diff < THRESHOLD:
        print(f"MAX Engine output correct (max diff {max_diff:.2e} < {THRESHOLD:.0e})")
    else:
        print(f"MAX Engine output differs (max diff {max_diff:.2e})")
        print("  Documenting in 05_results.md for analysis")

    print(f"\nMAX Engine inference: {max_latency * 1000:.1f} ms for 1s audio @16kHz")
    print(f"\n=== MAX ENGINE RESULT: SUCCESS ===")


if __name__ == "__main__":
    main()
