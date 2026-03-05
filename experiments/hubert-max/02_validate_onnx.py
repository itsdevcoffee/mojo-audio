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
