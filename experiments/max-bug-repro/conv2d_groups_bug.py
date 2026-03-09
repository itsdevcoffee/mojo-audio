#!/usr/bin/env python3
"""
Minimal repro for MAX Engine conv2d groups bug.

Expected: MAX grouped conv2d output ≈ numpy reference (max diff < 0.001)
Actual:   MAX grouped conv2d output differs wildly (max diff > 100.0)

System: MAX 26.1.0.dev2026010718, DGX Spark GB10 SM_121 ARM64, CUDA 13.0
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../../src"))

import numpy as np
from max import engine
from max.driver import CPU, Accelerator, Tensor, accelerator_count
from max.graph import Graph, TensorType, ops, DeviceRef, Dim
from max.dtype import DType


def numpy_grouped_conv1d_reference(x, filt, groups, padding):
    """Pure numpy grouped conv1d. Ground truth."""
    B, L, _, C_in = x.shape
    K = filt.shape[0]
    C_out = filt.shape[3]
    cin_g = C_in // groups
    cout_g = C_out // groups

    x_pad = np.pad(x, ((0,0), (padding, padding), (0,0), (0,0)))
    L_out = L + 2*padding - K + 1
    out = np.zeros((B, L_out, 1, C_out), dtype=np.float32)

    for g in range(groups):
        w_g = filt[:, :, :, g*cout_g:(g+1)*cout_g]  # [K, 1, cin_g, cout_g]
        w_flat = w_g.reshape(-1, cout_g)
        for l in range(L_out):
            patch = x_pad[:, l:l+K, :, g*cin_g:(g+1)*cin_g]
            out[:, l, 0, g*cout_g:(g+1)*cout_g] = patch.reshape(B, -1) @ w_flat

    return out


def run_max_grouped_conv(x_np, filt_np, groups, padding, use_gpu=False):
    """Run grouped conv2d via MAX Graph."""
    B, L, _, C_in = x_np.shape

    if use_gpu and accelerator_count() > 0:
        dev = Accelerator()
        dev_ref = DeviceRef.GPU(0)
    else:
        dev = CPU()
        dev_ref = DeviceRef.CPU()

    with Graph(
        "grouped_conv_test",
        input_types=[TensorType(DType.float32, [B, Dim("L"), 1, C_in], dev_ref)],
    ) as g:
        x = g.inputs[0]
        w = ops.constant(filt_np, device=dev_ref)
        out = ops.conv2d(x, w, stride=(1, 1), padding=(padding, padding, 0, 0), groups=groups)
        g.output(out)

    session = engine.InferenceSession(devices=[dev])
    model = session.load(g)

    if use_gpu and accelerator_count() > 0:
        inp = Tensor.from_numpy(x_np).to(dev)
    else:
        inp = x_np

    result = model.execute(inp)
    tensor = list(result.values())[0] if isinstance(result, dict) else result[0]
    return tensor.to_numpy()


def main():
    print("MAX Engine conv2d groups bug repro")
    print(f"MAX version: {engine.__version__}")
    print(f"GPU available: {accelerator_count() > 0}")
    print()

    rng = np.random.default_rng(42)

    # ===== Case 1: K=1 (control — should work) =====
    print("Case 1: groups=16, K=1, C_in=768 (control)")
    B, L, C_in, C_out, K, G = 1, 49, 768, 768, 1, 16
    cin_g = C_in // G
    x1 = rng.standard_normal((B, L, 1, C_in)).astype(np.float32)
    w1 = rng.standard_normal((K, 1, cin_g, C_out)).astype(np.float32)
    ref1 = numpy_grouped_conv1d_reference(x1, w1, G, padding=0)
    max1 = run_max_grouped_conv(x1, w1, G, padding=0, use_gpu=True)
    diff1 = np.abs(max1 - ref1).max()
    print(f"  Max diff (K=1):  {diff1:.6f}  {'PASS' if diff1 < 0.01 else 'FAIL'}")

    # ===== Case 2: K=3, small kernel =====
    print()
    print("Case 2: groups=16, K=3, C_in=768")
    K2 = 3
    w2 = rng.standard_normal((K2, 1, cin_g, C_out)).astype(np.float32)
    ref2 = numpy_grouped_conv1d_reference(x1, w2, G, padding=1)
    max2 = run_max_grouped_conv(x1, w2, G, padding=1, use_gpu=True)
    diff2 = np.abs(max2 - ref2[:, :L, :, :]).max()
    print(f"  Max diff (K=3):  {diff2:.6f}  {'PASS' if diff2 < 0.01 else 'FAIL'}")

    # ===== Case 3: K=128, large kernel — the failing case =====
    print()
    print("Case 3: groups=16, K=128, C_in=768 (pos_conv from HuBERT — BUG)")
    K3 = 128
    w3 = rng.standard_normal((K3, 1, cin_g, C_out)).astype(np.float32)
    ref3 = numpy_grouped_conv1d_reference(x1, w3, G, padding=64)
    max3_cpu = run_max_grouped_conv(x1, w3, G, padding=64, use_gpu=False)
    max3_gpu = run_max_grouped_conv(x1, w3, G, padding=64, use_gpu=True)
    diff_cpu = np.abs(max3_cpu[:, :L, :, :] - ref3[:, :L, :, :]).max()
    diff_gpu = np.abs(max3_gpu[:, :L, :, :] - ref3[:, :L, :, :]).max()
    print(f"  Max diff CPU (K=128): {diff_cpu:.6f}  {'PASS' if diff_cpu < 0.01 else 'BUG CONFIRMED'}")
    print(f"  Max diff GPU (K=128): {diff_gpu:.6f}  {'PASS' if diff_gpu < 0.01 else 'BUG CONFIRMED'}")

    # ===== Summary =====
    print()
    print("=" * 60)
    print("SUMMARY")
    print(f"  K=1   CPU diff: {diff1:.2e}  (expected < 1e-3)")
    print(f"  K=3   GPU diff: {diff2:.2e}  (expected < 1e-3)")
    print(f"  K=128 CPU diff: {diff_cpu:.2e}  (expected < 1e-3)")
    print(f"  K=128 GPU diff: {diff_gpu:.2e}  (expected < 1e-3)")
    print()
    if diff_cpu > 0.01 or diff_gpu > 0.01:
        print("BUG: ops.conv2d with groups > 1 and K > small threshold")
        print("     produces incorrect results. K=1 works, large K does not.")
        print()
        print("IMPACT: HuBERT/ContentVec pos_conv layer (groups=16, K=128)")
        print("        cannot run correctly on GPU. Workaround: numpy bridge.")
        print()
        print("ENVIRONMENT:")
        import platform, subprocess
        print(f"  Platform: {platform.machine()}")
        try:
            gpu = subprocess.run(
                ["nvidia-smi", "--query-gpu=name,compute_cap", "--format=csv,noheader"],
                capture_output=True, text=True, timeout=5
            ).stdout.strip()
            print(f"  GPU: {gpu}")
        except Exception:
            pass
        print(f"  MAX version: {engine.__version__}")


if __name__ == "__main__":
    main()
