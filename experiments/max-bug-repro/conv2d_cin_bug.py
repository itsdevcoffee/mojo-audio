"""Repro: ops.conv2d produces incorrect results when C_in >= 8 (groups=1).

MAX 26.3.0.dev2026032005, tested on CPU (x86_64) and GPU (RTX 4060 Ti).

The bug: conv2d output diverges from the mathematically correct result
(verified against both PyTorch and manual numpy convolution) when the
number of input channels is >= 8. The error grows with C_in.

C_in=4: exact match (0.0 diff)
C_in=8: max diff 0.020 (significant)
C_in=16: max diff 0.026
C_in=32: max diff 0.055
C_in=64: max diff 0.064
C_in=192: max diff 0.155

This blocks NSF-HiFiGAN (HiFiGAN vocoder for RVC voice conversion) from
producing numerically correct output, since all convolutions in the
network have C_in >= 32.

Usage:
    pixi run python experiments/max-bug-repro/conv2d_cin_bug.py
"""
import numpy as np
from max import engine
from max.graph import Graph, TensorType, ops, DeviceRef, Dim
from max.dtype import DType
from max.driver import CPU

def numpy_conv2d(x_nhwc, w_rscf, pad_h):
    """Ground-truth Conv2d (NHWC input, RSCF filter) via numpy loops."""
    B, H, W, C_in = x_nhwc.shape
    R, S, C_in_f, C_out = w_rscf.shape
    assert C_in == C_in_f and S == 1 and W == 1
    # Pad H dim
    x_pad = np.pad(x_nhwc, ((0,0), (pad_h, pad_h), (0,0), (0,0)))
    H_out = H  # same padding
    out = np.zeros((B, H_out, 1, C_out), dtype=np.float32)
    for b in range(B):
        for t in range(H_out):
            for co in range(C_out):
                val = 0.0
                for ci in range(C_in):
                    for r in range(R):
                        val += w_rscf[r, 0, ci, co] * x_pad[b, t + r, 0, ci]
                out[b, t, 0, co] = val
    return out

def max_conv2d(x_nhwc, w_rscf, pad_h):
    """Run the same conv via MAX Graph ops.conv2d."""
    B, H, W, C_in = x_nhwc.shape
    R, S, C_in_f, C_out = w_rscf.shape
    cpu_ref = DeviceRef.CPU()
    with Graph(
        "conv_test",
        input_types=[TensorType(DType.float32, [B, Dim("T"), 1, C_in], cpu_ref)],
    ) as g:
        inp = g.inputs[0]
        padded = ops.pad(inp, [0, 0, pad_h, pad_h, 0, 0, 0, 0])
        w = ops.constant(w_rscf, device=cpu_ref)
        out = ops.conv2d(padded, w, stride=(1, 1))
        g.output(out)
    model = engine.InferenceSession(devices=[CPU()]).load(g)
    result = model.execute(np.ascontiguousarray(x_nhwc))
    tensor = list(result.values())[0] if isinstance(result, dict) else result[0]
    return tensor.to_numpy()

def run_test(C_in, C_out, K, T=5):
    """Compare MAX conv2d against numpy ground truth."""
    rng = np.random.default_rng(42)
    # PyTorch-format weight [C_out, C_in, K] -> RSCF [K, 1, C_in, C_out]
    w_pt = rng.standard_normal((C_out, C_in, K)).astype(np.float32) * 0.01
    w_rscf = np.transpose(w_pt, (2, 1, 0))[:, np.newaxis, :, :]

    # NHWC input [1, T, 1, C_in]
    x_nhwc = rng.standard_normal((1, T, 1, C_in)).astype(np.float32) * 0.1

    pad = (K - 1) // 2

    np_out = numpy_conv2d(x_nhwc, w_rscf, pad)
    max_out = max_conv2d(x_nhwc, w_rscf, pad)

    diff = np.abs(np_out - max_out).max()
    status = "PASS" if diff < 1e-5 else "FAIL"
    print(f"  C_in={C_in:>3}, C_out={C_out:>3}, K={K}: max_diff={diff:.8f}  [{status}]")
    return diff

if __name__ == "__main__":
    print("=" * 70)
    print("MAX conv2d C_in bug repro")
    print("=" * 70)
    print(f"\nMAX version: 26.3.0.dev2026032005")
    print(f"All tests use groups=1, stride=1, 'same' padding, float32.\n")

    print("Test 1: Sweep C_in (K=3, C_out=2*C_in)")
    print("-" * 50)
    for C_in in [2, 4, 6, 8, 16, 32, 64, 128, 192]:
        run_test(C_in, C_in * 2, K=3)

    print("\nTest 2: Sweep C_in (K=7, C_out=512) — matches HiFiGAN conv_pre")
    print("-" * 50)
    for C_in in [4, 8, 32, 64, 192]:
        run_test(C_in, 512, K=7)

    print("\nTest 3: Fixed C_in=256, sweep K — check if kernel size matters")
    print("-" * 50)
    for K in [3, 5, 7, 11]:  # K=1 hits separate RSCF layout bug, skip
        run_test(256, 128, K=K)
