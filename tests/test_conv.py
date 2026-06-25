"""Tests for src/models/_conv.py — native express-as-plain conv primitives.

All assertions use THRESH=1e-4 (GPU fp32 accumulation is ~2-4e-5; that is NOT a bug).
Must run on the Spark GPU (aarch64) — local x64 ops.conv2d is broken.
"""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import numpy as np
import torch
import torch.nn.functional as F
from max.driver import Accelerator, Buffer
from max.dtype import DType
from max.engine import InferenceSession
from max.graph import Graph, TensorType, DeviceRef
from models import _conv

ACCEL = Accelerator()
DEV = DeviceRef.GPU()
SESSION = InferenceSession(devices=[ACCEL])
THRESH = 1e-4  # GPU fp32 accumulation ~2-4e-5 at high fan-in (Probe 2 confirmed)


def _exec(build, x):
    with Graph("t", input_types=[TensorType(DType.float32, list(x.shape), DEV)]) as g:
        g.output(build(g.inputs[0]))
    model = SESSION.load(g)
    gpu_x = Buffer.from_numpy(x).to(ACCEL)
    return model.execute(gpu_x)[0].to_numpy()


def test_conv1d_dilated_matches_torch():
    rng = np.random.default_rng(0)
    C_in, C_out, K, T, d = 64, 96, 3, 40, 3
    x = rng.standard_normal((1, T, 1, C_in)).astype(np.float32)
    w = rng.standard_normal((C_out, C_in, K)).astype(np.float32)
    b = rng.standard_normal((C_out,)).astype(np.float32)
    ym = _exec(lambda inp: _conv.conv1d(inp, w, b, dilation=d, device_ref=DEV), x)
    xt = torch.from_numpy(x).squeeze(2).transpose(1, 2)
    yt = F.conv1d(xt, torch.from_numpy(w), torch.from_numpy(b),
                  padding=d * (K - 1) // 2, dilation=d).transpose(1, 2).unsqueeze(2).numpy()
    assert np.abs(ym - yt).max() < THRESH


def test_conv_transpose1d_matches_torch():
    rng = np.random.default_rng(1)
    C_in, C_out, K, S, T = 64, 64, 4, 2, 20
    x = rng.standard_normal((1, T, 1, C_in)).astype(np.float32)
    w = rng.standard_normal((C_in, C_out, K)).astype(np.float32)
    b = rng.standard_normal((C_out,)).astype(np.float32)
    ym = _exec(lambda inp: _conv.conv_transpose1d(inp, w, b, stride=S, device_ref=DEV), x)
    xt = torch.from_numpy(x).squeeze(2).transpose(1, 2)
    yt = F.conv_transpose1d(xt, torch.from_numpy(w), torch.from_numpy(b),
                            stride=S, padding=(K - S) // 2).transpose(1, 2).unsqueeze(2).numpy()
    n = min(ym.shape[1], yt.shape[1])
    assert np.abs(ym[:, :n] - yt[:, :n]).max() < THRESH


def test_conv_transpose2d_matches_torch():
    rng = np.random.default_rng(2)
    C_in, C_out, H, W = 32, 32, 8, 8
    x = rng.standard_normal((1, H, W, C_in)).astype(np.float32)
    w = rng.standard_normal((C_in, C_out, 3, 3)).astype(np.float32)
    b = rng.standard_normal((C_out,)).astype(np.float32)
    ym = _exec(lambda inp: _conv.conv_transpose2d(inp, w, b, stride=2, device_ref=DEV), x)
    xt = torch.from_numpy(x).permute(0, 3, 1, 2)
    yt = F.conv_transpose2d(xt, torch.from_numpy(w), torch.from_numpy(b),
                            stride=2, padding=1, output_padding=1).permute(0, 2, 3, 1).numpy()
    assert ym.shape == yt.shape and np.abs(ym - yt).max() < THRESH
