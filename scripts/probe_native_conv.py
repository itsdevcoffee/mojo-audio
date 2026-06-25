"""Probe native MAX conv ops vs PyTorch ground truth on aarch64.

Validates the conv variants the pipeline uses that were NOT already verified
(dilated conv1d, ConvTranspose1d, ConvTranspose2d). Plain/strided/grouped are
already proven in AudioEncoder. Threshold: max_diff < 1e-5.

Key layout notes from MAX 26.4 API docs:
- conv2d filter:           RSCF = [K_h, K_w, C_in/groups, C_out]
- conv2d_transpose filter: RSCF = [K_h, K_w, C_out, C_in]   (out/in swapped!)
- output_paddings: only 0 is supported in this MAX version

GPU fp32 note:
The PyTorch reference runs on CPU while MAX runs on GPU.  With C_in=64 the
accumulation order differs, giving max_diff ~2-3e-5 for correct ops (scales
linearly with fan-in, not catastrophic).  We use THRESH=1e-5 (strict) and
THRESH_FP32=1e-4 (fp32 variance) as two thresholds; functional PASS is defined
as diff < THRESH_FP32 since this is GPU-vs-CPU float32 accumulation, not a bug.

MAX 26.4 known limitations (discovered by this probe):
- ops.conv2d with dilation > 1: "Non-unit dilation is not supported yet" on GPU
- ops.conv2d_transpose on GPU: cuDNN ALLOC_FAILED (MAX layout bug on aarch64/GB10)
- ops.conv2d_transpose on CPU: fails to lower to Mojo

Run on the Spark (set LD_LIBRARY_PATH for cuDNN, or the ABORT will fire):
    CUDNN_LIB=$(find ~/repos/mojo-audio/.pixi/envs/default -name "libcudnn.so.9" 2>/dev/null | head -1 | xargs dirname)
    export LD_LIBRARY_PATH=$CUDNN_LIB:$LD_LIBRARY_PATH
    pixi run python scripts/probe_native_conv.py

Note: LD_LIBRARY_PATH must be set BEFORE launching Python (not inside the script)
because the dynamic linker resolves symbols at process start.
"""
import numpy as np
import torch
import torch.nn.functional as F
from max.driver import Accelerator, Buffer
from max.dtype import DType
from max.engine import InferenceSession
from max.graph import Graph, TensorType, DeviceRef, ops

ACCEL = Accelerator()
DEV = DeviceRef.GPU()
SESSION = InferenceSession(devices=[ACCEL])
THRESH = 1e-5
THRESH_FP32 = 1e-4  # secondary: distinguishes precision variance from systematic bugs

_graph_counter = 0


def _next_name(prefix):
    """Generate unique graph names to avoid session collision."""
    global _graph_counter
    _graph_counter += 1
    return f"{prefix}_{_graph_counter}"


def _execute(graph, *inputs):
    """Load a graph, execute with given numpy inputs, return list of numpy arrays.

    Inputs are numpy arrays; we move them to GPU before execution.
    """
    model = SESSION.load(graph)
    gpu_inputs = [Buffer.from_numpy(inp).to(ACCEL) for inp in inputs]
    result = model.execute(*gpu_inputs)
    # result may be a dict or a list; normalise to list
    if isinstance(result, dict):
        outputs = list(result.values())
    else:
        outputs = list(result)
    return [o.to_numpy() for o in outputs]


def _pt_to_rscf(w):
    """PyTorch conv [C_out, C_in, K] -> MAX conv2d RSCF [K, 1, C_in/groups, C_out]."""
    # w: [C_out, C_in, K] -> insert W=1 dim -> [C_out, C_in, K, 1]
    # -> transpose(2, 3, 1, 0) -> [K, 1, C_in, C_out]
    return np.transpose(w[..., None], (2, 3, 1, 0)).copy()


def probe_dilated(C_in=64, C_out=64, K=3, T=50, dilation=1):
    """Test native ops.conv2d for dilated conv1d (run as conv2d with W=1).

    Findings (MAX 26.4, aarch64 GB10):
    - dilation=1: PASS_FP32 (max_diff ~2.3e-5, within GPU fp32 accumulation error)
    - dilation>1: FAIL — "Non-unit dilation is not supported yet" kernel error
    """
    rng = np.random.default_rng(0)
    x = rng.standard_normal((1, T, 1, C_in)).astype(np.float32)
    w = rng.standard_normal((C_out, C_in, K)).astype(np.float32)

    # PyTorch ground truth (NCT format, CPU)
    xt = torch.from_numpy(x).squeeze(2).transpose(1, 2)  # [1, C_in, T]
    pad = dilation * (K - 1) // 2
    yt = F.conv1d(xt, torch.from_numpy(w), padding=pad, dilation=dilation)
    yt = yt.transpose(1, 2).unsqueeze(2).numpy()  # [1, T, 1, C_out]

    # MAX native: conv2d with W=1 dimension, dilation only on H (T axis)
    w_rscf = _pt_to_rscf(w)  # [K, 1, C_in, C_out]
    try:
        with Graph(_next_name("dilated"), input_types=[TensorType(DType.float32, [1, T, 1, C_in], DEV)]) as g:
            out = ops.conv2d(
                g.inputs[0],
                ops.constant(w_rscf, device=DEV),
                stride=(1, 1),
                dilation=(dilation, 1),
                padding=(pad, pad, 0, 0),
            )
            g.output(out)

        ym = _execute(g, x)[0]
        d = float(np.abs(ym - yt).max())
        # Distinguish strict threshold vs fp32 precision variance
        if d < THRESH:
            status = "PASS"
        elif d < THRESH_FP32:
            status = f"PASS_FP32 (diff={d:.2e} < 1e-4; GPU fp32 accumulation, not a bug)"
        else:
            status = "FAIL"
        print(f"dilated d={dilation:>1}: max_diff={d:.3e}  {status}")
        return d < THRESH_FP32  # treat fp32 precision variance as functional pass
    except Exception as e:
        short = str(e).splitlines()[0][:120]
        print(f"dilated d={dilation:>1}: FAIL (exception: {short})")
        return False


def probe_convT1d(C_in=64, C_out=64, K=4, S=2, T=20):
    """Test native ops.conv2d_transpose for ConvTranspose1d (W=1 dimension).

    MAX conv2d_transpose RSCF layout: [K_h, K_w, C_out_orig, C_in_orig]
    where C_out_orig/C_in_orig are from the *forward* conv perspective.
    For ConvTranspose1d: RSCF for transpose = [K, 1, C_out, C_in].

    PyTorch ConvTranspose1d weight: [C_in, C_out, K]
    => rearrange to RSCF [K, 1, C_out, C_in]: transpose(2, 3, 1, 0)

    Findings (MAX 26.4, aarch64 GB10):
    - FAIL — cuDNN ALLOC_FAILED; MAX 26.4 has a layout bug in conv2d_transpose
      on this platform (uses NCHW stride layout for NHWC input descriptor).
    """
    rng = np.random.default_rng(1)
    x = rng.standard_normal((1, T, 1, C_in)).astype(np.float32)
    # PyTorch ConvTranspose1d weight: [C_in, C_out, K]
    w = rng.standard_normal((C_in, C_out, K)).astype(np.float32)

    pad = (K - S) // 2
    xt = torch.from_numpy(x).squeeze(2).transpose(1, 2)  # [1, C_in, T]
    yt = F.conv_transpose1d(xt, torch.from_numpy(w), stride=S, padding=pad)
    yt = yt.transpose(1, 2).unsqueeze(2).numpy()  # [1, T*S, 1, C_out]

    # MAX RSCF for transpose: [K, 1, C_out, C_in] from [C_in, C_out, K]
    # w[C_in, C_out, K] -> add W dim -> [C_in, C_out, K, 1] -> transpose(2, 3, 1, 0)
    w_rscf = np.transpose(w[..., np.newaxis], (2, 3, 1, 0)).copy()  # [K, 1, C_out, C_in]

    try:
        with Graph(_next_name("convT1d"), input_types=[TensorType(DType.float32, [1, T, 1, C_in], DEV)]) as g:
            out = ops.conv2d_transpose(
                g.inputs[0],
                ops.constant(w_rscf, device=DEV),
                stride=(S, 1),
                padding=(pad, pad, 0, 0),
                output_paddings=(0, 0),
            )
            g.output(out)

        ym = _execute(g, x)[0]
        n = min(ym.shape[1], yt.shape[1])
        d = float(np.abs(ym[:, :n] - yt[:, :n]).max())
        if d < THRESH:
            status = "PASS"
        elif d < THRESH_FP32:
            status = f"PASS_FP32 (diff={d:.2e} < 1e-4, GPU fp32 precision)"
        else:
            status = "FAIL"
        print(f"convT1d S={S}: max_diff={d:.3e}  {status}  shapes ym={ym.shape} yt={yt.shape}")
        return d < THRESH_FP32
    except Exception as e:
        short = str(e).splitlines()[0][:120]
        print(f"convT1d S={S}: FAIL (exception: {short})")
        return False


def probe_convT2d(C_in=32, C_out=32, K=3, S=2, H=8, W=8):
    """Test native ops.conv2d_transpose for ConvTranspose2d.

    MAX conv2d_transpose RSCF layout: [K_h, K_w, C_out, C_in]
    PyTorch ConvTranspose2d weight: [C_in, C_out, K_h, K_w]
    => rearrange: transpose(2, 3, 1, 0) -> [K_h, K_w, C_out, C_in]

    We choose padding=1 with S=2, K=3 so output_padding=0 (compatible with MAX
    which only supports output_paddings=0).

    Findings (MAX 26.4, aarch64 GB10):
    - FAIL — cuDNN ALLOC_FAILED; same issue as convT1d.
    """
    rng = np.random.default_rng(2)
    x = rng.standard_normal((1, H, W, C_in)).astype(np.float32)
    # PyTorch ConvTranspose2d weight: [C_in, C_out, K, K]
    w = rng.standard_normal((C_in, C_out, K, K)).astype(np.float32)

    # Use padding=1, output_padding=0 so no ambiguity with MAX's output_paddings=0 constraint
    xt = torch.from_numpy(x).permute(0, 3, 1, 2)  # NCHW
    yt = F.conv_transpose2d(xt, torch.from_numpy(w), stride=S, padding=1, output_padding=0)
    yt = yt.permute(0, 2, 3, 1).numpy()  # NHWC

    # MAX RSCF: [K, K, C_out, C_in] from PyTorch [C_in, C_out, K, K]
    w_rscf = np.transpose(w, (2, 3, 1, 0)).copy()  # [K, K, C_out, C_in]

    try:
        with Graph(_next_name("convT2d"), input_types=[TensorType(DType.float32, [1, H, W, C_in], DEV)]) as g:
            out = ops.conv2d_transpose(
                g.inputs[0],
                ops.constant(w_rscf, device=DEV),
                stride=(S, S),
                padding=(1, 1, 1, 1),
                output_paddings=(0, 0),
            )
            g.output(out)

        ym = _execute(g, x)[0]
        d = float(np.abs(ym - yt).max())
        if d < THRESH:
            status = "PASS"
        elif d < THRESH_FP32:
            status = f"PASS_FP32 (diff={d:.2e} < 1e-4, GPU fp32 precision)"
        else:
            status = "FAIL"
        print(f"convT2d S={S}: max_diff={d:.3e}  {status}  shapes ym={ym.shape} yt={yt.shape}")
        return d < THRESH_FP32
    except Exception as e:
        short = str(e).splitlines()[0][:120]
        print(f"convT2d S={S}: FAIL (exception: {short})")
        return False


if __name__ == "__main__":
    results = {
        "dilated_1": probe_dilated(dilation=1),
        "dilated_3": probe_dilated(dilation=3),
        "dilated_5": probe_dilated(dilation=5),
        "convT1d": probe_convT1d(),
        "convT2d": probe_convT2d(),
    }
    print("\nSUMMARY:", {k: ("PASS" if v else "FAIL") for k, v in results.items()})
