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


# ---------------------------------------------------------------------------
# PROBE 2: express-as-plain correctness checks
# ---------------------------------------------------------------------------

def _dilate_kernel(w_np: np.ndarray, dilation: int) -> np.ndarray:
    """Insert (dilation-1) zeros between kernel taps.

    Input:  [C_out, C_in, K]
    Output: [C_out, C_in, K_eff] where K_eff = (K-1)*dilation + 1
    """
    if dilation == 1:
        return w_np
    C_out, C_in, K = w_np.shape
    K_eff = (K - 1) * dilation + 1
    w_dilated = np.zeros((C_out, C_in, K_eff), dtype=w_np.dtype)
    w_dilated[:, :, ::dilation] = w_np
    return w_dilated


def probe_dilated_expanded(C_in=64, C_out=64, K=3, T=50, dilation=3):
    """Probe 2-A: dilated conv via kernel expansion (express-as-plain).

    A dilation-d, kernel-K conv equals a plain conv with kernel expanded to
    K_eff = (K-1)*d + 1 by inserting (d-1) zeros between taps.

    Strategy:
      - Expand kernel: w_eff [C_out, C_in, K_eff]
      - Convert to MAX RSCF [K_eff, 1, C_in, C_out]
      - Call ops.conv2d with dilation=(1,1) and symmetric padding (K_eff-1)//2

    Compare to: PyTorch F.conv1d with dilation=d and padding=d*(K-1)//2.
    Both should give identical same-length output.
    """
    rng = np.random.default_rng(10 + dilation)
    x = rng.standard_normal((1, T, 1, C_in)).astype(np.float32)
    # PyTorch Conv1d weight: [C_out, C_in, K]
    w = rng.standard_normal((C_out, C_in, K)).astype(np.float32)

    # PyTorch ground truth: F.conv1d with dilation
    xt = torch.from_numpy(x).squeeze(2).transpose(1, 2)  # [1, C_in, T]
    pad_pt = dilation * (K - 1) // 2
    yt = F.conv1d(xt, torch.from_numpy(w), padding=pad_pt, dilation=dilation)
    yt = yt.transpose(1, 2).unsqueeze(2).numpy()  # [1, T, 1, C_out]

    # Expand kernel: insert (d-1) zeros between taps
    w_eff = _dilate_kernel(w, dilation)  # [C_out, C_in, K_eff]
    K_eff = w_eff.shape[2]

    # Convert to MAX RSCF [K_eff, 1, C_in, C_out]: w[C_out, C_in, K_eff] -> transpose(2, 3, 1, 0)
    # Insert W=1 dim first: [C_out, C_in, K_eff, 1], then transpose(2, 3, 1, 0)
    w_rscf = np.transpose(w_eff[..., None], (2, 3, 1, 0)).copy()  # [K_eff, 1, C_in, C_out]

    pad = (K_eff - 1) // 2

    try:
        with Graph(_next_name("dilated_exp"), input_types=[TensorType(DType.float32, [1, T, 1, C_in], DEV)]) as g:
            out = ops.conv2d(
                g.inputs[0],
                ops.constant(w_rscf, device=DEV),
                stride=(1, 1),
                dilation=(1, 1),  # plain conv2d — dilation baked into expanded kernel
                padding=(pad, pad, 0, 0),
            )
            g.output(out)

        ym = _execute(g, x)[0]
        d = float(np.abs(ym - yt).max())
        if d < THRESH:
            status = "PASS"
        elif d < THRESH_FP32:
            status = f"PASS_FP32 (diff={d:.2e} < 1e-4; GPU fp32 accumulation)"
        else:
            status = f"FAIL (diff={d:.3e})"
        print(f"dilated_expanded d={dilation}: max_diff={d:.3e}  {status}  K_eff={K_eff}")
        return d < THRESH_FP32
    except Exception as e:
        short = str(e).splitlines()[0][:120]
        print(f"dilated_expanded d={dilation}: FAIL (exception: {short})")
        return False


def probe_convT1d_native(C_in=64, C_out=64, K=4, S=2, T=20):
    """Probe 2-B: ConvTranspose1d via zero-interleave + native plain ops.conv2d.

    The existing conv_transpose_1d() in _hifigan_graph.py already does:
      1. Zero-interleave (insert S-1 zeros between time steps)
      2. im2col + matmul with flipped kernel

    This probe replaces step 2 with a native ops.conv2d call (plain, d=1).

    Key details:
      - Zero-interleave the input: [1, T, 1, C_in] -> [1, T*S, 1, C_in]
      - Flip the kernel: w_pt[:, :, ::-1] (ConvTranspose = cross-correlation with flipped kernel)
        Note: PyTorch ConvTranspose1d weight [C_in, C_out, K] — flip along K axis
      - Convert flipped kernel to MAX conv2d RSCF [K, 1, C_in/groups, C_out]:
        w_flipped[C_in, C_out, K] -> transpose(2, 3, 1, 0) = [K, 1, C_out, C_in]
        But MAX conv2d RSCF is [K, 1, C_in, C_out], so we need to flip the in/out:
        For transpose conv expressed as plain conv, we do:
          - Input has C_in channels, output has C_out channels
          - PyTorch ConvTranspose1d w[C_in, C_out, K]: each output channel is a sum over input channels
          - Equivalent plain conv has filter: for each output channel c_out, kernel k:
            f[k, 1, c_in, c_out] = w_flipped[c_in, c_out, k]
          - So RSCF = [K, 1, C_in, C_out] from [C_in, C_out, K]: transpose(2, 3, 0, 1)
        Wait — re-derive carefully:
          w_pt[C_in, C_out, K]
          w_flipped[C_in, C_out, K] after [:, :, ::-1]
          MAX conv2d RSCF [K, 1, C_in/groups, C_out]:
            we want w_rscf[k, 0, c_in, c_out] = w_flipped[c_in, c_out, k]
            so: w_flipped[C_in, C_out, K] -> insert W dim -> [C_in, C_out, K, 1]
                -> transpose(2, 3, 0, 1) -> [K, 1, C_in, C_out]  ✓
      - Asymmetric padding (same as conv_transpose_1d in _hifigan_graph.py lines 77-78):
          pad_left = (K + S - 2) // 2
          pad_right = (K - S) // 2

    Compare to: PyTorch F.conv_transpose1d(x, w, stride=S, padding=(K-S)//2).
    """
    rng = np.random.default_rng(20)
    x_np = rng.standard_normal((1, T, 1, C_in)).astype(np.float32)
    # PyTorch ConvTranspose1d weight: [C_in, C_out, K]
    w_pt = rng.standard_normal((C_in, C_out, K)).astype(np.float32)

    pad = (K - S) // 2
    xt = torch.from_numpy(x_np).squeeze(2).transpose(1, 2)  # [1, C_in, T]
    yt = F.conv_transpose1d(xt, torch.from_numpy(w_pt), stride=S, padding=pad)
    yt = yt.transpose(1, 2).unsqueeze(2).numpy()  # [1, T*S, 1, C_out]

    T_out = T * S

    # Asymmetric padding matching _hifigan_graph.py lines 77-78
    pad_left = (K + S - 2) // 2
    pad_right = (K - S) // 2

    # Flipped kernel for cross-correlation: w_pt[:, :, ::-1]
    w_flipped = w_pt[:, :, ::-1].copy()  # [C_in, C_out, K]
    # Convert to MAX conv2d RSCF [K, 1, C_in, C_out]
    w_rscf = np.transpose(w_flipped[..., None], (2, 3, 0, 1)).copy()  # [K, 1, C_in, C_out]

    try:
        with Graph(_next_name("convT1d_native"),
                   input_types=[TensorType(DType.float32, [1, T, 1, C_in], DEV)]) as g:
            xi = g.inputs[0]

            # Step 1: zero-interleave along T
            # [1, T, 1, C_in] -> squeeze batch -> [T, 1, C_in]
            x_sq = ops.squeeze(xi, 0)          # [T, 1, C_in]
            x_ins = ops.unsqueeze(x_sq, 1)     # [T, 1, 1, C_in]
            x_padded = ops.pad(x_ins, [0, 0, 0, 0, 0, S - 1, 0, 0])  # [T, S, 1, C_in]
            x_merged = ops.reshape(x_padded, [T_out, 1, C_in])        # [T*S, 1, C_in]
            x_zi = ops.unsqueeze(x_merged, 0)  # [1, T*S, 1, C_in] — NHWC with W=1

            # Step 2: native plain conv2d with flipped kernel and asymmetric padding
            out = ops.conv2d(
                x_zi,
                ops.constant(w_rscf, device=DEV),
                stride=(1, 1),
                dilation=(1, 1),
                padding=(pad_left, pad_right, 0, 0),
            )
            g.output(out)

        ym = _execute(g, x_np)[0]
        n = min(ym.shape[1], yt.shape[1])
        d = float(np.abs(ym[:, :n] - yt[:, :n]).max())
        if d < THRESH:
            status = "PASS"
        elif d < THRESH_FP32:
            status = f"PASS_FP32 (diff={d:.2e} < 1e-4; GPU fp32 accumulation)"
        else:
            status = f"FAIL (diff={d:.3e})"
        print(f"convT1d_native S={S}: max_diff={d:.3e}  {status}  shapes ym={ym.shape} yt={yt.shape}")
        return d < THRESH_FP32
    except Exception as e:
        short = str(e).splitlines()[0][:120]
        print(f"convT1d_native S={S}: FAIL (exception: {short})")
        return False


def probe_convT2d_native(C_in=32, C_out=32, K=3, S=2, H=8, W_=8):
    """Probe 2-C: ConvTranspose2d via zero-interleave + native plain ops.conv2d.

    Strategy for 2D case (stride=2, K=3, padding=1, output_padding=0):
      - Zero-interleave both spatial dims: insert S-1 zeros between each row/col
      - Flip kernel along both spatial axes (ConvTranspose = cross-correlation with flipped kernel)
      - Pad with asymmetric padding: same formula as 1D case, applied to both H and W dims
      - Call plain ops.conv2d(dilation=(1,1))

    Asymmetric padding for 2D (each spatial dim independently):
      pad_before_h = pad_before_w = (K + S - 2) // 2
      pad_after_h  = pad_after_w  = (K - S) // 2

    Compare to: PyTorch F.conv_transpose2d(x, w, stride=2, padding=1, output_padding=0).
    Note: padding=1 in PyTorch ConvTranspose2d removes 1 row/col from each side,
    equivalent to choosing asymmetric pads:
      pad_before = (K-1) - padding = K-1-1 = K-2 = 1  (for K=3)
      pad_after  = K-1-padding - output_padding = 1 - 0 = 1
    For S=2, K=3: pad_before = (K+S-2)//2 = (3+2-2)//2 = 3//2 = 1, pad_after = (K-S)//2 = 1//2 = 0.

    Wait — need to reconcile with PyTorch. For conv_transpose2d with stride=S, padding=P, output_padding=OP:
      H_out = (H_in - 1) * S - 2*P + K + OP
    With S=2, K=3, P=1, OP=0: H_out = (H_in-1)*2 - 2 + 3 = 2*H_in - 1
    With S=2, H_in=8: H_out = 15

    Zero-interleave gives H_zi = (H_in-1)*S + 1 = 15 (for H_in=8, S=2).
    Then apply padding and convolve with K=3.
    For "same" output after zero-interleave: no padding needed when K=3 (the zero-interleave itself
    already centers things). Let's derive:
      H_out_target = 2*H_in - 1 = 15
      H_zi = 15
      H_out_after_conv = H_zi - K + 1 + pad_left + pad_right
      We want H_out = 15: 15 - 3 + 1 + pad_left + pad_right = 15 => pad_left + pad_right = 2
      Use pad_left = 1, pad_right = 1 (symmetric for K=3, S=2).
    """
    rng = np.random.default_rng(30)
    x_np = rng.standard_normal((1, H, W_, C_in)).astype(np.float32)
    # PyTorch ConvTranspose2d weight: [C_in, C_out, K, K]
    w_pt = rng.standard_normal((C_in, C_out, K, K)).astype(np.float32)

    # PyTorch ground truth: padding=1, output_padding=0 -> output (2*H-1, 2*W-1)
    xt = torch.from_numpy(x_np).permute(0, 3, 1, 2)  # NCHW
    yt = F.conv_transpose2d(xt, torch.from_numpy(w_pt), stride=S, padding=1, output_padding=0)
    yt = yt.permute(0, 2, 3, 1).numpy()  # NHWC

    H_out = yt.shape[1]
    W_out = yt.shape[2]

    # Zero-interleave both dims: H_zi = (H-1)*S + 1, W_zi = (W-1)*S + 1
    H_zi = (H - 1) * S + 1
    W_zi = (W_ - 1) * S + 1

    # Pad needed: H_out - (H_zi - K + 1) = pad_total; split symmetrically
    pad_total_h = H_out - H_zi + K - 1
    pad_total_w = W_out - W_zi + K - 1
    pad_before_h = pad_total_h // 2
    pad_after_h  = pad_total_h - pad_before_h
    pad_before_w = pad_total_w // 2
    pad_after_w  = pad_total_w - pad_before_w

    # Flipped kernel: flip along both spatial axes (last two dims in PyTorch layout)
    # w_pt[C_in, C_out, K_h, K_w] -> flip K_h and K_w -> w_flipped
    w_flipped = w_pt[:, :, ::-1, ::-1].copy()  # [C_in, C_out, K, K]
    # MAX conv2d RSCF [K, K, C_in, C_out]: w_flipped[C_in, C_out, K, K] -> transpose(2, 3, 0, 1)
    w_rscf = np.transpose(w_flipped, (2, 3, 0, 1)).copy()  # [K, K, C_in, C_out]

    try:
        with Graph(_next_name("convT2d_native"),
                   input_types=[TensorType(DType.float32, [1, H, W_, C_in], DEV)]) as g:
            xi = g.inputs[0]

            # Step 1: zero-interleave H axis (all 4D ops, no 5D tensors).
            # [B, H, W, C] -> flatten W*C into last dim -> [B, H, W*C]
            # -> insert H-axis: pad last dim to make room, reshape -> [B, H*S, W*C]
            # -> trim -> [B, H_zi, W*C] -> reshape back -> [B, H_zi, W, C]
            #
            # Trick: reshape to [B, H, W*C, 1], pad along dim-3 to [B, H, W*C, S],
            # reshape to [B, H*S, W*C//... wait, that doesn't work.
            #
            # Cleaner: use the same trick as conv_transpose_1d for each spatial dim,
            # treating the "other" spatial dim as part of channels.
            #
            # H-axis interleave:
            # Reshape [1, H, W, C] -> [1, H, W*C]: treat W*C as a unit.
            # Now in [B, H, WC] form, insert S-1 zeros in H:
            #   unsqueeze(2) -> [B, H, 1, WC]
            #   pad([0,0, 0,S-1, 0,0, 0,0]) -> [B, H, S, WC]
            #   reshape -> [B, H*S, WC]
            # slice [0:H_zi] -> [B, H_zi, WC]
            # reshape -> [B, H_zi, W, C]
            WC = W_ * C_in
            # H-axis zero-interleave (all ops strictly 4D; ops.transpose swaps only 2 axes).
            # ops.pad format: [pad_before_d0, pad_after_d0, pad_before_d1, pad_after_d1, ...]
            # Strategy: flatten W*C, unsqueeze an extra dim next to H, pad that dim, reshape.
            # [1, H, WC] -> unsqueeze dim2 -> [1, H, 1, WC]
            # pad dim2 [0,0, 0,0, 0,S-1, 0,0] -> [1, H, S, WC]
            # reshape [1, H*S, WC] -> slice [1, H_zi, WC] -> reshape [1, H_zi, W, C]
            x_flat_h = ops.reshape(xi, [1, H, WC])                          # [1, H, WC]
            x_flat_h = ops.unsqueeze(x_flat_h, 2)                           # [1, H, 1, WC]
            x_flat_h = ops.pad(x_flat_h, [0, 0, 0, 0, 0, S - 1, 0, 0])    # [1, H, S, WC]
            x_flat_h = ops.reshape(x_flat_h, [1, H * S, WC])                # [1, H*S, WC]
            x_flat_h = ops.slice_tensor(x_flat_h, [slice(None), slice(0, H_zi), slice(None)])
            x_h_zi = ops.reshape(x_flat_h, [1, H_zi, W_, C_in])            # [1, H_zi, W, C]

            # W-axis interleave of x_h_zi [1, H_zi, W, C]:
            # Treat H_zi as a batch axis: reshape [1, H_zi, W, C] -> [H_zi, W, 1, C]
            # pad dim2: [0,0, 0,0, 0,S-1, 0,0] -> [H_zi, W, S, C]
            # reshape [H_zi, W*S, C] -> slice [H_zi, W_zi, C] -> reshape [1, H_zi, W_zi, C]
            x_rsh_w = ops.reshape(x_h_zi, [H_zi, W_, 1, C_in])             # [H_zi, W, 1, C]
            x_rsh_w = ops.pad(x_rsh_w, [0, 0, 0, 0, 0, S - 1, 0, 0])      # [H_zi, W, S, C]
            x_rsh_w = ops.reshape(x_rsh_w, [H_zi, W_ * S, C_in])           # [H_zi, W*S, C]
            x_rsh_w = ops.slice_tensor(x_rsh_w, [slice(None), slice(0, W_zi), slice(None)])
            x_zi = ops.reshape(x_rsh_w, [1, H_zi, W_zi, C_in])             # [1, H_zi, W_zi, C]

            # Step 2: native plain conv2d with flipped kernel and computed padding
            out = ops.conv2d(
                x_zi,
                ops.constant(w_rscf, device=DEV),
                stride=(1, 1),
                dilation=(1, 1),
                padding=(pad_before_h, pad_after_h, pad_before_w, pad_after_w),
            )
            g.output(out)

        ym = _execute(g, x_np)[0]
        d = float(np.abs(ym - yt).max())
        if d < THRESH:
            status = "PASS"
        elif d < THRESH_FP32:
            status = f"PASS_FP32 (diff={d:.2e} < 1e-4; GPU fp32 accumulation)"
        else:
            status = f"FAIL (diff={d:.3e})"
        print(f"convT2d_native S={S} K={K}: max_diff={d:.3e}  {status}  shapes ym={ym.shape} yt={yt.shape}  pads=({pad_before_h},{pad_after_h},{pad_before_w},{pad_after_w})")
        return d < THRESH_FP32
    except Exception as e:
        short = str(e).splitlines()[0][:120]
        print(f"convT2d_native S={S} K={K}: FAIL (exception: {short})")
        return False


def probe_timing_dilated_expanded(C=512, K=3, dilation=5, T=400, n_warmup=2, n_timed=5):
    """Probe 2-D: wall-clock timing — native expanded-kernel vs im2col for HiFiGAN-sized dilated conv.

    Representative case: C_in=C_out=512, K=3, d=5, T=400 (HiFiGAN resblock conv).

    Two paths:
      A. Native: expand kernel to K_eff=11, run ops.conv2d(dilation=(1,1)) on GPU
      B. im2col: replicate _hifigan_graph.py conv1d() logic (dilate kernel, im2col matmul)

    Each path is compiled once, then timed over n_timed executions.
    Wall-clock includes model.execute() but excludes graph compile and buffer transfer.
    """
    import time

    rng = np.random.default_rng(99)
    x_np = rng.standard_normal((1, T, 1, C)).astype(np.float32)
    w_np = rng.standard_normal((C, C, K)).astype(np.float32)  # [C_out, C_in, K]

    K_eff = (K - 1) * dilation + 1
    pad = (K_eff - 1) // 2

    print(f"\n[timing] C={C}, K={K}, d={dilation}, T={T}, K_eff={K_eff}")

    # ---- Path A: Native expanded-kernel conv2d ----
    w_eff = _dilate_kernel(w_np, dilation)  # [C_out, C_in, K_eff]
    # RSCF: [K_eff, 1, C_in, C_out]
    w_rscf_a = np.transpose(w_eff[..., None], (2, 3, 1, 0)).copy()

    with Graph(_next_name("timing_native"),
               input_types=[TensorType(DType.float32, [1, T, 1, C], DEV)]) as g_a:
        out = ops.conv2d(
            g_a.inputs[0],
            ops.constant(w_rscf_a, device=DEV),
            stride=(1, 1),
            dilation=(1, 1),
            padding=(pad, pad, 0, 0),
        )
        g_a.output(out)

    model_a = SESSION.load(g_a)
    x_buf = Buffer.from_numpy(x_np).to(ACCEL)

    # Warmup
    for _ in range(n_warmup):
        model_a.execute(x_buf)

    t0 = time.perf_counter()
    for _ in range(n_timed):
        model_a.execute(x_buf)
    t_native = (time.perf_counter() - t0) / n_timed * 1000  # ms per call

    # ---- Path B: im2col matmul (replicates conv1d in _hifigan_graph.py) ----
    w_eff_b = _dilate_kernel(w_np, dilation)  # [C_out, C_in, K_eff]
    # Weight for matmul: [K_eff, C_in, C_out] -> reshape [K_eff*C_in, C_out]
    w_mat_b = np.transpose(w_eff_b, (2, 1, 0)).reshape(K_eff * C, C)
    pad_b = (K_eff - 1) // 2

    with Graph(_next_name("timing_im2col"),
               input_types=[TensorType(DType.float32, [1, T, 1, C], DEV)]) as g_b:
        xi = g_b.inputs[0]
        # Pad [1, T, 1, C] -> [1, T+2*pad, 1, C]
        x_padded = ops.pad(xi, [0, 0, pad_b, pad_b, 0, 0, 0, 0])
        # im2col: K_eff shifted slices, each [1, T, 1, C]
        slices = []
        for k in range(K_eff):
            s = ops.slice_tensor(
                x_padded,
                [slice(None), slice(k, k + T), slice(None), slice(None)],
            )
            slices.append(s)
        x_cols = ops.concat(slices, axis=3)  # [1, T, 1, K_eff*C]
        x_cols_sq = ops.squeeze(x_cols, 2)   # [1, T, K_eff*C]
        w_const_b = ops.constant(w_mat_b.astype(np.float32), device=DEV)
        out_b = ops.matmul(x_cols_sq, w_const_b)  # [1, T, C_out]
        out_b = ops.unsqueeze(out_b, 2)            # [1, T, 1, C_out]
        g_b.output(out_b)

    model_b = SESSION.load(g_b)

    # Warmup
    for _ in range(n_warmup):
        model_b.execute(x_buf)

    t0 = time.perf_counter()
    for _ in range(n_timed):
        model_b.execute(x_buf)
    t_im2col = (time.perf_counter() - t0) / n_timed * 1000  # ms per call

    ratio = t_im2col / t_native if t_native > 0 else float("inf")
    print(f"[timing] native (expanded kernel):  {t_native:.3f} ms/call")
    print(f"[timing] im2col (matmul):           {t_im2col:.3f} ms/call")
    print(f"[timing] speedup native vs im2col:  {ratio:.2f}x")

    return t_native, t_im2col


if __name__ == "__main__":
    # --- Probe 1 (original): direct native ops, expected to fail for d>1 and transpose ---
    print("=" * 60)
    print("PROBE 1: Direct native ops (expected failures for d>1, transpose)")
    print("=" * 60)
    results = {
        "dilated_1": probe_dilated(dilation=1),
        "dilated_3": probe_dilated(dilation=3),
        "dilated_5": probe_dilated(dilation=5),
        "convT1d": probe_convT1d(),
        "convT2d": probe_convT2d(),
    }
    print("\nSUMMARY P1:", {k: ("PASS" if v else "FAIL") for k, v in results.items()})

    # --- Probe 2: express-as-plain correctness ---
    print("\n" + "=" * 60)
    print("PROBE 2: Express-as-plain correctness (all should PASS)")
    print("=" * 60)
    results2 = {
        "dilated_expanded_d3": probe_dilated_expanded(dilation=3),
        "dilated_expanded_d5": probe_dilated_expanded(dilation=5),
        "convT1d_native":      probe_convT1d_native(),
        "convT2d_native":      probe_convT2d_native(),
    }
    print("\nSUMMARY P2:", {k: ("PASS" if v else "FAIL") for k, v in results2.items()})

    # --- Probe 2 timing ---
    print("\n" + "=" * 60)
    print("PROBE 2 TIMING: native expanded-kernel vs im2col (C=512, K=3, d=5, T=400)")
    print("=" * 60)
    probe_timing_dilated_expanded()
