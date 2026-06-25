"""Native MAX conv primitives via express-as-plain.

Native ops.conv2d on aarch64/26.4 supports ONLY plain dilation=1 (native
dilated conv and ops.conv2d_transpose are broken — see
.superpowers/sdd/task-1b-report.md). So dilation is done by kernel expansion
and transpose by zero-interleave, both reduced to native plain conv2d.
"""
import numpy as np
from max.graph import ops
from ._feature_extractor import _pt_weight_to_max  # [C_out,C_in,K] -> RSCF [K,1,C_in,C_out]


def _bias(b_np, device_ref):
    return ops.constant(np.asarray(b_np, dtype=np.float32), device=device_ref) if b_np is not None else None


def _dilate_kernel(w, d):
    """Insert (d-1) zeros between taps: [C_out,C_in,K] -> [C_out,C_in,(K-1)*d+1]."""
    if d == 1:
        return np.asarray(w, dtype=np.float32)
    C_out, C_in, K = w.shape
    w_eff = np.zeros((C_out, C_in, (K - 1) * d + 1), dtype=np.float32)
    w_eff[:, :, ::d] = w
    return w_eff


def conv1d(x, w_pt, b_np, dilation=1, groups=1, device_ref=None):
    """Conv1d as native plain conv2d; dilation handled by kernel expansion.
    x: NHWC [B,T,1,C_in]. w_pt: PyTorch [C_out,C_in,K]."""
    w_eff = _dilate_kernel(np.asarray(w_pt, dtype=np.float32), dilation)
    K_eff = w_eff.shape[2]
    w_max = _pt_weight_to_max(w_eff)  # [K_eff,1,C_in,C_out]
    pad = (K_eff - 1) // 2
    return ops.conv2d(
        x, ops.constant(w_max, device=device_ref),
        stride=(1, 1), dilation=(1, 1),
        padding=(pad, pad, 0, 0), groups=groups, bias=_bias(b_np, device_ref),
    )


def conv2d(x, w_max, b_np, stride=(1, 1), padding=(0, 0, 0, 0), groups=1, device_ref=None):
    """Direct native conv2d. w_max already MAX RSCF [kH,kW,C_in,C_out].

    For 1x1 kernels, uses matmul instead of ops.conv2d: ops.conv2d raises
    'no kernel registered for layout_transform_RSCF_to_KNkni' for kH=kW=1
    on CPU (aarch64 and x64). All other kernel sizes use native ops.conv2d.
    """
    w_max = np.asarray(w_max, dtype=np.float32)
    kH, kW, C_in, C_out = w_max.shape
    if kH == 1 and kW == 1:
        # 1x1 via matmul: squeeze batch, [H,W,C_in] @ [C_in,C_out] -> [H,W,C_out]
        w_2d = ops.constant(w_max.reshape(C_in, C_out), device=device_ref)
        x_sq = ops.squeeze(x, 0)
        out = ops.matmul(x_sq, w_2d)
        out = ops.unsqueeze(out, 0)
        if b_np is not None:
            out = ops.add(out, ops.constant(
                np.asarray(b_np, dtype=np.float32).reshape(1, 1, 1, -1), device=device_ref))
        return out
    return ops.conv2d(
        x, ops.constant(w_max, device=device_ref),
        stride=stride, dilation=(1, 1), padding=padding, groups=groups,
        bias=_bias(b_np, device_ref),
    )


def conv_transpose1d(x, w_pt, b_np, stride, device_ref=None):
    """ConvTranspose1d via zero-interleave + native plain conv2d.
    w_pt: PyTorch [C_in,C_out,K]. x: NHWC [B,T,1,C_in]. B must be 1."""
    w_pt = np.asarray(w_pt, dtype=np.float32)
    C_in, C_out, K = w_pt.shape
    S = stride
    w_flipped = w_pt[:, :, ::-1].copy()                          # convolution = flipped corr
    w_max = np.transpose(w_flipped[..., None], (2, 3, 0, 1)).copy()  # [K,1,C_in,C_out]
    T = x.shape[1]
    x_sq = ops.squeeze(x, 0)                                     # [T,1,C_in]
    x_ins = ops.unsqueeze(x_sq, 1)                               # [T,1,1,C_in]
    x_pad = ops.pad(x_ins, [0, 0, 0, S - 1, 0, 0, 0, 0])        # [T,S,1,C_in]
    x_merged = ops.reshape(x_pad, [T * S, 1, C_in])             # [T*S,1,C_in]
    x_zi = ops.unsqueeze(x_merged, 0)                           # [1,T*S,1,C_in]
    pad_left = (K + S - 2) // 2
    pad_right = (K - S) // 2
    return ops.conv2d(
        x_zi, ops.constant(w_max, device=device_ref),
        stride=(1, 1), dilation=(1, 1),
        padding=(pad_left, pad_right, 0, 0), bias=_bias(b_np, device_ref),
    )


def conv_transpose2d(x, w_pt, b_np, stride=2, device_ref=None):
    """ConvTranspose2d (S=2,K=3,P=1,output_pad=1) via 2D zero-interleave +
    native plain conv2d. w_pt: PyTorch [C_in,C_out,3,3]. H,W must be STATIC
    (RMVPE decoder dims are fixed). ops.transpose is 2-axis only — the
    reshape/pad/reshape chain below is the validated 4D interleave.

    Uses output_padding=1 convention (H -> 2H, W -> 2W): pad a zero after every
    input element including the last, so H_zi=2H instead of (H-1)*S+1.
    Matches PyTorch F.conv_transpose2d(stride=2, padding=1, output_padding=1).
    """
    w_pt = np.asarray(w_pt, dtype=np.float32)
    C_in, C_out, Kh, Kw = w_pt.shape
    S = stride
    w_flipped = w_pt[:, :, ::-1, ::-1].copy()
    w_max = np.transpose(w_flipped, (2, 3, 0, 1)).copy()        # [Kh,Kw,C_in,C_out]
    H = int(x.shape[1]); W = int(x.shape[2]); C = C_in
    # output_padding=1: H_zi = 2H (pad zero after every element including last)
    H_zi = H * S
    W_zi = W * S
    # H-axis interleave of [1,H,W,C] -> [1,2H,W,C]
    # squeeze batch, unsqueeze slot next to H, pad that slot, reshape to merge
    x_sq = ops.squeeze(x, 0)                                    # [H,W,C]
    x_ins = ops.unsqueeze(x_sq, 1)                              # [H,1,W,C]
    x_padH = ops.pad(x_ins, [0, 0, 0, 1, 0, 0, 0, 0])         # [H,2,W,C]
    x_r2 = ops.reshape(x_padH, [H * S, W, C])                  # [2H,W,C]
    x_zi_H = ops.unsqueeze(x_r2, 0)                             # [1,2H,W,C]
    # W-axis interleave: use transpose trick (W is static so W*2 provable)
    x_sq2 = ops.squeeze(x_zi_H, 0)                              # [2H,W,C]
    x_t = ops.transpose(x_sq2, 0, 1)                            # [W,2H,C]
    x_ins2 = ops.unsqueeze(x_t, 1)                              # [W,1,2H,C]
    x_padW = ops.pad(x_ins2, [0, 0, 0, 1, 0, 0, 0, 0])        # [W,2,2H,C]
    x_r4 = ops.reshape(x_padW, [W * S, H_zi, C])               # [2W,2H,C]
    x_t2 = ops.transpose(x_r4, 0, 1)                            # [2H,2W,C]
    x_zi = ops.unsqueeze(x_t2, 0)                               # [1,2H,2W,C]
    # Symmetric padding: for K=3,S=2,P=1,OP=1: pad=1 each side
    pad = Kh - 1 - 1  # K-1-P; for K=3,P=1 -> 1
    return ops.conv2d(
        x_zi, ops.constant(w_max, device=device_ref),
        stride=(1, 1), dilation=(1, 1),
        padding=(pad, pad, pad, pad), bias=_bias(b_np, device_ref),
    )
