"""
Audio resampler using windowed-sinc (Lanczos) interpolation.

Supports any rational sample rate conversion.
Quality controlled by kernel_size (default 64).

Common conversions for voice conversion pipeline:
  48000 -> 16000  (studio audio to HuBERT input)
  44100 -> 16000  (CD audio to HuBERT input)
  16000 -> 48000  (upsample converted output to DAW rate)
"""

from math import sin
from math.constants import pi


fn _sinc(x: Float32) -> Float32:
    """Normalized sinc: sin(pi*x) / (pi*x), with sinc(0) = 1."""
    if x == 0.0:
        return 1.0
    var px = pi * x
    return sin(px) / px


fn _lanczos_kernel(x: Float32, a: Int) -> Float32:
    """Lanczos kernel of order a. Zero outside [-a, a]."""
    var abs_x = x if x >= 0.0 else -x
    if abs_x >= Float32(a):
        return 0.0
    return _sinc(x) * _sinc(x / Float32(a))


fn resample(
    samples: List[Float32],
    src_rate: Int,
    dst_rate: Int,
    kernel_size: Int = 64,
) raises -> List[Float32]:
    """
    Resample audio from src_rate to dst_rate using Lanczos interpolation.

    Args:
        samples: Input audio samples (mono, Float32).
        src_rate: Source sample rate in Hz.
        dst_rate: Target sample rate in Hz.
        kernel_size: Lanczos kernel order (higher = better quality, slower).
                     Default 64 is a good balance for voice conversion.

    Returns:
        Resampled audio at dst_rate.

    Raises:
        Error if src_rate or dst_rate <= 0.
    """
    if src_rate <= 0 or dst_rate <= 0:
        raise Error("Sample rates must be positive")

    var n_src = len(samples)

    if src_rate == dst_rate:
        # Return an independent copy
        var copy = List[Float32]()
        for i in range(n_src):
            copy.append(samples[i])
        return copy^

    var ratio = Float32(dst_rate) / Float32(src_rate)
    var n_dst = Int(Float32(n_src) * ratio)

    var result = List[Float32]()
    for _ in range(n_dst):
        result.append(0.0)

    # Lanczos order (half-width of kernel in source samples)
    var a = kernel_size // 2

    for i in range(n_dst):
        # Position in source space
        var src_pos = Float32(i) / ratio

        # Kernel window in source space
        var src_pos_int = Int(src_pos)
        var start = src_pos_int - a + 1
        var end = src_pos_int + a

        var val = Float32(0.0)
        var weight_sum = Float32(0.0)

        for j in range(start, end + 1):
            # Clamp to valid range
            var j_clamped = j
            if j_clamped < 0:
                j_clamped = 0
            if j_clamped >= n_src:
                j_clamped = n_src - 1

            var kernel_x = src_pos - Float32(j)
            var w = _lanczos_kernel(kernel_x, a)
            val += samples[j_clamped] * w
            weight_sum += w

        if weight_sum != 0.0:
            result[i] = val / weight_sum
        else:
            result[i] = 0.0

    return result^


fn resample_to_16k(samples: List[Float32], src_rate: Int) raises -> List[Float32]:
    """Convenience: resample any rate to 16kHz (HuBERT standard input)."""
    return resample(samples, src_rate, 16000)


fn resample_to_48k(samples: List[Float32], src_rate: Int) raises -> List[Float32]:
    """Convenience: resample any rate to 48kHz (studio standard output)."""
    return resample(samples, src_rate, 48000)
