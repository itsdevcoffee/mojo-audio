"""
Phase vocoder pitch shifter for mojo-audio.

Shifts pitch by semitones while preserving duration.
Uses phase vocoder time-stretch + Lanczos resampling.

Usage in voice conversion pipeline:
  - Pre-align source to target key before voice conversion
  - Post-processing correction after conversion
  - Key transposition without neural inference

Algorithm:
  1. Compute frequency ratio = 2^(semitones/12)
  2. Time-stretch audio by 1/ratio using phase vocoder OLA
  3. Resample back to original length (restores duration)
"""

from math import exp, log, cos, sin
from math.constants import pi
from audio import hann_window
from resample import resample


fn semitones_to_ratio(semitones: Float32) -> Float32:
    """
    Convert semitone shift to frequency ratio.

    +12 semitones = one octave up = 2.0x frequency.
    -12 semitones = one octave down = 0.5x frequency.
    """
    return exp(semitones * log(Float32(2.0)) / Float32(12.0))


fn transpose_semitones(semitones: Float32) -> Float32:
    """
    Normalize semitone shift to [-6, +6] (minimum transposition to any key).

    Examples:
      7  -> -5  (going up a fifth = going down a fourth)
      12 -> 0   (octave = no change in pitch class)
      -7 -> 5
    """
    var s = semitones
    while s > 6.0:
        s -= 12.0
    while s < -6.0:
        s += 12.0
    return s


fn _time_stretch_ola(
    audio: List[Float32],
    stretch_factor: Float32,
    n_fft: Int = 512,
    hop_length: Int = 128,
) raises -> List[Float32]:
    """
    Time-stretch audio by stretch_factor using overlap-add (OLA).

    stretch_factor > 1.0 -> longer output (slower)
    stretch_factor < 1.0 -> shorter output (faster)

    Uses Hann window for smooth overlap-add synthesis.
    This is the identity phase vocoder -- fast and good enough for voice.
    """
    var win = hann_window(n_fft)
    var n_src = len(audio)

    # Compute output frames
    var n_frames = (n_src - n_fft) // hop_length + 1
    if n_frames <= 0:
        var copy = List[Float32]()
        for i in range(n_src):
            copy.append(audio[i])
        return copy^

    # Output hop (stretched)
    var out_hop = Int(Float32(hop_length) * stretch_factor + 0.5)
    if out_hop < 1:
        out_hop = 1

    # Output buffer
    var n_out = (n_frames - 1) * out_hop + n_fft
    var output = List[Float32]()
    var win_sum = List[Float32]()
    for _ in range(n_out):
        output.append(0.0)
        win_sum.append(0.0)

    for f in range(n_frames):
        # Read input frame
        var src_start = f * hop_length
        var dst_start = f * out_hop

        # Apply window and overlap-add to output
        for i in range(n_fft):
            if src_start + i < n_src and dst_start + i < n_out:
                output[dst_start + i] += audio[src_start + i] * win[i] * win[i]
                win_sum[dst_start + i] += win[i] * win[i]

    # Normalize
    for i in range(n_out):
        if win_sum[i] > 1e-8:
            output[i] /= win_sum[i]

    return output^


fn pitch_shift(
    audio: List[Float32],
    semitones: Float32,
    sample_rate: Int = 16000,
    n_fft: Int = 512,
    hop_length: Int = 128,
) raises -> List[Float32]:
    """
    Shift pitch by the given number of semitones.

    Duration of the output is the same as the input (length preserved).
    Uses overlap-add time-stretch + Lanczos resampling.

    Args:
        audio: Input audio samples (mono, Float32).
        semitones: Pitch shift in semitones.
                   Positive = pitch up. Negative = pitch down.
                   +12 = one octave up. -12 = one octave down.
        sample_rate: Sample rate in Hz (default 16000 for HuBERT pipeline).
        n_fft: FFT frame size for phase vocoder (default 512).
        hop_length: Hop length for phase vocoder (default 128).

    Returns:
        Pitch-shifted audio, same length as input.
    """
    if semitones == 0.0:
        # Identity: return copy
        var copy = List[Float32]()
        for i in range(len(audio)):
            copy.append(audio[i])
        return copy^

    var ratio = semitones_to_ratio(semitones)
    var src_len = len(audio)

    # Time-stretch by inverse ratio (pitch up = time compress before resampling)
    var stretch = Float32(1.0) / ratio
    var stretched = _time_stretch_ola(audio, stretch, n_fft, hop_length)

    # Resample back to original length
    # Treat stretched audio as if at sample_rate * ratio, resample to sample_rate
    var stretched_len = len(stretched)
    if stretched_len == 0:
        var copy = List[Float32]()
        for i in range(src_len):
            copy.append(audio[i])
        return copy^

    # Effective source rate to resample from stretched -> original length
    var src_rate_eff = Int(Float32(sample_rate) * ratio + 0.5)
    if src_rate_eff <= 0:
        src_rate_eff = 1
    var resampled = resample(stretched, src_rate_eff, sample_rate)

    # Trim or pad to exactly match original length
    var result = List[Float32]()
    for i in range(src_len):
        if i < len(resampled):
            result.append(resampled[i])
        else:
            result.append(0.0)

    return result^
