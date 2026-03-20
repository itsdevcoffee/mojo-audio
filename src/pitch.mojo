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

from math import exp, log, cos, sin, atan2, sqrt
from math.constants import pi
from audio import hann_window, _stft_complex, _irfft, Complex
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


fn _wrap_to_pi(x: Float32) -> Float32:
    """Wrap x to the range [-pi, pi]."""
    var result = x
    var two_pi = Float32(2.0) * pi
    while result > pi:
        result -= two_pi
    while result < -pi:
        result += two_pi
    return result


fn _time_stretch_pv(
    audio: List[Float32],
    stretch_factor: Float32,
    n_fft: Int = 512,
    hop_length: Int = 128,
) raises -> List[Float32]:
    """
    Time-stretch audio using a phase-locked phase vocoder.

    Tracks instantaneous frequency per bin and propagates phase correctly
    across stretched frames, eliminating the phasiness artifact of plain OLA
    on sustained tones (piano, held vocals).

    stretch_factor > 1.0 -> longer output (slower)
    stretch_factor < 1.0 -> shorter output (faster)

    Note: n_fft must be a power of 2 (default 512 satisfies this).
    """
    var n_src = len(audio)
    var n_frames_check = (n_src - n_fft) // hop_length + 1
    if n_frames_check <= 0:
        var copy = List[Float32]()
        for i in range(n_src):
            copy.append(audio[i])
        return copy^

    # Compute complex STFT: List[List[Complex]], shape (n_frames, n_fft//2+1)
    var frames = _stft_complex(audio, n_fft, hop_length)
    var n_frames = len(frames)
    if n_frames == 0:
        var copy = List[Float32]()
        for i in range(n_src):
            copy.append(audio[i])
        return copy^

    var n_bins = len(frames[0])  # n_fft//2+1

    # Output hop size (stretched)
    var out_hop = Int(Float32(hop_length) * stretch_factor + 0.5)
    if out_hop < 1:
        out_hop = 1

    # Synthesis phase accumulator — start with analysis phase of frame 0
    var synth_phase = List[Float32]()
    var prev_phase = List[Float32]()
    for k in range(n_bins):
        var c = frames[0][k].copy()
        var ph = atan2(c.imag, c.real)
        synth_phase.append(ph)
        prev_phase.append(ph)

    # Output buffer
    var out_len = (n_frames - 1) * out_hop + n_fft
    var output = List[Float32]()
    var win_sum = List[Float32]()
    for _ in range(out_len):
        output.append(0.0)
        win_sum.append(0.0)

    var win = hann_window(n_fft)
    var two_pi = Float32(2.0) * pi

    for f in range(n_frames):
        # Build phase-corrected synthesis frame
        var synth_frame = List[Complex]()
        for k in range(n_bins):
            var c = frames[f][k].copy()
            var mag = sqrt(c.real * c.real + c.imag * c.imag)

            if f > 0:
                # Instantaneous frequency via phase deviation
                var curr_phase = atan2(c.imag, c.real)
                var phi_expected = two_pi * Float32(k) * Float32(hop_length) / Float32(n_fft)
                var delta = curr_phase - prev_phase[k] - phi_expected
                var wrapped = _wrap_to_pi(delta)
                var omega = (phi_expected + wrapped) / Float32(hop_length)
                synth_phase[k] = synth_phase[k] + omega * Float32(out_hop)
                prev_phase[k] = curr_phase

            synth_frame.append(Complex(mag * cos(synth_phase[k]), mag * sin(synth_phase[k])))

        # IFFT to time domain (n_fft must be power of 2)
        var frame_td = _irfft(synth_frame, n_fft)

        # Windowed overlap-add
        var dst_start = f * out_hop
        for i in range(n_fft):
            if dst_start + i < out_len and i < len(frame_td):
                output[dst_start + i] += frame_td[i] * win[i]
                win_sum[dst_start + i] += win[i] * win[i]

    # Normalize by window sum
    for i in range(out_len):
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
    var stretched = _time_stretch_pv(audio, stretch, n_fft, hop_length)

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
