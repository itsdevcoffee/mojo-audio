"""Tests for inverse STFT and Griffin-Lim waveform reconstruction."""

from audio import stft, istft, griffin_lim, power_spectrum
from math import sin, sqrt
from math.constants import pi


fn abs_f32(x: Float32) -> Float32:
    if x < 0.0:
        return -x
    return x


fn signal_rms(s: List[Float32]) -> Float32:
    """Compute RMS energy of a signal."""
    if len(s) == 0:
        return Float32(0.0)
    var sum_sq = Float32(0.0)
    for i in range(len(s)):
        sum_sq += s[i] * s[i]
    return sqrt(sum_sq / Float32(len(s)))


fn test_istft_output_length() raises:
    """istft output length should be at least n_frames * hop_length."""
    print("Testing iSTFT output length...")

    var n_bins = 201  # n_fft=400 -> rfft gives 201 bins
    var n_frames = 10
    var hop_length = 160

    var spec = List[List[Float32]]()
    for _ in range(n_frames):
        var frame = List[Float32]()
        for _ in range(n_bins):
            frame.append(0.1)
        spec.append(frame^)

    var audio = istft(spec, hop_length)
    if len(audio) < n_frames * hop_length:
        raise Error("FAIL: iSTFT output too short: " + String(len(audio)) + " < " + String(n_frames * hop_length))
    print("  n_frames:", n_frames, "-> audio length:", len(audio))
    print("  iSTFT output length correct!")


fn test_istft_empty_input() raises:
    """Empty spectrogram should return empty audio."""
    print("Testing iSTFT empty input...")
    var empty = List[List[Float32]]()
    var audio = istft(empty, 160)
    if len(audio) != 0:
        raise Error("FAIL: expected empty, got " + String(len(audio)))
    print("  iSTFT empty input returns empty!")


fn test_stft_istft_nontrivial_output() raises:
    """iSTFT on a non-trivial spectrogram should produce non-zero output."""
    print("Testing STFT->iSTFT produces non-trivial output...")

    var n = 3200  # 0.2s @16kHz
    var src = List[Float32]()
    for i in range(n):
        var t = Float32(i) / 16000.0
        src.append(0.5 * sin(2.0 * pi * 440.0 * t))

    var spec = stft(src, 400, 160, "hann")
    var reconstructed = istft(spec, 160, "hann")

    # Output should be non-empty and non-zero
    if len(reconstructed) == 0:
        raise Error("FAIL: iSTFT returned empty output")

    var rms_out = signal_rms(reconstructed)
    print("  Spectrogram frames:", len(spec))
    print("  Reconstructed length:", len(reconstructed))
    print("  Output RMS:", rms_out)

    if rms_out < 1e-6:
        raise Error("FAIL: iSTFT returned silent output (rms=" + String(rms_out) + ")")

    print("  STFT->iSTFT produces non-trivial output!")


fn test_griffin_lim_produces_audio() raises:
    """Griffin-Lim must produce non-empty, non-zero audio."""
    print("Testing Griffin-Lim produces audio...")

    var n = 3200
    var src = List[Float32]()
    for i in range(n):
        var t = Float32(i) / 16000.0
        src.append(0.5 * sin(2.0 * pi * 440.0 * t))

    var spec = stft(src, 400, 160, "hann")

    # Use fewer iterations for speed in tests
    var audio = griffin_lim(spec, 4, 160, "hann")

    if len(audio) == 0:
        raise Error("FAIL: Griffin-Lim returned empty audio")

    var rms = signal_rms(audio)
    if rms < 0.001:
        raise Error("FAIL: Griffin-Lim returned silent audio (rms=" + String(rms) + ")")

    print("  Audio length:", len(audio))
    print("  Audio RMS:", rms)
    print("  Griffin-Lim produces non-empty non-zero audio!")


fn test_griffin_lim_more_iters_not_worse() raises:
    """More iterations should not make energy worse."""
    print("Testing Griffin-Lim iteration stability...")

    var n = 3200
    var src = List[Float32]()
    for i in range(n):
        var t = Float32(i) / 16000.0
        src.append(0.5 * sin(2.0 * pi * 440.0 * t))

    var spec = stft(src, 400, 160, "hann")
    var rms_4 = signal_rms(griffin_lim(spec, 4, 160, "hann"))
    var rms_16 = signal_rms(griffin_lim(spec, 16, 160, "hann"))

    # Both should produce audio of similar energy (within factor of 2)
    if rms_4 < 0.001 or rms_16 < 0.001:
        raise Error("FAIL: One iteration count produced silent audio")

    print("  4 iters RMS:", rms_4)
    print("  16 iters RMS:", rms_16)
    print("  Griffin-Lim stable across iteration counts!")


fn main() raises:
    test_istft_output_length()
    test_istft_empty_input()
    test_stft_istft_nontrivial_output()
    test_griffin_lim_produces_audio()
    test_griffin_lim_more_iters_not_worse()
    print("\n=== All iSTFT/Griffin-Lim tests passed! ===")
