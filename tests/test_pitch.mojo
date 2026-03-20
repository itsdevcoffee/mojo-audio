"""Tests for phase vocoder pitch shifter."""

from pitch import pitch_shift, semitones_to_ratio, transpose_semitones
from audio import rfft, Complex
from math import sin, sqrt
from math.constants import pi


fn abs_f32(x: Float32) -> Float32:
    if x < 0.0:
        return -x
    return x


fn assert_close(actual: Float32, expected: Float32, tol: Float32, msg: String) raises:
    if abs_f32(actual - expected) > tol:
        raise Error("FAIL: " + msg + " | got " + String(actual) + " expected " + String(expected))


fn make_tone(freq: Float32, sample_rate: Int, n: Int) -> List[Float32]:
    var s = List[Float32]()
    for i in range(n):
        var t = Float32(i) / Float32(sample_rate)
        s.append(0.5 * sin(2.0 * pi * freq * t))
    return s^


fn test_semitone_ratio_octave() raises:
    """12 semitones = 2.0x frequency ratio."""
    print("Testing semitone ratio calculation...")
    var ratio = semitones_to_ratio(12.0)
    assert_close(ratio, 2.0, 0.001, "12 semitones = 2x ratio")

    var ratio_zero = semitones_to_ratio(0.0)
    assert_close(ratio_zero, 1.0, 0.001, "0 semitones = 1x ratio")

    var ratio_down = semitones_to_ratio(-12.0)
    assert_close(ratio_down, 0.5, 0.001, "-12 semitones = 0.5x ratio")
    print("  ✓ Semitone ratios correct!")


fn test_pitch_shift_preserves_length() raises:
    """Pitch-shifted output must be same length as input."""
    print("Testing pitch shift preserves length...")
    var n = 4800  # 0.3s @16kHz
    var audio = make_tone(440.0, 16000, n)

    var shifted_2 = pitch_shift(audio, 2.0, 16000)
    if len(shifted_2) != n:
        raise Error("FAIL: pitch_shift(2.0) changed length: " + String(len(shifted_2)) + " vs " + String(n))
    print("  Semitones: 2.0 -> length:", len(shifted_2), "(OK)")

    var shifted_m2 = pitch_shift(audio, -2.0, 16000)
    if len(shifted_m2) != n:
        raise Error("FAIL: pitch_shift(-2.0) changed length: " + String(len(shifted_m2)) + " vs " + String(n))
    print("  Semitones: -2.0 -> length:", len(shifted_m2), "(OK)")

    var shifted_7 = pitch_shift(audio, 7.0, 16000)
    if len(shifted_7) != n:
        raise Error("FAIL: pitch_shift(7.0) changed length: " + String(len(shifted_7)) + " vs " + String(n))
    print("  Semitones: 7.0 -> length:", len(shifted_7), "(OK)")

    var shifted_12 = pitch_shift(audio, 12.0, 16000)
    if len(shifted_12) != n:
        raise Error("FAIL: pitch_shift(12.0) changed length: " + String(len(shifted_12)) + " vs " + String(n))
    print("  Semitones: 12.0 -> length:", len(shifted_12), "(OK)")

    print("  ✓ Length preserved!")


fn test_zero_shift_identity() raises:
    """0 semitone shift must return identical audio."""
    print("Testing zero shift is identity...")
    var audio = make_tone(440.0, 16000, 1600)
    var shifted = pitch_shift(audio, 0.0, 16000)

    if len(shifted) != len(audio):
        raise Error("FAIL: zero shift changed length")
    for i in range(len(audio)):
        assert_close(shifted[i], audio[i], 0.0001, "Sample " + String(i))
    print("  ✓ Zero shift is identity!")


fn test_transpose_normalization() raises:
    """transpose_semitones should normalize to [-6, +6]."""
    print("Testing semitone normalization...")
    assert_close(transpose_semitones(7.0), -5.0, 0.001, "7 -> -5")
    assert_close(transpose_semitones(-7.0), 5.0, 0.001, "-7 -> 5")
    assert_close(transpose_semitones(0.0), 0.0, 0.001, "0 -> 0")
    assert_close(transpose_semitones(12.0), 0.0, 0.001, "12 -> 0")
    assert_close(transpose_semitones(-12.0), 0.0, 0.001, "-12 -> 0")
    print("  ✓ Semitone normalization correct!")


fn test_pitch_shift_not_silent() raises:
    """Shifted audio must not be silent (energy check)."""
    print("Testing pitch shift is non-silent...")
    var audio = make_tone(440.0, 16000, 3200)
    var shifted = pitch_shift(audio, 5.0, 16000)

    var energy = Float32(0.0)
    for i in range(len(shifted)):
        energy += shifted[i] * shifted[i]
    energy /= Float32(len(shifted))

    if energy < 0.01:
        raise Error("FAIL: shifted audio is too quiet (RMS energy=" + String(energy) + ")")
    print("  Energy after shift:", energy, "(OK)")
    print("  ✓ Pitch-shifted audio is non-silent!")


fn fft_peak_bin(audio: List[Float32]) raises -> Int:
    """Return the FFT bin index with the highest magnitude."""
    var spectrum = rfft(audio)
    var peak_bin = 0
    var peak_mag = Float32(0.0)
    for k in range(len(spectrum)):
        var c = spectrum[k].copy()
        var mag = c.real * c.real + c.imag * c.imag
        if mag > peak_mag:
            peak_mag = mag
            peak_bin = k
    return peak_bin


fn test_pitch_shift_frequency() raises:
    """Pitch shift of +7 semitones on a 440Hz tone should raise peak to ~659Hz."""
    print("Testing pitch shift frequency accuracy...")
    var sr = 16000
    var n = 8192  # power of 2 for clean FFT bins
    var audio = make_tone(440.0, sr, n)

    var shifted = pitch_shift(audio, 7.0, sr)

    var peak_bin = fft_peak_bin(shifted)
    # Bin resolution: sr / n = 16000/8192 ≈ 1.95 Hz/bin
    # 440 * 2^(7/12) ≈ 659.26 Hz → bin ≈ 339
    var peak_freq = Float32(peak_bin) * Float32(sr) / Float32(n)
    var expected_freq = Float32(440.0) * semitones_to_ratio(7.0)
    var bin_width = Float32(sr) / Float32(n)

    if abs_f32(peak_freq - expected_freq) > bin_width * 5.0:
        raise Error(
            "FAIL: peak freq after +7st: got "
            + String(peak_freq)
            + "Hz, expected ~"
            + String(expected_freq)
            + "Hz"
        )
    print("  Peak frequency:", peak_freq, "Hz (expected ~", expected_freq, "Hz)")
    print("  ✓ Frequency accuracy OK!")


fn test_pitch_shift_energy_concentration() raises:
    """After pitch shift, most energy should be near the shifted frequency."""
    print("Testing energy concentration after pitch shift...")
    var sr = 16000
    var n = 8192
    var audio = make_tone(440.0, sr, n)

    var shifted = pitch_shift(audio, 7.0, sr)
    var spectrum = rfft(shifted)

    # Expected peak bin for ~659Hz
    var expected_freq = Float32(440.0) * semitones_to_ratio(7.0)
    var expected_bin = Int(expected_freq * Float32(n) / Float32(sr) + 0.5)

    # Energy in ±10 bins around expected vs total energy
    var peak_energy = Float32(0.0)
    var total_energy = Float32(0.0)
    for k in range(len(spectrum)):
        var c = spectrum[k].copy()
        var mag2 = c.real * c.real + c.imag * c.imag
        total_energy += mag2
        var dist = k - expected_bin
        if dist < 0:
            dist = -dist
        if dist <= 10:
            peak_energy += mag2

    if total_energy < 1e-8:
        raise Error("FAIL: shifted audio is silent")

    var concentration = peak_energy / total_energy
    if concentration < 0.5:
        raise Error(
            "FAIL: energy too spread out (concentration="
            + String(concentration)
            + "), phase vocoder may not be working"
        )
    print("  Energy concentration near target frequency:", concentration)
    print("  ✓ Phase vocoder energy concentration OK!")


fn main() raises:
    test_semitone_ratio_octave()
    test_pitch_shift_preserves_length()
    test_zero_shift_identity()
    test_transpose_normalization()
    test_pitch_shift_not_silent()
    test_pitch_shift_frequency()
    test_pitch_shift_energy_concentration()
    print("\n=== All pitch shifter tests passed! ===")
