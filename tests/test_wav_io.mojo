"""Tests for WAV file I/O."""

from wav_io import read_wav, write_wav


fn abs_f32(x: Float32) -> Float32:
    if x < 0.0:
        return -x
    return x


fn assert_close(actual: Float32, expected: Float32, tol: Float32, msg: String) raises:
    if abs_f32(actual - expected) > tol:
        raise Error("FAIL: " + msg + " | got " + String(actual) + " expected " + String(expected))


fn test_write_and_read_roundtrip() raises:
    """Write a known signal, read it back, verify samples match."""
    print("Testing WAV roundtrip...")

    # Generate a 1-second 440Hz sine at 16kHz
    var sample_rate = 16000
    var n_samples = 16000
    var samples = List[Float32]()
    from math import sin
    from math.constants import pi
    for i in range(n_samples):
        var t = Float32(i) / Float32(sample_rate)
        samples.append(sin(2.0 * pi * 440.0 * t) * 0.5)

    # Write then read
    write_wav("/tmp/test_mojo_audio.wav", samples, sample_rate)
    var wav_result = read_wav("/tmp/test_mojo_audio.wav")
    var read_sr = wav_result[1]
    var read_samples = wav_result[0].copy()

    assert_close(Float32(read_sr), Float32(sample_rate), 0.0, "Sample rate mismatch")
    assert_close(Float32(len(read_samples)), Float32(n_samples), 0.0, "Sample count mismatch")

    # Check first 100 samples are close (16-bit quantization: ~0.00003 error)
    for i in range(100):
        assert_close(read_samples[i], samples[i], 0.001, "Sample " + String(i) + " mismatch")

    print("  Sample rate:", read_sr, "Hz")
    print("  Sample count:", len(read_samples))
    print("  First sample:", read_samples[0])
    print("  Expected:    ", samples[0])
    print("  WAV roundtrip correct!")


fn test_clipping_on_write() raises:
    """Values > 1.0 or < -1.0 must be clamped, not wrap-around."""
    print("Testing WAV clipping on write...")

    var samples = List[Float32]()
    samples.append(1.5)   # should clamp to 1.0
    samples.append(-2.0)  # should clamp to -1.0
    samples.append(0.5)

    write_wav("/tmp/test_clip.wav", samples, 16000)
    var clip_result = read_wav("/tmp/test_clip.wav")
    var read_samples = clip_result[0].copy()

    # Clamped values should round-trip as max/min int16
    if read_samples[0] < 0.99:
        raise Error("FAIL: Positive clipping: got " + String(read_samples[0]))
    if read_samples[1] > -0.99:
        raise Error("FAIL: Negative clipping: got " + String(read_samples[1]))
    print("  Clipping handled correctly!")


fn main() raises:
    test_write_and_read_roundtrip()
    test_clipping_on_write()
    print("\n=== All WAV I/O tests passed! ===")
