"""Tests for WAV file I/O."""

from wav_io import read_wav, write_wav
from python import Python


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


fn test_32bit_pcm_read() raises:
    """Test reading 32-bit PCM WAV file."""
    print("Testing 32-bit PCM read...")
    # Write a 32-bit PCM WAV via Python
    var wave = Python.import_module("wave")
    var array_mod = Python.import_module("array")

    # Write known samples as 32-bit PCM
    var wf = wave.open("/tmp/test_32bit.wav", "wb")
    wf.setnchannels(1)
    wf.setsampwidth(4)
    wf.setframerate(16000)
    # Sample values: max positive, zero, max negative
    var arr = array_mod.array("i", Python.evaluate("[2147483647, 0, -2147483648]"))
    wf.writeframes(arr.tobytes())
    wf.close()

    # Read back with our function
    var samples: List[Float32]
    var sr: Int
    var result = read_wav("/tmp/test_32bit.wav")
    samples = result[0].copy()
    sr = Int(result[1])

    if sr != 16000:
        raise Error("FAIL: sample rate " + String(sr))
    if len(samples) != 3:
        raise Error("FAIL: expected 3 samples, got " + String(len(samples)))
    # Max positive should be ~1.0
    if samples[0] < 0.99:
        raise Error("FAIL: max positive sample " + String(samples[0]))
    # Zero should be ~0.0
    if abs_f32(samples[1]) > 0.001:
        raise Error("FAIL: zero sample " + String(samples[1]))
    # Max negative should be ~-1.0
    if samples[2] > -0.99:
        raise Error("FAIL: max negative sample " + String(samples[2]))
    print("  ✓ 32-bit PCM read works correctly!")


fn test_unsupported_format_raises() raises:
    """read_wav must raise on unsupported sample width (e.g. 8-bit)."""
    print("Testing unsupported format raises error...")
    var wave = Python.import_module("wave")
    var array_mod = Python.import_module("array")

    # Write an 8-bit WAV (sample width = 1)
    var wf = wave.open("/tmp/test_8bit.wav", "wb")
    wf.setnchannels(1)
    wf.setsampwidth(1)
    wf.setframerate(16000)
    var arr = array_mod.array("b", Python.evaluate("[0, 64, 127]"))
    wf.writeframes(arr.tobytes())
    wf.close()

    # Expect read_wav to raise
    var raised = False
    try:
        var _ = read_wav("/tmp/test_8bit.wav")
    except:
        raised = True

    if not raised:
        raise Error("FAIL: read_wav should raise for 8-bit WAV but did not")
    print("  ✓ Unsupported format raises correctly!")


fn test_stereo_mono_mixdown() raises:
    """Stereo WAV should be mixed down to mono by averaging channels."""
    print("Testing stereo mono mixdown...")
    var wave = Python.import_module("wave")
    var array_mod = Python.import_module("array")

    # Write stereo WAV: left=16384 (0.5), right=0 → mono should be ~0.25
    var wf = wave.open("/tmp/test_stereo.wav", "wb")
    wf.setnchannels(2)
    wf.setsampwidth(2)
    wf.setframerate(16000)
    # Interleaved: [left, right] = [16384, 0]
    var arr = array_mod.array("h", Python.evaluate("[16384, 0]"))
    wf.writeframes(arr.tobytes())
    wf.close()

    var result = read_wav("/tmp/test_stereo.wav")
    var samples = result[0].copy()

    if len(samples) != 1:
        raise Error("FAIL: expected 1 mono sample, got " + String(len(samples)))
    # 16384 / 32768.0 = 0.5, average with 0 = 0.25
    assert_close(samples[0], 0.25, 0.002, "Stereo mixdown average")
    print("  Sample value:", samples[0], "(expected ~0.25)")
    print("  ✓ Stereo mixdown correct!")


fn main() raises:
    test_write_and_read_roundtrip()
    test_clipping_on_write()
    test_32bit_pcm_read()
    test_unsupported_format_raises()
    test_stereo_mono_mixdown()
    print("\n=== All WAV I/O tests passed! ===")
