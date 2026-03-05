"""Tests for the Lanczos resampler."""

from resample import resample, resample_to_16k, resample_to_48k
from math import sin
from math.constants import pi


fn abs_f32(x: Float32) -> Float32:
    if x < 0.0:
        return -x
    return x


fn assert_close(actual: Float32, expected: Float32, tol: Float32, msg: String) raises:
    if abs_f32(actual - expected) > tol:
        raise Error("FAIL: " + msg + " | got " + String(actual) + " expected " + String(expected))


fn make_sine(freq: Float32, sample_rate: Int, n_samples: Int) -> List[Float32]:
    """Generate a pure sine wave."""
    var out = List[Float32]()
    for i in range(n_samples):
        var t = Float32(i) / Float32(sample_rate)
        out.append(sin(2.0 * pi * freq * t))
    return out^


fn test_identity_resample() raises:
    """Resampling at same rate must return identical signal."""
    print("Testing identity resample (16k -> 16k)...")
    var src = make_sine(440.0, 16000, 1600)
    var dst = resample(src, 16000, 16000)

    if len(dst) != len(src):
        raise Error("FAIL: Length mismatch: " + String(len(dst)) + " vs " + String(len(src)))
    for i in range(len(src)):
        assert_close(dst[i], src[i], 0.0001, "Sample " + String(i))
    print("  Identity resample correct!")


fn test_downsample_48k_to_16k() raises:
    """Downsample 48kHz -> 16kHz: output length should be 1/3."""
    print("Testing 48kHz -> 16kHz downsample...")
    var src = make_sine(440.0, 48000, 9600)  # 0.2 seconds (shorter for speed)
    var dst = resample(src, 48000, 16000)

    var expected_len = 3200
    var len_err = abs_f32(Float32(len(dst)) - Float32(expected_len))
    if len_err > Float32(expected_len) * 0.01:
        raise Error("FAIL: Output length " + String(len(dst)) + " expected ~" + String(expected_len))

    print("  Input:  9600 samples @48kHz")
    print("  Output:", len(dst), "samples @16kHz")
    print("  Downsample 48k->16k correct!")


fn test_downsample_44100_to_16k() raises:
    """Downsample 44.1kHz -> 16kHz (CD audio to HuBERT)."""
    print("Testing 44.1kHz -> 16kHz downsample...")
    var src = make_sine(440.0, 44100, 8820)  # 0.2 seconds
    var dst = resample(src, 44100, 16000)

    var expected_len = 3200
    var len_err = abs_f32(Float32(len(dst)) - Float32(expected_len))
    if len_err > Float32(expected_len) * 0.02:
        raise Error("FAIL: Output length " + String(len(dst)) + " expected ~" + String(expected_len))

    print("  Input:  8820 samples @44.1kHz")
    print("  Output:", len(dst), "samples @16kHz")
    print("  Downsample 44.1k->16k correct!")


fn test_upsample_16k_to_48k() raises:
    """Upsample 16kHz -> 48kHz: output length should be 3x."""
    print("Testing 16kHz -> 48kHz upsample...")
    var src = make_sine(440.0, 16000, 3200)  # 0.2 seconds
    var dst = resample(src, 16000, 48000)

    var expected_len = 9600
    var len_err = abs_f32(Float32(len(dst)) - Float32(expected_len))
    if len_err > Float32(expected_len) * 0.01:
        raise Error("FAIL: Output length " + String(len(dst)) + " expected ~" + String(expected_len))

    print("  Input:  3200 samples @16kHz")
    print("  Output:", len(dst), "samples @48kHz")
    print("  Upsample 16k->48k correct!")


fn test_amplitude_preservation() raises:
    """Resampled signal amplitude should be close to original."""
    print("Testing amplitude preservation through resample...")
    var src = List[Float32]()
    for i in range(9600):  # 0.2 seconds at 48kHz
        var t = Float32(i) / Float32(48000)
        src.append(0.8 * sin(2.0 * pi * 1000.0 * t))

    var dst = resample(src, 48000, 16000)

    var max_val = Float32(0.0)
    for i in range(len(dst)):
        var v = dst[i] if dst[i] >= 0.0 else -dst[i]
        if v > max_val:
            max_val = v

    # Allow 10% amplitude error (Lanczos has minor ripple at edges)
    assert_close(max_val, 0.8, 0.1, "Amplitude preservation")
    print("  Expected amplitude: 0.8")
    print("  Actual amplitude:  ", max_val)
    print("  Amplitude preserved through resample!")


fn test_convenience_functions() raises:
    """Test resample_to_16k and resample_to_48k helpers."""
    print("Testing convenience functions...")
    var src = make_sine(440.0, 48000, 4800)
    var dst_16k = resample_to_16k(src, 48000)
    var dst_48k = resample_to_48k(dst_16k, 16000)

    if len(dst_16k) < 1500 or len(dst_16k) > 1700:
        raise Error("FAIL: resample_to_16k output length unexpected: " + String(len(dst_16k)))
    if len(dst_48k) < 4600 or len(dst_48k) > 5000:
        raise Error("FAIL: resample_to_48k output length unexpected: " + String(len(dst_48k)))
    print("  Convenience functions work!")


fn main() raises:
    test_identity_resample()
    test_downsample_48k_to_16k()
    test_downsample_44100_to_16k()
    test_upsample_16k_to_48k()
    test_amplitude_preservation()
    test_convenience_functions()
    print("\n=== All resampler tests passed! ===")
