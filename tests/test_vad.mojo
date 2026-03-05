"""Tests for Voice Activity Detection."""

from vad import compute_rms_frames, frames_to_mask, trim_silence, get_voice_segments
from math import sin
from math.constants import pi


fn abs_f32(x: Float32) -> Float32:
    if x < 0.0:
        return -x
    return x


fn assert_close(actual: Float32, expected: Float32, tol: Float32, msg: String) raises:
    if abs_f32(actual - expected) > tol:
        raise Error("FAIL: " + msg + " | got " + String(actual) + " expected " + String(expected))


fn make_silence(n: Int) -> List[Float32]:
    var s = List[Float32]()
    for _ in range(n):
        s.append(0.0)
    return s^


fn make_tone(freq: Float32, sample_rate: Int, n: Int, amp: Float32 = 0.5) -> List[Float32]:
    var s = List[Float32]()
    for i in range(n):
        var t = Float32(i) / Float32(sample_rate)
        s.append(amp * sin(2.0 * pi * freq * t))
    return s^


fn append_lists(a: List[Float32], b: List[Float32]) -> List[Float32]:
    var out = List[Float32]()
    for i in range(len(a)):
        out.append(a[i])
    for i in range(len(b)):
        out.append(b[i])
    return out^


fn test_rms_silence_is_zero() raises:
    """Pure silence should give RMS ~= 0."""
    print("Testing RMS of silence...")
    var silent = make_silence(16000)
    var rms = compute_rms_frames(silent, 400, 160)

    for i in range(len(rms)):
        if rms[i] > 0.0001:
            raise Error("FAIL: silence frame " + String(i) + " has RMS " + String(rms[i]))
    print("  ✓ Silence gives zero RMS!")


fn test_rms_tone_is_nonzero() raises:
    """A tone should give nonzero RMS."""
    print("Testing RMS of tone...")
    var tone = make_tone(440.0, 16000, 16000, 0.5)
    var rms = compute_rms_frames(tone, 400, 160)

    for i in range(len(rms)):
        if rms[i] < 0.1:
            raise Error("FAIL: tone frame " + String(i) + " has RMS " + String(rms[i]) + " (too low)")
    print("  ✓ Tone gives nonzero RMS!")


fn test_trim_silence_removes_padding() raises:
    """Trim should remove leading/trailing silence."""
    print("Testing trim_silence...")

    # 0.5s silence + 0.5s tone + 0.5s silence
    var silence = make_silence(8000)
    var tone = make_tone(440.0, 16000, 8000, 0.5)
    var audio = append_lists(silence, append_lists(tone, silence))
    var trimmed = trim_silence(audio, 16000, 0.005, 400, 160, 50)

    # Trimmed should be much shorter than original 24000 samples
    if len(trimmed) >= 20000:
        raise Error("FAIL: trim_silence did not trim; output=" + String(len(trimmed)))

    # But should still contain the tone (at least 6000 samples)
    if len(trimmed) < 6000:
        raise Error("FAIL: trim_silence cut too aggressively; output=" + String(len(trimmed)))

    print("  Input length: ", len(audio))
    print("  Trimmed length:", len(trimmed))
    print("  ✓ Silence trimmed correctly!")


fn test_all_silence_returns_empty() raises:
    """All-silent audio should return empty list."""
    print("Testing all-silence returns empty...")
    var silent = make_silence(16000)
    var trimmed = trim_silence(silent, 16000)
    if len(trimmed) != 0:
        raise Error("FAIL: expected empty, got " + String(len(trimmed)))
    print("  ✓ All-silence returns empty!")


fn test_voice_segments_detected() raises:
    """get_voice_segments should find voice regions."""
    print("Testing voice segment detection...")

    var silence = make_silence(4800)   # 0.3s
    var tone = make_tone(440.0, 16000, 8000, 0.5)  # 0.5s
    var audio = append_lists(silence, append_lists(tone, silence))

    var segments = get_voice_segments(audio, 16000, 0.005)

    if len(segments) == 0:
        raise Error("FAIL: no voice segments found")

    print("  Found", len(segments), "voice segment(s)")
    print("  Segment 0: samples", segments[0][0], "to", segments[0][1])
    print("  ✓ Voice segments detected!")


fn test_frame_count() raises:
    """Frame count should follow (n - frame_size) // hop_length + 1."""
    print("Testing frame count formula...")
    var n_samples = 16000
    var frame_size = 400
    var hop_length = 160
    var expected_frames = (n_samples - frame_size) // hop_length + 1

    var audio = make_tone(440.0, 16000, n_samples, 0.5)
    var rms = compute_rms_frames(audio, frame_size, hop_length)

    if len(rms) != expected_frames:
        raise Error("FAIL: expected " + String(expected_frames) + " frames, got " + String(len(rms)))
    print("  Expected frames:", expected_frames, " Got:", len(rms))
    print("  ✓ Frame count correct!")


fn test_invalid_params_raise() raises:
    """Zero frame_size or hop_length must raise."""
    print("Testing invalid params raise...")

    var audio = make_tone(440.0, 16000, 1600, 0.5)

    var raised_frame = False
    try:
        var _ = compute_rms_frames(audio, 0, 160)
    except:
        raised_frame = True
    if not raised_frame:
        raise Error("FAIL: frame_size=0 should raise")

    var raised_hop = False
    try:
        var _ = compute_rms_frames(audio, 400, 0)
    except:
        raised_hop = True
    if not raised_hop:
        raise Error("FAIL: hop_length=0 should raise")

    print("  ✓ Invalid params raise correctly!")


fn main() raises:
    test_rms_silence_is_zero()
    test_rms_tone_is_nonzero()
    test_trim_silence_removes_padding()
    test_all_silence_returns_empty()
    test_voice_segments_detected()
    test_frame_count()
    test_invalid_params_raise()
    print("\n=== All VAD tests passed! ===")
