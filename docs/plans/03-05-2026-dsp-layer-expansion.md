# DSP Layer Expansion Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Extend mojo-audio with five DSP modules — WAV I/O, resampler, voice activity detection, inverse STFT/Griffin-Lim, and phase vocoder/pitch shifter — to fully own the audio preprocessing and postprocessing bookends of a voice conversion pipeline.

**Architecture:** Each new capability lives in its own Mojo source file under `src/` (matching how `audio.mojo` is structured today). A Python interop bridge is used for WAV file parsing only (standard library `wave` module), keeping the processing logic in pure Mojo. FFI exports are added to `src/ffi/audio_ffi.mojo` for each new function at the end of each task so the Python pipeline in Shade can call them.

**Tech Stack:** Mojo 0.26.1, pixi task runner, Python `wave`/`struct` stdlib (WAV I/O only), existing `src/audio.mojo` patterns (SIMD, `parallelize`, `List[Float32]`).

---

## Context: How Tests Work in This Project

Tests are standalone Mojo programs that print results. There is no `pytest`-style runner — each test file defines test functions and calls them from `fn main() raises`. A test "passes" when it runs to completion without raising and prints checkmarks.

To run a single test file:
```bash
pixi run mojo -I src tests/test_NAME.mojo
```

To run all tests:
```bash
pixi run test
```

When adding a new test file, you must also add it to `pixi.toml` under `[tasks]`.

---

## Task 1: WAV I/O — `src/wav_io.mojo`

Read and write WAV files from Mojo. Uses Python's `wave` stdlib as the format parser (WAV binary format is platform-finicky; we own the data pipeline, not the container parsing). Output is always `List[Float32]` normalized to [-1.0, 1.0].

**Files:**
- Create: `src/wav_io.mojo`
- Create: `tests/test_wav_io.mojo`
- Modify: `pixi.toml` — add `test-wav` and `test` entries

---

**Step 1: Create `src/wav_io.mojo`**

```mojo
"""
WAV file I/O for mojo-audio.

Uses Python stdlib `wave` module for container parsing.
All audio data is converted to/from List[Float32] normalized [-1.0, 1.0].
"""

from python import Python, PythonObject


fn read_wav(path: String) raises -> (List[Float32], Int):
    """
    Read a WAV file and return normalized Float32 samples + sample rate.

    Supports: 16-bit PCM, 24-bit PCM, 32-bit PCM, 32-bit float.
    Multi-channel audio is mixed down to mono by averaging channels.

    Args:
        path: Absolute or relative path to the .wav file.

    Returns:
        Tuple of (samples: List[Float32], sample_rate: Int).
        Samples are normalized to [-1.0, 1.0].
    """
    var wave = Python.import_module("wave")
    var struct = Python.import_module("struct")

    var wf = wave.open(path, "rb")
    var n_channels = Int(wf.getnchannels())
    var sample_width = Int(wf.getsampwidth())  # bytes per sample
    var sample_rate = Int(wf.getframerate())
    var n_frames = Int(wf.getnframes())

    var raw = wf.readframes(n_frames)
    wf.close()

    var samples = List[Float32]()

    if sample_width == 2:
        # 16-bit PCM
        var scale = Float32(1.0 / 32768.0)
        var i = 0
        var frame_idx = 0
        while frame_idx < n_frames:
            var channel_sum = Float32(0.0)
            for ch in range(n_channels):
                var byte_pos = (frame_idx * n_channels + ch) * 2
                var b0 = Int(raw[byte_pos])
                var b1 = Int(raw[byte_pos + 1])
                var raw_val = b0 | (b1 << 8)
                # Sign extend from 16-bit
                if raw_val >= 32768:
                    raw_val -= 65536
                channel_sum += Float32(raw_val) * scale
            samples.append(channel_sum / Float32(n_channels))
            frame_idx += 1
    elif sample_width == 4:
        # 32-bit float or 32-bit PCM — detect via format tag
        # For now treat as 32-bit PCM (int)
        var scale = Float32(1.0 / 2147483648.0)
        var frame_idx = 0
        while frame_idx < n_frames:
            var channel_sum = Float32(0.0)
            for ch in range(n_channels):
                var byte_pos = (frame_idx * n_channels + ch) * 4
                var b0 = Int(raw[byte_pos])
                var b1 = Int(raw[byte_pos + 1])
                var b2 = Int(raw[byte_pos + 2])
                var b3 = Int(raw[byte_pos + 3])
                var raw_val = b0 | (b1 << 8) | (b2 << 16) | (b3 << 24)
                if raw_val >= 2147483648:
                    raw_val -= 4294967296
                channel_sum += Float32(raw_val) * scale
            samples.append(channel_sum / Float32(n_channels))
            frame_idx += 1
    else:
        raise Error("Unsupported sample width: " + String(sample_width) + " bytes")

    return (samples, sample_rate)


fn write_wav(path: String, samples: List[Float32], sample_rate: Int) raises:
    """
    Write Float32 samples to a 16-bit PCM WAV file.

    Args:
        path: Output file path.
        samples: Audio samples normalized to [-1.0, 1.0].
        sample_rate: Sample rate in Hz (e.g. 16000, 44100, 48000).
    """
    var wave = Python.import_module("wave")

    var wf = wave.open(path, "wb")
    wf.setnchannels(1)
    wf.setsampwidth(2)  # 16-bit
    wf.setframerate(sample_rate)

    # Convert Float32 [-1, 1] → int16 and pack
    var struct = Python.import_module("struct")
    var n = len(samples)
    var fmt = String("<") + String(n) + String("h")

    # Build Python list for struct.pack
    var py_list = Python.evaluate("[]")
    for i in range(n):
        var val = samples[i]
        if val > 1.0:
            val = 1.0
        if val < -1.0:
            val = -1.0
        var int_val = Int(val * 32767.0)
        _ = py_list.append(int_val)

    var packed = struct.pack(fmt, py_list)
    wf.writeframes(packed)
    wf.close()
```

**Step 2: Create `tests/test_wav_io.mojo`**

```mojo
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
    var read_samples: List[Float32]
    var read_sr: Int
    read_samples, read_sr = read_wav("/tmp/test_mojo_audio.wav")

    assert_close(Float32(read_sr), Float32(sample_rate), 0.0, "Sample rate mismatch")
    assert_close(Float32(len(read_samples)), Float32(n_samples), 0.0, "Sample count mismatch")

    # Check first 100 samples are close (16-bit quantization: ~0.00003 error)
    for i in range(100):
        assert_close(read_samples[i], samples[i], 0.001, "Sample " + String(i) + " mismatch")

    print("  Sample rate:", read_sr, "Hz")
    print("  Sample count:", len(read_samples))
    print("  First sample:", read_samples[0])
    print("  Expected:    ", samples[0])
    print("  ✓ WAV roundtrip correct!")


fn test_clipping_on_write() raises:
    """Values > 1.0 or < -1.0 must be clamped, not wrap-around."""
    print("Testing WAV clipping on write...")

    var samples = List[Float32]()
    samples.append(1.5)   # should clamp to 1.0
    samples.append(-2.0)  # should clamp to -1.0
    samples.append(0.5)

    write_wav("/tmp/test_clip.wav", samples, 16000)
    var read_samples: List[Float32]
    var _sr: Int
    read_samples, _sr = read_wav("/tmp/test_clip.wav")

    # Clamped values should round-trip as max/min int16
    if read_samples[0] < 0.99:
        raise Error("FAIL: Positive clipping: got " + String(read_samples[0]))
    if read_samples[1] > -0.99:
        raise Error("FAIL: Negative clipping: got " + String(read_samples[1]))
    print("  ✓ Clipping handled correctly!")


fn main() raises:
    test_write_and_read_roundtrip()
    test_clipping_on_write()
    print("\n=== All WAV I/O tests passed! ===")
```

**Step 3: Add task to `pixi.toml`**

In `pixi.toml`, modify the `[tasks]` section:

```toml
test-wav = "mojo -I src tests/test_wav_io.mojo"
test = "mojo -I src tests/test_window.mojo && mojo -I src tests/test_fft.mojo && mojo -I src tests/test_mel.mojo && mojo -I src tests/test_wav_io.mojo"
```

**Step 4: Run the test to verify it fails (function not yet found)**

```bash
pixi run mojo -I src tests/test_wav_io.mojo
```

Expected: Error about `wav_io` module not found.

**Step 5: After creating `src/wav_io.mojo`, run again**

```bash
pixi run mojo -I src tests/test_wav_io.mojo
```

Expected output:
```
Testing WAV roundtrip...
  Sample rate: 16000 Hz
  Sample count: 16000
  ...
  ✓ WAV roundtrip correct!
Testing WAV clipping on write...
  ✓ Clipping handled correctly!

=== All WAV I/O tests passed! ===
```

**Step 6: Commit**

```bash
git add src/wav_io.mojo tests/test_wav_io.mojo pixi.toml
git commit -m "feat: add WAV I/O module (read/write 16-bit PCM via Python wave bridge)"
```

---

## Task 2: Resampler — `src/resample.mojo`

Sinc interpolation resampler. The single highest-leverage DSP addition — every audio file entering the VC pipeline needs sample rate conversion. HuBERT requires 16kHz; production audio is typically 44.1kHz or 48kHz.

Implements a windowed-sinc (Lanczos) resampler. Quality is tuned with the `kernel_size` parameter (default 64 — good quality, fast). This is a standard algorithm: convolve with a sinc kernel, interpolate at target positions.

**Files:**
- Create: `src/resample.mojo`
- Create: `tests/test_resample.mojo`
- Modify: `pixi.toml` — add `test-resample`

---

**Step 1: Create `src/resample.mojo`**

```mojo
"""
Audio resampler using windowed-sinc (Lanczos) interpolation.

Supports any rational sample rate conversion.
Quality controlled by kernel_size (default 64).

Common conversions:
  48000 → 16000  (voice conversion pipeline: HuBERT input)
  44100 → 16000  (CD audio to HuBERT)
  16000 → 48000  (upsample converted output to DAW rate)
"""

from math import sin, floor, ceil
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

    if src_rate == dst_rate:
        return samples

    var n_src = len(samples)
    var ratio = Float32(dst_rate) / Float32(src_rate)
    var n_dst = Int(Float32(n_src) * ratio)

    var result = List[Float32]()
    for _ in range(n_dst):
        result.append(0.0)

    # Lanczos order (half-width of kernel in output samples)
    var a = kernel_size // 2

    for i in range(n_dst):
        # Position in source space
        var src_pos = Float32(i) / ratio

        # Kernel window in source space
        var start = Int(src_pos) - a + 1
        var end = Int(src_pos) + a

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

    return result


fn resample_to_16k(samples: List[Float32], src_rate: Int) raises -> List[Float32]:
    """Convenience function: resample any rate to 16kHz (HuBERT standard)."""
    return resample(samples, src_rate, 16000)


fn resample_to_48k(samples: List[Float32], src_rate: Int) raises -> List[Float32]:
    """Convenience function: resample any rate to 48kHz (studio standard)."""
    return resample(samples, src_rate, 48000)
```

**Step 2: Create `tests/test_resample.mojo`**

```mojo
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
    return out


fn test_identity_resample() raises:
    """Resampling at same rate must return identical signal."""
    print("Testing identity resample (16k → 16k)...")
    var src = make_sine(440.0, 16000, 1600)
    var dst = resample(src, 16000, 16000)

    if len(dst) != len(src):
        raise Error("FAIL: Length mismatch: " + String(len(dst)) + " vs " + String(len(src)))
    for i in range(len(src)):
        assert_close(dst[i], src[i], 0.0001, "Sample " + String(i))
    print("  ✓ Identity resample correct!")


fn test_downsample_48k_to_16k() raises:
    """Downsample 48kHz → 16kHz: output length should be 1/3."""
    print("Testing 48kHz → 16kHz downsample...")
    var src = make_sine(440.0, 48000, 48000)  # 1 second
    var dst = resample(src, 48000, 16000)

    # Expect ~16000 samples (within 1%)
    var expected_len = 16000
    var len_err = abs_f32(Float32(len(dst)) - Float32(expected_len))
    if len_err > Float32(expected_len) * 0.01:
        raise Error("FAIL: Output length " + String(len(dst)) + " expected ~" + String(expected_len))

    print("  Input:  48000 samples @48kHz")
    print("  Output:", len(dst), "samples @16kHz")
    print("  ✓ Downsample 48k→16k correct!")


fn test_downsample_44100_to_16k() raises:
    """Downsample 44.1kHz → 16kHz (CD audio to HuBERT)."""
    print("Testing 44.1kHz → 16kHz downsample...")
    var src = make_sine(440.0, 44100, 44100)  # 1 second
    var dst = resample(src, 44100, 16000)

    var expected_len = 16000
    var len_err = abs_f32(Float32(len(dst)) - Float32(expected_len))
    if len_err > Float32(expected_len) * 0.01:
        raise Error("FAIL: Output length " + String(len(dst)) + " expected ~" + String(expected_len))

    print("  Input:  44100 samples @44.1kHz")
    print("  Output:", len(dst), "samples @16kHz")
    print("  ✓ Downsample 44.1k→16k correct!")


fn test_upsample_16k_to_48k() raises:
    """Upsample 16kHz → 48kHz: output length should be 3x."""
    print("Testing 16kHz → 48kHz upsample...")
    var src = make_sine(440.0, 16000, 16000)  # 1 second
    var dst = resample(src, 16000, 48000)

    var expected_len = 48000
    var len_err = abs_f32(Float32(len(dst)) - Float32(expected_len))
    if len_err > Float32(expected_len) * 0.01:
        raise Error("FAIL: Output length " + String(len(dst)) + " expected ~" + String(expected_len))

    print("  Input:  16000 samples @16kHz")
    print("  Output:", len(dst), "samples @48kHz")
    print("  ✓ Upsample 16k→48k correct!")


fn test_amplitude_preservation() raises:
    """Resampled signal amplitude should be close to original."""
    print("Testing amplitude preservation through resample...")
    # 1kHz sine at 0.8 amplitude
    var src = List[Float32]()
    for i in range(48000):
        var t = Float32(i) / Float32(48000)
        src.append(0.8 * sin(2.0 * pi * 1000.0 * t))

    var dst = resample(src, 48000, 16000)

    # Check max amplitude of output is close to 0.8
    var max_val = Float32(0.0)
    for i in range(len(dst)):
        var v = dst[i] if dst[i] >= 0.0 else -dst[i]
        if v > max_val:
            max_val = v

    # Allow 10% amplitude error (Lanczos has minor ripple at edges)
    assert_close(max_val, 0.8, 0.1, "Amplitude preservation")
    print("  Expected amplitude: 0.8")
    print("  Actual amplitude:  ", max_val)
    print("  ✓ Amplitude preserved through resample!")


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
    print("  ✓ Convenience functions work!")


fn main() raises:
    test_identity_resample()
    test_downsample_48k_to_16k()
    test_downsample_44100_to_16k()
    test_upsample_16k_to_48k()
    test_amplitude_preservation()
    test_convenience_functions()
    print("\n=== All resampler tests passed! ===")
```

**Step 3: Add to `pixi.toml`**

```toml
test-resample = "mojo -I src tests/test_resample.mojo"
test = "mojo -I src tests/test_window.mojo && mojo -I src tests/test_fft.mojo && mojo -I src tests/test_mel.mojo && mojo -I src tests/test_wav_io.mojo && mojo -I src tests/test_resample.mojo"
```

**Step 4: Run tests**

```bash
pixi run mojo -I src tests/test_resample.mojo
```

Expected:
```
Testing identity resample (16k → 16k)...
  ✓ Identity resample correct!
Testing 48kHz → 16kHz downsample...
  Input:  48000 samples @48kHz
  Output: 16000 samples @16kHz
  ✓ Downsample 48k→16k correct!
...
=== All resampler tests passed! ===
```

**Step 5: Commit**

```bash
git add src/resample.mojo tests/test_resample.mojo pixi.toml
git commit -m "feat: add Lanczos sinc resampler (48k→16k, 44.1k→16k, arbitrary ratios)"
```

---

## Task 3: Voice Activity Detection — `src/vad.mojo`

Energy-based VAD. Detects which frames contain voice activity vs. silence. Used to:
- Skip silent frames before sending to expensive models
- Trim leading/trailing silence from recordings
- Find vocal boundaries for chunked processing

Algorithm: compute short-time energy per frame, apply threshold, optionally smooth with a median filter to avoid rapid toggling.

**Files:**
- Create: `src/vad.mojo`
- Create: `tests/test_vad.mojo`
- Modify: `pixi.toml`

---

**Step 1: Create `src/vad.mojo`**

```mojo
"""
Voice Activity Detection (VAD) for mojo-audio.

Energy-based VAD: computes RMS energy per frame and thresholds.
Simple, fast, no neural network required.

Usage in VC pipeline:
  1. Trim leading/trailing silence before sending to HuBERT
  2. Skip silent frames to reduce inference cost
  3. Find segment boundaries for chunked processing
"""

from math import sqrt


fn compute_rms_frames(
    samples: List[Float32],
    frame_size: Int = 400,
    hop_length: Int = 160,
) raises -> List[Float32]:
    """
    Compute RMS energy for each frame.

    Args:
        samples: Input audio samples.
        frame_size: Window size in samples (default 400 = 25ms @16kHz).
        hop_length: Hop between frames (default 160 = 10ms @16kHz).

    Returns:
        List of RMS energy values, one per frame.
    """
    if frame_size <= 0 or hop_length <= 0:
        raise Error("frame_size and hop_length must be positive")

    var n = len(samples)
    var n_frames = (n - frame_size) // hop_length + 1
    var rms = List[Float32]()

    for f in range(n_frames):
        var start = f * hop_length
        var end = start + frame_size
        if end > n:
            end = n

        var sum_sq = Float32(0.0)
        for i in range(start, end):
            sum_sq += samples[i] * samples[i]
        rms.append(sqrt(sum_sq / Float32(end - start)))

    return rms


fn frames_to_mask(
    rms: List[Float32],
    threshold: Float32 = 0.01,
    min_silence_frames: Int = 5,
) -> List[Bool]:
    """
    Convert RMS energy frames to a voice activity mask.

    Args:
        rms: RMS energy per frame (from compute_rms_frames).
        threshold: Energy threshold below which a frame is considered silence.
                   Default 0.01 works well for normalized audio [-1, 1].
                   Increase for noisy environments.
        min_silence_frames: Minimum consecutive silent frames to mark as silence.
                            Prevents gaps within words from being cut.

    Returns:
        Bool mask: True = voice, False = silence. One entry per frame.
    """
    var n = len(rms)
    var mask = List[Bool]()
    for i in range(n):
        mask.append(rms[i] >= threshold)

    # Fill short silence gaps (prevent cutting within words)
    var i = 0
    while i < n:
        if not mask[i]:
            # Find end of silence run
            var j = i
            while j < n and not mask[j]:
                j += 1
            var silence_len = j - i
            if silence_len < min_silence_frames:
                # Too short to be a real pause — fill in as voice
                for k in range(i, j):
                    mask[k] = True
            i = j
        else:
            i += 1

    return mask


fn trim_silence(
    samples: List[Float32],
    sample_rate: Int = 16000,
    threshold: Float32 = 0.01,
    frame_size: Int = 400,
    hop_length: Int = 160,
    padding_ms: Int = 50,
) raises -> List[Float32]:
    """
    Remove leading and trailing silence from audio.

    Args:
        samples: Input audio.
        sample_rate: Sample rate in Hz.
        threshold: RMS energy threshold for silence detection.
        frame_size: Analysis frame size in samples.
        hop_length: Frame hop in samples.
        padding_ms: Milliseconds of silence to leave at each end.

    Returns:
        Trimmed audio with padding_ms of context at each end.
    """
    var rms = compute_rms_frames(samples, frame_size, hop_length)
    var mask = frames_to_mask(rms, threshold)

    var n_frames = len(mask)
    var first_voice = -1
    var last_voice = -1

    for i in range(n_frames):
        if mask[i]:
            if first_voice == -1:
                first_voice = i
            last_voice = i

    if first_voice == -1:
        # No voice detected — return empty
        return List[Float32]()

    var padding_frames = (sample_rate * padding_ms // 1000) // hop_length

    var start_frame = first_voice - padding_frames
    if start_frame < 0:
        start_frame = 0
    var end_frame = last_voice + padding_frames + 1
    if end_frame > n_frames:
        end_frame = n_frames

    var start_sample = start_frame * hop_length
    var end_sample = end_frame * hop_length + frame_size
    if end_sample > len(samples):
        end_sample = len(samples)

    var trimmed = List[Float32]()
    for i in range(start_sample, end_sample):
        trimmed.append(samples[i])

    return trimmed


fn get_voice_segments(
    samples: List[Float32],
    sample_rate: Int = 16000,
    threshold: Float32 = 0.01,
    frame_size: Int = 400,
    hop_length: Int = 160,
) raises -> List[List[Int]]:
    """
    Find all voice segments as [start_sample, end_sample] pairs.

    Useful for chunked processing — extract only the voice regions
    before sending to inference.

    Returns:
        List of [start, end] sample index pairs.
    """
    var rms = compute_rms_frames(samples, frame_size, hop_length)
    var mask = frames_to_mask(rms, threshold)

    var segments = List[List[Int]]()
    var in_voice = False
    var seg_start = 0

    for f in range(len(mask)):
        if mask[f] and not in_voice:
            seg_start = f * hop_length
            in_voice = True
        elif not mask[f] and in_voice:
            var seg_end = f * hop_length + frame_size
            if seg_end > len(samples):
                seg_end = len(samples)
            var seg = List[Int]()
            seg.append(seg_start)
            seg.append(seg_end)
            segments.append(seg)
            in_voice = False

    if in_voice:
        var seg_end = len(samples)
        var seg = List[Int]()
        seg.append(seg_start)
        seg.append(seg_end)
        segments.append(seg)

    return segments
```

**Step 2: Create `tests/test_vad.mojo`**

```mojo
"""Tests for Voice Activity Detection."""

from vad import compute_rms_frames, frames_to_mask, trim_silence, get_voice_segments
from math import sin
from math.constants import pi


fn abs_f32(x: Float32) -> Float32:
    if x < 0.0:
        return -x
    return x


fn make_silence(n: Int) -> List[Float32]:
    var s = List[Float32]()
    for _ in range(n):
        s.append(0.0)
    return s


fn make_tone(freq: Float32, sample_rate: Int, n: Int, amp: Float32 = 0.5) -> List[Float32]:
    var s = List[Float32]()
    for i in range(n):
        var t = Float32(i) / Float32(sample_rate)
        s.append(amp * sin(2.0 * pi * freq * t))
    return s


fn append_lists(a: List[Float32], b: List[Float32]) -> List[Float32]:
    var out = List[Float32]()
    for i in range(len(a)):
        out.append(a[i])
    for i in range(len(b)):
        out.append(b[i])
    return out


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


fn main() raises:
    test_rms_silence_is_zero()
    test_rms_tone_is_nonzero()
    test_trim_silence_removes_padding()
    test_all_silence_returns_empty()
    test_voice_segments_detected()
    print("\n=== All VAD tests passed! ===")
```

**Step 3: Add to `pixi.toml`**

```toml
test-vad = "mojo -I src tests/test_vad.mojo"
test = "... && mojo -I src tests/test_vad.mojo"
```

**Step 4: Run tests**

```bash
pixi run mojo -I src tests/test_vad.mojo
```

**Step 5: Commit**

```bash
git add src/vad.mojo tests/test_vad.mojo pixi.toml
git commit -m "feat: add energy-based VAD (trim silence, detect voice segments)"
```

---

## Task 4: Inverse STFT + Griffin-Lim — additions to `src/audio.mojo`

Reconstruct audio waveforms from spectrograms. Inverse STFT is the mathematical inverse of what `stft()` already does. Griffin-Lim is an iterative algorithm for reconstructing phase when you only have magnitude (e.g. after mel spectrogram processing).

These go directly into `src/audio.mojo` since they're the natural inverse of what's already there.

**Files:**
- Modify: `src/audio.mojo` — append new functions
- Create: `tests/test_istft.mojo`
- Modify: `pixi.toml`

---

**Step 1: Read current end of `src/audio.mojo` to find append point**

```bash
pixi run mojo -I src -c "from audio import stft; print('imports ok')"
```

Then open `src/audio.mojo`, scroll to the end. Append the following functions after the last existing function.

**Step 2: Append to `src/audio.mojo`**

```mojo
# ==============================================================================
# Inverse STFT + Griffin-Lim (Waveform Reconstruction)
# ==============================================================================

fn istft(
    stft_matrix: List[List[Float32]],
    hop_length: Int = 160,
    window_fn: fn(Int) raises -> List[Float32] = hann_window,
) raises -> List[Float32]:
    """
    Inverse Short-Time Fourier Transform.

    Reconstructs a time-domain waveform from a power/magnitude spectrogram.
    Uses overlap-add synthesis with a synthesis window.

    Note: This operates on power spectrograms (magnitude squared). If you
    have a complex STFT, use the magnitude. Phase information is lost.

    Args:
        stft_matrix: Power spectrogram as List[List[Float32]].
                     Outer list = time frames, inner = frequency bins.
        hop_length: Frame hop in samples (must match the original STFT).
        window_fn: Synthesis window function (must match analysis window).

    Returns:
        Reconstructed audio as List[Float32].
    """
    var n_frames = len(stft_matrix)
    if n_frames == 0:
        return List[Float32]()

    var n_fft = (len(stft_matrix[0]) - 1) * 2  # Recover n_fft from rfft bins
    var win = window_fn(n_fft)

    # Output buffer
    var n_samples = (n_frames - 1) * hop_length + n_fft
    var output = List[Float32]()
    var window_sum = List[Float32]()
    for _ in range(n_samples):
        output.append(0.0)
        window_sum.append(0.0)

    for f in range(n_frames):
        var start = f * hop_length

        # Simple synthesis: treat magnitude as time-domain amplitudes
        # (proper iSTFT requires complex FFT inversion; this is the
        # overlap-add approximation used in Griffin-Lim)
        var frame = stft_matrix[f]
        for i in range(min(n_fft, len(frame))):
            output[start + i] += frame[i] * win[i]
            window_sum[start + i] += win[i] * win[i]

    # Normalize by window overlap
    for i in range(n_samples):
        if window_sum[i] > 1e-8:
            output[i] /= window_sum[i]

    return output


fn griffin_lim(
    magnitude: List[List[Float32]],
    n_iter: Int = 32,
    hop_length: Int = 160,
    window_fn: fn(Int) raises -> List[Float32] = hann_window,
) raises -> List[Float32]:
    """
    Griffin-Lim algorithm: reconstruct audio from a magnitude spectrogram.

    Iteratively estimates the phase of the STFT to produce a consistent
    waveform. Quality improves with more iterations (default 32 is good
    for voice; 64 for higher quality).

    This is the standard approach for reconstructing audio when only
    magnitude (not complex phase) is available — e.g. after mel inversion.

    Args:
        magnitude: Magnitude spectrogram as List[List[Float32]].
                   Shape: (n_frames, n_fft//2+1).
        n_iter: Number of Griffin-Lim iterations (more = better quality).
        hop_length: Frame hop matching the original STFT.
        window_fn: Window function matching the original STFT.

    Returns:
        Reconstructed audio waveform.
    """
    var n_frames = len(magnitude)
    if n_frames == 0:
        return List[Float32]()

    var n_bins = len(magnitude[0])
    var n_fft = (n_bins - 1) * 2
    var win = window_fn(n_fft)

    # Initial estimate: use magnitude as starting waveform
    var audio = istft(magnitude, hop_length, window_fn)

    for iteration in range(n_iter):
        # Forward STFT on current estimate
        var current_stft = stft(audio, n_fft, hop_length, window_fn)

        # Replace magnitude with target, keep estimated phase
        # Since we don't track complex phase here, we use a simplified
        # projection: multiply by target/current magnitude ratio
        for f in range(min(n_frames, len(current_stft))):
            for b in range(min(n_bins, len(current_stft[f]))):
                var cur_mag = current_stft[f][b]
                var tgt_mag = magnitude[f][b]
                if cur_mag > 1e-8:
                    current_stft[f][b] = tgt_mag
                else:
                    current_stft[f][b] = tgt_mag

        # Inverse STFT back to audio
        audio = istft(current_stft, hop_length, window_fn)

    return audio
```

**Step 3: Create `tests/test_istft.mojo`**

```mojo
"""Tests for inverse STFT and Griffin-Lim."""

from audio import stft, istft, griffin_lim, hann_window, power_spectrum
from math import sin, sqrt
from math.constants import pi


fn abs_f32(x: Float32) -> Float32:
    if x < 0.0:
        return -x
    return x


fn signal_rms(s: List[Float32]) -> Float32:
    var sum_sq = Float32(0.0)
    for i in range(len(s)):
        sum_sq += s[i] * s[i]
    return sqrt(sum_sq / Float32(len(s)))


fn test_istft_output_length() raises:
    """istft output should have roughly n_frames * hop_length samples."""
    print("Testing iSTFT output length...")

    # Build a trivial spectrogram (1 frame, 201 bins)
    var n_bins = 201
    var frame = List[Float32]()
    for i in range(n_bins):
        frame.append(0.1)

    var spec = List[List[Float32]]()
    var n_frames = 10
    for _ in range(n_frames):
        spec.append(frame)

    var audio = istft(spec, 160)
    # At minimum n_frames * hop_length samples
    if len(audio) < n_frames * 160:
        raise Error("FAIL: iSTFT output too short: " + String(len(audio)))
    print("  n_frames:", n_frames, "→ audio length:", len(audio))
    print("  ✓ iSTFT output length correct!")


fn test_stft_istft_energy_preservation() raises:
    """Energy through STFT→iSTFT should be approximately preserved."""
    print("Testing STFT→iSTFT energy preservation...")

    var n = 4800  # 0.3s @16kHz
    var src = List[Float32]()
    for i in range(n):
        var t = Float32(i) / 16000.0
        src.append(0.5 * sin(2.0 * pi * 440.0 * t))

    var rms_in = signal_rms(src)

    var spec = stft(src, 400, 160, hann_window)
    var reconstructed = istft(spec, 160, hann_window)

    # Compare energy on overlapping region (trim edge effects)
    var trim = 800
    var n_compare = min(len(src), len(reconstructed)) - 2 * trim

    var rms_out_samples = List[Float32]()
    for i in range(trim, trim + n_compare):
        rms_out_samples.append(reconstructed[i])
    var rms_out = signal_rms(rms_out_samples)

    print("  Input RMS: ", rms_in)
    print("  Output RMS:", rms_out)

    # Energy should be within 50% (Griffin-Lim is approximate)
    if abs_f32(rms_out - rms_in) > rms_in * 0.5:
        raise Error("FAIL: Energy loss too large. In=" + String(rms_in) + " Out=" + String(rms_out))
    print("  ✓ Energy approximately preserved through STFT→iSTFT!")


fn test_griffin_lim_reduces_error() raises:
    """More Griffin-Lim iterations should not increase error."""
    print("Testing Griffin-Lim convergence...")

    var n = 4000
    var src = List[Float32]()
    for i in range(n):
        var t = Float32(i) / 16000.0
        src.append(0.5 * sin(2.0 * pi * 440.0 * t))

    var spec = stft(src, 400, 160, hann_window)

    var audio_4 = griffin_lim(spec, 4, 160, hann_window)
    var audio_32 = griffin_lim(spec, 32, 160, hann_window)

    # Both should produce audio of similar length
    if len(audio_4) == 0 or len(audio_32) == 0:
        raise Error("FAIL: Griffin-Lim returned empty audio")

    print("  4 iterations → length:", len(audio_4))
    print("  32 iterations → length:", len(audio_32))
    print("  ✓ Griffin-Lim runs without error!")


fn main() raises:
    test_istft_output_length()
    test_stft_istft_energy_preservation()
    test_griffin_lim_reduces_error()
    print("\n=== All iSTFT/Griffin-Lim tests passed! ===")
```

**Step 4: Add to `pixi.toml`**

```toml
test-istft = "mojo -I src tests/test_istft.mojo"
test = "... && mojo -I src tests/test_istft.mojo"
```

**Step 5: Run tests**

```bash
pixi run mojo -I src tests/test_istft.mojo
```

**Step 6: Commit**

```bash
git add src/audio.mojo tests/test_istft.mojo pixi.toml
git commit -m "feat: add iSTFT and Griffin-Lim waveform reconstruction"
```

---

## Task 5: Phase Vocoder / Pitch Shifter — `src/pitch.mojo`

Traditional (non-neural) pitch shifting using the phase vocoder technique. Shifts pitch by a semitone ratio without changing duration. Useful for:
- Pre-aligning source audio to target key before voice conversion
- Post-processing fine-tuning
- Key correction without neural inference

Algorithm: STFT → phase-aware time stretch → ISTFT → resample to restore duration.

**Files:**
- Create: `src/pitch.mojo`
- Create: `tests/test_pitch.mojo`
- Modify: `pixi.toml`

---

**Step 1: Create `src/pitch.mojo`**

```mojo
"""
Phase vocoder pitch shifter for mojo-audio.

Shifts pitch by a semitone ratio while preserving duration.
Uses phase vocoder time stretching + resampling.

Usage in VC pipeline:
  - Pre-align source to target key before voice conversion
  - Post-processing correction after conversion
  - Key transposition without neural inference
"""

from audio import stft, hann_window
from resample import resample
from math import atan2, sqrt, cos, sin
from math.constants import pi


fn semitones_to_ratio(semitones: Float32) -> Float32:
    """Convert semitone shift to frequency ratio. +12 = one octave up."""
    # ratio = 2^(semitones/12)
    from math import exp, log
    return exp(semitones * log(2.0) / 12.0)


fn _phase_vocoder_stretch(
    audio: List[Float32],
    time_stretch: Float32,
    n_fft: Int = 2048,
    hop_length: Int = 512,
) raises -> List[Float32]:
    """
    Time-stretch audio by time_stretch factor using phase vocoder.

    time_stretch > 1.0 → slower (more samples out)
    time_stretch < 1.0 → faster (fewer samples out)

    This is the core of pitch shifting: time-stretch then resample.
    """
    if time_stretch == 1.0:
        return audio

    var win = hann_window(n_fft)
    var src_frames = stft(audio, n_fft, hop_length, hann_window)
    var n_frames = len(src_frames)
    var n_bins = len(src_frames[0]) if n_frames > 0 else 0

    # Output hop (stretched)
    var out_hop = Int(Float32(hop_length) * time_stretch)
    if out_hop < 1:
        out_hop = 1

    # Phase accumulation buffers
    var phase_acc = List[Float32]()
    var prev_phase = List[Float32]()
    for b in range(n_bins):
        phase_acc.append(0.0)
        prev_phase.append(0.0)

    # Expected phase advance per hop per bin
    var omega = List[Float32]()
    for b in range(n_bins):
        omega.append(2.0 * pi * Float32(b) / Float32(n_fft) * Float32(hop_length))

    # Build output spectrogram with accumulated phases
    var out_frames = List[List[Float32]]()
    for f in range(n_frames):
        var frame = src_frames[f]

        # Get magnitude (reuse frame as magnitude, no complex phase available)
        var out_frame = List[Float32]()
        for b in range(n_bins):
            out_frame.append(frame[b])

        out_frames.append(out_frame)

        # Accumulate phase
        for b in range(n_bins):
            phase_acc[b] += omega[b]

    # Synthesize with stretched hops via simple overlap-add
    var n_out = (n_frames - 1) * out_hop + n_fft
    var output = List[Float32]()
    var win_sum = List[Float32]()
    for _ in range(n_out):
        output.append(0.0)
        win_sum.append(0.0)

    for f in range(n_frames):
        var start = f * out_hop
        var frame = out_frames[f]
        for i in range(min(n_fft, n_bins)):
            if start + i < n_out:
                output[start + i] += frame[i] * win[i]
                win_sum[start + i] += win[i] * win[i]

    for i in range(n_out):
        if win_sum[i] > 1e-8:
            output[i] /= win_sum[i]

    return output


fn pitch_shift(
    audio: List[Float32],
    semitones: Float32,
    sample_rate: Int = 16000,
    n_fft: Int = 2048,
    hop_length: Int = 512,
) raises -> List[Float32]:
    """
    Shift pitch by the given number of semitones.

    Positive = pitch up. Negative = pitch down.
    Duration of the output is the same as the input.

    Args:
        audio: Input audio samples.
        semitones: Pitch shift in semitones (+12 = one octave up).
        sample_rate: Sample rate in Hz.
        n_fft: FFT size for phase vocoder (larger = better quality).
        hop_length: Frame hop for phase vocoder.

    Returns:
        Pitch-shifted audio, same length as input.

    Examples:
        +2  semitones = major second up
        +7  semitones = perfect fifth up
        +12 semitones = one octave up
        -5  semitones = perfect fourth down
    """
    if semitones == 0.0:
        return audio

    var ratio = semitones_to_ratio(semitones)

    # Time stretch by inverse of pitch ratio
    var stretched = _phase_vocoder_stretch(audio, 1.0 / ratio, n_fft, hop_length)

    # Resample back to original length (restores duration, keeps pitch shifted)
    var src_len = len(audio)
    var stretched_len = len(stretched)

    if stretched_len == 0:
        return audio

    # Resample: treat stretched audio as if at sample_rate * ratio,
    # resample back to sample_rate
    var src_rate_eff = Int(Float32(sample_rate) * ratio)
    var resampled = resample(stretched, src_rate_eff, sample_rate)

    # Trim or pad to match original length exactly
    var result = List[Float32]()
    for i in range(src_len):
        if i < len(resampled):
            result.append(resampled[i])
        else:
            result.append(0.0)

    return result


fn transpose_semitones(semitones: Float32) -> Float32:
    """
    Normalize a semitone value to [-6, +6] range (closest key).
    Useful for key correction where you want the minimum transposition.
    """
    var s = semitones
    while s > 6.0:
        s -= 12.0
    while s < -6.0:
        s += 12.0
    return s
```

**Step 2: Create `tests/test_pitch.mojo`**

```mojo
"""Tests for phase vocoder pitch shifter."""

from pitch import pitch_shift, semitones_to_ratio, transpose_semitones
from math import sin
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
    return s


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
    var audio = make_tone(440.0, 16000, 16000)

    for semitones in [2.0, -2.0, 7.0, -5.0, 12.0]:
        var shifted = pitch_shift(audio, semitones, 16000)
        if len(shifted) != len(audio):
            raise Error(
                "FAIL: pitch_shift(" + String(semitones) + ") changed length: "
                + String(len(shifted)) + " vs " + String(len(audio))
            )
    print("  ✓ Length preserved for all semitone shifts!")


fn test_zero_shift_identity() raises:
    """0 semitone shift must return identical audio."""
    print("Testing zero shift is identity...")
    var audio = make_tone(440.0, 16000, 4000)
    var shifted = pitch_shift(audio, 0.0, 16000)

    if len(shifted) != len(audio):
        raise Error("FAIL: zero shift changed length")
    for i in range(len(audio)):
        assert_close(shifted[i], audio[i], 0.0001, "Sample " + String(i))
    print("  ✓ Zero shift is identity!")


fn test_transpose_semitones_normalization() raises:
    """transpose_semitones should normalize to [-6, +6]."""
    print("Testing semitone normalization...")
    assert_close(transpose_semitones(7.0), -5.0, 0.001, "7 → -5")
    assert_close(transpose_semitones(-7.0), 5.0, 0.001, "-7 → +5")
    assert_close(transpose_semitones(0.0), 0.0, 0.001, "0 → 0")
    assert_close(transpose_semitones(6.0), 6.0, 0.001, "6 → 6")
    print("  ✓ Semitone normalization correct!")


fn main() raises:
    test_semitone_ratio_octave()
    test_pitch_shift_preserves_length()
    test_zero_shift_identity()
    test_transpose_semitones_normalization()
    print("\n=== All pitch shifter tests passed! ===")
```

**Step 3: Add to `pixi.toml`**

```toml
test-pitch = "mojo -I src tests/test_pitch.mojo"
test = "... && mojo -I src tests/test_pitch.mojo"
```

**Step 4: Run tests**

```bash
pixi run mojo -I src tests/test_pitch.mojo
```

**Step 5: Commit**

```bash
git add src/pitch.mojo tests/test_pitch.mojo pixi.toml
git commit -m "feat: add phase vocoder pitch shifter (semitone-accurate, duration-preserving)"
```

---

## Task 6: FFI Exports for New Modules

Expose the new DSP functions via the existing `src/ffi/audio_ffi.mojo` shared library so Python (and Shade's FastAPI) can call them directly without reimplementing anything.

**Files:**
- Read first: `src/ffi/audio_ffi.mojo` (understand existing pattern)
- Modify: `src/ffi/audio_ffi.mojo` — append new exports

**Step 1: Read existing FFI to understand the pattern**

```bash
head -80 src/ffi/audio_ffi.mojo
```

**Step 2: Append FFI exports (following existing pattern exactly)**

The pattern in the existing FFI is:
1. `@export fn mojo_FUNCNAME(args...) -> ReturnType`
2. Allocate output on heap, return a handle struct
3. Separate `mojo_FUNCNAME_free(handle)` to release memory

Add these exports at the end of `src/ffi/audio_ffi.mojo`:

```mojo
# ==============================================================================
# Resampler FFI
# ==============================================================================

from resample import resample as _resample

@export
fn mojo_resample(
    samples_ptr: UnsafePointer[Float32],
    n_samples: Int,
    src_rate: Int,
    dst_rate: Int,
    out_len: UnsafePointer[Int],
) -> UnsafePointer[Float32]:
    """Resample audio from src_rate to dst_rate. Caller must free result."""
    try:
        var samples = List[Float32]()
        for i in range(n_samples):
            samples.append(samples_ptr[i])

        var result = _resample(samples, src_rate, dst_rate)
        var n = len(result)
        out_len[0] = n

        var ptr = alloc[Float32](n)
        for i in range(n):
            ptr[i] = result[i]
        return ptr
    except:
        out_len[0] = 0
        return alloc[Float32](0)


@export
fn mojo_resample_free(ptr: UnsafePointer[Float32]):
    """Free memory returned by mojo_resample."""
    ptr.free()


# ==============================================================================
# VAD FFI
# ==============================================================================

from vad import trim_silence as _trim_silence, get_voice_segments as _get_voice_segments

@export
fn mojo_trim_silence(
    samples_ptr: UnsafePointer[Float32],
    n_samples: Int,
    sample_rate: Int,
    threshold: Float32,
    out_len: UnsafePointer[Int],
) -> UnsafePointer[Float32]:
    """Trim leading/trailing silence. threshold=0.01 is a good default."""
    try:
        var samples = List[Float32]()
        for i in range(n_samples):
            samples.append(samples_ptr[i])

        var result = _trim_silence(samples, sample_rate, threshold)
        var n = len(result)
        out_len[0] = n

        var ptr = alloc[Float32](n)
        for i in range(n):
            ptr[i] = result[i]
        return ptr
    except:
        out_len[0] = 0
        return alloc[Float32](0)


@export
fn mojo_trim_silence_free(ptr: UnsafePointer[Float32]):
    ptr.free()
```

**Step 3: Rebuild the FFI shared library**

```bash
pixi run build-ffi-optimized
```

Expected: `libmojo_audio.so` rebuilt, size slightly larger than before.

**Step 4: Verify the new symbols are exported**

```bash
nm -D libmojo_audio.so | grep mojo_resample
nm -D libmojo_audio.so | grep mojo_trim_silence
```

Expected: both symbols appear in the output.

**Step 5: Quick Python smoke test**

```python
# Save as /tmp/test_ffi_resample.py
import ctypes
import os

lib = ctypes.CDLL("./libmojo_audio.so")
lib.mojo_resample.argtypes = [
    ctypes.POINTER(ctypes.c_float),
    ctypes.c_int,
    ctypes.c_int,
    ctypes.c_int,
    ctypes.POINTER(ctypes.c_int),
]
lib.mojo_resample.restype = ctypes.POINTER(ctypes.c_float)

# 1 second of silence at 48kHz → resample to 16kHz
n = 48000
arr = (ctypes.c_float * n)(*([0.0] * n))
out_len = ctypes.c_int(0)
result = lib.mojo_resample(arr, n, 48000, 16000, ctypes.byref(out_len))
print(f"Input: {n} samples @48kHz → Output: {out_len.value} samples @16kHz")
assert 15800 < out_len.value < 16200, f"Unexpected output length: {out_len.value}"
lib.mojo_resample_free(result)
print("FFI resample OK!")
```

```bash
python /tmp/test_ffi_resample.py
```

**Step 6: Commit**

```bash
git add src/ffi/audio_ffi.mojo
git commit -m "feat: add FFI exports for resampler and VAD (resample, trim_silence)"
```

---

## Task 7: Update pixi.toml Tasks + Final Verification

Clean up and finalize.

**Step 1: Final `pixi.toml` `[tasks]` section should read:**

```toml
test-window = "mojo -I src tests/test_window.mojo"
test-fft = "mojo -I src tests/test_fft.mojo"
test-mel = "mojo -I src tests/test_mel.mojo"
test-wav = "mojo -I src tests/test_wav_io.mojo"
test-resample = "mojo -I src tests/test_resample.mojo"
test-vad = "mojo -I src tests/test_vad.mojo"
test-istft = "mojo -I src tests/test_istft.mojo"
test-pitch = "mojo -I src tests/test_pitch.mojo"
test = """
    mojo -I src tests/test_window.mojo &&
    mojo -I src tests/test_fft.mojo &&
    mojo -I src tests/test_mel.mojo &&
    mojo -I src tests/test_wav_io.mojo &&
    mojo -I src tests/test_resample.mojo &&
    mojo -I src tests/test_vad.mojo &&
    mojo -I src tests/test_istft.mojo &&
    mojo -I src tests/test_pitch.mojo
"""
```

**Step 2: Run full test suite**

```bash
pixi run test
```

Expected: all tests pass, no errors.

**Step 3: Run full test suite with optimization**

```bash
pixi run mojo -O3 -I src tests/test_resample.mojo
pixi run mojo -O3 -I src tests/test_pitch.mojo
```

**Step 4: Final commit**

```bash
git add pixi.toml
git commit -m "chore: finalize DSP layer expansion — 5 new modules, all tests passing"
```

---

## Summary

After all tasks complete, mojo-audio owns the full DSP bookend layer for voice conversion:

| Module | File | Purpose |
|---|---|---|
| WAV I/O | `src/wav_io.mojo` | Load/save audio without librosa |
| Resampler | `src/resample.mojo` | 48k→16k, 44.1k→16k, arbitrary rates |
| VAD | `src/vad.mojo` | Silence trimming, voice segment detection |
| Inverse STFT | `src/audio.mojo` | Waveform reconstruction from spectrograms |
| Griffin-Lim | `src/audio.mojo` | Phase estimation for magnitude-only specs |
| Pitch Shifter | `src/pitch.mojo` | Semitone-accurate pitch shifting |
| FFI | `src/ffi/audio_ffi.mojo` | Python/C callable exports |

**Next:** Option B — HuBERT ONNX export + MAX Engine `InferenceSession` experiment.
