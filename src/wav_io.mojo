"""
WAV file I/O for mojo-audio.

Uses Python stdlib `wave` module for container parsing.
All audio data is converted to/from List[Float32] normalized [-1.0, 1.0].
"""

from python import Python


fn read_wav(path: String) raises -> Tuple[List[Float32], Int]:
    """
    Read a WAV file and return normalized Float32 samples + sample rate.

    Supports: 16-bit PCM, 32-bit PCM.
    Multi-channel audio is mixed down to mono by averaging channels.

    Args:
        path: Path to the .wav file.

    Returns:
        Tuple of (samples: List[Float32], sample_rate: Int).
        Samples are normalized to [-1.0, 1.0].
    """
    var wave_mod = Python.import_module("wave")

    var wf = wave_mod.open(path, "rb")
    try:
        var n_channels = Int(wf.getnchannels())
        var sample_width = Int(wf.getsampwidth())  # bytes per sample
        var sample_rate = Int(wf.getframerate())
        var n_frames = Int(wf.getnframes())

        var raw = wf.readframes(n_frames)

        var samples = List[Float32]()

        if sample_width == 2:
            # 16-bit PCM
            var scale = Float32(1.0 / 32768.0)
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
            # 32-bit PCM (int)
            # Use Float64 intermediate to avoid precision loss: Float32 has only 24 mantissa bits,
            # which cannot represent all 32-bit integer values exactly.
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
                    channel_sum += Float32(Float64(raw_val) * Float64(1.0 / 2147483648.0))
                samples.append(channel_sum / Float32(n_channels))
                frame_idx += 1
        else:
            raise Error("Unsupported sample width: " + String(sample_width) + " bytes")

        wf.close()
        return (samples^, sample_rate)
    except e:
        wf.close()
        raise e^


fn write_wav(path: String, samples: List[Float32], sample_rate: Int) raises:
    """
    Write Float32 samples to a 16-bit PCM WAV file.

    Args:
        path: Output file path.
        samples: Audio samples normalized to [-1.0, 1.0].
        sample_rate: Sample rate in Hz.
    """
    var wave_mod = Python.import_module("wave")
    var array_mod = Python.import_module("array")

    var wf = wave_mod.open(path, "wb")
    try:
        wf.setnchannels(1)
        wf.setsampwidth(2)  # 16-bit
        wf.setframerate(sample_rate)

        var n = len(samples)

        # Build Python list of int16 values. Each append is a Python FFI call (N total),
        # which is unavoidable without a more complex Mojo->Python buffer approach.
        # The key improvement is using array.array("h", ...).tobytes() which converts
        # to bytes in a single Python call rather than N struct.pack calls.
        var py_ints = Python.evaluate("[]")
        for i in range(n):
            var val = samples[i]
            if val > 1.0:
                val = 1.0
            if val < -1.0:
                val = -1.0
            _ = py_ints.append(Int(val * 32767.0))

        var arr = array_mod.array("h", py_ints)
        wf.writeframes(arr.tobytes())
        wf.close()
    except e:
        wf.close()
        raise e^
