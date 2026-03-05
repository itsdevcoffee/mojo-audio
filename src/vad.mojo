"""
Voice Activity Detection (VAD) for mojo-audio.

Energy-based VAD: computes RMS energy per frame and thresholds.
No neural network required — fast, zero-dependency.

Usage in voice conversion pipeline:
  - Trim leading/trailing silence before sending to HuBERT
  - Skip silent frames to reduce inference cost
  - Find segment boundaries for chunked processing
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

    Raises:
        Error if frame_size or hop_length <= 0.
    """
    if frame_size <= 0:
        raise Error("frame_size must be positive, got " + String(frame_size))
    if hop_length <= 0:
        raise Error("hop_length must be positive, got " + String(hop_length))

    var n = len(samples)
    var rms = List[Float32]()

    if n < frame_size:
        return rms^

    var n_frames = (n - frame_size) // hop_length + 1

    for f in range(n_frames):
        var start = f * hop_length
        var end = start + frame_size

        var sum_sq = Float32(0.0)
        for i in range(start, end):
            sum_sq += samples[i] * samples[i]
        rms.append(sqrt(sum_sq / Float32(frame_size)))

    return rms^


fn frames_to_mask(
    rms: List[Float32],
    threshold: Float32 = 0.01,
    min_silence_frames: Int = 5,
) -> List[Bool]:
    """
    Convert RMS energy frames to a voice activity boolean mask.

    Args:
        rms: RMS energy per frame.
        threshold: Energy threshold; frames below this are silence.
                   Default 0.01 works well for normalized audio [-1, 1].
        min_silence_frames: Minimum consecutive silent frames counted as silence.
                            Shorter silence gaps are filled in as voice.

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
            var j = i
            while j < n and not mask[j]:
                j += 1
            var silence_len = j - i
            if silence_len < min_silence_frames:
                for k in range(i, j):
                    mask[k] = True
            i = j
        else:
            i += 1

    return mask^


fn trim_silence(
    samples: List[Float32],
    sample_rate: Int = 16000,
    threshold: Float32 = 0.01,
    frame_size: Int = 400,
    hop_length: Int = 160,
    padding_ms: Int = 50,
) raises -> List[Float32]:
    """
    Remove leading and trailing silence, keeping a short padding context.

    Args:
        samples: Input audio.
        sample_rate: Sample rate in Hz.
        threshold: RMS energy threshold for silence detection.
        frame_size: Analysis frame size in samples.
        hop_length: Frame hop in samples.
        padding_ms: Milliseconds of context to leave at each end.

    Returns:
        Trimmed audio. Empty list if no voice detected.
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

    return trimmed^


fn get_voice_segments(
    samples: List[Float32],
    sample_rate: Int = 16000,
    threshold: Float32 = 0.01,
    frame_size: Int = 400,
    hop_length: Int = 160,
) raises -> List[List[Int]]:
    """
    Find all voice segments as [start_sample, end_sample] pairs.

    Useful for chunked inference — extract only voice regions
    before sending to HuBERT or VITS.

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
            segments.append(seg^)
            in_voice = False

    if in_voice:
        var seg = List[Int]()
        seg.append(seg_start)
        seg.append(len(samples))
        segments.append(seg^)

    return segments^
