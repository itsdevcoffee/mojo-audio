"""
Profile STFT/Mel Spectrogram Pipeline.

Measures time spent in each stage to identify optimization opportunities.
"""

from audio import (
    hann_window, apply_window_simd, rfft_simd, power_spectrum_simd,
    TwiddleFactorsSoA, next_power_of_2,
    create_mel_filterbank, apply_mel_filterbank
)
from time import perf_counter_ns
from math import log


fn profile_stft_stages(audio_seconds: Int) raises:
    """Profile each stage of the mel spectrogram pipeline."""
    print("\nProfiling " + String(audio_seconds) + "s audio:")
    print("-" * 50)

    var sample_rate = 16000
    var n_fft = 400
    var hop_length = 160
    var n_mels = 80
    var fft_size = next_power_of_2(n_fft)

    # Create test audio
    var audio = List[Float32]()
    for i in range(audio_seconds * sample_rate):
        audio.append(Float32(i % 100) / 100.0)

    var num_frames = (len(audio) - n_fft) // hop_length + 1
    var needed_bins = n_fft // 2 + 1

    # Stage 1: Window creation (one-time)
    var t0 = perf_counter_ns()
    var window = hann_window(n_fft)
    var t1 = perf_counter_ns()
    var window_time = Float64(t1 - t0) / 1_000_000.0

    # Stage 2: Twiddle precomputation (one-time)
    t0 = perf_counter_ns()
    var twiddles = TwiddleFactorsSoA(fft_size)
    t1 = perf_counter_ns()
    var twiddle_time = Float64(t1 - t0) / 1_000_000.0

    # Stage 3: Mel filterbank creation (one-time)
    t0 = perf_counter_ns()
    var filterbank = create_mel_filterbank(n_mels, n_fft, sample_rate)
    t1 = perf_counter_ns()
    var filterbank_time = Float64(t1 - t0) / 1_000_000.0

    # Measure per-frame processing
    var frame_extract_total: Int = 0
    var window_apply_total: Int = 0
    var rfft_total: Int = 0
    var power_total: Int = 0

    # Pre-allocate spectrogram
    var spectrogram = List[List[Float32]]()
    for _ in range(num_frames):
        var row = List[Float32]()
        for _ in range(needed_bins):
            row.append(0.0)
        spectrogram.append(row^)

    # Process all frames (sequential for accurate timing)
    for frame_idx in range(num_frames):
        var start = frame_idx * hop_length

        # Frame extraction
        t0 = perf_counter_ns()
        var frame = List[Float32]()
        for i in range(n_fft):
            if start + i < len(audio):
                frame.append(audio[start + i])
            else:
                frame.append(0.0)
        t1 = perf_counter_ns()
        frame_extract_total += Int(t1 - t0)

        # Window application
        t0 = perf_counter_ns()
        var windowed = apply_window_simd(frame, window)
        t1 = perf_counter_ns()
        window_apply_total += Int(t1 - t0)

        # RFFT
        t0 = perf_counter_ns()
        var fft_result = rfft_simd(windowed, twiddles)
        t1 = perf_counter_ns()
        rfft_total += Int(t1 - t0)

        # Power spectrum
        t0 = perf_counter_ns()
        var power = power_spectrum_simd(fft_result, Float32(n_fft))
        t1 = perf_counter_ns()
        power_total += Int(t1 - t0)

        # Store result
        for i in range(needed_bins):
            if i < len(power):
                spectrogram[frame_idx][i] = power[i]

    # Stage 5: Apply mel filterbank
    t0 = perf_counter_ns()
    var mel_spec = apply_mel_filterbank(spectrogram, filterbank)
    t1 = perf_counter_ns()
    var mel_apply_time = Float64(t1 - t0) / 1_000_000.0

    # Stage 6: Log scaling
    t0 = perf_counter_ns()
    for i in range(len(mel_spec)):
        for j in range(len(mel_spec[i])):
            mel_spec[i][j] = log(mel_spec[i][j] + 1e-10)
    t1 = perf_counter_ns()
    var log_time = Float64(t1 - t0) / 1_000_000.0

    # Convert totals to ms
    var frame_extract_ms = Float64(frame_extract_total) / 1_000_000.0
    var window_apply_ms = Float64(window_apply_total) / 1_000_000.0
    var rfft_ms = Float64(rfft_total) / 1_000_000.0
    var power_ms = Float64(power_total) / 1_000_000.0

    var total_ms = window_time + twiddle_time + filterbank_time +
                   frame_extract_ms + window_apply_ms + rfft_ms +
                   power_ms + mel_apply_time + log_time

    # Print results
    print("  Frames processed: " + String(num_frames))
    print("")
    print("  One-time setup:")
    print("    Window creation:    " + String(window_time)[:8] + " ms")
    print("    Twiddle precompute: " + String(twiddle_time)[:8] + " ms")
    print("    Filterbank create:  " + String(filterbank_time)[:8] + " ms")
    print("")
    print("  Per-frame processing (total across all frames):")
    print("    Frame extraction:   " + String(frame_extract_ms)[:8] + " ms (" +
          String(frame_extract_ms / total_ms * 100.0)[:5] + "%)")
    print("    Window application: " + String(window_apply_ms)[:8] + " ms (" +
          String(window_apply_ms / total_ms * 100.0)[:5] + "%)")
    print("    RFFT:               " + String(rfft_ms)[:8] + " ms (" +
          String(rfft_ms / total_ms * 100.0)[:5] + "%)")
    print("    Power spectrum:     " + String(power_ms)[:8] + " ms (" +
          String(power_ms / total_ms * 100.0)[:5] + "%)")
    print("")
    print("  Post-processing:")
    print("    Mel filterbank:     " + String(mel_apply_time)[:8] + " ms (" +
          String(mel_apply_time / total_ms * 100.0)[:5] + "%)")
    print("    Log scaling:        " + String(log_time)[:8] + " ms (" +
          String(log_time / total_ms * 100.0)[:5] + "%)")
    print("")
    print("  TOTAL:                " + String(total_ms)[:8] + " ms")
    print("  Throughput:           " + String(Float64(audio_seconds) / (total_ms / 1000.0))[:6] + "x realtime")


fn main() raises:
    print("\n" + "=" * 60)
    print("STFT/Mel Spectrogram Pipeline Profiler")
    print("=" * 60)
    print("\nBreaking down time spent in each stage...")

    profile_stft_stages(1)
    profile_stft_stages(10)
    profile_stft_stages(30)

    print("\n" + "=" * 60)
    print("Profile complete!")
    print("=" * 60 + "\n")
