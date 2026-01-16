# Long Audio Optimization - Beating librosa at 30s+

## Problem Statement

mojo-audio loses to librosa on longer audio (30s) for most FFT sizes:

| n_fft | 30s Mojo | 30s librosa | Gap |
|-------|----------|-------------|-----|
| 256 | 15.1ms | 19.1ms | Mojo 1.26x faster ✓ |
| 400 | 27.3ms | 19.4ms | librosa 1.41x faster ✗ |
| 512 | 33.4ms | 20.1ms | librosa 1.66x faster ✗ |
| 1024 | 60.0ms | 40.1ms | librosa 1.50x faster ✗ |

For short audio (1-10s), Mojo wins decisively. The issue is **O(n) overhead that scales with frame count**.

---

## Root Cause

Profiling shows time distribution for 30s audio:
- RFFT: 70-77% (already optimized)
- Mel filterbank: 7-12% (frame-by-frame - PROBLEM)
- Frame extraction: 5-6% (per-frame allocation)
- Window application: 4-5% (sequential)

**librosa's advantage**: Batch operations via NumPy/MKL that process all frames at once.

---

## Optimization 1: Batch Mel Filterbank (Highest Priority)

### Current Implementation
```mojo
# In mel_spectrogram() - processes ONE frame at a time
for frame_idx in range(num_frames):
    var power = power_spectrum_simd(fft_result, n_fft)
    var mel_frame = apply_mel_filterbank(power, mel_filterbank)
    # ... store mel_frame
```

### Target Implementation
```mojo
# Process ALL frames, then apply mel filterbank once
var all_power_spectra = List[List[Float32]]()  # [num_frames x freq_bins]

# Step 1: Compute all power spectra
for frame_idx in range(num_frames):
    var power = power_spectrum_simd(fft_result, n_fft)
    all_power_spectra.append(power)

# Step 2: Batch mel filterbank (matrix multiply)
var mel_output = batch_mel_filterbank(all_power_spectra, mel_filterbank)
```

### New Function Needed
```mojo
fn batch_mel_filterbank(
    power_spectra: List[List[Float32]],  # [num_frames x freq_bins]
    filterbank: MelFilterbank
) -> List[List[Float32]]:  # [n_mels x num_frames]
    """
    Apply mel filterbank to all frames at once.

    This is a matrix multiplication:
    output[mel, frame] = sum(filterbank[mel, freq] * power[frame, freq])

    Use SIMD to vectorize across frequencies.
    """
```

### Expected Gain
- Mel portion: 2-3x faster
- Overall: 10-15% improvement

---

## Optimization 2: Pre-allocated Output Buffers

### Current Issue
Memory allocated inside hot loop for each frame.

### Target
```mojo
fn stft_optimized(...) raises -> List[List[Float32]]:
    # Pre-allocate ALL memory before loop
    var output = List[List[Float32]](capacity=num_frames)
    var frame_buffer = List[Float32](capacity=n_fft)
    var windowed_buffer = List[Float32](capacity=n_fft)
    var fft_cache = TwiddleFactorsSoA(fft_size)

    for frame_idx in range(num_frames):
        # Reuse buffers instead of allocating
        extract_frame_into(audio, frame_idx, hop_length, frame_buffer)
        apply_window_into(frame_buffer, window, windowed_buffer)
        # ...
```

### New Functions Needed
- `extract_frame_into()` - extract frame into existing buffer
- `apply_window_into()` - apply window into existing buffer

### Expected Gain
- Reduces GC pressure
- 5-10% improvement on long audio

---

## Optimization 3: Parallel Frame Processing

### Current Issue
Frames processed sequentially, but they're independent.

### Target
```mojo
fn stft_parallel(...) raises -> List[List[Float32]]:
    var num_frames = calculate_num_frames(len(audio), n_fft, hop_length)
    var output = List[List[Float32]](capacity=num_frames)

    # Pre-size output
    for _ in range(num_frames):
        output.append(List[Float32]())

    # Process frames in parallel
    @parameter
    fn process_frame(frame_idx: Int):
        var frame = extract_frame(audio, frame_idx, hop_length, n_fft)
        var windowed = apply_window_simd(frame, window)
        var fft_result = rfft_simd(windowed, twiddles)
        var power = power_spectrum_simd(fft_result, n_fft)
        output[frame_idx] = power

    parallelize[process_frame](num_frames, num_physical_cores())

    # Apply mel filterbank (batch)
    return batch_mel_filterbank(output, mel_filterbank)
```

### Considerations
- Need thread-safe output storage
- May need per-thread FFT caches to avoid contention
- Only beneficial for large frame counts (>100 frames)

### Expected Gain
- 2-4x on multi-core for long audio
- Threshold: only parallelize if num_frames > 100

---

## Optimization 4: SIMD Mel Filterbank

### Current Implementation
Scalar loop over mel bands and frequencies.

### Target
```mojo
fn apply_mel_filterbank_simd(
    power: List[Float32],
    filterbank: MelFilterbank
) -> List[Float32]:
    var n_mels = filterbank.n_mels
    var n_freqs = len(power)
    var output = List[Float32](capacity=n_mels)

    alias simd_width = 8
    var power_ptr = power.unsafe_ptr()

    for mel in range(n_mels):
        var sum = SIMD[DType.float32, simd_width](0)
        var filter_ptr = filterbank.filters[mel].unsafe_ptr()

        var f = 0
        while f + simd_width <= n_freqs:
            var p = power_ptr.load[width=simd_width](f)
            var w = filter_ptr.load[width=simd_width](f)
            sum += p * w
            f += simd_width

        # Horizontal sum + scalar remainder
        var total = sum.reduce_add()
        while f < n_freqs:
            total += power[f] * filterbank.filters[mel][f]
            f += 1

        output.append(total)

    return output
```

### Expected Gain
- Mel application: 2-4x faster
- Overall: 5-8% improvement

---

## Implementation Order

1. **Batch Mel Filterbank** - Highest impact, moderate complexity
2. **SIMD Mel Filterbank** - Can combine with #1
3. **Pre-allocated Buffers** - Low complexity, good gains
4. **Parallel Frame Processing** - Highest complexity, best for very long audio

---

## Files to Modify

- `src/audio.mojo`:
  - `mel_spectrogram()` - restructure for batch processing
  - `stft()` - add pre-allocation, optional parallelization
  - Add `batch_mel_filterbank()` function
  - Add `apply_mel_filterbank_simd()` function

---

## Acceptance Criteria

1. **Correctness**: Output matches current implementation within 1e-4 tolerance
2. **Performance targets**:

| Duration | n_fft=400 Target | Current |
|----------|------------------|---------|
| 30s | < 20ms | 27.3ms |
| 10s | < 10ms | 11.8ms |
| 1s | < 1.5ms | 1.6ms |

3. **Tests pass**: `pixi run mojo -I src tests/test_fft.mojo`
4. **No regression**: Short audio (1s, 10s) should not get slower

---

## Validation Commands

```bash
# Run benchmarks
python ui/backend/run_benchmark.py mojo 30 10 400 160 80
python ui/backend/run_benchmark.py librosa 30 10 400 160 80

# Compare
pixi run bench
pixi run python benchmarks/compare_librosa.py

# Test correctness
pixi run mojo -I src tests/test_fft.mojo
```

---

## References

- Current mel_spectrogram: `src/audio.mojo:3173`
- Current stft: `src/audio.mojo` (search for `fn stft`)
- MelFilterbank struct: `src/audio.mojo` (search for `struct MelFilterbank`)
- Profiler: `benchmarks/profile_stft.mojo`
