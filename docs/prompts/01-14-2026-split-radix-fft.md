# Split-Radix FFT Implementation

## Context

We're building a high-performance audio processing library in Mojo for Whisper speech recognition. The FFT is the primary bottleneck in our mel spectrogram pipeline, taking 70-77% of total processing time.

**Current state:**
- Working radix-2 and radix-4 FFT implementations exist in `src/audio.mojo`
- Radix-4 only works for N = power of 4 (256, 1024, 4096...)
- For Whisper's n_fft=400, we pad to 512 which is NOT a power of 4
- A buggy split-radix attempt exists (marked "REMOVED") - feel free to delete or rewrite it

**Why split-radix:**
- Works on ANY power-of-2 size (including 512)
- ~20% fewer multiplications than radix-2
- Combines radix-2 and radix-4 butterflies optimally

## Goal

Implement a correct, SIMD-optimized split-radix FFT that passes all tests and improves performance for non-power-of-4 sizes like 512.

## Conditions of Done

1. **Correctness**: Split-radix output matches original `fft()` function within 1e-4 tolerance for all power-of-2 sizes from 8 to 2048

2. **Tests pass**: Add tests to `tests/test_fft.mojo` that verify:
   - Split-radix vs original FFT correctness
   - Multiple sizes (8, 16, 32, 64, 128, 256, 512, 1024)
   - SIMD version matches scalar version

3. **Integration**: Update `rfft_simd()` to use split-radix for non-power-of-4 sizes (the function already has a branch for this)

4. **Performance**: For N=512, split-radix should be faster than current radix-2 fallback

## Constraints

- Follow existing code patterns in `src/audio.mojo`
- Use 64-byte aligned memory allocation (see `Radix4TwiddleCache` for example)
- Mojo version in `pixi.toml` - verify syntax against current docs
- Run tests with: `pixi run mojo -I src tests/test_fft.mojo`
- Run benchmarks with: `pixi run bench`

## Reference

The existing working implementations to study:
- `fft_radix2_simd()` - radix-2 SIMD implementation
- `fft_radix4_cached_simd()` - radix-4 with twiddle cache
- `Radix4TwiddleCache` - struct pattern for precomputed twiddles

Compare against librosa with: `pixi run python benchmarks/compare_librosa.py`
