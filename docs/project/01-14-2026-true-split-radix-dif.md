# Task: True Split-Radix DIF Implementation

**Status:** Backlog
**Priority:** Low (current hybrid approach beats librosa)
**Estimated gain:** ~10-15% additional FFT speedup

---

## Context

Current split-radix implementation is a **hybrid**:
- Stages 1-2 (m=2, m=4): Proper split-radix butterflies
- Stages 3+ (m≥8): Falls back to radix-2 DIT butterflies

This works and beats librosa, but doesn't achieve the full theoretical benefit of split-radix (~33% fewer multiplications than radix-2).

**Why it's hybrid:** The original attempt mixed DIF (decimation-in-frequency) indexing patterns with DIT (decimation-in-time) bit-reversal, which are incompatible.

---

## What True Split-Radix Requires

1. **DIF formulation throughout** - Process from large butterflies down to small
2. **Bit-reversal at the END** - Not at the beginning like DIT
3. **L-shaped butterfly** - Combines radix-2 even outputs with radix-4-like odd outputs

**DIF Split-Radix decomposition:**
```
X[2k]   = DFT_{N/2}(x[n] + x[n+N/2])
X[4k+1] = DFT_{N/4}((x[n] - x[n+N/2]) - j(x[n+N/4] - x[n+3N/4])) * W_N^n
X[4k+3] = DFT_{N/4}((x[n] - x[n+N/2]) + j(x[n+N/4] - x[n+3N/4])) * W_N^{3n}
```

---

## Acceptance Criteria

- [ ] Passes all FFT correctness tests (tolerance 1e-4)
- [ ] Works for all power-of-2 sizes 8-4096
- [ ] N=512 benchmark shows improvement over current hybrid
- [ ] No regression for power-of-4 sizes (radix-4 path unaffected)

---

## Implementation Notes

- May want separate `fft_split_radix_dif()` function rather than modifying existing
- Twiddle cache structure may need adjustment for DIF access patterns
- SIMD version should process consecutive elements (DIF is naturally SIMD-friendly)
- Consider whether end bit-reversal can be fused with final output copy

---

## References

- Existing hybrid: `src/audio.mojo` → `fft_split_radix_simd()`
- Sorensen et al. (1987) - Original split-radix paper
- Duhamel & Hollmann (1984) - Split-radix complexity analysis
