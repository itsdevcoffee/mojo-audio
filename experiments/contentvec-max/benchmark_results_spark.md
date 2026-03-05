# AudioEncoder Benchmark Results

**Model:** `facebook/hubert-base-ls960`  
**Date:** 2026-03-05  
**Platform:** `aarch64`  Python 3.13.12  
**GPU:** NVIDIA GB10, 12.1  
**Input:** 16000 samples (1s @16kHz), batch=1  
**Iterations:** 20 (after 3 warmup)  

| Backend | Mean (ms) | Std (ms) | P95 (ms) | vs PyTorch CPU |
|---------|-----------|----------|----------|----------------|
| PyTorch CPU | 179.2 | 14.9 | 202.1 | 1.00x |
| PyTorch GPU | N/A | — | — | — |
| MAX Engine CPU | 100.1 | 8.1 | 108.8 | 1.79x |
| MAX Engine GPU | 197.1 | 0.6 | 197.9 | 0.91x |

## Notes

- MAX Engine CPU includes numpy bridge for pos_conv (MAX conv2d groups bug workaround)
- MAX Engine GPU uses GPU for CNN and transformer blocks; pos_conv runs on CPU
