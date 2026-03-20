# MAX Kernel Cache Findings

Investigation for Sprint 2, Item 2D.

## Summary

MAX Graph JIT compilation is fast enough in 26.3.0 that kernel caching is no longer a concern for cold-start latency.

## Results

### MAX 26.1.0 (old, Jan 2026 nightly)

- `from_pretrained` cold start: **~103-168s** (JIT compilation of 12-block transformer graph)
- Cache location: `~/.modular/.max_cache/mof/mef/<version>/`
- Cache stored compiled kernels as ~18MB files keyed by graph hash
- Second run with cache: **~4s**
- Cache was critical for usable cold-start times

### MAX 26.3.0 (current, Mar 2026 nightly)

- `from_pretrained` cold start: **~4-5s** consistently
- No cache directory created (`~/.modular/.max_cache/` not populated)
- Compilation speed improved ~25-30x vs 26.1.0
- Three sequential fresh-process runs: 4.19s, 4.84s, 4.48s
- Cache is no longer needed — compilation itself is fast enough

## Conclusion

No action needed. The MAX 26.3 compiler is fast enough that the 168s cold-start problem from Sprint 1 is resolved by the version upgrade alone. No cache configuration, environment variables, or workarounds required.

## Benchmark Environment

- Local: x86_64, RTX 4060 Ti, MAX 26.3.0.dev2026032005
- Model: facebook/hubert-base-ls960 (2 graphs: CNN+projection, pos_conv+transformer)
- Script: `experiments/max-cache-check/check_kernel_cache.py`
