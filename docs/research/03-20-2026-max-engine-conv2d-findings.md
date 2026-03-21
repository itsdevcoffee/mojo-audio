# MAX Engine conv2d Findings — Real Neural Network Implementation

**Date:** 2026-03-20
**Context:** Implementing NSF-HiFiGAN vocoder (RVC v2 voice conversion) using MAX Graph API
**MAX version:** 26.3.0.dev2026032005

---

## Summary

Building a real neural vocoder in MAX exposed 7 distinct issues with `ops.conv2d` and related ops. The unifying workaround — **im2col + matmul** — replaces all conv2d usage and achieves numerically perfect results (0.9998 correlation with PyTorch reference).

This is the kind of real-world stress test that pre-release frameworks need, and documents patterns useful to anyone building neural networks in MAX today.

---

## The Bugs

### 1. ops.conv2d produces wrong values for C_in >= 8

**Filed:** modular/modular#6248

The core conv2d operation returns **numerically incorrect results** when the input channel count is 8 or higher. This affects essentially every real neural network layer.

**Discovery:** conv1d layers in HiFiGAN resblocks (C=256) produced garbage outputs. Bisecting by channel count revealed the threshold at C_in=8.

**Impact:** Critical — makes conv2d unusable for most real workloads.

### 2. ops.conv2d compiler error with mixed stride configurations

**Not yet filed**

When a single MAX Graph contains multiple `ops.conv2d` calls with **different stride/padding configurations**, the compiler fails during `session.load()`:

```
ValueError: Graph compilation failed:
error: All operation types must have the same shape
```

Key details:
- Graph construction succeeds (ops validate at build time)
- Individual conv2d ops compile fine in isolation
- Failure occurs during compiler optimization/fusion passes
- Reproducible with as few as 2 conv2d ops (e.g., stride=12 + stride=4)

**Discovery:** HiFiGAN uses 4 noise_convs with strides [40, 4, 2, 1] plus 4 conv_transpose stages. Adding the second noise_conv triggered the error.

### 3. ops.conv2d fails with K=1, C_in=1

Layout transform bug when kernel size is 1 and input channels is 1 (RSCF format).

**Workaround:** Zero-pad kernel to K=3 with the original tap centered.

### 4. ops.rebind overwrites static dimensions

`ops.rebind(tensor, target_shape)` applies **all** dimensions from target_shape, not just symbolic ones. If your conv output has 512 channels but you rebind to input shape `[B, T, 1, 192]`, the channel dimension becomes 192.

**Discovery:** conv_pre (192→512 channels) output was rebinded to input shape, corrupting the channel count. Downstream conv_transpose_1d reshape then failed because element counts didn't match.

**Correct usage:** Construct the rebind shape explicitly:
```python
# Wrong: overwrites C_out with C_in
out = ops.rebind(out, x.shape)

# Right: only reconcile symbolic T dim
out = ops.rebind(out, [x.shape[0], x.shape[1], 1, C_out])
```

### 5-7. Missing ops

- **No `ops.conv_transpose`** — must implement via zero-interleave + forward conv
- **No dilated convolution** — must pre-expand kernel with zeros
- **No `ops.batch_norm`** — must bake into scale/offset at weight load time

---

## The Universal Workaround: im2col + matmul

A single pattern replaces all conv2d usage and handles every variant:

### Standard Conv1d (same padding, stride=1)
```
1. Pad input along T for "same" output
2. For each kernel position k: slice shifted copy of padded input
3. Concat slices along channel dim → [B, T, K*C_in]
4. Matmul with reshaped weight [K*C_in, C_out] → [B, T, C_out]
```

### ConvTranspose1d (upsampling)
```
1. Zero-interleave: insert S-1 zeros between time steps
2. Apply im2col conv1d with flipped kernel and asymmetric padding
```

### Strided Conv1d (downsampling, C_in=1)
```
1. Pad input along T
2. For each block b (K/stride blocks):
   - Slice a Ta-length region starting at b*stride
   - Reshape [B, Ta] → [B, T_out, stride] (groups stride-spaced samples)
3. Concat blocks → [B, T_out, K]
4. Matmul with [K, C_out]
```

### Dilated Conv1d
```
1. Pre-expand kernel: insert (dilation-1) zeros between taps
2. Apply standard im2col with expanded kernel
```

---

## Numerical Verification

| Component | MAX vs PyTorch | Notes |
|-----------|---------------|-------|
| conv1d (conv_pre) | max diff: 1e-6, corr: 1.000 | im2col + matmul |
| conv_transpose_1d | max diff: 0, corr: 1.000 | im2col + matmul |
| Full neural filter | max diff: 0.011, corr: 0.9998 | Same excitation signal |
| Full pipeline | corr: 0.659 | Different harmonic sources (expected) |

---

## Content Opportunities

### Blog Posts
1. **"Building a Neural Vocoder in MAX Engine — Every Bug We Hit (and How We Fixed Them)"**
   - Honest real-world experience report
   - Shows deep commitment to the ecosystem
   - Useful reference for other early adopters
   - Positions dev coffee as serious Mojo practitioners

2. **"im2col: The One Pattern That Replaces All of MAX's Broken conv2d"**
   - Technical deep-dive with diagrams
   - The stride-reshaping trick for downsampling is novel
   - Applicable to any neural network in MAX

3. **"From PyTorch to MAX: Porting a Real Voice Conversion Model"**
   - Architecture comparison
   - Weight loading patterns
   - NHWC layout conversion
   - Performance considerations

### Video Content
- Live debugging session (the "hunt" for why tests pass but real weights don't)
- Side-by-side PyTorch vs MAX execution walkthrough
- Demo: real-time voice conversion on DGX Spark

### Twitter/Social
- Thread: "We ported RVC v2 voice conversion to @modaborehq's MAX Engine. Here's what we learned..."
- Quick code snippets showing each workaround
- Tag Modular team, offer to pair on bug fixes

### Bug Reports to File
- Multi-stride conv2d compiler error (with minimal repro)
- ops.rebind static dimension behavior (with example)
- These show good-faith community engagement
