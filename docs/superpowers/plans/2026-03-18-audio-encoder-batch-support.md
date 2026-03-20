# AudioEncoder Batch > 1 Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add `batch_size: int = 1` to `AudioEncoder` so both MAX graphs accept `[B, ...]` inputs and `encode()` processes multi-sample batches.

**Architecture:** Static batch dimension baked into MAX graphs at construction time. The `batch_size` int flows through `from_pretrained` → `_from_weights` → graph construction + `_transformer_block_ops`. The `encode()` method validates input batch matches compiled size and loops pos_conv over the batch.

**Tech Stack:** Python, MAX Engine (MAX Graph API), numpy

**Spec:** `docs/superpowers/specs/2026-03-18-audio-encoder-batch-support-design.md`

---

### Task 1: Write failing batch tests

**Files:**
- Modify: `tests/test_audio_encoder.py:451-522` (add tests to `TestAudioEncoderShapes`)

These tests will fail until the implementation is complete. They use the existing `_make_full_weights` helper.

- [ ] **Step 1: Add 4 batch tests to `TestAudioEncoderShapes`**

Add after `test_encode_output_not_nan` (after line 522):

```python
    def test_encode_batch2_shape(self):
        """Batch=2: two 1s samples -> [2, 49, 768] on CPU."""
        import numpy as np
        from models.audio_encoder import AudioEncoder

        model = AudioEncoder._from_weights(self._make_full_weights(), device="cpu", batch_size=2)
        audio = np.zeros((2, 16000), dtype=np.float32)
        out = model.encode(audio)
        assert out.shape == (2, 49, 768), f"Expected (2,49,768) got {out.shape}"

    def test_encode_batch1_unchanged(self):
        """Default batch_size=1 still gives [1, 49, 768] (backward compat)."""
        import numpy as np
        from models.audio_encoder import AudioEncoder

        model = AudioEncoder._from_weights(self._make_full_weights(), device="cpu")
        audio = np.zeros((1, 16000), dtype=np.float32)
        out = model.encode(audio)
        assert out.shape == (1, 49, 768), f"Expected (1,49,768) got {out.shape}"

    def test_encode_batch2_not_nan(self):
        """Batch=2 output must not contain NaN or Inf."""
        import numpy as np
        from models.audio_encoder import AudioEncoder

        model = AudioEncoder._from_weights(self._make_full_weights(), device="cpu", batch_size=2)
        audio = np.random.randn(2, 16000).astype(np.float32) * 0.1
        out = model.encode(audio)
        assert not np.isnan(out).any(), "Batch output contains NaN"
        assert not np.isinf(out).any(), "Batch output contains Inf"

    def test_encode_batch2_samples_differ(self):
        """Different inputs in a batch must produce different outputs."""
        import numpy as np
        from models.audio_encoder import AudioEncoder

        rng = np.random.default_rng(123)
        model = AudioEncoder._from_weights(self._make_full_weights(), device="cpu", batch_size=2)
        audio = rng.standard_normal((2, 16000)).astype(np.float32)
        out = model.encode(audio)
        diff = np.abs(out[0] - out[1]).mean()
        assert diff > 0.001, f"Batch samples too similar (mean diff={diff:.6f})"
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pixi run test-models -k "test_encode_batch2_shape" -v 2>&1 | head -30`
Expected: FAIL (TypeError: `_from_weights()` got unexpected keyword argument `batch_size`)

- [ ] **Step 3: Commit failing tests**

```bash
git add tests/test_audio_encoder.py
git commit -m "test: add batch>1 tests for AudioEncoder (failing)"
```

---

### Task 2: Thread `batch_size` parameter through signatures

**Files:**
- Modify: `src/models/audio_encoder.py:8-26` (`_transformer_block_ops` signature + docstring)
- Modify: `src/models/audio_encoder.py:115-121` (`__init__`)
- Modify: `src/models/audio_encoder.py:123-129` (`_from_weights` signature)
- Modify: `src/models/audio_encoder.py:258` (constructor call)
- Modify: `src/models/audio_encoder.py:261-277` (`from_pretrained`)

This task ONLY adds the parameter — no graph or logic changes yet.

- [ ] **Step 1: Add `batch_size` to `_transformer_block_ops`**

Change line 8 from:
```python
def _transformer_block_ops(x, block_weights: dict, device_ref, heads: int = 12, hidden: int = 768):
```
to:
```python
def _transformer_block_ops(x, block_weights: dict, device_ref, heads: int = 12, hidden: int = 768, batch_size: int = 1):
```

Update docstring (line 17): `x: MAX graph TensorValue, shape [B, T, hidden].`
Update docstring (line 25): `MAX graph TensorValue, shape [B, T, hidden].`

- [ ] **Step 2: Add `batch_size` to `__init__`**

Change line 115 from:
```python
    def __init__(self, _model, _device, _model2=None, _pos_conv_weights=None, _enc_norm_weights=None):
```
to:
```python
    def __init__(self, _model, _device, _model2=None, _pos_conv_weights=None, _enc_norm_weights=None, _batch_size=1):
```

Add after line 121:
```python
        self._batch_size = _batch_size
```

- [ ] **Step 3: Add `batch_size` to `_from_weights`**

Change line 124 from:
```python
    def _from_weights(cls, weights: dict, device: str = "auto") -> "AudioEncoder":
```
to:
```python
    def _from_weights(cls, weights: dict, device: str = "auto", batch_size: int = 1) -> "AudioEncoder":
```

Update the constructor call on line 258 from:
```python
        return cls(_model=model1, _device=dev, _model2=model2,
                   _pos_conv_weights=pos_conv_w, _enc_norm_weights=enc_norm_w)
```
to:
```python
        return cls(_model=model1, _device=dev, _model2=model2,
                   _pos_conv_weights=pos_conv_w, _enc_norm_weights=enc_norm_w,
                   _batch_size=batch_size)
```

- [ ] **Step 4: Add `batch_size` to `from_pretrained`**

Change lines 262-266 from:
```python
    def from_pretrained(
        cls,
        model_id: str,
        device: str = "auto",
        cache_dir: str | None = None,
    ) -> "AudioEncoder":
```
to:
```python
    def from_pretrained(
        cls,
        model_id: str,
        device: str = "auto",
        cache_dir: str | None = None,
        batch_size: int = 1,
    ) -> "AudioEncoder":
```

Change line 277 from:
```python
        return cls._from_weights(weights, device=device)
```
to:
```python
        return cls._from_weights(weights, device=device, batch_size=batch_size)
```

- [ ] **Step 5: Verify existing tests still pass (no behavior change yet)**

Run: `pixi run test-models -k "not slow and not batch2" -v 2>&1 | tail -10`
Expected: All existing tests PASS (batch_size defaults to 1 everywhere)

---

### Task 3: Update `_transformer_block_ops` reshapes

**Files:**
- Modify: `src/models/audio_encoder.py:62-76` (Q/K/V reshape + context merge)

- [ ] **Step 1: Replace 4 hardcoded `[1, ...]` reshapes**

Change line 62 from:
```python
    q = _perm4(ops.reshape(q, [1, -1, heads, head_dim]), [0, 2, 1, 3])
    k = _perm4(ops.reshape(k, [1, -1, heads, head_dim]), [0, 2, 1, 3])
    v = _perm4(ops.reshape(v, [1, -1, heads, head_dim]), [0, 2, 1, 3])
```
to:
```python
    q = _perm4(ops.reshape(q, [batch_size, -1, heads, head_dim]), [0, 2, 1, 3])
    k = _perm4(ops.reshape(k, [batch_size, -1, heads, head_dim]), [0, 2, 1, 3])
    v = _perm4(ops.reshape(v, [batch_size, -1, heads, head_dim]), [0, 2, 1, 3])
```

Change line 76 from:
```python
    context = ops.reshape(context, [1, -1, hidden])
```
to:
```python
    context = ops.reshape(context, [batch_size, -1, hidden])
```

Update comments on lines 61, 68, 72, 74 to replace `[1,` with `[B,` for clarity.

---

### Task 4: Update Graph 1 — input type, CNN reshapes, GroupNorm

**Files:**
- Modify: `src/models/audio_encoder.py:158-218` (Graph 1 block)

This is the trickiest task. The GroupNorm must normalize per-sample (not across the batch).

- [ ] **Step 1: Update Graph 1 input type and comments**

Change line 160 comment to: `# Input: [B, L, 1, 1] NHWC audio → Output: [B, T, 768] features`

Change line 164 from:
```python
            input_types=[TensorType(DType.float32, [1, Dim("L"), 1, 1], device_ref)],
```
to:
```python
            input_types=[TensorType(DType.float32, [batch_size, Dim("L"), 1, 1], device_ref)],
```

- [ ] **Step 2: Update CNN reshape ops (3 sites — skip GroupNorm lines 183-199, handled in Step 3)**

Line 178 — after conv2d, reshape `[B, T', 1, c_out]` → `[B, T', c_out]`:
```python
                conv_out = ops.reshape(conv_out, [batch_size, -1, c_out])
```

Line 201 — GELU output: change `[1, -1, 1, c_out]` to `[batch_size, -1, 1, c_out]`:
```python
                x = ops.reshape(ops.gelu(conv_out), [batch_size, -1, 1, c_out])
```

Line 203 — final CNN reshape: change `[1, -1, 512]` to `[batch_size, -1, 512]`:
```python
            x = ops.reshape(x, [batch_size, -1, 512])
```

- [ ] **Step 3: Fix GroupNorm for batch > 1 (per-sample normalization)**

The current GroupNorm (lines 183-199) works for batch=1 by reshaping `[1, T', C]` → `[C, T']`. For batch > 1, we need `[B, T', C]` → `[B*C, T']` so each (sample, channel) pair normalizes independently.

Replace lines 183-199 with:

```python
                    # GroupNorm(num_groups=C, num_channels=C): normalize each channel c
                    # independently over the T' temporal dimension, per sample.
                    # [B, T', C] → transpose(1,2) → [B, C, T'] → reshape [B*C, T']
                    gn_in = ops.reshape(
                        ops.transpose(conv_out, 1, 2),  # [B, C, T']
                        [batch_size * c_out, -1],        # [B*C, T']
                    )
                    mean_ = ops.mean(gn_in, axis=1)                           # [B*C, 1]
                    diff_ = ops.sub(gn_in, mean_)                             # [B*C, T']
                    var_ = ops.mean(ops.mul(diff_, diff_), axis=1)            # [B*C, 1]
                    std_ = ops.sqrt(ops.add(var_, EPS_GN))                    # [B*C, 1]
                    normed_ = ops.div(diff_, std_)                            # [B*C, T']
                    # gamma/beta: [C] → tile to [B*C, 1] by repeating B times
                    gamma_1d = _const(weights[norm_w_key])                    # [C]
                    beta_1d = _const(weights[f"cnn.{i}.norm.bias"])           # [C]
                    gamma_tiled = ops.reshape(
                        ops.broadcast_to(
                            ops.reshape(gamma_1d, [1, c_out]),               # [1, C]
                            [batch_size, c_out],                              # [B, C]
                        ),
                        [batch_size * c_out, 1],                              # [B*C, 1]
                    )
                    beta_tiled = ops.reshape(
                        ops.broadcast_to(
                            ops.reshape(beta_1d, [1, c_out]),
                            [batch_size, c_out],
                        ),
                        [batch_size * c_out, 1],
                    )
                    gn_out = ops.add(ops.mul(normed_, gamma_tiled), beta_tiled)  # [B*C, T']
                    # Reshape back: [B*C, T'] → [B, C, T'] → transpose(1,2) → [B, T', C]
                    conv_out = ops.transpose(
                        ops.reshape(gn_out, [batch_size, c_out, -1]),        # [B, C, T']
                        1, 2,                                                 # [B, T', C]
                    )
```

**Why this works:** `ops.transpose(conv_out, 1, 2)` swaps T' and C giving `[B, C, T']`. Flattening to `[B*C, T']` means each row is one (sample, channel) pair. `ops.mean(axis=1)` normalizes each row independently — matching PyTorch GroupNorm(groups=C) behavior. gamma/beta are broadcast from `[C]` → `[B*C, 1]` by tiling B times.

**Note:** If MAX doesn't support `ops.broadcast_to`, use `ops.tile` instead:
```python
gamma_tiled = ops.reshape(
    ops.tile(ops.reshape(gamma_1d, [1, c_out]), [batch_size, 1]),
    [batch_size * c_out, 1],
)
```

- [ ] **Step 4: Verify Graph 1 compiles with batch_size=1**

Run: `pixi run test-models -k "test_encode_1s_shape" -v 2>&1 | tail -5`
Expected: PASS (batch_size defaults to 1, behavior unchanged)

---

### Task 5: Update Graph 2 — input type + pass `batch_size` to blocks

**Files:**
- Modify: `src/models/audio_encoder.py:228-243` (Graph 2 block)

- [ ] **Step 1: Update Graph 2 input type**

Change line 230 from:
```python
            input_types=[TensorType(DType.float32, [1, Dim("T"), 768], device_ref)],
```
to:
```python
            input_types=[TensorType(DType.float32, [batch_size, Dim("T"), 768], device_ref)],
```

- [ ] **Step 2: Pass `batch_size` to `_transformer_block_ops`**

Change line 241 from:
```python
                x = _transformer_block_ops(x, block_w, device_ref)
```
to:
```python
                x = _transformer_block_ops(x, block_w, device_ref, batch_size=batch_size)
```

Update comment on line 222: `# Input: [B, T, 768]` and line 232: `x = g2.inputs[0]  # [B, T, 768]`

---

### Task 6: Update `encode()` — validation, reshape, pos_conv batch loop

**Files:**
- Modify: `src/models/audio_encoder.py:279-362` (`encode` method)

- [ ] **Step 1: Update docstring and add input validation**

Replace lines 279-288 with:
```python
    def encode(self, audio: np.ndarray) -> np.ndarray:
        """Encode raw audio waveform to feature vectors.

        Args:
            audio: Float32 numpy array, shape [B, samples], 16kHz, normalized [-1, 1].
                   B must match the batch_size used at construction time.

        Returns:
            Float32 numpy array, shape [B, time_frames, 768].
            For 1s audio at batch=1: [1, 49, 768].
        """
        from max.driver import Accelerator, Tensor

        B = audio.shape[0]
        if B != self._batch_size:
            raise ValueError(
                f"Input batch size {B} does not match model batch_size {self._batch_size}. "
                f"Rebuild model with AudioEncoder._from_weights(..., batch_size={B})"
            )
```

- [ ] **Step 2: Update input reshape**

Change line 292 from:
```python
        audio_in = audio.reshape(1, -1, 1, 1).astype(np.float32)
```
to:
```python
        audio_in = audio.reshape(B, -1, 1, 1).astype(np.float32)
```

- [ ] **Step 3: Replace single-sample pos_conv with batch loop**

Replace lines 308-342 (the entire pos_conv block) with:

```python
            if self._pos_conv_weights is not None and self._pos_conv_weights["weight"] is not None:
                pw = self._pos_conv_weights["weight"]   # [768, 48, 128]
                pb = self._pos_conv_weights["bias"]     # [768] or None
                # features_np: [B, T, 768]
                C_in = 768
                groups = 16
                C_per_g = C_in // groups  # 48
                K = pw.shape[2]           # 128
                pad = K // 2              # 64
                from scipy.special import erf as scipy_erf

                pos_batch = np.zeros_like(features_np)  # [B, T, 768]
                for b in range(B):
                    x_bt = features_np[b]  # [T, 768]
                    T = x_bt.shape[0]
                    out_t = np.zeros((T, C_in), dtype=np.float32)
                    for grp in range(groups):
                        gs = grp * C_per_g
                        ge = gs + C_per_g
                        x_g = x_bt[:, gs:ge]  # [T, C_per_g]
                        x_g_pad = np.pad(x_g, ((pad, pad), (0, 0)))  # [T+128, C_per_g]
                        rows = np.stack([x_g_pad[t:t+K, :] for t in range(T + 1)], axis=0)
                        rows_2d = rows.reshape(T + 1, K * C_per_g)
                        w_g = pw[gs:ge, :, :]
                        w_2d = w_g.transpose(2, 1, 0).reshape(K * C_per_g, C_per_g)
                        out_g = (rows_2d @ w_2d)[:T, :]
                        out_t[:, gs:ge] = out_g
                    if pb is not None:
                        out_t += pb[np.newaxis, :]
                    # GELU activation
                    pos_batch[b] = (0.5 * out_t * (1.0 + scipy_erf(out_t / np.sqrt(2.0)))).astype(np.float32)
                features_np = features_np + pos_batch  # residual add [B, T, 768]
```

- [ ] **Step 4: Verify enc LayerNorm is batch-agnostic (no changes needed)**

Lines 348-351 use `axis=-1` which works on `[B, T, 768]`. No change needed. Confirm by reading the code.

---

### Task 7: Run all tests and verify

- [ ] **Step 1: Run existing tests (backward compat)**

Run: `pixi run test-models -k "not slow" -v 2>&1 | tail -20`
Expected: ALL tests pass including existing batch=1 tests

- [ ] **Step 2: Run new batch tests specifically**

Run: `pixi run test-models -k "batch2" -v 2>&1`
Expected: All 4 batch tests pass:
- `test_encode_batch2_shape` — `(2, 49, 768)` ✓
- `test_encode_batch1_unchanged` — `(1, 49, 768)` ✓
- `test_encode_batch2_not_nan` — no NaN/Inf ✓
- `test_encode_batch2_samples_differ` — outputs differ ✓

- [ ] **Step 3: Commit implementation**

```bash
git add src/models/audio_encoder.py tests/test_audio_encoder.py
git commit -m "feat: add batch_size support to AudioEncoder (Sprint 2, Item 2C)"
```

---

### Known Risks

1. **MAX `ops.broadcast_to` availability** — if not available, use `ops.tile` as fallback for gamma/beta tiling in GroupNorm. Check MAX API docs if compilation fails.
2. **Graph compilation time** — batch_size=2 doubles the static shapes, which may increase JIT compilation time. This is expected and acceptable.
3. **Memory** — batch=2 roughly doubles memory for activations. Random weights in tests use small float magnitudes (`* 0.02`) to avoid overflow.
