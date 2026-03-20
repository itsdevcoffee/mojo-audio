# AudioEncoder Batch > 1 Support

**Date:** 2026-03-18
**Status:** Approved
**Sprint:** 2, Item 2C

## Summary

Add `batch_size: int = 1` parameter to `AudioEncoder` so both MAX graphs accept `[B, ...]` inputs and `encode()` processes multi-sample batches.

## Changes

### `src/models/audio_encoder.py`

**`__init__`** — store `self._batch_size = batch_size` (new param, default 1).

**`_transformer_block_ops`** — add `batch_size: int = 1` param. Replace 4 hardcoded reshapes:

- `ops.reshape(q, [1, -1, heads, head_dim])` becomes `[batch_size, -1, heads, head_dim]` (same for k, v)
- `ops.reshape(context, [1, -1, hidden])` becomes `[batch_size, -1, hidden]`

**`_from_weights`** — add `batch_size: int = 1` param:

- Graph 1 input type: `[1, Dim("L"), 1, 1]` becomes `[batch_size, Dim("L"), 1, 1]`
- Graph 1 internal reshapes: all `[1, -1, ...]` patterns become `[batch_size, -1, ...]`
- GroupNorm: see GroupNorm Detail section below
- Graph 2 input type: `[1, Dim("T"), 768]` becomes `[batch_size, Dim("T"), 768]`
- Pass `batch_size` to `_transformer_block_ops` and to `cls(...)` constructor

**`from_pretrained`** — add `batch_size: int = 1`, forward to `_from_weights`

**`encode()`**:

- Update docstring: `[B, samples]` input, `[B, time_frames, 768]` output
- Validate: `raise ValueError` if `audio.shape[0] != self._batch_size` with descriptive message
- Reshape: `audio.reshape(self._batch_size, -1, 1, 1)`
- Pos conv bridge: allocate `out_batch = np.zeros_like(features_np)` shape `[B, T, 768]`, then `for b in range(B)`: extract `x_bt = features_np[b]` shape `[T, 768]`, run existing im2col grouped conv producing `out_t` shape `[T, 768]`, assign `out_batch[b] = out_t`. After the loop: apply bias, GELU, residual add `features_np = features_np + out_batch`
- Enc LayerNorm: already batch-agnostic (numpy ops use `axis=-1` on `[B, T, 768]`)

### `tests/test_audio_encoder.py`

Add to `TestAudioEncoderShapes`:

- `test_encode_batch2_shape`: build with `batch_size=2`, encode `(2, 16000)` input, assert `(2, 49, 768)`
- `test_encode_batch1_unchanged`: existing batch=1 still gives `(1, 49, 768)`
- `test_encode_batch2_not_nan`: batch=2 with random input, assert no NaN/Inf
- `test_encode_batch2_samples_differ`: two different inputs in a batch produce different outputs

### Unchanged

- `PitchExtractor` (not in Sprint 2 scope)
- Public `pitch_shift` API
- Weight loading (`_weight_loader.py`)

## GroupNorm Detail

**Correction from initial draft:** PyTorch `nn.GroupNorm` normalizes **per-sample**, not across the batch. Each `(batch_item, group)` pair computes its own mean/variance independently.

Current code (batch=1): transposes `[1, T', C]` to `[C, T']` and normalizes each row (channel) over T'.

With batch: the tensor is `[B, T', C]`. The correct approach is to reshape to `[B*C, T']` so each row is one (batch_item, channel) pair, normalize each row over T' independently, then reshape back to `[B, T', C]`.

Concretely in MAX graph ops:
1. Reshape `[B, T', C]` via transpose+reshape to `[B*C, T']`
2. `mean(axis=1)` → `[B*C, 1]`
3. Normalize per-row
4. Apply gamma/beta (broadcast across T')
5. Reshape back to `[B, T', C]`

## Definition of Done

- `AudioEncoder._from_weights(weights, device="cpu", batch_size=2)` works
- `model.encode(np.zeros((2, 16000), dtype=np.float32))` returns shape `[2, 49, 768]`
- All tests in `test_audio_encoder.py` pass without modification (backward compat)
- New batch=2 shape, NaN, and independence tests pass
