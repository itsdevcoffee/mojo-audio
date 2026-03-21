# VITS Synthesis Pipeline Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build the remaining RVC v2 inference components (enc_p, flow, speaker embedding) in MAX Graph and wire them with the existing AudioEncoder, RMVPE, and HiFiGAN into a single `VoiceConverter.convert()` API.

**Architecture:** Two new MAX Graphs: (1) `vits_encoder` — TextEncoder that projects ContentVec 768-dim features + pitch embedding → prior mean/logvar in 192-dim latent space via 6-layer relative-position transformer, (2) `vits_flow` — 4-stage reverse normalizing flow with WaveNet dilated convs and baked speaker embedding. Reparameterization (sampling) in numpy between the two graphs. Speaker conditioning baked into HiFiGAN's conv_pre.bias at load time.

**Tech Stack:** Python, MAX Engine (MAX Graph API), numpy, torch (weight loading only)

**Spec:** `docs/superpowers/specs/2026-03-21-vits-synthesis-design.md`

---

## File Map

| File | Responsibility |
|---|---|
| `src/models/_vits_weight_loader.py` | Extract `enc_p.*`, `flow.*`, `emb_g.*` from RVC checkpoint; weight-norm reconstruction; bake speaker embedding + HiFiGAN cond into conv_pre.bias |
| `src/models/_vits_graph.py` | MAX Graph construction: TextEncoder (enc_p) + ResidualCouplingBlock (flow reverse) |
| `src/models/voice_converter.py` | `VoiceConverter` class: orchestrates AudioEncoder → RMVPE → enc_p → sample → flow → HiFiGAN |
| `tests/test_vits.py` | Weight loader, enc_p shape/numerical, flow shape/numerical, integration tests |
| `tests/_rvc_pytorch_reference.py` | Extend with `run_enc_p_reference`, `run_flow_reference`, `run_full_vits_reference` |

**Reference files** (read, don't modify unless noted):
- `src/models/_hifigan_graph.py` — im2col conv1d, leaky_relu, conv_transpose patterns (reuse directly)
- `src/models/_hifigan_weight_loader.py` — `reconstruct_weight_norm`, `extract_hifigan_weights` patterns
- `src/models/hifigan.py` — NSFHiFiGAN class API pattern
- `src/models/audio_encoder.py` — Multi-head attention (absolute pos), transformer block pattern
- `src/models/_hifigan_weight_loader.py:70-98` — `reconstruct_weight_norm` (import and reuse)
- `src/models/hifigan.py` — Modify: fold baked `cond(g)` into conv_pre.bias at load time

**Applio reference** (read for correctness checks):
- `/home/maskkiller/repos/Applio/rvc/lib/algorithm/synthesizers.py:206-243` — inference path
- `/home/maskkiller/repos/Applio/rvc/lib/algorithm/encoders.py:88-144` — TextEncoder
- `/home/maskkiller/repos/Applio/rvc/lib/algorithm/residuals.py:103-262` — Flow + CouplingLayer
- `/home/maskkiller/repos/Applio/rvc/lib/algorithm/modules.py:5-117` — WaveNet
- `/home/maskkiller/repos/Applio/rvc/lib/algorithm/attentions.py:6-186` — MultiHeadAttention + FFN
- `/home/maskkiller/repos/Applio/rvc/lib/algorithm/commons.py:89-103` — fused_add_tanh_sigmoid_multiply
- `/home/maskkiller/repos/Applio/rvc/infer/pipeline.py:293-376` — voice_conversion function

**RVC checkpoints for testing:**
- `/home/maskkiller/Downloads/voice files/extracted/theweeknd biggest data set/theweekv1.pth` (48k)
- `/home/maskkiller/Downloads/voice files/extracted/drake/drake_e1000_s14000.pth` (40k)

---

### Task 1: Weight loader — extract enc_p, flow, emb_g weights

**Files:**
- Create: `src/models/_vits_weight_loader.py`
- Create: `tests/test_vits.py` (initial test file)

The weight loader extracts TextEncoder and flow weights from RVC checkpoints, reconstructs weight-normalized layers, and bakes the speaker embedding.

- [ ] **Step 1: Write weight extraction tests**

Test that `extract_vits_weights` correctly:
1. Extracts `enc_p.*` keys (strips prefix)
2. Extracts `flow.*` keys (strips prefix, skips Flip layers which have no params)
3. Extracts `emb_g.weight` and returns baked `g` vector `[256, 1]`
4. Reconstructs weight-norm pairs (`.parametrizations.weight.original0/1` → plain `.weight`)
5. All values are float32

Use fake state_dict with synthetic weights (no checkpoint download needed).

**Important:** Start `tests/test_vits.py` with the same sys.path boilerplate as `tests/test_hifigan.py`:
```python
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
```

Run: `pixi run python -m pytest tests/test_vits.py::TestVITSWeightLoader -v`
Expected: FAIL (module not found)

- [ ] **Step 2: Implement weight loader**

Create `src/models/_vits_weight_loader.py` with:
- `extract_vits_weights(state_dict) -> dict` — extract enc_p.* and flow.* keys, reconstruct weight-norm
- `extract_speaker_embedding(state_dict, sid=0) -> np.ndarray` — return `emb_g.weight[sid]` as `[256, 1]`
- `bake_hifigan_cond(hifigan_weights, g) -> None` — compute `cond(g)` and fold into `conv_pre.bias` in-place
- `load_vits_weights(checkpoint_path) -> (vits_weights, hifigan_weights, config)` — full loading pipeline

Reuse `reconstruct_weight_norm` from `_hifigan_weight_loader.py`.

Handle modern PyTorch parametrizations: detect `.parametrizations.weight.original0` (weight_g) / `.original1` (weight_v) key patterns in addition to the old `.weight_g` / `.weight_v` pattern.

Run: `pixi run python -m pytest tests/test_vits.py::TestVITSWeightLoader -v`
Expected: PASS

- [ ] **Step 3: Test with real checkpoint**

Add a test that loads `theweekv1.pth` and verifies:
- enc_p weights have expected shapes (emb_phone.weight = [192, 768], proj.weight = [384, 192, 1], etc.)
- flow weights have 4 coupling layers (indices 0, 2, 4, 6)
- Speaker embedding g has shape [256, 1]
- Baked cond(g) modifies conv_pre.bias shape correctly

Mark with `@pytest.mark.skipif` if checkpoint not present.

Run: `pixi run python -m pytest tests/test_vits.py -v`
Expected: PASS

- [ ] **Step 4: Commit**

```bash
git add src/models/_vits_weight_loader.py tests/test_vits.py
git commit -m "feat(vits): weight loader for enc_p, flow, and speaker embedding (Task 1)"
```

---

### Task 2: PyTorch reference — enc_p and flow

**Files:**
- Modify: `tests/_rvc_pytorch_reference.py`

Extend the existing PyTorch reference with functions to run enc_p and flow independently, for numerical comparison.

- [ ] **Step 1: Add enc_p reference function**

Add `run_enc_p_reference(checkpoint_path, features, pitch, sr_tag)` that:
1. Loads the RVC model (reuse `load_rvc_generator` pattern but load full `Synthesizer`)
2. Runs `model.enc_p(phone, pitch, lengths)` → returns (mean, logvar, mask) as numpy

- [ ] **Step 2: Add flow reference function**

Add `run_flow_reference(checkpoint_path, z_p, mask, sid, sr_tag)` that:
1. Loads model, extracts baked `g = model.emb_g(sid).unsqueeze(-1)`
2. Runs `model.flow(z_p, mask, g=g, reverse=True)` → returns z as numpy

- [ ] **Step 3: Add full VITS inference reference**

Add `run_full_vits_reference(checkpoint_path, features, pitch, pitchf, sr_tag)` that:
1. Loads model, runs full `model.infer(phone, lengths, pitch, pitchf, sid)` → returns audio as numpy

- [ ] **Step 4: Smoke test references**

Write a quick script that runs all three reference functions with the Weeknd checkpoint and verifies shapes + no NaN.

- [ ] **Step 5: Commit**

```bash
git add tests/_rvc_pytorch_reference.py
git commit -m "feat(vits): PyTorch reference functions for enc_p, flow, full VITS (Task 2)"
```

---

### Task 3: Flow graph — WaveNet + ResidualCouplingLayer + Flip

**Files:**
- Create: `src/models/_vits_graph.py`
- Add to: `tests/test_vits.py`

Build the normalizing flow in MAX Graph. Start with WaveNet (the innermost building block), then coupling layer, then full flow block. The flow is simpler than enc_p (no attention), so build it first.

- [ ] **Step 1: Write WaveNet shape test**

Test that a single WaveNet forward pass produces correct output shape:
- Input: [1, 192, T] + mask [1, 1, T] + g [256, 1]
- Output: [1, 192, T]

Use random weights matching the spec config (hidden=192, k=5, dil=1, n_layers=3, gin=256).

Run: `pixi run python -m pytest tests/test_vits.py::TestWaveNet -v`
Expected: FAIL

- [ ] **Step 2: Implement WaveNet graph builder**

In `_vits_graph.py`, implement `build_wavenet(x, mask, g, weights, config, device_ref)`:
1. Compute `g_all = cond_layer(g)` — Conv1d(256, 1152, k=1) via im2col, producing all conditioning at once
2. For each layer i (0..2):
   - `x_in = in_layers[i](x)` — Conv1d(192, 384, k=5, pad=2) via im2col
   - `g_l = g_all[:, i*384:(i+1)*384, :]` — slice conditioning
   - `acts = tanh(x_in[:,:192] + g_l[:,:192]) * sigmoid(x_in[:,192:] + g_l[:,192:])` — gated activation
   - `res_skip = res_skip_layers[i](acts)` — Conv1d
   - If i < n-1: x = (x + res_skip[:,:192,:]) * mask; output += res_skip[:,192:,:]
   - If i == n-1: output += res_skip
3. Return output * mask

**Key pattern:** All Conv1d uses im2col from `_hifigan_graph.py`. Import `conv1d` function directly.

**Layout note:** WaveNet operates in [B, C, T] (channel-first) unlike HiFiGAN's NHWC. Handle the layout difference in the graph builder — either transpose to NHWC for conv1d calls, or implement a channel-first im2col variant. Decision: transpose to NHWC `[B, T, 1, C]` before each conv1d call and back, matching the established pattern.

Run: `pixi run python -m pytest tests/test_vits.py::TestWaveNet -v`
Expected: PASS

- [ ] **Step 3: Write ResidualCouplingLayer reverse test**

Test single coupling layer in reverse mode:
- Input: x [1, 192, T] + mask [1, 1, T] + g [256, 1]
- Output: [1, 192, T] (same shape)
- Verify: output differs from input (the coupling transform changed x1)

- [ ] **Step 4: Implement ResidualCouplingLayer reverse**

`build_coupling_layer_reverse(x, mask, g, weights, config, device_ref)`:
1. Split x into x0 [1, 96, T] and x1 [1, 96, T] along channel dim
2. h = pre_conv(x0) * mask — Conv1d(96, 192, k=1)
3. h = wavenet(h, mask, g, weights) — WaveNet from step 2
4. m = post_conv(h) * mask — Conv1d(192, 96, k=1)
5. x1 = (x1 - m) * mask — reverse coupling (mean_only)
6. Concat [x0, x1] → output

- [ ] **Step 5: Write Flip test**

Test that Flip reverses channel dimension:
- Input: [1, 192, T] with known values
- Output: channels reversed

- [ ] **Step 6: Implement Flip**

`flip_channels(x)`: reverse along channel dimension using `ops.slice_tensor` with reversed indices, or reshape tricks. Since channel dim is static (192), this can use `ops.concat` of reversed slices or a single `ops.gather` along dim 1.

- [ ] **Step 7: Write full flow reverse test**

Test `build_flow_graph` with 4 coupling layers + 4 flips (8 modules, reversed order):
- Input: z_p [1, 192, T] + mask [1, 1, T]
- Output: z [1, 192, T]
- Baked speaker embedding g as constant in graph

- [ ] **Step 8: Implement full flow graph**

`build_flow_graph(weights, config, device, batch_size=1)`:
1. Create graph with inputs: z_p [B, 192, T], mask [B, 1, T]
2. Bake g as ops.constant
3. Iterate modules in reverse: [Flip, CouplingLayer(3), Flip, CouplingLayer(2), Flip, CouplingLayer(1), Flip, CouplingLayer(0)]
4. Return compiled graph

- [ ] **Step 9: Numerical validation against PyTorch**

Compare `build_flow_graph` output against `run_flow_reference` with frozen z_p:
- Load real checkpoint weights
- Generate fixed z_p and mask
- Run both, compare: max diff < 1e-3, correlation > 0.999

- [ ] **Step 10: Commit**

```bash
git add src/models/_vits_graph.py tests/test_vits.py
git commit -m "feat(vits): flow graph — WaveNet + coupling layers + numerical validation (Task 3)"
```

---

### Task 4: TextEncoder graph — linear projection + pitch embedding + transformer

**Files:**
- Modify: `src/models/_vits_graph.py`
- Add to: `tests/test_vits.py`

Build the TextEncoder (enc_p) MAX Graph. This is the most complex new component — it has a 6-layer transformer with relative positional attention.

- [ ] **Step 1: Write FFN test**

Test feed-forward network: Conv1d(192, 768, k=3, pad=1) → ReLU → Conv1d(768, 192, k=3, pad=1)
- Input: [1, 192, T] + mask
- Output: [1, 192, T]

- [ ] **Step 2: Implement FFN**

`build_ffn(x, mask, weights, device_ref)`:
1. y = conv1d(x, w1, b1) — im2col, k=3
2. y = relu(y)
3. y = conv1d(y, w2, b2) — im2col, k=3
4. Return y * mask

- [ ] **Step 3: Write relative positional attention test**

Test multi-head attention with relative positional encoding:
- Input: x [1, 192, T] + mask [1, 1, T]
- Output: [1, 192, T]
- Config: n_heads=2, window_size=10, heads_share=True
- Verify output shape and no NaN

- [ ] **Step 4: Implement relative positional attention**

`build_rel_attention(x, mask, weights, config, device_ref)`:
1. Q, K, V = conv1d projections (192→192 each, k=1)
2. Reshape Q, K, V to [B, n_heads, T, head_dim] where head_dim=96
3. Compute attention scores: QK^T / sqrt(head_dim)
4. Add relative position bias from `emb_rel_k` [1, 21, 96]:
   - For each query position q and key position k, add bias from `emb_rel_k[0, clip(q-k+window, 0, 2*window), :]`
   - This requires building a relative position index matrix and gathering from emb_rel_k
5. Apply attention mask
6. Softmax
7. Weighted sum of V, add relative value bias from `emb_rel_v`
8. Reshape back to [B, 192, T]
9. Output projection conv1d (192→192, k=1)

**Note:** The relative position indexing is the trickiest part. For symbolic T, we may need to compute the index matrix at runtime or use a fixed max-T approach. If MAX doesn't support dynamic indexing well, fall back to computing relative attention in numpy and passing it as a graph input.

- [ ] **Step 5: Write LayerNorm test**

Test LayerNorm with gamma/beta:
- Input: [1, 192, T]
- Verify matches PyTorch LayerNorm output

- [ ] **Step 6: Implement LayerNorm**

`build_layer_norm(x, gamma, beta, device_ref)`:
1. Compute mean and variance along channel dim
2. Normalize: (x - mean) / sqrt(var + eps)
3. Scale and shift: gamma * normalized + beta

Note: this is channel-wise LayerNorm (like VITS), normalizing over the channel dimension for each time step. Different from the standard PyTorch LayerNorm which normalizes over the last dim.

- [ ] **Step 7: Write transformer encoder test**

Test 6-layer transformer encoder:
- Input: [1, 192, T] + mask
- Output: [1, 192, T]

- [ ] **Step 8: Implement transformer encoder**

`build_encoder(x, mask, weights, config, device_ref)`:
For each layer i (0..5):
1. y = rel_attention(x, mask, attn_weights[i])
2. x = layer_norm_1(x + y)
3. y = ffn(x, mask, ffn_weights[i])
4. x = layer_norm_2(x + y)
Return x * mask

- [ ] **Step 9: Write full enc_p test**

Test complete TextEncoder:
- Input: features [1, 20, 768] + pitch [1, 20] (int) + lengths [1]
- Output: mean [1, 192, 20], logvar [1, 192, 20], mask [1, 1, 20]

- [ ] **Step 10: Implement full enc_p graph**

`build_enc_p_graph(weights, config, device, batch_size=1)`:
1. Inputs: features [B, T, 768], pitch [B, T] (int32), lengths [B] (int32)
2. x = emb_phone(features) — Linear(768, 192) via matmul
3. x = x + emb_pitch(pitch) — Embedding lookup via gather
4. x = x * sqrt(192)
5. x = leaky_relu(x, alpha=0.1)
6. x = transpose to [B, 192, T]
7. Build mask from lengths
8. x = encoder(x, mask) — 6-layer transformer
9. stats = proj(x) * mask — Conv1d(192, 384, k=1)
10. Split stats into mean [B, 192, T] and logvar [B, 192, T]
11. Output: mean, logvar, mask

- [ ] **Step 11: Numerical validation against PyTorch**

Compare `build_enc_p_graph` output against `run_enc_p_reference`:
- Load real checkpoint
- Generate fixed features and pitch
- Run both, compare mean and logvar: max diff < 1e-3

- [ ] **Step 12: Commit**

```bash
git add src/models/_vits_graph.py tests/test_vits.py
git commit -m "feat(vits): enc_p graph — TextEncoder with relative positional attention (Task 4)"
```

---

### Task 5: Speaker conditioning in HiFiGAN

**Files:**
- Modify: `src/models/_vits_weight_loader.py` (add `bake_hifigan_cond` function — already created in Task 1)
- Add to: `tests/test_vits.py`

Fold the baked speaker embedding into HiFiGAN's conv_pre.bias so the existing HiFiGAN graph works unchanged. No changes to `hifigan.py` or `_hifigan_graph.py` needed.

- [ ] **Step 1: Write cond baking test**

Test that `bake_hifigan_cond(weights, g)`:
1. Computes `cond_out = Conv1d(g)` using cond.weight and cond.bias
2. Adds result to conv_pre.bias
3. Removes cond.weight and cond.bias from weights dict

- [ ] **Step 2: Implement cond baking**

In `_vits_weight_loader.py` (already created in Task 1), the `bake_hifigan_cond` function:
1. `g` is [256, 1], cond.weight is [512, 256, 1], cond.bias is [512]
2. `cond_out = g.T @ cond.weight.squeeze(-1).T + cond.bias` → [512]
3. `weights['conv_pre.bias'] += cond_out`

- [ ] **Step 3: Verify HiFiGAN output changes with cond**

Compare HiFiGAN output with and without baked cond — verify the output differs (speaker conditioning has effect).

- [ ] **Step 4: Commit**

```bash
git add src/models/_vits_weight_loader.py tests/test_vits.py
git commit -m "feat(vits): bake speaker conditioning into HiFiGAN conv_pre.bias (Task 5)"
```

---

### Task 6: VoiceConverter orchestrator

**Files:**
- Create: `src/models/voice_converter.py`
- Add to: `tests/test_vits.py`

Wire the full pipeline: AudioEncoder → RMVPE → enc_p → sample → flow → HiFiGAN.

- [ ] **Step 1: Write VoiceConverter shape test**

Test `VoiceConverter._from_weights()` builds and runs:
- Input: synthetic weights (random, matching all required shapes)
- Verify `convert(audio)` returns audio with expected length

This is a shape/smoke test only — numerical correctness comes in step 4.

- [ ] **Step 2: Implement VoiceConverter**

```python
class VoiceConverter:
    def __init__(self, _audio_encoder, _pitch_extractor, _enc_p_model,
                 _flow_model, _hifigan, _config):
        ...

    @classmethod
    def from_pretrained(cls, checkpoint_path, hubert_path="facebook/hubert-base-ls960",
                        rmvpe_path="lj1995/VoiceConversionWebUI", device="auto"):
        # 1. Load checkpoint
        # 2. Extract weights for all components
        # 3. Bake speaker embedding into HiFiGAN
        # 4. Build all MAX graphs
        # 5. Load AudioEncoder and PitchExtractor from their pretrained sources
        ...

    def convert(self, audio, pitch_shift=0, sr=16000):
        # 1. Resample to 16kHz if needed
        # 2. AudioEncoder.encode(audio) → features [B, T, 768]
        # 3. PitchExtractor.extract(audio) → f0 Hz
        # 4. Interpolate features 2x
        # 5. Apply pitch shift to f0
        # 6. Quantize f0 → pitch bins (0-255)
        # 7. Build mask from lengths
        # 8. enc_p(features, pitch, lengths) → mean, logvar, mask
        # 9. z_p = (mean + exp(logvar) * randn * 0.66666) * mask
        # 10. flow(z_p, mask) → z
        # 11. HiFiGAN(z * mask, f0) → audio
        ...
```

- [ ] **Step 3: Write from_pretrained integration test**

Test loading from real checkpoint (Weeknd 48k):
- `VoiceConverter.from_pretrained(checkpoint_path)` succeeds
- All sub-models loaded
- `convert(audio)` produces valid output (no NaN, reasonable length)

Mark with `@pytest.mark.skipif` if checkpoint or HuBERT weights not present.

- [ ] **Step 4: End-to-end numerical comparison**

Run full pipeline vs `run_full_vits_reference`:
- Same frozen random seed for z_p sampling
- Compare spectrograms + correlation
- Freeze z_p directly (pass same sampled tensor to both) for deterministic comparison

- [ ] **Step 5: End-to-end listen test setup**

Write a script `experiments/vits-comparison/compare.py` that:
1. Takes an input WAV + voice model .pth
2. Runs both Applio and our VoiceConverter
3. Saves both outputs as WAVs for A/B listening
4. Prints spectrogram diff stats

- [ ] **Step 6: Commit**

```bash
git add src/models/voice_converter.py tests/test_vits.py experiments/vits-comparison/compare.py
git commit -m "feat(vits): VoiceConverter end-to-end pipeline (Task 6)"
```

---

### Task 7: Pixi test task + final validation

**Files:**
- Modify: `pixi.toml` (add test-vits task)
- Add to: `tests/test_vits.py` (regression tests)

- [ ] **Step 1: Add pixi task**

Add `test-vits` task to `pixi.toml`:
```toml
[tasks.test-vits]
cmd = "pytest tests/test_vits.py -v -m 'not slow'"
```

- [ ] **Step 2: Run full test suite**

```bash
pixi run test-vits        # VITS tests
pixi run test-hifigan     # HiFiGAN regression (verify cond baking didn't break anything)
pixi run test-models      # AudioEncoder regression
```

All must pass.

- [ ] **Step 3: Test with multiple checkpoints**

Verify with at least 2 different voice models:
- Weeknd (48k): `theweekv1.pth`
- Drake (40k): `drake_e1000_s14000.pth`

Both should load, convert, and produce valid audio.

- [ ] **Step 4: Update roadmap**

Mark Sprint 4 items as complete in `docs/project/03-06-2026-roadmap.md`.
Add FAISS index retrieval to future roadmap items.

- [ ] **Step 5: Commit**

```bash
git add pixi.toml tests/test_vits.py docs/project/03-06-2026-roadmap.md
git commit -m "feat(vits): pixi test task + multi-checkpoint validation (Task 7)"
```

---

## Implementation Notes

### Conv1d Layout Convention

The existing `_hifigan_graph.py` conv1d operates in NHWC: `[B, T, 1, C]`. VITS operates in channel-first: `[B, C, T]`. For each conv1d call in the VITS graph:
1. Transpose input from `[B, C, T]` → `[B, T, 1, C]` (NHWC)
2. Call `conv1d()` from `_hifigan_graph.py`
3. Transpose output back from `[B, T, 1, C]` → `[B, C, T]`

Alternatively, write a thin wrapper `conv1d_bct(x, w, b, ...)` that handles the transpose internally.

### Embedding Lookup

For pitch embedding `Embedding(256, 192)` with integer input:
- If MAX supports `ops.gather`: use directly
- If not: convert pitch to one-hot `[B, T, 256]` and matmul with weight `[256, 192]`

### Relative Position Attention

The relative position bias requires building an index matrix of shape `[T, T]` where `idx[q, k] = clip(q - k + window, 0, 2*window)`. Since T is symbolic:
- **Preferred:** Generate the relative position index matrix in numpy for a max-T, pass as graph constant
- **Fallback:** Compute relative attention entirely in numpy, pass attention output as graph input

### Sequence Mask

Generate in numpy before graph execution: `mask[b, :, :lengths[b]] = 1.0`. Pass as graph input `[B, 1, T]`.

### Task Dependencies

```
Task 1 (weight loader) ──→ Task 3 (flow) ──→ Task 6 (VoiceConverter)
Task 2 (PT reference)  ──→ Task 3, Task 4     ↗
                           Task 4 (enc_p) ───→
Task 1 ──→ Task 5 (HiFi cond) ──→ Task 6
Task 6 ──→ Task 7 (validation)
```

Tasks 1 and 2 can be done in parallel.
Tasks 3 and 4 can be done in parallel (after Task 1).
Task 5 depends on Task 1.
Task 6 depends on Tasks 3, 4, 5.
Task 7 depends on Task 6.
