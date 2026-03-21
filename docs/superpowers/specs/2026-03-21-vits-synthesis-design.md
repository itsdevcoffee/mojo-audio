# VITS Synthesis Pipeline — Sprint 4 Design Spec

**Date:** 2026-03-21
**Sprint:** 4 — "Full VITS Synthesis"
**Goal:** Wire the remaining RVC v2 components so that ContentVec features + F0 → audio, end-to-end in MAX Graph. Combined with existing AudioEncoder, RMVPE, and HiFiGAN.

---

## 1. Inference Data Flow

The RVC v2 inference path (from Applio `synthesizers.py:206-243`):

```
Input: raw audio [1, N] @16kHz
  │
  ├─→ AudioEncoder (HuBERT/ContentVec)           ← DONE
  │     → features [1, T, 768]
  │     → interpolate 2x → [1, T*2, 768]
  │
  ├─→ RMVPE PitchExtractor                        ← DONE
  │     → f0 [1, T*2] Hz
  │     → quantized pitch [1, T*2] (long, 0-255)
  │
  ├─→ (Optional) FAISS index retrieval             ← SKIPPED (future roadmap)
  │     → blend features with speaker embeddings
  │
  ├─→ (Optional) Pitch protection blending          ← SKIPPED (future roadmap)
  │
  ▼
enc_p (TextEncoder)                                 ← NEW
  features [1, T*2, 768] + pitch [1, T*2]
  → prior mean m_p [1, 192, T*2]
  → prior log-var logs_p [1, 192, T*2]
  → mask x_mask [1, 1, T*2]
  │
  ▼
Reparameterize                                      ← NEW (trivial)
  z_p = (m_p + exp(logs_p) * randn * 0.66666) * x_mask
  → z_p [1, 192, T*2]
  │
  ▼
flow (ResidualCouplingBlock, reverse=True)           ← NEW
  z_p → z [1, 192, T*2]
  (speaker conditioning: baked emb_g(0) constant)
  │
  ▼
dec (NSF-HiFiGAN)                                   ← DONE
  z [1, 192, T*2] + f0 [1, T*2]
  → audio [1, T_audio]
```

---

## 2. Components to Build

### 2.1 TextEncoder (`enc_p`)

**Source:** `Applio/rvc/lib/algorithm/encoders.py:88-144`

Architecture:
1. `emb_phone`: Linear(768, 192) — project ContentVec features to hidden dim
2. `emb_pitch`: Embedding(256, 192) — learned pitch embedding, added to phone features
3. Scale by √192, LeakyReLU(0.1)
4. Transpose to [B, 192, T]
5. `encoder`: 6-layer transformer (see §2.1.1)
6. `proj`: Conv1d(192, 384, k=1) — project to mean + log-variance
7. Split into mean [B, 192, T] and log-var [B, 192, T]
8. Apply sequence mask

**RVC v2 default config** (from `synthesizers.py` constructor):
- `out_channels` = 192 (`inter_channels`)
- `hidden_channels` = 192
- `filter_channels` = 768
- `n_heads` = 2
- `n_layers` = 6
- `kernel_size` = 3
- `p_dropout` = 0 (inference)
- `embedding_dim` = 768 (`text_enc_hidden_dim`)

**Weights from checkpoint:** `enc_p.*` keys

#### 2.1.1 Encoder (Transformer)

**Source:** `Applio/rvc/lib/algorithm/encoders.py:11-85`

6 layers, each:
1. MultiHeadAttention(192, 192, n_heads=2, window_size=10) + residual + LayerNorm
2. FFN(192, 192, filter=768, kernel=3) + residual + LayerNorm

**MultiHeadAttention** (`attentions.py:6`):
- Conv1d projections for Q, K, V (192→192 each)
- Relative positional encoding with window_size=10
- heads_share=True (one set of relative position embeddings shared across heads)
- Conv1d output projection (192→192)
- No dropout at inference

**FFN** (`attentions.py:188`):
- Conv1d(192, 768, k=3, pad=1) → ReLU → Conv1d(768, 192, k=3, pad=1)
- No dropout at inference

**Note on existing code:** We already have multi-head attention in `AudioEncoder._from_weights`. The attention pattern here is similar but uses relative positional encoding instead of absolute. This is a new attention variant we need to implement.

### 2.2 ResidualCouplingBlock (`flow`)

**Source:** `Applio/rvc/lib/algorithm/residuals.py:103-178`

Architecture: 4 coupling flows, interleaved with Flip layers.
Total: 8 modules = [CouplingLayer, Flip, CouplingLayer, Flip, CouplingLayer, Flip, CouplingLayer, Flip]

At inference (reverse=True): iterate in reverse order.

**RVC v2 default config:**
- `channels` = 192 (`inter_channels`)
- `hidden_channels` = 192
- `kernel_size` = 5
- `dilation_rate` = 1
- `n_layers` = 3 (WaveNet layers per coupling)
- `n_flows` = 4
- `gin_channels` = 256

**Weights from checkpoint:** `flow.*` keys

#### 2.2.1 ResidualCouplingLayer (mean-only)

**Source:** `residuals.py:182-258`

RVC uses `mean_only=True`, which simplifies the coupling:
- Split input x into x0, x1 along channel dim (96 + 96)
- x0 → pre Conv1d(96, 192, k=1) → WaveNet(192, k=5, dil=1, n=3, gin=256) → post Conv1d(192, 96, k=1) → m
- Forward: x1 = m + x1 (additive coupling, no scaling since mean_only)
- **Reverse: x1 = x1 - m** (simple subtraction)
- Concat [x0, x1] → output

Since `mean_only=True`, `logs` is always zero, so `exp(logs) = 1`. The coupling is purely additive — no element-wise multiplication needed in the reverse pass.

#### 2.2.2 Flip

**Source:** `residuals.py:87-100`

Trivially flips channels: `torch.flip(x, [1])` — reverses the channel dimension.
At inference (reverse=True): same operation (flip is its own inverse).

#### 2.2.3 WaveNet

**Source:** `modules.py:5-117`

3-layer WaveNet with gated activation:
- Per layer i (dilation = 1^i = 1 for all, since dilation_rate=1):
  - `in_layers[i]`: Conv1d(192, 384, k=5, dil=1, pad=2) → produces [B, 384, T]
  - Add speaker conditioning: `cond_layer` Conv1d(256, 1152, k=1), slice 384 channels per layer
  - Gated activation: `tanh(x_in[:,:192] + g_l[:,:192]) * sigmoid(x_in[:,192:] + g_l[:,192:])`
  - `res_skip_layers[i]`: Conv1d(192, 384, k=1) for i < n-1, Conv1d(192, 192, k=1) for last
  - Split res_skip into residual (192) + skip (192), or just skip for last layer
  - x = (x + residual) * mask; output += skip
- Return output * mask

**Speaker conditioning:** The `cond_layer` is a single Conv1d(256, 384*3=1152, k=1) that produces conditioning for all 3 WaveNet layers at once. The output is sliced by layer index.

### 2.3 Speaker Embedding (baked)

`emb_g`: Embedding(n_speakers, 256). For single-speaker models, extract `emb_g.weight[0]` at load time as a constant `[256, 1]` tensor (unsqueezed for broadcasting along time).

This constant `g` is passed to:
- `flow` → each ResidualCouplingLayer's WaveNet `cond_layer`
- `dec` (HiFiGAN) → currently not implemented (HiFiGAN's `cond` layer). **Decision: add `g` conditioning to HiFiGAN in this sprint.**

### 2.4 VoiceConverter (orchestrator)

Public API that wires the full pipeline:

```python
class VoiceConverter:
    @classmethod
    def from_pretrained(cls, checkpoint_path, device="auto"):
        """Load all components from a single RVC .pth checkpoint."""
        ...

    def convert(self, audio, pitch_shift=0, sr=16000):
        """Convert audio to target voice.

        Args:
            audio: [B, N] float32 input audio at sr Hz
            pitch_shift: semitones to shift (0 = no shift)
            sr: input sample rate

        Returns:
            [B, T_audio] float32 converted audio at target sample rate
        """
        ...
```

Internal steps:
1. Resample to 16kHz if needed
2. AudioEncoder → features [B, T, 768]
3. RMVPE → f0 [B, T_frames] Hz
4. Interpolate features 2x to match f0 frame rate
5. Apply pitch shift to f0 (multiply by 2^(semitones/12))
6. Quantize f0 to pitch bins (0-255)
7. Build sequence mask from lengths
8. enc_p(features, pitch, lengths) → mean, logvar, mask
9. Sample z_p = (mean + exp(logvar) * randn * 0.66666) * mask
10. flow(z_p, mask, g, reverse=True) → z
11. HiFiGAN(z * mask, f0, g) → audio

---

## 3. MAX Graph Architecture

### 3.1 Graph Strategy

Two new MAX Graphs (following the pattern from AudioEncoder and HiFiGAN):

**Graph 1: `vits_encoder`** — enc_p
- Input: features [B, T, 768], pitch [B, T] (int), length [B]
- Output: mean [B, 192, T], logvar [B, 192, T], mask [B, 1, T]
- Contains: Linear projection, pitch embedding, 6 transformer layers, output projection

**Graph 2: `vits_flow`** — flow (reverse)
- Input: z_p [B, 192, T], mask [B, 1, T]
- Output: z [B, 192, T]
- Contains: 4 reverse coupling layers with WaveNet + Flip, baked speaker embedding
- Speaker conditioning `g` is baked as a constant in the graph

**Reparameterization** (z_p sampling) stays in numpy — it's one line of code with random sampling.

### 3.2 Op Requirements

New ops needed (not in existing MAX graph code):

| Op | Used in | Notes |
|----|---------|-------|
| Relative positional attention | enc_p transformer | New attention variant vs AudioEncoder's absolute pos |
| Embedding lookup (int→float) | enc_p pitch embedding | Can implement as gather or one-hot matmul |
| Sequence mask generation | enc_p | Can generate in numpy, pass as graph input |
| Channel flip | flow Flip layers | `ops.slice_tensor` to reverse channel dim |
| Channel split/concat | flow coupling layers | Split along channel dim into halves |
| Gated activation (tanh * sigmoid) | flow WaveNet | Already have tanh and sigmoid ops |

All conv1d operations use the existing im2col + matmul workaround.

### 3.3 Speaker Conditioning in HiFiGAN

The existing HiFiGAN graph does NOT apply speaker conditioning (`gin_channels`). The RVC decoder uses it:
```python
# In HiFiGANNSFGenerator.forward():
if g is not None:
    x = x + self.cond(g)  # cond = Conv1d(256, 512, k=1)
```

This is a single Conv1d(256, 512, k=1) applied after conv_pre, adding speaker identity to the signal. Since `g` is baked as a constant, this becomes a static bias addition — `cond(g)` is a fixed [1, 512, 1] vector broadcast across time.

**Implementation:** Bake `cond(g)` at weight load time and fold it into `conv_pre.bias`. No graph changes needed.

---

## 4. Weight Loading

### 4.1 Keys to Extract

From RVC `.pth` checkpoint (under `weight` key):

**enc_p** (key names shown after weight-norm reconstruction — see §4.2):
```
enc_p.emb_phone.weight              [192, 768]
enc_p.emb_phone.bias                [192]
enc_p.emb_pitch.weight              [256, 192]
enc_p.encoder.attn_layers.{i}.conv_q.weight  [192, 192, 1]  (i=0..5)
enc_p.encoder.attn_layers.{i}.conv_q.bias    [192]
enc_p.encoder.attn_layers.{i}.conv_k.weight  [192, 192, 1]
enc_p.encoder.attn_layers.{i}.conv_k.bias    [192]
enc_p.encoder.attn_layers.{i}.conv_v.weight  [192, 192, 1]
enc_p.encoder.attn_layers.{i}.conv_v.bias    [192]
enc_p.encoder.attn_layers.{i}.conv_o.weight  [192, 192, 1]
enc_p.encoder.attn_layers.{i}.conv_o.bias    [192]
enc_p.encoder.attn_layers.{i}.emb_rel_k      [1, 21, 96]   (n_heads_rel=1, 2*window+1, head_dim)
enc_p.encoder.attn_layers.{i}.emb_rel_v      [1, 21, 96]
enc_p.encoder.norm_layers_1.{i}.gamma        [192]
enc_p.encoder.norm_layers_1.{i}.beta         [192]
enc_p.encoder.ffn_layers.{i}.conv_1.weight   [768, 192, 3]
enc_p.encoder.ffn_layers.{i}.conv_1.bias     [768]
enc_p.encoder.ffn_layers.{i}.conv_2.weight   [192, 768, 3]
enc_p.encoder.ffn_layers.{i}.conv_2.bias     [192]
enc_p.encoder.norm_layers_2.{i}.gamma        [192]
enc_p.encoder.norm_layers_2.{i}.beta         [192]
enc_p.proj.weight                    [384, 192, 1]
enc_p.proj.bias                      [384]
```

**flow** (key names shown after weight-norm reconstruction — see §4.2):

Checkpoint stores weight-norm parameters as `.parametrizations.weight.original0` (weight_g)
and `.original1` (weight_v). The weight loader reconstructs these into plain `.weight` keys.

```
flow.flows.{f}.pre.weight           [192, 96, 1]      (f=0,2,4,6 — coupling layers only)
flow.flows.{f}.pre.bias             [192]
flow.flows.{f}.enc.in_layers.{l}.weight  [384, 192, 5]  (l=0..2, reconstructed from weight-norm)
flow.flows.{f}.enc.in_layers.{l}.bias    [384]
flow.flows.{f}.enc.res_skip_layers.{l}.weight  [384, 192, 1] for l<2, [192, 192, 1] for l=2
flow.flows.{f}.enc.res_skip_layers.{l}.bias    [384] for l<2, [192] for l=2
flow.flows.{f}.enc.cond_layer.weight [1152, 256, 1]     (reconstructed from weight-norm)
flow.flows.{f}.enc.cond_layer.bias   [1152]
flow.flows.{f}.post.weight          [96, 192, 1]
flow.flows.{f}.post.bias            [96]
```

**Speaker embedding:**
```
emb_g.weight                         [n_speakers, 256]
```

**HiFiGAN conditioning** (already loaded but unused — `upsample_initial_channel` is config-dependent, default 512):
```
dec.cond.weight                      [upsample_initial_channel, 256, 1]
dec.cond.bias                        [upsample_initial_channel]
```

### 4.2 Weight-Norm Reconstruction

Same pattern as HiFiGAN: detect `weight_v` + `weight_g` pairs, reconstruct:
```python
weight = weight_v * (weight_g / norm(weight_v))
```

The existing `reconstruct_weight_norm` function in `_hifigan_weight_loader.py` can be reused.

---

## 5. Validation Strategy

### 5.1 Component-Level (Deterministic)

For each component, freeze inputs and compare MAX vs PyTorch:

| Test | Input | Target |
|------|-------|--------|
| enc_p forward | Random features [1, 20, 768] + pitch [1, 20] | mean, logvar match < 1e-3 |
| Single ResidualCouplingLayer reverse | Frozen z_p [1, 192, 20] + mask | output match < 1e-3 |
| Full flow reverse | Frozen z_p [1, 192, 20] + mask + baked g | output match < 1e-3 |
| HiFiGAN with speaker cond | Frozen latents + f0 + baked g | output match < 1e-2 |

### 5.2 End-to-End (Perceptual)

1. Run Applio on a test WAV with a specific voice model
2. Run our VoiceConverter on the same WAV with the same model
3. Compare: spectrogram visual similarity, listen test
4. Acceptable: output sounds like the same voice conversion (harmonic source differences are expected)

### 5.3 PyTorch Reference

Extend `tests/_rvc_pytorch_reference.py` with:
- `run_enc_p_reference(checkpoint, features, pitch)` → mean, logvar
- `run_flow_reference(checkpoint, z_p, mask)` → z
- `run_full_vits_reference(checkpoint, features, f0)` → audio

---

## 6. What's NOT in Scope

| Item | Why | When |
|------|-----|------|
| FAISS index retrieval | Optional feature, CPU-side, bolt-on later | Post-Sprint 4 |
| Pitch protection blending | Quality refinement, not core pipeline | Post-Sprint 4 |
| Multi-speaker embedding | All current models are single-speaker | If needed |
| Posterior encoder (`enc_q`) | Training only | Never |
| Duration predictor | Not used in VC inference | Never |
| Batch > 1 for VITS graphs | Single-speaker inference is B=1 | Before Sprint 6 |

---

## 7. File Structure

```
src/models/voice_converter.py           — Public VoiceConverter API (orchestrator)
src/models/_vits_graph.py               — MAX graphs for enc_p + flow
src/models/_vits_weight_loader.py       — Extract enc_p.*, flow.*, emb_g.* weights
tests/test_vits.py                      — Component + integration tests
tests/_rvc_pytorch_reference.py         — Extended with enc_p + flow references
```

---

## 8. Risk Assessment

| Risk | Likelihood | Impact | Mitigation |
|------|-----------|--------|------------|
| Relative positional attention fails in MAX | Medium | High | Fall back to no relative pos (test quality impact) |
| WaveNet dilated convs hit same conv2d bugs | High | Low | im2col workaround already proven |
| Flow reverse numerical drift | Low | Medium | Component-level validation catches this early |
| Large graph compile time (enc_p + flow are big) | Medium | Low | Split into 2 separate graphs |
| Speaker conditioning in HiFiGAN changes output | Low | Low | Bake cond(g) into conv_pre.bias — no graph change |
