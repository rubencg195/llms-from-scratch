---
jupytext:
  text_representation:
    extension: .md
    format_name: light
    format_version: '1.5'
kernelspec:
  display_name: Python 3 (LLMs)
  language: python
  name: python3
---

# Section 6.1: Why Separate Encoders Kill Your VRAM

**Goal:** Compare parameter and activation memory for ViT+LLM vs encoder-free patch pipeline.

## What You Need to Know First

Everything here builds on ideas you already met in Phases 1–5 — no outside knowledge needed:

- **Parameters and `nn.Linear`** — a layer's "weights" are just numbers the model learns. A linear layer holds `in × out + out` of them, which is exactly what we count below.
- **VRAM** — the memory on your GPU. Both the model's weights *and* the temporary tensors created while training have to fit inside it.
- **Embeddings and `d_model`** — every token is represented by a vector of length `d_model`; that shared width is the "language" all our components speak.

If those feel familiar, you're ready. We use only basic arithmetic and a little PyTorch.

## From Text-Only to Multimodal

Teaching LLMs to *see* is the next frontier in language modeling. A text-only model processes
the world through one narrow channel — strings of tokens. But language about the physical
world constantly references spatial relationships ("the cat *on* the left"), colors, textures,
and object identities that are expensive to describe in words but trivial to perceive visually.

The dominant approach in production multimodal LLMs (LLaVA, GPT-4V, Gemini) is to bolt a
**pre-trained Vision Transformer (ViT)** onto the LLM's input. (A ViT is just a transformer —
the same architecture you already know — that was trained on images instead of text.) This
works, but it comes at
an enormous VRAM cost — the encoder alone can exceed 300M parameters and require dedicated
activation memory. In this lab we quantify that cost and motivate the *encoder-free* design
we will build instead: a single linear patch projector that turns raw pixels into LLM-compatible
vectors with orders of magnitude fewer parameters.

---

## The ViT Architecture (Detailed Breakdown)

A Vision Transformer processes an image through three stages:

1. **Patch Embedding** — The image is split into fixed-size patches (typically 14×14 or 16×16).
   Each patch is linearly projected to the model dimension, producing one token per patch.
   A [CLS] token is prepended.

2. **Transformer Blocks** — Standard multi-head self-attention layers process the sequence
   of patch tokens. Position embeddings encode spatial location.

3. **Pooling / Projection** — Either the [CLS] token or mean-pool of all patches is projected
   to the final representation dimension.

The key insight: *all* those transformer layers exist solely to pre-process vision features
before handing them to the LLM. If we remove them, the LLM itself becomes the vision processor.

---

## Parameter Comparison: Encoders vs Linear Projector

```python
def count_linear_params(in_f, out_f):
    return in_f * out_f + out_f

d_model = 512

# ViT-B/16: 12 layers, hidden=768, heads=12, patch_size=16
vit_b_params = (
    count_linear_params(768, 768) * 4 * 12  # QKV+Out per layer
    + count_linear_params(768, 3072) * 12    # FFN up
    + count_linear_params(3072, 768) * 12    # FFN down
    + count_linear_params(768, d_model)      # final projection
)

# ViT-L/14: 24 layers, hidden=1024, heads=16
vit_l_params = (
    count_linear_params(1024, 1024) * 4 * 24
    + count_linear_params(1024, 4096) * 24
    + count_linear_params(4096, 1024) * 24
    + count_linear_params(1024, d_model)
)

# ViT-H/14: 32 layers, hidden=1280, heads=16
vit_h_params = (
    count_linear_params(1280, 1280) * 4 * 32
    + count_linear_params(1280, 5120) * 32
    + count_linear_params(5120, 1280) * 32
    + count_linear_params(1280, d_model)
)

# Encoder-free: one projection per patch
patch_dim = 16 * 16 * 3  # 768
projector_params = count_linear_params(patch_dim, d_model)

print("=" * 60)
print(f"{'Model':<20} {'Params (M)':<15} {'Ratio vs Projector'}")
print("-" * 60)
print(f"{'ViT-B/16':<20} {vit_b_params/1e6:<15.1f} {vit_b_params/projector_params:.0f}x")
print(f"{'ViT-L/14':<20} {vit_l_params/1e6:<15.1f} {vit_l_params/projector_params:.0f}x")
print(f"{'ViT-H/14':<20} {vit_h_params/1e6:<15.1f} {vit_h_params/projector_params:.0f}x")
print(f"{'Linear Projector':<20} {projector_params/1e6:<15.3f} {'1x'}")
print("=" * 60)
```

The table makes it clear: even the smallest ViT carries **~200x more parameters** than a
simple linear projector. On a 10 GB budget, those extra millions of parameters consume
memory that could be used for longer text context or larger batch sizes.

---

## Activation Memory During Training

Beyond parameter storage, the real VRAM killer during training is **activation memory** —
the intermediate tensors saved for backpropagation. Every attention layer in the encoder
stores Q, K, V projections and the attention matrix.

```python
import torch

B, T_text, T_patch = 2, 128, 9  # 48/16 = 3 patches per side -> 9 patches
d = 512

text_act = B * T_text * d * 4  # fp32 bytes
patch_act = B * T_patch * d * 4

# ViT-B activations: 197 patches (224/16=14 -> 14*14+1 CLS) * 12 layers
vit_seq = 197
vit_layers = 12
vit_act_per_layer = B * vit_seq * 768 * 4  # just hidden states
vit_attn_per_layer = B * 12 * vit_seq * vit_seq * 4  # attention matrices
vit_total_act = vit_layers * (vit_act_per_layer + vit_attn_per_layer)

print("Text activations:           ", f"{text_act / 1024:.1f} KB")
print("Patch activations (ours):   ", f"{patch_act / 1024:.1f} KB")
print("ViT activations (all layers):", f"{vit_total_act / (1024**2):.1f} MB")
print()
print(f"ViT uses {vit_total_act / patch_act:.0f}x more activation memory than our projector!")
```

## Gradient Checkpointing for Encoder is Expensive

When the ViT encoder is fine-tuned (not frozen), you must store activations for
backpropagation. **Gradient checkpointing** trades compute for memory by recomputing
activations during the backward pass rather than storing them. But even with checkpointing,
the encoder adds ~30-40% overhead because you still need at least one activation per
checkpoint boundary.

```python
def estimate_vram_mb(model_params_m, seq_len, d, batch, layers,
                     fp16=True, grad_ckpt=False):
    """Rough VRAM estimate in MB for training."""
    bytes_per_elem = 2 if fp16 else 4
    # Parameters + gradients + optimizer states (AdamW: 2 extra copies)
    param_bytes = model_params_m * 1e6 * bytes_per_elem
    optimizer_bytes = model_params_m * 1e6 * 4 * 2  # fp32 moments
    # Activations (simplified: one hidden state per layer + attention)
    if grad_ckpt:
        act_layers = layers ** 0.5  # sqrt(layers) checkpoints
    else:
        act_layers = layers
    act_bytes = batch * seq_len * d * bytes_per_elem * act_layers * 2
    total = param_bytes + optimizer_bytes + act_bytes
    return total / (1024 ** 2)

print("ViT-B training VRAM estimates:")
print(f"  No checkpointing:   {estimate_vram_mb(86, 197, 768, 2, 12):.0f} MB")
print(f"  With checkpointing: {estimate_vram_mb(86, 197, 768, 2, 12, grad_ckpt=True):.0f} MB")
print()
print("Our projector training VRAM:")
print(f"  Linear projector:    {estimate_vram_mb(0.4, 9, 512, 2, 1):.0f} MB")
```

---

## Visualization: VRAM Usage Comparison

```python
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

models = ['ViT-B + LLM\n(frozen enc)', 'ViT-B + LLM\n(finetuned)', 'ViT-L + LLM\n(frozen enc)', 'Our: Linear\nProjector + LLM']
# Approximate VRAM in MB for batch=2, fp16
params_mb = [86*2, 86*2, 304*2, 0.4*2]  # model params in fp16
kv_act_mb = [45, 180, 95, 5]  # activations
optimizer_mb = [0, 86*4*2, 0, 0.4*4*2]  # optimizer states (only if training encoder)
llm_base = 200  # our LLM base cost

# Convert optimizer to MB
optimizer_mb = [o / 1024 for o in optimizer_mb]

total = [p + a + o + llm_base for p, a, o in zip(params_mb, kv_act_mb, optimizer_mb)]

fig, ax = plt.subplots(figsize=(10, 5))
bars = ax.bar(models, total, color=['#e74c3c', '#c0392b', '#8e44ad', '#27ae60'], edgecolor='black')
ax.axhline(y=10000, color='red', linestyle='--', linewidth=2, label='10 GB budget')
ax.set_ylabel('Approximate VRAM (MB)')
ax.set_title('VRAM Budget: Encoder-Based vs Encoder-Free Multimodal')
ax.legend()

for bar, val in zip(bars, total):
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 20,
            f'{val:.0f} MB', ha='center', fontsize=10)

plt.tight_layout()
plt.savefig('vram_comparison.png', dpi=100, bbox_inches='tight')
plt.close()
print("Saved vram_comparison.png")
```

---

## Exercise: Compute VRAM Budget Remaining for Text

After allocating VRAM for the vision component, how much is left for text context?
Compute the remaining budget under both approaches:

```python
TOTAL_BUDGET_MB = 10_000  # 10 GB

# Our LLM: 8 layers, d=512, 8 heads
llm_params_mb = 30 * 2  # ~30M params in fp16
llm_optimizer_mb = 30 * 4 * 2  # AdamW fp32 moments

# Scenario 1: ViT-B encoder (frozen, so no optimizer for it)
vit_params_mb = 86 * 2  # fp16 params
vit_act_mb = 45  # frozen forward only

remaining_with_vit = TOTAL_BUDGET_MB - llm_params_mb - llm_optimizer_mb - vit_params_mb - vit_act_mb

# Scenario 2: Linear projector
proj_params_mb = 0.4 * 2
proj_act_mb = 0.1

remaining_with_proj = TOTAL_BUDGET_MB - llm_params_mb - llm_optimizer_mb - proj_params_mb - proj_act_mb

# How many text tokens can we fit in the remaining budget?
# KV cache per token: 2 * n_layers * n_heads * head_dim * 2 bytes (fp16)
kv_per_token_bytes = 2 * 8 * 8 * 64 * 2
max_tokens_vit = int((remaining_with_vit * 1024 * 1024) / kv_per_token_bytes)
max_tokens_proj = int((remaining_with_proj * 1024 * 1024) / kv_per_token_bytes)

print("VRAM remaining for text context:")
print(f"  With ViT encoder:     {remaining_with_vit:.0f} MB -> ~{max_tokens_vit:,} tokens")
print(f"  With linear projector: {remaining_with_proj:.0f} MB -> ~{max_tokens_proj:,} tokens")
print(f"  Extra tokens gained:   {max_tokens_proj - max_tokens_vit:,}")
```

---

## Where This Leads Next

Now that we've shown *why* a heavy vision encoder is too expensive, Section 6.2 builds the
lightweight alternative: a single linear "patch projector" that turns raw pixels into vectors
the LLM can read. The VRAM we just freed up is exactly what makes that encoder-free design possible.

## Key Takeaway

Encoder-free multimodal design is not about laziness — it is a **VRAM-conscious architectural
decision**. By replacing a ViT encoder (86M–632M params) with a single linear projection
(~0.4M params), we free up hundreds of megabytes of memory for longer text contexts, larger
batch sizes, or additional modalities. The LLM itself learns to interpret raw patch vectors,
making it both simpler and more memory-efficient. On a 10 GB budget, this is the difference
between a 2K context window and a 50K+ context window.

## Further Reading (Optional)

**Optional — you do NOT need these to continue. They are for curious students who want the original sources.**

- Dosovitskiy et al. (2020). *An Image is Worth 16x16 Words (ViT)*. ICLR.
- Radford et al. (2021). *Learning Transferable Visual Models From Natural Language Supervision (CLIP)*. ICML.
