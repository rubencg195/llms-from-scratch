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

# Section 1.3: RoPE — Rotating Word Vectors like Hands on a Clock

**Goal:** Implement Rotary Position Embeddings: apply 2D rotations to paired dimensions based on token position.

## What You Need to Know First

- **Embeddings as vectors** (Section 1.2) — RoPE adjusts the very vectors you just learned to build.
- **The dot product as similarity** (Section 0.2) — RoPE's key trick is what it does to dot products between tokens.
- **Tensor slicing and shapes** (Section 0.1) — we split vectors into pairs and recombine them.
- **High-school picture of rotating an arrow / sine and cosine** — just the intuition that spinning an arrow keeps its length but changes its direction.

These all come from earlier sections, so no outside knowledge is required. New terms like *positional encoding* and *relative position* are explained inline. (A **positional encoding** is extra information added to each token's vector that says *where* in the sentence the token appears.)

## Why Position Matters

Consider these two sentences:
- "**dog** bites **man**" — everyday event
- "**man** bites **dog**" — headline news!

The words are identical, but **position changes meaning completely**. Without position information, attention treats both sequences identically — it's just a bag of words. Position embeddings tell the model *where* each token sits in the sequence.

## Comparing Position Encoding Approaches

| Method | Type | Generalization | Used By |
|--------|------|----------------|---------|
| Sinusoidal (original Transformer) | Fixed, absolute | Moderate | Vaswani et al. 2017 |
| Learned absolute | Trained, absolute | Poor beyond training length | GPT-2 |
| ALiBi | Bias on attention scores | Good | BLOOM |
| **RoPE** | Rotation-based, relative | Excellent | LLaMA, Mistral, Gemma |

**Why RoPE wins:** It encodes **relative** position via rotation angles. The dot product between two rotated vectors depends only on their distance $(m - n)$, not their absolute positions. This means a model trained on length 2048 can generalize to longer sequences.

## Motivation

Absolute position embeddings don't generalize well. RoPE encodes **relative** distance via rotation angles.

For position $m$ and dimension pair $(2k, 2k+1)$:

$$\begin{pmatrix} x' \\ y' \end{pmatrix} = \begin{pmatrix} \cos m\theta_k & -\sin m\theta_k \\ \sin m\theta_k & \cos m\theta_k \end{pmatrix} \begin{pmatrix} x \\ y \end{pmatrix}$$

where $\theta_k = \text{base}^{-2k/d}$ (lower-frequency rotations for later dimensions).

```python
import torch
import math
import matplotlib.pyplot as plt
import numpy as np

def precompute_freqs(dim, seq_len, base=10000.0, device="cpu"):
    """dim must be even — rotate pairs."""
    inv_freq = 1.0 / (base ** (torch.arange(0, dim, 2, device=device).float() / dim))
    t = torch.arange(seq_len, device=device).float()
    freqs = torch.outer(t, inv_freq)  # (seq_len, dim/2)
    cos = freqs.cos()
    sin = freqs.sin()
    return cos, sin

def apply_rope(x, cos, sin):
    """x: (batch, heads, seq, head_dim)"""
    d = x.shape[-1]
    x1 = x[..., 0::2]
    x2 = x[..., 1::2]
    cos = cos.unsqueeze(0).unsqueeze(0)  # broadcast
    sin = sin.unsqueeze(0).unsqueeze(0)
    rot1 = x1 * cos - x2 * sin
    rot2 = x1 * sin + x2 * cos
    out = torch.stack((rot1, rot2), dim=-1).flatten(-2)
    return out

B, H, T, D = 2, 4, 16, 32
x = torch.randn(B, H, T, D)
cos, sin = precompute_freqs(D, T)
y = apply_rope(x, cos, sin)
print("RoPE output shape:", y.shape)
assert y.shape == x.shape
```

## Visualization: Rotation Angles Across Positions and Dimensions

Each dimension pair rotates at a different frequency — early dimensions rotate fast (capture local patterns), later dimensions rotate slowly (capture long-range structure).

```python
D_vis = 32
T_vis = 64
cos_vis, sin_vis = precompute_freqs(D_vis, T_vis)

# The rotation angle for position m, dimension pair k is: m * theta_k
inv_freq = 1.0 / (10000.0 ** (torch.arange(0, D_vis, 2).float() / D_vis))
positions = torch.arange(T_vis).float()
angles = torch.outer(positions, inv_freq)  # (T, D/2)

fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# Left: rotation angles as heatmap
im = axes[0].imshow(angles.numpy(), aspect="auto", cmap="twilight")
axes[0].set_xlabel("Dimension Pair Index (k)")
axes[0].set_ylabel("Position (m)")
axes[0].set_title("RoPE Rotation Angles (radians)")
plt.colorbar(im, ax=axes[0])

# Right: cos/sin for selected dimension pairs
for k in [0, 4, 8, 12, 15]:
    axes[1].plot(positions.numpy(), angles[:, k].numpy(), label=f"dim pair {k}", alpha=0.8)
axes[1].set_xlabel("Position")
axes[1].set_ylabel("Angle (radians)")
axes[1].set_title("Rotation Angle vs Position (per dimension pair)")
axes[1].legend(fontsize=9)
axes[1].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig("rope_angles.png", dpi=120)
plt.show()
print("Saved rope_angles.png")
print("Notice: dim pair 0 rotates fastest (local), pair 15 rotates slowest (long-range)")
```

## Relative distance property

Dot product between RoPE(q at m) and RoPE(k at n) depends on **m − n**, not absolute positions.

```python
q = torch.randn(1, 1, T, D)
k = torch.randn(1, 1, T, D)
cos, sin = precompute_freqs(D, T)
q_r = apply_rope(q, cos, sin)
k_r = apply_rope(k, cos, sin)
scores = (q_r @ k_r.transpose(-2, -1)) / math.sqrt(D)
print("attention scores shape:", scores.shape)
```

## Demonstrating Relative Distance with Numbers

Let's prove that the dot product after RoPE depends only on the **distance** between positions, not their absolute values.

```python
torch.manual_seed(0)
D_demo = 8  # small for clarity
q_vec = torch.randn(1, 1, 1, D_demo)  # a single query vector
k_vec = torch.randn(1, 1, 1, D_demo)  # a single key vector

# Compute dot product when q is at position 3, k is at position 7 (distance = 4)
T_max = 20
cos_all, sin_all = precompute_freqs(D_demo, T_max)

def rope_dot(q, k, pos_q, pos_k, cos_all, sin_all):
    """Compute dot product between q at pos_q and k at pos_k after RoPE."""
    cos_q = cos_all[pos_q:pos_q+1].unsqueeze(0).unsqueeze(0)
    sin_q = sin_all[pos_q:pos_q+1].unsqueeze(0).unsqueeze(0)
    cos_k = cos_all[pos_k:pos_k+1].unsqueeze(0).unsqueeze(0)
    sin_k = sin_all[pos_k:pos_k+1].unsqueeze(0).unsqueeze(0)

    q1, q2 = q[..., 0::2], q[..., 1::2]
    k1, k2 = k[..., 0::2], k[..., 1::2]

    q_rot = torch.cat([q1 * cos_q - q2 * sin_q, q1 * sin_q + q2 * cos_q], dim=-1)
    k_rot = torch.cat([k1 * cos_k - k2 * sin_k, k1 * sin_k + k2 * cos_k], dim=-1)

    return (q_rot * k_rot).sum().item()

# Same distance (4), different absolute positions
dot_3_7 = rope_dot(q_vec, k_vec, 3, 7, cos_all, sin_all)
dot_5_9 = rope_dot(q_vec, k_vec, 5, 9, cos_all, sin_all)
dot_10_14 = rope_dot(q_vec, k_vec, 10, 14, cos_all, sin_all)

# Different distance (2)
dot_3_5 = rope_dot(q_vec, k_vec, 3, 5, cos_all, sin_all)
dot_7_9 = rope_dot(q_vec, k_vec, 7, 9, cos_all, sin_all)

print("Same relative distance (4):")
print(f"  pos(3,7):   dot = {dot_3_7:.4f}")
print(f"  pos(5,9):   dot = {dot_5_9:.4f}")
print(f"  pos(10,14): dot = {dot_10_14:.4f}")
print(f"  → All equal! Position-invariant for same distance.")

print(f"\nSame relative distance (2):")
print(f"  pos(3,5): dot = {dot_3_5:.4f}")
print(f"  pos(7,9): dot = {dot_7_9:.4f}")
print(f"  → Also equal! Different from distance-4 values.")
```

## Integrate into attention (preview)

RoPE is applied to **Q** and **K** only, not **V** — standard in LLaMA, Mistral, Gemma families.

```python
print("RoPE ready for Section 1.4 MultiHeadAttention")
```

## Exercise: Verify RoPE Preserves Vector Norms

A rotation in 2D preserves the length (norm) of the vector — it only changes direction, not magnitude. Since RoPE applies independent 2D rotations to each dimension pair, it should preserve the overall vector norm.

```python
torch.manual_seed(42)
B_test, H_test, T_test, D_test = 4, 8, 32, 64
x_test = torch.randn(B_test, H_test, T_test, D_test)
cos_test, sin_test = precompute_freqs(D_test, T_test)
x_rotated = apply_rope(x_test, cos_test, sin_test)

# Compare norms before and after rotation
norm_before = x_test.norm(dim=-1)
norm_after = x_rotated.norm(dim=-1)

max_diff = (norm_before - norm_after).abs().max().item()
mean_diff = (norm_before - norm_after).abs().mean().item()

print(f"Norm before RoPE (sample): {norm_before[0, 0, :5].tolist()}")
print(f"Norm after RoPE (sample):  {norm_after[0, 0, :5].tolist()}")
print(f"Max absolute difference: {max_diff:.2e}")
print(f"Mean absolute difference: {mean_diff:.2e}")
print(f"Norms preserved: {max_diff < 1e-5}")
print("\n✓ RoPE is norm-preserving (it's a rotation, not a scaling)")
```

## Where This Leads Next

You now have position-aware Q and K vectors. Section 1.4 (**attention**) is where they finally get used: each token's query is compared (via dot product) against every other token's key to decide "which words should I pay attention to?" RoPE is what makes those comparisons aware of word order.

## Key Takeaway

- **Position information** is essential: without it, "dog bites man" and "man bites dog" are indistinguishable to attention.
- **RoPE** encodes position by **rotating** dimension pairs at different frequencies — fast rotations for local patterns, slow for long-range.
- The key property: dot products after RoPE depend only on **relative distance** $(m-n)$, not absolute positions — this enables length generalization.
- **Rotation preserves norms** — RoPE changes the direction of vectors but not their magnitude, keeping the attention score scale stable.
- RoPE is applied only to **Q** and **K** (not V) — it tells the attention mechanism "how far apart" tokens are.

## Checkpoint

You understand how RoPE encodes position through rotation. Next: **attention** (Section 1.4) — where Q, K, V come together to compute "which tokens matter."

## Further Reading (Optional)

**Optional — you do NOT need these to continue. They are for curious students who want the original sources.**

- Su et al. (2021). *RoFormer: Enhanced Transformer with Rotary Position Embedding*. arXiv:2104.09864.
- Vaswani et al. (2017). *Attention Is All You Need* (sinusoidal positions). NeurIPS.
