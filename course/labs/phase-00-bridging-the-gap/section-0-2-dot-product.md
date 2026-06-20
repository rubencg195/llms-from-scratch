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

# Section 0.2: The Dot Product — Calculating Similarity with Basic Multiplication

**Goal:** Implement dot products manually and with `@` / `torch.matmul`, and connect them to cosine similarity.

## Definition

For vectors $\mathbf{a}$ and $\mathbf{b}$ of length $n$:

$$\mathbf{a} \cdot \mathbf{b} = \sum_{i=1}^{n} a_i b_i$$

Large dot product → vectors point in similar directions.

```python
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
import numpy as np

a = torch.tensor([1.0, 0.0, 0.0])
b = torch.tensor([0.9, 0.1, 0.0])
c = torch.tensor([0.0, 1.0, 0.0])

def dot_manual(x, y):
    return (x * y).sum()

print("a·b:", dot_manual(a, b))
print("a·c:", dot_manual(a, c))
```

## Geometric Interpretation

The dot product has a beautiful geometric meaning:

$$\mathbf{a} \cdot \mathbf{b} = \|\mathbf{a}\| \|\mathbf{b}\| \cos\theta$$

where $\theta$ is the angle between the vectors. This means:
- **Positive** dot product → vectors point in roughly the same direction ($\theta < 90°$)
- **Zero** dot product → vectors are orthogonal ($\theta = 90°$)
- **Negative** dot product → vectors point in opposite directions ($\theta > 90°$)

```python
# Visualize two 2D vectors and the angle between them
v1 = torch.tensor([3.0, 1.0])
v2 = torch.tensor([1.0, 2.5])

cos_angle = torch.dot(v1, v2) / (v1.norm() * v2.norm())
angle_rad = torch.acos(cos_angle)
angle_deg = angle_rad * 180 / np.pi

fig, ax = plt.subplots(1, 1, figsize=(6, 6))
origin = [0, 0]

ax.quiver(*origin, v1[0], v1[1], angles="xy", scale_units="xy", scale=1,
          color="steelblue", linewidth=2, label=f"v1 = [{v1[0]:.0f}, {v1[1]:.0f}]")
ax.quiver(*origin, v2[0], v2[1], angles="xy", scale_units="xy", scale=1,
          color="darkorange", linewidth=2, label=f"v2 = [{v2[0]:.0f}, {v2[1]:.0f}]")

# Draw the angle arc
theta_range = np.linspace(0, angle_rad.item(), 30)
arc_r = 0.8
ax.plot(arc_r * np.cos(np.arctan2(v1[1], v1[0]).item() - theta_range + angle_rad.item()),
        arc_r * np.sin(np.arctan2(v1[1], v1[0]).item() - theta_range + angle_rad.item()),
        "g--", alpha=0.6)

ax.set_xlim(-0.5, 4)
ax.set_ylim(-0.5, 3.5)
ax.set_aspect("equal")
ax.grid(True, alpha=0.3)
ax.axhline(0, color="k", linewidth=0.5)
ax.axvline(0, color="k", linewidth=0.5)
ax.set_title(f"Dot product = {torch.dot(v1, v2):.1f}, Angle = {angle_deg:.1f}°")
ax.legend(fontsize=11)
plt.tight_layout()
plt.savefig("dot_product_geometry.png", dpi=120)
plt.show()
print(f"Angle between vectors: {angle_deg:.1f}°")
```

## PyTorch built-ins

```python
print("torch.dot:", torch.dot(a, b))
print("@ operator:", a @ b)

# Batched dot products — shape (batch, features)
batch_a = torch.randn(8, 64)
batch_b = torch.randn(8, 64)
batched = (batch_a * batch_b).sum(dim=-1)
print("batched dots shape:", batched.shape)
```

## Matrix multiply = many dot products

If `A` is `(m, k)` and `B` is `(k, n)`, then `(A @ B)[i, j]` is the dot product of row $i$ of $A$ with column $j$ of $B$.

```python
A = torch.randn(4, 8)
B = torch.randn(8, 3)
C = A @ B
print("A @ B shape:", C.shape)
```

This is exactly what a **linear layer** does (Section 0.4).

## Dot Products in Neural Networks

The dot product is the most fundamental operation in deep learning. Here's where it appears:

**1. Linear layers:** `y = x @ W.T + b` — each output neuron computes a dot product with its weight vector.

**2. Attention scores:** In transformers, the "how much should token A attend to token B?" question is answered by a dot product: `score(A, B) = query_A · key_B`.

**3. Embedding similarity:** Finding the most similar word to a query embedding is a nearest-neighbor search using dot products.

```python
# Simulating a linear layer: dot product of input with each weight row
d_in, d_out = 64, 10
W = torch.randn(d_out, d_in)  # 10 neurons, each with 64 weights
x = torch.randn(d_in)          # one input vector

# Each output is a dot product
output = W @ x  # shape (10,) — same as [torch.dot(W[i], x) for i in range(10)]
print("Linear layer output shape:", output.shape)
print("Manual check:", torch.allclose(output[0], torch.dot(W[0], x)))
```

## Cosine similarity

Normalize vectors, then dot product → value in $[-1, 1]$.

```python
def cosine_sim(x, y, eps=1e-8):
    x_n = x / (x.norm() + eps)
    y_n = y / (y.norm() + eps)
    return (x_n * y_n).sum()

print("cosine(a, b):", cosine_sim(a, b))
print("cosine(a, c):", cosine_sim(a, c))
```

## Softmax: Turning Dot Products into Probabilities

In attention, raw dot product scores can be any real number. **Softmax** converts them into a probability distribution (non-negative, sums to 1):

$$\text{softmax}(z_i) = \frac{e^{z_i}}{\sum_j e^{z_j}}$$

```python
# Raw attention scores (dot products between query and 5 keys)
scores = torch.tensor([2.0, 1.0, 0.1, -1.0, 3.0])
print("Raw scores:", scores)

# Softmax converts to probabilities
weights = F.softmax(scores, dim=0)
print("Attention weights:", weights)
print("Sum:", weights.sum().item())  # always 1.0

# Temperature scaling: lower temp → sharper distribution
for temp in [0.5, 1.0, 2.0]:
    w = F.softmax(scores / temp, dim=0)
    print(f"  temp={temp}: {w.numpy().round(3)} (max={w.max():.3f})")
```

## Visualization: Attention Score Heatmap

In a real transformer, each token computes dot products with every other token to decide "who to attend to." Let's visualize this pattern.

```python
torch.manual_seed(42)
seq_len = 8
d_k = 16
tokens = ["The", "cat", "sat", "on", "the", "warm", "soft", "mat"]

Q = torch.randn(seq_len, d_k)
K = torch.randn(seq_len, d_k)

# Compute attention scores
attn_scores = Q @ K.T / (d_k ** 0.5)

# Apply causal mask (can't look into the future)
mask = torch.triu(torch.ones(seq_len, seq_len), diagonal=1).bool()
attn_scores.masked_fill_(mask, float("-inf"))

# Softmax to get weights
attn_weights = F.softmax(attn_scores, dim=-1)

plt.figure(figsize=(7, 6))
plt.imshow(attn_weights.detach().numpy(), cmap="Blues", vmin=0, vmax=1)
plt.colorbar(label="Attention Weight")
plt.xticks(range(seq_len), tokens, rotation=45, ha="right")
plt.yticks(range(seq_len), tokens)
plt.xlabel("Key (attending to)")
plt.ylabel("Query (attending from)")
plt.title("Causal Attention Weights (Dot Product + Softmax)")
plt.tight_layout()
plt.savefig("attention_heatmap_preview.png", dpi=120)
plt.show()
print("Saved attention_heatmap_preview.png")
```

## Exercise 1: Attention Scores

Given random query `q` shape `(64,)` and keys `K` shape `(128, 64)`, compute attention **scores** as dot products `q @ K.T` shape `(128,)`.

```python
torch.manual_seed(0)
q = torch.randn(64)
K = torch.randn(128, 64)
scores = q @ K.T
print("scores shape:", scores.shape)
print("top-3 key indices:", scores.topk(3).indices.tolist())
```

## Exercise 2: Word Similarity Search with Random Embeddings

Simulate a semantic search: given a vocabulary of "word embeddings," find the most similar word to a query using cosine similarity.

```python
torch.manual_seed(7)
vocab = ["king", "queen", "man", "woman", "prince", "princess", "boy", "girl",
         "apple", "banana", "car", "truck"]
d_emb = 32
embeddings = torch.randn(len(vocab), d_emb)

# Make related words more similar by adding shared components
royalty_signal = torch.randn(d_emb)
embeddings[0] += 2 * royalty_signal  # king
embeddings[1] += 2 * royalty_signal  # queen
embeddings[4] += 1.5 * royalty_signal  # prince
embeddings[5] += 1.5 * royalty_signal  # princess

# Query: "king"
query = embeddings[0]

# Compute cosine similarity with all words
norms = embeddings.norm(dim=1, keepdim=True)
cos_sims = (embeddings @ query) / (norms.squeeze() * query.norm())

# Show top-5 most similar
top5 = cos_sims.topk(5)
print("Most similar to 'king':")
for idx, sim in zip(top5.indices.tolist(), top5.values.tolist()):
    print(f"  {vocab[idx]:>10s}: cosine = {sim:.3f}")
```

## Key Takeaway

- The **dot product** measures how aligned two vectors are — it is the atomic similarity operation in all of deep learning.
- **Matrix multiplication** is just many dot products computed in parallel — it's what makes linear layers and attention efficient.
- **Cosine similarity** normalizes the dot product to $[-1, 1]$, removing magnitude and keeping only direction.
- **Softmax** converts raw dot-product scores into a probability distribution — this is how attention decides "how much to focus" on each token.
- Every time a transformer processes text, it computes billions of dot products. Understanding them geometrically gives you intuition for what the model is "thinking."

## Checkpoint

You understand dot products as similarity, matrix multiply as batched dots, and softmax as the "focus selector." Next: **autograd** (Section 0.3) — how PyTorch automatically computes gradients to learn.
