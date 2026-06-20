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

# Section 1.4: The Attention Mechanism — Math for "Which Words Matter?"

**Goal:** Implement causal multi-head self-attention with RoPE and a causal mask.

## What You Need to Know First

- **The dot product and softmax** (Section 0.2) — attention scores are dot products turned into weights by softmax.
- **`nn.Linear` layers** (Section 0.4) — Q, K, and V are produced by linear layers.
- **Embeddings** (Section 1.2) and **RoPE** (Section 1.3) — attention runs on position-aware token vectors.

Everything here was built up across Phase 0 and the earlier parts of Phase 1, so no outside knowledge is needed. New terms like *causal mask*, *self-attention*, and *residual connection* are explained inline. (A **causal mask** simply blocks each token from "looking ahead" at words that come later, so the model can only use the past to predict the next word.)

## Attention as a Database Query

The attention mechanism is best understood as a **soft database lookup**:

| Component | Database Analogy | What It Does |
|-----------|-----------------|--------------|
| **Query (Q)** | Your search query | "What am I looking for?" |
| **Key (K)** | Index/label on each entry | "What do I contain?" |
| **Value (V)** | The actual data stored | "Here's my content" |

Each token generates a Q, K, and V vector. The Query "asks a question," the Keys "advertise what they have," and the dot product Q·K determines how relevant each Key is. The Values then deliver content, weighted by relevance.

$$\text{Attention}(Q, K, V) = \text{softmax}\left(\frac{QK^T}{\sqrt{d_k}}\right) V$$

The $\sqrt{d_k}$ scaling prevents dot products from growing too large (which would make softmax saturate into a one-hot vector).

```python
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
import numpy as np

# Paste RoPE helpers from Section 1.3 or import from your module
def precompute_freqs(dim, seq_len, base=10000.0, device="cpu"):
    inv_freq = 1.0 / (base ** (torch.arange(0, dim, 2, device=device).float() / dim))
    t = torch.arange(seq_len, device=device).float()
    freqs = torch.outer(t, inv_freq)
    return freqs.cos(), freqs.sin()

def apply_rope(x, cos, sin):
    x1, x2 = x[..., 0::2], x[..., 1::2]
    cos = cos.unsqueeze(0).unsqueeze(0)
    sin = sin.unsqueeze(0).unsqueeze(0)
    rot1 = x1 * cos - x2 * sin
    rot2 = x1 * sin + x2 * cos
    return torch.stack((rot1, rot2), dim=-1).flatten(-2)
```

## Visualization: Attention Weight Heatmap

Before diving into the full implementation, let's visualize what attention looks like on a short sentence.

```python
torch.manual_seed(42)
tokens = ["The", "cat", "sat", "on", "the", "mat"]
seq_len = len(tokens)
d_k = 16

# Simulate Q and K (random for now — in practice they come from learned projections)
Q = torch.randn(seq_len, d_k)
K = torch.randn(seq_len, d_k)

# Compute raw attention scores
raw_scores = Q @ K.T / math.sqrt(d_k)

# Apply causal mask (token can only attend to itself and earlier tokens)
causal_mask = torch.triu(torch.ones(seq_len, seq_len), diagonal=1).bool()
raw_scores.masked_fill_(causal_mask, float("-inf"))

# Softmax to get attention weights
attn_weights = F.softmax(raw_scores, dim=-1)

plt.figure(figsize=(7, 6))
plt.imshow(attn_weights.detach().numpy(), cmap="Blues", vmin=0, vmax=1)
plt.colorbar(label="Attention Weight")
plt.xticks(range(seq_len), tokens, rotation=45, ha="right")
plt.yticks(range(seq_len), tokens)
plt.xlabel("Key (attending TO)")
plt.ylabel("Query (attending FROM)")
plt.title("Causal Self-Attention Weights\n(each row sums to 1, upper triangle masked)")
for i in range(seq_len):
    for j in range(seq_len):
        val = attn_weights[i, j].item()
        if val > 0.01:
            plt.text(j, i, f"{val:.2f}", ha="center", va="center", fontsize=8)
plt.tight_layout()
plt.savefig("attention_weights_demo.png", dpi=120)
plt.show()
print("Saved attention_weights_demo.png")
```

## Multi-Head Attention: Why Multiple Heads?

A single attention head learns **one type of relationship** (e.g., "attend to the previous word"). By using multiple heads in parallel, the model can simultaneously learn:
- Head 1: syntactic relationships (subject→verb)
- Head 2: coreference (pronoun→noun it refers to)
- Head 3: positional patterns (attend to adjacent tokens)
- Head 4: semantic similarity (attend to related concepts)

Each head operates independently with its own Q, K, V projections, then their outputs are concatenated and projected back.

## Multi-head attention module

```python
class CausalSelfAttention(nn.Module):
    def __init__(self, d_model, n_heads, block_size, dropout=0.1):
        super().__init__()
        assert d_model % n_heads == 0
        self.n_heads = n_heads
        self.head_dim = d_model // n_heads
        self.qkv = nn.Linear(d_model, 3 * d_model)
        self.proj = nn.Linear(d_model, d_model)
        self.dropout = nn.Dropout(dropout)
        self.block_size = block_size
        self.register_buffer(
            "mask",
            torch.tril(torch.ones(block_size, block_size)).view(1, 1, block_size, block_size),
        )

    def forward(self, x, cos, sin):
        B, T, C = x.shape
        qkv = self.qkv(x).reshape(B, T, 3, self.n_heads, self.head_dim).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]
        q = apply_rope(q, cos, sin)
        k = apply_rope(k, cos, sin)
        att = (q @ k.transpose(-2, -1)) / math.sqrt(self.head_dim)
        att = att.masked_fill(self.mask[:, :, :T, :T] == 0, float("-inf"))
        att = F.softmax(att, dim=-1)
        att = self.dropout(att)
        out = att @ v
        out = out.transpose(1, 2).contiguous().view(B, T, C)
        return self.proj(out)
```

## Test forward pass

```python
device = "cuda" if torch.cuda.is_available() else "cpu"
d_model, n_heads, T = 512, 8, 32
block_size = 256
x = torch.randn(2, T, d_model, device=device)
cos, sin = precompute_freqs(d_model // n_heads, block_size, device=device)
attn = CausalSelfAttention(d_model, n_heads, block_size).to(device)
y = attn(x, cos[:T], sin[:T])
print("output:", y.shape)
```

## Putting It Together: The Transformer Block (Pre-LN Residual)

A Transformer block combines attention with a feedforward network (MLP), connected via **residual connections** and **layer normalization**. Modern LLMs use "Pre-LN" (normalize before the sublayer, not after):

```
x ──────────────────────────────────── + ──────────────────── + ── output
     │                                  ↑         │            ↑
     └→ LayerNorm → Attention ──────────┘         └→ LN → MLP ┘
```

**Why residual connections?** Without them, gradients vanish in deep networks. The residual path provides a "gradient highway" that lets information flow directly from early layers to late layers.

**Why Pre-LN?** It stabilizes training by normalizing inputs to each sublayer, allowing higher learning rates and faster convergence.

## Transformer block sketch

```python
class Block(nn.Module):
    def __init__(self, d_model, n_heads, block_size):
        super().__init__()
        self.ln1 = nn.LayerNorm(d_model)
        self.attn = CausalSelfAttention(d_model, n_heads, block_size)
        self.ln2 = nn.LayerNorm(d_model)
        self.mlp = nn.Sequential(
            nn.Linear(d_model, 4 * d_model),
            nn.GELU(),
            nn.Linear(4 * d_model, d_model),
        )

    def forward(self, x, cos, sin):
        x = x + self.attn(self.ln1(x), cos, sin)
        x = x + self.mlp(self.ln2(x))
        return x
```

## Verify: Attention Pattern Visualization

Let's generate an attention pattern from our real module and visualize which tokens attend to which, across multiple heads.

```python
# Run attention and capture the weights for visualization
class AttentionWithWeights(nn.Module):
    """Variant that returns attention weights for visualization."""
    def __init__(self, d_model, n_heads, block_size):
        super().__init__()
        self.n_heads = n_heads
        self.head_dim = d_model // n_heads
        self.qkv = nn.Linear(d_model, 3 * d_model)
        self.block_size = block_size
        mask = torch.tril(torch.ones(block_size, block_size)).view(1, 1, block_size, block_size)
        self.register_buffer("mask", mask)

    def forward(self, x, cos, sin):
        B, T, C = x.shape
        qkv = self.qkv(x).reshape(B, T, 3, self.n_heads, self.head_dim).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]
        q = apply_rope(q, cos, sin)
        k = apply_rope(k, cos, sin)
        att = (q @ k.transpose(-2, -1)) / math.sqrt(self.head_dim)
        att = att.masked_fill(self.mask[:, :, :T, :T] == 0, float("-inf"))
        att = F.softmax(att, dim=-1)
        return att  # (B, H, T, T)

torch.manual_seed(0)
T_demo = 8
d_demo = 64
n_heads_demo = 4
x_demo = torch.randn(1, T_demo, d_demo)
cos_demo, sin_demo = precompute_freqs(d_demo // n_heads_demo, T_demo)
attn_viz = AttentionWithWeights(d_demo, n_heads_demo, T_demo)
weights = attn_viz(x_demo, cos_demo, sin_demo)  # (1, 4, 8, 8)

fig, axes = plt.subplots(1, 4, figsize=(16, 4))
token_labels = [f"t{i}" for i in range(T_demo)]

for h in range(n_heads_demo):
    ax = axes[h]
    w = weights[0, h].detach().numpy()
    ax.imshow(w, cmap="Blues", vmin=0, vmax=w.max())
    ax.set_xticks(range(T_demo))
    ax.set_yticks(range(T_demo))
    ax.set_xticklabels(token_labels, fontsize=8)
    ax.set_yticklabels(token_labels, fontsize=8)
    ax.set_title(f"Head {h}", fontsize=11)
    ax.set_xlabel("Key")
    if h == 0:
        ax.set_ylabel("Query")

plt.suptitle("Multi-Head Attention Patterns (each head learns different relationships)", y=1.02)
plt.tight_layout()
plt.savefig("multihead_attention_patterns.png", dpi=120)
plt.show()
print("Saved multihead_attention_patterns.png")
print("Each head attends to different positions — this is the power of multi-head attention")
```

## Where This Leads Next

You now have the full Transformer block: attention + MLP + residuals. Section 1.5 (**the training loop**) stacks several of these blocks into a complete ~80M-parameter GPT and trains it on TinyStories using the very same forward → loss → backward → step loop you first met back in Section 0.3 — only now at full scale.

## Key Takeaway

- **Attention** is a soft database lookup: Query asks, Key advertises, Value delivers — weighted by Q·K relevance scores.
- **Causal masking** ensures autoregressive generation: each token can only attend to itself and earlier tokens (no peeking at the future).
- **Multi-head attention** runs multiple parallel attention operations, each learning different relationship types (syntax, semantics, position).
- **The Transformer block** = Pre-LayerNorm + Attention + Residual + Pre-LayerNorm + MLP + Residual — this pattern repeats 8–100+ times in modern LLMs.
- **Scaling by $\sqrt{d_k}$** prevents attention scores from becoming too large, keeping softmax gradients healthy.

Attention is the **context highlighter** — everything else in Phase 1 stacks these blocks.

## Checkpoint

You've built causal multi-head attention with RoPE. Next: **the training loop** (Section 1.5) — assembling the full GPT model and training on TinyStories.

## Further Reading (Optional)

**Optional — you do NOT need these to continue. They are for curious students who want the original sources.**

- Vaswani et al. (2017). *Attention Is All You Need*. NeurIPS.
- Bahdanau, Cho, & Bengio (2015). *Neural Machine Translation by Jointly Learning to Align and Translate*. ICLR.
- Dao et al. (2022). *FlashAttention: Fast and Memory-Efficient Exact Attention with IO-Awareness*. NeurIPS.
