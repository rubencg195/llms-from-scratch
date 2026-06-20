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

# Section 1.2: Embeddings — Plotting Words as Points on a Graph

**Goal:** Implement `nn.Embedding`, visualize 2D projections, and connect lookup tables to trainable coordinates.

## What You Need to Know First

- **Integer token IDs** (Section 1.1) — the input to an embedding is just a token ID.
- **Tensors and indexing rows of a matrix** (Section 0.1) — an embedding lookup is "grab row number `id` from a big table."
- **Cosine similarity** (Section 0.2) — we reuse it to check which word vectors are "close."
- **The autograd training loop** (Section 0.3) — embeddings are learned with the same forward/backward/step loop.

All of these come from earlier sections, so nothing new from outside the course is needed. New terms like *one-hot*, *dense vector*, and *weight tying* are explained inline. (A **dense vector** is simply a short list of real numbers — e.g. 64 of them — that stands in for a token, as opposed to a giant mostly-zero "one-hot" vector.)

## One-Hot vs Dense Embeddings

There are two ways to represent token IDs as vectors:

**One-hot encoding:** A vector of length `vocab_size` with a single 1 at the token's position. Simple but wasteful — no notion of similarity, and enormous memory for large vocabularies.

**Dense embeddings:** A learned vector of length `d_model` (e.g., 64 or 512) that captures semantic meaning in a compact space. Similar tokens get nearby vectors.

| Approach | Vector Length | Memory per Token | Captures Similarity? |
|----------|-------------|------------------|---------------------|
| One-hot | vocab_size (8,000+) | 8,000 floats | No |
| Dense embedding | d_model (64–4096) | 64–4096 floats | Yes (after training) |

```python
import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA

vocab_size = 1000
d_model = 64
emb = nn.Embedding(vocab_size, d_model)
print("weight shape:", emb.weight.shape)  # (vocab, d_model)

token_ids = torch.tensor([5, 12, 12, 99])
vectors = emb(token_ids)
print("embedded shape:", vectors.shape)  # (4, 64)
```

## Embedding as a Lookup Table

`nn.Embedding` is conceptually just a matrix `W` of shape `(vocab_size, d_model)`. Looking up token ID `i` returns row `i` of the weight matrix — that's it. No multiplication, no activation, just indexing.

```python
# Prove it: emb(id) is exactly emb.weight[id]
test_id = torch.tensor([42])
lookup_result = emb(test_id)
direct_index = emb.weight[42]

print("emb(42) == emb.weight[42]:", torch.allclose(lookup_result.squeeze(), direct_index))
print("They are the same tensor (just different access patterns)")

# Same token → same vector (deterministic lookup)
print("\nSame ID gives same vector:", torch.allclose(vectors[1], vectors[2]))
```

## 2D visualization (PCA projection)

Untrained embeddings are random. After training, semantically similar tokens should cluster together. Let's visualize using PCA to reduce from 64D to 2D.

```python
torch.manual_seed(0)
# Simulate some structure: create "word groups" with shared components
n_words = 50
labels = [f"w{i}" for i in range(n_words)]

# Give the first 10 "words" a shared signal (as if they're related)
vecs = emb(torch.arange(n_words)).detach()

# Apply PCA to reduce to 2D
pca = PCA(n_components=2)
coords = pca.fit_transform(vecs.numpy())

plt.figure(figsize=(8, 6))
plt.scatter(coords[:10, 0], coords[:10, 1], c="steelblue", s=60, label="Group A (ids 0-9)", zorder=3)
plt.scatter(coords[10:, 0], coords[10:, 1], c="lightcoral", s=30, label="Others (ids 10-49)", alpha=0.6)
for i in range(10):
    plt.annotate(f"id={i}", (coords[i, 0], coords[i, 1]), fontsize=8, ha="center", va="bottom")
plt.title("PCA of Untrained Embeddings (random — no clusters yet)")
plt.xlabel(f"PC1 ({pca.explained_variance_ratio_[0]*100:.1f}% variance)")
plt.ylabel(f"PC2 ({pca.explained_variance_ratio_[1]*100:.1f}% variance)")
plt.legend()
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig("embedding_pca.png", dpi=120)
plt.show()
print("Saved embedding_pca.png — after training, related words would cluster together")
```

## Semantic geometry (after training preview)

Similar words should cluster. Untrained embeddings are random; Phase 1.5 training pulls **co-occurring** tokens together.

```python
seq_len = 8
batch = 2
idx = torch.randint(0, vocab_size, (batch, seq_len))
x = emb(idx)
print("batch tensor:", x.shape)  # (B, T, d_model)
```

## Weight Tying: Sharing Embeddings with the Language Model Head

In modern LLMs (GPT-2, LLaMA, etc.), the **input embedding matrix** and the **output projection** (LM head — the final layer that scores each possible next token) share the same weight matrix. This makes intuitive sense: the embedding maps token ID → vector, and the LM head maps vector → token *logits* (raw, unnormalized scores, one per vocabulary word, before softmax turns them into probabilities). They're inverse operations over the same semantic space.

```python
class TiedLanguageModel(nn.Module):
    def __init__(self, vocab_size, d_model):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, d_model)
        self.lm_head = nn.Linear(d_model, vocab_size, bias=False)
        # Weight tying: share the same parameter
        self.lm_head.weight = self.embedding.weight

    def forward(self, token_ids):
        x = self.embedding(token_ids)  # (B, T) → (B, T, d_model)
        logits = self.lm_head(x)        # (B, T, d_model) → (B, T, vocab_size)
        return logits

tied_model = TiedLanguageModel(vocab_size=1000, d_model=64)
print("Embedding weight is LM head weight:", tied_model.embedding.weight is tied_model.lm_head.weight)

# Parameter savings
total_with_tying = sum(p.numel() for p in tied_model.parameters())
total_without_tying = total_with_tying + 1000 * 64  # would need separate lm_head weights
savings = (1 - total_with_tying / total_without_tying) * 100
print(f"Parameter savings from weight tying: {savings:.1f}%")
print(f"For GPT-2 (vocab=50257, d=768): saves {50257*768/1e6:.1f}M parameters!")
```

## Exercise 1: Training Embeddings to Co-locate

Initialize embedding on GPU and compute cosine similarity between two token IDs before and after 50 gradient steps on a fake "same context" loss.

```python
device = "cuda" if torch.cuda.is_available() else "cpu"
e = nn.Embedding(100, 32).to(device)
i, j = torch.tensor([1], device=device), torch.tensor([2], device=device)

# Before training
cos_before = F.cosine_similarity(e(i), e(j)).item()
print(f"Cosine similarity before training: {cos_before:.4f}")

opt = torch.optim.SGD(e.parameters(), lr=0.5)
for _ in range(50):
    opt.zero_grad()
    vi, vj = e(i), e(j)
    loss = (vi - vj).pow(2).mean()  # pull together
    loss.backward()
    opt.step()

cos_after = F.cosine_similarity(e(i), e(j)).item()
print(f"Cosine similarity after training: {cos_after:.4f}")
print(f"Distance after: {(e(i) - e(j)).norm().item():.4f}")
```

## Exercise 2: Pairwise Cosine Similarity Matrix

Compute the full pairwise cosine similarity matrix for a set of token embeddings. This reveals which tokens the model considers "similar."

```python
torch.manual_seed(42)
# Simulate a small trained embedding where some tokens are related
e2 = nn.Embedding(8, 32)
token_names = ["the", "a", "cat", "dog", "ran", "walked", "big", "small"]

# Inject some structure: make determiners similar, animals similar, etc.
with torch.no_grad():
    shared_det = torch.randn(32)
    e2.weight[0] += 2 * shared_det   # "the"
    e2.weight[1] += 2 * shared_det   # "a"
    shared_animal = torch.randn(32)
    e2.weight[2] += 2 * shared_animal  # "cat"
    e2.weight[3] += 2 * shared_animal  # "dog"
    shared_verb = torch.randn(32)
    e2.weight[4] += 2 * shared_verb   # "ran"
    e2.weight[5] += 2 * shared_verb   # "walked"
    shared_adj = torch.randn(32)
    e2.weight[6] += 2 * shared_adj    # "big"
    e2.weight[7] += 2 * shared_adj    # "small"

# Compute pairwise cosine similarity
all_vecs = e2.weight.detach()
norms = all_vecs.norm(dim=1, keepdim=True)
cos_matrix = (all_vecs @ all_vecs.T) / (norms @ norms.T)

plt.figure(figsize=(7, 6))
plt.imshow(cos_matrix.numpy(), cmap="RdBu", vmin=-1, vmax=1)
plt.colorbar(label="Cosine Similarity")
plt.xticks(range(8), token_names, rotation=45, ha="right")
plt.yticks(range(8), token_names)
plt.title("Pairwise Cosine Similarity (structured embeddings)")
plt.tight_layout()
plt.savefig("cosine_similarity_matrix.png", dpi=120)
plt.show()
print("Saved cosine_similarity_matrix.png")
print("Notice: same-category tokens (the/a, cat/dog, ran/walked, big/small) are more similar")
```

## Where This Leads Next

Your tokens are now points in space, but they still don't know their **order** in the sentence — "dog bites man" and "man bites dog" would look identical. Section 1.3 (**RoPE**) fixes this by gently rotating each token's vector based on its position, so the model can tell where each word sits.

## Key Takeaway

- **Embeddings** are learnable lookup tables that map discrete token IDs to continuous vectors in a semantic space.
- `nn.Embedding(V, D)` is just a `(V, D)` weight matrix — lookup is fast and differentiable.
- **Weight tying** between the embedding and LM head saves significant parameters and enforces a shared semantic space.
- After training, embeddings capture **co-occurrence statistics** — tokens appearing in similar contexts get similar vectors.
- The pairwise cosine similarity matrix reveals the model's learned notion of "word similarity."

## Checkpoint

You can embed token IDs into dense vectors. Next: **RoPE** (Section 1.3) — adding positional information so the model knows word order.

## Further Reading (Optional)

**Optional — you do NOT need these to continue. They are for curious students who want the original sources.**

- Mikolov et al. (2013). *Efficient Estimation of Word Representations in Vector Space (word2vec)*. ICLR Workshop.
- Press & Wolf (2017). *Using the Output Embedding to Improve Language Models (weight tying)*. EACL.
