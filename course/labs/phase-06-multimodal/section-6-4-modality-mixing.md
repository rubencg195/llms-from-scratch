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

# Section 6.4: Modality Mixing — Stitching Image Tensors and Text Tensors Together

**Goal:** Build a single sequence `[IMG] patches... [TEXT] tokens...` and run through Phase 1 backbone.

## The Key Insight: Images and Text Live in the Same Vector Space

After projection, an image patch and a text token are both just vectors in $\mathbb{R}^{d}$.
The transformer doesn't know (or care) whether a vector came from a pixel patch or a word
embedding — it only sees a sequence of $d$-dimensional vectors and computes attention over
them.

This is the unification that makes multimodal models possible: by projecting different
modalities into the *same* embedding space, we let a single transformer reason over all of
them simultaneously. The modality embedding acts as a gentle "tag" that tells the model
which vectors came from which source, while the core computation remains modality-agnostic.

---

## Modality Indicators

How does the model know which tokens are image vs text? We use a **modality embedding** —
a small lookup table that adds a learned bias depending on the token's source:

- ID 0 → image patch
- ID 1 → text token

This is analogous to segment embeddings in BERT (sentence A vs sentence B).

```python
import torch
import torch.nn as nn
import torch.nn.functional as F

d_model = 512
vocab = 8000

tok_emb = nn.Embedding(vocab, d_model)
proj = nn.Linear(16 * 16 * 3, d_model)
modality_emb = nn.Embedding(2, d_model)  # 0=image, 1=text

def stitch_sequence(patch_vectors, token_ids):
    """
    patch_vectors: (P, d_model) — projected image patches
    token_ids: (T,) — text token indices
    Returns: (P+T, d_model) — unified multimodal sequence
    """
    text_vec = tok_emb(token_ids) + modality_emb(torch.ones(len(token_ids), dtype=torch.long))
    img_vec = patch_vectors + modality_emb(torch.zeros(len(patch_vectors), dtype=torch.long))
    return torch.cat([img_vec, text_vec], dim=0)

P = torch.randn(9, d_model)
T = torch.tensor([101, 242, 88, 5])  # fake caption ids
seq = stitch_sequence(P, T)
print("multimodal seq len:", seq.shape)  # (13, 512)
print("  9 image patches + 4 text tokens = 13 total")
```

---

## Causal Attention Over Mixed-Modal Sequence

The attention mask is causal: every token can attend to itself and all prior tokens.
Image patches attend to each other (bidirectional within the image prefix is also valid),
and text tokens attend to all patches plus prior text.

```python
T_total = seq.shape[0]
causal = torch.tril(torch.ones(T_total, T_total))
print("causal mask shape:", causal.shape)
print("text token at pos 12 can see all 13 positions:", causal[12].sum().item())
print("image patch at pos 0 can only see itself:", causal[0].sum().item())
```

---

## Full Multimodal Forward Pass (2-Layer Transformer)

Let's build a minimal multimodal transformer and run a complete forward pass:

```python
class MultimodalTransformer(nn.Module):
    def __init__(self, d_model, n_heads, n_layers, vocab_size):
        super().__init__()
        self.tok_emb = nn.Embedding(vocab_size, d_model)
        self.patch_proj = nn.Linear(16 * 16 * 3, d_model)
        self.modality_emb = nn.Embedding(2, d_model)
        self.pos_emb = nn.Embedding(256, d_model)

        self.layers = nn.ModuleList([
            nn.TransformerEncoderLayer(
                d_model=d_model, nhead=n_heads, dim_feedforward=4*d_model,
                dropout=0.0, activation='gelu', batch_first=True
            ) for _ in range(n_layers)
        ])
        self.ln_f = nn.LayerNorm(d_model)
        self.lm_head = nn.Linear(d_model, vocab_size, bias=False)

    def forward(self, patch_vectors, token_ids):
        """
        patch_vectors: (B, P, patch_dim)
        token_ids: (B, T)
        """
        B = patch_vectors.shape[0]
        P = patch_vectors.shape[1]
        T = token_ids.shape[1]

        img_emb = self.patch_proj(patch_vectors) + self.modality_emb(
            torch.zeros(B, P, dtype=torch.long))
        txt_emb = self.tok_emb(token_ids) + self.modality_emb(
            torch.ones(B, T, dtype=torch.long))

        seq = torch.cat([img_emb, txt_emb], dim=1)  # (B, P+T, d)
        seq = seq + self.pos_emb(torch.arange(P + T))

        # Causal mask
        S = P + T
        mask = torch.triu(torch.ones(S, S), diagonal=1).bool()

        for layer in self.layers:
            seq = layer(seq, src_mask=mask)

        seq = self.ln_f(seq)
        logits = self.lm_head(seq)
        return logits

model = MultimodalTransformer(d_model=512, n_heads=8, n_layers=2, vocab_size=8000)
print(f"Model parameters: {sum(p.numel() for p in model.parameters()) / 1e6:.1f}M")

# Forward pass
batch_patches = torch.randn(2, 9, 768)  # 2 images, 9 patches each
batch_tokens = torch.randint(0, 8000, (2, 20))  # 2 captions, 20 tokens each
logits = model(batch_patches, batch_tokens)
print(f"Output logits: {logits.shape}")  # (2, 29, 8000)
```

---

## Training Objective: Masked Loss on Text Only

We predict caption tokens autoregressively. Image patches use masked loss **0** — we never
ask the model to "predict" image patches from text (that would be image generation, a
different task). This is similar to the user-turn masking in Phase 2.

```python
def multimodal_loss(logits, token_ids, n_patches):
    """
    logits: (B, P+T, vocab)
    token_ids: (B, T)
    n_patches: int — number of image patches to skip
    """
    B, S, V = logits.shape
    T = token_ids.shape[1]

    # Only compute loss on text positions (shifted by 1 for next-token prediction)
    text_logits = logits[:, n_patches:-1, :]  # predict from patch_end to second-to-last
    text_targets = token_ids[:, 1:]           # shifted targets

    loss = F.cross_entropy(text_logits.reshape(-1, V), text_targets.reshape(-1))
    return loss

loss = multimodal_loss(logits, batch_tokens, n_patches=9)
print(f"Training loss: {loss.item():.4f}")
```

---

## Training Loop with Image-Caption Pairs

```python
torch.manual_seed(42)
optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)

# Synthetic training data: random patches + random captions
n_train_steps = 30
losses = []

for step in range(n_train_steps):
    patches = torch.randn(4, 9, 768)  # batch of 4 images
    captions = torch.randint(0, 8000, (4, 15))  # batch of 4 captions

    logits = model(patches, captions)
    loss = multimodal_loss(logits, captions, n_patches=9)

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    losses.append(loss.item())

    if step % 10 == 0:
        print(f"Step {step:3d} | Loss: {loss.item():.4f}")

print(f"Final loss: {losses[-1]:.4f} (started at {losses[0]:.4f})")
```

---

## Visualization: Attention Pattern Between Image and Text

Let's inspect which image patches the text tokens attend to most:

```python
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

# Get attention weights from a forward pass
model.eval()
patches_viz = torch.randn(1, 9, 768)
caption_viz = torch.randint(0, 8000, (1, 8))

# Manual forward to extract attention
with torch.no_grad():
    P = 9
    T_cap = 8
    img_emb = model.patch_proj(patches_viz) + model.modality_emb(torch.zeros(1, P, dtype=torch.long))
    txt_emb = model.tok_emb(caption_viz) + model.modality_emb(torch.ones(1, T_cap, dtype=torch.long))
    seq_viz = torch.cat([img_emb, txt_emb], dim=1)
    seq_viz = seq_viz + model.pos_emb(torch.arange(P + T_cap))

    # Compute attention scores manually for layer 0
    layer0 = model.layers[0]
    h = layer0.norm1(seq_viz)
    q = h @ layer0.self_attn.in_proj_weight[:512].T + layer0.self_attn.in_proj_bias[:512]
    k = h @ layer0.self_attn.in_proj_weight[512:1024].T + layer0.self_attn.in_proj_bias[512:1024]

    # Compute raw attention scores (simplified, single head view)
    attn_scores = (q @ k.transpose(-2, -1)) / (512 ** 0.5)
    # Apply causal mask
    S = P + T_cap
    mask = torch.triu(torch.ones(S, S), diagonal=1).bool()
    attn_scores.masked_fill_(mask.unsqueeze(0), float('-inf'))
    attn_weights = torch.softmax(attn_scores, dim=-1)

# Plot: how much text tokens attend to image patches
text_to_img_attn = attn_weights[0, P:, :P].numpy()  # (T, P)

fig, ax = plt.subplots(figsize=(8, 5))
im = ax.imshow(text_to_img_attn, cmap='Blues', aspect='auto')
ax.set_xlabel('Image Patch Index (3×3 grid)')
ax.set_ylabel('Text Token Position')
ax.set_title('Text → Image Attention Weights (Layer 0)')
ax.set_xticks(range(9))
ax.set_xticklabels([f'({i//3},{i%3})' for i in range(9)], fontsize=8)
plt.colorbar(im, ax=ax)
plt.tight_layout()
plt.savefig('attention_pattern.png', dpi=100, bbox_inches='tight')
plt.close()
print("Saved attention_pattern.png")
```

---

## Exercise: Generate a Caption Given Image Patches (Greedy Decode)

```python
@torch.no_grad()
def greedy_caption(model, patches, max_len=20, bos_token=1, eos_token=2):
    """
    Greedy autoregressive captioning.
    patches: (1, P, patch_dim)
    """
    model.eval()
    generated = [bos_token]

    for _ in range(max_len):
        token_ids = torch.tensor([generated]).long()
        logits = model(patches, token_ids)
        # Next token prediction from last position
        next_logits = logits[0, -1, :]
        next_token = next_logits.argmax().item()
        if next_token == eos_token:
            break
        generated.append(next_token)

    return generated

# Generate caption for random patches (will be random tokens since untrained on real data)
test_patches = torch.randn(1, 9, 768)
caption_ids = greedy_caption(model, test_patches)
print(f"Generated caption token IDs ({len(caption_ids)} tokens): {caption_ids[:10]}...")
print("(These are random since we trained on synthetic data — with real image-caption")
print(" pairs, this would produce meaningful captions)")
```

---

## Key Takeaway

Modality mixing is elegantly simple: project image patches and text tokens into the same
vector space, add modality indicators so the model knows which is which, concatenate them
into one sequence, and let the transformer's attention mechanism figure out cross-modal
relationships. The training loss applies *only* to text token predictions — the model
learns to "read" the image by learning that certain patch patterns predict certain words.
Fine-tune with TinyImage-Stories pairs `[image] → caption` and the same LLM that generates
text now generates text *about images*.
