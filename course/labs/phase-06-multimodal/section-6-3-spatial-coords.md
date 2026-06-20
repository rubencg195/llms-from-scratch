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

# Section 6.3: Spatial Coordinates — Teaching the Model Left vs. Right

**Goal:** Add learnable row/column embeddings to patch tokens before mixing with text.

## Without Position Info, the Model Sees a Bag of Patches, Not a Grid

Consider the caption "the cat is on the **left** side of the image." Without spatial
position information, every patch token is interchangeable — the model has no way to
distinguish the top-left patch from the bottom-right. The sequence becomes an unordered
*bag of patches*, destroying the very spatial structure that makes vision useful.

Position embeddings inject geometric structure back into the representation. They tell
the transformer: "This patch came from row 0, column 2" — enabling spatial reasoning,
object localization, and grounding of directional language.

We explore two approaches:
1. **Learnable embeddings** — separate row and column lookup tables (simple, effective)
2. **Fixed sinusoidal 2D embeddings** — no extra parameters, generalizes to unseen resolutions

---

## Approach 1: Learnable Row/Column Embeddings

```python
import torch
import torch.nn as nn

grid_h, grid_w = 3, 3  # 48/16
d_model = 512

row_emb = nn.Embedding(grid_h, d_model)
col_emb = nn.Embedding(grid_w, d_model)

def add_spatial_coords(patch_tokens):
    """patch_tokens: (N, d_model) row-major flatten"""
    n = patch_tokens.shape[0]
    rows = torch.arange(n) // grid_w
    cols = torch.arange(n) % grid_w
    return patch_tokens + row_emb(rows) + col_emb(cols)

x = torch.randn(9, d_model)
y = add_spatial_coords(x)
print("with spatial bias:", y.shape)
print("corner vs center diff:", (y[0] - y[4]).norm().item())
```

---

## Approach 2: Fixed Sinusoidal 2D Position Embeddings

Sinusoidal embeddings have a key advantage: they can be computed for *any* grid resolution
without retraining. This is the same idea as 1D sinusoidal embeddings from "Attention Is
All You Need," extended to two spatial dimensions.

```python
import numpy as np

def sinusoidal_2d_embeddings(grid_h, grid_w, d_model):
    """
    Generate fixed 2D sinusoidal position embeddings.
    Each position gets a d_model-dimensional vector encoding its (row, col).
    """
    assert d_model % 4 == 0, "d_model must be divisible by 4 for 2D sinusoidal"
    d_half = d_model // 2

    # Row embeddings
    row_pos = np.arange(grid_h)[:, np.newaxis]  # (H, 1)
    col_pos = np.arange(grid_w)[:, np.newaxis]  # (W, 1)

    dim = np.arange(d_half // 2)[np.newaxis, :]  # (1, d_half/2)
    freq = 1.0 / (10000 ** (2 * dim / d_half))

    row_sin = np.sin(row_pos * freq)  # (H, d_half/2)
    row_cos = np.cos(row_pos * freq)  # (H, d_half/2)
    col_sin = np.sin(col_pos * freq)  # (W, d_half/2)
    col_cos = np.cos(col_pos * freq)  # (W, d_half/2)

    # Combine: for each (r, c) pair, concat [row_sin, row_cos, col_sin, col_cos]
    embeddings = np.zeros((grid_h, grid_w, d_model))
    for r in range(grid_h):
        for c in range(grid_w):
            embeddings[r, c] = np.concatenate([
                row_sin[r], row_cos[r], col_sin[c], col_cos[c]
            ])

    return torch.tensor(embeddings, dtype=torch.float32).reshape(grid_h * grid_w, d_model)

sin_2d = sinusoidal_2d_embeddings(3, 3, d_model)
print("Sinusoidal 2D embeddings shape:", sin_2d.shape)
print("Norm of position 0:", sin_2d[0].norm().item())
print("Norm of position 8:", sin_2d[8].norm().item())
```

---

## Comparison: Learnable vs Sinusoidal

```python
# Similarity between positions using learnable embeddings
with torch.no_grad():
    learnable_embs = row_emb.weight[torch.arange(9) // 3] + col_emb.weight[torch.arange(9) % 3]
    learnable_sim = torch.cosine_similarity(
        learnable_embs.unsqueeze(0), learnable_embs.unsqueeze(1), dim=-1
    )

# Similarity between positions using sinusoidal embeddings
sin_sim = torch.cosine_similarity(
    sin_2d.unsqueeze(0), sin_2d.unsqueeze(1), dim=-1
)

print(f"Learnable position similarity (corners 0,8): {learnable_sim[0, 8].item():.4f}")
print(f"Sinusoidal position similarity (corners 0,8): {sin_sim[0, 8].item():.4f}")
print()
print(f"Learnable adjacent similarity (0,1): {learnable_sim[0, 1].item():.4f}")
print(f"Sinusoidal adjacent similarity (0,1): {sin_sim[0, 1].item():.4f}")
```

---

## Visualization: Position Embedding Similarity Matrix

This heatmap shows which positions the model considers "nearby." After training,
adjacent patches should have higher similarity than distant ones.

```python
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

fig, axes = plt.subplots(1, 2, figsize=(12, 5))

# Sinusoidal similarity matrix
im0 = axes[0].imshow(sin_sim.numpy(), cmap='RdBu_r', vmin=-1, vmax=1)
axes[0].set_title('Sinusoidal 2D Position Similarity')
axes[0].set_xlabel('Patch Index')
axes[0].set_ylabel('Patch Index')
labels = [f'({i//3},{i%3})' for i in range(9)]
axes[0].set_xticks(range(9))
axes[0].set_xticklabels(labels, fontsize=7, rotation=45)
axes[0].set_yticks(range(9))
axes[0].set_yticklabels(labels, fontsize=7)

# Learnable similarity (random init)
im1 = axes[1].imshow(learnable_sim.numpy(), cmap='RdBu_r', vmin=-1, vmax=1)
axes[1].set_title('Learnable Position Similarity (random init)')
axes[1].set_xlabel('Patch Index')
axes[1].set_ylabel('Patch Index')
axes[1].set_xticks(range(9))
axes[1].set_xticklabels(labels, fontsize=7, rotation=45)
axes[1].set_yticks(range(9))
axes[1].set_yticklabels(labels, fontsize=7)

plt.colorbar(im0, ax=axes[0], fraction=0.046)
plt.colorbar(im1, ax=axes[1], fraction=0.046)
plt.tight_layout()
plt.savefig('position_similarity.png', dpi=100, bbox_inches='tight')
plt.close()
print("Saved position_similarity.png")
print("Note: sinusoidal shows structured similarity; learnable is random before training")
```

---

## Interpolating Position Embeddings for Different Resolutions

A key practical question: what if at inference time we receive a 96×96 image (6×6 grid)
but trained on 48×48 (3×3 grid)? Sinusoidal embeddings handle this natively. For learnable
embeddings, we must *interpolate*.

```python
def interpolate_pos_embeddings(pos_emb, old_grid, new_grid):
    """
    Bilinearly interpolate learned position embeddings to a new grid size.
    pos_emb: (old_h * old_w, d_model)
    """
    old_h, old_w = old_grid
    new_h, new_w = new_grid
    d = pos_emb.shape[-1]

    # Reshape to spatial grid
    pos_grid = pos_emb.reshape(1, old_h, old_w, d).permute(0, 3, 1, 2)  # (1, d, h, w)

    # Bilinear interpolation
    interpolated = torch.nn.functional.interpolate(
        pos_grid, size=(new_h, new_w), mode='bilinear', align_corners=True
    )

    return interpolated.permute(0, 2, 3, 1).reshape(new_h * new_w, d)

# Interpolate our 3x3 sinusoidal embeddings to 6x6
new_pos = interpolate_pos_embeddings(sin_2d, (3, 3), (6, 6))
print(f"Interpolated from {sin_2d.shape} to {new_pos.shape}")
print(f"Original grid: 3×3 = 9 positions")
print(f"New grid: 6×6 = 36 positions")

# Verify smoothness: adjacent positions in new grid should be similar
adj_sim = torch.cosine_similarity(new_pos[0:1], new_pos[1:2], dim=-1)
diag_sim = torch.cosine_similarity(new_pos[0:1], new_pos[7:8], dim=-1)
print(f"Adjacent similarity in 6x6 grid: {adj_sim.item():.4f}")
print(f"Diagonal similarity in 6x6 grid: {diag_sim.item():.4f}")
```

---

## Normalized Coordinates (MLP Alternative)

Instead of discrete embeddings, we can map continuous normalized coordinates through an MLP.
This naturally handles any resolution without interpolation.

```python
def norm_coords(n):
    rows = (torch.arange(n) // grid_w).float() / (grid_h - 1)
    cols = (torch.arange(n) % grid_w).float() / (grid_w - 1)
    return torch.stack([rows, cols], dim=-1)

coord_mlp = nn.Linear(2, d_model)
coords = norm_coords(9)
bias = coord_mlp(coords)
print("MLP coord bias:", bias.shape)
print("Coordinate values (row, col):")
print(coords)
```

---

## Exercise: Verify Spatial Similarity After Training

Train the learnable position embeddings briefly on a spatial proximity objective and verify
that adjacent patches end up with more similar embeddings than distant patches.

```python
# Simple training: make nearby positions have similar embeddings
# Loss: adjacent pairs should have high cosine similarity

torch.manual_seed(42)
row_emb_train = nn.Embedding(grid_h, d_model)
col_emb_train = nn.Embedding(grid_w, d_model)
optimizer = torch.optim.Adam(
    list(row_emb_train.parameters()) + list(col_emb_train.parameters()), lr=0.01
)

def get_all_embeddings():
    rows = torch.arange(9) // grid_w
    cols = torch.arange(9) % grid_w
    return row_emb_train(rows) + col_emb_train(cols)

# Define adjacency: positions that differ by 1 in row or column
adjacent_pairs = []
for i in range(9):
    for j in range(i+1, 9):
        ri, ci = i // 3, i % 3
        rj, cj = j // 3, j % 3
        if abs(ri - rj) + abs(ci - cj) == 1:
            adjacent_pairs.append((i, j))

distant_pairs = [(0, 8), (2, 6), (0, 6), (2, 8)]  # corners
print(f"Adjacent pairs: {len(adjacent_pairs)}, Distant pairs: {len(distant_pairs)}")

for step in range(200):
    embs = get_all_embeddings()
    loss = torch.tensor(0.0)
    # Pull adjacent together
    for i, j in adjacent_pairs:
        sim = torch.cosine_similarity(embs[i:i+1], embs[j:j+1])
        loss = loss - sim.mean()
    # Push distant apart
    for i, j in distant_pairs:
        sim = torch.cosine_similarity(embs[i:i+1], embs[j:j+1])
        loss = loss + sim.mean()

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

# Check results
with torch.no_grad():
    final_embs = get_all_embeddings()
    adj_sims = [torch.cosine_similarity(final_embs[i:i+1], final_embs[j:j+1]).item()
                for i, j in adjacent_pairs]
    dist_sims = [torch.cosine_similarity(final_embs[i:i+1], final_embs[j:j+1]).item()
                 for i, j in distant_pairs]

print(f"\nAfter training:")
print(f"  Mean adjacent similarity:  {np.mean(adj_sims):.4f}")
print(f"  Mean distant similarity:   {np.mean(dist_sims):.4f}")
print(f"  Gap (should be positive):  {np.mean(adj_sims) - np.mean(dist_sims):.4f}")
```

---

## Key Takeaway

Spatial position embeddings transform an orderless bag of patch tokens into a structured
grid. The model learns that position (0,0) is "top-left" and position (2,2) is "bottom-right,"
enabling spatial reasoning in captions. Learnable embeddings are simple and effective for
fixed resolutions; sinusoidal embeddings generalize across resolutions without retraining.
Either way, the signal lets captions like "cat on the **left**" align with patch geometry.
