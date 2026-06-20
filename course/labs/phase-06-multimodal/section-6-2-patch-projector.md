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

# Section 6.2: Chopping Images into Algebra — The Linear Patch Projector

**Goal:** Split $48 \times 48$ RGB image into $16 \times 16$ patches and project each to `d_model`.

## What You Need to Know First

This section only assumes a few things you've already seen — no outside knowledge required:

- **From Section 6.1:** a "patch projector" is a single linear layer that replaces an expensive vision encoder, saving VRAM.
- **`nn.Linear`** — a layer that maps an input vector to an output vector by multiplying with learned weights; here it maps a flattened patch to a `d_model`-length vector.
- **Token embeddings and `d_model`** — text tokens are already vectors of length `d_model`; our goal is to make image patches into vectors of that same length so the LLM treats them identically.
- **An RGB image as numbers** — a picture is just a grid of pixels, each with 3 numbers (red, green, blue) between 0 and 255.

If that's familiar, you have everything you need.

## The Patch Projector is the "Retina"

In biological vision, the retina converts raw photon patterns into neural signals that the
brain can process. Our patch projector plays the same role: it transforms raw pixel intensities
into dense vectors that live in the same mathematical space as text token embeddings.

This is a powerful simplification. Instead of learning hierarchical visual features through
dozens of convolutional or attention layers (as ViT does), we trust that a *single linear
transformation* plus the LLM's own transformer layers will be sufficient. The LLM becomes
the visual cortex.

The pipeline: `Image → Patches → Normalize → Project → Ready for LLM`

---

## Patch Size Tradeoffs

The choice of patch size is a fundamental design decision:

| Patch Size | Tokens for 48×48 | Tokens for 224×224 | Detail Level | Compute Cost |
|-----------|------------------|--------------------|--------------|--------------|
| 8×8       | 36               | 784                | Very High    | Very High    |
| 16×16     | 9                | 196                | Medium       | Medium       |
| 32×32     | ~2 (pad needed)  | 49                 | Low          | Low          |

**Smaller patches = more tokens = more spatial detail** but exponentially more compute in
the LLM's self-attention (the "O(n²)" just means: double the number of tokens and the
attention work roughly quadruples). For our 10 GB budget with
48×48 images, 16×16 patches (9 tokens) is the sweet spot.

---

## Core Implementation

```python
import torch
import torch.nn as nn
import numpy as np

patch_size = 16
d_model = 512
patch_dim = patch_size * patch_size * 3

projector = nn.Linear(patch_dim, d_model)

def image_to_patches(img_array, patch_size=16):
    """img_array: (H, W, 3) uint8 -> patches: (N, patch_dim) float32"""
    H, W, C = img_array.shape
    assert H % patch_size == 0 and W % patch_size == 0
    ph, pw = H // patch_size, W // patch_size
    patches = img_array.reshape(ph, patch_size, pw, patch_size, C)
    patches = patches.transpose(0, 2, 1, 3, 4)
    patches = patches.reshape(ph * pw, patch_size * patch_size * C)
    return patches.astype(np.float32) / 255.0

# Synthetic 48x48 image with an orange blob
arr = np.zeros((48, 48, 3), dtype=np.uint8)
arr[16:32, 16:32] = [255, 128, 0]  # orange square in center
patches = image_to_patches(arr)
print("num patches:", patches.shape[0], "patch dim:", patches.shape[1])

tokens = projector(torch.from_numpy(patches))
print("projected tokens:", tokens.shape)  # (9, 512)
```

---

## Visualization: Image and Patch Grid

Let's visualize what the model actually "sees" — the original image split into a grid of
patches with borders drawn.

```python
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

fig, axes = plt.subplots(1, 3, figsize=(12, 4))

# Original image
axes[0].imshow(arr)
axes[0].set_title('Original 48×48 Image')
axes[0].axis('off')

# Image with patch grid overlay
axes[1].imshow(arr)
for i in range(1, 3):
    axes[1].axhline(y=i*16 - 0.5, color='red', linewidth=2)
    axes[1].axvline(x=i*16 - 0.5, color='red', linewidth=2)
axes[1].set_title('3×3 Patch Grid (16×16 each)')
axes[1].axis('off')

# Individual patches laid out
patch_imgs = patches.reshape(3, 3, 16, 16, 3)
grid_img = np.zeros((16*3 + 4, 16*3 + 4, 3))
for r in range(3):
    for c in range(3):
        y_start = r * (16 + 2)
        x_start = c * (16 + 2)
        grid_img[y_start:y_start+16, x_start:x_start+16] = patch_imgs[r, c]

axes[2].imshow(grid_img)
axes[2].set_title('Patches Separated (with gaps)')
axes[2].axis('off')

plt.tight_layout()
plt.savefig('patch_grid.png', dpi=100, bbox_inches='tight')
plt.close()
print("Saved patch_grid.png")
```

---

## Patch Normalization

Raw pixel values in [0, 1] have a skewed distribution. Normalizing with ImageNet-style
mean/std (or computed from our dataset) improves training stability by centering the input
distribution around zero with unit variance.

```python
# ImageNet normalization constants (per channel)
IMAGENET_MEAN = np.array([0.485, 0.456, 0.406])
IMAGENET_STD = np.array([0.229, 0.224, 0.225])

def normalize_patches(patches, mean=IMAGENET_MEAN, std=IMAGENET_STD):
    """
    patches: (N, patch_size*patch_size*3) in [0, 1]
    Applies per-channel normalization.
    """
    N, D = patches.shape
    ps = int((D // 3) ** 0.5)  # patch_size
    reshaped = patches.reshape(N, ps, ps, 3)
    normalized = (reshaped - mean) / std
    return normalized.reshape(N, D).astype(np.float32)

patches_norm = normalize_patches(patches)
print(f"Before normalization - mean: {patches.mean():.4f}, std: {patches.std():.4f}")
print(f"After normalization  - mean: {patches_norm.mean():.4f}, std: {patches_norm.std():.4f}")
```

---

## Full Pipeline: Image → Patches → Normalize → Project

```python
class PatchProjector(nn.Module):
    def __init__(self, patch_size, d_model, normalize=True):
        super().__init__()
        self.patch_size = patch_size
        self.patch_dim = patch_size * patch_size * 3
        self.proj = nn.Linear(self.patch_dim, d_model)
        self.normalize = normalize
        if normalize:
            self.register_buffer('mean', torch.tensor(IMAGENET_MEAN).float())
            self.register_buffer('std', torch.tensor(IMAGENET_STD).float())

    def patchify(self, img_tensor):
        """img_tensor: (B, 3, H, W) -> (B, N, patch_dim)"""
        B, C, H, W = img_tensor.shape
        ps = self.patch_size
        ph, pw = H // ps, W // ps
        x = img_tensor.reshape(B, C, ph, ps, pw, ps)
        x = x.permute(0, 2, 4, 3, 5, 1)  # (B, ph, pw, ps, ps, C)
        x = x.reshape(B, ph * pw, self.patch_dim)
        return x

    def forward(self, img_tensor):
        """img_tensor: (B, 3, H, W) in [0, 1] -> (B, N, d_model)"""
        patches = self.patchify(img_tensor)
        if self.normalize:
            B, N, D = patches.shape
            ps = self.patch_size
            patches = patches.reshape(B, N, ps, ps, 3)
            patches = (patches - self.mean) / self.std
            patches = patches.reshape(B, N, D)
        return self.proj(patches)

# Test the full pipeline
projector_module = PatchProjector(patch_size=16, d_model=512)
img_batch = torch.from_numpy(arr).permute(2, 0, 1).unsqueeze(0).float() / 255.0
print("Input image batch:", img_batch.shape)

output_tokens = projector_module(img_batch)
print("Output tokens:", output_tokens.shape)  # (1, 9, 512)
print("Ready for LLM input!")
```

---

## Optional: Load Real TinyImage-Stories Sample

```python
try:
    from datasets import load_dataset
    ds = load_dataset("chenghao/tiny_image_stories", split="train[:1]")
    print("dataset columns:", ds.column_names)
except Exception as e:
    print("Use synthetic image if dataset unavailable:", e)
```

---

## Exercise: Compare Different Patch Sizes

Try patchifying with different sizes and observe the token count / dimension tradeoff:

```python
print("Patch Size Comparison for 48×48 image:")
print(f"{'Patch Size':<12} {'Num Tokens':<12} {'Patch Dim':<12} {'Proj Params':<12}")
print("-" * 48)

for ps in [8, 16, 32]:
    if 48 % ps != 0:
        # Need to pad image for 32x32 patches (48/32 is not integer)
        effective_h = ((48 + ps - 1) // ps) * ps
        n_patches = (effective_h // ps) ** 2
    else:
        n_patches = (48 // ps) ** 2
    pdim = ps * ps * 3
    proj_params = pdim * d_model + d_model
    print(f"{ps}×{ps:<8} {n_patches:<12} {pdim:<12} {proj_params:,}")

print()
print("For a 224×224 image (standard ImageNet):")
print(f"{'Patch Size':<12} {'Num Tokens':<12} {'Attention Cost O(n²)'}")
print("-" * 48)
for ps in [8, 16, 32]:
    n = (224 // ps) ** 2
    cost = n * n
    print(f"{ps}×{ps:<8} {n:<12} {cost:,}")
```

---

## Where This Leads Next

Right now all 9 patch vectors are interchangeable — the model can't tell the top-left patch
from the bottom-right one. Section 6.3 fixes that by adding spatial position information so
the model knows *where* each patch came from in the image grid.

## Key Takeaway

The patch projector is the simplest possible bridge between pixel space and token space:
a single linear layer that maps flattened, normalized patches into the LLM's embedding
dimension. By choosing 16×16 patches for 48×48 images, we get exactly 9 visual tokens —
cheap enough to prepend to any text sequence without blowing our VRAM budget. Each patch
becomes one "token row" in the LLM sequence, treated identically to text embeddings from
that point forward.

## Further Reading (Optional)

**Optional — you do NOT need these to continue. They are for curious students who want the original sources.**

- Dosovitskiy et al. (2020). *An Image is Worth 16x16 Words (ViT)*. ICLR.
- Bavishi et al. (2023). *Fuyu-8B: A Multimodal Architecture for AI Agents*. Adept AI.
