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

# Section 5.2: PolarQuant — Using Matrix Rotation to Smooth Outliers

**Goal:** Apply a random orthogonal rotation to KV vectors before quantization; compare reconstruction MSE with/without rotation.

> **Heads-up on the name:** "PolarQuant" is a teaching name we use in this course for the broader *rotate-then-quantize* family of methods (such as QuIP# and QuaRot); it is not a separate published algorithm.

## What You Need to Know First

This section adds one new idea — rotation — on top of things you already know. No outside knowledge needed:

- **The KV cache and quantization** from Section 5.1 — quantizing means snapping each number to one of a small set of allowed levels.
- **A vector and its "variance"** — variance just measures how spread out a dimension's values are; an *outlier* dimension is one that is spread out far more than the rest.
- **Matrix multiplication** — multiplying a vector by a matrix to get a new vector. A *rotation* is a special multiply that turns the data without changing any distances.
- **MSE (mean squared error)** — the average of the squared differences between the original values and the reconstructed ones; smaller MSE = better.

## The Outlier Problem — Visualized

Neural network activations are not uniformly distributed. In transformer KV caches, it's
common to find **outlier dimensions** — a single feature channel that has 10–100× the
variance of all others. This is catastrophic for quantization:

- Quantization maps a continuous range `[min, max]` to a discrete set of levels.
- If one dimension has range `[-100, 100]` while others have range `[-1, 1]`, the
  quantization grid is stretched by the outlier.
- The 99 "normal" dimensions get only a tiny fraction of the available levels, losing
  most of their information.

The insight of PolarQuant: if we **rotate** the vector space before quantizing, we
can spread the outlier's energy across all dimensions, making the distribution more
uniform and quantization more effective.

```python
import torch
import matplotlib.pyplot as plt
import numpy as np
```

## Demonstrating the Outlier Effect

```python
torch.manual_seed(0)
d = 64
x = torch.randn(1000, d)
x[:, 0] *= 20  # outlier dimension — 20x larger variance

fig, axes = plt.subplots(1, 2, figsize=(12, 4))
axes[0].hist(x[:, 0].numpy(), bins=50, alpha=0.7, color="#C44E52", label="Dim 0 (outlier)")
axes[0].hist(x[:, 1].numpy(), bins=50, alpha=0.7, color="#4C72B0", label="Dim 1 (normal)")
axes[0].set_title("Distribution of Two Dimensions")
axes[0].legend()
axes[0].set_xlabel("Value")

per_dim_std = x.std(dim=0).numpy()
axes[1].bar(range(d), per_dim_std, color=["#C44E52"] + ["#4C72B0"] * (d-1))
axes[1].set_xlabel("Dimension")
axes[1].set_ylabel("Std Dev")
axes[1].set_title("Per-Dimension Standard Deviation")

plt.tight_layout()
plt.show()

print(f"Dim 0 std: {per_dim_std[0]:.2f}, Mean of others: {per_dim_std[1:].mean():.2f}")
print(f"Outlier ratio: {per_dim_std[0] / per_dim_std[1:].mean():.1f}x")
```

## What Is an Orthogonal Matrix?

An orthogonal matrix $R$ is a square matrix where $R^T R = R R^T = I$. This means:

1. **Rotation preserves distances:** $\|Rx\| = \|x\|$ for any vector $x$.
2. **Dot products are unchanged:** $(Rx)^T(Ry) = x^T R^T R y = x^T y$.
3. **Information is perfectly preserved:** rotation is lossless and exactly invertible via $R^T$.

For quantization, this means: rotating before quantizing and de-rotating after dequantizing
introduces **no additional error** from the rotation itself — all error comes purely from
the quantization step, which now operates on a better-conditioned distribution.

```python
def random_orthogonal(d, device="cpu"):
    q, _ = torch.linalg.qr(torch.randn(d, d, device=device))
    return q

R = random_orthogonal(d)
print("R^T @ R ≈ I?", torch.allclose(R.T @ R, torch.eye(d), atol=1e-5))

x_sample = torch.randn(10, d)
norms_before = torch.norm(x_sample, dim=-1)
norms_after = torch.norm(x_sample @ R, dim=-1)
print("Norms preserved?", torch.allclose(norms_before, norms_after, atol=1e-5))

dots_before = (x_sample[:5] @ x_sample[5:].T)
dots_after = ((x_sample[:5] @ R) @ (x_sample[5:] @ R).T)
print("Dot products preserved?", torch.allclose(dots_before, dots_after, atol=1e-4))
```

## Visualization: 2D Scatter Before and After Rotation

Let's see what rotation does to a 2D slice of data with an outlier dimension.

```python
torch.manual_seed(42)
x_2d = torch.randn(500, 2)
x_2d[:, 0] *= 10  # outlier in dim 0

R_2d = random_orthogonal(2)
x_2d_rot = x_2d @ R_2d

fig, axes = plt.subplots(1, 2, figsize=(12, 5))

axes[0].scatter(x_2d[:, 0].numpy(), x_2d[:, 1].numpy(), alpha=0.3, s=10, c="#C44E52")
axes[0].set_title("Before Rotation (Outlier in Dim 0)")
axes[0].set_xlabel("Dim 0")
axes[0].set_ylabel("Dim 1")
axes[0].set_xlim(-35, 35)
axes[0].set_ylim(-35, 35)
axes[0].set_aspect("equal")
axes[0].axhline(0, color="gray", linewidth=0.5)
axes[0].axvline(0, color="gray", linewidth=0.5)

axes[1].scatter(x_2d_rot[:, 0].numpy(), x_2d_rot[:, 1].numpy(), alpha=0.3, s=10, c="#4C72B0")
axes[1].set_title("After Rotation (Energy Spread)")
axes[1].set_xlabel("Dim 0'")
axes[1].set_ylabel("Dim 1'")
axes[1].set_xlim(-35, 35)
axes[1].set_ylim(-35, 35)
axes[1].set_aspect("equal")
axes[1].axhline(0, color="gray", linewidth=0.5)
axes[1].axvline(0, color="gray", linewidth=0.5)

plt.suptitle("Rotation Distributes Outlier Energy Across Dimensions", fontsize=13, y=1.02)
plt.tight_layout()
plt.show()

print(f"Before rotation — Dim 0 std: {x_2d[:, 0].std():.2f}, Dim 1 std: {x_2d[:, 1].std():.2f}")
print(f"After rotation  — Dim 0 std: {x_2d_rot[:, 0].std():.2f}, Dim 1 std: {x_2d_rot[:, 1].std():.2f}")
```

## Why Rotation Helps Quantization

The core insight: uniform quantization allocates grid levels **equally across the entire
value range**. If one dimension stretches the range, all dimensions suffer.

After rotation:
- The outlier energy is **distributed** across all dimensions.
- Per-dimension variance becomes more uniform.
- The quantization grid covers the actual data distribution more efficiently.
- Result: **lower reconstruction MSE** at the same bit width.

```python
def rotate(x, R):
    return x @ R

def quantize_dequant(x, bits=4):
    qmin, qmax = -(2 ** (bits - 1)), 2 ** (bits - 1) - 1
    mn, mx = x.min(), x.max()
    s = (mx - mn).clamp(min=1e-8) / (qmax - qmin)
    zp = qmin - mn / s
    xi = torch.round(x / s + zp).clamp(qmin, qmax)
    return (xi - zp) * s

torch.manual_seed(0)
d = 64
x = torch.randn(1000, d)
x[:, 0] *= 20

R = random_orthogonal(d)
x_rot = rotate(x, R)

mse_plain = (x - quantize_dequant(x)).pow(2).mean()
x_back = rotate(quantize_dequant(x_rot), R.T)
mse_rot = (x - x_back).pow(2).mean()
print("MSE without rotation:", mse_plain.item())
print("MSE with PolarQuant rotation:", mse_rot.item())
print(f"Improvement: {(1 - mse_rot/mse_plain)*100:.1f}% lower MSE")
```

## Hadamard Matrices as Fast Alternatives

Random orthogonal matrices require $O(d^2)$ storage and $O(d^2)$ multiply per vector.
**Hadamard matrices** are structured orthogonal matrices with entries $\pm 1/\sqrt{d}$
that enable $O(d \log d)$ transforms via the Fast Walsh-Hadamard Transform (FWHT).

A Hadamard matrix $H_n$ of order $n = 2^k$ is constructed recursively:

$$H_1 = [1], \quad H_{2n} = \frac{1}{\sqrt{2}} \begin{bmatrix} H_n & H_n \\ H_n & -H_n \end{bmatrix}$$

```python
def hadamard_matrix(d):
    """Construct a normalized Hadamard matrix of size d (must be power of 2)."""
    assert d > 0 and (d & (d - 1)) == 0, "d must be a power of 2"
    H = torch.ones(1, 1)
    while H.shape[0] < d:
        H = torch.cat([
            torch.cat([H, H], dim=1),
            torch.cat([H, -H], dim=1),
        ], dim=0) / (2 ** 0.5)
    return H

H = hadamard_matrix(d)
print("Hadamard shape:", H.shape)
print("H^T @ H ≈ I?", torch.allclose(H.T @ H, torch.eye(d), atol=1e-5))
print("All entries ±1/√d?", torch.allclose(H.abs(), torch.ones_like(H) / (d ** 0.5), atol=1e-5))
```

## Exercise: Compare MSE — Hadamard vs Random Orthogonal

```python
torch.manual_seed(0)
x = torch.randn(1000, d)
x[:, 0] *= 20  # outlier

R_random = random_orthogonal(d)
H_norm = hadamard_matrix(d)

mse_no_rot = (x - quantize_dequant(x)).pow(2).mean().item()

x_rand_rot = x @ R_random
x_rand_back = quantize_dequant(x_rand_rot) @ R_random.T
mse_random = (x - x_rand_back).pow(2).mean().item()

x_had_rot = x @ H_norm
x_had_back = quantize_dequant(x_had_rot) @ H_norm.T
mse_hadamard = (x - x_had_back).pow(2).mean().item()

print(f"{'Method':<25} | {'MSE':>10} | {'vs No Rotation':>15}")
print("-" * 56)
print(f"{'No rotation':<25} | {mse_no_rot:>10.6f} | {'baseline':>15}")
print(f"{'Random orthogonal':<25} | {mse_random:>10.6f} | {(1 - mse_random/mse_no_rot)*100:>14.1f}%")
print(f"{'Hadamard':<25} | {mse_hadamard:>10.6f} | {(1 - mse_hadamard/mse_no_rot)*100:>14.1f}%")

fig, ax = plt.subplots(figsize=(8, 4))
methods = ["No Rotation", "Random\nOrthogonal", "Hadamard"]
mses = [mse_no_rot, mse_random, mse_hadamard]
bars = ax.bar(methods, mses, color=["#C44E52", "#4C72B0", "#55A868"])
for bar, mse in zip(bars, mses):
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.001,
            f"{mse:.4f}", ha="center", va="bottom", fontsize=10)
ax.set_ylabel("Reconstruction MSE")
ax.set_title("Quantization MSE: No Rotation vs Random Orthogonal vs Hadamard")
plt.tight_layout()
plt.show()
```

## Apply Per Head

```python
B, H, T, D = 2, 8, 128, 64
kv = torch.randn(B, H, T, D)
R = random_orthogonal(D, device=kv.device)
kv_rot = kv @ R
print("rotated KV shape:", kv_rot.shape)
```

Store $R$ once per layer/head — de-rotate after dequant when attending.

---

## Where This Leads Next

Rotation gives you a *well-behaved* distribution that quantizes cleanly. **Section 5.3** uses that to actually pack the values down to ~3.5 bits each — squeezing the KV cache about 4.5× smaller and turning the theory here into real memory savings.

## Key Takeaway

Outlier dimensions destroy uniform quantization by stretching the value range far beyond
what "normal" dimensions need. **PolarQuant** solves this by rotating the vector space with
an orthogonal matrix before quantizing — spreading outlier energy across all dimensions.
Since rotation preserves distances and dot products, the only error comes from the now
better-conditioned quantization step. Hadamard matrices provide a structured alternative
to random orthogonal matrices with $O(d \log d)$ transform cost instead of $O(d^2)$, while
achieving comparable MSE improvement.

## Further Reading (Optional)

**Optional — you do NOT need these to continue. They are for curious students who want the original sources.**

- Tseng et al. (2024). *QuIP#: Even Better LLM Quantization with Hadamard Incoherence and Lattice Codebooks*. ICML.
- Ashkboos et al. (2024). *QuaRot: Outlier-Free 4-Bit Inference in Rotated LLMs*. arXiv:2404.00456.
