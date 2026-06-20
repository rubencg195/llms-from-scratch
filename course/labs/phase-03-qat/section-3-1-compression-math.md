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

# Section 3.1: The Math of Compression — Min, Max, and Rounding

**Goal:** Manually quantize a float tensor to 4-bit integers and measure reconstruction error.

## What You Need to Know First

This kicks off Phase 3. It only needs a little high-school algebra and the tensor basics from earlier phases — no calculus yet.

- **Weights and tensors** — a model's learned numbers, stored as big grids of values (tensors). You met these in Phase 1.
- **Floating-point vs integers** — floats (like 0.0314) can store fractions but take more memory; integers (like 5) are exact whole numbers and take less.
- **Bits** — the storage budget per number. Fewer bits = smaller file but less precision. "4-bit" means only 16 possible values.
- **Rounding and min/max** — finding the smallest and largest values in a list and snapping each value to the nearest allowed step. That is the whole idea.
- **MSE (mean squared error)** — average of the squared differences; here it measures how far the rounded numbers drifted from the originals.

If you can find a min, a max, and round to the nearest step, you can follow this entire section.

## Why Smaller Numbers Matter

Modern LLMs have billions of parameters stored as 32-bit (or 16-bit) floating-point numbers. At inference time, the bottleneck is almost never compute — it's **memory bandwidth**. Every token generation requires reading the entire model's weights from memory.

| Factor | FP32 | FP16 | INT8 | INT4 |
|--------|------|------|------|------|
| Bits per weight | 32 | 16 | 8 | 4 |
| 7B model size | 28 GB | 14 GB | 7 GB | 3.5 GB |
| Memory BW needed | 1× | 0.5× | 0.25× | 0.125× |
| Fits on consumer GPU? | No (most) | Barely | Yes | Easily |
| Energy per inference | 1× | ~0.5× | ~0.25× | ~0.15× |

Quantization shrinks each weight from 32 bits to fewer bits by:
1. Finding the range of values [min, max]
2. Dividing that range into 2^n evenly-spaced "bins"
3. Rounding each float to its nearest bin center

The trade-off is **reconstruction error** — the quantized weights don't exactly equal the originals. Our goal is to minimize this error while maximizing compression.

## Core Quantization: Scale and Zero-Point

```python
import torch

torch.manual_seed(42)
x = torch.randn(1000) * 0.5
qmin, qmax = -8, 7  # 4-bit signed: 2^4 = 16 levels

x_min, x_max = x.min(), x.max()
scale = (x_max - x_min) / (qmax - qmin)
zero_point = qmin - x_min / scale

def quantize(x, scale, zp, qmin, qmax):
    x_int = torch.round(x / scale + zp).clamp(qmin, qmax)
    return x_int

def dequantize(x_int, scale, zp):
    return (x_int - zp) * scale

x_int = quantize(x, scale, zero_point, qmin, qmax)
x_hat = dequantize(x_int, scale, zero_point)
mse = (x - x_hat).pow(2).mean()
print(f"MSE: {mse.item():.6f}")
print(f"unique int levels used: {x_int.unique().numel()} / 16 possible")
print(f"scale: {scale.item():.6f}, zero_point: {zero_point.item():.3f}")
```

## Visualization: Original vs Quantized Reconstruction

```python
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

fig, axes = plt.subplots(1, 3, figsize=(15, 4))

axes[0].hist(x.numpy(), bins=50, alpha=0.7, color="steelblue", edgecolor="black", linewidth=0.5)
axes[0].set_title("Original FP32 Values")
axes[0].set_xlabel("Value")
axes[0].set_ylabel("Count")

axes[1].hist(x_hat.numpy(), bins=50, alpha=0.7, color="coral", edgecolor="black", linewidth=0.5)
axes[1].set_title("Reconstructed (4-bit Quantized)")
axes[1].set_xlabel("Value")

error = (x - x_hat).numpy()
axes[2].hist(error, bins=50, alpha=0.7, color="seagreen", edgecolor="black", linewidth=0.5)
axes[2].set_title(f"Quantization Error (MSE={mse.item():.5f})")
axes[2].set_xlabel("Error")
axes[2].axvline(0, color="red", linestyle="--", alpha=0.5)

plt.tight_layout()
plt.savefig("quantization_histogram.png", dpi=100, bbox_inches="tight")
plt.close()
print("Saved quantization_histogram.png")
```

## Asymmetric vs Symmetric Quantization

There are two main approaches to choosing scale and zero-point:

**Asymmetric quantization** maps the actual [min, max] range to [qmin, qmax]. The zero-point is generally non-zero, meaning floating-point 0.0 doesn't map exactly to integer 0. This uses the full integer range but complicates computation.

**Symmetric quantization** forces zero_point = 0 and uses the same scale for positive and negative values. This maps [-absmax, +absmax] to [qmin, qmax]. It "wastes" some range if the distribution is asymmetric, but simplifies the math (especially for matmul).

```python
def quantize_symmetric(x, n_bits=4):
    """Symmetric quantization: zero maps to zero."""
    qmax = 2 ** (n_bits - 1) - 1
    qmin = -(2 ** (n_bits - 1))
    absmax = x.abs().max()
    scale = absmax / qmax
    x_int = torch.round(x / scale).clamp(qmin, qmax)
    x_hat = x_int * scale
    return x_hat, scale

def quantize_asymmetric(x, n_bits=4):
    """Asymmetric quantization: uses full range."""
    qmax = 2 ** (n_bits - 1) - 1
    qmin = -(2 ** (n_bits - 1))
    x_min, x_max = x.min(), x.max()
    scale = (x_max - x_min) / (qmax - qmin)
    zp = qmin - x_min / scale
    x_int = torch.round(x / scale + zp).clamp(qmin, qmax)
    x_hat = (x_int - zp) * scale
    return x_hat, scale

x_sym, scale_sym = quantize_symmetric(x)
x_asym, scale_asym = quantize_asymmetric(x)

mse_sym = (x - x_sym).pow(2).mean()
mse_asym = (x - x_asym).pow(2).mean()

print(f"Symmetric  MSE: {mse_sym.item():.6f} (scale={scale_sym.item():.6f})")
print(f"Asymmetric MSE: {mse_asym.item():.6f} (scale={scale_asym.item():.6f})")
print(f"Winner: {'Asymmetric' if mse_asym < mse_sym else 'Symmetric'}")
```

## Quantization Granularity: Per-Tensor, Per-Channel, Per-Group

The coarser the granularity, the fewer scale/zero-point values we store — but the higher the error. There are three common strategies:

```python
W = torch.randn(64, 128)  # Simulated linear weight matrix

def quantize_per_tensor(W, n_bits=4):
    """Single scale for entire tensor."""
    x_hat, _ = quantize_asymmetric(W.flatten(), n_bits)
    return x_hat.view_as(W)

def quantize_per_channel(W, n_bits=4):
    """One scale per output channel (row)."""
    qmax = 2 ** (n_bits - 1) - 1
    qmin = -(2 ** (n_bits - 1))
    rows = []
    for i in range(W.shape[0]):
        row = W[i]
        s = (row.max() - row.min()) / (qmax - qmin)
        z = qmin - row.min() / s
        qi = torch.round(row / s + z).clamp(qmin, qmax)
        rows.append((qi - z) * s)
    return torch.stack(rows)

def quantize_per_group(W, n_bits=4, group_size=32):
    """One scale per group of consecutive elements."""
    qmax = 2 ** (n_bits - 1) - 1
    qmin = -(2 ** (n_bits - 1))
    flat = W.flatten()
    n = flat.numel()
    pad = (group_size - n % group_size) % group_size
    if pad > 0:
        flat = torch.cat([flat, torch.zeros(pad)])
    groups = flat.view(-1, group_size)
    result = torch.zeros_like(groups)
    for i in range(groups.shape[0]):
        g = groups[i]
        s = (g.max() - g.min()).clamp(min=1e-8) / (qmax - qmin)
        z = qmin - g.min() / s
        qi = torch.round(g / s + z).clamp(qmin, qmax)
        result[i] = (qi - z) * s
    return result.flatten()[:n].view_as(W)

W_pt = quantize_per_tensor(W)
W_pc = quantize_per_channel(W)
W_pg = quantize_per_group(W, group_size=32)

mse_pt = (W - W_pt).pow(2).mean().item()
mse_pc = (W - W_pc).pow(2).mean().item()
mse_pg = (W - W_pg).pow(2).mean().item()

print(f"Per-tensor  MSE: {mse_pt:.6f}")
print(f"Per-channel MSE: {mse_pc:.6f}")
print(f"Per-group   MSE: {mse_pg:.6f} (group_size=32)")
print(f"\nBest → Worst: per-group < per-channel < per-tensor (usually)")
```

## Outlier Sensitivity Demo

A single outlier can destroy quantization quality for the entire tensor because it stretches the scale to cover an extreme range, leaving very few bins for the remaining values.

```python
x_bad = x.clone()
x_bad[0] = 50.0  # single outlier
x_min_bad, x_max_bad = x_bad.min(), x_bad.max()
scale_bad = (x_max_bad - x_min_bad) / (qmax - qmin)
zp_bad = qmin - x_min_bad / scale_bad
x_hat_bad = dequantize(quantize(x_bad, scale_bad, zp_bad, qmin, qmax), scale_bad, zp_bad)
mse_bad = (x_bad - x_hat_bad).pow(2).mean().item()
print(f"MSE without outlier: {mse.item():.6f}")
print(f"MSE with outlier:    {mse_bad:.6f}")
print(f"Degradation:         {mse_bad / mse.item():.1f}×")
```

## Comparing MSE Across Different Bit Widths

```python
bit_widths = [2, 3, 4, 8]
results = []

for bits in bit_widths:
    qmax_b = 2 ** (bits - 1) - 1
    qmin_b = -(2 ** (bits - 1))
    n_levels = 2 ** bits
    s = (x.max() - x.min()) / (qmax_b - qmin_b)
    z = qmin_b - x.min() / s
    x_q = torch.round(x / s + z).clamp(qmin_b, qmax_b)
    x_r = (x_q - z) * s
    mse_b = (x - x_r).pow(2).mean().item()
    max_err = (x - x_r).abs().max().item()
    results.append({"bits": bits, "levels": n_levels, "mse": mse_b, "max_error": max_err})

print(f"{'Bits':<6} {'Levels':<8} {'MSE':<12} {'Max Error':<12} {'Compression':<12}")
print("-" * 50)
for r in results:
    compression = f"{32 / r['bits']:.1f}×"
    print(f"{r['bits']:<6} {r['levels']:<8} {r['mse']:<12.6f} {r['max_error']:<12.6f} {compression:<12}")
```

## Per-Channel vs Per-Tensor on Weight Matrix

```python
scales = []
recon = []
for i in range(W.shape[0]):
    row = W[i]
    s = (row.max() - row.min()) / (qmax - qmin)
    z = qmin - row.min() / s
    qi = quantize(row, s, z, qmin, qmax)
    recon.append(dequantize(qi, s, z))
W_hat = torch.stack(recon)
print(f"per-channel MSE: {(W - W_hat).pow(2).mean().item():.6f}")
```

## Exercise: Implement 8-Bit Quantization and Compare MSE to 4-Bit

Implement 8-bit quantization (256 levels) and compare the reconstruction error against the 4-bit version on both a normal distribution and a distribution with outliers.

```python
def quantize_n_bit(x, n_bits):
    """General n-bit asymmetric quantization."""
    qmax = 2 ** (n_bits - 1) - 1
    qmin = -(2 ** (n_bits - 1))
    x_min, x_max = x.min(), x.max()
    scale = (x_max - x_min).clamp(min=1e-8) / (qmax - qmin)
    zp = qmin - x_min / scale
    x_int = torch.round(x / scale + zp).clamp(qmin, qmax)
    x_hat = (x_int - zp) * scale
    return x_hat

torch.manual_seed(123)
test_normal = torch.randn(5000) * 0.5
test_outlier = test_normal.clone()
test_outlier[:10] = torch.tensor([10.0, -10.0, 8.0, -8.0, 15.0, -15.0, 12.0, -12.0, 6.0, -6.0])

print("=== Normal Distribution ===")
for bits in [4, 8]:
    recon = quantize_n_bit(test_normal, bits)
    mse_val = (test_normal - recon).pow(2).mean().item()
    print(f"  {bits}-bit MSE: {mse_val:.8f}")

print("\n=== Distribution with Outliers ===")
for bits in [4, 8]:
    recon = quantize_n_bit(test_outlier, bits)
    mse_val = (test_outlier - recon).pow(2).mean().item()
    print(f"  {bits}-bit MSE: {mse_val:.8f}")

recon_4 = quantize_n_bit(test_normal, 4)
recon_8 = quantize_n_bit(test_normal, 8)
ratio = (test_normal - recon_4).pow(2).mean() / (test_normal - recon_8).pow(2).mean()
print(f"\n4-bit error is {ratio.item():.1f}× higher than 8-bit")
print(f"But 4-bit gives {8/4:.0f}× more compression than 8-bit")
```

## Where This Leads Next

So far we have quantized weights *after the fact* with plain NumPy-style math. Section 3.2 wraps this `quantize → dequantize` round-trip inside an `nn.Module` so it can run live during a model's forward pass — the first step toward training a model that expects to be quantized.

---

## Key Takeaway

Quantization is a **lossy compression** that trades reconstruction accuracy for memory savings. The critical variables are:

1. **Bit width** — fewer bits = more compression but higher MSE; 4-bit is the sweet spot for LLMs (8× compression with acceptable error)
2. **Granularity** — per-group > per-channel > per-tensor in accuracy; the cost is storing more scale/zero-point metadata
3. **Outliers** — a single extreme value can destroy quantization quality for thousands of normal values; this motivates clipping, rotation (Phase 5), and QAT (Phase 3)
4. **Symmetric vs asymmetric** — symmetric is simpler and faster for inference (no zero-point subtraction in matmul), but wastes range for skewed distributions

Phase 5 rotation attacks the outlier problem; QAT (next sections) teaches weights to *fit* the quantization grid during training so reconstruction error is minimal at inference time.

## Further Reading (Optional)

**Optional — you do NOT need these to continue. They are for curious students who want the original sources.**

- Nagel et al. (2021). *A White Paper on Neural Network Quantization*. arXiv:2106.08295.
- Jacob et al. (2018). *Quantization and Training of Neural Networks for Efficient Integer-Arithmetic-Only Inference*. CVPR.
