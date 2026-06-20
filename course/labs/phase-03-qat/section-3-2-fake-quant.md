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

# Section 3.2: Faking Low-Precision Math in PyTorch

**Goal:** Wrap `quantize → dequantize` in an `nn.Module` applied to activations and weights during forward pass.

## What You Need to Know First

This builds directly on Section 3.1 and the PyTorch module basics from Phase 1.

- **Quantize and dequantize** (Section 3.1) — round a float onto the integer grid, then map it back to a float that is *close* but not identical.
- **`nn.Module`** — PyTorch's standard wrapper for a reusable layer; it has a `forward()` method that runs when you call it. You used these in Phase 1.
- **Activations vs weights** — weights are the learned numbers stored in the model; activations are the temporary numbers that flow through it as data passes forward.
- **Calibration** — measuring typical value ranges over several batches so our scale/zero-point are stable instead of jumpy.
- **Outliers** — rare, extreme values that stretch the range and hurt quantization (introduced in Section 3.1).

No calculus is needed yet — that arrives in Section 3.3.

## "Fake" vs "Real" Quantization

There are two distinct stages of quantization in practice:

| Aspect | Fake Quantization (Training) | Real Quantization (Deployment) |
|--------|------------------------------|-------------------------------|
| **Precision** | All math in FP32 | Actual INT4/INT8 arithmetic |
| **Purpose** | Simulate quantization error during training | Achieve inference speedup |
| **Data type** | `torch.float32` throughout | `torch.int8` or packed INT4 |
| **Speed** | Slower (extra ops) | Faster (hardware INT units) |
| **Gradient** | Available (via STE) | N/A (inference only) |

During **QAT (Quantization-Aware Training)**, we use fake quantization: the weights and activations are quantized and immediately dequantized back to float. The tensor stays in FP32 memory, but its values are *constrained* to only the levels representable in low precision. This lets the model "feel" the rounding error during training and adapt its weights accordingly.

At deployment, we replace fake quant with real integer operations — but the weights are already optimized for it.

## The Basic FakeQuantize Module

```python
import torch
import torch.nn as nn

class FakeQuantize(nn.Module):
    def __init__(self, qmin=-8, qmax=7):
        super().__init__()
        self.qmin = qmin
        self.qmax = qmax

    def forward(self, x):
        min_val = x.min()
        max_val = x.max()
        scale = (max_val - min_val).clamp(min=1e-8) / (self.qmax - self.qmin)
        zp = self.qmin - min_val / scale
        x_int = torch.round(x / scale + zp).clamp(self.qmin, self.qmax)
        x_deq = (x_int - zp) * scale
        return x_deq

fq = FakeQuantize()
x = torch.randn(4, 64, requires_grad=True)
y = fq(x)
loss = y.sum()
loss.backward()
print("grad norm:", x.grad.norm().item())
```

## Calibration: Setting Scale and Zero-Point from Data

In the basic `FakeQuantize` above, we compute min/max fresh on every forward pass. This is unstable — batch statistics fluctuate wildly. **Calibration** collects statistics over multiple batches to find stable scale/zero-point values.

The key question: *how* do we summarize the distribution of values we've seen?

```python
class CalibratedFakeQuantize(nn.Module):
    """FakeQuantize with running min/max calibration."""
    def __init__(self, qmin=-8, qmax=7, momentum=0.1):
        super().__init__()
        self.qmin = qmin
        self.qmax = qmax
        self.momentum = momentum
        self.register_buffer("running_min", torch.tensor(float("inf")))
        self.register_buffer("running_max", torch.tensor(float("-inf")))
        self.register_buffer("calibrated", torch.tensor(False))

    def update_stats(self, x):
        """Update running statistics during calibration."""
        batch_min = x.detach().min()
        batch_max = x.detach().max()
        if not self.calibrated:
            self.running_min = batch_min
            self.running_max = batch_max
            self.calibrated.fill_(True)
        else:
            self.running_min = (1 - self.momentum) * self.running_min + self.momentum * batch_min
            self.running_max = (1 - self.momentum) * self.running_max + self.momentum * batch_max

    def forward(self, x):
        if self.training:
            self.update_stats(x)
        min_val = self.running_min if self.calibrated else x.min()
        max_val = self.running_max if self.calibrated else x.max()
        scale = (max_val - min_val).clamp(min=1e-8) / (self.qmax - self.qmin)
        zp = self.qmin - min_val / scale
        x_int = torch.round(x / scale + zp).clamp(self.qmin, self.qmax)
        return (x_int - zp) * scale

cfq = CalibratedFakeQuantize()
cfq.train()
for i in range(20):
    batch = torch.randn(32, 64) * (0.5 + 0.1 * i)
    _ = cfq(batch)

print(f"Calibrated range: [{cfq.running_min.item():.3f}, {cfq.running_max.item():.3f}]")
```

## Observer Patterns

Different observer strategies handle outliers differently:

| Observer | Method | Pros | Cons |
|----------|--------|------|------|
| **Min-Max** | Track global min/max | Simple, no clipping | Sensitive to outliers |
| **Moving Average** | EMA of batch min/max | Smooths noise | May lag behind distribution shifts |
| **Percentile** | Use Pth percentile | Robust to outliers | Clips extreme values |
| **MSE-optimal** | Minimize reconstruction MSE | Best accuracy | Expensive to compute |

```python
class PercentileObserver(nn.Module):
    """Observer that clips at specified percentiles to handle outliers."""
    def __init__(self, percentile=99.9):
        super().__init__()
        self.percentile = percentile
        self.register_buffer("observed_min", torch.tensor(float("inf")))
        self.register_buffer("observed_max", torch.tensor(float("-inf")))
        self.all_values = []

    def observe(self, x):
        """Collect values for percentile computation."""
        self.all_values.append(x.detach().flatten())

    def compute_range(self):
        """Compute percentile-clipped range from all observations."""
        all_data = torch.cat(self.all_values)
        sorted_data = torch.sort(all_data).values
        n = len(sorted_data)
        low_idx = int(n * (100 - self.percentile) / 100)
        high_idx = int(n * self.percentile / 100) - 1
        self.observed_min = sorted_data[low_idx]
        self.observed_max = sorted_data[high_idx]
        return self.observed_min, self.observed_max

observer = PercentileObserver(percentile=99.9)
for _ in range(10):
    data = torch.randn(100, 64)
    data[0, 0] = 100.0  # inject outlier each batch
    observer.observe(data)

pmin, pmax = observer.compute_range()
true_min = torch.cat(observer.all_values).min()
true_max = torch.cat(observer.all_values).max()
print(f"True range:       [{true_min.item():.2f}, {true_max.item():.2f}]")
print(f"Percentile range: [{pmin.item():.2f}, {pmax.item():.2f}]")
print(f"Outliers clipped: {true_max.item() - pmax.item():.1f} above, {pmin.item() - true_min.item():.1f} below")
```

## Quantized Linear Layer

```python
class QLinear(nn.Module):
    def __init__(self, in_f, out_f):
        super().__init__()
        self.weight = nn.Parameter(torch.randn(out_f, in_f) * 0.02)
        self.bias = nn.Parameter(torch.zeros(out_f))
        self.w_quant = FakeQuantize()
        self.a_quant = FakeQuantize()

    def forward(self, x):
        w_q = self.w_quant(self.weight)
        x_q = self.a_quant(x)
        return nn.functional.linear(x_q, w_q, self.bias)

layer = QLinear(128, 128)
inp = torch.randn(8, 128)
out = layer(inp)
print("output shape:", out.shape)
```

## Comparison: Quantizing a Pretrained vs Random Layer

A pretrained layer has weights concentrated in a narrow range (due to training dynamics), making it more quantization-friendly than random initialization.

```python
torch.manual_seed(42)
random_layer = nn.Linear(256, 256)
pretrained_layer = nn.Linear(256, 256)
nn.init.normal_(pretrained_layer.weight, mean=0, std=0.02)

fq_test = FakeQuantize(qmin=-8, qmax=7)

with torch.no_grad():
    random_q = fq_test(random_layer.weight)
    pretrained_q = fq_test(pretrained_layer.weight)

mse_random = (random_layer.weight - random_q).pow(2).mean().item()
mse_pretrained = (pretrained_layer.weight - pretrained_q).pow(2).mean().item()

print(f"Random init  weight std: {random_layer.weight.std().item():.4f}")
print(f"Pretrained   weight std: {pretrained_layer.weight.std().item():.4f}")
print(f"\nQuantization MSE (random):     {mse_random:.6f}")
print(f"Quantization MSE (pretrained): {mse_pretrained:.6f}")
print(f"Ratio: random is {mse_random / mse_pretrained:.1f}× worse")
```

## Visualization: Weight Distribution Before and After Fake Quantization

```python
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

torch.manual_seed(42)
weights = torch.randn(1000) * 0.02 + 0.001

fq_viz = FakeQuantize(qmin=-8, qmax=7)
with torch.no_grad():
    weights_q = fq_viz(weights)

fig, axes = plt.subplots(1, 2, figsize=(12, 4))

axes[0].hist(weights.numpy(), bins=60, alpha=0.7, color="steelblue", edgecolor="black", linewidth=0.3)
axes[0].set_title("Original Weights (FP32)")
axes[0].set_xlabel("Value")
axes[0].set_ylabel("Count")
axes[0].axvline(weights.mean().item(), color="red", linestyle="--", label=f"mean={weights.mean():.4f}")
axes[0].legend()

axes[1].hist(weights_q.numpy(), bins=60, alpha=0.7, color="coral", edgecolor="black", linewidth=0.3)
axes[1].set_title("After Fake Quantization (4-bit)")
axes[1].set_xlabel("Value")
unique_vals = weights_q.unique()
axes[1].axvline(weights_q.mean().item(), color="red", linestyle="--",
                label=f"mean={weights_q.mean():.4f}\n{len(unique_vals)} unique levels")
axes[1].legend()

plt.tight_layout()
plt.savefig("fake_quant_distribution.png", dpi=100, bbox_inches="tight")
plt.close()
print("Saved fake_quant_distribution.png")
print(f"Original unique values: {weights.unique().numel()}")
print(f"Quantized unique values: {unique_vals.numel()}")
```

## Memory Footprint Estimate

```python
def param_bytes(model, bits=32):
    n = sum(p.numel() for p in model.parameters())
    return n * bits / 8

dense = nn.Linear(512, 512)
print(f"FP32 MB: {param_bytes(dense, 32) / 1e6:.3f}")
print(f"INT8 equivalent MB: {param_bytes(dense, 8) / 1e6:.3f}")
print(f"INT4 equivalent MB: {param_bytes(dense, 4) / 1e6:.3f}")
print(f"\nCompression ratio (FP32 → INT4): {32 / 4:.0f}×")
```

Fake quant simulates INT4 while tensors remain FP32 in memory — real deployment packs bits (outside lab scope).

## Exercise: Implement a Percentile-Based Observer that Clips Outliers at 99.9th Percentile

Build a `FakeQuantize` module that uses percentile-based clipping instead of raw min/max. Compare its quantization error to standard min/max on data with outliers.

```python
class PercentileFakeQuantize(nn.Module):
    """FakeQuantize that clips to [p_low, p_high] before computing scale."""
    def __init__(self, qmin=-8, qmax=7, percentile=99.9):
        super().__init__()
        self.qmin = qmin
        self.qmax = qmax
        self.low_pct = (100.0 - percentile) / 100.0
        self.high_pct = percentile / 100.0

    def forward(self, x):
        flat = x.detach().flatten()
        sorted_vals = torch.sort(flat).values
        n = len(sorted_vals)
        low_val = sorted_vals[int(n * self.low_pct)]
        high_val = sorted_vals[int(n * self.high_pct) - 1]

        x_clamped = x.clamp(low_val, high_val)
        scale = (high_val - low_val).clamp(min=1e-8) / (self.qmax - self.qmin)
        zp = self.qmin - low_val / scale
        x_int = torch.round(x_clamped / scale + zp).clamp(self.qmin, self.qmax)
        return (x_int - zp) * scale

torch.manual_seed(7)
clean_data = torch.randn(2000) * 0.5
outlier_data = clean_data.clone()
outlier_data[:20] = torch.randn(20) * 10  # inject outliers

fq_minmax = FakeQuantize(qmin=-8, qmax=7)
fq_pct = PercentileFakeQuantize(qmin=-8, qmax=7, percentile=99.9)

recon_minmax = fq_minmax(outlier_data)
recon_pct = fq_pct(outlier_data)

mse_minmax = (outlier_data - recon_minmax).pow(2).mean().item()
mse_pct = (outlier_data - recon_pct).pow(2).mean().item()

print(f"Min-Max FakeQuant MSE:     {mse_minmax:.6f}")
print(f"Percentile FakeQuant MSE:  {mse_pct:.6f}")
print(f"Improvement: {mse_minmax / mse_pct:.2f}× lower error with percentile clipping")

recon_clean_mm = fq_minmax(clean_data)
recon_clean_pct = fq_pct(clean_data)
print(f"\nOn clean data (no outliers):")
print(f"  Min-Max MSE:     {(clean_data - recon_clean_mm).pow(2).mean().item():.6f}")
print(f"  Percentile MSE:  {(clean_data - recon_clean_pct).pow(2).mean().item():.6f}")
```

## Where This Leads Next

Our fake-quant module works in the forward pass, but there is a hidden problem: `round()` blocks gradients, so the model cannot learn through it. Section 3.3 introduces the **Straight-Through Estimator (STE)**, the gradient trick that makes fake quantization trainable.

---

## Key Takeaway

Fake quantization is the **training-time simulation** of low-precision inference. Key insights:

1. **Fake ≠ free** — it adds a quantize→dequantize round-trip to every forward pass, introducing rounding error that the model must learn to tolerate
2. **Calibration is critical** — computing scale/zero-point from a single batch is noisy; running statistics (EMA) or percentile-based methods produce more stable quantization parameters
3. **Observers matter** — min-max is simple but fails with outliers; percentile clipping at 99.9% can reduce MSE by 2-5× on realistic weight distributions
4. **Pretrained > random** — models already trained to convergence have weight distributions concentrated near zero, making them inherently more quantization-friendly
5. **FP32 in memory** — fake quant doesn't save memory during training; the savings come only at deployment when weights are actually stored as integers

The next section introduces the **Straight-Through Estimator (STE)** — the gradient trick that makes fake quantization differentiable so we can actually backpropagate through the rounding operation.

## Further Reading (Optional)

**Optional — you do NOT need these to continue. They are for curious students who want the original sources.**

- Jacob et al. (2018). *Quantization and Training of Neural Networks ...*. CVPR.
- Krishnamoorthi (2018). *Quantizing deep convolutional networks for efficient inference: A whitepaper*. arXiv:1806.08342.
