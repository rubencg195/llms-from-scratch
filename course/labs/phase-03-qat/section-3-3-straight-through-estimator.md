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

# Section 3.3: The Straight-Through Estimator (Bypassing PyTorch's Rounding Block)

**Goal:** Implement STE so gradients flow through `round()` during QAT.

## What You Need to Know First

This section explains one clever trick. It builds on the fake-quant module from Section 3.2 plus the idea of a gradient from Phase 1.

- **Fake quantization** (Section 3.2) — the `quantize → dequantize` round-trip that includes a `round()` step.
- **Gradient** — a number telling each weight which way to nudge to lower the loss. "Zero gradient" means the weight gets no instruction and stops learning.
- **A staircase function** — `round()` is flat between steps and jumps suddenly; "flat" means slope zero, which is exactly the problem.
- **Forward vs backward pass** — forward computes the output; backward computes gradients. The STE simply uses a *different* rule for each.
- **`torch.autograd.Function`** — PyTorch's hook for defining custom forward and backward behavior.

The only "math" is recognizing that a flat line has zero slope, and pretending it has slope 1 instead.

## The Fundamental Problem: Rounding Has Zero Gradient

The `round()` function is a staircase — flat everywhere except at the half-integers where it jumps. Mathematically:

$$\frac{d}{dx}\text{round}(x) = 0 \quad \text{almost everywhere}$$

This means PyTorch's autograd computes **zero gradient** for any operation downstream of rounding. Without a workaround, backpropagation through quantize→dequantize produces no learning signal, and training stalls completely.

## Gradient Flow Diagram

The STE resolves this by using **different functions** for forward and backward passes:

| Pass | Operation | Formula | Gradient |
|------|-----------|---------|----------|
| **Forward** | Round to nearest integer | $y = \text{round}(x)$ | N/A (used for output) |
| **Backward** | Identity (pretend round didn't happen) | $\frac{\partial y}{\partial x} \approx 1$ | $\frac{\partial L}{\partial x} = \frac{\partial L}{\partial y}$ |

Visually: gradients "pass straight through" the rounding operation as if it were a wire.

```
Forward:  x ──→ [round()] ──→ y ──→ loss
                    ↓
Backward: ∂L/∂x ←── [identity] ←── ∂L/∂y ←── loss
```

## The STE Implementation

```python
import torch

class RoundSTE(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x):
        return torch.round(x)

    @staticmethod
    def backward(ctx, grad_output):
        return grad_output  # identity — pretend round was linear

round_ste = RoundSTE.apply

x = torch.randn(10, requires_grad=True)
y = round_ste(x)
loss = (y - torch.randn(10)).pow(2).mean()
loss.backward()
print("x grad (non-zero):", x.grad.abs().mean().item())
print("Gradient exists:", x.grad is not None and x.grad.abs().sum().item() > 0)
```

## Demonstrating the Problem: Without STE

Let's prove that standard `torch.round()` blocks all gradients:

```python
x_no_ste = torch.randn(10, requires_grad=True)
y_no_ste = torch.round(x_no_ste)
loss_no_ste = (y_no_ste - torch.randn(10)).pow(2).mean()
loss_no_ste.backward()
print("Without STE - grad:", x_no_ste.grad)
print("All zeros:", x_no_ste.grad.abs().sum().item() == 0.0)
```

## STE Inside Fake Quantization

```python
import torch.nn as nn

class FakeQuantSTE(nn.Module):
    def __init__(self, qmin=-8, qmax=7):
        super().__init__()
        self.qmin, self.qmax = qmin, qmax

    def forward(self, x):
        min_val, max_val = x.detach().min(), x.detach().max()
        scale = (max_val - min_val).clamp(min=1e-8) / (self.qmax - self.qmin)
        zp = self.qmin - min_val / scale
        x_scaled = x / scale + zp
        x_int = RoundSTE.apply(x_scaled).clamp(self.qmin, self.qmax)
        return (x_int - zp) * scale
```

## Full Comparison: STE vs No-STE Training (200 Steps Each)

This is the definitive test. We train two identical models on the same task — one using STE and one without. The STE model should converge; the no-STE model should stall.

```python
import torch.nn.functional as F

torch.manual_seed(42)

def train_with_ste(n_steps=200, lr=0.01):
    """Train with STE — gradients flow through rounding."""
    w = nn.Parameter(torch.randn(64, 64) * 0.1)
    fq = FakeQuantSTE()
    opt = torch.optim.Adam([w], lr=lr)
    target = torch.randn(64, 64) * 0.05
    losses = []
    for step in range(n_steps):
        opt.zero_grad()
        w_q = fq(w)
        loss = (w_q - target).pow(2).mean()
        loss.backward()
        opt.step()
        losses.append(loss.item())
    return losses

def fake_quant_no_ste(x, qmin=-8, qmax=7):
    """Quantize without STE — detach blocks gradients."""
    min_val, max_val = x.detach().min(), x.detach().max()
    scale = (max_val - min_val).clamp(min=1e-8) / (qmax - qmin)
    zp = qmin - min_val / scale
    x_int = torch.round(x / scale + zp).clamp(qmin, qmax)
    return ((x_int - zp) * scale).detach() + x - x.detach()  # still passes grad but via original x only

def train_without_ste(n_steps=200, lr=0.01):
    """Train without STE — use standard round() which zeros gradients."""
    w = nn.Parameter(torch.randn(64, 64) * 0.1)
    opt = torch.optim.Adam([w], lr=lr)
    target = torch.randn(64, 64) * 0.05
    losses = []
    for step in range(n_steps):
        opt.zero_grad()
        min_val, max_val = w.detach().min(), w.detach().max()
        scale = (max_val - min_val).clamp(min=1e-8) / (7 - (-8))
        zp = -8 - min_val / scale
        w_q = torch.round(w / scale + zp).clamp(-8, 7)
        w_q = (w_q - zp) * scale
        loss = (w_q - target).pow(2).mean()
        loss.backward()
        opt.step()
        losses.append(loss.item())
    return losses

losses_ste = train_with_ste(200)
losses_no_ste = train_without_ste(200)

print(f"STE — Initial loss: {losses_ste[0]:.4f}, Final loss: {losses_ste[-1]:.4f}")
print(f"No STE — Initial loss: {losses_no_ste[0]:.4f}, Final loss: {losses_no_ste[-1]:.4f}")
print(f"\nSTE reduction: {(1 - losses_ste[-1] / losses_ste[0]) * 100:.1f}%")
print(f"No-STE reduction: {(1 - losses_no_ste[-1] / losses_no_ste[0]) * 100:.1f}%")
```

## Visualization: Side-by-Side Loss Curves

```python
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

fig, axes = plt.subplots(1, 2, figsize=(14, 5))

axes[0].plot(losses_ste, color="green", linewidth=1.5, label="With STE")
axes[0].plot(losses_no_ste, color="red", linewidth=1.5, label="Without STE")
axes[0].set_xlabel("Training Step")
axes[0].set_ylabel("MSE Loss")
axes[0].set_title("STE vs No-STE: Loss Curves")
axes[0].legend()
axes[0].grid(True, alpha=0.3)
axes[0].set_yscale("log")

axes[1].plot(losses_ste[:50], color="green", linewidth=2, label="STE (converging)")
axes[1].plot(losses_no_ste[:50], color="red", linewidth=2, label="No-STE (stalled)")
axes[1].set_xlabel("Training Step (first 50)")
axes[1].set_ylabel("MSE Loss")
axes[1].set_title("Early Training: STE Enables Convergence")
axes[1].legend()
axes[1].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig("ste_comparison.png", dpi=100, bbox_inches="tight")
plt.close()
print("Saved ste_comparison.png")
```

## Demonstrating Weight Convergence with STE

Let's track how close the quantized weights get to the target over training:

```python
torch.manual_seed(42)
w = nn.Parameter(torch.randn(64, 64) * 0.1)
fq = FakeQuantSTE()
opt = torch.optim.Adam([w], lr=0.01)
target = torch.randn(64, 64) * 0.05

weight_distances = []
for step in range(200):
    opt.zero_grad()
    w_q = fq(w)
    loss = (w_q - target).pow(2).mean()
    loss.backward()
    opt.step()
    with torch.no_grad():
        dist = (fq(w) - target).pow(2).mean().item()
        weight_distances.append(dist)

print(f"Initial distance to target: {weight_distances[0]:.6f}")
print(f"Final distance to target:   {weight_distances[-1]:.6f}")
print(f"Convergence: {weight_distances[0] / max(weight_distances[-1], 1e-8):.1f}× closer")
```

## STE Variants

The vanilla STE uses identity for the backward pass, but other approximations exist:

### 1. Clipping STE
Passes gradients only when the input is within the quantization range. Outside the range, gradient is clipped to zero (since clamping happens anyway).

### 2. Smooth STE (Sigmoid Approximation)
Replaces the staircase with a smooth function whose derivative is well-defined and non-zero.

```python
class ClippingSTE(torch.autograd.Function):
    """STE that clips gradient to zero outside [qmin, qmax] range."""
    @staticmethod
    def forward(ctx, x, qmin, qmax):
        ctx.save_for_backward(x)
        ctx.qmin = qmin
        ctx.qmax = qmax
        return torch.round(x).clamp(qmin, qmax)

    @staticmethod
    def backward(ctx, grad_output):
        x, = ctx.saved_tensors
        mask = (x >= ctx.qmin) & (x <= ctx.qmax)
        return grad_output * mask.float(), None, None

class SmoothSTE(torch.autograd.Function):
    """STE using a sum of sigmoids as a smooth staircase approximation."""
    @staticmethod
    def forward(ctx, x, temperature=10.0):
        ctx.save_for_backward(x)
        ctx.temperature = temperature
        return torch.round(x)

    @staticmethod
    def backward(ctx, grad_output):
        x, = ctx.saved_tensors
        frac = x - torch.floor(x)
        smooth_grad = ctx.temperature * torch.sigmoid(ctx.temperature * (frac - 0.5)) * \
                      (1 - torch.sigmoid(ctx.temperature * (frac - 0.5)))
        grad = torch.where(smooth_grad > 0.01, smooth_grad, torch.ones_like(smooth_grad))
        return grad_output * grad, None

x_test = torch.linspace(-2, 2, 100, requires_grad=True)

y_vanilla = RoundSTE.apply(x_test)
y_vanilla.sum().backward()
grad_vanilla = x_test.grad.clone()

x_test.grad = None
y_clip = ClippingSTE.apply(x_test, -1.0, 1.0)
y_clip.sum().backward()
grad_clip = x_test.grad.clone()

x_test.grad = None
y_smooth = SmoothSTE.apply(x_test, 10.0)
y_smooth.sum().backward()
grad_smooth = x_test.grad.clone()

print(f"Vanilla STE grad mean: {grad_vanilla.abs().mean():.4f} (constant 1.0)")
print(f"Clipping STE grad mean: {grad_clip.abs().mean():.4f} (zeros outside range)")
print(f"Smooth STE grad mean: {grad_smooth.abs().mean():.4f} (varies smoothly)")
```

## Training Comparison: Vanilla STE vs Clipping STE vs Smooth STE

```python
def train_variant(ste_fn, n_steps=200, lr=0.01):
    """Train using a specific STE variant."""
    torch.manual_seed(42)
    w = nn.Parameter(torch.randn(64, 64) * 0.1)
    opt = torch.optim.Adam([w], lr=lr)
    target = torch.randn(64, 64) * 0.05
    losses = []
    for step in range(n_steps):
        opt.zero_grad()
        min_val, max_val = w.detach().min(), w.detach().max()
        scale = (max_val - min_val).clamp(min=1e-8) / 15
        zp = -8 - min_val / scale
        x_scaled = w / scale + zp
        x_int = ste_fn(x_scaled)
        w_q = (x_int - zp) * scale
        loss = (w_q - target).pow(2).mean()
        loss.backward()
        opt.step()
        losses.append(loss.item())
    return losses

losses_vanilla = train_variant(lambda x: RoundSTE.apply(x).clamp(-8, 7))
losses_clip = train_variant(lambda x: ClippingSTE.apply(x, -8.0, 7.0))
losses_smooth = train_variant(lambda x: SmoothSTE.apply(x, 10.0).clamp(-8, 7))

print(f"{'Variant':<15} {'Initial':<10} {'Final':<10} {'Reduction':<10}")
print("-" * 45)
for name, l in [("Vanilla", losses_vanilla), ("Clipping", losses_clip), ("Smooth", losses_smooth)]:
    reduction = (1 - l[-1] / l[0]) * 100
    print(f"{name:<15} {l[0]:<10.4f} {l[-1]:<10.4f} {reduction:<10.1f}%")
```

## Compare: No STE (Detach Quant)

```python
w2 = nn.Parameter(torch.randn(64, 64))
try:
    y = fake_quant_no_ste(w2)
    y.sum().backward()
    grad_norm = w2.grad.norm().item() if w2.grad is not None else 0
    print(f"Grad norm without proper STE: {grad_norm:.6f}")
    print("Note: This variant passes gradient via identity residual, not true STE")
except RuntimeError as e:
    print("Expected: no grad through detached quant")
```

## Exercise: Implement a "Smooth" STE Using a Sigmoid Approximation

Implement a smooth version of STE where the backward pass uses a sigmoid-based soft staircase instead of a hard identity. Compare training convergence with vanilla STE over 200 steps.

The idea: approximate `round(x)` with `floor(x) + sigmoid(k * (x - floor(x) - 0.5))` where `k` controls sharpness. As k→∞, this approaches the true staircase.

```python
class SigmoidSTE(torch.autograd.Function):
    """Smooth STE: forward uses round(), backward uses sigmoid derivative."""
    @staticmethod
    def forward(ctx, x, sharpness=20.0):
        ctx.save_for_backward(x)
        ctx.sharpness = sharpness
        return torch.round(x)

    @staticmethod
    def backward(ctx, grad_output):
        x, = ctx.saved_tensors
        k = ctx.sharpness
        frac = x - torch.floor(x) - 0.5
        sig = torch.sigmoid(k * frac)
        derivative = k * sig * (1 - sig)
        derivative = derivative.clamp(min=0.1, max=2.0)
        return grad_output * derivative, None

def train_sigmoid_ste(sharpness=20.0, n_steps=200, lr=0.01):
    torch.manual_seed(42)
    w = nn.Parameter(torch.randn(64, 64) * 0.1)
    opt = torch.optim.Adam([w], lr=lr)
    target = torch.randn(64, 64) * 0.05
    fq = FakeQuantSTE()
    losses = []
    for step in range(n_steps):
        opt.zero_grad()
        min_val, max_val = w.detach().min(), w.detach().max()
        scale = (max_val - min_val).clamp(min=1e-8) / 15
        zp = -8 - min_val / scale
        x_scaled = w / scale + zp
        x_int = SigmoidSTE.apply(x_scaled, sharpness).clamp(-8, 7)
        w_q = (x_int - zp) * scale
        loss = (w_q - target).pow(2).mean()
        loss.backward()
        opt.step()
        losses.append(loss.item())
    return losses

losses_sig_5 = train_sigmoid_ste(sharpness=5.0)
losses_sig_20 = train_sigmoid_ste(sharpness=20.0)
losses_sig_50 = train_sigmoid_ste(sharpness=50.0)
losses_vanilla_ref = train_with_ste(200)

print(f"{'Method':<25} {'Final Loss':<12} {'Reduction':<10}")
print("-" * 47)
for name, l in [("Vanilla STE", losses_vanilla_ref),
                ("Sigmoid k=5", losses_sig_5),
                ("Sigmoid k=20", losses_sig_20),
                ("Sigmoid k=50", losses_sig_50)]:
    reduction = (1 - l[-1] / l[0]) * 100
    print(f"{name:<25} {l[-1]:<12.6f} {reduction:<10.1f}%")
```

## Where This Leads Next

You now have every piece of QAT: a quantizer (3.1), a fake-quant module (3.2), and a way to train through it (STE, this section). Section 3.4 assembles them into a real fine-tuning run that alternates datasets and measures whether the quantized model keeps its quality.

---

## Key Takeaway

The Straight-Through Estimator is the **enabling trick** for Quantization-Aware Training. Without it, gradients are zero (or undefined) through rounding, and training cannot proceed. Key points:

1. **The problem** — `round()` has zero derivative almost everywhere, killing gradient flow
2. **The STE solution** — use `round()` in forward, but pretend it's `identity` in backward; this is a biased but practically effective gradient estimator
3. **It works** — STE-trained models converge to the quantized target; without STE, loss remains flat
4. **Variants exist** — clipping STE zeros gradients outside the valid range (prevents divergence); smooth STE uses sigmoid derivatives for more informative gradients near bin boundaries
5. **Industry standard** — PyTorch's `torch.quantization.FakeQuantize` uses STE internally; all major QAT papers rely on it

STE is the standard trick in QAT papers and PyTorch `torch.quantization` fake quant modules. Next, we combine STE with real training data to fine-tune a model that survives integer rounding at deployment time.

## Further Reading (Optional)

**Optional — you do NOT need these to continue. They are for curious students who want the original sources.**

- Bengio, Léonard, & Courville (2013). *Estimating or Propagating Gradients Through Stochastic Neurons (STE)*. arXiv:1308.3432.
- Courbariaux et al. (2016). *Binarized Neural Networks*. NeurIPS.
