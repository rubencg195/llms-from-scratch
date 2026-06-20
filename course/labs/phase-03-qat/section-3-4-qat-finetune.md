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

# Section 3.4: Fine-Tuning the Model to Survive Integer Rounding

**Goal:** Alternate TinyStories + Glaive batches while QAT-wrapped layers stay trainable; compare perplexity before and after quantization.

## Why Alternating Datasets Matters

Catastrophic forgetting is the biggest risk when specializing a model. If you only train on quantization-friendly data, the model may lose its language ability. By alternating between TinyStories (grammar, fluency) and Glaive (tool-calling structure), we preserve both skills while the weights learn to tolerate rounding.

## The STE-enabled Fake Quantization Module

This is the production-grade version that properly uses the Straight-Through Estimator from Section 3.3 so gradients flow through the rounding operation.

```python
import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class RoundSTE(torch.autograd.Function):
    """Straight-Through Estimator for rounding."""
    @staticmethod
    def forward(ctx, x):
        return torch.round(x)

    @staticmethod
    def backward(ctx, grad_output):
        return grad_output

class FakeQuantSTE(nn.Module):
    """Fake quantization with proper STE gradient bypass."""
    def __init__(self, qmin=-8, qmax=7):
        super().__init__()
        self.qmin = qmin
        self.qmax = qmax

    def forward(self, x):
        if not self.training:
            return x
        min_val = x.detach().min()
        max_val = x.detach().max()
        scale = (max_val - min_val).clamp(min=1e-8) / (self.qmax - self.qmin)
        zp = self.qmin - min_val / scale
        x_scaled = x / scale + zp
        x_int = RoundSTE.apply(x_scaled).clamp(self.qmin, self.qmax)
        return (x_int - zp) * scale
```

## QAT-Wrapped Linear Layer

Each `QATLinear` quantizes both weights and activations in the forward pass. During backprop, STE ensures gradients pass through the rounding as if it were an identity operation.

```python
class QATLinear(nn.Linear):
    def __init__(self, in_features, out_features, bias=True):
        super().__init__(in_features, out_features, bias)
        self.w_quant = FakeQuantSTE()
        self.a_quant = FakeQuantSTE()

    def forward(self, x):
        w_q = self.w_quant(self.weight)
        x_q = self.a_quant(x)
        return F.linear(x_q, w_q, self.bias)
```

## Alternating Batch Iterator

Simulates drawing from two data sources. In production, replace the random tensors with real tokenized TinyStories and Glaive sequences.

```python
def mixed_batches(n_steps=100, batch_size=8, seq_len=64, vocab=8000):
    """Yield (source_name, input_ids, target_ids) alternating sources."""
    for step in range(n_steps):
        source = "tinystories" if step % 2 == 0 else "glaive"
        x = torch.randint(0, vocab, (batch_size, seq_len))
        y = torch.randint(0, vocab, (batch_size, seq_len))
        yield source, x, y
```

## Build and Train the QAT Model

```python
device = "cuda" if torch.cuda.is_available() else "cpu"

model = nn.Sequential(
    nn.Embedding(8000, 256),
    QATLinear(256, 256),
    nn.ReLU(),
    QATLinear(256, 8000),
).to(device)

opt = torch.optim.AdamW(model.parameters(), lr=5e-5)
ce = nn.CrossEntropyLoss()

losses = {"tinystories": [], "glaive": []}

model.train()
for step, (source, x, y) in enumerate(mixed_batches(n_steps=60)):
    x, y = x.to(device), y.to(device)
    logits = model(x)
    loss = ce(logits.view(-1, 8000), y.view(-1))
    opt.zero_grad()
    loss.backward()
    opt.step()
    losses[source].append(loss.item())
    if step % 10 == 0:
        print(f"[{source}] step {step} loss {loss.item():.3f}")
```

## Visualization: Training Losses by Source Over Steps

```python
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

fig, axes = plt.subplots(1, 2, figsize=(14, 5))

axes[0].plot(losses["tinystories"], color="steelblue", linewidth=1.5,
             marker="o", markersize=3, label="TinyStories")
axes[0].plot(losses["glaive"], color="coral", linewidth=1.5,
             marker="s", markersize=3, label="Glaive")
axes[0].set_xlabel("Batch Index (within source)")
axes[0].set_ylabel("Cross-Entropy Loss")
axes[0].set_title("QAT Training: Loss by Data Source")
axes[0].legend()
axes[0].grid(True, alpha=0.3)

all_losses = []
for i in range(max(len(losses["tinystories"]), len(losses["glaive"]))):
    if i < len(losses["tinystories"]):
        all_losses.append(("tinystories", losses["tinystories"][i]))
    if i < len(losses["glaive"]):
        all_losses.append(("glaive", losses["glaive"][i]))

combined = [v for _, v in all_losses]
colors = ["steelblue" if s == "tinystories" else "coral" for s, _ in all_losses]
axes[1].bar(range(len(combined)), combined, color=colors, alpha=0.7, width=1.0)
axes[1].set_xlabel("Global Step")
axes[1].set_ylabel("Cross-Entropy Loss")
axes[1].set_title("Interleaved Training (blue=TinyStories, orange=Glaive)")
axes[1].grid(True, alpha=0.3, axis="y")

plt.tight_layout()
plt.savefig("qat_training_losses.png", dpi=100, bbox_inches="tight")
plt.close()
print("Saved qat_training_losses.png")
print(f"TinyStories — mean loss: {sum(losses['tinystories'])/len(losses['tinystories']):.3f}")
print(f"Glaive      — mean loss: {sum(losses['glaive'])/len(losses['glaive']):.3f}")
```

## Perplexity Computation

Perplexity is the standard metric for language model quality: $\text{PPL} = e^{\text{CE loss}}$. Lower is better. We compute it on held-out data to assess how much the quantization hurts.

```python
@torch.no_grad()
def compute_perplexity(model, n_batches=20, batch_size=16, seq_len=64, vocab=8000):
    """Compute perplexity on synthetic test data."""
    model.eval()
    total_loss = 0.0
    total_tokens = 0

    for _ in range(n_batches):
        x = torch.randint(0, vocab, (batch_size, seq_len), device=device)
        y = torch.randint(0, vocab, (batch_size, seq_len), device=device)
        logits = model(x)
        loss = F.cross_entropy(logits.view(-1, vocab), y.view(-1), reduction="sum")
        total_loss += loss.item()
        total_tokens += batch_size * seq_len

    avg_ce = total_loss / total_tokens
    perplexity = math.exp(avg_ce)
    model.train()
    return avg_ce, perplexity

ce_qat, ppl_qat = compute_perplexity(model)
print(f"QAT Model — CE: {ce_qat:.4f}, Perplexity: {ppl_qat:.2f}")
```

## Compare: QAT vs Unquantized Baseline

```python
baseline = nn.Sequential(
    nn.Embedding(8000, 256),
    nn.Linear(256, 256),
    nn.ReLU(),
    nn.Linear(256, 8000),
).to(device)

ce_base, ppl_base = compute_perplexity(baseline)
print(f"Baseline (untrained, no quant) — CE: {ce_base:.4f}, PPL: {ppl_base:.2f}")
print(f"QAT (trained, quantized)       — CE: {ce_qat:.4f}, PPL: {ppl_qat:.2f}")
print(f"\nPPL gap: {abs(ppl_qat - ppl_base):.2f}")

baseline.eval()
model.eval()
with torch.no_grad():
    test_x = torch.randint(0, 8000, (16, 64), device=device)
    test_y = torch.randint(0, 8000, (16, 64), device=device)
    base_loss = ce(baseline(test_x).view(-1, 8000), test_y.view(-1))
    qat_loss = ce(model(test_x).view(-1, 8000), test_y.view(-1))
    print(f"\nSingle-batch comparison:")
    print(f"  baseline CE: {base_loss.item():.3f}")
    print(f"  QAT CE:      {qat_loss.item():.3f}")
    print(f"  gap:         {abs(qat_loss.item() - base_loss.item()):.3f}")
```

## Exercise: Compare QAT with 4-Bit vs 8-Bit Precision

Train two QAT models — one with 4-bit fake quantization (qmin=-8, qmax=7) and one with 8-bit (qmin=-128, qmax=127). Compare their final loss and perplexity.

```python
class QATLinearNBit(nn.Linear):
    """QAT linear layer with configurable bit width."""
    def __init__(self, in_features, out_features, n_bits=4, bias=True):
        super().__init__(in_features, out_features, bias)
        qmax = 2 ** (n_bits - 1) - 1
        qmin = -(2 ** (n_bits - 1))
        self.w_quant = FakeQuantSTE(qmin=qmin, qmax=qmax)
        self.a_quant = FakeQuantSTE(qmin=qmin, qmax=qmax)

    def forward(self, x):
        w_q = self.w_quant(self.weight)
        x_q = self.a_quant(x)
        return F.linear(x_q, w_q, self.bias)

def train_qat_model(n_bits, n_steps=60, lr=5e-5):
    """Train a QAT model with specified bit width."""
    m = nn.Sequential(
        nn.Embedding(8000, 256),
        QATLinearNBit(256, 256, n_bits=n_bits),
        nn.ReLU(),
        QATLinearNBit(256, 8000, n_bits=n_bits),
    ).to(device)

    optimizer = torch.optim.AdamW(m.parameters(), lr=lr)
    m.train()
    step_losses = []
    for step, (source, x, y) in enumerate(mixed_batches(n_steps=n_steps)):
        x, y = x.to(device), y.to(device)
        logits = m(x)
        loss = ce(logits.view(-1, 8000), y.view(-1))
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        step_losses.append(loss.item())

    return m, step_losses

model_4bit, losses_4bit = train_qat_model(n_bits=4)
model_8bit, losses_8bit = train_qat_model(n_bits=8)

ce_4, ppl_4 = compute_perplexity(model_4bit)
ce_8, ppl_8 = compute_perplexity(model_8bit)

print(f"{'Precision':<12} {'Final Loss':<12} {'Perplexity':<12} {'Compression':<12}")
print("-" * 48)
print(f"{'4-bit':<12} {losses_4bit[-1]:<12.4f} {ppl_4:<12.2f} {'8×':<12}")
print(f"{'8-bit':<12} {losses_8bit[-1]:<12.4f} {ppl_8:<12.2f} {'4×':<12}")
print(f"\n4-bit has {((ppl_4 - ppl_8) / ppl_8 * 100):.1f}% higher perplexity but 2× more compression")
```

## Save Checkpoint

```python
import os
os.makedirs("checkpoints", exist_ok=True)
torch.save(model.state_dict(), "checkpoints/phase3_qat.pt")
print("Saved checkpoints/phase3_qat.pt")
if device == "cuda":
    print("Peak VRAM (MB):", torch.cuda.max_memory_allocated() / 1e6)
```

---

## Key Takeaway

- STE keeps gradients alive through `round()` — without it, QAT training stalls.
- Alternating datasets prevents catastrophic forgetting of either skill.
- QAT loss should stay within ~10% of the unquantized baseline; if it diverges, reduce learning rate or increase `qmax` bits.
- **Perplexity** is the right evaluation metric: $\text{PPL} = e^{\text{CE}}$. A 4-bit QAT model typically shows 5-15% higher PPL than 8-bit, but the 2× additional compression is often worth it.
- The 4-bit vs 8-bit trade-off is the core deployment decision: more compression means slightly worse quality, but fits on cheaper hardware.

**Next:** Phase 4 — Mixture of Experts, where we make the model smarter without making it slower.
