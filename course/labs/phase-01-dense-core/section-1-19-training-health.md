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

# Section 1.19: Training Health — Init, Activations & Gradients

**Goal:** Learn Karpathy's *makemore* Part 3 diagnostics — check that your network is healthy **before** you burn GPU hours on Section 1.5.

## What You Need to Know First

- **Attention and FFN blocks** (Sections 1.3–1.4) — or at least `nn.Linear` + LayerNorm from Phase 0.
- **The training loop skeleton** (Section 0.3) — forward, loss, backward, step.

Run this lab **after Section 1.4, before Section 1.5**. It takes minutes and catches the most common "loss is NaN / flat / exploding" failures.

## Why initialization matters

If weights start too large, activations explode. Too small, signals vanish. GPT-style models use small random init:

```python
import torch
import torch.nn as nn

def init_gpt_style(module):
    if isinstance(module, nn.Linear):
        nn.init.normal_(module.weight, mean=0.0, std=0.02)
        if module.bias is not None:
            nn.init.zeros_(module.bias)
    elif isinstance(module, nn.Embedding):
        nn.init.normal_(module.weight, mean=0.0, std=0.02)

# Toy stack: embedding → linear → GELU → linear (like a tiny FFN)
class TinyBlock(nn.Module):
    def __init__(self, vocab=8000, d=512):
        super().__init__()
        self.emb = nn.Embedding(vocab, d)
        self.fc1 = nn.Linear(d, 4 * d)
        self.fc2 = nn.Linear(4 * d, d)
        self.ln = nn.LayerNorm(d)

    def forward(self, idx):
        x = self.emb(idx)
        x = x + self.fc2(torch.nn.functional.gelu(self.fc1(self.ln(x))))
        return x

model = TinyBlock()
model.apply(init_gpt_style)
print("Init OK — weights ~ N(0, 0.02²)")
```

## Activation statistics (forward pass health)

Healthy activations stay near unit scale layer-to-layer:

```python
activations = {}

def hook(name):
    def fn(_module, _inp, out):
        t = out if isinstance(out, torch.Tensor) else out[0]
        activations[name] = {
            "mean": t.mean().item(),
            "std": t.std().item(),
            "max": t.abs().max().item(),
        }
    return fn

model.ln.register_forward_hook(hook("after_ln"))
model.fc1.register_forward_hook(hook("after_fc1"))
model.fc2.register_forward_hook(hook("after_fc2"))

idx = torch.randint(0, 8000, (4, 32))  # batch=4, seq=32
_ = model(idx)

for name, stats in activations.items():
    print(f"{name:12s}  mean={stats['mean']:+.3f}  std={stats['std']:.3f}  max={stats['max']:.2f}")
```

**Red flags:**
- `std > 10` — activations blowing up; lower init or add LayerNorm earlier.
- `std < 0.01` — vanishing signal; check dead neurons.
- `mean` drifting far from 0 — consider LayerNorm placement.

## Gradient statistics (backward pass health)

```python
model.zero_grad()
logits = model(idx)
loss = logits.pow(2).mean()  # dummy loss — just to trigger backward
loss.backward()

grad_norms = {}
for name, p in model.named_parameters():
    if p.grad is not None:
        grad_norms[name] = p.grad.norm().item()

for name, g in sorted(grad_norms.items(), key=lambda x: -x[1])[:6]:
    print(f"{name:20s}  grad_norm={g:.4f}")
```

**Healthy range:** most gradient norms between `1e-4` and `1e-1` early in training.

**Red flags:**
- Any norm `> 10` — exploding; use gradient clipping (Section 1.5 uses `max_norm=1.0`).
- All norms `< 1e-7` — vanishing; check init, depth, or dead GELU/ReLU units.

## Compare bad init vs good init

```python
def bad_init(module):
    if isinstance(module, nn.Linear):
        nn.init.normal_(module.weight, mean=0.0, std=1.0)  # 50× too large!

bad = TinyBlock()
bad.apply(bad_init)
bad(idx)  # forward
bad.zero_grad()
bad(idx).pow(2).mean().backward()
print("Bad init max activation:", max(p.abs().max().item() for p in bad.parameters()))
print("Bad init max grad:", max(p.grad.norm().item() for p in bad.parameters() if p.grad is not None))
```

Run the same hooks on `bad` — you will see activations in the hundreds and gradients that explode.

## Perplexity sanity check (after a few training steps)

Once you start training in Section 1.5, track **perplexity** = $e^{\text{loss}}$:

```python
import math

def perplexity(avg_ce_loss):
    return math.exp(avg_ce_loss)

# Examples for TinyStories-scale model:
for loss in [8.0, 5.0, 3.5, 2.5]:
    print(f"loss={loss:.1f}  perplexity={perplexity(loss):.0f}  (~{perplexity(loss):.0f} equally likely tokens)")
```

Random guessing over 8000 tokens → loss $\approx \ln(8000) \approx 9.0$, perplexity $\approx 8000$.
A trained Phase 1 model should reach loss ~2.5–4.0 (perplexity ~12–55).

## Pre-flight checklist for Section 1.5

Before launching the full training run:

1. **Init** — `std=0.02` on Linear/Embedding weights.
2. **Forward** — activation std roughly 0.5–2.0 at each layer on random input.
3. **Backward** — no gradient norm above ~1.0 before clipping.
4. **Loss** — first batch loss should be ~8–10 (near random), not NaN.
5. **Generation** — after ~1000 steps, output should beat random characters.

## Exercise: gradient clipping demo

```python
torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
total_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
print(f"Total grad norm after clip: {total_norm:.4f}")
```

Apply this in every step of Section 1.5's training loop.

## Where This Leads Next

Section 1.5 runs the full autoregressive training loop. Use the hooks from this lab for the first 10 batches — if stats look healthy, let it run for hours with confidence.

## Key Takeaway

- **Small init** (`std=0.02`) keeps activations stable in deep Transformers.
- **Monitor activation std** forward and **gradient norms** backward — cheap insurance.
- **Perplexity** = $e^{\text{loss}}$; random ≈ vocab size, trained ≈ 12–55 on TinyStories.
- **Gradient clipping** prevents one bad batch from destroying weights.

## Checkpoint

You can diagnose a sick network in minutes. Next: **Section 1.5 — Full Training Loop**.

## Further Reading (Optional)

- Karpathy, *makemore* Part 3 — activation/gradient statistics and batch normalization intuition.
- Radford et al. (2019). *GPT-2* — documents weight init and training hyperparameters.
- Glorot & Bengio (2010). *Understanding the difficulty of training deep feedforward neural networks*. AISTATS.
