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

# Section 8.3: The Surprise Metric — Using Prediction Error as a Trigger to Learn

**Goal:** Gate memory writes on surprise threshold; skip TTT when prediction is confident.

## What You Need to Know First

This section decides *when* to write to memory. The prerequisites are small:

- **From Sections 8.1–8.2:** writing to memory means a Test-Time-Training step, which costs compute — so we don't want to do it every token.
- **"Surprise" = prediction error** — simply how far the model's prediction was from the truth (here, mean squared error). Big error = surprising.
- **A threshold / gate** — a cutoff value; if surprise is above it, we write, otherwise we skip.
- **Exponential moving average (EMA)** — a running average that gently follows recent values, used so the threshold adapts to how hard the current text is. `new = (1-α)·old + α·latest`.

No new math beyond squaring a difference and taking a running average.

## Learning from Surprise — The Brain Does This Too

Neuroscience tells us that the brain doesn't memorize *everything* — it selectively encodes
information that violates predictions. When you walk your usual route to work, you form no
new memories. But if a building is demolished overnight, that surprise triggers encoding.

This is exactly the principle behind surprise-gated memory writes in Titans:
- "Then they discussed the weather" → low surprise → skip memory write
- "My new password is X7k#9mQ" → high surprise → trigger TTT step

The surprise metric is simply the prediction error: how badly did the memory predict the
current input? High error = surprising = worth remembering.

---

## Basic Surprise Gate

```python
import torch
import torch.nn as nn

class SurpriseGate:
    def __init__(self, threshold=0.5):
        self.threshold = threshold

    def should_write(self, pred, target):
        surprise = (pred - target).pow(2).mean().item()
        return surprise > self.threshold, surprise

d = 128
memory = nn.Linear(d, d)
gate = SurpriseGate(threshold=0.3)

x = torch.randn(d)
target = torch.randn(d)
pred = memory(x)
write, surprise = gate.should_write(pred, target)
print(f"High surprise: {surprise:.4f} -> write: {write}")

# Low-surprise case: target matches prediction
target2 = pred.detach()
pred2 = memory(x)
write2, s2 = gate.should_write(pred2, target2)
print(f"Low surprise:  {s2:.4f} -> write: {write2}")
```

---

## Adaptive Threshold: Running Mean of Surprise

A fixed threshold is fragile — it doesn't adapt to the difficulty of the current text.
Technical content will naturally have higher prediction error than casual conversation.
An **adaptive threshold** based on the running mean of surprise automatically calibrates:

```python
class AdaptiveSurpriseGate:
    def __init__(self, alpha=0.1, multiplier=1.5):
        """
        alpha: EMA smoothing factor (smaller = slower adaptation)
        multiplier: write if surprise > multiplier * running_mean
        """
        self.alpha = alpha
        self.multiplier = multiplier
        self.running_mean = None

    def should_write(self, pred, target):
        surprise = (pred - target).pow(2).mean().item()

        if self.running_mean is None:
            self.running_mean = surprise
            return True, surprise  # always write the first token

        threshold = self.multiplier * self.running_mean
        should = surprise > threshold

        # Update running mean (EMA)
        self.running_mean = (1 - self.alpha) * self.running_mean + self.alpha * surprise

        return should, surprise

    @property
    def current_threshold(self):
        if self.running_mean is None:
            return 0.0
        return self.multiplier * self.running_mean

adaptive_gate = AdaptiveSurpriseGate(alpha=0.1, multiplier=1.5)
print("Adaptive gate: threshold adjusts to content difficulty")
```

---

## Demonstration: Repeated Words vs New Fact

Feed a sequence where most tokens are predictable (repeated patterns) but one contains
genuinely new information. The surprise gate should trigger only on the novel token.

```python
torch.manual_seed(42)
mem_demo = nn.Linear(d, d)

# Simulate token embeddings: 20 "boring" tokens followed by 1 "surprising" token
boring_embeddings = torch.randn(1, d).expand(20, -1)  # same token repeated
surprising_embedding = torch.randn(1, d) * 3  # very different

all_embeddings = torch.cat([boring_embeddings, surprising_embedding], dim=0)

gate_demo = AdaptiveSurpriseGate(alpha=0.2, multiplier=2.0)
surprises = []
writes = []
thresholds = []

for i in range(all_embeddings.shape[0]):
    x = all_embeddings[i]
    target = all_embeddings[min(i + 1, len(all_embeddings) - 1)]  # next token
    pred = mem_demo(x)
    write, s = gate_demo.should_write(pred, target)
    surprises.append(s)
    writes.append(write)
    thresholds.append(gate_demo.current_threshold)

print("Token-by-token surprise analysis:")
print(f"{'Token':<8} {'Surprise':<12} {'Threshold':<12} {'Write?'}")
print("-" * 40)
for i in [0, 1, 5, 10, 15, 19, 20]:  # sample positions
    label = "NEW FACT" if i == 20 else f"repeat {i}"
    print(f"{label:<8} {surprises[i]:<12.4f} {thresholds[i]:<12.4f} {'YES' if writes[i] else 'no'}")

n_writes = sum(writes)
print(f"\nTotal writes: {n_writes}/{len(writes)} ({n_writes/len(writes)*100:.0f}%)")
print("Only genuinely surprising tokens trigger memory updates!")
```

---

## Visualization: Surprise Values Over a Sequence with Threshold Line

```python
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

fig, ax = plt.subplots(figsize=(12, 5))

x_positions = range(len(surprises))
colors = ['red' if w else 'steelblue' for w in writes]

ax.bar(x_positions, surprises, color=colors, alpha=0.7, label='Surprise value')
ax.plot(x_positions, thresholds, 'k--', linewidth=2, label='Adaptive threshold')
ax.axvline(x=20, color='green', linestyle=':', linewidth=2, label='New fact injected')

ax.set_xlabel('Token Position')
ax.set_ylabel('Surprise (MSE)')
ax.set_title('Surprise-Gated Memory Writes: Only Novel Information Triggers Updates')
ax.legend()

# Add annotation
ax.annotate('New fact\n(high surprise)', xy=(20, surprises[20]),
            xytext=(15, surprises[20] * 0.8),
            arrowprops=dict(arrowstyle='->', color='green'),
            fontsize=10, color='green')

plt.tight_layout()
plt.savefig('surprise_gate.png', dpi=100, bbox_inches='tight')
plt.close()
print("Saved surprise_gate.png")
print("Red bars = writes triggered, Blue bars = skipped")
```

---

## Batch Policy: TTT Update If Surprised

```python
# Import NeuralMemory concept from 8.2
class SimpleNeuralMemory(nn.Module):
    def __init__(self, d):
        super().__init__()
        self.proj = nn.Linear(d, d)
        self.memory = nn.Parameter(torch.zeros(d, d))

    def read(self, x):
        k = self.proj(x)
        att = torch.softmax(k @ self.memory, dim=-1)
        return att

    def write_delta(self, key, value, lr=0.01):
        with torch.no_grad():
            k = self.proj(key).mean(dim=0) if key.dim() > 1 else self.proj(key)
            v = self.proj(value).mean(dim=0) if value.dim() > 1 else self.proj(value)
            self.memory += lr * torch.outer(v, k)

def ttt_update_if_surprised(memory_module, x, target, gate, lr=0.01):
    """Only update memory if the prediction is surprising."""
    pred = memory_module.read(x.unsqueeze(0)).squeeze(0)
    should_write, s = gate.should_write(pred, target)
    if should_write:
        memory_module.write_delta(x, target, lr=lr)
    return should_write, s

# Run on a sequence
torch.manual_seed(42)
seq_mem = SimpleNeuralMemory(d)
seq_gate = AdaptiveSurpriseGate(alpha=0.1, multiplier=2.0)

n_writes_total = 0
for i in range(50):
    x = torch.randn(d) if i != 30 else torch.randn(d) * 5  # spike at position 30
    target = torch.randn(d) if i != 30 else torch.randn(d) * 5
    wrote, s = ttt_update_if_surprised(seq_mem, x, target, seq_gate)
    n_writes_total += int(wrote)

print(f"Sequence of 50 tokens: {n_writes_total} writes triggered")
print("Maps to Titans: unexpected facts (new name, passcode) trigger write; small talk skipped.")
```

---

## Exercise: Implement Exponential Moving Average Threshold

Build a more sophisticated adaptive threshold with configurable warmup and bounds:

```python
class EMAThreshold:
    def __init__(self, alpha=0.05, multiplier=2.0, warmup=10,
                 min_threshold=0.01, max_threshold=10.0):
        self.alpha = alpha
        self.multiplier = multiplier
        self.warmup = warmup
        self.min_threshold = min_threshold
        self.max_threshold = max_threshold
        self.ema = 0.0
        self.count = 0
        self.history = []

    def update_and_check(self, surprise_value):
        self.count += 1
        self.history.append(surprise_value)

        # During warmup, always write and accumulate statistics
        if self.count <= self.warmup:
            self.ema = sum(self.history) / len(self.history)
            return True

        # Compute adaptive threshold with bounds
        threshold = max(self.min_threshold,
                       min(self.max_threshold, self.multiplier * self.ema))

        # EMA update
        self.ema = (1 - self.alpha) * self.ema + self.alpha * surprise_value

        return surprise_value > threshold

    @property
    def threshold(self):
        return max(self.min_threshold,
                  min(self.max_threshold, self.multiplier * self.ema))

# Test with varying difficulty levels
ema_thresh = EMAThreshold(alpha=0.05, multiplier=2.0, warmup=10)

# Phase 1: easy tokens (low surprise)
easy_surprises = [0.1 + 0.05 * torch.randn(1).item() for _ in range(20)]
# Phase 2: hard tokens (high baseline surprise)
hard_surprises = [0.8 + 0.1 * torch.randn(1).item() for _ in range(20)]
# Phase 3: one genuinely novel fact among easy tokens
novel_surprises = [0.1 + 0.05 * torch.randn(1).item() for _ in range(9)] + [2.5]

all_surprises = easy_surprises + hard_surprises + novel_surprises
write_decisions = []
threshold_trace = []

for s in all_surprises:
    write = ema_thresh.update_and_check(s)
    write_decisions.append(write)
    threshold_trace.append(ema_thresh.threshold)

print("Adaptive Threshold Behavior:")
print(f"  Easy phase (tokens 0-19):   {sum(write_decisions[:20])}/{20} writes")
print(f"  Hard phase (tokens 20-39):  {sum(write_decisions[20:40])}/{20} writes")
print(f"  Novel fact (token 49):      write={write_decisions[49]}")
print(f"\n  Threshold adapted: {threshold_trace[0]:.3f} -> {threshold_trace[25]:.3f} -> {threshold_trace[49]:.3f}")
print("  The threshold rises during hard content, preventing spurious writes,")
print("  but the novel fact at 2.5 still exceeds the adapted threshold.")
```

---

## Where This Leads Next

We can now decide *what* to remember. Section 8.4 handles the opposite problem — *forgetting*.
It adds decay and momentum so old, unreinforced facts fade away, keeping the fixed-size memory
from filling up and turning into noise.

## Key Takeaway

The surprise metric transforms TTT from "learn everything" to "learn what matters."
By gating memory writes on prediction error relative to an adaptive running threshold,
we ensure that only genuinely novel information (new names, codes, facts) triggers the
expensive inner-loop gradient step. Repeated or predictable content passes through without
modifying the memory. This mirrors biological learning: the hippocampus encodes surprising
events, not routine ones. The adaptive EMA threshold automatically calibrates to content
difficulty — technical papers have higher baseline surprise than casual chat.

## Further Reading (Optional)

**Optional — you do NOT need these to continue. They are for curious students who want the original sources.**

- Behrouz, Zhong, & Mirrokni (2024). *Titans: Learning to Memorize at Test Time*. arXiv:2501.00663.
- Itti & Baldi (2009). *Bayesian Surprise Attracts Human Attention*. Vision Research.
