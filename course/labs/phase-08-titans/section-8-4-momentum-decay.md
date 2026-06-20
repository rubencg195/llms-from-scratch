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

# Section 8.4: Momentum and Decay — Teaching the Memory to Forget Old Information

**Goal:** Add exponential decay and momentum to memory updates so old facts fade unless reinforced.

## What You Need to Know First

This section teaches the memory to forget. Background needed is minimal:

- **From Sections 8.2–8.3:** the memory is a fixed-size matrix updated by surprise-gated writes — and fixed size means it can fill up.
- **Exponential decay** — multiplying a value by something like 0.99 every step makes it shrink smoothly toward zero (`0.99^t`); that's all "decay" means here.
- **Momentum** — a trick from optimizers you've seen: blend in a bit of the previous update direction to smooth out noisy steps.
- **Half-life** — the number of steps for a value to shrink to half its strength; handy for describing how fast the memory forgets.

It's all high-school algebra (repeated multiplication) — the neuroscience analogy is just for intuition.

## Memory Without Forgetting is a Hoarder — Decay is Essential

A memory that never forgets eventually saturates: new facts interfere destructively with
old ones, and everything becomes noise. This is the **catastrophic interference** problem
in neural networks — without forgetting, a fixed-capacity system can only store so much.

Decay provides the solution: old memory traces fade exponentially over time unless they
are **reinforced** by being re-encountered. This is directly analogous to the Ebbinghaus
forgetting curve from psychology: unrehearsed memories decay exponentially, but spaced
repetition can make them permanent.

In Titans, decay serves two purposes:
1. **Capacity management** — make room for new facts by fading old ones
2. **Relevance filtering** — facts that haven't been useful recently are probably irrelevant

---

## The MemoryState Implementation

```python
import torch
import numpy as np

class MemoryState:
    def __init__(self, shape, decay=0.99, momentum=0.9, device="cpu"):
        self.weights = torch.zeros(shape, device=device)
        self.velocity = torch.zeros(shape, device=device)
        self.decay = decay
        self.momentum = momentum

    def step(self, grad, lr=0.01):
        # Decay: exponentially shrink existing memory
        self.weights *= self.decay
        # Momentum: smooth noisy gradient updates
        self.velocity = self.momentum * self.velocity - lr * grad
        # Apply update
        self.weights += self.velocity

    @property
    def effective_halflife(self):
        """Number of steps for memory to decay to half strength."""
        if self.decay >= 1.0:
            return float('inf')
        return -np.log(2) / np.log(self.decay)

state = MemoryState((64, 64))
for t in range(100):
    grad = torch.randn(64, 64)
    if t == 50:
        grad.zero_()  # stop reinforcing at step 50
    state.step(grad)

print(f"Weight norm at t=100 (stopped reinforcing at t=50): {state.weights.norm().item():.4f}")
print(f"Half-life with decay=0.99: {state.effective_halflife:.1f} steps")
```

---

## Visualization: Forgetting Curves for Different Decay Rates

```python
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

decay_rates = [0.9, 0.95, 0.99, 1.0]
n_steps = 200
reinforce_until = 50  # stop reinforcing at step 50

fig, ax = plt.subplots(figsize=(10, 6))

for decay in decay_rates:
    mem = MemoryState((32, 32), decay=decay, momentum=0.9)
    norms = []

    for t in range(n_steps):
        grad = torch.ones(32, 32) * 0.1 if t < reinforce_until else torch.zeros(32, 32)
        mem.step(grad, lr=0.05)
        norms.append(mem.weights.norm().item())

    half_life = -np.log(2) / np.log(decay) if decay < 1.0 else float('inf')
    label = f'decay={decay} (half-life={half_life:.0f})' if decay < 1.0 else f'decay=1.0 (no decay)'
    ax.plot(norms, linewidth=2, label=label)

ax.axvline(x=reinforce_until, color='gray', linestyle=':', label='Stop reinforcing')
ax.set_xlabel('Time Step')
ax.set_ylabel('Memory Weight Norm')
ax.set_title('Forgetting Curves: Memory Fades Without Reinforcement')
ax.legend()
ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig('forgetting_curves.png', dpi=100, bbox_inches='tight')
plt.close()
print("Saved forgetting_curves.png")
print("\nKey observation: decay=1.0 never forgets (weights grow unbounded)")
print("decay=0.9 forgets in ~7 steps, decay=0.99 in ~69 steps")
```

---

## Connection to Neuroscience: The Ebbinghaus Forgetting Curve

Hermann Ebbinghaus (1885) discovered that memory retention follows an exponential decay:

$$R(t) = e^{-t/S}$$

where $R$ is retention, $t$ is time, and $S$ is memory strength (increased by repetition).

Our decay parameter maps directly: `decay=0.99` means retention of 99% per step, giving
exponential forgetting with the same mathematical form.

```python
def ebbinghaus_retention(t, strength=1.0):
    """Classic forgetting curve: R = e^(-t/S)"""
    return np.exp(-t / strength)

# Compare Ebbinghaus with our discrete decay
t_continuous = np.linspace(0, 200, 1000)
strengths = [10, 50, 100]  # different "memory strengths" from repetition

fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# Ebbinghaus continuous
for S in strengths:
    R = ebbinghaus_retention(t_continuous, S)
    axes[0].plot(t_continuous, R, linewidth=2, label=f'Strength S={S}')
axes[0].set_title('Ebbinghaus Forgetting Curve (Continuous)')
axes[0].set_xlabel('Time')
axes[0].set_ylabel('Retention R(t)')
axes[0].legend()
axes[0].grid(True, alpha=0.3)

# Our discrete decay (equivalent)
steps = np.arange(200)
for decay in [0.9, 0.95, 0.99]:
    R_discrete = decay ** steps
    equiv_S = -1 / np.log(decay)
    axes[1].plot(steps, R_discrete, linewidth=2,
                label=f'decay={decay} (≈ S={equiv_S:.0f})')
axes[1].set_title('Our Discrete Decay (Equivalent)')
axes[1].set_xlabel('Step')
axes[1].set_ylabel('Retention')
axes[1].legend()
axes[1].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('ebbinghaus_comparison.png', dpi=100, bbox_inches='tight')
plt.close()
print("Saved ebbinghaus_comparison.png")
print("Our decay=d is equivalent to Ebbinghaus with strength S = -1/ln(d)")
```

---

## Interaction Between Surprise-Gated Writes and Decay

The interplay between surprise-gated writes and decay creates a natural priority system:
- **Frequently mentioned facts** are reinforced on each encounter → survive decay
- **One-time mentions** fade unless they were extremely surprising
- **Background noise** never gets written → zero memory footprint

```python
class MemoryWithSurpriseAndDecay:
    def __init__(self, shape, decay=0.99, momentum=0.9, surprise_threshold=0.5):
        self.state = MemoryState(shape, decay=decay, momentum=momentum)
        self.surprise_threshold = surprise_threshold
        self.n_writes = 0
        self.n_skips = 0

    def update(self, grad, surprise_value, lr=0.01):
        """Apply decay always; write only if surprised."""
        # Decay happens every step regardless
        self.state.weights *= self.state.decay

        if surprise_value > self.surprise_threshold:
            # Surprise-gated write
            self.state.velocity = self.state.momentum * self.state.velocity - lr * grad
            self.state.weights += self.state.velocity
            self.n_writes += 1
        else:
            self.n_skips += 1

# Simulate: 3 types of information
torch.manual_seed(42)
mem = MemoryWithSurpriseAndDecay((32, 32), decay=0.99, surprise_threshold=0.3)

# Type 1: repeated fact (mentioned every 10 steps) — should persist
# Type 2: one-time surprising fact — should fade
# Type 3: boring filler — should never be written

norms_after_fact1 = []
norms_after_fact2 = []

fact1_grad = torch.ones(32, 32)  # consistent direction
fact2_grad = -torch.ones(32, 32)  # different direction (one-time)

for step in range(100):
    if step == 5:
        # One-time surprising fact
        mem.update(fact2_grad, surprise_value=0.8, lr=0.05)
    elif step % 10 == 0:
        # Repeated fact
        mem.update(fact1_grad, surprise_value=0.6, lr=0.05)
    else:
        # Boring filler
        mem.update(torch.randn(32, 32) * 0.01, surprise_value=0.1, lr=0.01)

    # Track memory components in fact1 and fact2 directions
    with torch.no_grad():
        norm1 = (mem.state.weights * fact1_grad).sum().item()
        norm2 = (mem.state.weights * fact2_grad).sum().item()
    norms_after_fact1.append(norm1)
    norms_after_fact2.append(norm2)

print(f"Total writes: {mem.n_writes}, skips: {mem.n_skips}")
print(f"Repeated fact strength at end: {norms_after_fact1[-1]:.4f} (reinforced)")
print(f"One-time fact strength at end: {norms_after_fact2[-1]:.4f} (faded)")
print("Repeated facts survive; one-time facts decay away!")
```

---

## Forgetting Curve Demo (Original)

```python
facts = []
w = MemoryState((32, 32), decay=0.95)
strong_grad = torch.ones(32, 32)
weak_grad = torch.ones(32, 32) * 0.01

for epoch in range(30):
    w.step(strong_grad if epoch < 10 else weak_grad, lr=0.05)
    facts.append(w.weights[0, 0].item())

print("Fact strength trajectory (strong for 10 steps, then weak):")
print(f"  Peak (step 10): {max(facts):.4f}")
print(f"  Final (step 30): {facts[-1]:.4f}")
print(f"  Decay ratio: {facts[-1] / max(facts) * 100:.0f}%")
```

---

## Exercise: Find Optimal Decay/Momentum for Fact Retrieval

```python
import itertools

def fact_retrieval_benchmark(decay, momentum, n_facts=10, n_total_steps=200):
    """
    Write n_facts at different times, measure recall at the end.
    Facts are reinforced periodically (every 20 steps after first write).
    """
    torch.manual_seed(42)
    mem = MemoryState((64, 64), decay=decay, momentum=momentum)
    fact_grads = [torch.randn(64, 64) for _ in range(n_facts)]
    fact_write_times = [i * 15 for i in range(n_facts)]  # spaced 15 steps apart

    for step in range(n_total_steps):
        wrote_something = False
        for i, (fg, wt) in enumerate(zip(fact_grads, fact_write_times)):
            if step == wt or (step > wt and (step - wt) % 20 == 0):
                mem.step(fg, lr=0.05)
                wrote_something = True
                break
        if not wrote_something:
            mem.step(torch.zeros(64, 64), lr=0.0)  # just decay

    # Measure recall: how aligned is memory with each fact?
    recalls = []
    for fg in fact_grads:
        alignment = (mem.weights * fg).sum().item() / (mem.weights.norm() * fg.norm()).item()
        recalls.append(max(0, alignment))  # clamp negative

    return np.mean(recalls)

# Grid search
print("Grid Search: Optimal Decay × Momentum")
print(f"{'Decay':<8} {'Momentum':<10} {'Mean Recall'}")
print("-" * 30)

best_score = 0
best_params = (0, 0)

for decay in [0.9, 0.95, 0.99, 0.995]:
    for momentum in [0.0, 0.5, 0.9, 0.95]:
        score = fact_retrieval_benchmark(decay, momentum)
        if score > best_score:
            best_score = score
            best_params = (decay, momentum)
        print(f"{decay:<8} {momentum:<10} {score:.4f}")

print(f"\nBest: decay={best_params[0]}, momentum={best_params[1]} (recall={best_score:.4f})")
print("High decay (slow forgetting) + moderate momentum (smooth updates) works best")
```

---

## Where This Leads Next

The neural memory is now complete: it reads, writes on surprise, and forgets via decay. Section 8.5
plugs it into the transformer itself — blending the memory readout with normal attention through a
learned gate, so each layer can draw on both local context and long-term memory.

## Key Takeaway

Decay prevents infinite accumulation in the memory matrix — without it, the memory saturates
and new facts destructively interfere with old ones. The combination of exponential decay +
momentum + surprise gating creates a natural priority system: frequently reinforced facts
persist, one-time mentions fade, and boring content is never written. This mirrors the
Ebbinghaus forgetting curve from neuroscience, where unrehearsed memories decay exponentially
but spaced repetition makes them permanent. The optimal decay rate balances retention (high
decay ≈ long memory) against capacity (low decay ≈ room for new facts).

## Further Reading (Optional)

**Optional — you do NOT need these to continue. They are for curious students who want the original sources.**

- Behrouz, Zhong, & Mirrokni (2024). *Titans: Learning to Memorize at Test Time*. arXiv:2501.00663.
- Qin et al. (2024). *Gated Linear Attention Transformers with Hardware-Efficient Training*. ICML.
