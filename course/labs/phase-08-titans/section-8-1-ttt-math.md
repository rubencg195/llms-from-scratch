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

# Section 8.1: Test-Time Training (TTT) — The Math of Learning While Chatting

**Goal:** Perform one gradient step on a tiny memory network at inference time when prediction error is high.

## What You Need to Know First

Phase 8 reuses the training tools you already know — now applied at an unexpected time. You need:

- **Gradient descent and a "gradient step"** — from the training phases: compute how wrong a prediction is (the loss), then nudge the weights in the direction that lowers it. That's all an "inner-loop step" is.
- **`nn.Linear` / a small MLP** — a tiny network whose weights we'll treat as the model's notepad.
- **Training time vs inference time** — normally weights freeze after training; the twist here is updating a few weights *during* inference (while chatting).
- **KV cache (from the attention phases)** — the running store of past tokens that lets attention look back; here it's the thing we're trying to replace with something fixed-size.

The notation $\nabla_{\theta}\mathcal{L}$ just means "the gradient of the loss with respect to the weights" — the direction to nudge them.

## What If the Model Could Learn During the Conversation?

This is the radical idea behind Test-Time Training (TTT): instead of treating model weights
as frozen after pretraining, we allow a small subset of parameters to *update themselves*
during inference. When the model encounters surprising new information ("My new passcode
is 7742"), it performs a gradient step to memorize it — just as a human would.

Standard transformers handle new information through the KV cache: stuff it into the context
window and hope attention retrieves it later. But the KV cache grows linearly with sequence
length, eventually exceeding memory. TTT offers a fixed-size alternative: a small neural
network whose weights *are* the memory, updated by gradient descent at test time.

**Key distinction:**
- **Outer loop** (pretraining): optimize all parameters on the training corpus
- **Inner loop** (TTT): optimize only the memory network on the current conversation

---

## The Inner-Loop Objective

The TTT inner loop has a simple objective: minimize prediction error on the current input.
Given a new token $x_t$ and a target $y_t$ (what the memory should produce), we compute:

$$\mathcal{L}_{\text{inner}}(\theta_m) = \| f_{\theta_m}(x_t) - y_t \|^2$$

Then update: $\theta_m \leftarrow \theta_m - \eta \nabla_{\theta_m} \mathcal{L}_{\text{inner}}$

This is literally one step of gradient descent, happening *during inference*.

```python
import torch
import torch.nn as nn

class MemoryMLP(nn.Module):
    def __init__(self, d):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(d, d), nn.Tanh(), nn.Linear(d, d))

    def forward(self, x):
        return self.net(x)

d = 64
memory = MemoryMLP(d)
x_new = torch.randn(d)
target = torch.randn(d)  # what memory should recall

# Inner-loop TTT step
inner_lr = 0.01
pred = memory(x_new)
surprise = (pred - target).pow(2).mean()
grads = torch.autograd.grad(surprise, memory.parameters(), create_graph=False)
with torch.no_grad():
    for p, g in zip(memory.parameters(), grads):
        p -= inner_lr * g

pred2 = memory(x_new)
print("surprise before:", surprise.item())
print("surprise after one TTT step:", (pred2 - target).pow(2).mean().item())
```

---

## Outer-Loop vs Inner-Loop Gradients

The outer loop and inner loop solve different problems with different gradient signals:

| | Outer Loop (Pretraining) | Inner Loop (TTT) |
|---|---|---|
| **When** | Before deployment | During inference |
| **Data** | Entire training corpus | Current conversation |
| **Parameters** | All model weights | Only memory network |
| **Objective** | Next-token prediction | Minimize surprise on new facts |
| **Learning rate** | Small (1e-4 to 1e-3) | Larger (0.01 to 0.1) |
| **Steps** | Millions | 1-10 per token |

```python
# Demonstrate the difference in gradient magnitudes and directions
torch.manual_seed(42)
memory_compare = MemoryMLP(d)

# Simulate outer-loop gradient (average over many samples)
outer_grads = []
for _ in range(100):
    x = torch.randn(d)
    t = torch.randn(d)
    pred = memory_compare(x)
    loss = (pred - t).pow(2).mean()
    grads = torch.autograd.grad(loss, memory_compare.parameters(), retain_graph=False)
    outer_grads.append([g.clone() for g in grads])

avg_outer_grad = [sum(g[i] for g in outer_grads) / len(outer_grads)
                  for i in range(len(outer_grads[0]))]

# Simulate inner-loop gradient (single sample, high surprise)
x_surprising = torch.randn(d) * 3  # unusual input
t_surprising = torch.randn(d) * 3
pred = memory_compare(x_surprising)
inner_loss = (pred - t_surprising).pow(2).mean()
inner_grads = torch.autograd.grad(inner_loss, memory_compare.parameters())

print("Gradient magnitude comparison:")
for i, (og, ig) in enumerate(zip(avg_outer_grad, inner_grads)):
    print(f"  Layer {i}: outer={og.norm().item():.4f}, inner={ig.norm().item():.4f}, "
          f"ratio={ig.norm().item() / max(og.norm().item(), 1e-8):.1f}x")
print("\nInner-loop gradients are larger (single surprising sample vs averaged corpus)")
```

---

## Multi-Step TTT: Improvement Over 1, 5, 10 Inner Steps

More inner steps = better memorization of the current fact, but more compute per token.

```python
def ttt_multi_step(memory_net, x, target, inner_lr=0.01, n_steps=1):
    """Perform n_steps of TTT on the memory network."""
    surprises = []
    for step in range(n_steps):
        pred = memory_net(x)
        surprise = (pred - target).pow(2).mean()
        surprises.append(surprise.item())
        grads = torch.autograd.grad(surprise, memory_net.parameters(), create_graph=False)
        with torch.no_grad():
            for p, g in zip(memory_net.parameters(), grads):
                p -= inner_lr * g
    return surprises

# Compare different step counts
print(f"{'Steps':<8} {'Initial Surprise':<20} {'Final Surprise':<20} {'Reduction'}")
print("-" * 65)

for n_steps in [1, 5, 10, 20]:
    torch.manual_seed(0)
    mem = MemoryMLP(d)
    x_test = torch.randn(d)
    t_test = torch.randn(d)
    surprises = ttt_multi_step(mem, x_test, t_test, inner_lr=0.02, n_steps=n_steps)
    reduction = (1 - surprises[-1] / surprises[0]) * 100
    print(f"{n_steps:<8} {surprises[0]:<20.6f} {surprises[-1]:<20.6f} {reduction:.1f}%")
```

---

## Visualization: Surprise Decreasing Over TTT Steps

```python
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

torch.manual_seed(0)
mem_viz = MemoryMLP(d)
x_viz = torch.randn(d)
t_viz = torch.randn(d)
surprises_20 = ttt_multi_step(mem_viz, x_viz, t_viz, inner_lr=0.02, n_steps=20)

fig, ax = plt.subplots(figsize=(8, 5))
ax.plot(range(1, 21), surprises_20, 'o-', color='steelblue', linewidth=2, markersize=6)
ax.set_xlabel('TTT Inner Steps')
ax.set_ylabel('Surprise (MSE)')
ax.set_title('Test-Time Training: Surprise Decreases with Inner Steps')
ax.axhline(y=surprises_20[0] * 0.1, color='red', linestyle='--',
           label='90% reduction threshold')
ax.legend()
ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig('ttt_surprise.png', dpi=100, bbox_inches='tight')
plt.close()
print("Saved ttt_surprise.png")
print(f"After 20 steps: {(1 - surprises_20[-1]/surprises_20[0])*100:.1f}% surprise reduction")
```

---

## Computational Cost of TTT

TTT is not free — each inner step requires a full forward+backward pass through the memory
network. For a memory with $M$ parameters, each TTT step costs ~$3M$ FLOPs (forward + backward).

```python
def estimate_ttt_cost(d_mem, n_steps_per_token, tokens_per_second=12.5):
    """Estimate TTT compute cost."""
    # Memory MLP: Linear(d, d) + Tanh + Linear(d, d)
    mem_params = d_mem * d_mem + d_mem + d_mem * d_mem + d_mem  # two linear layers
    flops_per_step = 3 * mem_params  # forward + backward ≈ 3x forward
    flops_per_token = flops_per_step * n_steps_per_token
    flops_per_second = flops_per_token * tokens_per_second

    return {
        'mem_params': mem_params,
        'flops_per_token': flops_per_token,
        'flops_per_second': flops_per_second,
        'ms_per_token_approx': flops_per_token / 1e9 * 0.5,  # rough: 2 TFLOPS on consumer GPU
    }

print("TTT Computational Cost Analysis:")
print(f"{'d_mem':<8} {'Steps':<8} {'Params':<12} {'FLOPS/token':<15} {'~ms/token'}")
print("-" * 55)
for d_mem in [64, 128, 256]:
    for steps in [1, 5]:
        cost = estimate_ttt_cost(d_mem, steps)
        print(f"{d_mem:<8} {steps:<8} {cost['mem_params']:<12,} "
              f"{cost['flops_per_token']:<15,} {cost['ms_per_token_approx']:.3f}")

print("\nConclusion: TTT adds <1ms per token for d_mem=64 — negligible on modern GPUs")
```

---

## TTT vs Static KV Cache

| | Static KV Cache | TTT Memory |
|---|---|---|
| Memory usage | ∝ sequence length | Fixed param count |
| Past access | Read-only | **Writable** weights |
| 100K tokens | 100K × 2 × d bytes | Same as 1 token |
| New fact | Hope attention finds it | Gradient-write it |

```python
# Concrete comparison
d_model_kv = 512
n_layers_kv = 8
n_heads_kv = 8
head_dim = d_model_kv // n_heads_kv

for seq_len in [1000, 10000, 100000]:
    kv_bytes = 2 * n_layers_kv * seq_len * d_model_kv * 2  # 2 for K,V; 2 for fp16
    ttt_bytes = 64 * 64 * 2 * 4  # two 64x64 matrices (memory MLP), fp32
    print(f"Seq={seq_len:>7}: KV={kv_bytes/1e6:.1f} MB, TTT={ttt_bytes/1e6:.3f} MB, "
          f"ratio={kv_bytes/ttt_bytes:.0f}x")

print("\nTitans uses surprise-gated TTT instead of unbounded cache growth.")
```

---

## Exercise: Implement TTT with Gradient Clipping for Stability

Large surprise values can produce exploding gradients in the inner loop. Add gradient
clipping to make TTT stable:

```python
def ttt_step_clipped(memory_net, x, target, inner_lr=0.01, max_grad_norm=1.0):
    """One TTT step with gradient clipping for stability."""
    pred = memory_net(x)
    surprise = (pred - target).pow(2).mean()
    grads = torch.autograd.grad(surprise, memory_net.parameters(), create_graph=False)

    # Compute total gradient norm
    total_norm = torch.sqrt(sum(g.pow(2).sum() for g in grads))

    # Clip if necessary
    clip_coeff = max_grad_norm / (total_norm + 1e-8)
    clip_coeff = min(clip_coeff, 1.0)

    with torch.no_grad():
        for p, g in zip(memory_net.parameters(), grads):
            p -= inner_lr * g * clip_coeff

    return surprise.item(), total_norm.item(), clip_coeff

# Test: normal input vs adversarial large input
torch.manual_seed(42)
mem_clip = MemoryMLP(d)

# Normal input
x_normal = torch.randn(d)
t_normal = torch.randn(d)
s1, norm1, clip1 = ttt_step_clipped(mem_clip, x_normal, t_normal)

# Large surprising input (would cause gradient explosion without clipping)
x_large = torch.randn(d) * 10
t_large = torch.randn(d) * 10
s2, norm2, clip2 = ttt_step_clipped(mem_clip, x_large, t_large)

print("Gradient Clipping Analysis:")
print(f"  Normal input:  surprise={s1:.4f}, grad_norm={norm1:.4f}, clip_coeff={clip1:.4f}")
print(f"  Large input:   surprise={s2:.4f}, grad_norm={norm2:.4f}, clip_coeff={clip2:.4f}")
print(f"\n  Without clipping, the large input would update weights by {norm2:.1f}x too much!")
```

---

## Where This Leads Next

We've shown a memory network *can* learn during inference, but we used a bare MLP. Section 8.2
turns that idea into a proper, reusable `NeuralMemory` module with clean read and write paths —
the actual building block the rest of Phase 8 plugs into.

## Key Takeaway

Test-Time Training is a paradigm shift: the model's memory network *learns during inference*
via gradient descent. Each surprising token triggers 1-10 inner optimization steps, writing
new information into fixed-size parameters rather than appending to a growing KV cache.
The cost is modest (~1ms/token for d=64) but the benefit is profound: O(1) memory for
arbitrarily long contexts. Combined with surprise gating (Section 8.3), only genuinely
novel information triggers writes — mundane tokens pass through without updating the memory.

## Further Reading (Optional)

**Optional — you do NOT need these to continue. They are for curious students who want the original sources.**

- Sun et al. (2024). *Learning to (Learn at Test Time): RNNs with Expressive Hidden States*. arXiv:2407.04620.
- Finn, Abbeel, & Levine (2017). *Model-Agnostic Meta-Learning (MAML)*. ICML.
