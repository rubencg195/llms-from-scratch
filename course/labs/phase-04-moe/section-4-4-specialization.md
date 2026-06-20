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

# Section 4.4: Tracking Expert Specialization

**Goal:** Log which experts activate on OpenWebText vs Glaive batches and visualize specialization.

## Why Do We Expect Experts to Specialize?

The combination of sparse routing and load balancing creates a natural pressure toward
specialization. Here's the intuition:

1. **Different data domains have different statistical patterns.** JSON tokens (brackets,
   colons, quoted keys) have very different co-occurrence statistics from narrative prose.
2. **The router learns to group similar tokens.** Through backpropagation, the router
   discovers that certain hidden-state patterns benefit from certain expert weights.
3. **Load balancing prevents collapse.** Without it, all tokens would go to one expert.
   With it, experts are forced to "claim" distinct token populations.
4. **Each expert's weights adapt to its assigned tokens.** Over training, Expert 0 might
   develop weights that are excellent for JSON structure tokens, while Expert 2 excels at
   natural language continuation.

This isn't guaranteed — experts may learn overlapping functions, especially with few
training steps or too-strong load balancing. But when it works, you get interpretable
functional specialization: different experts handle different "types" of language.

## Expert Statistics Tracker

```python
import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
import numpy as np
from collections import defaultdict

class ExpertStats:
    def __init__(self, n_experts):
        self.n_experts = n_experts
        self.counts = defaultdict(lambda: torch.zeros(n_experts))

    def update(self, domain, expert_idx):
        c = torch.bincount(expert_idx.view(-1), minlength=self.n_experts).float()
        self.counts[domain] += c

    def report(self):
        for domain, c in self.counts.items():
            pct = (c / c.sum() * 100).tolist()
            print(domain, [f"{p:.1f}%" for p in pct])

stats = ExpertStats(4)
stats.update("webtext", torch.tensor([0, 0, 1, 2, 0, 3]))
stats.update("glaive", torch.tensor([3, 3, 2, 3, 1, 3]))
stats.report()
```

## MoE Layer with Load Balancing

```python
class ExpertFFN(nn.Module):
    def __init__(self, d_model, expansion=4):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(d_model, expansion * d_model),
            nn.GELU(),
            nn.Linear(expansion * d_model, d_model),
        )
    def forward(self, x):
        return self.net(x)

class MoELayer(nn.Module):
    def __init__(self, d_model, n_experts=4):
        super().__init__()
        self.router = nn.Linear(d_model, n_experts)
        self.experts = nn.ModuleList([ExpertFFN(d_model) for _ in range(n_experts)])

    def forward(self, x):
        B, T, C = x.shape
        logits = self.router(x)
        probs = torch.softmax(logits, dim=-1)
        choice = probs.argmax(dim=-1)
        out = torch.zeros_like(x)
        for e, expert in enumerate(self.experts):
            mask = (choice == e)
            if mask.any():
                out[mask] = expert(x[mask])
        return out, probs

def load_balance_loss(probs, expert_idx, n_experts):
    one_hot = F.one_hot(expert_idx, n_experts).float()
    f = one_hot.mean(dim=(0, 1))
    P = probs.mean(dim=(0, 1))
    return n_experts * (f * P).sum()
```

## Hook During Training

```python
def train_step_moe(moe_layer, x, domain, stats):
    y, probs = moe_layer(x)
    expert_idx = probs.argmax(dim=-1)
    stats.update(domain, expert_idx.cpu())
    lb = load_balance_loss(probs, expert_idx, len(moe_layer.experts))
    return y, lb
```

## Full MoE Training Loop on Mixed Domains (100 Steps)

We'll simulate training on two domains — "webtext" (prose-like) and "glaive" (structured/JSON-like) —
using different synthetic distributions to mimic domain differences.

```python
torch.manual_seed(42)
d_model = 128
n_experts = 4

moe = MoELayer(d_model, n_experts)
optimizer = torch.optim.Adam(moe.parameters(), lr=1e-3)
stats = ExpertStats(n_experts)

target_layer = nn.Linear(d_model, d_model)
for p in target_layer.parameters():
    p.requires_grad = False

step_log = {"step": [], "lm_loss": [], "balance_loss": [], "total_loss": []}
alpha = 0.01

for step in range(100):
    if step % 2 == 0:
        domain = "webtext"
        x = torch.randn(2, 32, d_model)
    else:
        domain = "glaive"
        x = torch.randn(2, 32, d_model) * 0.5
        x[:, :, :d_model // 4] *= 3.0

    y, probs = moe(x)
    expert_idx = probs.argmax(dim=-1)

    target = target_layer(x).detach()
    lm_loss = F.mse_loss(y, target)
    bl = load_balance_loss(probs, expert_idx, n_experts)
    loss = lm_loss + alpha * bl

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    stats.update(domain, expert_idx.cpu())

    step_log["step"].append(step)
    step_log["lm_loss"].append(lm_loss.item())
    step_log["balance_loss"].append(bl.item())
    step_log["total_loss"].append(loss.item())

    if step % 20 == 0:
        print(f"Step {step:3d} | LM: {lm_loss.item():.4f} | Balance: {bl.item():.4f} | Total: {loss.item():.4f}")

print("\nFinal expert utilization by domain:")
stats.report()
```

## Visualization: Expert Usage per Domain (Stacked Bar Chart)

```python
domains = list(stats.counts.keys())
n_experts_vis = stats.n_experts
expert_colors = ["#4C72B0", "#55A868", "#C44E52", "#8172B3"]

fig, ax = plt.subplots(figsize=(8, 5))

domain_data = {}
for domain in domains:
    c = stats.counts[domain]
    pct = (c / c.sum() * 100).numpy()
    domain_data[domain] = pct

x_pos = np.arange(len(domains))
bar_width = 0.5
bottoms = np.zeros(len(domains))

for e in range(n_experts_vis):
    values = [domain_data[d][e] for d in domains]
    ax.bar(x_pos, values, bar_width, bottom=bottoms,
           label=f"Expert {e}", color=expert_colors[e])
    for i, v in enumerate(values):
        if v > 5:
            ax.text(x_pos[i], bottoms[i] + v / 2, f"{v:.1f}%",
                    ha="center", va="center", fontsize=9, fontweight="bold", color="white")
    bottoms += values

ax.set_xticks(x_pos)
ax.set_xticklabels(domains, fontsize=12)
ax.set_ylabel("Percentage of Tokens (%)", fontsize=11)
ax.set_title("Expert Usage by Domain (Stacked)", fontsize=13)
ax.legend(loc="upper right")
ax.set_ylim(0, 105)
plt.tight_layout()
plt.show()
```

## What Do Experts Learn? — Analyzing Weight Norms

One way to understand what experts have learned is to examine their weight norms.
Experts that process more varied or larger-magnitude inputs tend to develop larger weights.
We can also look at which input dimensions each expert is most sensitive to.

```python
fig, axes = plt.subplots(1, 2, figsize=(12, 4))

weight_norms = []
for e, expert in enumerate(moe.experts):
    w1 = expert.net[0].weight
    w2 = expert.net[2].weight
    weight_norms.append({
        "expert": e,
        "W1_norm": w1.norm().item(),
        "W2_norm": w2.norm().item(),
        "W1_mean_abs": w1.abs().mean().item(),
    })
    print(f"Expert {e}: W1 norm={w1.norm().item():.2f}, W2 norm={w2.norm().item():.2f}")

axes[0].bar(
    range(n_experts),
    [w["W1_norm"] for w in weight_norms],
    color=expert_colors, alpha=0.8
)
axes[0].set_xlabel("Expert")
axes[0].set_ylabel("Frobenius Norm")
axes[0].set_title("First Layer Weight Norms")
axes[0].set_xticks(range(n_experts))

for e, expert in enumerate(moe.experts):
    w1 = expert.net[0].weight
    dim_sensitivity = w1.abs().mean(dim=0).detach().numpy()
    axes[1].plot(dim_sensitivity[:32], label=f"Expert {e}", color=expert_colors[e], alpha=0.7)
axes[1].set_xlabel("Input Dimension (first 32)")
axes[1].set_ylabel("Mean |Weight|")
axes[1].set_title("Per-Dimension Sensitivity by Expert")
axes[1].legend(fontsize=8)

plt.tight_layout()
plt.show()
```

## Expert Merging — When Two Experts Learn Similar Things

Sometimes two experts converge to similar functions. This wastes capacity. We can detect
this by measuring **cosine similarity** between expert weight matrices. If two experts
are nearly identical, one could be pruned or reinitialized.

```python
def expert_similarity(experts):
    """Compute pairwise cosine similarity between expert weight matrices."""
    n = len(experts)
    sim_matrix = torch.zeros(n, n)
    flat_weights = []
    for expert in experts:
        w = torch.cat([p.view(-1) for p in expert.parameters()])
        flat_weights.append(w)

    for i in range(n):
        for j in range(n):
            sim_matrix[i, j] = F.cosine_similarity(
                flat_weights[i].unsqueeze(0),
                flat_weights[j].unsqueeze(0)
            ).item()
    return sim_matrix

sim = expert_similarity(moe.experts)
print("Expert similarity matrix:")
print(sim.numpy().round(3))

fig, ax = plt.subplots(figsize=(5, 4))
im = ax.imshow(sim.numpy(), cmap="RdYlGn", vmin=-1, vmax=1)
for i in range(n_experts):
    for j in range(n_experts):
        ax.text(j, i, f"{sim[i, j]:.2f}", ha="center", va="center", fontsize=10)
ax.set_xticks(range(n_experts))
ax.set_yticks(range(n_experts))
ax.set_xticklabels([f"Expert {i}" for i in range(n_experts)])
ax.set_yticklabels([f"Expert {i}" for i in range(n_experts)])
ax.set_title("Expert Weight Cosine Similarity")
plt.colorbar(im, ax=ax, shrink=0.8)
plt.tight_layout()
plt.show()

high_sim_pairs = []
for i in range(n_experts):
    for j in range(i + 1, n_experts):
        if sim[i, j] > 0.8:
            high_sim_pairs.append((i, j, sim[i, j].item()))

if high_sim_pairs:
    print("\nMerge candidates (cosine sim > 0.8):")
    for i, j, s in high_sim_pairs:
        print(f"  Expert {i} ↔ Expert {j}: similarity = {s:.3f}")
else:
    print("\nNo expert pairs with similarity > 0.8 — experts are reasonably diverse.")
```

## Exercise: Train Longer and Track Specialization Over Time

Train for more steps and log expert utilization per domain at regular intervals
to observe how specialization evolves.

```python
torch.manual_seed(123)
moe_long = MoELayer(d_model, n_experts)
optimizer_long = torch.optim.Adam(moe_long.parameters(), lr=1e-3)

checkpoints = []
checkpoint_interval = 50
n_steps = 300

for step in range(n_steps):
    if step % 2 == 0:
        domain = "webtext"
        x = torch.randn(2, 32, d_model)
    else:
        domain = "glaive"
        x = torch.randn(2, 32, d_model) * 0.5
        x[:, :, :d_model // 4] *= 3.0

    y, probs = moe_long(x)
    expert_idx = probs.argmax(dim=-1)

    target = target_layer(x).detach()
    lm_loss = F.mse_loss(y, target)
    bl = load_balance_loss(probs, expert_idx, n_experts)
    loss = lm_loss + 0.01 * bl

    optimizer_long.zero_grad()
    loss.backward()
    optimizer_long.step()

    if step % checkpoint_interval == 0 or step == n_steps - 1:
        eval_stats = ExpertStats(n_experts)
        for eval_domain, gen_fn in [
            ("webtext", lambda: torch.randn(4, 32, d_model)),
            ("glaive", lambda: torch.randn(4, 32, d_model) * 0.5),
        ]:
            with torch.no_grad():
                ex = gen_fn()
                _, ep = moe_long(ex)
                ei = ep.argmax(dim=-1)
                eval_stats.update(eval_domain, ei)
        checkpoint = {"step": step}
        for d in ["webtext", "glaive"]:
            c = eval_stats.counts[d]
            checkpoint[d] = (c / c.sum()).tolist()
        checkpoints.append(checkpoint)

fig, axes = plt.subplots(1, len(checkpoints), figsize=(3.5 * len(checkpoints), 4), sharey=True)
if len(checkpoints) == 1:
    axes = [axes]

for ax_i, cp in zip(axes, checkpoints):
    x_pos = np.arange(n_experts)
    w = 0.35
    ax_i.bar(x_pos - w/2, cp["webtext"], w, label="webtext", color="#4C72B0", alpha=0.8)
    ax_i.bar(x_pos + w/2, cp["glaive"], w, label="glaive", color="#C44E52", alpha=0.8)
    ax_i.set_title(f"Step {cp['step']}")
    ax_i.set_xticks(x_pos)
    ax_i.set_xticklabels([f"E{i}" for i in range(n_experts)])
    ax_i.axhline(y=1/n_experts, color="black", linestyle="--", alpha=0.3)
    if ax_i == axes[0]:
        ax_i.set_ylabel("Token Fraction")
    ax_i.legend(fontsize=7)

plt.suptitle("Expert Specialization Over Training", fontsize=13, y=1.02)
plt.tight_layout()
plt.show()
```

## Save Results

```python
import json
import os
os.makedirs("logs", exist_ok=True)

results = {}
for domain, c in stats.counts.items():
    results[domain] = (c / c.sum() * 100).tolist()

with open("logs/expert_specialization.json", "w") as f:
    json.dump(results, f, indent=2)

print("Saved expert utilization to logs/expert_specialization.json")
print(json.dumps(results, indent=2))
```

---

## Key Takeaway

Expert specialization is the **payoff** of MoE — it's why we use multiple experts instead
of one larger FFN. Through training on mixed-domain data with load balancing, different
experts naturally develop affinity for different token types. We can track this by logging
per-domain routing statistics and visualizing them as stacked bar charts. Weight norm
analysis and cosine similarity between experts reveal **what** experts have learned and
whether any are redundant (candidates for merging or reinitialization). Specialization
deepens with more training — early in training the router is nearly random, but after
hundreds of steps, clear domain preferences emerge.
