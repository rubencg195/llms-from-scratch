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

# Section 4.2: Creating 4 "Expert" Sub-Networks

**Goal:** Replace dense FFN with four parallel experts; only one runs per token (same FLOPs as single FFN).

## What You Need to Know First

This section combines pieces you have already seen — no new outside knowledge required:

- **The router from Section 4.1** — the tiny linear layer + softmax that picks one expert per token.
- **A feed-forward network (FFN)** — the standard "expand then shrink" block of two linear layers with an activation in between, which you built in earlier phases.
- **Parameters vs. compute (FLOPs)** — parameters are how many numbers the model stores; FLOPs are how much arithmetic it does per token. MoE keeps FLOPs low while growing parameters.

If those are familiar, you are ready.

## The Receptionist + Specialists Analogy

Picture a medical clinic with **one receptionist** and **four specialist doctors**. Every patient
walks through the same front door and talks to the receptionist (the router). The receptionist
doesn't examine anyone — they just read the intake form and say "go to Room 2."

The patient then sees **only one doctor** (one expert FFN). The other three doctors sit idle
for that patient — they're available for different patients arriving at the same time.

This is exactly how a Mixture-of-Experts layer works:

- **Receptionist (Router):** A tiny `nn.Linear(d_model, n_experts)` — examines each token's
  hidden state and assigns it to one specialist.
- **Specialists (Expert FFNs):** Four identical-architecture but independently-parameterized
  feedforward networks. Each one *could* handle any token, but they develop different
  specializations through training.
- **The key insight:** Total parameters = 4× a single FFN, but compute per token = 1× FFN.
  You get the **capacity** of a much larger model at the **cost** of a smaller one.

## Expert and MoE Layer Implementation

```python
import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
import time

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
        choice = probs.argmax(dim=-1)  # (B, T)
        out = torch.zeros_like(x)
        for e, expert in enumerate(self.experts):
            mask = choice == e
            if mask.any():
                out[mask] = expert(x[mask])
        return out, probs

d_model = 512
moe = MoELayer(d_model)
x = torch.randn(2, 32, d_model, device="cuda" if torch.cuda.is_available() else "cpu")
moe = moe.to(x.device)
y, probs = moe(x)
print("output:", y.shape)
```

## Capacity Factor: What Happens When an Expert Is Full?

In production MoE systems, each expert has a **capacity** — a maximum number of tokens it can
process in one forward pass. The capacity factor $C_f$ determines this limit:

$$\text{expert\_capacity} = C_f \times \frac{T}{E}$$

where $T$ is the total tokens and $E$ is the number of experts. If tokens are perfectly balanced,
each expert gets $T/E$ tokens. The capacity factor $C_f$ provides headroom (typically 1.0–1.5).

**When an expert's buffer is full:**
- Extra tokens assigned to that expert are **dropped** — they pass through unchanged (identity).
- This is a form of regularization but also means information is lost.
- Too-small $C_f$ → many dropped tokens → degraded quality.
- Too-large $C_f$ → wasted memory from oversized buffers.

```python
def route_with_capacity(x, experts, router_linear, capacity_factor=1.25):
    B, T, C = x.shape
    n_experts = len(experts)
    logits = router_linear(x)
    probs = torch.softmax(logits, dim=-1)
    choice = probs.argmax(dim=-1)

    capacity = int(capacity_factor * T / n_experts)
    out = x.clone()
    dropped = 0

    for e, expert in enumerate(experts):
        mask = (choice == e)
        positions = mask.nonzero(as_tuple=False)
        if len(positions) > capacity:
            positions = positions[:capacity]
            dropped += mask.sum().item() - capacity
        if len(positions) > 0:
            indices = positions[:, 0], positions[:, 1]
            out[indices] = expert(x[indices])

    return out, probs, dropped

x_cap = torch.randn(1, 64, d_model, device=x.device)
moe_cap = MoELayer(d_model).to(x.device)
y_cap, p_cap, n_dropped = route_with_capacity(
    x_cap, moe_cap.experts, moe_cap.router, capacity_factor=1.0
)
print(f"Tokens dropped due to capacity overflow: {n_dropped}")
```

## Parameter Count: Total vs Active

```python
def count_params(m):
    return sum(p.numel() for p in m.parameters())

dense_ffn = ExpertFFN(d_model)
print(f"Single expert params:  {count_params(dense_ffn):,}")
print(f"MoE total params:      {count_params(moe):,}")
print(f"Router params:         {count_params(nn.Linear(d_model, 4)):,}")
print(f"Active per token ≈ single expert + router = {count_params(dense_ffn) + count_params(nn.Linear(d_model, 4)):,}")
```

## Visualization: Parameter Count Breakdown (Router vs Experts)

```python
router_params = count_params(nn.Linear(d_model, 4))
expert_params = [count_params(exp) for exp in moe.experts]

labels = [f"Expert {i}" for i in range(len(expert_params))] + ["Router"]
sizes = expert_params + [router_params]
colors = ["#4C72B0", "#55A868", "#C44E52", "#8172B3", "#FFA500"]

fig, ax = plt.subplots(figsize=(7, 7))
wedges, texts, autotexts = ax.pie(
    sizes, labels=labels, colors=colors, autopct="%1.1f%%",
    startangle=90, textprops={"fontsize": 11}
)
ax.set_title(f"MoE Parameter Distribution\n(Total: {sum(sizes):,} params)", fontsize=13)
plt.tight_layout()
plt.show()

print(f"\nRouter is only {router_params/sum(sizes)*100:.2f}% of total parameters")
print("Yet it controls 100% of the computation graph.")
```

## Comparing Forward Pass Time: Dense FFN vs MoE

Both a single dense FFN and our 4-expert MoE process the same tokens — but MoE has 4×
the parameters while activating only ~1× per token. Let's measure the wall-clock difference.

```python
dense = ExpertFFN(d_model).to(x.device)
moe_time_test = MoELayer(d_model).to(x.device)
x_bench = torch.randn(4, 128, d_model, device=x.device)

n_trials = 50

t0 = time.perf_counter()
for _ in range(n_trials):
    _ = dense(x_bench)
dense_time = (time.perf_counter() - t0) / n_trials

t0 = time.perf_counter()
for _ in range(n_trials):
    _ = moe_time_test(x_bench)
moe_time = (time.perf_counter() - t0) / n_trials

print(f"Dense FFN forward:  {dense_time*1000:.2f} ms")
print(f"MoE (4 experts) forward: {moe_time*1000:.2f} ms")
print(f"MoE / Dense ratio:  {moe_time/dense_time:.2f}x")
print(f"\nMoE has {count_params(moe_time_test)/count_params(dense):.1f}x more parameters")
print("but per-token compute is similar since only 1 expert runs per token.")
```

## Swap into Transformer Block

Replace `self.mlp` in Phase 1 `Block` with `MoELayer` — attention unchanged; VRAM at
inference tracks **one expert**, not four.

```python
class MoETransformerBlock(nn.Module):
    """Transformer block with MoE replacing the dense FFN."""
    def __init__(self, d_model, n_heads, n_experts=4):
        super().__init__()
        self.ln1 = nn.LayerNorm(d_model)
        self.attn = nn.MultiheadAttention(d_model, n_heads, batch_first=True)
        self.ln2 = nn.LayerNorm(d_model)
        self.moe = MoELayer(d_model, n_experts)

    def forward(self, x):
        # Self-attention with residual
        x_norm = self.ln1(x)
        attn_out, _ = self.attn(x_norm, x_norm, x_norm)
        x = x + attn_out
        # MoE FFN with residual
        x_norm = self.ln2(x)
        moe_out, probs = self.moe(x_norm)
        x = x + moe_out
        return x, probs

block = MoETransformerBlock(d_model=512, n_heads=8, n_experts=4).to(x.device)
x_block = torch.randn(2, 32, 512, device=x.device)
out_block, probs_block = block(x_block)
print("Block output shape:", out_block.shape)
print("Block total params:", f"{count_params(block):,}")
```

## Exercise: Vary Number of Experts and Compare

Experiment with different expert counts (2, 4, 8) and observe how total parameter count
and routing patterns change.

```python
results = []
torch.manual_seed(42)
x_vary = torch.randn(2, 64, d_model)

for n_exp in [2, 4, 8]:
    layer = MoELayer(d_model, n_experts=n_exp)
    y_v, probs_v = layer(x_vary)
    choice_v = probs_v.argmax(dim=-1)
    counts_v = torch.bincount(choice_v.view(-1), minlength=n_exp)
    utilization = (counts_v.float() / counts_v.sum()).tolist()
    total_p = count_params(layer)
    active_p = count_params(layer.experts[0]) + count_params(nn.Linear(d_model, n_exp))

    results.append({
        "n_experts": n_exp,
        "total_params": total_p,
        "active_params": active_p,
        "utilization": utilization,
    })
    print(f"\n--- {n_exp} Experts ---")
    print(f"  Total params:  {total_p:,}")
    print(f"  Active params: {active_p:,}")
    print(f"  Utilization:   {[f'{u:.1%}' for u in utilization]}")

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
ns = [r["n_experts"] for r in results]
ax1.bar(ns, [r["total_params"] for r in results], color="#4C72B0", alpha=0.7, label="Total")
ax1.bar(ns, [r["active_params"] for r in results], color="#55A868", alpha=0.7, label="Active per token")
ax1.set_xlabel("Number of Experts")
ax1.set_ylabel("Parameter Count")
ax1.set_title("Total vs Active Parameters")
ax1.legend()
ax1.set_xticks(ns)

for r in results:
    ax2.bar(
        [f"E{e}\n(n={r['n_experts']})" for e in range(r["n_experts"])],
        r["utilization"], alpha=0.7
    )
ax2.set_ylabel("Token Fraction")
ax2.set_title("Expert Utilization (Untrained)")
plt.tight_layout()
plt.show()
```

---

## Where This Leads Next

You now have several experts and a router, but nothing yet stops the router from sending almost every token to the *same* expert. In **Section 4.3** you will add a load-balancing loss — a small amount of extra math that encourages the router to spread tokens evenly so all of your experts actually get used.

## Key Takeaway

A Mixture-of-Experts layer multiplies model **capacity** (total parameters) without
proportionally increasing per-token **compute**. Each expert is a standard FFN; the router
picks one per token. The capacity factor prevents any single expert from being overwhelmed.
When swapped into a Transformer block, attention stays unchanged — only the FFN is replaced.
More experts means more parameters but the same per-token cost, which is the fundamental
scaling insight behind models like GShard, Switch Transformer, and Mixtral.

## Further Reading (Optional)

**Optional — you do NOT need these to continue. They are for curious students who want the original sources.**

- Jiang et al. (2024). *Mixtral of Experts*. arXiv:2401.04088.
- Lepikhin et al. (2020). *GShard*. arXiv:2006.16668.
