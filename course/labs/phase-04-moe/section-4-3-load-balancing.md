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

# Section 4.3: Load Balancing — Math to Stop the Router from Overworking Expert 1

**Goal:** Add auxiliary loss encouraging uniform expert utilization (Switch Transformer style).

## What You Need to Know First

This section is mostly about adding one extra term to the loss. You already have all the building blocks:

- **The router and experts** from Sections 4.1–4.2 — and the fact that the router learns only from the loss.
- **A loss function** — a single number measuring "how wrong" the model is, which we make smaller during training.
- **Adding up a probability** — the only math here is averaging numbers and multiplying two averages together; high-school algebra is enough.
- **An "auxiliary loss"** simply means a *second*, helper loss added on top of the main one to encourage good behavior (here: even use of experts).

No external background is needed — just these ideas.

## The Expert Collapse Problem

Without any balancing incentive, MoE training frequently **collapses** — the router learns
to send almost all tokens to a single expert, and the other experts receive near-zero
gradient updates. This is a self-reinforcing loop:

1. By random initialization, Expert 0 is slightly better than the rest.
2. The router sends more tokens to Expert 0 → Expert 0 gets more gradient updates.
3. Expert 0 improves further → the router sends **even more** tokens to Expert 0.
4. Experts 1–3 stagnate → eventually 99%+ of tokens go to Expert 0.

The result: you have 4× the parameters but effectively a **single** FFN. All the capacity
of MoE is wasted. This is **expert collapse**, and it's one of the key failure modes that
motivated the Switch Transformer's auxiliary load-balancing loss.

```python
import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
```

## Deriving the Switch Transformer Load Balancing Loss

The Switch Transformer paper (Fedus et al., 2021) introduces an auxiliary loss with two
components. Let's derive it step by step.

**Setup:** We have $E$ experts, a batch of $N = B \times T$ tokens, router probabilities
$p_{i,e}$ for token $i$ and expert $e$, and hard routing decisions $c_i = \arg\max_e p_{i,e}$
(the $\arg\max$ just picks *which* expert has the highest probability for that token).

**Step 1 — Fraction of tokens routed to each expert:**

$$f_e = \frac{1}{N} \sum_{i=1}^{N} \mathbf{1}[c_i = e]$$

This is the empirical load: what fraction of tokens actually went to expert $e$.
If perfectly balanced, $f_e = 1/E$ for all $e$.

**Step 2 — Mean router probability per expert:**

$$P_e = \frac{1}{N} \sum_{i=1}^{N} p_{i,e}$$

This is the average softmax probability the router assigns to expert $e$ across all tokens.

**Step 3 — The auxiliary loss:**

$$\mathcal{L}_{\text{balance}} = E \cdot \sum_{e=1}^{E} f_e \cdot P_e$$

The multiplication $f_e \cdot P_e$ penalizes experts that **both** receive many tokens (high $f_e$)
**and** have high average probability (high $P_e$). The factor $E$ normalizes so that the
loss equals 1.0 under perfect balance.

**Why this works:** The gradient of $\mathcal{L}_{\text{balance}}$ w.r.t. the router weights
pushes the router to increase probability for underutilized experts and decrease it for
overloaded ones. Crucially, $f_e$ involves a non-differentiable argmax, but $P_e$ is
fully differentiable — so the gradient flows through $P_e$.

## Implementation

```python
def load_balance_loss(probs, expert_idx, n_experts):
    """
    probs: (B, T, E) softmax from router
    expert_idx: (B, T) hard choices
    """
    one_hot = F.one_hot(expert_idx, n_experts).float()
    f = one_hot.mean(dim=(0, 1))  # (E,) fraction of tokens per expert
    P = probs.mean(dim=(0, 1))    # (E,) mean probability per expert
    return (n_experts * (f * P).sum())

B, T, E = 4, 64, 4
probs = torch.softmax(torch.randn(B, T, E), dim=-1)
expert_idx = probs.argmax(dim=-1)
lb = load_balance_loss(probs, expert_idx, E)
print("load balance loss:", lb.item())
```

## Visualizing Expert Utilization: Before and After Balance Loss

Let's simulate a biased router (most tokens go to Expert 0) and show how the balance
loss would correct it over optimization steps.

```python
torch.manual_seed(0)

biased_logits = torch.randn(B, T, E)
biased_logits[:, :, 0] += 3.0
biased_probs = torch.softmax(biased_logits, dim=-1)
biased_idx = biased_probs.argmax(dim=-1)
biased_counts = torch.bincount(biased_idx.view(-1), minlength=E).float()
biased_util = (biased_counts / biased_counts.sum()).numpy()

uniform_logits = torch.randn(B, T, E) * 0.1
uniform_probs = torch.softmax(uniform_logits, dim=-1)
uniform_idx = uniform_probs.argmax(dim=-1)
uniform_counts = torch.bincount(uniform_idx.view(-1), minlength=E).float()
uniform_util = (uniform_counts / uniform_counts.sum()).numpy()

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4), sharey=True)
expert_labels = [f"Expert {i}" for i in range(E)]
colors = ["#4C72B0", "#55A868", "#C44E52", "#8172B3"]

ax1.bar(expert_labels, biased_util, color=colors)
ax1.set_title("Before Balance Loss\n(Expert Collapse)")
ax1.set_ylabel("Fraction of Tokens")
ax1.axhline(y=1/E, color="black", linestyle="--", alpha=0.5, label="Ideal (1/E)")
ax1.legend()

ax2.bar(expert_labels, uniform_util, color=colors)
ax2.set_title("After Balance Loss\n(Roughly Uniform)")
ax2.axhline(y=1/E, color="black", linestyle="--", alpha=0.5, label="Ideal (1/E)")
ax2.legend()

plt.suptitle("Expert Utilization Before vs After Load Balancing", fontsize=13, y=1.02)
plt.tight_layout()
plt.show()

print(f"Balance loss (biased):  {load_balance_loss(biased_probs, biased_idx, E).item():.4f}")
print(f"Balance loss (uniform): {load_balance_loss(uniform_probs, uniform_idx, E).item():.4f}")
```

## Combined Training Objective

```python
def total_loss(lm_loss, probs, expert_idx, n_experts, alpha=0.01):
    return lm_loss + alpha * load_balance_loss(probs, expert_idx, n_experts)

lm = torch.tensor(2.3)
loss = total_loss(lm, probs, expert_idx, E)
print("total:", loss.item())
```

## Tuning Alpha: The Balance Coefficient

The coefficient $\alpha$ controls how strongly the balance loss influences training.
Getting it right is critical:

| Alpha | Behavior |
|-------|----------|
| 0.0   | No balancing — pure LM loss. High risk of expert collapse. |
| 0.01  | Standard setting (Switch Transformer default). Gentle push toward balance. |
| 0.1   | Strong balancing. May sacrifice some LM quality for better utilization. |
| 1.0   | Extreme — balance loss dominates. Router becomes near-uniform random, defeating the purpose of having experts specialize. |

```python
torch.manual_seed(42)
test_probs = torch.softmax(torch.randn(B, T, E), dim=-1)
test_idx = test_probs.argmax(dim=-1)
lm_loss = torch.tensor(2.5)

alphas = [0.0, 0.001, 0.01, 0.1, 1.0]
print(f"{'alpha':>8} | {'LM loss':>10} | {'Balance':>10} | {'Total':>10}")
print("-" * 48)
for alpha in alphas:
    bl = load_balance_loss(test_probs, test_idx, E)
    total = lm_loss + alpha * bl
    print(f"{alpha:>8.3f} | {lm_loss.item():>10.4f} | {alpha * bl.item():>10.4f} | {total.item():>10.4f}")
```

## Training Loop: Watching Balance Improve Over Steps

Let's create a simple MoE layer with a trainable router and observe how the load balance
loss drives expert utilization toward uniformity during training.

```python
class SimpleMoE(nn.Module):
    def __init__(self, d_model, n_experts):
        super().__init__()
        self.router = nn.Linear(d_model, n_experts)
        self.n_experts = n_experts

    def forward(self, x):
        logits = self.router(x)
        probs = torch.softmax(logits, dim=-1)
        expert_idx = probs.argmax(dim=-1)
        return probs, expert_idx

d_model, n_exp = 64, 4
model = SimpleMoE(d_model, n_exp)
optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

history = {"step": [], "balance_loss": [], "utilization": []}

for step in range(200):
    x = torch.randn(4, 32, d_model)
    probs, expert_idx = model(x)
    loss = load_balance_loss(probs, expert_idx, n_exp)

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    if step % 20 == 0:
        counts = torch.bincount(expert_idx.view(-1), minlength=n_exp).float()
        util = (counts / counts.sum()).tolist()
        history["step"].append(step)
        history["balance_loss"].append(loss.item())
        history["utilization"].append(util)
        if step % 40 == 0:
            print(f"Step {step:3d} | Balance loss: {loss.item():.4f} | Util: {[f'{u:.2f}' for u in util]}")

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
ax1.plot(history["step"], history["balance_loss"], "b-o", markersize=4)
ax1.set_xlabel("Step")
ax1.set_ylabel("Balance Loss")
ax1.set_title("Load Balance Loss Over Training")

utils_arr = torch.tensor(history["utilization"])
for e in range(n_exp):
    ax2.plot(history["step"], utils_arr[:, e], "-o", markersize=4, label=f"Expert {e}")
ax2.axhline(y=1/n_exp, color="black", linestyle="--", alpha=0.5, label="Ideal")
ax2.set_xlabel("Step")
ax2.set_ylabel("Token Fraction")
ax2.set_title("Expert Utilization Over Training")
ax2.legend()

plt.tight_layout()
plt.show()
```

## Monitor Imbalance

```python
def expert_histogram(expert_idx, n_experts):
    c = torch.bincount(expert_idx.view(-1), minlength=n_experts).float()
    return (c / c.sum()).tolist()

print("utilization:", expert_histogram(expert_idx, E))
```

## Exercise: Implement Capacity Factor Token Dropping

Implement a routing function that enforces a capacity limit per expert. When an expert's
buffer is full, additional tokens assigned to it are **dropped** (passed through as identity).
Track how many tokens are dropped at different capacity factors.

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

def route_with_capacity_drop(x, experts, router_linear, n_experts, capacity_factor=1.0):
    """
    Route tokens to experts with a capacity limit.
    Tokens exceeding capacity pass through unchanged (identity).
    Returns: output, probs, n_dropped
    """
    B, T, C = x.shape
    logits = router_linear(x)
    probs = torch.softmax(logits, dim=-1)
    expert_idx = probs.argmax(dim=-1)

    capacity = max(1, int(capacity_factor * (B * T) / n_experts))
    out = x.clone()
    total_dropped = 0

    for e in range(n_experts):
        mask = (expert_idx == e)
        positions = mask.nonzero(as_tuple=False)
        n_assigned = len(positions)
        if n_assigned > capacity:
            total_dropped += n_assigned - capacity
            positions = positions[:capacity]
        if len(positions) > 0:
            idx_b, idx_t = positions[:, 0], positions[:, 1]
            out[idx_b, idx_t] = experts[e](x[idx_b, idx_t])

    return out, probs, total_dropped

d_test = 128
n_test = 4
router_test = nn.Linear(d_test, n_test)
experts_test = nn.ModuleList([ExpertFFN(d_test) for _ in range(n_test)])
x_test = torch.randn(4, 64, d_test)

print(f"{'C_f':>6} | {'Dropped':>8} | {'Drop Rate':>10}")
print("-" * 32)
for cf in [0.5, 0.75, 1.0, 1.25, 1.5, 2.0]:
    _, _, dropped = route_with_capacity_drop(x_test, experts_test, router_test, n_test, cf)
    total_tokens = x_test.shape[0] * x_test.shape[1]
    print(f"{cf:>6.2f} | {dropped:>8d} | {dropped/total_tokens:>9.1%}")
```

---

## Where This Leads Next

With load balancing in place, your experts will finally each receive a fair share of tokens. In **Section 4.4** you will study the *payoff*: tracking how those balanced experts naturally **specialize** — learning to handle different kinds of text (for example prose vs. structured data).

## Key Takeaway

Expert collapse — where the router funnels all tokens to one expert — is the central failure
mode of MoE training. The Switch Transformer **load balancing loss** $\mathcal{L} = E \sum_e f_e P_e$
provides a differentiable incentive for uniform utilization. The coefficient $\alpha$ must be
tuned carefully: too low allows collapse, too high forces uniform-random routing that
prevents specialization. Monitoring expert utilization histograms during training is
essential for diagnosing MoE health.

## Further Reading (Optional)

**Optional — you do NOT need these to continue. They are for curious students who want the original sources.**

- Fedus, Zoph, & Shazeer (2021). *Switch Transformers* (load balancing loss). JMLR.
- Zoph et al. (2022). *ST-MoE: Designing Stable and Transferable Sparse Expert Models*. arXiv:2202.08906.
