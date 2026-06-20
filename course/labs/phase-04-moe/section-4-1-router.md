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

# Section 4.1: The Router — Building the Triage Desk with Softmax Probabilities

**Goal:** Map each token hidden state to expert probabilities and select top-1 expert.

## What You Need to Know First

Everything here builds on ideas you have already met earlier in the course — you do **not** need any new outside knowledge:

- **A linear layer** (`nn.Linear`) — just a learnable matrix multiply that turns a vector of numbers into another vector of numbers.
- **Softmax** — the function that turns a list of raw scores into probabilities that are all positive and add up to 1.
- **A token's hidden state** — the vector of numbers that represents one word/token as it flows through the model.
- **Gradients and backpropagation** — how the model nudges its weights to reduce the loss. That is the only "training signal" the router ever gets.

If those four ideas feel familiar, you are fully prepared for this section.

## The Hospital Triage Analogy

Imagine a busy hospital emergency room. Every patient who walks in first meets a **triage nurse** —
a single person whose job is not to treat anyone, but to quickly assess each patient and decide
**which specialist** should see them. A patient with chest pain goes to cardiology; a broken arm
goes to orthopedics; a skin rash goes to dermatology.

A **router** in a Mixture-of-Experts model (an "MoE" — a model that holds several small sub-networks called *experts* and uses only a few of them per token) works exactly the same way:

| Hospital | MoE Model |
|----------|-----------|
| Patient arrives | Token hidden state enters the MoE layer |
| Triage nurse examines symptoms | Router's linear layer projects the hidden state |
| Nurse assigns a specialist | Softmax picks the highest-probability expert |
| Specialist treats the patient | Selected expert FFN (feed-forward network — the standard two-linear-layer block) processes the token |

The router itself is tiny — just a single `nn.Linear(d_model, n_experts)` — but its decisions
determine the entire computation graph. Every other parameter in the layer is gated by this
small projection.

## Building the Router

```python
import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt

d_model = 512
n_experts = 4

class Router(nn.Module):
    def __init__(self, d_model, n_experts):
        super().__init__()
        self.gate = nn.Linear(d_model, n_experts)

    def forward(self, x, temperature=1.0):
        # x: (B, T, C)
        logits = self.gate(x)
        probs = F.softmax(logits / temperature, dim=-1)
        expert_idx = probs.argmax(dim=-1)
        return probs, expert_idx

router = Router(d_model, n_experts)
x = torch.randn(2, 16, d_model)
probs, idx = router(x)
print("probs shape:", probs.shape)
print("selected experts:", idx[0].tolist())
```

## Softmax Temperature — Controlling Routing Sharpness

The softmax temperature $\tau$ controls how **decisive** the router is. Given logits $z_1, \dots, z_E$:

$$p_i = \frac{\exp(z_i / \tau)}{\sum_j \exp(z_j / \tau)}$$

- **Low temperature** ($\tau \to 0$): probabilities collapse to a one-hot vector — the router is
  extremely confident. Good for sparse compute but poor for gradient flow.
- **High temperature** ($\tau \to \infty$): probabilities approach uniform $1/E$ — the router is
  indecisive. Every expert gets nearly equal probability.
- **$\tau = 1$**: standard softmax — the default starting point.

```python
torch.manual_seed(42)
sample_logits = torch.tensor([[2.0, 1.0, 0.5, -0.5]])

temperatures = [0.1, 0.5, 1.0, 2.0, 5.0]
fig, axes = plt.subplots(1, len(temperatures), figsize=(15, 3), sharey=True)
for ax, temp in zip(axes, temperatures):
    p = F.softmax(sample_logits / temp, dim=-1).squeeze().detach()
    ax.bar(range(n_experts), p.numpy(), color=["#4C72B0", "#55A868", "#C44E52", "#8172B3"])
    ax.set_title(f"τ = {temp}")
    ax.set_xlabel("Expert")
    ax.set_ylim(0, 1.05)
    if ax == axes[0]:
        ax.set_ylabel("Probability")
plt.suptitle("Effect of Temperature on Router Probabilities", fontsize=13, y=1.05)
plt.tight_layout()
plt.show()
```

## Visualizing Router Probabilities for Sample Tokens

Let's see how the untrained router distributes probability mass across experts for
several random token embeddings.

```python
torch.manual_seed(7)
sample_tokens = torch.randn(5, 1, d_model)
labels = [f"Token {i}" for i in range(5)]

fig, ax = plt.subplots(figsize=(8, 4))
bar_width = 0.18
expert_colors = ["#4C72B0", "#55A868", "#C44E52", "#8172B3"]

for i, tok in enumerate(sample_tokens):
    p, _ = router(tok.unsqueeze(0))
    p = p.squeeze().detach().numpy()
    for e in range(n_experts):
        ax.bar(i + e * bar_width, p[e], width=bar_width,
               color=expert_colors[e], label=f"Expert {e}" if i == 0 else "")

ax.set_xticks([i + 1.5 * bar_width for i in range(5)])
ax.set_xticklabels(labels)
ax.set_ylabel("Router Probability")
ax.set_title("Router Probability Distribution per Token (Untrained)")
ax.legend()
plt.tight_layout()
plt.show()
```

## Top-1 Routing (Sparse Compute)

In top-1 routing, each token is processed by **exactly one** expert. This is the simplest
and most compute-efficient strategy — per-token FLOPs equal a single expert FFN.

```python
def route_top1(x, experts, router):
    probs, expert_idx = router(x)
    B, T, C = x.shape
    out = torch.zeros_like(x)
    for e, expert in enumerate(experts):
        mask = (expert_idx == e)
        if mask.any():
            out[mask] = expert(x[mask])
    return out, probs
```

## Top-1 vs Top-2 Routing — Trade-offs

| Property | Top-1 | Top-2 |
|----------|-------|-------|
| Experts evaluated per token | 1 | 2 |
| FLOPs per token | 1× FFN | 2× FFN |
| Gradient flow | Only through selected expert | Through both experts |
| Capacity pressure | Lower | Higher (each expert sees ~2× tokens) |
| Output formula | $y = \text{Expert}_k(x)$ | $y = w_1 \cdot E_1(x) + w_2 \cdot E_2(x)$ |

Top-2 routing gives smoother gradients to the router because **two** expert outputs
contribute to the loss, and the weighting coefficients $w_1, w_2$ are differentiable
functions of the router probabilities. The downside: you need a **capacity factor** —
each expert's buffer must accommodate roughly twice as many tokens.

## Gradient Flow Through Routing

A subtle but critical point: the router **learns** which expert to select, and it learns
this purely from the downstream language modeling loss.

The gradient path is:

$$\frac{\partial \mathcal{L}}{\partial W_{\text{router}}} = \frac{\partial \mathcal{L}}{\partial y} \cdot \frac{\partial y}{\partial w_i} \cdot \frac{\partial w_i}{\partial W_{\text{router}}}$$

where $w_i$ are the softmax routing weights. The `argmax` operation itself is not
differentiable, but the softmax weights that multiply expert outputs **are**. In top-2
routing the output is $y = w_1 E_1(x) + w_2 E_2(x)$, so gradients flow through $w_1$ and
$w_2$ back to the router weights.

In top-1 routing, only the **selected expert** produces gradients — the router gets a
learning signal only from the combination of "which expert was picked" and "how well it did."

```python
router_for_grad = Router(d_model, n_experts)
x_test = torch.randn(1, 4, d_model, requires_grad=True)
probs_test, _ = router_for_grad(x_test)

loss_proxy = probs_test.sum()
loss_proxy.backward()
print("Router gate weight grad norm:", router_for_grad.gate.weight.grad.norm().item())
print("Input grad norm:", x_test.grad.norm().item())
print("Both are nonzero — gradients flow through softmax routing.")
```

## Load Preview

```python
counts = torch.bincount(idx.view(-1), minlength=n_experts)
print("tokens per expert:", counts.tolist())
```

## Exercise: Implement Top-2 Routing

Implement a `route_top2` function where each token is processed by its **two** highest-probability
experts, and the output is the weighted sum of both expert outputs (weighted by their
renormalized routing probabilities).

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

def route_top2(x, experts, router):
    """
    Top-2 gating: each token is processed by its top-2 experts.
    Output = w1 * Expert_1(x) + w2 * Expert_2(x), where w1 + w2 = 1.
    """
    probs, _ = router(x)
    B, T, C = x.shape
    n_experts = len(experts)

    top2_probs, top2_idx = probs.topk(2, dim=-1)  # (B, T, 2)
    top2_weights = top2_probs / top2_probs.sum(dim=-1, keepdim=True)  # renormalize

    out = torch.zeros_like(x)
    for k in range(2):
        for e in range(n_experts):
            mask = (top2_idx[:, :, k] == e)
            if mask.any():
                expert_out = experts[e](x[mask])
                out[mask] += top2_weights[:, :, k][mask].unsqueeze(-1) * expert_out
    return out, probs

experts = nn.ModuleList([ExpertFFN(d_model) for _ in range(n_experts)])
x_ex = torch.randn(2, 8, d_model)
y_top2, p_top2 = route_top2(x_ex, experts, router)
print("Top-2 output shape:", y_top2.shape)

top2_idx = p_top2.topk(2, dim=-1).indices
print("Top-2 expert selections (batch 0):", top2_idx[0].tolist())
```

---

## Where This Leads Next

Now that the router can decide *which* expert should handle each token, the next step is to actually build those experts. In **Section 4.2** you will create four parallel expert sub-networks and wire them to this router, getting the capacity of a much bigger model while paying the compute cost of a small one.

## Key Takeaway

The **router** is the decision-maker of MoE — a single linear layer followed by softmax that
assigns each token to one (or two) experts. Temperature controls sharpness of these assignments.
Top-1 routing is maximally sparse but has limited gradient flow; top-2 routing doubles compute
per token but gives the router richer learning signals. In either case, the router learns
**entirely from the downstream task loss** — it discovers which expert is best for which
pattern through backpropagation, not through any explicit labeling.

Next lab replaces per-expert Python loop with batched masking for GPU efficiency.

## Further Reading (Optional)

**Optional — you do NOT need these to continue. They are for curious students who want the original sources.**

- Shazeer et al. (2017). *Outrageously Large Neural Networks: The Sparsely-Gated Mixture-of-Experts Layer*. ICLR.
- Fedus, Zoph, & Shazeer (2021). *Switch Transformers*. JMLR.
