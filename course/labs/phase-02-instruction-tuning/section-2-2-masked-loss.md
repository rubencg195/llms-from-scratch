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

# Section 2.2: The Masked Loss Function — Multiplying "User" Errors by Zero

**Goal:** Build a boolean loss mask so only assistant tokens contribute to cross-entropy.

## What You Need to Know First

This section continues directly from Section 2.1 — you already have all the background you need.

- **Chat templates and role regions** — from Section 2.1, you know each token belongs to a "user" or "assistant" region. The "Train?" column from that section becomes our mask here.
- **Cross-entropy loss** — the standard number that measures how wrong a prediction is; bigger means more wrong. We computed it in Phase 1.
- **A mask** — just a list of 1s and 0s the same length as the tokens; multiplying by 0 "turns off" a position, multiplying by 1 keeps it.
- **Gradients** — the signal that tells each weight how to change; "zero gradient" simply means "no learning happens here."

No new math beyond multiplying and averaging is required.

## Why We Don't Train on User Tokens

In a chat-formatted training example, roughly half the tokens belong to the user's question. If we include these in the loss, two bad things happen:

1. **Wasted capacity.** The model spends parameters learning to predict what humans *ask* — something it will never need to do at inference time. Every gradient step on user tokens is a gradient step *not* spent improving response quality.

2. **Objective contamination.** During inference, the model generates token-by-token in the assistant region. If it was also trained to predict user tokens (via teacher forcing), it may "drift" into generating question-like text — because that's what maximized likelihood on half the training data.

The solution is simple: compute cross-entropy on every position, then **zero out** the loss for any position that falls in a user (or padding) region. Only assistant-region errors propagate gradients backward.

## Visualizing the Mask

Consider a tokenized sequence with 8 positions. The table below shows how role labels and mask values align:

| Position | 0 | 1 | 2 | 3 | 4 | 5 | 6 | 7 |
|----------|---|---|---|---|---|---|---|---|
| **Token** | `<\|user\|>` | `What` | `is` | `<\|asst\|>` | `The` | `answer` | `is` | `<\|pad\|>` |
| **Role** | user | user | user | assistant | assistant | assistant | assistant | pad |
| **Mask** | 0 | 0 | 0 | 1 | 1 | 1 | 1 | 0 |
| **Loss** | ✗ | ✗ | ✗ | ✓ | ✓ | ✓ | ✓ | ✗ |

The mask is a `float32` tensor of the same shape as the target — multiply element-wise with the per-token loss, then divide by the number of unmasked positions.

## Core Implementation: Manual Mask Multiplication

```python
import torch
import torch.nn.functional as F

seq = torch.tensor([1, 1, 1, 2, 2, 2, 2, 0])
loss_mask = torch.tensor([0, 0, 0, 1, 1, 1, 1, 0], dtype=torch.float32)

logits = torch.randn(8, 100)  # (T, vocab)
targets = torch.randint(0, 100, (8,))

def masked_cross_entropy(logits, targets, mask, ignore_index=-100):
    T, V = logits.shape
    loss = F.cross_entropy(logits, targets, reduction="none")
    loss = loss * mask
    denom = mask.sum().clamp(min=1)
    return loss.sum() / denom

print("masked loss:", masked_cross_entropy(logits, targets, loss_mask).item())
```

## The `ignore_index=-100` Alternative

PyTorch's `F.cross_entropy` has a built-in mechanism for ignoring certain target positions: the `ignore_index` parameter. When a target equals `ignore_index` (default -100), that position contributes **zero** to both the loss and the normalization denominator.

This is semantically equivalent to our manual masking but avoids the explicit mask tensor:

```python
def masked_ce_via_ignore_index(logits, targets, mask):
    """Replace masked positions with ignore_index=-100 in targets."""
    masked_targets = targets.clone()
    masked_targets[mask == 0] = -100
    return F.cross_entropy(logits, masked_targets, ignore_index=-100)

loss_manual = masked_cross_entropy(logits, targets, loss_mask)
loss_ignore = masked_ce_via_ignore_index(logits, targets, loss_mask)
print(f"Manual mask loss:       {loss_manual.item():.6f}")
print(f"ignore_index loss:      {loss_ignore.item():.6f}")
print(f"Difference:             {abs(loss_manual.item() - loss_ignore.item()):.8f}")
```

The two approaches should give numerically identical results (up to floating-point precision). The `ignore_index` approach is slightly faster because PyTorch skips the softmax computation for ignored positions entirely.

## Build Mask from Chat Template Spans

```python
def build_assistant_mask(token_roles):
    """token_roles: list of 'user' | 'assistant' | 'pad' per token"""
    mask = torch.tensor([1.0 if r == "assistant" else 0.0 for r in token_roles])
    return mask

roles = ["user", "user", "assistant", "assistant", "assistant", "pad"]
m = build_assistant_mask(roles)
print("mask:", m)
```

## Batched Training Step

```python
def batch_masked_ce(logits, targets, masks):
    """logits: (B,T,V), targets: (B,T), masks: (B,T)"""
    B, T, V = logits.shape
    loss = F.cross_entropy(logits.view(B * T, V), targets.view(B * T), reduction="none")
    loss = loss.view(B, T) * masks
    return loss.sum() / masks.sum().clamp(min=1)

B, T, V = 4, 16, 8000
logits = torch.randn(B, T, V)
targets = torch.randint(0, V, (B, T))
masks = (torch.rand(B, T) > 0.5).float()
print("batch loss:", batch_masked_ce(logits, targets, masks).item())
```

## Why Masking Matters — Gradient Analysis

Without masking, the model learns to **predict user questions** — wasting capacity and hurting inference (teacher forcing on wrong objective). We can prove masking works by examining gradient magnitudes at each position.

```python
demo_logits = torch.randn(8, 100, requires_grad=True)
demo_targets = torch.randint(0, 100, (8,))
demo_mask = torch.tensor([0, 0, 0, 1, 1, 1, 1, 0], dtype=torch.float32)

demo_loss = masked_cross_entropy(demo_logits, demo_targets, demo_mask)
demo_loss.backward()

per_token_grad = demo_logits.grad.norm(dim=-1)
print("grad magnitude per position:", per_token_grad.tolist())
print("user positions (0-2) grads ≈ 0:", per_token_grad[:3].sum().item() < 1e-6)
print("assistant positions (3-6) grads > 0:", per_token_grad[3:7].sum().item() > 0)
```

## Visualization: Loss Contribution Per Position

This bar chart shows the per-position loss values before and after applying the mask. Masked positions contribute exactly zero.

```python
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

viz_logits = torch.randn(8, 100)
viz_targets = torch.randint(0, 100, (8,))
viz_mask = torch.tensor([0, 0, 0, 1, 1, 1, 1, 0], dtype=torch.float32)

raw_loss = F.cross_entropy(viz_logits, viz_targets, reduction="none")
masked_loss_vals = raw_loss * viz_mask

fig, axes = plt.subplots(1, 2, figsize=(12, 4))

positions = list(range(8))
labels = ["usr", "usr", "usr", "asst", "asst", "asst", "asst", "pad"]
colors = ["#ff6b6b" if l == "usr" else "#51cf66" if l == "asst" else "#868e96" for l in labels]

axes[0].bar(positions, raw_loss.detach().numpy(), color=colors)
axes[0].set_title("Raw Loss (all positions)")
axes[0].set_xlabel("Position")
axes[0].set_ylabel("Cross-Entropy Loss")
axes[0].set_xticks(positions)
axes[0].set_xticklabels(labels)

axes[1].bar(positions, masked_loss_vals.detach().numpy(), color=colors)
axes[1].set_title("Masked Loss (only assistant)")
axes[1].set_xlabel("Position")
axes[1].set_ylabel("Cross-Entropy Loss")
axes[1].set_xticks(positions)
axes[1].set_xticklabels(labels)

plt.tight_layout()
plt.savefig("masked_loss_visualization.png", dpi=100, bbox_inches="tight")
plt.close()
print("Saved masked_loss_visualization.png")
print(f"Total raw loss:    {raw_loss.sum().item():.4f}")
print(f"Total masked loss: {masked_loss_vals.sum().item():.4f}")
print(f"Positions contributing: {int(viz_mask.sum().item())} / {len(viz_mask)}")
```

## Exercise: Implement Masking Using `ignore_index` and Compare

Implement a complete training step using the `ignore_index=-100` approach instead of explicit mask multiplication. Then compare the gradients from both methods to verify they are equivalent.

```python
def training_step_manual_mask(logits, targets, mask):
    """Standard approach: multiply loss by mask."""
    loss_per_token = F.cross_entropy(logits, targets, reduction="none")
    masked = loss_per_token * mask
    return masked.sum() / mask.sum().clamp(min=1)

def training_step_ignore_index(logits, targets, mask):
    """Alternative: use ignore_index=-100 for masked positions."""
    modified_targets = targets.clone()
    modified_targets[mask == 0] = -100
    return F.cross_entropy(logits, modified_targets, ignore_index=-100)

torch.manual_seed(42)
ex_logits = torch.randn(8, 100, requires_grad=True)
ex_targets = torch.randint(0, 100, (8,))
ex_mask = torch.tensor([0, 0, 0, 1, 1, 1, 1, 0], dtype=torch.float32)

loss_a = training_step_manual_mask(ex_logits, ex_targets, ex_mask)
loss_a.backward()
grad_a = ex_logits.grad.clone()

ex_logits.grad = None

loss_b = training_step_ignore_index(ex_logits, ex_targets, ex_mask)
loss_b.backward()
grad_b = ex_logits.grad.clone()

print(f"Loss (manual mask):    {loss_a.item():.6f}")
print(f"Loss (ignore_index):   {loss_b.item():.6f}")
print(f"Max gradient diff:     {(grad_a - grad_b).abs().max().item():.2e}")
print(f"Gradients equivalent:  {torch.allclose(grad_a, grad_b, atol=1e-6)}")
```

## Where This Leads Next

You now have the two core building blocks of instruction tuning: a chat template (Section 2.1) and a masked loss (this section). Section 2.3 puts them together into a full fine-tuning loop on the real GSM8K math dataset, where the mask is built automatically from the template spans.

---

## Key Takeaway

The masked loss function is the mechanism that teaches the model *only how to respond*, not how to ask questions. Two equivalent implementations exist:

1. **Explicit mask multiplication** — flexible, lets you apply fractional weights or curriculum-based scaling per position
2. **`ignore_index=-100`** — simpler API, slightly faster, but binary (on/off only)

Both produce identical gradients. The key insight is that masking ensures **zero gradient flows to user-region logits**, meaning the model's parameters are updated exclusively to improve response quality. This is not a minor optimization — without masking, you waste ~50% of your training compute learning the wrong objective.

## Further Reading (Optional)

**Optional — you do NOT need these to continue. They are for curious students who want the original sources.**

- Ouyang et al. (2022). *Training language models to follow instructions with human feedback*. NeurIPS.
- Wang et al. (2023). *Self-Instruct: Aligning Language Models with Self-Generated Instructions*. ACL.
