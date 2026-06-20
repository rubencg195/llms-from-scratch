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

# Section 0.3: PyTorch Autograd — The Automated "Blame Game" for Errors

**Goal:** Understand `requires_grad`, forward/backward passes, and one optimizer step on a toy linear model.

## What You Need to Know First

- **Tensors and basic tensor math** (Section 0.1) — especially creating tensors and doing arithmetic on them.
- **The dot product / matrix multiply** (Section 0.2) — the `x @ w` step that turns inputs into a prediction.
- **The high-school idea of a slope** — a *gradient* is just "how steeply the error changes if I nudge a number," i.e. a slope.

Everything here builds directly on Sections 0.1 and 0.2, so no new outside knowledge is needed. Jargon like *gradient*, *backpropagation*, and *optimizer* is explained in plain words as it comes up. (A **gradient** is a slope that says which direction to push each weight to make the error smaller; **backpropagation** is just applying that slope rule backwards through the calculation.)

## The training loop skeleton

1. Forward → prediction  
2. Loss → scalar error  
3. `backward()` → gradients  
4. `optimizer.step()` → update weights  

## The Computation Graph

PyTorch silently builds a **directed acyclic graph (DAG)** as you compute. Each node is an operation; edges carry tensors. When you call `loss.backward()`, PyTorch walks this graph **backwards** (hence "backpropagation"), applying the chain rule at each node to compute $\frac{\partial \text{loss}}{\partial \text{parameter}}$ for every learnable weight.

```
                    ┌──────────┐
       x ──────────│  x @ w   │──── prediction
                    └────┬─────┘           │
       w (learnable) ────┘                 │
                                           ▼
                                    ┌─────────────┐
       y_true ─────────────────────│  MSE Loss    │──── loss (scalar)
                                    └──────┬──────┘
                                           │
                        backward()         │
                        ◄──────────────────┘
                                    │
                         ┌──────────┴──────────┐
                         │  ∂loss/∂w = ...     │
                         │  ∂loss/∂b = ...     │
                         └─────────────────────┘
```

Key insight: you never write gradient formulas by hand. Autograd handles arbitrary compositions — attention layers, LayerNorm, MoE routers — all automatically.

```python
import torch
import matplotlib.pyplot as plt

# True line: y = 2x + 1
torch.manual_seed(42)
X = torch.linspace(0, 1, 100).unsqueeze(1)
y_true = 2 * X + 1 + 0.05 * torch.randn(100, 1)

# Learnable parameters — start wrong on purpose
w = torch.tensor([[0.0]], requires_grad=True)
b = torch.tensor([[0.0]], requires_grad=True)

def predict(x):
    return x @ w + b

loss_fn = torch.nn.MSELoss()
```

## One training step

```python
y_pred = predict(X)
loss = loss_fn(y_pred, y_true)
print("loss before:", loss.item())

loss.backward()
print("gradient w:", w.grad.item())
print("gradient b:", b.grad.item())

with torch.no_grad():
    w -= 0.1 * w.grad
    b -= 0.1 * b.grad
    w.grad.zero_()
    b.grad.zero_()

print("w, b after one step:", w.item(), b.item())
```

## Why `zero_grad()` Is Essential: Gradient Accumulation

PyTorch **accumulates** gradients by default — calling `backward()` **adds** to existing `.grad` values rather than replacing them. This is useful for gradient accumulation (simulating larger batch sizes), but dangerous if you forget to zero them out.

```python
# Demonstrate gradient accumulation bug
w_demo = torch.tensor([[1.0]], requires_grad=True)
for i in range(3):
    loss = (w_demo * 2).sum()
    loss.backward()
    print(f"Step {i}: w_demo.grad = {w_demo.grad.item()} (accumulating!)")

# Fix: zero before each backward
w_demo.grad.zero_()
print(f"After zero_grad: w_demo.grad = {w_demo.grad.item()}")
```

## Full loop with Adam

```python
w = torch.nn.Parameter(torch.tensor([[0.0]]))
b = torch.nn.Parameter(torch.tensor([[0.0]]))
opt = torch.optim.Adam([w, b], lr=0.05)

losses = []
for step in range(200):
    opt.zero_grad()
    y_pred = X @ w + b
    loss = loss_fn(y_pred, y_true)
    loss.backward()
    opt.step()
    losses.append(loss.item())
    if step % 50 == 0:
        print(f"step {step}: loss={loss.item():.4f}, w={w.item():.3f}, b={b.item():.3f}")
```

## Visualization: Loss Curve Over Training

The loss curve tells you whether training is working. A healthy curve drops steeply at first (easy gains), then plateaus (diminishing returns).

```python
plt.figure(figsize=(8, 4))
plt.plot(losses, linewidth=2, color="steelblue")
plt.xlabel("Training Step")
plt.ylabel("MSE Loss")
plt.title("Loss Curve: Learning y = 2x + 1")
plt.yscale("log")
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig("loss_curve_autograd.png", dpi=120)
plt.show()
print("Saved loss_curve_autograd.png")
```

## Learning Rate: Too High vs Too Low

The learning rate (lr) is the most important hyperparameter. It controls how big each step is:
- **Too low:** Training converges, but painfully slowly (thousands of wasted steps)
- **Too high:** Parameters overshoot the optimum, loss oscillates or diverges
- **Just right:** Fast convergence to a good solution

```python
fig, axes = plt.subplots(1, 3, figsize=(14, 4))
learning_rates = [0.001, 0.05, 1.5]
titles = ["Too Low (lr=0.001)", "Just Right (lr=0.05)", "Too High (lr=1.5)"]

for ax, lr, title in zip(axes, learning_rates, titles):
    w_exp = torch.nn.Parameter(torch.tensor([[0.0]]))
    b_exp = torch.nn.Parameter(torch.tensor([[0.0]]))
    opt_exp = torch.optim.SGD([w_exp, b_exp], lr=lr)
    exp_losses = []

    for step in range(200):
        opt_exp.zero_grad()
        y_pred = X @ w_exp + b_exp
        loss = loss_fn(y_pred, y_true)
        loss.backward()
        opt_exp.step()
        exp_losses.append(min(loss.item(), 100))  # clip for visualization

    ax.plot(exp_losses, linewidth=2)
    ax.set_title(title, fontsize=12)
    ax.set_xlabel("Step")
    ax.set_ylabel("Loss")
    ax.set_ylim(0, max(exp_losses[0], 5))
    ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig("learning_rate_comparison.png", dpi=120)
plt.show()
print("Saved learning_rate_comparison.png")
```

## Why autograd scales

PyTorch builds a **computation graph**. One `loss.backward()` propagates through millions of ops—attention, LayerNorm, MoE routers—automatically. The cost of backward is roughly 2× the cost of forward.

## Exercise 1: Train on GPU

Train the same toy line on **GPU** if available. Log final `w` and `b`; they should be near `2.0` and `1.0`.

```python
device = "cuda" if torch.cuda.is_available() else "cpu"
X_d = X.to(device)
y_d = y_true.to(device)
w = torch.nn.Parameter(torch.zeros(1, 1, device=device))
b = torch.nn.Parameter(torch.zeros(1, 1, device=device))
opt = torch.optim.Adam([w, b], lr=0.05)

for _ in range(300):
    opt.zero_grad()
    loss = loss_fn(X_d @ w + b, y_d)
    loss.backward()
    opt.step()

print("learned w, b:", w.item(), b.item())
```

## Exercise 2: Fit a Quadratic

Extend the linear model to learn $y = ax^2 + bx + c$. The true function is $y = 0.5x^2 - 1.5x + 2$.

```python
torch.manual_seed(7)
X_quad = torch.linspace(-3, 3, 200).unsqueeze(1).to(device)
y_quad = (0.5 * X_quad**2 - 1.5 * X_quad + 2 + 0.1 * torch.randn_like(X_quad)).to(device)

# Learnable parameters for y = a*x^2 + b*x + c
a = torch.nn.Parameter(torch.tensor([[0.0]], device=device))
b_param = torch.nn.Parameter(torch.tensor([[0.0]], device=device))
c = torch.nn.Parameter(torch.tensor([[0.0]], device=device))
opt = torch.optim.Adam([a, b_param, c], lr=0.05)

quad_losses = []
for step in range(500):
    opt.zero_grad()
    y_pred = a * X_quad**2 + b_param * X_quad + c
    loss = loss_fn(y_pred, y_quad)
    loss.backward()
    opt.step()
    quad_losses.append(loss.item())
    if step % 100 == 0:
        print(f"step {step}: loss={loss.item():.4f}, a={a.item():.3f}, b={b_param.item():.3f}, c={c.item():.3f}")

print(f"\nLearned: y = {a.item():.3f}x² + {b_param.item():.3f}x + {c.item():.3f}")
print(f"True:    y = 0.500x² + -1.500x + 2.000")
```

## Where This Leads Next

You can now train a model with two parameters. Section 0.4 keeps the exact same loop (forward → loss → backward → step) but swaps the toy `w` and `b` for a real **`nn.Linear` layer** stacked with a non-linearity — the actual building block that every Transformer is made of.

## Key Takeaway

- **Autograd** builds a computation graph during the forward pass and uses the chain rule during `backward()` to compute all gradients automatically.
- Always call `zero_grad()` before `backward()` — PyTorch accumulates gradients by default.
- The **learning rate** is the most critical hyperparameter: too low wastes compute, too high causes divergence.
- The **loss curve** is your primary diagnostic tool — if it's not going down, something is wrong.
- These same mechanics (forward → loss → backward → step) scale unchanged from 2-parameter toy models to 70-billion-parameter LLMs.

## Checkpoint

You understand autograd, gradient descent, and learning rate dynamics. Next: **building your first neural layer** (Section 0.4) — stacking linear operations with non-linearities.

## Further Reading (Optional)

**Optional — you do NOT need these to continue. They are for curious students who want the original sources.**

- Rumelhart, Hinton, & Williams (1986). *Learning representations by back-propagating errors*. Nature.
- Baydin et al. (2018). *Automatic Differentiation in Machine Learning: a Survey*. JMLR.
- Kingma & Ba (2015). *Adam: A Method for Stochastic Optimization*. ICLR.
