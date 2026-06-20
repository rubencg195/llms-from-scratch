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

# Section 0.4: Building Your First $y = mx + b$ Neural Layer

**Goal:** Implement `nn.Linear`, stack two layers with a non-linearity, and train on synthetic classification data.

## One linear layer

`nn.Linear(in_features, out_features)` stores $W$ shape `(out, in)` and $b$ shape `(out,)`.

Forward: $y = x W^T + b$

```python
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import numpy as np

layer = nn.Linear(3, 2)
x = torch.randn(5, 3)  # batch of 5 vectors
y = layer(x)
print("input:", x.shape, "output:", y.shape)
print("weight shape:", layer.weight.shape)
```

## Why Non-Linearities Matter: The XOR Problem

A single linear layer can only learn **linear** decision boundaries (straight lines/planes). Many real-world problems are not linearly separable. The classic example is XOR:

```python
# XOR: no single line can separate the classes
xor_X = torch.tensor([[0., 0.], [0., 1.], [1., 0.], [1., 1.]])
xor_y = torch.tensor([[0.], [1.], [1.], [0.]])  # XOR labels

# A linear model CANNOT solve this
linear_model = nn.Linear(2, 1)
opt_lin = torch.optim.Adam(linear_model.parameters(), lr=0.1)

for _ in range(1000):
    opt_lin.zero_grad()
    pred = linear_model(xor_X)
    loss = nn.BCEWithLogitsLoss()(pred, xor_y)
    loss.backward()
    opt_lin.step()

preds_linear = torch.sigmoid(linear_model(xor_X))
print("Linear model on XOR (cannot solve it):")
for i in range(4):
    print(f"  input={xor_X[i].tolist()} → pred={preds_linear[i].item():.3f}, true={xor_y[i].item():.0f}")
```

```python
# A 2-layer network WITH non-linearity CAN solve XOR
class XORNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(2, 8)
        self.fc2 = nn.Linear(8, 1)

    def forward(self, x):
        x = torch.relu(self.fc1(x))  # non-linearity is the key!
        return self.fc2(x)

torch.manual_seed(42)
xor_net = XORNet()
opt_xor = torch.optim.Adam(xor_net.parameters(), lr=0.05)

for _ in range(2000):
    opt_xor.zero_grad()
    loss = nn.BCEWithLogitsLoss()(xor_net(xor_X), xor_y)
    loss.backward()
    opt_xor.step()

preds_xor = torch.sigmoid(xor_net(xor_X))
print("\nNon-linear model on XOR (solves it!):")
for i in range(4):
    print(f"  input={xor_X[i].tolist()} → pred={preds_xor[i].item():.3f}, true={xor_y[i].item():.0f}")
```

## Two layers = tiny network

```python
class TinyNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(2, 16)
        self.fc2 = nn.Linear(16, 1)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        return self.fc2(x)

model = TinyNet()
print(model)
```

## Synthetic dataset

Random 2D points; label 1 if $x_0 + x_1 > 0$.

```python
torch.manual_seed(0)
N = 512
X = torch.randn(N, 2)
y = (X[:, 0] + X[:, 1] > 0).float().unsqueeze(1)

device = "cuda" if torch.cuda.is_available() else "cpu"
model = TinyNet().to(device)
X, y = X.to(device), y.to(device)
opt = torch.optim.Adam(model.parameters(), lr=1e-2)
bce = nn.BCEWithLogitsLoss()
```

## Training with Loss Curve

```python
train_losses = []
for epoch in range(200):
    opt.zero_grad()
    logits = model(X)
    loss = bce(logits, y)
    loss.backward()
    opt.step()
    train_losses.append(loss.item())
    if epoch % 50 == 0:
        acc = ((logits > 0).float() == y).float().mean()
        print(f"epoch {epoch}: loss={loss.item():.3f} acc={acc.item():.3f}")

# Final accuracy
acc = ((model(X) > 0).float() == y).float().mean()
print(f"\nFinal accuracy: {acc.item():.3f}")
```

```python
plt.figure(figsize=(8, 4))
plt.plot(train_losses, linewidth=2, color="steelblue")
plt.xlabel("Epoch")
plt.ylabel("BCE Loss")
plt.title("Training Loss Curve — TinyNet on Classification")
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig("first_layer_loss.png", dpi=120)
plt.show()
print("Saved first_layer_loss.png")
```

## Visualization: Decision Boundary

Let's visualize what the network learned — the decision boundary separating class 0 from class 1.

```python
# Create a grid of points to classify
xx, yy = np.meshgrid(np.linspace(-3, 3, 200), np.linspace(-3, 3, 200))
grid = torch.tensor(np.c_[xx.ravel(), yy.ravel()], dtype=torch.float32).to(device)

with torch.no_grad():
    grid_preds = torch.sigmoid(model(grid)).cpu().numpy().reshape(xx.shape)

X_cpu = X.cpu().numpy()
y_cpu = y.cpu().numpy().ravel()

plt.figure(figsize=(7, 6))
plt.contourf(xx, yy, grid_preds, levels=50, cmap="RdBu", alpha=0.8)
plt.colorbar(label="P(class=1)")
plt.scatter(X_cpu[y_cpu == 0, 0], X_cpu[y_cpu == 0, 1], c="red", s=10, alpha=0.5, label="class 0")
plt.scatter(X_cpu[y_cpu == 1, 0], X_cpu[y_cpu == 1, 1], c="blue", s=10, alpha=0.5, label="class 1")
plt.contour(xx, yy, grid_preds, levels=[0.5], colors="black", linewidths=2)
plt.xlabel("$x_0$")
plt.ylabel("$x_1$")
plt.title("Decision Boundary of TinyNet (black line = 50% threshold)")
plt.legend()
plt.tight_layout()
plt.savefig("decision_boundary.png", dpi=120)
plt.show()
print("Saved decision_boundary.png")
```

## Model Introspection: What Did the Network Learn?

Neural networks are often called "black boxes," but we can examine what they learned by looking at their weights.

```python
print("=== Layer 1 (fc1): 2 inputs → 16 hidden neurons ===")
print(f"Weight shape: {model.fc1.weight.shape}")
print(f"Bias shape: {model.fc1.bias.shape}")
print(f"\nFirst 4 neuron weight vectors:")
for i in range(4):
    w = model.fc1.weight[i].detach().cpu().numpy()
    b = model.fc1.bias[i].detach().cpu().item()
    print(f"  neuron {i}: w=[{w[0]:.3f}, {w[1]:.3f}], b={b:.3f}")

print(f"\n=== Layer 2 (fc2): 16 hidden → 1 output ===")
print(f"Weight shape: {model.fc2.weight.shape}")
print(f"These weights determine how much each hidden neuron contributes to the final decision.")

# Total parameters
total = sum(p.numel() for p in model.parameters())
print(f"\nTotal parameters: {total}")
```

## Overfitting vs Generalization

In the real world, we care about performance on **unseen data**, not just the training set. Let's demonstrate the train/test split concept.

```python
torch.manual_seed(99)
N_total = 1024
X_all = torch.randn(N_total, 2, device=device)
y_all = (X_all[:, 0] + X_all[:, 1] > 0).float().unsqueeze(1)

# 80/20 split
n_train = int(0.8 * N_total)
X_train, X_test = X_all[:n_train], X_all[n_train:]
y_train, y_test = y_all[:n_train], y_all[n_train:]

model2 = TinyNet().to(device)
opt2 = torch.optim.Adam(model2.parameters(), lr=1e-2)

train_losses2, test_losses2 = [], []
for epoch in range(300):
    # Train
    opt2.zero_grad()
    loss_train = bce(model2(X_train), y_train)
    loss_train.backward()
    opt2.step()
    train_losses2.append(loss_train.item())

    # Evaluate (no gradients needed)
    with torch.no_grad():
        loss_test = bce(model2(X_test), y_test)
        test_losses2.append(loss_test.item())

train_acc = ((model2(X_train) > 0).float() == y_train).float().mean()
test_acc = ((model2(X_test) > 0).float() == y_test).float().mean()
print(f"Train accuracy: {train_acc.item():.3f}")
print(f"Test accuracy:  {test_acc.item():.3f}")
```

```python
plt.figure(figsize=(8, 4))
plt.plot(train_losses2, label="Train Loss", linewidth=2)
plt.plot(test_losses2, label="Test Loss", linewidth=2, linestyle="--")
plt.xlabel("Epoch")
plt.ylabel("BCE Loss")
plt.title("Train vs Test Loss — Monitoring for Overfitting")
plt.legend()
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig("overfit_monitor.png", dpi=120)
plt.show()
print("Saved overfit_monitor.png")
```

## Phase 0 wrap-up

You built the atomic unit of every LLM layer: **matrix multiply + bias + non-linearity**. Phase 1 stacks hundreds of these into a Transformer.

```python
if device == "cuda":
    print("Peak VRAM (MB):", torch.cuda.max_memory_allocated() / 1e6)
```

## Key Takeaway

- **Non-linearities** (ReLU, GELU, etc.) are what give neural networks their power — without them, any depth collapses to a single linear transformation.
- **Decision boundaries** show what the network learned geometrically — deeper networks can learn more complex boundaries.
- **Model introspection** (examining weights) reveals what each neuron "looks for" in the input.
- **Train/test splits** prevent self-delusion — always measure performance on data the model hasn't seen during training.
- The building block of a Transformer is exactly this: `Linear → NonLinearity → Linear`, repeated hundreds of times with attention in between.

## Phase 0 Complete → What's Next

You now have all the prerequisites for building a Transformer:
- **Tensors** (Section 0.1): the data structure
- **Dot products** (Section 0.2): the similarity operation
- **Autograd** (Section 0.3): the learning mechanism
- **Linear layers** (Section 0.4): the computation unit

In **Phase 1**, we assemble these into a GPT: tokenization → embeddings → position encoding (RoPE) → attention → training loop.
