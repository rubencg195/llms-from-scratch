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

# Section 0.25: Micrograd — Build Autograd From Scratch

**Goal:** Implement a tiny scalar autograd engine (like Karpathy's *micrograd*) so backpropagation is not a black box before you use PyTorch.

## What You Need to Know First

- **Tensors and basic arithmetic** (Section 0.1).
- **The dot product idea** (Section 0.2) — a neuron is multiply-and-add.
- **High-school chain rule** — if $y = f(g(x))$, then $\frac{dy}{dx} = \frac{dy}{dg}\cdot\frac{dg}{dx}$. You do not need calculus beyond that.

Do this section **before** Section 0.3 if you want to understand *why* `loss.backward()` works. Section 0.3 then shows the production version in PyTorch.

## Why build your own autograd?

PyTorch's `loss.backward()` feels like magic until you see the mechanism once:

1. Every operation records its inputs in a graph.
2. Each operation knows its **local derivative** (how much its output changes when its input nudges).
3. `backward()` walks the graph in reverse, multiplying local derivatives (the chain rule).

Karpathy's *micrograd* does exactly this in ~100 lines. We replicate the core idea here.

## The `Value` class

```python
class Value:
    """A scalar number that remembers how it was computed."""

    def __init__(self, data, _children=(), _op=""):
        self.data = float(data)
        self.grad = 0.0
        self._backward = lambda: None
        self._prev = set(_children)
        self._op = _op

    def __repr__(self):
        return f"Value(data={self.data:.4f}, grad={self.grad:.4f})"

    def __add__(self, other):
        other = other if isinstance(other, Value) else Value(other)
        out = Value(self.data + other.data, (self, other), "+")

        def _backward():
            self.grad += out.grad
            other.grad += out.grad

        out._backward = _backward
        return out

    def __mul__(self, other):
        other = other if isinstance(other, Value) else Value(other)
        out = Value(self.data * other.data, (self, other), "*")

        def _backward():
            self.grad += other.data * out.grad
            other.grad += self.data * out.grad

        out._backward = _backward
        return out

    def __pow__(self, other):
        assert isinstance(other, (int, float))
        out = Value(self.data ** other, (self,), f"**{other}")

        def _backward():
            self.grad += other * (self.data ** (other - 1)) * out.grad

        out._backward = _backward
        return out

    def relu(self):
        out = Value(max(0, self.data), (self,), "ReLU")

        def _backward():
            self.grad += (1.0 if self.data > 0 else 0.0) * out.grad

        out._backward = _backward
        return out

    def backward(self):
        topo = []
        visited = set()

        def build_topo(v):
            if v not in visited:
                visited.add(v)
                for child in v._prev:
                    build_topo(child)
                topo.append(v)

        build_topo(self)
        self.grad = 1.0
        for node in reversed(topo):
            node._backward()
```

## Worked example: $f = (a \cdot b + c)^2$

```python
a = Value(2.0)
b = Value(-3.0)
c = Value(10.0)

# Build graph: d = a*b, e = d + c, f = e^2
d = a * b          # -6
e = d + c          # 4
f = e ** 2         # 16

f.backward()

print("Forward:", f" d={d.data}, e={e.data}, f={f.data}")
print("Gradients:")
print(f"  df/da = {a.grad}")   # 2*e*b = 2*4*(-3) = -24
print(f"  df/db = {b.grad}")   # 2*e*a = 2*4*2 = 16
print(f"  df/dc = {c.grad}")   # 2*e = 8
```

**Check by hand:** $f = (ab + c)^2$. Let $u = ab + c$. Then $\frac{df}{du} = 2u = 8$.
- $\frac{df}{da} = 8 \cdot b = -24$ ✓
- $\frac{df}{db} = 8 \cdot a = 16$ ✓
- $\frac{df}{dc} = 8$ ✓

## A two-neuron network

Same math as $y = mx + b$, but every weight is a `Value`:

```python
# Inputs
x1 = Value(2.0)
x2 = Value(0.5)

# Weights (start wrong on purpose)
w1 = Value(-0.5)
w2 = Value(1.0)
b = Value(0.0)

# Forward: neuron = ReLU(w1*x1 + w2*x2 + b)
n = (x1 * w1 + x2 * w2 + b).relu()
label = Value(1.0)

# MSE loss
loss = (n - label) ** 2
loss.backward()

print(f"prediction={n.data:.3f}, loss={loss.data:.4f}")
print(f"grad w1={w1.grad:.4f}, grad w2={w2.grad:.4f}, grad b={b.grad:.4f}")

# One manual gradient-descent step (lr = 0.01)
lr = 0.01
w1.data -= lr * w1.grad
w2.data -= lr * w2.grad
b.data -= lr * b.grad
print(f"after step: w1={w1.data:.4f}, w2={w2.data:.4f}, b={b.data:.4f}")
```

Run this loop 50 times and watch the loss drop — same training loop as Section 0.3, but you built the engine.

## Visualizing the graph

```python
def trace(root):
    nodes, seen = [], set()

    def visit(v):
        if v not in seen:
            seen.add(v)
            for child in v._prev:
                visit(child)
            nodes.append(v)

    visit(root)
    return nodes

for v in trace(f):
    print(f"{v._op:>6}  data={v.data:7.3f}  grad={v.grad:7.3f}")
```

Each line is one node. Gradients flow **backward** through this list.

## Exercise: add subtraction and division

Extend `Value` with `__sub__` and `__truediv__`. Hint:

- $a - b$: gradient $+1$ to $a$, $-1$ to $b$.
- $a / b$: treat as $a \cdot b^{-1}$; use the power rule for $b$.

Test on $f = (a - b) / c$ with $a=6, b=2, c=2$ and verify gradients numerically:

```python
def num_grad(f, x, eps=1e-5):
    x.data += eps
    f1 = f()
    x.data -= 2 * eps
    f2 = f()
    x.data += eps
    return (f1 - f2) / (2 * eps)
```

## Where This Leads Next

Section 0.3 uses PyTorch's built-in autograd — the same chain rule, scaled to millions of parameters and GPU kernels. Section 0.4 stacks `nn.Linear` layers; every weight update still follows the pattern you just traced by hand.

## Key Takeaway

- **Autograd** = record operations forward, apply chain rule backward.
- Each operation stores a tiny `_backward` function with its **local derivative**.
- Training is just: forward → loss → `backward()` → nudge weights opposite the gradient.
- PyTorch's `loss.backward()` is micrograd at industrial scale.

## Checkpoint

You can explain backprop without invoking magic. Next: **Section 0.3 — PyTorch Autograd** (the production version).

## Further Reading (Optional)

- Karpathy's *micrograd* repository (github.com/karpathy/micrograd).
- Karpathy, *Neural Networks: Zero to Hero* — micrograd and backprop ninja videos.
- Rumelhart, Hinton, & Williams (1986). *Learning representations by back-propagating errors*. Nature.
