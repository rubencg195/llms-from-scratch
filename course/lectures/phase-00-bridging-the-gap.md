---
title: "Phase 0: Bridging the Gap"
subtitle: "The Mathematical Primer"
author: "LLMs From Scratch"
---

# Phase 0: Bridging the Gap

## The Mathematical Primer

From Python loops and $y = mx + b$ to tensors and autograd.

<!-- notes: Welcome slide. Set the tone: this course takes you from knowing Python basics to training a Titans-class model on a single RTX 3080. Phase 0 is the on-ramp — no ML experience required, only curiosity. -->

---

## Course Welcome

**LLMs From Scratch** — a 9-phase journey:

| Phase | Milestone |
|-------|-----------|
| 0 | Math primer (you are here) |
| 1 | 80M Transformer on TinyStories |
| 2 | Instruction tuning & tool calling |
| 3 | Quantization-aware training |
| 4 | Mixture of Experts |
| 5–8 | KV compression, RL, memory (Titans) |

**Start:** Python loops and `y = mx + b`
**End:** A Titans-class model trained entirely on one RTX 3080

<!-- notes: Emphasize that every line of training code will be written by the student. No black-box API calls — we build the internals. The RTX 3080 constraint forces us to think carefully about memory, which is the real engineering skill. -->

---

## Before You Begin (Prerequisites)

This is the very first phase, so the entry bar is intentionally low.

**You need only two things:**

- **Basic Python** — variables, lists, `for` loops, and functions. That's it.
- **The algebra equation $y = mx + b$** — a straight line with a slope $m$ and an intercept $b$, exactly as taught in high school.

**Everything else is taught here, from zero.** You do **not** need calculus, linear algebra, statistics, or any prior machine-learning experience. Terms like *tensor*, *gradient*, and *autograd* are all introduced in this phase.

> If you can read a `for` loop and you remember $y = mx + b$, you are ready.

<!-- notes: Reassure nervous students directly. This is the on-ramp phase, so the only true prerequisites are basic Python and the line equation y = mx + b. Calculus is NOT required — autograd handles derivatives for us. Every other concept (tensors, dot products, loss, autograd, GPUs) is defined from scratch in the slides that follow. -->

---

## Learning objectives

- Represent data as **tensors** (lists of numbers on the GPU)
- Compute **dot products** as similarity scores
- Use **PyTorch autograd** to adjust weights automatically
- Implement a single **linear layer** ($y = mx + b$ at scale)
- Understand **loss functions** and why we need them
- Know why GPUs matter for deep learning

<!-- notes: By the end of Phase 0, students should be able to write a forward pass, compute a loss, call backward(), and update weights. That's the entire training loop in miniature. -->

---

## What is a tensor?

A tensor is just a **multi-dimensional array of numbers**.

```
0D  Scalar      42              a single number
1D  Vector      [3, 1, 4]      a list
2D  Matrix      [[1, 2],       a table / spreadsheet
                 [3, 4]]
3D  Cube        [[[...], ...]]  e.g., a color image
4D  Batch       [[[[...]]]]    e.g., a batch of images
```

**Concrete examples:**

- A single pixel (RGB) = 3 numbers → **1D tensor**, shape `[3]`
- A 28×28 grayscale image = 784 numbers → **2D tensor**, shape `[28, 28]`
- A color photo = height × width × 3 → **3D tensor**, shape `[H, W, 3]`
- A training batch of 16 color photos → **4D tensor**, shape `[16, H, W, 3]`

<!-- notes: The key insight is that EVERYTHING in deep learning — text, images, audio — gets converted into tensors before the model sees it. A sentence becomes a 2D tensor of shape [seq_len, embedding_dim]. An entire batch of sentences becomes 3D. -->

---

## Why tensors over Python lists?

| Python list | PyTorch tensor |
|-------------|----------------|
| Lives in RAM | Can live on **GPU** |
| Slow `for` loops | **Vectorized** CUDA ops |
| No gradients | **Autograd** tracks errors |
| No shape metadata | `.shape`, `.dtype`, `.device` |

```python
import torch
a = torch.tensor([1.0, 2.0, 3.0])   # lives on CPU
a = a.cuda()                          # now on GPU
b = a * 2                             # parallel multiply — no loop
```

A GPU can multiply **billions** of numbers per second because it has thousands of cores doing the same operation simultaneously.

<!-- notes: Demonstrate in lab 0.1. Stress that the syntax looks like NumPy, but `.cuda()` moves computation to the GPU. The autograd engine is the real superpower: it records every operation so it can compute gradients automatically. -->

---

## The core intuition: $y = mx + b$

A neural network is a long chain of linear equations:

$$y = mx + b$$

- $x$ = input data
- $m$ = **weight** (learned slope)
- $b$ = **bias** (learned intercept)

You learned this in algebra. A neuron is **literally** this equation.

<!-- notes: This is the single most important slide for students who are nervous about the math. If you understand y = mx + b, you understand the atom of deep learning. Everything else is scaling this up and adding non-linearities. -->

---

## From one neuron to a network

```
Single neuron:     y = w·x + b         (1 weight, 1 bias)

A layer of 4:      y₁ = w₁·x + b₁     (4 weights, 4 biases)
                   y₂ = w₂·x + b₂
                   y₃ = w₃·x + b₃
                   y₄ = w₄·x + b₄

In matrix form:    Y = W·X + B         (weight MATRIX)

A network:         Layer1 → ReLU → Layer2 → ReLU → Layer3
                   (chain of matrix multiplies with non-linearities)
```

**ReLU** (the non-linearity above) just means "keep positive numbers, set negatives to zero" — a simple bend that lets stacked lines learn curves.

At scale, $x$, $m$, and $b$ are not scalars — they are **tensors**.

- A single neuron: **2 parameters** $(w, b)$
- A layer of 512 neurons taking 512 inputs: **262,144 + 512 = 262,656 parameters**
- Our Phase 1 model: **~80 million parameters**

<!-- notes: Walk through the scaling visually. The jump from 2 parameters to 80 million is just repeating y=mx+b in a matrix and stacking layers. The non-linearity (ReLU) between layers is what lets the network learn curves, not just lines. -->

---

## The dot product — worked example

Given two equal-length vectors, multiply matching entries and sum:

$$\mathbf{a} \cdot \mathbf{b} = \sum_i a_i b_i$$

**Example:** $\mathbf{a} = [1, 2, 3]$, $\mathbf{b} = [4, 5, 6]$

$$\mathbf{a} \cdot \mathbf{b} = (1 \times 4) + (2 \times 5) + (3 \times 6) = 4 + 10 + 18 = 32$$

**Interpretation:** How similar are two directions in space?

- Vectors pointing the **same** way → large positive dot product
- Vectors at **right angles** → dot product is 0
- Vectors pointing **opposite** → large negative dot product

Attention (Phase 1) is dot products all the way down.

<!-- notes: Do this calculation live. Then ask: what if a = [1,0,0] and b = [0,1,0]? The dot product is 0 — they're orthogonal, meaning unrelated. In attention, the query and key vectors are compared via dot product to decide which tokens should attend to which. If the dot product is high, the model pays attention to that token. -->

---

## What is a loss function?

The loss is a **single number** that measures how wrong the model is.

**Mean Squared Error (MSE)** — for regression:

$$\mathcal{L}_{\text{MSE}} = \frac{1}{n} \sum_{i=1}^{n} (y_i - \hat{y}_i)^2$$

Predict 3.0 when truth is 5.0 → error = $(5-3)^2 = 4$

**Cross-Entropy** — for classification (next-token prediction):

$$\mathcal{L}_{\text{CE}} = -\sum_{c} y_c \log(\hat{y}_c)$$

Model says "cat" has 80% probability, truth is "cat" → low loss.
Model says "cat" has 2% probability, truth is "cat" → **high** loss.

The entire goal of training: **make the loss smaller**.

<!-- notes: MSE is intuitive — it's just average squared error. Cross-entropy is what we actually use for language models. Intuition: cross-entropy punishes confident wrong answers harshly. If you say 2% for the right word, you get a huge penalty. If you say 80%, small penalty. This is why language models learn to spread probability mass wisely. -->

---

## Autograd: the blame game

1. **Forward pass:** compute prediction $\hat{y} = f(x; \theta)$
2. **Loss:** measure wrongness (single number)
3. **Backward pass:** `loss.backward()` distributes blame to every weight
4. **Optimizer step:** nudge weights to reduce loss

```
         x ──→ [Linear] ──→ [ReLU] ──→ [Linear] ──→ ŷ
                  ↓                        ↓
               w₁, b₁                  w₂, b₂
                  ↑                        ↑
              ∂L/∂w₁  ←── backward ←──  ∂L/∂w₂  ←── Loss
```

You never hand-write calculus for 80 million parameters — **autograd does it**.

<!-- notes: The computation graph is the key concept. PyTorch records every operation during the forward pass as a directed acyclic graph. When you call .backward(), it walks the graph in reverse, applying the chain rule at each node. This is backpropagation. The beauty is that you write only the forward pass; PyTorch derives the backward pass automatically. -->

---

## The computation graph — closer look

Consider $L = (w \cdot x - y)^2$. PyTorch builds:

```
  w ───┐
       ├──→ [multiply] ──→ [subtract y] ──→ [square] ──→ L
  x ───┘

  Forward:   w·x = 6    →   6 - 5 = 1    →   1² = 1

  Backward:  ∂L/∂w ← chain rule through each node
             = 2(wx-y) · x
             = 2(1) · 2 = 4
```

Every tensor with `requires_grad=True` gets a `.grad` attribute after `loss.backward()`.

<!-- notes: Walk through this with w=3, x=2, y=5. Forward: 3*2=6, 6-5=1, 1^2=1. Backward: derivative of square is 2(wx-y)=2, times derivative of multiply w.r.t. w is x=2, so dL/dw=4. PyTorch does exactly this, just for millions of parameters simultaneously. This is the chain rule, automated. -->

---

## Micrograd: autograd in 100 lines (Lab 0.25)

Before trusting PyTorch, build it once:

```python
class Value:
    def __init__(self, data, _children=(), _op=""):
        self.data = data
        self.grad = 0.0
        self._backward = lambda: None
        self._prev = set(_children)

    def __mul__(self, other):
        out = Value(self.data * other.data, (self, other), "*")
        def _backward():
            self.grad += other.data * out.grad
            other.grad += self.data * out.grad
        out._backward = _backward
        return out
    # ... +, ReLU, backward() walks the graph in reverse
```

**Karpathy's insight:** every op stores its local derivative. `backward()` multiplies them via the chain rule — exactly what `loss.backward()` does at scale.

<!-- notes: Lab 0.25 implements the full Value class. Students who do this lab report that Section 0.3 suddenly makes sense. The "backprop ninja" exercise — manually tracing gradients through a small graph — is the single best debug skill for when training breaks later. -->

---

## Why GPUs?

| | CPU (Ryzen 5800X) | GPU (RTX 3080) |
|---|---|---|
| Cores | 16 threads | **8,704** CUDA cores |
| FP32 TFLOPS | ~0.5 | **~30** |
| FP16 TFLOPS | ~1 | **~60** |
| Memory BW | ~50 GB/s | **~760 GB/s** |

A matrix multiply of two $[4096 \times 4096]$ matrices:

- CPU: ~140 billion operations → **~280 seconds** naively
- GPU: same 140 billion operations → **~5 ms** (parallel)

**Analogy:** A CPU is a sports car (fast, one lane). A GPU is a highway (slower per lane, but **thousands of lanes**).

<!-- notes: The memory bandwidth number is often more important than raw FLOPS. Models are frequently memory-bound, not compute-bound, especially during inference. This is why quantization (Phase 3) matters so much — smaller numbers = more fit in cache = faster. The RTX 3080's 10GB VRAM is our constraint throughout the course. -->

---

## Phase 0 lab map

| Lab | Topic | Key skill |
|-----|-------|-----------|
| 0.1 | Tensors | Create, reshape, move to GPU |
| 0.2 | Dot product | Similarity via multiplication |
| 0.25 | **Micrograd** | Build autograd from scratch (chain rule) |
| 0.3 | Autograd | `loss.backward()`, `.grad` in PyTorch |
| 0.4 | First neural layer | $y = Wx + b$ in PyTorch |

Dataset: **synthetic random arrays** (no downloads needed).

All labs run on CPU or GPU — Phase 0 uses negligible VRAM.

<!-- notes: Lab 0.25 (micrograd) is optional but strongly recommended for beginners — Karpathy's approach of building autograd once before trusting PyTorch makes Section 0.3 click. Each lab is a Jupyter notebook. Encourage students to modify and break things. -->

---

## Hardware note

All code targets **RTX 3080 (10 GB VRAM)**.

Phase 0 uses negligible VRAM — focus on **correctness** and **intuition**.

```
VRAM Budget across the course:

Phase 0:  ~0.01 GB  (toy tensors)
Phase 1:  ~4-6 GB   (80M model + gradients)
Phase 2:  ~3-4 GB   (fine-tuning, shorter seqs)
Phase 3:  ~3 GB     (quantized weights)
Phase 4:  ~6-8 GB   (MoE experts share VRAM)
```

<!-- notes: The VRAM budget is intentionally tight. Real ML engineering is about fitting your ambitions into your hardware. We'll learn gradient checkpointing, mixed precision, and quantization precisely because we need them, not as academic exercises. -->

---

## Key takeaways

1. **Tensors** are just arrays of numbers — the universal data format
2. **$y = mx + b$** is the atom; a network is millions of atoms chained together
3. **Dot products** measure similarity — the heart of attention
4. **Loss** measures wrongness; **autograd** distributes blame
5. **GPUs** let us do trillions of simple operations per second

These five ideas carry you through the entire course.

<!-- notes: Before moving to questions, reiterate that Phase 0 is intentionally gentle. If students understand these five bullet points, they're ready for Phase 1 where we build a real Transformer. The math doesn't get harder — it just gets bigger. -->

---

## Bridge to the Next Phase

**What you built in Phase 0:** the complete training loop in miniature — tensors to hold data, $y = mx + b$ as the building block, the dot product as a similarity score, a loss to measure wrongness, and autograd to fix the weights.

**The single thread into Phase 1:** the **dot product** you computed by hand becomes **self-attention**. In Phase 1, every word is turned into a vector, and the model uses dot products between those vectors to decide which words should "pay attention" to which — that is the entire heart of the Transformer.

You already own all five tools. Phase 1 just arranges them into a real language model.

<!-- notes: Make the connection explicit and concrete. The dot product from slide "The dot product — worked example" is literally the operation inside attention (QK^T). The y=mx+b linear layer becomes the projections (W_Q, W_K, W_V) and the FFN. Autograd and cross-entropy loss carry over unchanged into the Phase 1 training loop. Nothing new mathematically — just bigger and arranged into a Transformer. -->

---

## Questions?

Next: **Phase 1 — The Dense Core**

We will build a full 80M-parameter Transformer from scratch, train it on TinyStories, and watch it learn to write children's stories.

Preview of Phase 1 concepts:
- Tokenization and embeddings
- Self-attention (dot products at scale)
- Rotary position encodings
- The autoregressive training loop

<!-- notes: Take questions. Common ones: "Do I need to know calculus?" (No — autograd handles it, but understanding the chain rule helps intuition.) "What if I don't have an RTX 3080?" (Any CUDA GPU works; CPU works too but slower.) "How long does Phase 1 training take?" (About 2-4 hours on an RTX 3080.) -->

---

## Further Reading (Optional)

**These papers are optional enrichment — you do NOT need to read any of them to continue the course.**

- Rumelhart, Hinton, & Williams (1986). *Learning representations by back-propagating errors*. Nature.
- Paszke et al. (2019). *PyTorch: An Imperative Style, High-Performance Deep Learning Library*. NeurIPS.
- Goodfellow, Bengio, & Courville (2016). *Deep Learning*. MIT Press. (free at deeplearningbook.org)
- Nielsen (2015). *Neural Networks and Deep Learning*. (free online book)

<!-- notes: Strictly optional. If a student is curious where backprop came from, the 1986 Rumelhart paper is the classic. The PyTorch paper explains the autograd engine we lean on. Goodfellow and Nielsen are both free and beginner-friendly companions, but the course is fully self-contained without them. -->
