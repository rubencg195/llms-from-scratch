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

# Section 0.1: Tensors — Just Fancy Python Lists of Numbers

**Goal:** Move from Python lists to PyTorch tensors and perform element-wise math on CPU and GPU.

## What You Need to Know First

- **Basic Python lists** — how to make a list like `[1, 2, 3]` and a list of lists like `[[1, 2], [3, 4]]`.
- **The algebra idea of a slope** — that a line like $y = 2x + 1$ goes up by a fixed amount for each step (we lean on this intuition later when numbers get scaled and shifted).

This is the very first section, so that is genuinely all you need — no prior PyTorch, calculus, or machine-learning knowledge is assumed. ("Element-wise" just means an operation is applied to each number in the tensor separately.)

## What Is a Tensor, Really?

Think of a tensor as a **spreadsheet that can have any number of dimensions**:

| Rank | Math Name | Everyday Analogy | Example Shape |
|------|-----------|-------------------|---------------|
| 0 | Scalar | A single temperature reading | `()` |
| 1 | Vector | A row of sensor readings | `(768,)` |
| 2 | Matrix | A spreadsheet (rows × cols) | `(32, 768)` |
| 3 | 3-D Tensor | A stack of spreadsheets | `(4, 32, 768)` |
| 4+ | N-D Tensor | Batch of stacks of spreadsheets | `(2, 8, 32, 64)` |

In an LLM, a typical hidden state is a 3-D tensor shaped `(batch_size, sequence_length, d_model)` — a batch of sequences, each containing `d_model`-dimensional vectors at every token position.

## Why this matters

Every input to an LLM—token IDs, embeddings, activations—is a **tensor**. If you can create and slice tensors, you can read any model code.

```python
import torch
import matplotlib.pyplot as plt
import time

# A scalar (0-D tensor)
x = torch.tensor(3.14)
print("scalar:", x, "shape:", x.shape)

# A vector (1-D) — like a row of features
vec = torch.tensor([1.0, 2.0, 3.0])
print("vector shape:", vec.shape)

# A matrix (2-D) — batch of vectors
mat = torch.randn(4, 3)  # 4 rows, 3 columns
print("matrix shape:", mat.shape)
print(mat)
```

## From lists to tensors

```python
python_list = [[1, 2], [3, 4]]
t = torch.tensor(python_list, dtype=torch.float32)
print(t)
print("dtype:", t.dtype)
```

## Tensor Dtypes: Precision Matters

Different numerical precisions trade off between memory, speed, and accuracy. This matters enormously when training or running LLMs.

| dtype | Bits | Use Case |
|-------|------|----------|
| `float32` | 32 | Default training precision; safe for learning |
| `float16` | 16 | Mixed-precision training; 2× memory savings |
| `bfloat16` | 16 | Same range as fp32, less precision; preferred for LLMs |
| `int8` | 8 | Quantized inference; 4× memory savings |
| `int64` | 64 | Token IDs (integers don't need decimals) |

```python
f32 = torch.randn(1000, 1000, dtype=torch.float32)
f16 = f32.to(torch.float16)
bf16 = f32.to(torch.bfloat16)

print(f"float32: {f32.element_size()} bytes/element → {f32.nbytes / 1e6:.1f} MB")
print(f"float16: {f16.element_size()} bytes/element → {f16.nbytes / 1e6:.1f} MB")
print(f"bfloat16: {bf16.element_size()} bytes/element → {bf16.nbytes / 1e6:.1f} MB")

# Precision difference: float16 can lose information
big_val = torch.tensor(65504.0, dtype=torch.float16)  # max float16
print(f"\nMax float16 value: {big_val.item()}")
print(f"float16 overflow: {(big_val + 1).item()}")  # inf!
print(f"bfloat16 handles larger range: {torch.tensor(65504.0, dtype=torch.bfloat16) + 1}")
```

## GPU placement (RTX 3080)

```python
device = "cuda" if torch.cuda.is_available() else "cpu"
print("Using device:", device)

t_gpu = torch.randn(1024, 1024, device=device)
# Element-wise ops run on GPU automatically
y = t_gpu * 2.0 + 1.0
print("result device:", y.device)
if device == "cuda":
    print("VRAM allocated (MB):", torch.cuda.memory_allocated() / 1e6)
```

## Indexing and reshaping

```python
a = torch.arange(12).reshape(3, 4)
print("original:\n", a)
print("row 1:", a[1])
print("column 2:", a[:, 2])
print("flatten:", a.reshape(-1).shape)
```

## Visualization: 2D Tensor Heatmap

Visualizing tensor values as heatmaps is invaluable for debugging attention patterns, embeddings, and weight matrices later in this course.

```python
torch.manual_seed(42)
heatmap_data = torch.randn(8, 12)

plt.figure(figsize=(8, 4))
plt.imshow(heatmap_data.numpy(), cmap="coolwarm", aspect="auto")
plt.colorbar(label="Value")
plt.title("2D Tensor Visualized as Heatmap")
plt.xlabel("Columns (features)")
plt.ylabel("Rows (batch/sequence positions)")
plt.tight_layout()
plt.savefig("tensor_heatmap.png", dpi=120)
plt.show()
print("Saved tensor_heatmap.png")
```

## Broadcasting Rules

Broadcasting lets PyTorch perform element-wise operations on tensors of **different shapes** without copying data. The rules:

1. Dimensions are compared from the **right** (trailing dimensions)
2. Two dimensions are compatible if they are **equal** or one of them is **1**
3. A dimension of size 1 is "stretched" to match the other

```python
# Rule in action: (3, 4) + (4,) — the vector is broadcast across rows
matrix = torch.ones(3, 4)
row_vec = torch.tensor([1.0, 2.0, 3.0, 4.0])
result = matrix + row_vec
print("(3,4) + (4,) =", result.shape)
print(result)

# Column broadcast: (3, 4) + (3, 1) — the column is broadcast across columns
col_vec = torch.tensor([[10.0], [20.0], [30.0]])
result2 = matrix + col_vec
print("\n(3,4) + (3,1) =", result2.shape)
print(result2)

# Common LLM pattern: apply a per-head scale factor
# scores shape: (batch, heads, seq, seq) * scale shape: (1, heads, 1, 1)
batch, heads, seq = 2, 8, 16
scores = torch.randn(batch, heads, seq, seq)
scale = torch.randn(1, heads, 1, 1)  # different scale per head
scaled_scores = scores * scale
print(f"\nAttention broadcast: {scores.shape} * {scale.shape} → {scaled_scores.shape}")
```

## Exercise 1: Basic Statistics

Create a tensor of shape `(8, 16)` with random normal values on your GPU (or CPU). Compute its mean and standard deviation along dimension 0.

```python
ex = torch.randn(8, 16, device=device)
print("mean per column:", ex.mean(dim=0).shape)
print("std per column:", ex.std(dim=0).shape)
```

## Exercise 2: Matrix Multiplication Timing — CPU vs GPU

Large matrix multiplies are the core operation in every neural network forward pass. Let's measure how much faster GPU is.

```python
sizes = [256, 512, 1024, 2048, 4096]
cpu_times = []
gpu_times = []

for n in sizes:
    a_cpu = torch.randn(n, n)
    b_cpu = torch.randn(n, n)

    # CPU timing
    start = time.time()
    _ = a_cpu @ b_cpu
    cpu_times.append(time.time() - start)

    if device == "cuda":
        a_gpu = a_cpu.to(device)
        b_gpu = b_cpu.to(device)
        torch.cuda.synchronize()
        start = time.time()
        _ = a_gpu @ b_gpu
        torch.cuda.synchronize()
        gpu_times.append(time.time() - start)

print(f"{'Size':<8} {'CPU (ms)':<12} {'GPU (ms)':<12} {'Speedup':<10}")
print("-" * 42)
for i, n in enumerate(sizes):
    cpu_ms = cpu_times[i] * 1000
    if gpu_times:
        gpu_ms = gpu_times[i] * 1000
        speedup = cpu_ms / gpu_ms if gpu_ms > 0 else float("inf")
        print(f"{n:<8} {cpu_ms:<12.2f} {gpu_ms:<12.2f} {speedup:<10.1f}x")
    else:
        print(f"{n:<8} {cpu_ms:<12.2f} {'N/A':<12} {'N/A':<10}")
```

## Where This Leads Next

Now that you can build and slice tensors, the next section (0.2) introduces the single most important thing you can *do* with two tensors: the **dot product**. It is just "multiply matching numbers and add them up," and it turns out to be how neural networks measure similarity between vectors.

## Key Takeaway

- **Tensors** are n-dimensional arrays optimized for GPU parallelism — they are the universal data structure of deep learning.
- **Dtypes** control precision/memory tradeoffs: `float32` for stable training, `float16`/`bfloat16` for speed, `int8` for deployment.
- **Broadcasting** eliminates explicit loops and enables elegant vectorized code.
- **GPU acceleration** provides 10–100× speedups on matrix operations — the foundation of every neural network forward and backward pass.

## Checkpoint

You can create tensors, move them to CUDA, index them, and reshape them. Next: **dot products** (Section 0.2) — the atomic operation that computes similarity between vectors.

## Further Reading (Optional)

**Optional — you do NOT need these to continue. They are for curious students who want the original sources.**

- Paszke et al. (2019). *PyTorch: An Imperative Style, High-Performance Deep Learning Library*. NeurIPS.
- Harris et al. (2020). *Array programming with NumPy*. Nature.
