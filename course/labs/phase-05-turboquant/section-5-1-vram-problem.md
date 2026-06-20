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

# Section 5.1: Why Long Contexts Crash Your 3080 (The VRAM Problem)

**Goal:** Estimate KV cache bytes vs sequence length and find max context for 10 GB.

## What You Need to Know First

This section is mostly counting bytes — basic multiplication is enough. Helpful background you have already met:

- **Attention's keys (K) and values (V)** — the vectors every token produces so that later tokens can "look back" at it.
- **Autoregressive generation** — the model writes one token at a time, each time re-reading everything before it.
- **VRAM** — the memory on your GPU; like RAM, but for the graphics card, and it is limited (e.g. 10 GB on an RTX 3080).
- **FP16** — a number format that uses 16 bits = 2 bytes per value (half the size of normal FP32).

If you can multiply a few numbers together, you can follow every formula here.

## The Hidden Cost of Generation

Training and inference have fundamentally different VRAM profiles. During **training**,
the dominant cost is activations (stored for backpropagation) and optimizer states (Adam
stores 2 extra copies of every parameter). But during **inference** — especially
autoregressive generation — a different beast emerges: the **KV cache**.

Every time the model generates a new token, it needs to attend to **all previous tokens**.
To avoid recomputing all key/value projections from scratch at each step, we cache them.
This cache grows **linearly** with sequence length:

$$\text{KV cache} = 2 \times n_{\text{layers}} \times n_{\text{heads}} \times d_{\text{head}} \times T \times \text{bytes\_per\_elem}$$

The factor of 2 is for keys **and** values. For our 80M parameter model with 8 layers,
8 heads, and head dimension 64, this adds up fast.

**Why gradient checkpointing won't help:** Gradient checkpointing saves VRAM during
training by recomputing activations during the backward pass. But during inference there
is no backward pass — the KV cache is the irreducible cost of fast autoregressive generation.

## KV Cache Size Computation

```python
import torch
import matplotlib.pyplot as plt

def kv_cache_bytes(n_layers, n_heads, head_dim, seq_len, bytes_per_elem=2):
    # K and V each: layers * seq * heads * head_dim
    per_tensor = n_layers * seq_len * n_heads * head_dim * bytes_per_elem
    return 2 * per_tensor

cfg = dict(n_layers=8, n_heads=8, head_dim=64)  # d_model=512
for seq in [512, 2048, 8192, 16384]:
    mb = kv_cache_bytes(**cfg, seq_len=seq) / 1e6
    print(f"seq={seq}: KV cache ~{mb:.0f} MB (FP16)")

vram_budget_mb = 10_000 * 0.6  # 60% for KV, rest for weights/activations
print("Rough KV budget (MB):", vram_budget_mb)
```

## Exact KV Cache Sizes for Our 80M Model

Our model: 8 layers, 8 heads, head dimension 64 (d_model = 512), FP16 (2 bytes per element).

```python
seq_lengths = [512, 1024, 2048, 4096, 8192]
print(f"{'Seq Length':>12} | {'KV Cache (MB)':>14} | {'KV Cache (GB)':>14}")
print("-" * 46)
kv_sizes_mb = []
for seq in seq_lengths:
    mb = kv_cache_bytes(**cfg, seq_len=seq) / 1e6
    gb = mb / 1000
    kv_sizes_mb.append(mb)
    print(f"{seq:>12,} | {mb:>14.1f} | {gb:>14.3f}")
```

## Where Does VRAM Go? — Pie Chart Breakdown

Let's visualize the VRAM breakdown for our 80M model generating at 2048 tokens.

```python
model_weights_mb = 80e6 * 2 / 1e6
kv_cache_2048 = kv_cache_bytes(**cfg, seq_len=2048) / 1e6
activations_mb = 2048 * 512 * 2 * 8 / 1e6
optimizer_mb = 0  # inference only

categories = ["Model Weights\n(FP16)", "KV Cache\n(2048 tokens)", "Activations\n(est.)"]
sizes = [model_weights_mb, kv_cache_2048, activations_mb]

fig, ax = plt.subplots(figsize=(7, 7))
colors = ["#4C72B0", "#C44E52", "#55A868"]
wedges, texts, autotexts = ax.pie(
    sizes, labels=categories, colors=colors, autopct=lambda p: f"{p:.1f}%\n({p*sum(sizes)/100:.0f} MB)",
    startangle=90, textprops={"fontsize": 11}
)
ax.set_title("VRAM Breakdown During Inference (seq_len=2048)\n80M Parameter Model", fontsize=13)
plt.tight_layout()
plt.show()

for cat, s in zip(categories, sizes):
    print(f"  {cat.replace(chr(10), ' ')}: {s:.1f} MB")
print(f"  Total: {sum(sizes):.1f} MB")
```

## Visualization: KV Cache Growth vs Sequence Length

```python
seq_range = list(range(128, 16385, 128))
kv_mbs = [kv_cache_bytes(**cfg, seq_len=s) / 1e6 for s in seq_range]

fig, ax = plt.subplots(figsize=(10, 5))
ax.plot(seq_range, kv_mbs, "b-", linewidth=2)
ax.fill_between(seq_range, kv_mbs, alpha=0.15, color="blue")

budget_60pct = 10_000 * 0.6
ax.axhline(y=budget_60pct, color="red", linestyle="--", linewidth=1.5, label=f"60% of 10GB = {budget_60pct:.0f} MB")

for seq_mark in [512, 2048, 8192]:
    kv_mark = kv_cache_bytes(**cfg, seq_len=seq_mark) / 1e6
    ax.annotate(f"{seq_mark} tokens\n{kv_mark:.0f} MB", xy=(seq_mark, kv_mark),
                xytext=(seq_mark + 500, kv_mark + 200),
                arrowprops=dict(arrowstyle="->", color="gray"),
                fontsize=9, ha="left")

ax.set_xlabel("Sequence Length (tokens)", fontsize=12)
ax.set_ylabel("KV Cache Size (MB)", fontsize=12)
ax.set_title("KV Cache Memory vs Sequence Length (FP16, 80M Model)", fontsize=13)
ax.legend(fontsize=11)
ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.show()
```

## What Happens When You Run Out of VRAM

When the KV cache exceeds available GPU memory, several things can happen:

| Symptom | Cause |
|---------|-------|
| `CUDA out of memory` error | PyTorch can't allocate the next KV slice |
| Silent slowdown | OS starts swapping GPU memory to CPU (unified memory on some GPUs) |
| Generation stops mid-sequence | OOM during the autoregressive loop |
| Corrupted outputs | Partial allocations succeed but tensors are truncated |

**Why common training tricks don't help:**
- **Gradient checkpointing:** Trades compute for memory during backward pass. No backward pass during inference → no benefit.
- **Mixed precision (AMP):** KV cache is already FP16. FP32→FP16 is a one-time 2× savings, already applied.
- **Batch size reduction:** Helps, but batch=1 is already the minimum for single-user inference.

The only real solutions are: (1) reduce precision below FP16 (quantization — storing each number with fewer bits), (2) evict old KV entries (sliding window), or (3) compress the cache (TurboQuant — the cache-shrinking technique we build across this phase).

## Measure Live Allocation During Generate

```python
device = "cuda" if torch.cuda.is_available() else "cpu"
if device == "cuda":
    torch.cuda.reset_peak_memory_stats()
    cache = []
    for t in range(256):
        k = torch.randn(8, 8, 64, device=device)  # one token slice
        v = torch.randn(8, 8, 64, device=device)
        cache.append((k, v))
    peak = torch.cuda.max_memory_allocated() / 1e6
    print("Peak VRAM after 256 fake layers-step (MB):", peak)
else:
    print("CUDA not available — use formulas above.")
```

## Exercise: Compute Maximum Sequence Length for 10 GB VRAM

Given our 80M parameter model (FP16 weights), compute the **maximum sequence length** before
hitting a 10 GB VRAM budget. Account for model weights, a conservative activation estimate,
and the KV cache.

```python
total_vram_bytes = 10 * 1024**3  # 10 GB
model_weight_bytes = 80e6 * 2     # FP16 = 2 bytes per param
overhead_bytes = 200 * 1024**2    # ~200 MB for CUDA context, activations, etc.

available_for_kv = total_vram_bytes - model_weight_bytes - overhead_bytes

per_token_kv_bytes = 2 * cfg["n_layers"] * cfg["n_heads"] * cfg["head_dim"] * 2  # K+V, FP16

max_seq_len = int(available_for_kv / per_token_kv_bytes)

print(f"Total VRAM budget:        {total_vram_bytes / 1e9:.1f} GB")
print(f"Model weights (FP16):     {model_weight_bytes / 1e6:.0f} MB")
print(f"Overhead estimate:        {overhead_bytes / 1e6:.0f} MB")
print(f"Available for KV cache:   {available_for_kv / 1e6:.0f} MB")
print(f"KV bytes per token:       {per_token_kv_bytes:,} bytes")
print(f"Maximum sequence length:  {max_seq_len:,} tokens")
print(f"\nVerification: KV at max_seq = {kv_cache_bytes(**cfg, seq_len=max_seq_len) / 1e6:.1f} MB")
print(f"Total VRAM used: {(model_weight_bytes + overhead_bytes + kv_cache_bytes(**cfg, seq_len=max_seq_len)) / 1e9:.2f} GB")
```

---

## Where This Leads Next

You have now measured exactly *why* long contexts blow up VRAM. **Section 5.2 (PolarQuant)** takes the first step toward fixing it: rotating the KV vectors before quantizing so that a few "outlier" numbers stop wrecking the compression — the groundwork for the 3.5-bit cache you build in Section 5.3.

## Key Takeaway

During autoregressive generation, the **KV cache** grows linearly with sequence length and
quickly dominates VRAM — eclipsing model weights for long contexts. For our 80M model at FP16,
the cache reaches ~134 MB at 8192 tokens. Training-time tricks like gradient checkpointing
are irrelevant at inference. The only paths to longer contexts are **quantizing** the KV cache
(the focus of Sections 5.2–5.3), using sliding-window attention, or compressing cached
representations — which is exactly what TurboQuant achieves.

## Further Reading (Optional)

**Optional — you do NOT need these to continue. They are for curious students who want the original sources.**

- Pope et al. (2023). *Efficiently Scaling Transformer Inference*. MLSys.
- Ainslie et al. (2023). *GQA: Training Generalized Multi-Query Transformer Models*. EMNLP.
