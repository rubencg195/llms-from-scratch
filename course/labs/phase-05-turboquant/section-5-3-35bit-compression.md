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

# Section 5.3: 3.5-Bit Compression — Writing the Shorthand Notes

**Goal:** Pack quantized KV values at 3.5 bits/elem (conceptual nibble packing) and decompress for attention.

## The Memory Savings Story

Going from FP16 (16 bits) to 3.5 bits per element is a **~4.57× compression** ratio:

$$\text{compression ratio} = \frac{16}{3.5} \approx 4.57\times$$

For our 80M model at 8192 tokens, this means:

| Format | Bits/elem | KV Cache Size | Savings |
|--------|-----------|---------------|---------|
| FP16   | 16        | ~134 MB       | baseline |
| INT8   | 8         | ~67 MB        | 2× |
| INT4   | 4         | ~34 MB        | 4× |
| 3.5-bit| 3.5       | ~29 MB        | 4.57× |

This compression enables context lengths that would otherwise OOM on consumer GPUs.

## Non-uniform Codebooks — How K-Means Enables Fractional Bits

Standard quantization uses **uniform** grids: evenly-spaced levels between min and max.
But real weight/activation distributions are not uniform — they're typically bell-shaped
with heavy tails.

**Non-uniform codebooks** use k-means clustering to find the optimal set of representative
values (centroids). With $k$ centroids, you need $\lceil\log_2 k\rceil$ bits to store
the cluster index. The trick for 3.5 bits: use **11 centroids** ($\log_2 11 \approx 3.46$ bits).

In practice, pairs of values share a nibble boundary, or a lookup table maps 11 levels
into packed storage.

```python
import torch
import time
import matplotlib.pyplot as plt

def kmeans_codebook(data, n_clusters=11, n_iters=20):
    """Simple 1D k-means to find non-uniform quantization levels."""
    flat = data.view(-1)
    indices = torch.randperm(len(flat))[:n_clusters]
    centroids = flat[indices].clone()

    for _ in range(n_iters):
        dists = (flat.unsqueeze(1) - centroids.unsqueeze(0)).abs()
        assignments = dists.argmin(dim=1)
        for c in range(n_clusters):
            mask = (assignments == c)
            if mask.any():
                centroids[c] = flat[mask].mean()
    return centroids.sort().values

torch.manual_seed(42)
sample_data = torch.randn(10000)
codebook = kmeans_codebook(sample_data, n_clusters=11)
print("3.5-bit codebook (11 levels):")
print([f"{v:.3f}" for v in codebook.tolist()])
```

## Compression and Decompression

```python
BITS = 3.5  # 11 levels in 4-bit storage with shared codebook — simplified lab uses 4-bit
QMIN, QMAX = -8, 7

def compress_kv(tensor, R):
    x = tensor @ R
    mn, mx = x.min(), x.max()
    scale = (mx - mn).clamp(min=1e-8) / (QMAX - QMIN)
    zp = QMIN - mn / scale
    codes = torch.round(x / scale + zp).clamp(QMIN, QMAX).to(torch.int8)
    meta = {"scale": scale.item(), "zp": zp.item(), "R": R}
    return codes, meta

def decompress_kv(codes, meta):
    x = (codes.float() - meta["zp"]) * meta["scale"]
    return x @ meta["R"].T

def random_orthogonal(d):
    q, _ = torch.linalg.qr(torch.randn(d, d))
    return q

T, D = 512, 64
k = torch.randn(T, D)
R = random_orthogonal(D)
codes, meta = compress_kv(k, R)
k_hat = decompress_kv(codes, meta)
print("compression MSE:", (k - k_hat).pow(2).mean().item())
print("codes dtype/size:", codes.dtype, codes.numel())
```

## Better TurboKVCache with Per-Head Rotation Matrices

The improved cache stores a separate rotation matrix per **layer and head**, enabling
each attention head to have its own optimal rotation for its value distribution.

```python
class TurboKVCache:
    """KV cache with per-head rotation matrices and 3.5-bit compression."""
    def __init__(self, n_layers, n_heads, d_head):
        self.n_layers = n_layers
        self.n_heads = n_heads
        self.d_head = d_head
        self.store = [[[] for _ in range(n_heads)] for _ in range(n_layers)]
        self.R = [[random_orthogonal(d_head) for _ in range(n_heads)] for _ in range(n_layers)]

    def append(self, layer, head, k, v):
        """Compress and store a single token's K and V for one layer/head."""
        R_lh = self.R[layer][head]
        ck, mk = compress_kv(k, R_lh)
        cv, mv = compress_kv(v, R_lh)
        self.store[layer][head].append((ck, mk, cv, mv))

    def get(self, layer, head):
        """Decompress all cached K, V for a given layer/head."""
        if not self.store[layer][head]:
            return None, None
        ks, vs = [], []
        for ck, mk, cv, mv in self.store[layer][head]:
            ks.append(decompress_kv(ck, mk))
            vs.append(decompress_kv(cv, mv))
        return torch.stack(ks), torch.stack(vs)

    def get_layer(self, layer):
        """Decompress all heads for a layer, returning (K, V) of shape (n_heads, T, D)."""
        all_k, all_v = [], []
        for h in range(self.n_heads):
            k, v = self.get(layer, h)
            if k is not None:
                all_k.append(k)
                all_v.append(v)
        if not all_k:
            return None, None
        return torch.stack(all_k), torch.stack(all_v)

    def memory_bytes(self):
        """Estimate compressed cache size in bytes."""
        total = 0
        for layer in self.store:
            for head in layer:
                for ck, mk, cv, mv in head:
                    total += ck.numel() + cv.numel()  # int8 = 1 byte each
        return total

n_layers, n_heads, d_head = 8, 8, 64
cache = TurboKVCache(n_layers, n_heads, d_head)

for t in range(32):
    for layer in range(n_layers):
        for head in range(n_heads):
            k_tok = torch.randn(d_head)
            v_tok = torch.randn(d_head)
            cache.append(layer, head, k_tok.unsqueeze(0), v_tok.unsqueeze(0))

K, V = cache.get_layer(0)
print(f"Layer 0 restored K shape: {K.shape}")
print(f"Layer 0 restored V shape: {V.shape}")
print(f"Compressed cache size: {cache.memory_bytes():,} bytes ({cache.memory_bytes()/1024:.1f} KB)")
```

## Memory Usage Comparison: FP16 vs INT8 vs 3.5-bit

```python
def cache_size_mb(n_layers, n_heads, d_head, seq_len, bits_per_elem):
    total_elements = 2 * n_layers * n_heads * d_head * seq_len  # K + V
    total_bits = total_elements * bits_per_elem
    return total_bits / 8 / 1e6

cfg = {"n_layers": 8, "n_heads": 8, "d_head": 64}
seq_lengths = [512, 1024, 2048, 4096, 8192]
formats = [
    ("FP16", 16, "#4C72B0"),
    ("INT8", 8, "#55A868"),
    ("3.5-bit", 3.5, "#C44E52"),
]

print(f"{'Seq Len':>8} | {'FP16 (MB)':>10} | {'INT8 (MB)':>10} | {'3.5-bit (MB)':>12} | {'FP16→3.5b Savings':>18}")
print("-" * 70)
for seq in seq_lengths:
    fp16 = cache_size_mb(**cfg, seq_len=seq, bits_per_elem=16)
    int8 = cache_size_mb(**cfg, seq_len=seq, bits_per_elem=8)
    bit35 = cache_size_mb(**cfg, seq_len=seq, bits_per_elem=3.5)
    print(f"{seq:>8,} | {fp16:>10.1f} | {int8:>10.1f} | {bit35:>12.1f} | {fp16/bit35:>17.1f}×")

fig, ax = plt.subplots(figsize=(10, 5))
for name, bits, color in formats:
    sizes = [cache_size_mb(**cfg, seq_len=s, bits_per_elem=bits) for s in seq_lengths]
    ax.plot(seq_lengths, sizes, "-o", label=f"{name} ({bits} bits)", color=color, linewidth=2)

ax.set_xlabel("Sequence Length", fontsize=12)
ax.set_ylabel("KV Cache Size (MB)", fontsize=12)
ax.set_title("KV Cache Memory: FP16 vs INT8 vs 3.5-bit", fontsize=13)
ax.legend(fontsize=11)
ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.show()
```

## Integration with Attention Computation

During attention, the KV cache is decompressed on-the-fly: decompress → attend → discard
the decompressed tensors. Only the compressed codes persist in memory.

```python
def turbo_attention(query, cache, layer, head, d_head):
    """
    Compute attention using compressed KV cache.
    1. Decompress K, V for this layer/head
    2. Compute standard scaled dot-product attention
    3. Decompressed tensors are transient — only compressed cache persists
    """
    K, V = cache.get(layer, head)
    if K is None:
        return torch.zeros_like(query)

    K = K.squeeze(1)  # (T, D)
    V = V.squeeze(1)  # (T, D)

    scores = (query @ K.T) / (d_head ** 0.5)
    attn_weights = torch.softmax(scores, dim=-1)
    output = attn_weights @ V
    # K, V go out of scope here — only compressed cache remains in memory
    return output

query = torch.randn(1, d_head)
attn_out = turbo_attention(query, cache, layer=0, head=0, d_head=d_head)
print(f"Attention output shape: {attn_out.shape}")
```

## Exercise: Benchmark Compress + Decompress Latency vs Memory Savings

Measure the time cost of compression and decompression at various sequence lengths,
and compare it to the memory savings achieved.

```python
d_head = 64
R_bench = random_orthogonal(d_head)

results = []
for seq_len in [128, 512, 1024, 2048, 4096]:
    k_data = torch.randn(seq_len, d_head)

    t0 = time.perf_counter()
    for _ in range(100):
        codes_b, meta_b = compress_kv(k_data, R_bench)
    compress_ms = (time.perf_counter() - t0) / 100 * 1000

    t0 = time.perf_counter()
    for _ in range(100):
        _ = decompress_kv(codes_b, meta_b)
    decompress_ms = (time.perf_counter() - t0) / 100 * 1000

    fp16_bytes = k_data.numel() * 2
    compressed_bytes = codes_b.numel() * 1
    savings = fp16_bytes / compressed_bytes

    results.append({
        "seq_len": seq_len,
        "compress_ms": compress_ms,
        "decompress_ms": decompress_ms,
        "savings": savings,
    })

print(f"{'Seq Len':>8} | {'Compress (ms)':>13} | {'Decompress (ms)':>15} | {'Memory Savings':>14}")
print("-" * 58)
for r in results:
    print(f"{r['seq_len']:>8,} | {r['compress_ms']:>13.3f} | {r['decompress_ms']:>15.3f} | {r['savings']:>13.1f}×")

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
sl = [r["seq_len"] for r in results]
ax1.plot(sl, [r["compress_ms"] for r in results], "-o", color="#4C72B0", label="Compress")
ax1.plot(sl, [r["decompress_ms"] for r in results], "-s", color="#C44E52", label="Decompress")
ax1.set_xlabel("Sequence Length")
ax1.set_ylabel("Latency (ms)")
ax1.set_title("Compress/Decompress Latency")
ax1.legend()
ax1.grid(True, alpha=0.3)

ax2.bar(range(len(sl)), [r["savings"] for r in results], color="#55A868")
ax2.set_xticks(range(len(sl)))
ax2.set_xticklabels(sl)
ax2.set_xlabel("Sequence Length")
ax2.set_ylabel("Memory Savings (×)")
ax2.set_title("Compression Ratio")
ax2.grid(True, alpha=0.3, axis="y")

plt.tight_layout()
plt.show()
```

**3.5-bit note:** Production TurboQuant uses non-uniform codebooks; lab uses 4-bit uniform as teaching proxy.

---

## Key Takeaway

3.5-bit KV cache compression achieves ~4.5× memory savings over FP16, enabling significantly
longer context windows on the same hardware. The key enabler is **non-uniform codebooks** via
k-means clustering, which allocate representation levels where the data actually lives rather
than on a uniform grid. The per-head rotation matrix from PolarQuant (Section 5.2) ensures
outliers don't degrade quantization quality. During attention, compressed KV entries are
decompressed on-the-fly — only the compressed codes persist in GPU memory, while the
decompressed tensors exist only for the duration of a single attention computation.
