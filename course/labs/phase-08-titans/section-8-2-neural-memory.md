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

# Section 8.2: Coding the Core Neural Memory Module

**Goal:** Implement `NeuralMemory` with read/write paths and separate fast weights updated at test time.

## What You Need to Know First

This section codes the memory module. Everything it leans on is already in your toolkit:

- **From Section 8.1:** the memory is a small set of weights we can update *during* inference (Test-Time Training).
- **`nn.Linear` and `nn.Parameter`** — `nn.Parameter` is just a tensor the module treats as a learnable weight; here the memory matrix is one big parameter.
- **Matrix multiplication and softmax** — reading the memory is "multiply the query by the matrix, softmax the result" — the same attention-style math from Phase 3.
- **Outer product** — multiplying a column vector by a row vector to get a matrix; we use it to "write" a key→value association into the memory.

Only high-school-level linear algebra (multiply, average) is needed.

## A Tiny Network Inside the Network

The neural memory module is a small MLP whose weights serve as associative storage. Unlike
a KV cache (which stores explicit key-value pairs), the memory stores information
*implicitly in its weights* — the same way your brain stores facts in synaptic connections
rather than in individual neurons.

The architecture has two paths:
1. **Read path** — query the memory: project a query, multiply by the memory matrix, read out a value
2. **Write path** — update the memory: compute an outer-product update gated by surprise

Think of the memory matrix as a lookup table encoded in a weight matrix: writing stores
a key-value association, reading retrieves the value given a key.

---

## The Read Path

The read path converts a query into a memory readout:
1. Project the query through `write_proj` to get a key vector
2. Multiply the key by the memory matrix (acts like attention over stored patterns)
3. Apply softmax to get attention-like weights
4. Project back through `read_proj` to get the output

```python
import torch
import torch.nn as nn

class NeuralMemory(nn.Module):
    def __init__(self, d_model, d_mem=None):
        super().__init__()
        d_mem = d_mem or d_model
        self.d_mem = d_mem
        self.write_proj = nn.Linear(d_model, d_mem)
        self.read_proj = nn.Linear(d_mem, d_model)
        # The memory matrix: this is what gets updated at test time
        self.memory = nn.Parameter(torch.zeros(d_mem, d_mem) * 0.01)

    def read(self, query):
        """
        Read from memory.
        query: (B, T, d_model) -> output: (B, T, d_model)
        """
        k = self.write_proj(query)  # (B, T, d_mem)
        # Attend over memory matrix rows
        att = torch.softmax(k @ self.memory, dim=-1)  # (B, T, d_mem)
        return self.read_proj(att)  # (B, T, d_model)

    def write_delta(self, key, value, lr=0.01):
        """
        Outer-product style update — test-time only.
        Writes the association (key -> value) into memory.
        """
        with torch.no_grad():
            k = self.write_proj(key).mean(dim=(0, 1))   # (d_mem,)
            v = self.write_proj(value).mean(dim=(0, 1))  # (d_mem,)
            self.memory += lr * torch.outer(v, k)

d_model = 512
mem = NeuralMemory(d_model)
q = torch.randn(2, 8, d_model)
print("Read path output shape:", mem.read(q).shape)
print("Memory matrix shape:", mem.memory.shape)
print("Memory norm (before writes):", mem.memory.norm().item())
```

---

## The Write Path (Surprise-Gated Outer-Product Update)

Writing to memory uses the outer product of the value and key vectors. This creates a
rank-1 update to the memory matrix that encodes the association "when you see this key,
recall this value."

```python
# Write a single fact
key_input = torch.randn(1, 1, d_model)   # the "query" that should trigger recall
value_input = torch.randn(1, 1, d_model)  # the "answer" to store

print("Before write:")
readout_before = mem.read(key_input)
print(f"  Memory norm: {mem.memory.norm().item():.4f}")

mem.write_delta(key_input, value_input, lr=0.05)

print("\nAfter write:")
readout_after = mem.read(key_input)
print(f"  Memory norm: {mem.memory.norm().item():.4f}")

# Did the write make the readout closer to the value?
target_proj = mem.write_proj(value_input).mean(dim=(0, 1))
read_proj_before = mem.write_proj(readout_before).mean(dim=(0, 1))
read_proj_after = mem.write_proj(readout_after).mean(dim=(0, 1))

cos_sim_before = torch.cosine_similarity(read_proj_before.unsqueeze(0), target_proj.unsqueeze(0))
cos_sim_after = torch.cosine_similarity(read_proj_after.unsqueeze(0), target_proj.unsqueeze(0))
print(f"  Cosine sim to value (before write): {cos_sim_before.item():.4f}")
print(f"  Cosine sim to value (after write):  {cos_sim_after.item():.4f}")
```

---

## Memory Capacity: How Many Facts Can It Store?

The memory matrix is $d_{mem} \times d_{mem}$. Each write is a rank-1 update. In theory,
a rank-$r$ matrix can store at most $r$ independent facts. Since $d_{mem}$ rank-1 updates
can fill the matrix, capacity ≈ $d_{mem}$ facts before interference.

```python
def test_memory_capacity(d_model, d_mem, n_facts, write_lr=0.05):
    """Write n_facts and measure how many can be accurately recalled."""
    mem_test = NeuralMemory(d_model, d_mem)
    torch.manual_seed(42)

    # Generate random key-value pairs as "facts"
    keys = [torch.randn(1, 1, d_model) for _ in range(n_facts)]
    values = [torch.randn(1, 1, d_model) for _ in range(n_facts)]

    # Write all facts
    for k, v in zip(keys, values):
        mem_test.write_delta(k, v, lr=write_lr)

    # Read back and measure accuracy
    correct = 0
    similarities = []
    for k, v in zip(keys, values):
        readout = mem_test.read(k)
        sim = torch.cosine_similarity(
            readout.view(1, -1), v.view(1, -1)).item()
        similarities.append(sim)
        if sim > 0.3:  # threshold for "recalled"
            correct += 1

    return correct, n_facts, similarities

print("Memory Capacity Test (d_mem=512):")
print(f"{'N Facts':<10} {'Recalled':<10} {'Accuracy':<10} {'Avg Similarity'}")
print("-" * 45)
for n in [5, 10, 25, 50, 100]:
    correct, total, sims = test_memory_capacity(512, 512, n)
    import numpy as np
    print(f"{n:<10} {correct:<10} {correct/total*100:.0f}%{'':<6} {np.mean(sims):.4f}")
```

---

## Demonstration: Write 5 Facts, Read Them Back

```python
torch.manual_seed(123)
demo_mem = NeuralMemory(d_model=128, d_mem=128)

# Simulate 5 "facts" as distinct key-value pairs
fact_names = ["Alice's code", "Bob's color", "Meeting room", "WiFi password", "Project name"]
keys = [torch.randn(1, 1, 128) for _ in range(5)]
values = [torch.randn(1, 1, 128) for _ in range(5)]

# Write each fact
for i, (k, v) in enumerate(zip(keys, values)):
    demo_mem.write_delta(k, v, lr=0.1)

# Read back each fact and check accuracy
print("Fact Recall Test:")
print(f"{'Fact':<20} {'Cosine Similarity':<20} {'Recalled?'}")
print("-" * 50)
for i, (k, v, name) in enumerate(zip(keys, values, fact_names)):
    readout = demo_mem.read(k)
    sim = torch.cosine_similarity(readout.view(1, -1), v.view(1, -1)).item()
    recalled = "YES" if sim > 0.2 else "NO"
    print(f"{name:<20} {sim:<20.4f} {recalled}")
```

---

## Visualization: Memory Matrix Heatmap Before and After Writes

```python
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

torch.manual_seed(0)
viz_mem = NeuralMemory(d_model=64, d_mem=64)

# Snapshot before writes
mem_before = viz_mem.memory.detach().clone().numpy()

# Write 10 facts
for _ in range(10):
    k = torch.randn(1, 1, 64)
    v = torch.randn(1, 1, 64)
    viz_mem.write_delta(k, v, lr=0.1)

mem_after = viz_mem.memory.detach().numpy()

fig, axes = plt.subplots(1, 3, figsize=(15, 4))

im0 = axes[0].imshow(mem_before[:32, :32], cmap='RdBu_r', vmin=-0.1, vmax=0.1)
axes[0].set_title('Memory Matrix (Before Writes)')
axes[0].set_xlabel('Column')
axes[0].set_ylabel('Row')
plt.colorbar(im0, ax=axes[0], fraction=0.046)

im1 = axes[1].imshow(mem_after[:32, :32], cmap='RdBu_r', vmin=-0.5, vmax=0.5)
axes[1].set_title('Memory Matrix (After 10 Writes)')
axes[1].set_xlabel('Column')
axes[1].set_ylabel('Row')
plt.colorbar(im1, ax=axes[1], fraction=0.046)

# Difference
diff = mem_after[:32, :32] - mem_before[:32, :32]
im2 = axes[2].imshow(diff, cmap='RdBu_r', vmin=-0.5, vmax=0.5)
axes[2].set_title('Difference (What Was Written)')
axes[2].set_xlabel('Column')
axes[2].set_ylabel('Row')
plt.colorbar(im2, ax=axes[2], fraction=0.046)

plt.tight_layout()
plt.savefig('memory_heatmap.png', dpi=100, bbox_inches='tight')
plt.close()
print("Saved memory_heatmap.png")
print(f"Memory matrix rank after writes: {np.linalg.matrix_rank(mem_after)}")
```

---

## Exercise: Test Memory Capacity — Keep Adding Facts Until Recall Drops

```python
torch.manual_seed(42)
capacity_mem = NeuralMemory(d_model=128, d_mem=128)

all_keys = []
all_values = []
recall_curve = []

print("Capacity Stress Test: adding facts until recall drops below 50%")
print(f"{'N Facts':<10} {'Recall Rate':<15} {'Avg Similarity'}")
print("-" * 40)

for n in range(1, 201):
    # Add a new fact
    k = torch.randn(1, 1, 128)
    v = torch.randn(1, 1, 128)
    capacity_mem.write_delta(k, v, lr=0.05)
    all_keys.append(k)
    all_values.append(v)

    # Check recall of ALL facts periodically
    if n % 10 == 0 or n <= 5:
        correct = 0
        sims = []
        for ki, vi in zip(all_keys, all_values):
            readout = capacity_mem.read(ki)
            sim = torch.cosine_similarity(readout.view(1, -1), vi.view(1, -1)).item()
            sims.append(sim)
            if sim > 0.2:
                correct += 1
        recall_rate = correct / n
        recall_curve.append((n, recall_rate))
        avg_sim = np.mean(sims)
        print(f"{n:<10} {recall_rate*100:.1f}%{'':<10} {avg_sim:.4f}")

        if recall_rate < 0.5:
            print(f"\n  Memory saturated at ~{n} facts for d_mem=128!")
            break

print("\nThe memory has finite capacity — just like a real neural network.")
print("Decay (Section 8.4) prevents saturation by forgetting old, unreinforced facts.")
```

---

## Integration Point

Insert `NeuralMemory.read()` alongside attention context — Phase 8.5 merges the streams.
The memory provides long-term context that doesn't grow with sequence length.

---

## Where This Leads Next

Our memory writes on *every* call, which quickly saturates it. Section 8.3 adds the missing
piece of intelligence: a "surprise" gate that decides *when* a fact is worth writing — so the
model only spends a memory write on genuinely new information.

## Key Takeaway

The neural memory module is a fixed-size associative store encoded as a weight matrix.
Reading queries the matrix via key projection and softmax attention; writing updates it via
surprise-gated outer-product updates. Capacity is bounded by $d_{mem}$ (the matrix rank),
after which old facts interfere with new ones — motivating the decay mechanism in Section 8.4.
Unlike a KV cache that grows to gigabytes, this memory stays at $d_{mem}^2 \times 4$ bytes
regardless of conversation length.

## Further Reading (Optional)

**Optional — you do NOT need these to continue. They are for curious students who want the original sources.**

- Graves, Wayne, & Danihelka (2014). *Neural Turing Machines*. arXiv:1410.5401.
- Behrouz, Zhong, & Mirrokni (2024). *Titans: Learning to Memorize at Test Time*. arXiv:2501.00663.
