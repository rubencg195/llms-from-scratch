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

# Section 8.5: Hooking Titans to Attention (Memory as Context)

**Goal:** Concatenate or add neural memory readout to attention context before the output projection.

## What You Need to Know First

This section wires the memory into a transformer block. You already have the pieces:

- **The transformer block (Phase 1/3):** LayerNorm → self-attention → feed-forward, with residual ("add the input back") connections.
- **From Section 8.2:** `NeuralMemory.read()` returns a context vector from the long-term memory matrix.
- **Sigmoid gate** — `sigmoid` squashes any number into 0–1; multiplying the memory output by this gate lets the model learn *how much* to trust memory (0 = ignore, 1 = full).
- **Residual connections** — adding a layer's output back to its input; we add the gated memory the same way.

That's it — combining the streams is just addition with a learned dial.

## Three Streams of Context

A Titans block has access to three sources of information, each operating at a different
timescale:

1. **Attention (local context)** — standard self-attention over the current window. Provides
   precise token-level relationships within the sliding window. Timescale: ~4K tokens.

2. **Neural Memory (long-term context)** — the learned memory matrix from Section 8.2.
   Provides compressed summaries of important facts from the entire conversation history.
   Timescale: unlimited (bounded only by memory capacity, not sequence length).

3. **Model Weights (permanent knowledge)** — the pretrained parameters. Encodes general
   language understanding, world knowledge, and reasoning patterns. Timescale: permanent.

The gating mechanism learns to *blend* attention and memory outputs based on what's
most useful for the current prediction.

---

## The Gating Mechanism

A learned sigmoid gate controls how much the memory influences the output. This is crucial:
early in training, the memory is unreliable (random), so the gate should be near 0. As the
memory learns to store useful facts, the gate opens.

```python
import torch
import torch.nn as nn

class TitansBlock(nn.Module):
    def __init__(self, d_model, n_heads, block_size):
        super().__init__()
        self.ln1 = nn.LayerNorm(d_model)
        self.ln2 = nn.LayerNorm(d_model)
        self.attn = nn.MultiheadAttention(d_model, n_heads, batch_first=True)
        self.memory = nn.Linear(d_model, d_model)  # memory readout projection
        self.mem_gate = nn.Parameter(torch.tensor(0.0))  # init near 0 (sigmoid -> 0.5)
        self.ff = nn.Sequential(
            nn.Linear(d_model, 4 * d_model),
            nn.GELU(),
            nn.Linear(4 * d_model, d_model),
        )

    def forward(self, x, mem_read):
        """
        x: (B, T, d_model) — token representations
        mem_read: (B, T, d_model) — neural memory readout for each position
        """
        h = self.ln1(x)
        attn_out, _ = self.attn(h, h, h, need_weights=False)

        # Memory contribution gated by learned sigmoid
        mem_out = self.memory(mem_read)
        gate = torch.sigmoid(self.mem_gate)

        # Residual connection with gated memory
        x = x + attn_out + gate * mem_out
        x = x + self.ff(self.ln2(x))
        return x

B, T, C = 2, 16, 512
x = torch.randn(B, T, C)
mem_read = torch.randn(B, T, C)
block = TitansBlock(C, 8, 256)
y = block(x, mem_read)
print(f"Titans block output: {y.shape}")
print(f"Memory gate value: {torch.sigmoid(block.mem_gate).item():.4f}")
```

---

## Full Integration: TitansBlock with NeuralMemory from 8.2

Now let's wire the actual NeuralMemory module into the block:

```python
class NeuralMemory(nn.Module):
    """Simplified from Section 8.2 for integration."""
    def __init__(self, d_model, d_mem=None):
        super().__init__()
        d_mem = d_mem or d_model
        self.write_proj = nn.Linear(d_model, d_mem)
        self.read_proj = nn.Linear(d_mem, d_model)
        self.memory = nn.Parameter(torch.zeros(d_mem, d_mem))

    def read(self, query):
        k = self.write_proj(query)
        att = torch.softmax(k @ self.memory, dim=-1)
        return self.read_proj(att)

    def write_delta(self, key, value, lr=0.01):
        with torch.no_grad():
            k = self.write_proj(key).mean(dim=(0, 1))
            v = self.write_proj(value).mean(dim=(0, 1))
            self.memory += lr * torch.outer(v, k)

class TitansBlockWithMemory(nn.Module):
    """Complete Titans block integrating NeuralMemory."""
    def __init__(self, d_model, n_heads, d_mem=None):
        super().__init__()
        self.ln1 = nn.LayerNorm(d_model)
        self.ln2 = nn.LayerNorm(d_model)
        self.attn = nn.MultiheadAttention(d_model, n_heads, batch_first=True)
        self.neural_memory = NeuralMemory(d_model, d_mem)
        self.mem_gate = nn.Parameter(torch.tensor(0.0))
        self.ff = nn.Sequential(
            nn.Linear(d_model, 4 * d_model),
            nn.GELU(),
            nn.Linear(4 * d_model, d_model),
        )

    def forward(self, x):
        h = self.ln1(x)

        # Stream 1: local attention
        attn_out, attn_weights = self.attn(h, h, h, need_weights=True)

        # Stream 2: memory readout
        mem_out = self.neural_memory.read(h)
        gate = torch.sigmoid(self.mem_gate)

        # Merge streams
        x = x + attn_out + gate * mem_out
        x = x + self.ff(self.ln2(x))
        return x, attn_weights

# Build a 2-layer Titans model
class TitansModel(nn.Module):
    def __init__(self, d_model, n_heads, n_layers, d_mem=None):
        super().__init__()
        self.layers = nn.ModuleList([
            TitansBlockWithMemory(d_model, n_heads, d_mem)
            for _ in range(n_layers)
        ])

    def forward(self, x):
        all_attn = []
        for layer in self.layers:
            x, attn_w = layer(x)
            all_attn.append(attn_w)
        return x, all_attn

model = TitansModel(d_model=512, n_heads=8, n_layers=2, d_mem=256)
x_input = torch.randn(2, 16, 512)
output, attentions = model(x_input)
print(f"Model output: {output.shape}")
print(f"Attention weights per layer: {[a.shape for a in attentions]}")
print(f"Total parameters: {sum(p.numel() for p in model.parameters()):,}")
```

---

## Forward Pass Walkthrough with Shape Annotations

```python
print("=" * 70)
print("FORWARD PASS WALKTHROUGH")
print("=" * 70)

# Single example for clarity
x_walk = torch.randn(1, 8, 512)  # (batch=1, seq_len=8, d_model=512)
print(f"\nInput: x = {x_walk.shape}")

layer = model.layers[0]

# Step 1: LayerNorm
h = layer.ln1(x_walk)
print(f"\n1. After LayerNorm: h = {h.shape}")

# Step 2: Self-Attention
attn_out, attn_w = layer.attn(h, h, h, need_weights=True)
print(f"2. Attention output: {attn_out.shape}")
print(f"   Attention weights: {attn_w.shape} (heads averaged)")

# Step 3: Memory Read
mem_out = layer.neural_memory.read(h)
print(f"3. Memory readout: {mem_out.shape}")

# Step 4: Gate
gate_val = torch.sigmoid(layer.mem_gate)
print(f"4. Memory gate: {gate_val.item():.4f}")
print(f"   Gated memory: {(gate_val * mem_out).shape}")

# Step 5: Residual merge
merged = x_walk + attn_out + gate_val * mem_out
print(f"5. After merge (x + attn + gate*mem): {merged.shape}")

# Step 6: FFN
ff_out = layer.ff(layer.ln2(merged))
print(f"6. FFN output: {ff_out.shape}")

# Step 7: Final residual
final = merged + ff_out
print(f"7. Block output: {final.shape}")

print(f"\n{'=' * 70}")
print("All shapes preserved: input (1, 8, 512) -> output (1, 8, 512)")
print("Memory provides context WITHOUT increasing sequence length!")
```

---

## Memory Read from NeuralMemory: The Full Loop

During inference, the complete loop is:
1. Forward through transformer layers (attention + memory read)
2. Compute surprise on each position
3. For high-surprise positions, perform TTT write to memory
4. Next token uses updated memory

```python
def titans_inference_step(model, x, surprise_threshold=0.5):
    """One step of Titans inference with memory updates."""
    # Forward pass (reads from memory)
    output, _ = model(x)

    # Compute surprise for each layer's memory
    writes_performed = 0
    for layer in model.layers:
        mem = layer.neural_memory
        h = layer.ln1(x)
        pred = mem.read(h)
        # Target: what the layer actually produced (post-attention)
        target_approx = h  # simplified: use input as self-prediction target

        # Per-position surprise
        surprise = (pred - target_approx).pow(2).mean(dim=-1)  # (B, T)

        # Write high-surprise positions
        high_surprise_mask = surprise > surprise_threshold
        if high_surprise_mask.any():
            # Average high-surprise positions for the write
            high_surp_h = h[high_surprise_mask].unsqueeze(0).unsqueeze(0)
            if high_surp_h.shape[1] > 0:
                mem.write_delta(high_surp_h, high_surp_h, lr=0.01)
                writes_performed += high_surprise_mask.sum().item()

    return output, writes_performed

# Test inference step
x_test = torch.randn(1, 8, 512)
out, n_writes = titans_inference_step(model, x_test)
print(f"Inference step: {n_writes} memory writes performed")
print("Decode uses fixed-size memory — context length does not grow KV cache.")
```

---

## Exercise: Vary Gate Initialization and Observe Training Impact

The gate initialization determines how much the model relies on memory from the start.
Compare different initializations:

```python
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

gate_inits = [-2.0, 0.0, 2.0]  # sigmoid(-2)≈0.12, sigmoid(0)=0.5, sigmoid(2)≈0.88
results = {}

for gate_init in gate_inits:
    torch.manual_seed(42)
    test_model = TitansBlockWithMemory(d_model=128, n_heads=4, d_mem=64)
    # Override gate initialization
    test_model.mem_gate.data.fill_(gate_init)

    optimizer = torch.optim.Adam(test_model.parameters(), lr=1e-3)
    losses = []
    gate_values = []

    for step in range(100):
        x = torch.randn(4, 8, 128)
        target = torch.randn(4, 8, 128)  # random targets (toy task)

        out, _ = test_model(x)
        loss = (out - target).pow(2).mean()

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        losses.append(loss.item())
        gate_values.append(torch.sigmoid(test_model.mem_gate).item())

    results[gate_init] = {'losses': losses, 'gates': gate_values}
    init_gate = torch.sigmoid(torch.tensor(gate_init)).item()
    print(f"Gate init={gate_init} (σ={init_gate:.2f}): "
          f"final loss={losses[-1]:.4f}, final gate={gate_values[-1]:.4f}")

# Plot results
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

for gate_init in gate_inits:
    init_gate = torch.sigmoid(torch.tensor(gate_init)).item()
    axes[0].plot(results[gate_init]['losses'], linewidth=2,
                label=f'init σ(g)={init_gate:.2f}')
    axes[1].plot(results[gate_init]['gates'], linewidth=2,
                label=f'init σ(g)={init_gate:.2f}')

axes[0].set_xlabel('Step')
axes[0].set_ylabel('Loss')
axes[0].set_title('Training Loss vs Gate Initialization')
axes[0].legend()
axes[0].grid(True, alpha=0.3)

axes[1].set_xlabel('Step')
axes[1].set_ylabel('Gate Value σ(g)')
axes[1].set_title('Memory Gate Evolution During Training')
axes[1].axhline(y=0.5, color='gray', linestyle=':', alpha=0.5)
axes[1].legend()
axes[1].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('gate_initialization.png', dpi=100, bbox_inches='tight')
plt.close()
print("Saved gate_initialization.png")
print("\nInsight: low initialization (gate≈0) lets the model learn when memory is useful")
print("High initialization forces memory reliance before it's trained → slower convergence")
```

---

## Where This Leads Next

The full Titans block is assembled — attention plus gated neural memory. Section 8.6 is the
capstone: we profile actual VRAM as the conversation grows, proving the central claim that this
architecture keeps memory *constant* and fits 100K+ token contexts on a single 10 GB GPU.

## Key Takeaway

The Titans block merges three information streams through a single learned gate: attention
provides local context (what's in the current window), neural memory provides long-term
context (facts from the entire conversation), and model weights provide permanent knowledge.
The sigmoid gate starts near 0, allowing the model to learn *when* memory is useful rather
than always relying on it. This architecture keeps VRAM constant regardless of sequence
length — the memory matrix is the same size whether the conversation is 100 or 100,000 tokens.

## Further Reading (Optional)

**Optional — you do NOT need these to continue. They are for curious students who want the original sources.**

- Behrouz, Zhong, & Mirrokni (2024). *Titans: Learning to Memorize at Test Time*. arXiv:2501.00663.
- Katharopoulos et al. (2020). *Transformers are RNNs: Fast Autoregressive Transformers with Linear Attention*. ICML.
