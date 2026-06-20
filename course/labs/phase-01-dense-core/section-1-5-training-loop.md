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

# Section 1.5: The Autoregressive Training Loop on the RTX 3080

**Goal:** Assemble GPT-style model (~80M params), train on TinyStories, and save `phase1_80m.pt`.

## What You Need to Know First

- **The autograd training loop** (Section 0.3) — forward → loss → `backward()` → `optimizer.step()`, used here unchanged at scale.
- **Tokenization and embeddings** (Sections 1.1–1.2) — to turn TinyStories text into token IDs and vectors.
- **RoPE and attention** (Sections 1.3–1.4) — the Transformer blocks being trained here are exactly the ones you built.
- **Softmax** (Section 0.2) — used both inside attention and to turn final scores into next-token probabilities.

Every ingredient was assembled in earlier sections, so nothing new from outside the course is required. New terms like *autoregressive*, *cross-entropy*, and *gradient clipping* are explained inline. (**Autoregressive** just means the model generates one token at a time, each new token depending on the ones already produced.)

## What Is Autoregressive Language Modeling?

The training objective is simple: **predict the next token**. Given a sequence of tokens $[t_1, t_2, \ldots, t_n]$, the model learns to predict $t_{i+1}$ from $[t_1, \ldots, t_i]$ for every position simultaneously.

```
Input:    [The] [cat] [sat] [on] [the]
Target:   [cat] [sat] [on]  [the] [mat]
```

The loss is **cross-entropy** — a standard score that is small when the model puts high probability on the correct next token and large when it is confidently wrong — measured between the model's predicted probability distribution over the vocabulary and the actual next token. This is computed at every position in parallel (thanks to the causal mask hiding future tokens).

At generation time, we sample from the predicted distribution to produce one token at a time — hence "autoregressive."

## Model config (VRAM-safe)

```python
CONFIG = dict(
    vocab_size=8000,
    d_model=512,
    n_layers=8,
    n_heads=8,
    block_size=256,
    dropout=0.1,
)

def count_params(model):
    return sum(p.numel() for p in model.parameters())

print("Target: ~80M parameters")
print(f"Config: {CONFIG}")
```

## Minimal GPT wrapper

```python
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
import matplotlib.pyplot as plt
import time

def precompute_freqs(dim, seq_len, base=10000.0, device="cpu"):
    inv_freq = 1.0 / (base ** (torch.arange(0, dim, 2, device=device).float() / dim))
    t = torch.arange(seq_len, device=device).float()
    freqs = torch.outer(t, inv_freq)
    return freqs.cos(), freqs.sin()

def apply_rope(x, cos, sin):
    x1, x2 = x[..., 0::2], x[..., 1::2]
    cos = cos.unsqueeze(0).unsqueeze(0)
    sin = sin.unsqueeze(0).unsqueeze(0)
    rot1 = x1 * cos - x2 * sin
    rot2 = x1 * sin + x2 * cos
    return torch.stack((rot1, rot2), dim=-1).flatten(-2)

class CausalSelfAttention(nn.Module):
    def __init__(self, d_model, n_heads, block_size, dropout=0.1):
        super().__init__()
        assert d_model % n_heads == 0
        self.n_heads = n_heads
        self.head_dim = d_model // n_heads
        self.qkv = nn.Linear(d_model, 3 * d_model)
        self.proj = nn.Linear(d_model, d_model)
        self.dropout = nn.Dropout(dropout)
        self.register_buffer(
            "mask",
            torch.tril(torch.ones(block_size, block_size)).view(1, 1, block_size, block_size),
        )

    def forward(self, x, cos, sin):
        B, T, C = x.shape
        qkv = self.qkv(x).reshape(B, T, 3, self.n_heads, self.head_dim).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]
        q = apply_rope(q, cos, sin)
        k = apply_rope(k, cos, sin)
        att = (q @ k.transpose(-2, -1)) / math.sqrt(self.head_dim)
        att = att.masked_fill(self.mask[:, :, :T, :T] == 0, float("-inf"))
        att = F.softmax(att, dim=-1)
        att = self.dropout(att)
        out = (att @ v).transpose(1, 2).contiguous().view(B, T, C)
        return self.proj(out)

class Block(nn.Module):
    def __init__(self, d_model, n_heads, block_size, dropout=0.1):
        super().__init__()
        self.ln1 = nn.LayerNorm(d_model)
        self.attn = CausalSelfAttention(d_model, n_heads, block_size, dropout)
        self.ln2 = nn.LayerNorm(d_model)
        self.mlp = nn.Sequential(
            nn.Linear(d_model, 4 * d_model),
            nn.GELU(),
            nn.Linear(4 * d_model, d_model),
            nn.Dropout(dropout),
        )

    def forward(self, x, cos, sin):
        x = x + self.attn(self.ln1(x), cos, sin)
        x = x + self.mlp(self.ln2(x))
        return x

class GPT(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        self.tok_emb = nn.Embedding(cfg["vocab_size"], cfg["d_model"])
        self.blocks = nn.ModuleList([
            Block(cfg["d_model"], cfg["n_heads"], cfg["block_size"], cfg["dropout"])
            for _ in range(cfg["n_layers"])
        ])
        self.ln_f = nn.LayerNorm(cfg["d_model"])
        self.lm_head = nn.Linear(cfg["d_model"], cfg["vocab_size"], bias=False)
        # Weight tying (Section 1.2)
        self.lm_head.weight = self.tok_emb.weight

    def forward(self, idx, cos, sin):
        x = self.tok_emb(idx)
        for block in self.blocks:
            x = block(x, cos, sin)
        x = self.ln_f(x)
        return self.lm_head(x)
```

## Loading TinyStories Data

For a real training run, we load actual text data from HuggingFace's `datasets` library. Here we tokenize with a simple whitespace scheme (swap for BPE in production).

```python
from datasets import load_dataset
from collections import Counter

# Load a small slice for this lab (increase for full training)
print("Loading TinyStories...")
ds = load_dataset("roneneldan/TinyStories", split="train[:2%]")
print(f"Loaded {len(ds)} stories")

# Build vocabulary from corpus
def build_vocab(dataset, max_docs=2000, vocab_size=7999):
    counter = Counter()
    for i, row in enumerate(dataset):
        if i >= max_docs:
            break
        for word in row["text"].lower().split():
            counter[word] += 1
    words = [w for w, _ in counter.most_common(vocab_size)]
    vocab = {w: i + 1 for i, w in enumerate(words)}
    vocab["<|unk|>"] = 0
    return vocab

vocab = build_vocab(ds, vocab_size=CONFIG["vocab_size"] - 1)
print(f"Vocabulary: {len(vocab)} tokens")
```

```python
class TinyStoriesDataset(Dataset):
    def __init__(self, dataset, vocab, seq_len, max_stories=2000):
        self.seq_len = seq_len
        # Tokenize all stories into one long stream
        all_tokens = []
        for i, row in enumerate(dataset):
            if i >= max_stories:
                break
            tokens = [vocab.get(w, 0) for w in row["text"].lower().split()]
            all_tokens.extend(tokens)

        self.data = torch.tensor(all_tokens, dtype=torch.long)
        self.n_samples = (len(self.data) - 1) // seq_len
        print(f"Total tokens: {len(self.data):,}, Samples: {self.n_samples}")

    def __len__(self):
        return self.n_samples

    def __getitem__(self, i):
        start = i * self.seq_len
        chunk = self.data[start : start + self.seq_len + 1]
        return chunk[:-1], chunk[1:]  # inputs, targets (next token)

device = "cuda" if torch.cuda.is_available() else "cpu"
cfg = CONFIG
model = GPT(cfg).to(device)
print(f"Parameters: {count_params(model)/1e6:.1f}M")

cos, sin = precompute_freqs(cfg["d_model"] // cfg["n_heads"], cfg["block_size"], device=device)
train_ds = TinyStoriesDataset(ds, vocab, cfg["block_size"], max_stories=2000)
loader = DataLoader(train_ds, batch_size=8, shuffle=True, drop_last=True)
```

## Learning Rate Scheduling: Cosine Annealing

A constant learning rate is suboptimal. **Cosine annealing** starts with a warmup phase (linearly increasing lr) and then smoothly decays the learning rate following a cosine curve. This is the standard schedule for LLM training.

```python
max_lr = 3e-4
min_lr = 3e-5
warmup_steps = 50
max_steps = len(loader)  # one epoch

def get_lr(step):
    """Cosine annealing with linear warmup."""
    if step < warmup_steps:
        return max_lr * (step + 1) / warmup_steps
    progress = (step - warmup_steps) / max(1, max_steps - warmup_steps)
    return min_lr + 0.5 * (max_lr - min_lr) * (1 + math.cos(math.pi * progress))

# Visualize the schedule
lr_schedule = [get_lr(s) for s in range(max_steps)]
plt.figure(figsize=(8, 3))
plt.plot(lr_schedule, linewidth=2)
plt.xlabel("Step")
plt.ylabel("Learning Rate")
plt.title("Cosine Annealing with Linear Warmup")
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig("lr_schedule.png", dpi=120)
plt.show()
print("Saved lr_schedule.png")
```

## Training loop with gradient clipping

```python
opt = torch.optim.AdamW(model.parameters(), lr=max_lr, weight_decay=0.01)
max_grad_norm = 1.0  # gradient clipping threshold

model.train()
losses = []
step_times = []
start_time = time.time()

for step, (x, y) in enumerate(loader):
    if step >= 200:  # increase for full training run
        break

    # Update learning rate
    lr = get_lr(step)
    for param_group in opt.param_groups:
        param_group["lr"] = lr

    x, y = x.to(device), y.to(device)
    logits = model(x, cos, sin)
    loss = F.cross_entropy(logits.view(-1, cfg["vocab_size"]), y.view(-1))

    opt.zero_grad()
    loss.backward()

    # Gradient clipping: prevent exploding gradients
    grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)

    opt.step()
    losses.append(loss.item())

    if step % 20 == 0:
        elapsed = time.time() - start_time
        tokens_per_sec = (step + 1) * 8 * cfg["block_size"] / elapsed if elapsed > 0 else 0
        print(f"step {step:4d} | loss {loss.item():.3f} | lr {lr:.2e} | "
              f"grad_norm {grad_norm:.2f} | tok/s {tokens_per_sec:.0f}")
        if device == "cuda":
            print(f"         VRAM MB: {torch.cuda.max_memory_allocated() / 1e6:.0f}")

total_time = time.time() - start_time
print(f"\nTraining complete: {len(losses)} steps in {total_time:.1f}s")
```

## Training Loss Plot

```python
plt.figure(figsize=(10, 4))
plt.plot(losses, alpha=0.3, color="steelblue", label="Raw loss")
# Smoothed loss (moving average)
window = 10
if len(losses) > window:
    smoothed = [sum(losses[max(0,i-window):i+1])/min(i+1, window) for i in range(len(losses))]
    plt.plot(smoothed, linewidth=2, color="darkblue", label=f"Smoothed (window={window})")
plt.xlabel("Step")
plt.ylabel("Cross-Entropy Loss")
plt.title("Training Loss — GPT on TinyStories")
plt.legend()
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig("training_loss.png", dpi=120)
plt.show()
print("Saved training_loss.png")
print(f"Initial loss: {losses[0]:.3f} (random = -ln(1/{cfg['vocab_size']}) = {math.log(cfg['vocab_size']):.3f})")
print(f"Final loss: {losses[-1]:.3f}")
```

## Text Generation: Sampling from the Model

The whole point of training is to generate text. Here's how autoregressive generation works:
1. Start with a prompt (sequence of token IDs)
2. Forward pass → get logits for next token
3. Sample from the distribution (with temperature)
4. Append the new token and repeat

```python
@torch.no_grad()
def generate(model, prompt_ids, max_new_tokens=50, temperature=0.8, top_k=50):
    """Autoregressive text generation with top-k sampling."""
    model.eval()
    ids = prompt_ids.clone()

    for _ in range(max_new_tokens):
        # Crop to block_size if needed
        context = ids[:, -cfg["block_size"]:]
        T_ctx = context.shape[1]

        logits = model(context, cos[:T_ctx], sin[:T_ctx])
        logits = logits[:, -1, :] / temperature  # last position only

        # Top-k filtering
        if top_k > 0:
            topk_vals, _ = logits.topk(top_k)
            logits[logits < topk_vals[:, -1:]] = float("-inf")

        probs = F.softmax(logits, dim=-1)
        next_id = torch.multinomial(probs, num_samples=1)
        ids = torch.cat([ids, next_id], dim=1)

    model.train()
    return ids

# Build inverse vocab for decoding
itos = {i: w for w, i in vocab.items()}

def decode_ids(ids):
    return " ".join(itos.get(i, "<unk>") for i in ids.tolist())

# Generate from a prompt
prompt_text = "once upon a"
prompt_ids = torch.tensor([[vocab.get(w, 0) for w in prompt_text.split()]], device=device)
generated = generate(model, prompt_ids, max_new_tokens=30, temperature=0.9)

print(f"Prompt: '{prompt_text}'")
print(f"Generated: '{decode_ids(generated[0])}'")
print("\n(Note: with only 200 steps of training, output will be mostly nonsensical)")
print("Full training (10k+ steps) produces coherent TinyStories-style text.")
```

## VRAM Profiling

Understanding memory usage is critical for fitting models on consumer GPUs. Here's a breakdown of where VRAM goes.

```python
if device == "cuda":
    print("=== VRAM Profiling ===")
    print(f"Peak VRAM allocated: {torch.cuda.max_memory_allocated() / 1e6:.0f} MB")
    print(f"Current VRAM allocated: {torch.cuda.memory_allocated() / 1e6:.0f} MB")
    print(f"VRAM reserved (cached): {torch.cuda.memory_reserved() / 1e6:.0f} MB")

    # Estimate memory breakdown
    param_memory = sum(p.numel() * p.element_size() for p in model.parameters()) / 1e6
    grad_memory = param_memory  # gradients same size as params
    optimizer_memory = 2 * param_memory  # Adam stores m and v

    print(f"\nEstimated breakdown:")
    print(f"  Model parameters:  {param_memory:.0f} MB")
    print(f"  Gradients:         {grad_memory:.0f} MB")
    print(f"  Optimizer states:  {optimizer_memory:.0f} MB (Adam m + v)")
    print(f"  Activations:       ~{torch.cuda.max_memory_allocated()/1e6 - param_memory - grad_memory - optimizer_memory:.0f} MB")
    print(f"  Total estimated:   {param_memory + grad_memory + optimizer_memory:.0f} MB (+ activations)")
    print(f"\n  RTX 3080 has 10 GB — {'SAFE' if torch.cuda.max_memory_allocated()/1e9 < 9 else 'TIGHT'}!")

    torch.cuda.reset_peak_memory_stats()
else:
    print("Running on CPU — no VRAM profiling available")
    print("Tip: Use 'nvidia-smi' in terminal to monitor GPU usage during training")
```

## Save checkpoint

```python
import os
os.makedirs("checkpoints", exist_ok=True)
checkpoint = {
    "config": cfg,
    "model": model.state_dict(),
    "optimizer": opt.state_dict(),
    "step": len(losses),
    "loss": losses[-1] if losses else None,
    "vocab": vocab,
}
torch.save(checkpoint, "checkpoints/phase1_80m.pt")
print("Saved checkpoints/phase1_80m.pt")
print(f"Checkpoint contains: model weights, optimizer state, vocab, and config")
```

## Where This Leads Next

You've trained a complete GPT end-to-end — that closes the loop from "what is a tensor?" all the way to generating text. Phase 2 picks up from this exact checkpoint to **scale things up**: more data, longer training, proper evaluation metrics, and efficiency tricks like mixed-precision and multi-GPU training, so your small but real model grows into a genuinely capable one.

## Key Takeaway

- **Autoregressive LM training** predicts the next token at every position simultaneously — the causal mask ensures no future information leaks.
- **Cosine annealing** with linear warmup is the standard LR schedule for LLM training — it balances fast early learning with stable late convergence.
- **Gradient clipping** prevents training instabilities from exploding gradients — essential for deep transformers.
- **Text generation** is just repeated forward passes + sampling — temperature controls creativity vs. coherence.
- **VRAM budget:** parameters + gradients + optimizer states + activations must all fit in GPU memory. Our 80M model fits comfortably on an RTX 3080 at batch size 8.

## Phase 1 Checkpoint Summary

You've built a complete GPT from scratch:

| Section | Component | What You Built |
|---------|-----------|----------------|
| 1.1 | Tokenization | Text ↔ integer IDs |
| 1.2 | Embeddings | ID → dense vector lookup |
| 1.3 | RoPE | Position encoding via rotation |
| 1.4 | Attention | Multi-head causal self-attention |
| 1.5 | Training | Full autoregressive training loop |

**Next steps (Phase 2):** Scale up — larger data, longer training, evaluation metrics, and techniques like mixed-precision training and distributed data parallel.

**Instructor note:** For a full training run, increase `max_stories` to the full dataset and train for 10,000+ steps. Expect ~2 hours on an RTX 3080 for noticeable grammar and storytelling ability. VRAM should stay under ~8 GB at batch 8.

## Further Reading (Optional)

**Optional — you do NOT need these to continue. They are for curious students who want the original sources.**

- Radford et al. (2019). *Language Models are Unsupervised Multitask Learners (GPT-2)*. OpenAI.
- Eldan & Li (2023). *TinyStories*. arXiv:2305.07759.
- Loshchilov & Hutter (2017). *SGDR: Stochastic Gradient Descent with Warm Restarts (cosine schedule)*. ICLR.
