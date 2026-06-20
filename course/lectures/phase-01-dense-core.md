---
title: "Phase 1: The Dense Core"
subtitle: "Teaching Math to Read"
author: "LLMs From Scratch"
---

# Phase 1: The Dense Core

## Teaching Math to Read

Build a modern ~80M-parameter Transformer from scratch on TinyStories.

<!-- notes: This is the flagship phase. By the end, students will have a working language model that generates coherent children's stories. Every component is hand-coded — no HuggingFace Trainer, no shortcuts. -->

---

## Before You Begin (Prerequisites)

Everything on this list was taught in **Phase 0** — nothing new from outside is required.

- **Tensors** — multi-dimensional arrays of numbers (Phase 0).
- **The dot product** — multiply matching entries and sum; it measures similarity (Phase 0). This is the engine of attention.
- **$y = mx + b$ as a linear layer** — a weight matrix times an input plus a bias (Phase 0). Every projection in this phase is just this at scale.
- **Loss + autograd** — cross-entropy measures wrongness, `loss.backward()` distributes the blame, the optimizer nudges the weights (Phase 0).
- **Basic Python** — loops, functions, and classes.

> If Phase 0's five takeaways feel comfortable, you have everything you need. The math here is the same — only bigger.

<!-- notes: Anchor students back to Phase 0 explicitly. The dot product becomes attention, the linear layer becomes Q/K/V projections and the FFN, and the autograd training loop is reused almost verbatim. No external reading, no calculus, no prior NLP knowledge is assumed. -->

---

## Learning objectives

- **Tokenize** text into integer IDs using BPE
- Map IDs to **embeddings** (coordinates in meaning-space)
- Apply **RoPE** for relative position encoding
- Implement **multi-head self-attention** with causal masking
- Build the full **Transformer block** (Pre-LN architecture)
- Run the **autoregressive training loop** on RTX 3080

<!-- notes: Stress that we are building EVERY component. By the end of lab 1.5, students will have trained a model that can complete sentences like "Once upon a time, there was a little..." with grammatically correct, sometimes creative continuations. -->

---

## The Transformer block diagram (Pre-LN)

Our model stacks 8 identical blocks with this structure:

```
Input tokens
    ↓
[Token Embedding + Positional Encoding (RoPE)]
    ↓
┌─────────────────────────────────────────┐
│  ╔═══════════════════════════════════╗  │
│  ║  x ─────────────────────┐        ║  │
│  ║  ↓                      │        ║  │  ×8
│  ║  LayerNorm              │        ║  │  layers
│  ║  ↓                      │        ║  │
│  ║  Multi-Head Attention    │        ║  │
│  ║  ↓                      │        ║  │
│  ║  + ←────────────────────┘  (residual)║
│  ║  ↓                                ║  │
│  ║  x ─────────────────────┐        ║  │
│  ║  ↓                      │        ║  │
│  ║  LayerNorm              │        ║  │
│  ║  ↓                      │        ║  │
│  ║  Feed-Forward (FFN)     │        ║  │
│  ║  ↓                      │        ║  │
│  ║  + ←────────────────────┘  (residual)║
│  ╚═══════════════════════════════════╝  │
└─────────────────────────────────────────┘
    ↓
[Final LayerNorm]
    ↓
[LM Head (linear → vocab logits)]
    ↓
Next-token probabilities
```

**Pre-LN** = LayerNorm *before* each sub-layer (more stable training than Post-LN).

**LayerNorm** (the raw scores → stable scale step) rescales each token's vector to a steady mean and spread so training doesn't blow up. **Logits** are the raw, pre-probability scores the model produces for every possible next token (turned into probabilities by softmax).

<!-- notes: Draw this on the board if possible. The residual connections (the + arrows) are critical — they let gradients flow directly back through the network without vanishing. Pre-LN is the modern standard used by GPT-2 and later. Post-LN (original "Attention Is All You Need") puts LayerNorm after the addition, which is less stable. -->

---

## Tokenization: from text to integers

**Byte Pair Encoding (BPE)** — the standard for modern LLMs.

Algorithm:
1. Start with individual characters: `l o w e s t`
2. Count all adjacent pairs, merge the most frequent
3. Repeat until vocabulary reaches target size

**Worked example:**

```
Corpus:  "low lower lowest"

Step 0:  l o w   l o w e r   l o w e s t
Step 1:  merge (l, o) → "lo"
         lo w   lo w e r   lo w e s t
Step 2:  merge (lo, w) → "low"
         low   low e r   low e s t
Step 3:  merge (e, s) → "es"
         low   low e r   low es t
Step 4:  merge (es, t) → "est"
         low   low e r   low est
```

Result: `"lowest"` → tokens `["low", "est"]` → IDs `[142, 897]`

<!-- notes: BPE is elegant because it's data-driven. Common words like "the" become single tokens. Rare words get split into subwords. This means the model never encounters an unknown word — it can always fall back to character-level tokens. Our TinyStories vocabulary is ~8k tokens, which is tiny compared to GPT-4's ~100k, but sufficient for children's language. -->

---

## Why not one-hot encoding?

**One-hot:** each token is a sparse vector of size $|V|$

```
Token "cat" (ID 42, vocab=8000):
  [0, 0, ..., 0, 1, 0, ..., 0]    ← 8000 dimensions, only one is 1
                  ^
               position 42
```

Problems:
- **Wasteful:** 8000 numbers to say one thing
- **No similarity:** `dot("cat", "kitten")` = 0 (orthogonal!)
- **No learning:** distances are meaningless

**Dense embeddings:** each token is a learned vector in $\mathbb{R}^{d}$

```
"cat"    → [0.21, -0.87, 0.55, ..., 0.12]    (512 dims)
"kitten" → [0.19, -0.82, 0.61, ..., 0.14]    (nearby!)
"car"    → [-0.45, 0.33, 0.11, ..., -0.72]   (far away)
```

The embedding table is a learnable matrix $E \in \mathbb{R}^{|V| \times d}$. Look up row by token ID.

<!-- notes: This is one of the most important ideas in NLP. One-hot vectors live in a space where every word is equidistant from every other word. Embeddings learn a geometry where semantically similar words cluster together. The embedding matrix E is just a lookup table that gets trained by gradient descent along with everything else. -->

---

## Rotary Position Embeddings (RoPE)

**Problem:** Self-attention treats input as a *set* — it doesn't know word order.

**Solution:** Rotate each token's vector by an angle proportional to its position.

The rotation matrix for position $m$ at dimension pair $(2i, 2i+1)$:

$$R_m = \begin{pmatrix} \cos(m\theta_i) & -\sin(m\theta_i) \\ \sin(m\theta_i) & \cos(m\theta_i) \end{pmatrix}$$

where $\theta_i = 10000^{-2i/d}$

**The clock analogy:**

```
Position 0:  ●──→         (no rotation)
Position 1:  ●  ╲         (small rotation)
Position 2:  ●    ↓       (more rotation)
Position 3:  ●  ╱         (even more)

Low-freq dims:   ⟲ slow clock hand (hour hand)
High-freq dims:  ⟲ fast clock hand (second hand)
```

**Key property:** the dot product $q_m^T k_n$ depends only on the *relative* distance $m - n$, not absolute positions. This lets the model generalize to unseen sequence lengths.

<!-- notes: RoPE is used by LLaMA, Mistral, and most modern open models. The core insight is that rotation preserves vector norms while encoding position in the angle. When you take the dot product of two rotated vectors, the rotation angles subtract, giving you relative position. The different frequency bands (theta_i) work like a clock: the "hour hand" dimensions change slowly across positions, encoding coarse position, while the "second hand" dimensions change rapidly, encoding fine position. -->

---

## Attention = learned highlighting

$$\text{Attention}(Q, K, V) = \text{softmax}\left(\frac{QK^T}{\sqrt{d_k}}\right) V$$

**Softmax** turns a list of raw scores into positive probabilities that add up to 1 (the bigger the score, the bigger its share).

- **Query** $Q$: "What am I looking for?"
- **Key** $K$: "What do I contain?"
- **Value** $V$: "Here's my content if selected"

Step by step (for one query token attending to 4 keys):

```
1. Dot products:    scores = Q·K^T = [2.1, 0.3, -0.5, 1.8]
2. Scale:           scores / √d_k  = [0.66, 0.09, -0.16, 0.57]
3. Mask future:     [-inf for future tokens]
4. Softmax:         weights = [0.38, 0.22, 0.17, 0.23]   (sum to 1)
5. Weighted sum:    output = 0.38·V₁ + 0.22·V₂ + 0.17·V₃ + 0.23·V₄
```

**Multi-head:** Run $h$ parallel attention operations with different $W_Q, W_K, W_V$ projections, then concatenate. Each head can learn a different notion of "relevance."

<!-- notes: The scaling by sqrt(d_k) prevents dot products from growing large in high dimensions, which would push softmax into regions with vanishing gradients. Multi-head attention is like having multiple "perspectives" — one head might track subject-verb agreement, another might track coreference, another might focus on recent context. With 8 heads and d_model=512, each head works in d_k=64 dimensions. -->

---

## The causal mask

In autoregressive generation, token $t$ must **not** see tokens $t+1, t+2, \ldots$

We enforce this with a **lower-triangular mask** applied before softmax:

```
Token:    The   cat   sat   on
The     [  0   -∞    -∞    -∞  ]
cat     [  0    0    -∞    -∞  ]
sat     [  0    0     0    -∞  ]
on      [  0    0     0     0  ]
```

$-\infty$ entries become $0$ probability after softmax → future tokens are invisible.

**Without** this mask, the model could "cheat" during training by looking at the answer.

<!-- notes: This is the difference between an encoder (BERT, which sees all tokens bidirectionally) and a decoder (GPT, which sees only past tokens). The mask is just a matrix of zeros and negative infinities added to the attention scores before softmax. In PyTorch: torch.triu(torch.ones(T,T) * float('-inf'), diagonal=1). Simple but critical. -->

---

## The feed-forward network (FFN)

Each Transformer block contains an FFN applied **independently** to each token:

$$\text{FFN}(x) = \text{GELU}(xW_1 + b_1)W_2 + b_2$$

**Expand-contract pattern:**

```
Input:   x ∈ ℝ^512  (d_model)
         ↓
Expand:  xW₁ → ℝ^2048  (4 × d_model)    ← "think in high dimensions"
         ↓
GELU:    non-linearity
         ↓
Contract: ×W₂ → ℝ^512  (back to d_model) ← "compress insight"
```

**Why GELU over ReLU?**

```
ReLU:   max(0, x)          — hard cutoff at 0, kills gradients
GELU:   x · Φ(x)           — smooth, allows small negative flow
```

GELU is the standard in GPT-2, BERT, and all modern Transformers. It provides smoother gradients during training.

<!-- notes: The FFN is where most of the model's "knowledge" is stored. Attention decides WHAT to look at; the FFN processes what it sees. The 4x expansion factor is conventional — it gives the network a high-dimensional workspace to perform nonlinear transformations before projecting back. In MoE (Phase 4), we replace this single FFN with multiple expert FFNs, each potentially storing different knowledge. -->

---

## Parameter budget — where do 80M params go?

| Component | Shape | Params | % |
|-----------|-------|--------|---|
| Token embedding | $8000 \times 512$ | 4.1M | 5% |
| Per-layer Attention ($W_Q, W_K, W_V, W_O$) | $4 \times 512 \times 512$ | 1.05M × 8 | 10.5% |
| Per-layer FFN ($W_1, W_2$) | $512 \times 2048 + 2048 \times 512$ | 2.1M × 8 | 21% |
| Per-layer LayerNorms (×2) | $2 \times 512$ | 0.008M × 8 | <1% |
| LM head (weight-tied with embedding) | — | 0 (shared) | 0% |
| **Total** | | **~80M** | |

Key observations:
- **FFN dominates** (~21% per layer × 8 layers ≈ the bulk of parameters)
- **Weight tying** between embedding and LM head saves 4M params
- LayerNorms are negligible in parameter count but critical for training stability

<!-- notes: Have students calculate these numbers themselves in lab. The parameter budget reveals WHY MoE (Phase 4) targets the FFN: it's where most parameters live, and it's also the most "parallelizable" component because it operates per-token with no cross-token interaction. Understanding this table builds intuition for every architectural decision that follows. -->

---

## TinyStories dataset

Simple vocabulary (~3-year-old level). An 80M model learns **grammar in hours**, not weeks on Wikipedia.

**Why TinyStories?**

- Small vocabulary → small embedding table → less VRAM
- Short stories → short sequences → fits in 256 block size
- Simple grammar → 80M params is **sufficient** (not undertrained)
- Fast feedback loop: see coherent output after ~1 hour of training

**Sample from TinyStories:**

```
Once upon a time, there was a little girl named Lily. She liked to
play with her toys. One day, she found a big red ball in the park...
```

Download via Hugging Face `datasets` in the labs.

<!-- notes: TinyStories was created by Microsoft Research specifically to show that small models CAN learn language well if the data matches their capacity. GPT-4-level models were used to generate millions of simple stories. This is a curriculum design choice: we want students to see success quickly. A 80M model on Wikipedia would produce gibberish; on TinyStories it produces surprisingly coherent stories. -->

---

## Training recipe

| Hyperparameter | Value | Why |
|----------------|-------|-----|
| Batch size | 8–16 | Fits in 10GB VRAM |
| Sequence length | 256 | TinyStories are short |
| Learning rate | 3e-4 | Adam sweet spot for this scale |
| Warmup steps | 500 | Stabilize early training |
| LR schedule | Cosine decay | Smooth annealing to 1e-5 |
| Gradient clipping | 1.0 (max norm) | Prevent exploding gradients |
| Optimizer | AdamW | Weight decay = 0.01 |
| Precision | FP16 (mixed) | 2× speed, half memory |
| Total steps | ~20k | ~2-4 hours on RTX 3080 |

**The training loop in pseudocode:**

```
for batch in dataloader:
    logits = model(batch.input_ids)           # forward
    loss = cross_entropy(logits, batch.targets) # compute loss
    loss.backward()                           # backprop
    clip_grad_norm_(model.parameters(), 1.0)  # clip
    optimizer.step()                          # update weights
    scheduler.step()                          # adjust LR
    optimizer.zero_grad()                     # reset gradients
```

<!-- notes: Walk through each hyperparameter choice. Batch size is constrained by VRAM — larger is generally better for training stability but we can't fit more. The warmup prevents the model from taking huge steps with random weights. Cosine decay is standard — it lets the model explore early and fine-tune late. Gradient clipping at 1.0 prevents any single batch from causing a catastrophic weight update. -->

---

## Lab map

| Lab | Topic | What you build |
|-----|-------|----------------|
| 1.1 | Tokenization | BPE tokenizer from scratch |
| 1.2 | Embeddings | Embedding layer + positional encoding |
| 1.3 | RoPE | Rotation matrices applied to Q, K |
| 1.4 | Attention | Multi-head causal self-attention |
| 1.5 | Training loop | Full training with mixed precision |

Each lab builds on the previous one. By lab 1.5, you assemble all components into a complete Transformer.

<!-- notes: Labs are designed to be done in order. Each lab imports code from the previous one. Encourage students to actually read their generated text at each checkpoint — watching the model go from random characters to coherent sentences is the most rewarding part of the course. -->

---

## Deliverable

Checkpoint `phase1_80m.pt` — reused through Phase 8.

This single checkpoint is the **foundation** for everything that follows:
- Phase 2 fine-tunes it to follow instructions
- Phase 3 quantizes it to 4-bit
- Phase 4 grafts MoE experts onto its FFN layers
- Phases 5–8 add KV compression, RL, and memory

**Treat this checkpoint with care** — a bad Phase 1 model means bad everything else.

<!-- notes: The checkpoint contains the full model state_dict and optimizer state. Students should save both so they can resume training if needed. Recommend saving checkpoints every 5000 steps. The final checkpoint should achieve a validation perplexity around 15-25 on TinyStories, meaning the model is genuinely surprised by roughly 15-25 possible next tokens on average. -->

---

## Bridge to the Next Phase

**What you built in Phase 1:** a full ~80M-parameter Transformer — BPE tokenizer, embeddings, RoPE positions, multi-head causal attention, FFN blocks, and an autoregressive training loop — saved as `phase1_80m.pt`.

**The thread into Phase 2:** Phase 2 does **not** change a single layer of this architecture. It reuses the *exact same model and the same cross-entropy loss*, but changes **what data we train on** and **which tokens we score**. The causal, next-token prediction you mastered here is the only mechanism Phase 2 needs — it simply formats text as a chat conversation and grades only the assistant's replies (masked loss).

So carry forward two ideas: **next-token prediction** and the **`phase1_80m.pt` checkpoint**. Phase 2 builds an assistant on top of both.

<!-- notes: The key continuity message: instruction tuning is NOT a new architecture. Same Transformer, same loss function, same training loop. The only differences are the dataset (chat-formatted) and a mask that zeroes out loss on user tokens. Students who understand that Phase 1's model just predicts the next token will immediately grasp why fine-tuning on Q&A data makes it answer questions. -->

---

## Next

**Phase 2:** Instruction tuning and JSON tool calling.

We will teach this base model to:
- Follow a structured chat format
- Think step-by-step with `<|Thought|>` tags
- Solve arithmetic problems (GSM8K)
- Emit valid JSON for tool/function calling

<!-- notes: Preview: Phase 2 is where the model goes from "babbling storyteller" to "useful assistant." The key technique is masked loss — we only grade the model on its own responses, not on the user's prompts. This is a simple but powerful idea that makes instruction tuning work. -->

---

## Further Reading (Optional)

**These papers are optional enrichment — you do NOT need to read any of them to continue the course.**

- Vaswani et al. (2017). *Attention Is All You Need*. NeurIPS.
- Sennrich, Haddow, & Birch (2016). *Neural Machine Translation of Rare Words with Subword Units (BPE)*. ACL.
- Su et al. (2021). *RoFormer: Enhanced Transformer with Rotary Position Embedding*. arXiv:2104.09864.
- Radford et al. (2019). *Language Models are Unsupervised Multitask Learners (GPT-2)*. OpenAI.
- Eldan & Li (2023). *TinyStories: How Small Can Language Models Be and Still Speak Coherent English?*. arXiv:2305.07759.
- Ba, Kiros, & Hinton (2016). *Layer Normalization*. arXiv:1607.06450.
- Hendrycks & Gimpel (2016). *Gaussian Error Linear Units (GELUs)*. arXiv:1606.08415.

<!-- notes: Strictly optional. Vaswani et al. is the original Transformer; the Sennrich BPE paper underpins our tokenizer; RoFormer introduced RoPE; GPT-2 is the Pre-LN decoder we mirror; TinyStories justifies the dataset choice; the Layer Normalization and GELU papers cover two components we use directly. Every concept is already explained in the slides — these are for the curious. -->
