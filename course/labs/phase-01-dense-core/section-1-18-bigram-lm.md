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

# Section 1.18: Bigram Language Model — Count, Normalize, Generate

**Goal:** Build the simplest possible language model (bigram counts) before the full Transformer, following Karpathy's *makemore* Part 1 ladder.

## What You Need to Know First

- **Tokenization** (Section 1.1) — text becomes a list of token IDs or words.
- **Probability intuition** — if "the" is followed by "cat" 40 times and "dog" 10 times out of 50, then $P(\text{cat} \mid \text{the}) \approx 0.8$.

Do this section **right after Section 1.1**. It shows what a language model *is* (next-token prediction) without any neural network.

## What is a language model?

A language model assigns a probability to the **next token** given everything before it:

$$P(w_t \mid w_1, w_2, \ldots, w_{t-1})$$

A **bigram** model approximates this with only the previous token:

$$P(w_t \mid w_{t-1})$$

It is crude, but it trains in seconds, generates readable text, and makes the Transformer's job concrete.

## Load a tiny corpus

```python
from collections import Counter, defaultdict
from datasets import load_dataset
import torch
import torch.nn.functional as F

ds = load_dataset("roneneldan/TinyStories", split="train[:0.5%]")
text = " ".join(ds[i]["text"] for i in range(min(500, len(ds)))).lower()
words = text.split()
print(f"Corpus: {len(words):,} words, {len(set(words)):,} unique")
print("Sample:", " ".join(words[:20]))
```

## Count bigrams

```python
bigram_counts = defaultdict(Counter)
for w1, w2 in zip(words, words[1:]):
    bigram_counts[w1][w2] += 1

# What follows "the"?
print("After 'the':", bigram_counts["the"].most_common(8))
```

## Normalize rows to probabilities

Each row is a conditional distribution over next words:

```python
def bigram_probs(counts_dict):
    """counts_dict[w1][w2] = count → tensor of P(w2 | w1)."""
    vocab = sorted(set(counts_dict.keys()) | {w2 for c in counts_dict.values() for w2 in c})
    stoi = {w: i for i, w in enumerate(vocab)}
    itos = {i: w for w, i in stoi.items()}
    n = len(vocab)
    M = torch.zeros(n, n)

    for w1, next_counts in counts_dict.items():
        i = stoi[w1]
        row = torch.tensor([next_counts.get(w2, 0) for w2 in vocab], dtype=torch.float32)
        M[i] = row / row.sum()  # normalize row

    return M, stoi, itos

P, stoi, itos = bigram_probs(bigram_counts)
print(f"Bigram matrix shape: {P.shape}")  # [vocab, vocab]
```

## Sample text (Karpathy-style generation loop)

```python
import random

def sample_bigram(P, stoi, itos, n_tokens=40, start="once"):
    if start not in stoi:
        start = random.choice(list(stoi.keys()))
    out = [start]
    for _ in range(n_tokens - 1):
        i = stoi[out[-1]]
        probs = P[i]
        j = torch.multinomial(probs, num_samples=1).item()
        out.append(itos[j])
    return " ".join(out)

print(sample_bigram(P, stoi, itos, start="once"))
print(sample_bigram(P, stoi, itos, start="the"))
```

Same loop as Section 1.5's `generate()` — only the probability source changes.

## Negative log-likelihood (the training objective)

Cross-entropy on a bigram model is just $-\log P(w_t \mid w_{t-1})$:

```python
def bigram_nll(words, P, stoi):
    nll = 0.0
    n = 0
    for w1, w2 in zip(words, words[1:]):
        if w1 in stoi and w2 in stoi:
            p = P[stoi[w1], stoi[w2]].item()
            nll -= torch.log(torch.tensor(p + 1e-9)).item()
            n += 1
    return nll / max(n, 1)

print(f"Average NLL (lower = better): {bigram_nll(words[:2000], P, stoi):.3f}")
```

The Transformer's cross-entropy loss is this same idea, but with a **neural network** producing the probability row instead of a count table.

## Bigram vs neural: the upgrade path

| | Bigram (this lab) | Transformer (Sections 1.2–1.5) |
|---|---|---|
| Context | 1 previous token | All previous tokens (via attention) |
| Parameters | $V \times V$ table | ~80M learned weights |
| Training | Count + divide | Gradient descent |
| Quality | Repetitive, local | Coherent stories |

```python
# Bigram row for "the" — fixed counts
i = stoi["the"]
print("Bigram top-5 after 'the':", [itos[j] for j in P[i].topk(5).indices.tolist()])

# Neural model (later): embedding("the") → attention over full context → logits
# Same output shape: [vocab_size] probabilities. Different mechanism.
```

## Exercise: trigram model

Extend to $P(w_t \mid w_{t-2}, w_{t-1})$ using `defaultdict(Counter)` keyed by `(w1, w2)` tuples. Compare sample quality vs bigram.

## Where This Leads Next

Section 1.2 replaces the count table with **learned embeddings** — each word becomes a vector. Section 1.4 adds **attention** so the model sees more than one previous word. Section 1.5 runs the full training loop you already understand from the bigram sampling loop.

## Key Takeaway

- A language model = **next-token probability distribution**.
- **Generation** = sample → append → repeat (identical loop at every scale).
- **Cross-entropy** = average $-\log P(\text{correct next token})$.
- The Transformer is a bigram model's context window expanded to the entire prefix.

## Checkpoint

You built and sampled from a language model with zero neural nets. Next: **Section 1.2 — Embeddings**.

## Further Reading (Optional)

- Karpathy, *makemore* Part 1 — bigram language model.
- Bengio et al. (2003). *A Neural Probabilistic Language Model*. JMLR. (the MLP step between bigram and Transformer)
