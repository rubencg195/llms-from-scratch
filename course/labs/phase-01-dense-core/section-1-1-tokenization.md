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

# Section 1.1: Tokenization — Turning Words into Python Dictionaries

**Goal:** Train a small BPE tokenizer on TinyStories and encode/decode text to integer sequences.

## Why Tokenization Matters

Before a language model can process text, it must convert characters into **integers** (token IDs). The choice of tokenization strategy creates a fundamental tradeoff:

| Strategy | Vocab Size | Sequence Length | Problem |
|----------|-----------|-----------------|---------|
| Character-level | ~256 | Very long | Sequences too long for attention (quadratic cost) |
| Word-level | 100,000+ | Short | Huge embedding table, can't handle typos/new words |
| Subword (BPE) | 8,000–100,000 | Medium | Best tradeoff — handles rare words via subword pieces |

**The core tension:** smaller vocabulary → longer sequences → more compute per sample. Larger vocabulary → shorter sequences → more memory for the embedding matrix. Modern LLMs use BPE with 32k–128k tokens as the sweet spot.

```python
from datasets import load_dataset
from collections import Counter
import torch
import tiktoken

# Stream a subset — full dataset is large
ds = load_dataset("roneneldan/TinyStories", split="train[:1%]")
sample = ds[0]["text"]
print(sample[:300])
```

## Character-level baseline

The simplest approach: one token per character. Simple but creates extremely long sequences.

```python
text = "the cat sat"
chars = sorted(set(text))
stoi = {c: i for i, c in enumerate(chars)}
itos = {i: c for c, i in stoi.items()}

encode = lambda s: [stoi[c] for c in s]
decode = lambda ids: "".join(itos[i] for i in ids)

ids = encode(text)
print("ids:", ids)
print("decoded:", decode(ids))
print(f"Vocab size: {len(chars)}, Sequence length: {len(ids)}")
print(f"Ratio (seq_len / text_len): {len(ids) / len(text):.1f}x — every char is a token")
```

## BPE with tiktoken (production-style)

```python
# GPT-2 encoding as reference; lab builds TinyStories-specific vocab in steps below
enc = tiktoken.get_encoding("gpt2")
tokens = enc.encode("Once upon a time")
print("BPE token ids:", tokens)
print("tokens:", [enc.decode([t]) for t in tokens])
print(f"4 words → {len(tokens)} tokens (subword compression)")
```

## Subword Tokenization: The BPE Merge Algorithm

Byte-Pair Encoding (BPE) starts with individual characters and **iteratively merges** the most frequent adjacent pair into a new token. This builds a vocabulary bottom-up:

1. Start with character-level tokens
2. Count all adjacent pairs in the corpus
3. Merge the most frequent pair into a single new token
4. Repeat until desired vocab size is reached

```python
def get_pairs(word_freqs):
    """Count frequency of each adjacent symbol pair across the corpus."""
    pairs = Counter()
    for word, freq in word_freqs.items():
        symbols = word.split()
        for i in range(len(symbols) - 1):
            pairs[(symbols[i], symbols[i + 1])] += freq
    return pairs

def merge_pair(pair, word_freqs):
    """Merge all occurrences of the most frequent pair."""
    new_word_freqs = {}
    bigram = " ".join(pair)
    replacement = "".join(pair)
    for word, freq in word_freqs.items():
        new_word = word.replace(bigram, replacement)
        new_word_freqs[new_word] = freq
    return new_word_freqs

# Build corpus: each word represented as space-separated characters
corpus_text = " ".join(row["text"][:200] for row in ds[:100])
words = corpus_text.lower().split()
word_freq_raw = Counter(words)

# Represent each word as space-separated characters + end-of-word marker
word_freqs = {}
for word, freq in word_freq_raw.most_common(500):
    word_freqs[" ".join(list(word)) + " </w>"] = freq

print("Initial vocabulary (characters):")
vocab_bpe = set()
for word in word_freqs:
    for symbol in word.split():
        vocab_bpe.add(symbol)
print(f"  {len(vocab_bpe)} unique symbols")
print(f"  Sample: {sorted(list(vocab_bpe))[:20]}")
```

```python
# Run BPE merges
num_merges = 50
merges = []

for i in range(num_merges):
    pairs = get_pairs(word_freqs)
    if not pairs:
        break
    best_pair = max(pairs, key=pairs.get)
    merges.append(best_pair)
    word_freqs = merge_pair(best_pair, word_freqs)

    if i < 10 or i % 10 == 9:
        print(f"Merge {i+1:2d}: '{best_pair[0]}' + '{best_pair[1]}' → '{''.join(best_pair)}' (freq={pairs[best_pair]})")

print(f"\nAfter {num_merges} merges, vocabulary grew from {len(vocab_bpe)} → {len(vocab_bpe) + num_merges} tokens")
print("Learned merge rules (first 10):")
for p in merges[:10]:
    print(f"  {''.join(p)}")
```

## Build vocab from corpus (simplified WordPiece loop)

```python
def word_freqs_fn(corpus, max_docs=500):
    counter = Counter()
    for i, row in enumerate(corpus):
        if i >= max_docs:
            break
        for word in row["text"].lower().split():
            counter[word] += 1
    return counter

freqs = word_freqs_fn(ds)
print("top words:", freqs.most_common(10))
vocab_size = 8000  # target for 80M model
print(f"Target vocab size: {vocab_size}")
```

## Vocabulary Coverage Analysis

A critical question: what percentage of the text can your vocabulary cover? Tokens not in the vocabulary become "UNK" (unknown), losing information.

```python
# Build minimal vocab from most common 7999 words + UNK at 0
words_list = [w for w, _ in freqs.most_common(7999)]
vocab = {w: i + 1 for i, w in enumerate(words_list)}
vocab["<|unk|>"] = 0

# Measure coverage on a sample of the dataset
total_tokens = 0
known_tokens = 0
oov_examples = []

for i, row in enumerate(ds):
    if i >= 200:
        break
    for word in row["text"].lower().split():
        total_tokens += 1
        if word in vocab:
            known_tokens += 1
        elif len(oov_examples) < 20:
            oov_examples.append(word)

coverage = known_tokens / total_tokens * 100
print(f"Vocabulary coverage: {coverage:.1f}% of tokens are known")
print(f"OOV rate: {100 - coverage:.1f}% of tokens are unknown")
print(f"\nSample OOV words: {oov_examples[:10]}")
print(f"\nThis is why BPE is better: subword pieces cover rare words without a massive vocab.")
```

## Save encodings for training

```python
def simple_whitespace_encode(s, v):
    """Map known words to id; unknown → 0 (UNK)."""
    return [v.get(w, 0) for w in s.lower().split()]

seq = simple_whitespace_encode(sample, vocab)
print("sequence length:", len(seq))
print("first 20 ids:", seq[:20])
```

**Production note:** Phase 1 final checkpoint uses a proper BPE merge table; this lab teaches the **ID ↔ text** contract.

## Exercise: Compare Tokenization Strategies

Compare how character-level, word-level, and BPE tokenize the same text. Observe the sequence length vs vocabulary size tradeoff.

```python
test_sentence = "The little girl was playing with her beautiful golden retriever in the garden."

# Character-level
char_tokens = list(test_sentence)
print(f"Character-level: {len(char_tokens)} tokens, vocab ~256")

# Word-level
word_tokens = test_sentence.split()
print(f"Word-level: {len(word_tokens)} tokens, vocab ~100k+")

# BPE (GPT-2)
bpe_tokens = enc.encode(test_sentence)
bpe_decoded = [enc.decode([t]) for t in bpe_tokens]
print(f"BPE (GPT-2): {len(bpe_tokens)} tokens, vocab 50,257")
print(f"  BPE pieces: {bpe_decoded}")

# Compression ratio
print(f"\nCompression ratios (vs character-level):")
print(f"  Word-level: {len(char_tokens)/len(word_tokens):.1f}x compression")
print(f"  BPE:        {len(char_tokens)/len(bpe_tokens):.1f}x compression")
```

## Key Takeaway

- **Tokenization** converts raw text into integer sequences that models can process — it's the first and last step in every LLM pipeline.
- **BPE** (Byte-Pair Encoding) achieves the best tradeoff: moderate vocab size, reasonable sequence lengths, and graceful handling of rare/unseen words via subword decomposition.
- The **merge algorithm** is elegant: iteratively combine the most frequent adjacent pair until you reach your target vocabulary size.
- **Vocabulary coverage** analysis reveals how much information you lose to UNK tokens — BPE minimizes this by construction.
- In Phase 1, we use a vocab of 8,000 tokens — small enough for fast training on an RTX 3080, large enough to capture TinyStories grammar.

## Checkpoint

You can encode text ↔ integers. Next: **embeddings** (Section 1.2) — turning those integer IDs into learnable dense vectors.
