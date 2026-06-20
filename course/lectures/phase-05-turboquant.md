---
title: "Phase 5: TurboQuant"
subtitle: "Squeezing Short-Term Memory"
author: "LLMs From Scratch"
---

# Phase 5: TurboQuant

## Long context on 10 GB VRAM via compressed KV cache

---

## Before You Begin (Prerequisites)

Everything you need was taught in earlier phases — **no outside knowledge required**.

- **From Phase 1 you need:** attention (Query/Key/Value vectors) and the **KV cache** concept — the stored Keys and Values that let the model avoid recomputing past tokens.
- **From Phase 0 you need:** tensors and basic matrix multiplication — that's all the "math" here really is.
- **High-school algebra is enough:** if you can read $x_q = \text{round}(\dots)$ and picture a number line, you're ready.

New terms like *quantization*, *outliers*, and *orthogonal rotation* are all defined on the slides as they appear. Relax — we build up slowly.

<!-- notes: Reassure students here. The only hard prerequisite is comfort with the KV cache from Phase 1 — if they remember that the model stores Keys and Values for every past token, they have what they need. Everything else (quantization, rotation) is introduced from scratch in this deck. Emphasize that the equations look scarier than they are: they are mostly rounding and multiplication. -->

---

## Learning objectives

- Diagnose **KV cache VRAM** growth
- Apply **PolarQuant** rotation to smooth outliers
- Store cache at **3.5 bits** per value
- Pass **Needle-in-a-Haystack** at 8k tokens

<!-- notes: This phase tackles the single biggest deployment bottleneck after model weights: the KV cache. Students will leave with a working quantized cache that fits 8k tokens where FP16 couldn't fit 2k. -->

---

## Context Length Arms Race

| Model | Max Context | Year |
|-------|------------|------|
| GPT-3 | 2,048 | 2020 |
| GPT-4 | 128,000 | 2023 |
| Claude 3 | 200,000 | 2024 |
| Gemini 1.5 Pro | 1,000,000+ | 2024 |

Long context unlocks **full-codebase reasoning**, **book-length summarization**, and **multi-document RAG without chunking**.

<!-- notes: The industry trend is clear — every generation doubles or 10x's context length. But longer context means proportionally more memory for the KV cache. This is the fundamental tension we solve in this phase. Real applications like legal doc review or code analysis demand 100k+ tokens. -->

---

## The VRAM problem

Each generated token stores Key and Value tensors for **every layer**.

$$\text{KV bytes} = 2 \times L \times H \times d_h \times s \times b_{\text{dtype}}$$

Where $L$ = layers, $H$ = heads, $d_h$ = head dim, $s$ = seq length, $b_{\text{dtype}}$ = bytes per value.

<!-- notes: The factor of 2 accounts for both K and V. Notice this is linear in sequence length — double the context, double the cache. That's manageable in theory, but at FP16 it eats VRAM fast. Let's put real numbers on this. -->

---

## Concrete VRAM numbers: our 80M model

Our model: $L{=}12$, $H{=}12$, $d_h{=}64$, FP16 ($b{=}2$ bytes)

| Seq Length | KV Cache Size | Remaining for Weights+Activations |
|-----------|--------------|----------------------------------|
| 512 | **18.9 MB** | Comfortable |
| 2,048 | **75.5 MB** | Fine |
| 8,192 | **302 MB** | Tight on 10 GB |
| 16,384 | **604 MB** | Over budget |
| 65,536 | **2.4 GB** | Impossible |

$$\text{Example: } 2 \times 12 \times 12 \times 64 \times 8192 \times 2 = 301{,}989{,}888 \text{ bytes} \approx 302 \text{ MB}$$

<!-- notes: These numbers are for our tiny 80M model. Scale to a 70B model and the KV cache at 128k tokens is over 40 GB — larger than the model weights themselves. That's why every frontier lab is working on KV cache compression. -->

---

## Why Not Just Buy More VRAM?

| GPU | VRAM | Cost |
|-----|------|------|
| RTX 4090 | 24 GB | $1,600 |
| A100 80GB | 80 GB | $15,000 |
| H100 80GB | 80 GB | $30,000+ |
| 8× H100 node | 640 GB | $250,000+ |

- **Quadratic scaling problem**: doubling context requires doubling VRAM for cache *and* quadratic attention cost
- **Democratization**: most researchers and startups have 10–24 GB GPUs
- **Edge deployment**: phones, laptops have 4–16 GB *shared* memory

Compression lets a $1,600 GPU do what a $30,000 GPU does naively.

<!-- notes: This isn't just about cost — it's about access. If only Google and OpenAI can run long-context models, the research community can't reproduce or improve on their work. Quantized KV caches are a democratization tool. -->

---

## Outliers break naive quantization

*Quantization* = storing numbers with fewer bits by snapping them to a small set of allowed values (like rounding prices to the nearest 5 cents). *Outliers* = a few unusually large values that stretch the scale.

**Min-max quantization**: map $[\min, \max] \to [0, 2^b - 1]$

$$x_q = \text{round}\!\left(\frac{x - \min}{\max - \min} \cdot (2^b - 1)\right)$$

**The problem**: imagine 127 values in $[-1, 1]$ and one outlier at $100$.

- Full range becomes $[-1, 100]$
- 99% of values map to the **bottom 1%** of quantization bins
- Effective precision for typical values: **< 1 bit**

<!-- notes: This is like having a thermometer that goes from -10 to 1000 degrees. For everyday temperatures between 0 and 40, you can't distinguish anything. One extreme reading ruins the scale for all the normal ones. -->

---

## Visualizing outlier channels

Imagine a histogram of activation magnitudes across 64 head dimensions:

```
Dim 0-62:  ████████ (range: -2 to +2)
Dim 63:    ████████████████████████████████████████ (range: -200 to +200)
```

One channel has **100× the dynamic range** of the others. Min-max quantization uses that one channel's range for *all* channels, destroying precision everywhere else.

This is called the **outlier channel** phenomenon — first documented in LLM.int8() (Dettmers et al., 2022).

<!-- notes: This pattern appears consistently across transformer models. Certain dimensions in the hidden state carry disproportionately large magnitudes. The insight of rotation-based methods is that these outliers aren't inherent to the information — they're an artifact of the basis the model learned. Change the basis, and the outliers spread out. -->

---

## Hadamard / Orthogonal Rotation

**Key insight**: outlier energy is concentrated in a few dimensions. An orthogonal rotation **spreads** that energy across all dimensions.

Given rotation matrix $R$ (where $R R^T = I$):

$$K_{\text{rot}} = K \cdot R$$

**Geometric analogy**: imagine a long thin ellipse (outlier = one stretched axis). Rotation turns it into a more circular shape — same total energy, but evenly distributed.

**Why orthogonal?** Preserves norms and dot products:

$$q \cdot k = (qR) \cdot (kR)$$

Attention scores are **identical** after rotation, so we lose nothing.

<!-- notes: This is the core mathematical trick. A Hadamard matrix is a specific efficient orthogonal matrix where every entry is ±1/√n, making the matrix-vector product O(n log n) via the fast Walsh-Hadamard transform. But any random orthogonal matrix works — Hadamard is just fast. The key property is that multiplying by an orthogonal matrix redistributes magnitude across dimensions without changing the geometry of the space. -->

---

## PolarQuant pipeline: step by step

```
Original K,V (FP16)
       │
       ▼
   ┌──────────┐
   │  Rotate   │  K_rot = K · R,  V_rot = V · R
   └──────────┘
       │
       ▼
   ┌──────────┐
   │ Quantize  │  Per-group min-max → INT8 codes + scales
   └──────────┘
       │
       ▼
   ┌──────────┐
   │  Store    │  3.5-bit packed integers in cache
   └──────────┘
       │  (on attention read)
       ▼
   ┌──────────┐
   │Dequantize │  INT codes → FP16 using stored scales
   └──────────┘
       │
       ▼
   ┌──────────┐
   │ De-rotate │  K_hat = K_dq · R^T
   └──────────┘
       │
       ▼
   Use in standard attention
```

Because $R$ is orthogonal, $R^T = R^{-1}$ — de-rotation is free.

<!-- notes: The rotation matrix R is computed once at initialization and stored. It's the same for every token, so there's no per-token overhead to generate it. The only runtime cost is two matrix multiplications (rotate and de-rotate), which are tiny compared to attention itself. The big win is that after rotation, the value distribution is much more uniform, so quantization loses far less precision. -->

---

## 3.5-bit encoding: fractional bits

Standard quantization uses integer bit widths: 8-bit (256 levels), 4-bit (16 levels).

**3.5-bit idea**: use a **non-uniform codebook** with $\lceil 2^{3.5} \rceil = 12$ levels.

| Approach | Levels | Bits/value | Storage savings vs FP16 |
|----------|--------|-----------|------------------------|
| INT8 | 256 | 8.0 | 2× |
| INT4 | 16 | 4.0 | 4× |
| 3.5-bit | 12 | 3.5 | 4.57× |
| INT2 | 4 | 2.0 | 8× |

**Implementation**: pack pairs of 3.5-bit values into 7 bits, or groups of 8 values into 28 bits.

Non-uniform spacing clusters more levels near the peak of the value distribution (like $\mu$-law companding in telephony).

<!-- notes: The 3.5-bit encoding is a sweet spot — you get almost 5× compression over FP16 while retaining enough precision that the post-rotation values quantize cleanly. Below 3 bits, quality degrades sharply because you can't represent the distribution faithfully. The non-uniform codebook is learned from calibration data — you run a few hundred tokens through the model, collect the post-rotation KV distributions, and fit a 12-level codebook that minimizes reconstruction error. -->

---

## Compression savings for our model

With 3.5-bit TurboQuant at 8,192 tokens:

| Method | Cache Size | Savings |
|--------|-----------|---------|
| FP16 (baseline) | 302 MB | 1× |
| INT8 naive | 151 MB | 2× |
| INT4 naive | 75.5 MB | 4× (but broken — outliers) |
| TurboQuant 3.5-bit | **66 MB** | **4.57×** |

$302 \text{ MB} \div 4.57 \approx 66 \text{ MB}$ — now 8k context fits easily in 10 GB.

At 16k tokens: $604 \div 4.57 \approx 132$ MB. Still fine.

<!-- notes: This is the practical payoff. We went from "8k doesn't fit" to "16k fits comfortably" with a single technique. Combine with grouped-query attention (GQA — several attention heads share one set of Keys/Values to shrink the cache) or multi-query attention (MQA — all heads share one set) and you can push even further. The frontier labs stack all these techniques. -->

---

## Evaluation: Needle in a Haystack

**Setup**: hide a fact like `"The magic passcode is cyan"` at a specific **depth** in a long filler document. Query at the end: *"What is the magic passcode?"*

**Methodology**: sweep two axes:

- **Context length**: 1k, 2k, 4k, 8k tokens
- **Needle depth**: 0%, 25%, 50%, 75%, 100% through document

**Heatmap result**: each cell = retrieval accuracy

```
Depth↓ \ Length→   1k    2k    4k    8k
    0% (start)    100%  100%  100%   98%
   25%            100%  100%   97%   95%
   50%            100%   99%   95%   90%
   75%            100%   98%   93%   87%
  100% (end)      100%  100%  100%   99%
```

**Goal**: maintain >90% accuracy across all cells at 8k with TurboQuant.

<!-- notes: The heatmap reveals where the model's memory breaks down. Deep needles in long contexts are hardest because the model must attend accurately over thousands of tokens. KV cache quantization introduces noise that can make these long-range retrievals fail. Our test verifies that TurboQuant's rotation smooths the quantization noise enough to preserve retrieval at 8k. Notice that start and end are easiest — primacy and recency effects are real even in transformers. -->

---

## Quality control: what can go wrong

| Failure Mode | Symptom | Fix |
|-------------|---------|-----|
| Too few bits | Garbled output at long context | Increase to 4-bit |
| Bad rotation | Outliers persist after rotation | Verify $R$ is orthogonal ($R R^T \approx I$) |
| Scale overflow | NaN in attention | Use per-head scaling |
| Group size mismatch | Accuracy drops at boundaries | Align groups to head dim |

Always compare **perplexity** (a score of how "surprised" the model is by text — lower is better) with and without quantization on a held-out set.

<!-- notes: Debugging quantized models can be tricky because failures are often silent — the model produces fluent but wrong text. Always have a quantitative eval (perplexity + needle test) alongside qualitative inspection. -->

---

## Lab map

| Lab | Topic |
|-----|-------|
| 5.1 | VRAM problem |
| 5.2 | PolarQuant rotation |
| 5.3 | 3.5-bit compression |
| 5.4 | Needle-in-a-Haystack test |

<!-- notes: Lab 5.1 profiles the actual KV cache size at various sequence lengths. Lab 5.2 implements the rotation and verifies outlier smoothing. Lab 5.3 builds the quantizer with non-uniform codebook. Lab 5.4 puts it all together with the needle benchmark. -->

---

## Key takeaways

1. **KV cache is the bottleneck** for long-context inference, not model weights
2. **Outlier channels** make naive quantization useless — rotation fixes this
3. **3.5-bit encoding** hits the sweet spot: 4.57× compression, minimal quality loss
4. **Needle-in-a-Haystack** is the gold-standard eval for long-context fidelity

<!-- notes: The techniques in this phase are directly applicable to production. Every major inference framework (vLLM, TensorRT-LLM, llama.cpp) has some form of KV cache quantization. Understanding the math here means you can debug and improve these systems. -->

---

## Bridge to the Next Phase

You just learned to **shrink the KV cache** so longer sequences fit in 10 GB. Why does that matter for **Phase 6 (multimodal)**?

- In Phase 6, every image becomes a string of **patch tokens** added to the sequence — a single picture can cost dozens or hundreds of extra tokens.
- More tokens → a **bigger KV cache**. The compression skills from this phase are exactly what keep image + text sequences affordable.
- Same idea, new payload: Phase 5 squeezed *text* memory; Phase 6 adds *vision* into that same memory budget.

So: **efficient memory is the runway** that lets us bolt on new modalities next.

<!-- notes: The goal of this bridge is to make students feel the phases connect, not stand alone. The KV cache is the shared resource across every later phase. Once images turn into patch tokens (Phase 6) and audio turns into codec tokens (Phase 7), they all flow through the same attention and the same cache we just learned to compress. Frame Phase 5 as the "memory budget" phase that everything afterward spends from. -->

---

## Further Reading (Optional)

**These papers are optional enrichment — you do NOT need to read any of them to continue the course.**

- Ainslie et al. (2023). *GQA: Training Generalized Multi-Query Transformer Models from Multi-Head Checkpoints*. EMNLP.
- Chee et al. (2023). *QuIP: 2-Bit Quantization of Large Language Models With Guarantees*. NeurIPS.
- Tseng et al. (2024). *QuIP#: Even Better LLM Quantization with Hadamard Incoherence and Lattice Codebooks*. ICML.
- Ashkboos et al. (2024). *QuaRot: Outlier-Free 4-Bit Inference in Rotated LLMs*. arXiv:2404.00456.
- Liu et al. (2024). *KIVI: A Tuning-Free Asymmetric 2bit Quantization for KV Cache*. ICML.
- Kamradt (2023). *Needle In A Haystack — Pressure Testing LLMs*. (GitHub: gkamradt/LLMTest_NeedleInAHaystack)

*Aside:* "PolarQuant/TurboQuant" in this course are **teaching names** for the rotation-then-quantize family pioneered by **QuIP#/QuaRot**.

<!-- notes: Keep the pressure off — these are enrichment, not homework. If a student is curious where our "PolarQuant" and "TurboQuant" names come from, point them to QuIP# and QuaRot, which introduced the rotate-then-quantize trick we teach. KIVI is the closest real analog to our KV-cache-specific quantization, and Kamradt's Needle-in-a-Haystack is the eval we use in Lab 5.4. -->

---

## Next

**Phase 6:** Encoder-free multimodal (Gemma 4 style).
