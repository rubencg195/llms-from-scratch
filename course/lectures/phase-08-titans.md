---
title: "Phase 8: Google's Titans Architecture"
subtitle: "Living Neural Memory"
author: "LLMs From Scratch"
---

# Phase 8: Google's Titans Architecture

## Replace static history with memory that learns while you chat

---

## Before You Begin (Prerequisites)

This capstone reuses the whole course — **no external knowledge required**.

- **From Phase 0 you need:** *gradient descent* — adjusting weights a little in the direction that lowers the loss. Titans does this *during inference*, but it's the same one step you've done since day one.
- **From Phase 1 you need:** *attention* and the *KV cache* — the "short-term memory" Titans sits alongside.
- **From Phase 5 you need:** the pain that **the KV cache grows with context** — Titans is the fix.
- **From Phases 6–7 you need:** the habit of stacking new modules onto the same transformer; here we add a memory module.
- **High-school algebra is enough:** the update is one line, $\theta_{t+1} = \theta_t - \eta \nabla_\theta \mathcal{L}$, which just means "nudge the weights downhill."

A *gradient* is simply the direction of steepest increase; we step the opposite way to improve.

<!-- notes: This is the capstone, so the prerequisite list is intentionally a recap of the arc. The single most important prerequisite is gradient descent from Phase 0 — students already know it; the only twist is that Titans runs it at inference time on a tiny memory network. Reassure them that the scary-looking TTT equation is the exact same update rule from training, just applied per token during chat. Everything else (attention, KV cache, modular stacking) they've practiced for seven phases. -->

---

## Learning objectives

- **Test-Time Training (TTT):** gradient updates during inference
- Implement **neural memory module**
- **Surprise metric** triggers writes
- **Momentum + decay** for forgetting
- Integrate memory with **attention**
- Prove **O(1) decode** VRAM on RTX 3080

<!-- notes: This is the capstone phase. We take everything built in Phases 0-7 and add the final frontier capability: memory that grows and adapts during a conversation. This is the key innovation from Google's Titans paper — the idea that a model can literally learn new facts in real-time by performing gradient descent on its own memory weights during inference. Students will implement the full Titans architecture and demonstrate O(1) decode-step memory. -->

---

## The Memory Problem in LLMs

Standard attention: $O(n^2)$ compute, $O(n)$ KV cache memory.

| Context Length | Attention FLOPs | KV Cache (80M model, FP16) |
|---------------|----------------|--------------------------|
| 1k | 1× | 37 MB |
| 4k | 16× | 151 MB |
| 16k | 256× | 604 MB |
| 64k | 4,096× | 2.4 GB |
| 256k | 65,536× | 9.6 GB |

Even with Phase 5's TurboQuant (4.57× compression), 256k tokens need 2.1 GB just for cache.

**Fundamental issue**: KV cache stores *every token ever seen*. Do you really need all of them?

<!-- notes: Think about how human memory works. You don't remember every word of a conversation verbatim. You remember the key facts, the important context, the emotional beats. You compress and abstract. KV cache is like a tape recorder — it stores everything literally, which is both wasteful and eventually impossible. Titans introduces a memory that works more like human memory: it stores compressed representations that can be queried, and it forgets unimportant details over time. -->

---

## Meta-Learning Meets Inference

**Test-Time Training (TTT)** is inner-loop optimization during inference — a concept from *meta-learning* (training a model to be good at *learning new things quickly*, i.e. "learning to learn").

Connection to MAML/Reptile:

| Concept | Meta-Learning (MAML) | Titans TTT |
|---------|---------------------|-----------|
| Outer loop | Pre-training | Pre-training the full model |
| Inner loop | Few-shot adaptation | Per-token memory update |
| What adapts | Task-specific head | Memory network weights |
| Update rule | Gradient descent | Gradient descent |
| When | At test time | At inference time |

**Key insight**: the memory module's weights $\theta$ are *designed* to be updated at inference time. Pre-training teaches the model *how to learn* from new tokens.

<!-- notes: This connection to meta-learning is deep and important. In MAML, you train a model such that a few gradient steps on a new task produce good performance. In Titans, you train the memory module such that a few gradient steps on new tokens produce good memorization. The outer loop (pre-training) doesn't just learn facts — it learns a learning algorithm. The inner loop (inference) executes that algorithm on new data. This is why Titans can memorize new facts from a single exposure — the pre-training has prepared the weight space for rapid adaptation. -->

---

## TTT math: the update rule

At each token $x_t$, the memory network $f_\theta$ predicts the next representation. The update:

$$\theta_{t+1} = \theta_t - \eta \nabla_\theta \mathcal{L}(f_\theta(x_t), y_t)$$

Where:
- $\theta_t$ = memory weights at step $t$ (a small MLP, ~100K params)
- $f_\theta(x_t)$ = memory's prediction given current token
- $y_t$ = target (the actual next token representation)
- $\mathcal{L}$ = MSE loss: $\|f_\theta(x_t) - y_t\|^2$
- $\eta$ = inner learning rate (hyperparameter, typically $10^{-3}$ to $10^{-2}$)

**One gradient step per token** — not SGD over a dataset, just a single update.

The memory *literally learns* from each token it sees.

<!-- notes: This is remarkably simple. It's just one step of gradient descent, the same operation we've been doing since Phase 0. The difference is that we're doing it during inference, not training. The gradient tells the memory "here's what you predicted, here's what actually happened, adjust your weights to be more accurate next time." Over hundreds of tokens, the memory accumulates knowledge about the current conversation — names, facts, relationships — encoded in its weights rather than in explicit key-value pairs. -->

---

## The Titans Architecture: three memory systems

```
┌─────────────────────────────────────────────────────┐
│                    Titans Model                      │
│                                                     │
│  ┌─────────────┐  ┌──────────────┐  ┌────────────┐ │
│  │ Short-Term  │  │  Long-Term   │  │ Persistent │ │
│  │  (Attention) │  │(Neural Memory)│  │  (Frozen)  │ │
│  │             │  │              │  │            │ │
│  │ Standard    │  │ MLP with TTT │  │ Pre-trained│ │
│  │ self-attn   │  │ updates at   │  │ weights,   │ │
│  │ over recent │  │ inference    │  │ no updates │ │
│  │ tokens      │  │ time         │  │ at all     │ │
│  └──────┬──────┘  └──────┬───────┘  └─────┬──────┘ │
│         │                │                │        │
│         └────────┬───────┘                │        │
│                  │ (gated combination)     │        │
│                  ▼                         │        │
│         ┌──────────────┐                  │        │
│         │  Gate + Add  │←─────────────────┘        │
│         └──────────────┘                           │
│                  │                                  │
│                  ▼                                  │
│            Output h_t                               │
└─────────────────────────────────────────────────────┘
```

This mirrors human cognition: working memory (attention) + episodic memory (neural memory) + semantic knowledge (frozen weights).

<!-- notes: The three-component design is inspired by cognitive science. Working memory (attention) holds the last few hundred tokens in full fidelity — like your awareness of the current sentence. Episodic memory (neural memory) stores compressed facts from the entire conversation — like remembering that the user's name is Alice and she works at Google. Semantic memory (frozen weights) contains everything learned during pre-training — like knowing that Paris is the capital of France. Each system has different capacity, update rules, and access patterns. -->

---

## Surprise metric: when to write

Not every token deserves a memory update. The **surprise metric** gates writes:

**Prediction error**:

$$e_t = \|f_\theta(x_t) - y_t\|^2$$

**Gated update**:

$$\theta_{t+1} = \begin{cases} \theta_t - \eta \nabla_\theta \mathcal{L} & \text{if } e_t > \tau \text{ (surprised)} \\ \theta_t & \text{if } e_t \leq \tau \text{ (expected)} \end{cases}$$

Where $\tau$ is a threshold (learned or fixed).

**Intuition**: "The quick brown fox" → low surprise, skip. "The patient's blood type is AB-negative" → high surprise, memorize.

<!-- notes: This is computationally important. Without the surprise gate, you'd run a backward pass at every token — expensive. With it, you only update on ~10-30% of tokens (the surprising ones). Common words, filler, and predictable sequences are skipped. The threshold tau can be fixed (e.g., top 20% of error values) or learned as a parameter. In practice, the model learns to allocate its memory budget to the most informative tokens. This is similar to how human attention works — you tune out predictable speech and perk up when something unexpected is said. -->

---

## Momentum and decay: controlled forgetting

Raw gradient updates accumulate without bound. **Exponential moving average (EMA)** — a running average that weights recent values more than old ones — provides stability:

$$M_{t+1} = \beta M_t + (1-\beta) \Delta_t$$

$$\theta_{t+1} = \theta_t + M_{t+1}$$

Where:
- $M_t$ = momentum buffer (running average of gradients)
- $\Delta_t = -\eta \nabla_\theta \mathcal{L}$ (current update)
- $\beta$ = decay factor (typically 0.9–0.99)

**Why decay prevents unbounded growth**:
- Without decay ($\beta = 1$): weights drift arbitrarily far → instability
- With decay ($\beta = 0.95$): old updates fade with half-life $\approx \frac{\ln 2}{1 - \beta} \approx 14$ steps
- **Recent facts are weighted more** than ancient ones — natural forgetting

<!-- notes: This is the same momentum used in optimizers like Adam and SGD with momentum, but applied during inference. The decay factor beta controls the memory's "attention span." High beta (0.99) means the memory retains facts for hundreds of tokens. Low beta (0.9) means it forgets after ~14 tokens. In practice, you want different decay rates for different types of information — but a single beta in the 0.95 range works well as a starting point. The EMA also smooths noisy gradients, preventing the memory from overreacting to single unusual tokens. -->

---

## Memory as context: gated integration

The memory readout is **gated** and added to the attention output:

$$h_{\text{mem}} = f_\theta(x_t)$$

$$h_{\text{attn}} = \text{MultiHeadAttention}(x_t, \text{KV cache})$$

$$h = h_{\text{attn}} + \sigma(g) \cdot h_{\text{mem}}$$

Where:
- $g \in \mathbb{R}^{d_{\text{model}}}$ is a **learned gate vector**
- $\sigma(g)$ = sigmoid, element-wise in $[0, 1]$
- When $\sigma(g_i) \approx 0$: dimension $i$ uses only attention (short-term)
- When $\sigma(g_i) \approx 1$: dimension $i$ blends in memory (long-term)

The gate lets each dimension **choose** between local context and long-range memory.

<!-- notes: The gating mechanism is crucial. Without it, you'd always add the memory output, which could interfere with attention when short-term context is sufficient. The learned gate discovers which dimensions benefit from memory and which don't. In practice, the gate learns to activate memory for factual recall queries and suppress it for syntactic/grammatical decisions that depend only on local context. This is analogous to the gating in LSTMs and GRUs, but applied at the architecture level rather than the cell level. -->

---

## O(1) vs O(n) complexity comparison

| Metric | Standard KV Cache | Titans Neural Memory |
|--------|-------------------|---------------------|
| **Storage** per decode step | $O(n)$ — grows with context | $O(1)$ — fixed MLP size |
| **Compute** per decode step | $O(n)$ — attend over all keys | $O(1)$ — single forward pass |
| **Memory at 1k tokens** | 37 MB | 0.4 MB |
| **Memory at 16k tokens** | 604 MB | 0.4 MB |
| **Memory at 256k tokens** | 9.6 GB | **0.4 MB** |
| **Memory at 1M tokens** | 38.4 GB | **0.4 MB** |

The neural memory is a small MLP (~100K params = 0.4 MB in FP32). Its size is **independent** of how many tokens it has seen.

Trade-off: fixed capacity means the memory *must* forget — it can't store everything perfectly. The surprise metric and decay manage what's kept and what's dropped.

<!-- notes: This is the fundamental advantage of Titans. At a million tokens, the KV cache would need 38 GB — more than most GPUs have. The neural memory stays at 0.4 MB regardless. Of course, there's no free lunch — the neural memory has limited capacity and lossy compression. But for practical applications like long conversations, the trade-off is excellent. The model retains key facts (names, numbers, decisions) while gracefully forgetting filler and repeated information. In the hybrid architecture (attention + memory), you get the best of both worlds: perfect short-term recall from attention and compressed long-term storage from memory. -->

---

## Putting it all together: Phase 0 → Phase 8

| Phase | What We Built | Key Concept |
|-------|--------------|-------------|
| 0 | Tensors & autograd | Backpropagation from scratch |
| 1 | 80M-param transformer | Attention, FFN, LayerNorm |
| 2 | Chat fine-tuning | Instruction following, masking |
| 3 | RLHF / DPO | Human preference alignment |
| 4 | MoE routing | Sparse expert selection |
| 5 | TurboQuant KV cache | Rotation + quantization |
| 6 | Multimodal vision | Encoder-free patch projection |
| 7 | Full-duplex audio | Neural codec + dual-stream |
| **8** | **Titans memory** | **TTT + neural memory + surprise** |

From raw tensor operations to a **multimodal, memory-augmented, quantized, alignment-tuned LLM** — the 2026 frontier stack.

<!-- notes: Take a moment to appreciate the journey. In Phase 0, students couldn't multiply matrices. Now they have a model that can see images, hear speech, remember facts across arbitrarily long conversations, and fit on a consumer GPU. Every component builds on the previous ones — attention enables the transformer, the transformer enables chat, chat enables alignment, alignment enables safe deployment, quantization enables long context, and memory enables infinite context. This is the complete stack that production LLMs use today. -->

---

## Capstone evaluation checklist

| Test | Criterion | Pass Condition |
|------|-----------|---------------|
| **Load checkpoint** | All Phase 1–7 components present | Model loads without error |
| **Write to memory** | Feed 5,000-token document with 10 facts | Surprise metric triggers ≥8 writes |
| **Retrieve facts** | Query each fact after 5,000 more filler tokens | ≥7/10 correct retrievals |
| **VRAM budget** | Profile peak GPU memory during retrieval | **< 10 GB** on RTX 3080 |
| **O(1) decode** | Measure decode time at 1k, 5k, 10k context | Time constant ±10% |
| **Momentum decay** | Verify old facts fade after sufficient new input | Graceful degradation, not cliff |

Students must pass **all six** tests to complete the course.

<!-- notes: The capstone is designed to be integrative. Loading the checkpoint verifies all previous phases work together. The memory write test checks that the surprise metric correctly identifies important tokens. The retrieval test is the real exam — can the model recall facts that are no longer in the attention window? The VRAM test ensures the implementation is practical, not just theoretically correct. The O(1) decode test verifies the fundamental scaling property. And the decay test checks that the memory degrades gracefully rather than catastrophically. -->

---

## Dataset

**Extended Complex Fact Associations** — 20k+ word scenarios with interlinked facts.

Example scenario:
- "Dr. Sarah Chen works at Memorial Hospital in Boston."
- [5,000 words of filler narrative]
- "Who does Dr. Chen work for?" → "Memorial Hospital"
- [5,000 more words]
- "Where is Dr. Chen's workplace located?" → "Boston"

Facts require **multi-hop reasoning** across the memory.

<!-- notes: The dataset is specifically designed to test memory, not just language modeling. The facts are interlinked so that answering some questions requires combining information from multiple memory writes. The filler text ensures the facts are long out of the attention window by the time they're queried. This is a harder test than Needle-in-a-Haystack because it requires not just retrieval but reasoning over retrieved facts. -->

---

## Lab map

| Lab | Topic |
|-----|-------|
| 8.1 | TTT math |
| 8.2 | Neural memory module |
| 8.3 | Surprise metric |
| 8.4 | Momentum & decay |
| 8.5 | Memory + attention |
| 8.6 | VRAM profiling |

<!-- notes: Lab 8.1 implements the basic TTT update rule and verifies gradient computation. Lab 8.2 builds the neural memory MLP and tests memorization on simple sequences. Lab 8.3 adds the surprise gate and verifies selective writing. Lab 8.4 implements EMA momentum with decay and tests forgetting curves. Lab 8.5 integrates memory with the full transformer via gated addition. Lab 8.6 profiles VRAM across sequence lengths and demonstrates O(1) scaling. -->

---

## Key takeaways

1. **KV cache grows linearly** with context — unsustainable for long conversations
2. **Test-Time Training** turns inference into a learning process: one gradient step per surprising token
3. **Neural memory** stores compressed facts in MLP weights — **O(1) space** regardless of context length
4. **Surprise gating** saves compute by only updating on informative tokens
5. **Momentum + decay** provide controlled forgetting — recent facts weighted over old ones
6. The **Titans architecture** combines attention (short-term), memory (long-term), and frozen weights (permanent knowledge)

<!-- notes: This phase represents the cutting edge of LLM architecture research. The ideas here — test-time training, neural memory, surprise-gated updates — are actively being explored by Google, Meta, and others. Students who understand this material are prepared to read and contribute to frontier research papers. The combination of all eight phases gives a complete picture of how modern LLMs work, from the lowest level (tensor operations) to the highest level (memory-augmented multimodal reasoning). -->

---

## Bridge: Course Capstone

This is the end of the road — so let's connect the whole journey, **Phase 0 → Phase 8**.

- **Phase 0–1:** you learned *gradient descent* and built *attention* — the engine and the short-term memory.
- **Phase 2–3:** you made the model *follow instructions* and *align* to human preferences — turning a predictor into an assistant.
- **Phase 4:** *Mixture-of-Experts* let it scale capacity without scaling cost.
- **Phase 5:** *TurboQuant* compressed the KV cache so long contexts fit on small GPUs.
- **Phase 6–7:** you tokenized *vision* and *audio*, proving the transformer is a universal sequence engine.
- **Phase 8:** *Titans* adds long-term memory that **learns at test time**, so the model can remember forever in O(1) space.

Every phase fed the next: gradients → attention → chat → alignment → scale → efficient memory → new senses → living memory. **You now hold the complete 2026 frontier stack.**

<!-- notes: This is the emotional and intellectual payoff of the entire course. Walk students back through the dependency chain explicitly so they see it was never a pile of disconnected tricks — each phase was a prerequisite for the next. Gradient descent (Phase 0) is literally the mechanism that powers Titans' test-time training (Phase 8), closing the loop. End on empowerment: they didn't just learn about LLMs, they built every layer of one. -->

---

## Capstone

You built an 80M stack from tensors to **live-updating memory** — the 2026 frontier.

Congratulations. You now understand, at implementation depth, every major component of a modern LLM.

<!-- notes: This is a genuine accomplishment. Most ML engineers use LLMs as black boxes. Students in this course have built every component from scratch: autograd, attention, training loops, alignment, quantization, multimodal processing, audio generation, and neural memory. They can debug, modify, and improve any part of the stack. That's the difference between using a tool and understanding a tool. -->

---

## Further Reading (Optional)

**These papers are optional enrichment — you do NOT need to read any of them to continue the course.**

- Graves, Wayne, & Danihelka (2014). *Neural Turing Machines*. arXiv:1410.5401.
- Finn, Abbeel, & Levine (2017). *Model-Agnostic Meta-Learning for Fast Adaptation of Deep Networks (MAML)*. ICML.
- Katharopoulos et al. (2020). *Transformers are RNNs: Fast Autoregressive Transformers with Linear Attention*. ICML.
- Gu & Dao (2023). *Mamba: Linear-Time Sequence Modeling with Selective State Spaces*. arXiv:2312.00752.
- Sun et al. (2024). *Learning to (Learn at Test Time): RNNs with Expressive Hidden States*. arXiv:2407.04620.
- Behrouz, Zhong, & Mirrokni (2024). *Titans: Learning to Memorize at Test Time*. arXiv:2501.00663.

<!-- notes: This is the final reading list of the whole course, so frame it as a launchpad into research, not an assignment. Neural Turing Machines is the ancestor of learnable memory; MAML is the meta-learning foundation behind test-time training; Linear Attention and Mamba are the efficient-sequence-model lineage; the Sun et al. TTT paper and the Behrouz et al. Titans paper are the direct sources for everything we built this phase. A motivated student can now read Titans end to end and recognize every idea. All optional. -->

