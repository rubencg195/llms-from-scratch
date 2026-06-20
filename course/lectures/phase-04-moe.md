---
title: "Phase 4: Mixture of Experts"
subtitle: "Smarter Without Slower"
author: "LLMs From Scratch"
---

# Phase 4: Mixture of Experts (MoE)

## Route tokens to specialists — one expert fires per token

<!-- notes: MoE is one of the most important architectural innovations in modern LLMs. Mixtral 8x7B showed that a model with 46B total parameters but only 12B active per token can match or beat dense 70B models at much lower inference cost. We're implementing the same idea at our 80M scale. -->

---

## Learning objectives

- Understand the **scaling problem** with dense models
- Build a **router** with softmax probabilities
- Instantiate **4 expert** feed-forward networks
- Implement **load-balancing** loss to prevent collapse
- Track **specialization** across topics
- Understand the **total vs active** parameter distinction

<!-- notes: By the end of Phase 4, students will have a model with 4× the parameters of the Phase 1 model but the same inference cost per token. The router learns to send math tokens to one expert and story tokens to another. This is conditional computation — and it's the future of efficient scaling. -->

---

## The scaling problem

Dense models are **wasteful**: every parameter activates for every token.

```
Dense model (80M params):
  "2 + 2 = ?"     → all 80M params activate
  "Once upon..."  → all 80M params activate (same ones!)

Question: Does a math token really need the "fairy tale" parameters?
```

**The inefficiency:**
- A model trained on diverse data stores diverse knowledge
- But each token only needs a **tiny fraction** of that knowledge
- Dense models pay full compute cost regardless of token difficulty

**Empirical evidence:**
- Simple tokens ("the", "and") activate similar patterns to complex tokens ("eigenvalue", "mitochondria")
- Most neurons fire weakly for most inputs — **sparse activation is natural**

<!-- notes: This slide motivates the entire MoE approach. Ask students: when YOU answer a math question, do you use the same brain regions as when you write a story? No — different tasks activate different neural pathways. Dense models don't have this luxury; every weight participates in every computation. MoE introduces specialization, which is more brain-like and more efficient. The sparse activation observation comes from work on lottery ticket hypothesis and neural network pruning. -->

---

## Sparse conditional computation

**Key idea:** only activate the parameters **relevant** to each token.

```
Dense FFN (Phase 1):
  ┌─────────────────────────────┐
  │   Every token → FFN         │  80M params active, 80M total
  │   (all parameters fire)     │
  └─────────────────────────────┘

MoE FFN (Phase 4):
  ┌─────────────────────────────┐
  │   Token → Router → Expert₂  │  80M params active, 320M total
  │   (only 1 of 4 experts)     │
  └─────────────────────────────┘
```

**Benefits:**
- **4× parameters** = 4× knowledge capacity
- **1× compute** per token = same inference speed
- **1× memory** per forward pass = same VRAM for activations

**The catch:** you need 4× memory to **store** all expert weights (but only 1× to **run** each token).

<!-- notes: This is the fundamental tradeoff of MoE. You get more capacity (more total parameters, more knowledge storage) without paying more compute per token. But you DO need more memory to store the weights. In our setting with an RTX 3080 (10GB), we can fit 4 experts because each expert is the same size as the original FFN, and we only need one expert's activations in memory at a time. In data center settings, experts can be spread across multiple GPUs (expert parallelism). -->

---

## The router — how tokens choose experts

The router is a simple linear layer followed by softmax:

$$p = \text{softmax}(W_r \cdot h + b_r)$$

where $h$ is the token's hidden state and $p \in \mathbb{R}^{N_{\text{experts}}}$.

**Example** (4 experts):

```
Token hidden state h = [0.5, -0.3, 0.8, ...]  (d_model = 512)
                ↓
Router linear: W_r ∈ ℝ^(512 × 4)
                ↓
Logits:        [2.1,  0.3,  -0.5,  1.2]
                ↓
Softmax:       [0.52, 0.09,  0.04, 0.21]  ← probability distribution
                ↓
Top-1:         Expert 0 selected (p=0.52)
                ↓
Output:        y = p₀ · Expert₀(h) = 0.52 · Expert₀(h)
```

**Top-1 vs Top-2 routing:**

| Strategy | Active experts | Compute | Used by |
|----------|---------------|---------|---------|
| Top-1 | 1 per token | 1× FFN | Switch Transformer |
| Top-2 | 2 per token | 2× FFN | Mixtral, GShard |

We use **top-1** for simplicity and to match our VRAM budget.

<!-- notes: The router is deceptively simple — just a linear layer and argmax. But its behavior is critical. If it learns a good routing policy, each expert specializes and the model becomes more capable. If it degenerates (always routing to the same expert), the model is no better than a dense model with wasted parameters. Top-2 routing gives better quality because each token gets two "opinions," but costs 2× the FFN compute. Mixtral uses top-2 with 8 experts, so 2 of 8 fire per token. We use top-1 with 4 experts for simplicity. -->

---

## Expert architecture

Each expert is a **full FFN** — identical in structure to the Phase 1 FFN:

```
Expert_i(x) = GELU(x · W₁ⁱ + b₁ⁱ) · W₂ⁱ + b₂ⁱ

Where W₁ⁱ ∈ ℝ^(d_model × d_ff), W₂ⁱ ∈ ℝ^(d_ff × d_model)
```

**MoE block replaces the dense FFN in each Transformer layer:**

```
Before (Phase 1 dense):
  LayerNorm → Attention → Residual → LayerNorm → [FFN] → Residual
                                                   ↑
                                              one FFN for all

After (Phase 4 MoE):
  LayerNorm → Attention → Residual → LayerNorm → [Router → Expert_k] → Residual
                                                   ↑
                                              router picks 1 of 4

Each expert:
  ┌──────────────────────────────────────┐
  │  x (512) → W₁ (512×2048) → GELU    │
  │         → W₂ (2048×512)  → output   │
  └──────────────────────────────────────┘
  Parameters per expert: 512×2048 + 2048×512 = 2.1M
  Total for 4 experts: 8.4M per layer
  Dense FFN was: 2.1M per layer
```

**Experts share no weights** — each develops independent specialization.

<!-- notes: A common misconception is that MoE experts have special architectures. They don't — each expert is a vanilla FFN, identical to what Phase 1 used. The magic is in the routing, not in the expert design. The parameter count scales linearly with the number of experts. With 4 experts per layer and 8 layers, we go from 8×2.1M=16.8M FFN params to 8×8.4M=67.2M FFN params. Adding the non-FFN parameters (attention, embedding, etc.), total model size roughly triples. -->

---

## Load balancing — the critical loss term

**Without balancing**, the router degenerates:

```
Step 0:    Expert 0 gets slightly better gradients
Step 100:  Expert 0 gets 40% of tokens (more training → gets better)
Step 1000: Expert 0 gets 90% of tokens (snowball effect)
Step 5000: Expert 0 gets 99% — other experts are untrained dead weight
```

This is called **expert collapse** — and it ruins the model.

<!-- notes: Expert collapse is the #1 failure mode in MoE training. It happens because of a positive feedback loop: a slightly better expert gets more tokens, which gives it more gradients, which makes it even better, which attracts even more tokens. It's like a restaurant review system where one restaurant with a slight early lead gets all the customers while equally good restaurants languish. The solution is an auxiliary loss that explicitly encourages balance. -->

---

## The auxiliary load-balancing loss

From the **Switch Transformer** paper (Fedus et al., 2021):

$$\mathcal{L}_{\text{balance}} = \alpha \cdot N \cdot \sum_{i=1}^{N} f_i \cdot P_i$$

Where:
- $N$ = number of experts (4 in our case)
- $f_i$ = **fraction** of tokens routed to expert $i$ in this batch
- $P_i$ = **mean router probability** assigned to expert $i$ in this batch
- $\alpha$ = balancing coefficient (typically 0.01)

**Perfectly balanced:** $f_i = P_i = 1/N$ for all $i$

$$\mathcal{L}_{\text{balance}}^{\text{ideal}} = \alpha \cdot N \cdot N \cdot \frac{1}{N} \cdot \frac{1}{N} = \alpha$$

**Collapsed (all tokens → expert 0):** $f_0 = 1, P_0 \approx 1$

$$\mathcal{L}_{\text{balance}}^{\text{collapsed}} = \alpha \cdot N \cdot (1 \cdot 1 + 0 + 0 + 0) = \alpha \cdot N$$

The collapsed case has $N\times$ higher loss → gradient pushes toward balance.

**Total training loss:**

$$\mathcal{L} = \mathcal{L}_{\text{LM}} + \mathcal{L}_{\text{balance}}$$

<!-- notes: Walk through the math carefully. The f_i · P_i product is clever: f_i measures actual routing decisions (hard, non-differentiable), P_i measures router probabilities (soft, differentiable). The gradient flows through P_i, pushing the router to spread probability mass more evenly. The α coefficient controls how much we care about balance vs language modeling quality — too high and we sacrifice quality for perfect balance, too low and collapse still happens. 0.01 is a good default. -->

---

## Expert collapse — what it looks like

**Healthy routing** (tokens per expert per batch of 1000):

```
Expert 0: ████████████████████████ 260 tokens (26%)
Expert 1: ██████████████████████   230 tokens (23%)
Expert 2: ████████████████████████ 250 tokens (25%)
Expert 3: ████████████████████████ 260 tokens (26%)
```

**Collapsed routing** (without load balancing):

```
Expert 0: ████████████████████████████████████████████████ 950 tokens (95%)
Expert 1: █ 20 tokens (2%)
Expert 2: █ 15 tokens (1.5%)
Expert 3: █ 15 tokens (1.5%)
```

**Symptoms:**
- Loss plateaus early (only 1 expert is learning)
- Token diversity per expert drops (expert sees same patterns)
- Model quality degrades to worse than dense baseline
- 75% of parameters are wasted (untrained dead experts)

<!-- notes: Show real training curves from a collapsed run in the lab. The loss curve is diagnostic — it plateaus much higher than a balanced run. Students should monitor expert token counts during training (we log this in lab 4.3). If any expert consistently gets <10% or >40% of tokens, the balancing coefficient α needs adjustment. Healthy routing doesn't need to be perfectly uniform, but should be roughly balanced with natural variation based on input distribution. -->

---

## Total vs active parameters

This is the key insight of MoE: **capacity without cost**.

```
                      Total params    Active params/token    Inference FLOPS
Dense (Phase 1):      80M              80M                    1×
MoE 4-expert:         ~250M            ~80M                   ~1×
```

**How it works:**

```
For each token:
  1. Router selects 1 expert     (tiny compute: 512 × 4 = 2048 ops)
  2. Token passes through that   (same as dense FFN: ~2.1M ops)
     expert only
  3. Other 3 experts: idle       (no computation, no memory bandwidth)

Total stored weights:  250M  (all 4 experts in GPU memory)
Weights used per token: 80M  (attention + 1 expert + embedding)
```

**Analogy:** A hospital has 100 doctors (total), but each patient sees only 1-2 specialists (active). The hospital's total knowledge is 100 doctors' worth, but the cost per patient visit is only 1-2 doctors' time.

<!-- notes: This distinction trips up many students. The model FILE is 4× larger (you need memory to store all experts), but the inference COMPUTE per token is the same as a dense model. This means: (1) inference speed is roughly the same, (2) training requires more GPU memory to hold all expert weights, (3) the model has more capacity to learn and store knowledge. In our RTX 3080 setting, we can store all 4 experts in VRAM because each is only ~2.1M parameters. At Mixtral scale (8×7B), experts are distributed across GPUs. -->

---

## Real-world MoE models

| Model | Experts | Top-k | Total params | Active params | Key innovation |
|-------|---------|-------|-------------|---------------|----------------|
| **Switch Transformer** (Google, 2021) | 128 | 1 | 1.6T | 12.5B | Simplified routing |
| **GShard** (Google, 2020) | 2048 | 2 | 600B | ~1.5B | Distributed experts |
| **Mixtral 8×7B** (Mistral, 2023) | 8 | 2 | 46.7B | 12.9B | Open-source MoE |
| **DeepSeek-V2** (2024) | 160 | 6 | 236B | 21B | Fine-grained experts |
| **Our model** | 4 | 1 | ~250M | ~80M | Educational |

**Trends:**
- Industry is moving toward **more, smaller experts** (DeepSeek-V2: 160 experts)
- Top-2 is becoming standard (but top-1 is simpler to implement)
- Expert parallelism distributes experts across GPUs in large clusters

<!-- notes: These numbers help students see where our toy model fits in the landscape. We use 4 experts with top-1 routing, which is the simplest possible MoE. Mixtral's 8×7B showed that MoE can be competitive with much larger dense models (Mixtral matches LLaMA 2 70B while using only 12.9B active params). The efficiency gain is real and measurable. DeepSeek-V2 pushes this further with fine-grained experts — each expert is smaller but there are many more of them, allowing finer-grained specialization. -->

---

## Datasets and routing specialization

**Training data:**

**OpenWebText** (broad topics) + **Glaive** (code/tools) — natural separation helps routing.

**What we hope to see:**

```
Expert 0: Specializes in narrative/story tokens
Expert 1: Specializes in math/arithmetic tokens
Expert 2: Specializes in code/tool tokens
Expert 3: General-purpose (function words, punctuation)
```

**How we track specialization** (Lab 4.4):
- Log which expert each token routes to
- Aggregate by **topic** (classify tokens by source dataset)
- Build a confusion matrix: `expert × topic`
- Measure **mutual information** between routing and topic

**Reality check:** With only 4 experts, specialization is coarse. With 128 experts (Switch), individual experts can specialize in things like "French cooking vocabulary" or "legal terminology."

<!-- notes: The specialization tracking lab is one of the most interesting in the course. Students get to SEE whether their router learned something meaningful. In practice, with 4 experts, you often see one expert grabbing function words (the, and, is), one handling content words from stories, one handling numbers and math operators, and one as a generalist. The assignment between experts and topics isn't controlled — it emerges naturally from the training data distribution. This is unsupervised specialization, and it's quite remarkable when it works. -->

---

## Lab map

| Lab | Topic | What you build |
|-----|-------|----------------|
| 4.1 | Router + softmax | Linear router with temperature |
| 4.2 | Four expert sub-networks | MoE layer replacing dense FFN |
| 4.3 | Load balancing | Auxiliary loss + monitoring |
| 4.4 | Specialization tracking | Expert-topic confusion matrix |

**Deliverable:** `phase4_moe.pt` — a Mixture of Experts model with 4× capacity.

<!-- notes: Lab 4.1 starts simple — just the router in isolation, tested on synthetic data to verify it can learn to route different inputs to different experts. Lab 4.2 integrates the router with actual expert FFNs inside the Transformer block. Lab 4.3 adds the balancing loss and monitoring — students should watch the token counts per expert converge to roughly uniform. Lab 4.4 is analysis: after training, examine what each expert learned to specialize in. -->

---

## Implementation notes for RTX 3080

| Concern | Solution |
|---------|----------|
| 4× FFN memory | Each expert is only 2.1M params; total MoE fits in 10GB |
| Router overhead | Negligible — one small linear layer |
| Batch routing | All tokens in a batch can route to different experts |
| Gradient flow | Router gradients + expert gradients + balance loss gradients |

**VRAM budget:**

```
Token embedding:         4.1M  ×  2 bytes = 8.2 MB
Attention (8 layers):    8.4M  ×  2 bytes = 16.8 MB
MoE FFN (8 layers, 4 exp): 67.2M × 2 bytes = 134.4 MB
Other (LN, router, LM head): ~5M × 2 bytes = 10 MB
Optimizer states (AdamW):     ~3× model    = ~500 MB
Activations + gradients:                   = ~2-4 GB
                                     Total ≈ ~3-5 GB  ✓
```

Fits comfortably in 10 GB VRAM.

<!-- notes: Walk through the VRAM calculation to show students that despite having 4× the FFN parameters, the model still fits on our GPU. The key insight is that MoE at small scale is almost free — the memory overhead is manageable. At large scale (Mixtral with 8×7B experts = 56B total), you need expert parallelism across multiple GPUs. Our educational setting is much simpler. Encourage students to monitor nvidia-smi during training to see actual VRAM usage. -->

---

## Next

**Phase 5:** TurboQuant — compress the KV cache to 3.5 bits.

We'll tackle the **other** memory bottleneck: not the model weights (Phase 3 handled that), but the attention cache that grows with every generated token.

Preview:
- KV cache grows linearly with sequence length
- Quantize cached keys and values on-the-fly
- Mixed-precision: important heads get more bits

<!-- notes: Phase 5 is where we start combining ideas from previous phases. We used quantization in Phase 3 for weights; now we apply similar ideas to the KV cache. We used attention in Phase 1; now we need to compress its memory footprint. The course is designed so that each phase builds on all previous phases, creating a deep understanding of how these techniques interact. -->
