---
title: "Phase 2: Instruction Tuning & Tool Calling"
subtitle: "Teaching the Model to Obey"
author: "LLMs From Scratch"
---

# Phase 2: Instruction Tuning & Tool Calling

## From babbling to useful answers

Train the Phase 1 model to follow chat format and emit JSON tool calls.

<!-- notes: Phase 2 is the transformation point. Our Phase 1 model can write children's stories, but it can't answer questions, follow instructions, or use tools. Instruction tuning is what turns a base language model into an assistant. This is the same process (conceptually) that turns GPT base into ChatGPT. -->

---

## Before You Begin (Prerequisites)

Everything here was built in **Phase 1** (and rests on **Phase 0**) — no outside knowledge needed.

- **A trained base Transformer** — the `phase1_80m.pt` checkpoint you produced in Phase 1. We fine-tune *that exact model*.
- **Next-token prediction & cross-entropy loss** — how the model is trained (Phase 0 + Phase 1). We only adjust *which* tokens get scored.
- **Tokenization** — text becomes integer IDs via BPE (Phase 1). Here we add a few new special tokens for chat roles.
- **The training loop** — forward, loss, `backward()`, optimizer step (Phase 0). Identical here, just with a mask.
- **Basic Python & JSON** — dictionaries and lists are enough to follow tool calling.

> Nothing in this phase changes the model's architecture. If you trained the Phase 1 model, you are ready.

<!-- notes: Reassure students that Phase 2 reuses everything: the architecture is untouched, the loss is the same cross-entropy, and the training loop is the Phase 0/1 loop plus a mask. The only genuinely new ideas — chat templates and masked loss — are introduced from scratch in the upcoming slides. JSON is just Python dicts. -->

---

## Learning objectives

- Design **chat templates** with `<|Thought|>` tags
- Apply **masked loss** (grade only assistant tokens)
- Fine-tune on **GSM8K** step-by-step math
- Output valid **Python dict / JSON** for tools
- Handle **multi-turn** conversations with accumulated context

<!-- notes: By the end of this phase, students will have a model that can: (1) answer questions in a structured format, (2) show its reasoning in thought tags, (3) solve elementary math problems step by step, and (4) emit structured JSON when it needs to call a tool like a calculator. -->

---

## Base LM vs Instruct LM

**Base model** (Phase 1) — continues text:

```
Prompt:  "What is 2 + 2?"
Output:  "What is 2 + 3? What is 2 + 4? Here are more
          math problems for your worksheet..."
```

**Instruct model** (Phase 2) — answers questions:

```
Prompt:  "What is 2 + 2?"
Output:  "<|Thought|>I need to add 2 and 2. 2 + 2 = 4.<|/Thought|>
          The answer is 4."
```

The architecture is **identical**. The only difference is **what data we train on** and **which tokens we compute loss on**.

<!-- notes: This side-by-side comparison is the "aha moment" for many students. The base model isn't wrong — it's doing exactly what it was trained to do: predict likely next tokens given the context. "What is 2 + 2?" in a children's story context is likely followed by more questions. The instruct model learned a different pattern: questions are followed by answers. Same weights, different training data, wildly different behavior. -->

---

## The chat template

We structure conversations using special tokens:

```
<|system|>You are a helpful math tutor.<|/system|>
<|user|>What is 15 × 7?<|/user|>
<|assistant|>
<|Thought|>
I need to multiply 15 by 7.
15 × 7 = 105
<|/Thought|>
The answer is 105.
<|/assistant|>
```

**Token-level view** (simplified IDs):

```
Tokens:  [SYS] You are... [/SYS] [USR] What is... [/USR] [AST] [THK] I need... [/THK] The answer... [/AST]
IDs:     8001  ...         8002   8003  ...        8004   8005  8006  ...       8007   ...           8008
Mask:    0     0           0      0     0          0      1     1     1         1      1             1
```

Only tokens with **mask = 1** (assistant turns) contribute to the loss.

<!-- notes: The special tokens ([SYS], [USR], etc.) are added to the tokenizer vocabulary. They never appeared in TinyStories, so their embeddings start random and are learned during fine-tuning. The mask is critical — without it, the model would be trained to predict the user's questions, which is not what we want. We want it to learn to produce good answers given questions. -->

---

## The Thought Tag paradigm

**Chain-of-Thought (CoT)** reasoning via structured tags:

```
<|Thought|>
  Step 1: Identify the operation (multiplication)
  Step 2: Break it down: 15 × 7 = (10 × 7) + (5 × 7)
  Step 3: 70 + 35 = 105
<|/Thought|>
```

**Why tags instead of free-form "Let me think..."?**

- **Parseable:** code can extract reasoning vs. final answer
- **Trainable:** we can selectively mask/weight thought tokens
- **Inspectable:** users (or evaluators) can audit the reasoning chain

**Connection to frontier models:**
- OpenAI o1/o3: hidden chain-of-thought before responding
- DeepSeek-R1: `<think>...</think>` tags, trained with RL
- Our approach: same concept, simpler implementation with supervised fine-tuning

<!-- notes: The thought tag approach is a simplified version of what DeepSeek-R1 calls "thinking tokens." In R1, the model was trained with reinforcement learning to produce useful reasoning in think tags. We use supervised fine-tuning with GSM8K solutions as the ground-truth reasoning chains. The key insight is the same: giving the model a "scratchpad" dramatically improves accuracy on multi-step problems because the model can break complex reasoning into sequential steps. -->

---

## Masked loss — visual explanation

The loss function only penalizes **assistant** tokens:

```
Tokens:   <|user|>  What  is  2+2  ?  <|assistant|>  The  answer  is  4  .

Role:     [  user  ] [  user  ] ...    [ assistant ] [    assistant    ]

Loss
weight:    0     0    0    0   0   0       1         1     1      1  1  1

            ╰── ignored ──╯                ╰─── gradients flow here ───╯
```

**Implementation:**

```python
labels = input_ids.clone()
labels[user_mask] = -100        # PyTorch ignores -100 in cross_entropy
loss = F.cross_entropy(logits.view(-1, V), labels.view(-1), ignore_index=-100)
```

Here `logits` are the model's raw, pre-probability scores for each possible next token. Setting user token labels to `-100` is the standard PyTorch convention for "don't compute loss here."

<!-- notes: This is deceptively simple but incredibly important. Without masking, the model would spend half its capacity learning to mimic user questions — which is useless for an assistant. The -100 trick is a PyTorch convention: F.cross_entropy skips any target with this value. In practice, you build the mask by tracking which tokens came from which role during template formatting. -->

---

## GSM8K dataset deep dive

**GSM8K** = Grade School Math 8K — 8,500 elementary math problems with step-by-step solutions.

**Example problem:**

```
Q: Natalia sold clips to 48 of her friends in April, and then she sold
   half as many clips in May. How many clips did Natalia sell altogether
   in April and May?

A: Natalia sold 48/2 = <<48/2=24>>24 clips in May.
   Natalia sold 48+24 = <<48+24=72>>72 clips altogether in April and May.
   #### 72
```

**Key features:**
- `<<expression=result>>` annotations mark calculator steps
- `#### 72` marks the **final numerical answer** (used for eval)
- Solutions are **2-8 steps** — perfect for our small model

**We reformat for our template:**

```
<|user|>Natalia sold clips to 48 of her friends in April...<|/user|>
<|assistant|>
<|Thought|>
Step 1: Clips in May = 48 / 2 = 24
Step 2: Total = 48 + 24 = 72
<|/Thought|>
Natalia sold 72 clips altogether.
<|/assistant|>
```

<!-- notes: GSM8K is a gold-standard benchmark for testing mathematical reasoning in LLMs. The problems require genuine multi-step reasoning, not just pattern matching. Our 80M model won't achieve GPT-4 level accuracy (92%), but it can learn the FORMAT of step-by-step reasoning, which is the pedagogical goal. Expect around 10-25% accuracy on held-out GSM8K with our tiny model — still impressive for 80M parameters! The #### marker is only used during evaluation to extract the final answer for automated scoring. -->

---

## Function calling schema

The model learns to emit structured JSON when it needs an external tool:

**Tool registry** (provided in system prompt):

```json
{
  "tools": [
    {
      "name": "calculator",
      "description": "Evaluate a math expression",
      "parameters": {
        "expression": {"type": "string", "description": "e.g. '12 * 8'"}
      }
    }
  ]
}
```

**Model output:**

```json
{"name": "calculator", "arguments": {"expression": "48 / 2"}}
```

**The executor loop:**

```
User asks question
    → Model emits JSON tool call
        → Python executor parses JSON
            → Runs calculator("48 / 2") → 24
                → Result fed back as new message
                    → Model generates final answer using result
```

<!-- notes: This is a simplified version of what OpenAI calls "function calling" and what Anthropic calls "tool use." The key challenge for training is teaching the model to emit VALID JSON — one missing quote or bracket and the executor fails. We train on the Glaive function calling dataset which provides thousands of examples of well-formatted tool calls. In practice, our small model will sometimes produce malformed JSON; we'll discuss error handling strategies in the lab. -->

---

## Multi-turn conversations

Real conversations have **multiple exchanges**. Context accumulates:

```
<|user|>What is 15 × 7?<|/user|>
<|assistant|>
<|Thought|>15 × 7 = 105<|/Thought|>
The answer is 105.
<|/assistant|>
<|user|>Now divide that by 3.<|/user|>
<|assistant|>
<|Thought|>"That" refers to 105. 105 / 3 = 35.<|/Thought|>
105 ÷ 3 = 35.
<|/assistant|>
```

**Challenges:**
- **Context window:** all prior turns must fit in 256 tokens (our block size)
- **Coreference:** "that," "it," "the previous answer" require understanding history
- **Loss masking:** must correctly mask ALL user turns, not just the first

**In training:** we concatenate full conversations and apply the role mask across the entire sequence.

<!-- notes: Multi-turn is where things get tricky. The model sees the entire conversation history as one long sequence, and the mask must correctly identify which parts are user (mask=0) and which are assistant (mask=1). With our 256 token block size, we can typically fit 2-3 turns before running out of context. In production systems with 8k-128k context windows, multi-turn conversations can go on for pages. -->

---

## Datasets and data mixing

| Dataset | Role | % of training |
|---------|------|---------------|
| **GSM8K** | Chain-of-thought arithmetic | ~70% |
| **Glaive Function Calling** | JSON tool call format | ~30% |

**Why mix?** A model trained only on math forgets how to call tools. A model trained only on tools can't reason about math.

**Data mixing strategy:**

```python
for step in range(total_steps):
    if random.random() < 0.7:
        batch = next(gsm8k_loader)      # math reasoning
    else:
        batch = next(glaive_loader)      # tool calling
```

The 70/30 ratio is tuned empirically in the labs — students experiment with different ratios.

<!-- notes: Data mixing is a recurring theme in ML. The ratio matters more than you might think — 90/10 works poorly for tool calling, 50/50 makes math performance worse. The 70/30 sweet spot was found by the curriculum designers through experimentation. In production LLMs, data mixing involves dozens of datasets with carefully tuned ratios, often using techniques like data ablation studies and scaling laws. -->

---

## Training details

| Hyperparameter | Value | Notes |
|----------------|-------|-------|
| Base checkpoint | `phase1_80m.pt` | Start from pretrained |
| Learning rate | 2e-5 | 10× smaller than pretraining |
| Epochs | 3 | Over combined dataset |
| Batch size | 8 | Same VRAM as Phase 1 |
| Warmup | 100 steps | Short (weights already good) |

An **epoch** is one full pass over the training dataset; *3 epochs* means the model sees every example three times.

**Why lower learning rate?**
- Pretrained weights are already useful — large LR would destroy them
- We want to **add** instruction-following ability, not **replace** language knowledge
- This is the difference between pretraining (learning language) and fine-tuning (learning behavior)

<!-- notes: The learning rate difference between pretraining and fine-tuning is one of the most important practical concepts. If you use the same 3e-4 learning rate from Phase 1, you'll get catastrophic forgetting — the model will learn to format responses but forget how to write coherent English. The 2e-5 rate lets the model gently adjust its behavior while preserving its language modeling capabilities. -->

---

## Post-training: SFT is not the end (Lab 2.5)

ChatGPT-style models go through **three stages**:

```
Base LM (Phase 1)  →  SFT (Labs 2.1–2.3)  →  Preference alignment (Lab 2.5)
   predicts text       follows format           prefers helpful answers
```

**SFT** imitates expert demonstrations (GSM8K solutions, tool-call examples).

**DPO (Direct Preference Optimization)** learns from pairs:
- **Chosen:** concise, correct, safe answer
- **Rejected:** verbose, wrong, or unhelpful answer

One loss pushes the model toward `chosen` relative to a frozen SFT copy — no reward model or PPO required for learning purposes.

<!-- notes: Karpathy's Deep Dive into LLMs explains this pipeline for beginners. SFT teaches skills; alignment teaches judgment. We skip full RLHF/PPO complexity and teach DPO as the accessible entry point. DeepSeek-R1 adds RL on verifiable math rewards — optional reading for students who want the frontier story. -->

---

## VRAM note

Same 80M backbone — fine-tuning uses **less** memory than pretraining:

- **Shorter sequences:** reformatted data typically < 200 tokens
- **Smaller learning rate:** optimizer states stay small
- **No new parameters:** we're reusing the Phase 1 architecture

```
VRAM comparison:
  Phase 1 (pretraining):  ~4-6 GB  (batch=8, seq=256)
  Phase 2 (fine-tuning):  ~3-4 GB  (batch=8, seq=200 avg)
```

<!-- notes: Fine-tuning efficiency is one reason why instruction tuning is so popular. You take a huge pretrained model (which cost millions to train) and fine-tune it with a tiny dataset (8k examples) for a few epochs. The compute cost goes from months on clusters to hours on a single GPU. This democratization of fine-tuning is why there are so many "chat" variants of open models. -->

---

## Lab map

| Lab | Topic | What you build |
|-----|-------|----------------|
| 2.1 | Chat templates | Token formatter with role markers |
| 2.2 | Masked loss | Role-aware loss masking |
| 2.3 | GSM8K training | CoT fine-tuning with thought tags |
| 2.4 | JSON tool output | Tool call + executor loop |
| 2.5 | Preference alignment | DPO loss on chosen vs rejected pairs |

**Deliverable:** `phase2_instruct.pt` — an instruction-following model that can reason about math and call tools.

<!-- notes: Lab 2.5 introduces DPO as the beginner-friendly alignment step after SFT. Lab 2.4 is the most fun — students write a Python executor that parses the model's JSON output, runs a calculator, and feeds the result back. When it works end-to-end, it feels like magic: the model asks for a calculator, your code runs it, and the model incorporates the result into its answer. -->

---

## Bridge to the Next Phase

**What you built in Phase 2:** an instruction-following model (`phase2_instruct.pt`) that obeys a chat template, reasons in `<|Thought|>` tags, solves GSM8K-style math, and emits valid JSON tool calls — all by reusing the Phase 1 weights with masked loss.

**The thread into Phase 3:** Phase 2 gave you a model that is *smart but heavy* (16-bit weights). Phase 3 makes it *small* without making it dumb. The crucial concept that carries over is **data mixing**: here you mixed GSM8K and Glaive to teach two skills at once; in Phase 3 you will mix TinyStories + GSM8K + Glaive during quantization so the model keeps all the abilities it just learned (avoiding "catastrophic forgetting"). Also carry forward the idea that **fine-tuning uses a much smaller learning rate** than pretraining — Phase 3 goes smaller still.

<!-- notes: Two explicit hooks into Phase 3. First, data mixing: the 70/30 GSM8K/Glaive mix here becomes a 40/30/30 mix in Phase 3 to preserve all prior skills during quantization. Second, the gentle-fine-tuning mindset (low LR to avoid destroying good weights) continues — Phase 3 uses an even smaller LR. The model produced here, phase2_instruct.pt, is the literal starting checkpoint for Phase 3. -->

---

## Next

**Phase 3:** Quantization-aware training (QAT).

We will teach the model to survive **rounding** — compressing 16-bit floats to 4-bit integers while preserving accuracy.

Preview:
- Why quantize? (cost, speed, edge deployment)
- Fake quantization in the forward pass
- The Straight-Through Estimator trick
- Training with mixed-precision data

<!-- notes: Phase 3 is where we transition from "making the model smarter" to "making the model smaller." Quantization is essential for deployment — nobody wants to run an FP16 model on a phone. But naive quantization (just rounding weights) destroys quality. QAT teaches the model to be robust to rounding during training itself. -->

---

## Further Reading (Optional)

**These papers are optional enrichment — you do NOT need to read any of them to continue the course.**

- Ouyang et al. (2022). *Training language models to follow instructions with human feedback (InstructGPT)*. NeurIPS.
- Wei et al. (2022). *Chain-of-Thought Prompting Elicits Reasoning in Large Language Models*. NeurIPS.
- Cobbe et al. (2021). *Training Verifiers to Solve Math Word Problems (GSM8K)*. arXiv:2110.14168.
- Schick et al. (2023). *Toolformer: Language Models Can Teach Themselves to Use Tools*. NeurIPS.
- Wei et al. (2021). *Finetuned Language Models Are Zero-Shot Learners (FLAN)*. ICLR.
- DeepSeek-AI (2025). *DeepSeek-R1: Incentivizing Reasoning Capability in LLMs via Reinforcement Learning*. arXiv:2501.12948.

<!-- notes: Strictly optional. InstructGPT and FLAN are the foundational instruction-tuning papers; the Chain-of-Thought paper motivates our Thought tags; GSM8K is the dataset we fine-tune on; Toolformer underlies tool/function calling; DeepSeek-R1 is the modern reasoning-via-RL model whose <think> tags mirror our approach. All concepts are already covered in the slides. -->
