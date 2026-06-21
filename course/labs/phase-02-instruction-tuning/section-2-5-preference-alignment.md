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

# Section 2.5: Preference Alignment — From SFT to DPO

**Goal:** Understand the post-training pipeline Karpathy covers in *Deep Dive into LLMs* — supervised fine-tuning (SFT) is not the last step; preference alignment teaches the model **which answers humans prefer**.

## What You Need to Know First

- **Chat templates & masked loss** (Sections 2.1–2.2) — how instruction tuning works.
- **GSM8K fine-tuning** (Section 2.3) — SFT on demonstration data.
- **Cross-entropy loss** (Phase 0/1) — still the building block.

Do this section **after Section 2.3** (or after 2.4). It explains what comes *after* SFT without requiring a full RL implementation.

## The three-stage pipeline

Modern chat models typically go through:

```
Base LM (Phase 1)
    ↓  SFT on demonstrations (Phase 2 labs 2.1–2.3)
Instruct LM — can answer, but may be verbose, unsafe, or hedgy
    ↓  Preference alignment (this lab)
Helpful, harmless, concise assistant
```

**SFT** teaches *format* ("answer like this example").
**Preference alignment** teaches *judgment* ("this answer is better than that one").

## Preference pairs

Human labelers (or a stronger model) rank two responses to the same prompt:

```python
prompt = "Explain gravity to a 5-year-old."

chosen = (
    "Gravity is like an invisible hug from the Earth. "
    "It pulls everything toward the ground so we don't float away."
)

rejected = (
    "Gravity is a fundamental interaction described by general relativity "
    "and approximated by Newton's law F = G m1 m2 / r²..."
)
```

The model should increase probability of `chosen` and decrease `rejected`.

## RLHF (conceptual — what InstructGPT did)

1. Train a **reward model** on human rankings.
2. Use **PPO** (policy gradient) to push the LM toward high-reward outputs.

Powerful but fragile: reward hacking, unstable training, many moving parts.

## DPO — Direct Preference Optimization (beginner-friendly)

DPO skips the reward model and PPO loop. One loss directly on preference pairs:

$$\mathcal{L}_{\text{DPO}} = -\log \sigma\left(\beta \left[\log \frac{\pi_\theta(y_w \mid x)}{\pi_{\text{ref}}(y_w \mid x)} - \log \frac{\pi_\theta(y_l \mid x)}{\pi_{\text{ref}}(y_l \mid x)}\right]\right)$$

In plain words:
- $\pi_\theta$ = your fine-tuned model
- $\pi_{\text{ref}}$ = frozen copy from SFT (the "before" model)
- $y_w$ = chosen (winner), $y_l$ = rejected (loser)
- $\beta$ = how strongly to push preferences (typical: 0.1–0.5)

The model learns to prefer `chosen` **relative to what it would have said before**.

## Toy DPO loss in PyTorch

```python
import torch
import torch.nn.functional as F

def dpo_loss(logp_chosen, logp_rejected, logp_ref_chosen, logp_ref_rejected, beta=0.1):
    """
    logp_* = sum of log-probabilities of the response tokens (assistant turn only).
    """
    pi_logratio = logp_chosen - logp_rejected
    ref_logratio = logp_ref_chosen - logp_ref_rejected
    logits = beta * (pi_logratio - ref_logratio)
    return -F.logsigmoid(logits).mean()

# Simulated log-probs from a batch (negative = higher probability)
logp_chosen = torch.tensor([-2.1, -1.8])
logp_rejected = torch.tensor([-3.5, -4.0])
logp_ref_chosen = torch.tensor([-2.3, -2.0])
logp_ref_rejected = torch.tensor([-2.8, -3.2])

loss = dpo_loss(logp_chosen, logp_rejected, logp_ref_chosen, logp_ref_rejected)
print(f"DPO loss: {loss.item():.4f}")  # lower when chosen is preferred over rejected
```

## Computing log-probs from your Phase 2 model

```python
def sequence_logprob(model, input_ids, mask):
    """
    Sum log P(token) over positions where mask==1 (assistant tokens only).
    input_ids: [1, T], mask: [1, T] with 1 on assistant tokens.
    """
    with torch.no_grad():  # remove for training
        logits = model(input_ids)  # [1, T, vocab]
    log_probs = F.log_softmax(logits[:, :-1], dim=-1)
    targets = input_ids[:, 1:]
    token_logp = log_probs.gather(-1, targets.unsqueeze(-1)).squeeze(-1)
    m = mask[:, 1:].float()
    return (token_logp * m).sum(dim=-1) / m.sum(dim=-1).clamp(min=1)
```

During DPO training you run this twice per pair (chosen + rejected) on both $\pi_\theta$ and frozen $\pi_{\text{ref}}$.

## Minimal preference dataset format

```python
preference_examples = [
    {
        "prompt": "<|user|>What is 2+2?<|/user|>",
        "chosen": "<|assistant|>2 + 2 = 4.<|/assistant|>",
        "rejected": "<|assistant|>I'm not sure, maybe 5?<|/assistant|>",
    },
    {
        "prompt": "<|user|>Write a haiku about rain.<|/user|>",
        "chosen": "<|assistant|>Soft drops on the roof\nPuddles grow along the path\nSpring smells like wet earth<|/assistant|>",
        "rejected": "<|assistant|>Rain is water falling from clouds. Rain happens when evaporation occurs.<|/assistant|>",
    },
]
```

In practice you need hundreds to thousands of pairs. Open datasets: *Anthropic HH-RLHF*, *UltraFeedback*.

## SFT vs DPO — when to use which

| Stage | Data | Teaches |
|-------|------|---------|
| SFT (2.1–2.3) | Expert demonstrations | Format, skills, reasoning style |
| DPO (this lab) | A vs B rankings | Helpfulness, tone, safety, conciseness |

**Rule of thumb:** SFT first (your model must know *how* to answer). DPO second (nudge *which* answers are better).

## What we skip (and why)

Full **PPO + reward model** training is production-grade complexity. DPO gives 80% of the benefit for learning purposes. Frontier models (DeepSeek-R1, o1) add **RL on verifiable rewards** (math/code correctness) — see optional reading.

## Exercise: preference pair curation

Write 5 preference pairs for your Phase 2 model's failure modes (too verbose, refuses easy questions, bad JSON). Format them as `prompt / chosen / rejected`.

## Where This Leads Next

Phase 3 quantizes the aligned model. The preference-aligned checkpoint should be saved as `phase2_instruct_dpo.pt` if you run DPO; otherwise `phase2_instruct.pt` from Section 2.3 is fine.

## Key Takeaway

- **SFT** = imitate demonstrations; **DPO** = prefer better answers over worse ones.
- DPO compares your model to a **frozen SFT reference** — prevents drifting too far.
- The post-training stack is: **base → SFT → preference alignment** (→ optional RL on verifiable tasks).
- You do not need PPO to understand alignment — DPO is the beginner-friendly entry point.

## Checkpoint

You understand why ChatGPT is not "just more SFT." Next: **Phase 3 — Quantization**.

## Further Reading (Optional)

- Ouyang et al. (2022). *Training language models to follow instructions with human feedback (InstructGPT)*. NeurIPS.
- Rafailov et al. (2023). *Direct Preference Optimization*. NeurIPS.
- Karpathy, *Deep Dive into LLMs like ChatGPT* — post-training stages explained for beginners.
- DeepSeek-AI (2025). *DeepSeek-R1* — RL on verifiable rewards for reasoning.
