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

# Section 2.3: Training on GSM8K Step-by-Step Logic

**Goal:** Load GSM8K, format chain-of-thought examples, and fine-tune with masked loss. We will build a complete training loop using a small demo model, implement proper mask construction from template spans, evaluate generation quality, and plot the loss curve.

## What You Need to Know First

This is the capstone of Phase 2 — it reuses everything from the previous two sections, plus the basic training loop from Phase 1.

- **Chat templates** (Section 2.1) — formatting a question + reasoning + answer into one string with `<|user|>`, `<|assistant|>`, and `<|Thought|>` markers.
- **Masked loss** (Section 2.2) — computing loss only on assistant tokens.
- **A training loop** — the repeated cycle of: predict, measure loss, `backward()`, `optimizer.step()`. You saw this in Phase 1.
- **GSM8K** — a public dataset of grade-school math word problems, each with a worked, step-by-step solution ending in `#### <number>`.
- **Chain-of-thought** — just a fancy name for "showing your work" before giving the final answer.

Plain Python and the PyTorch basics from Phase 1 are all you need.

## Load the Dataset

```python
from datasets import load_dataset

gsm = load_dataset("gsm8k", "main", split="train[:2%]")
row = gsm[0]
print("Q:", row["question"])
print("A:", row["answer"][:200])
print(f"\nDataset size: {len(gsm)} examples")
```

## Parse Final Numeric Answer

GSM8K answers end with `#### <number>`. We need to split reasoning from the final answer for our chat template.

```python
def parse_gsm8k_answer(text):
    final = text.split("####")[-1].strip()
    reasoning = text.split("####")[0].strip()
    return reasoning, final

reasoning, final = parse_gsm8k_answer(row["answer"])
print("reasoning:", reasoning[:120], "...")
print("final:", final)
```

## Chat Template and Mask Builder

The key function `build_mask_from_template` identifies assistant spans in the formatted text by locating the `<|assistant|>` and `<|end|>` markers. Only tokens within assistant regions get mask=1.

```python
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

SPECIAL = {
    "user": "<|user|>",
    "assistant": "<|assistant|>",
    "thought": "<|Thought|>",
    "end": "<|end|>",
}

def gsm8k_to_chat(question, reasoning, final_answer):
    thought = reasoning.strip()
    answer = f"The answer is {final_answer.strip()}."
    return f"{SPECIAL['user']}{question}{SPECIAL['end']}{SPECIAL['assistant']}{SPECIAL['thought']}{thought}{SPECIAL['end']}{answer}{SPECIAL['end']}"

def build_mask_from_template(text, max_len):
    """Build a binary mask where 1 = assistant region (trainable).

    Strategy: find all <|assistant|>...<|end|> spans and mark those character
    positions. Then map character positions to token positions (1 char = 1 token
    in our simplified tokenization).
    """
    mask = torch.zeros(max_len, dtype=torch.float32)
    asst_marker = SPECIAL["assistant"]
    end_marker = SPECIAL["end"]

    i = 0
    while i < len(text):
        asst_start = text.find(asst_marker, i)
        if asst_start == -1:
            break
        region_start = asst_start + len(asst_marker)

        depth = 0
        j = region_start
        region_end = len(text)
        while j < len(text):
            if text[j:].startswith(end_marker):
                if depth == 0:
                    end_of_thought = j + len(end_marker)
                    remaining = text[end_of_thought:]
                    final_end = remaining.find(end_marker)
                    if final_end != -1:
                        region_end = end_of_thought + final_end + len(end_marker)
                    else:
                        region_end = end_of_thought
                    break
                depth -= 1
                j += len(end_marker)
            elif text[j:].startswith(SPECIAL["thought"]):
                depth += 1
                j += len(SPECIAL["thought"])
            else:
                j += 1

        char_start = min(asst_start, max_len)
        char_end = min(region_end, max_len)
        mask[char_start:char_end] = 1.0
        i = region_end

    return mask

sample = gsm8k_to_chat("What is 2+3?", "Add 2 and 3 to get 5.", "5")
sample_mask = build_mask_from_template(sample, len(sample))
print("Sample text length:", len(sample))
print("Masked positions:", int(sample_mask.sum().item()))
print("Mask ratio:", f"{sample_mask.mean().item():.2%}")
```

## Build Training Dataset

```python
class GSM8KChatDataset(Dataset):
    def __init__(self, hf_split, formatter, max_len=256):
        self.rows = hf_split
        self.formatter = formatter
        self.max_len = max_len

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, i):
        r = self.rows[i]
        reasoning, final = parse_gsm8k_answer(r["answer"])
        text = self.formatter(r["question"], reasoning, final)
        ids = [(ord(c) % 8000) for c in text[: self.max_len]]
        ids = ids + [0] * (self.max_len - len(ids))
        x = torch.tensor(ids[:-1])
        y = torch.tensor(ids[1:])
        mask = build_mask_from_template(text, self.max_len - 1)
        return x, y, mask

ds = GSM8KChatDataset(gsm, gsm8k_to_chat)
loader = DataLoader(ds, batch_size=4, shuffle=True)
x, y, m = next(iter(loader))
print("shapes:", x.shape, y.shape, m.shape)
print("mask density:", f"{m.mean().item():.2%}")
```

## Define the Demo Model

Since the Phase 1 checkpoint may not be available in all environments, we define a small transformer-like model that can train quickly for demonstration purposes.

```python
class DemoLM(nn.Module):
    """Minimal language model for demonstrating the training loop."""
    def __init__(self, vocab_size=8000, d_model=128, n_heads=4, n_layers=2, max_len=255):
        super().__init__()
        self.embed = nn.Embedding(vocab_size, d_model)
        self.pos_embed = nn.Embedding(max_len, d_model)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=n_heads, dim_feedforward=d_model * 4,
            dropout=0.1, batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=n_layers)
        self.head = nn.Linear(d_model, vocab_size)

    def forward(self, x):
        B, T = x.shape
        positions = torch.arange(T, device=x.device).unsqueeze(0).expand(B, T)
        h = self.embed(x) + self.pos_embed(positions)
        causal_mask = nn.Transformer.generate_square_subsequent_mask(T, device=x.device)
        h = self.transformer(h, mask=causal_mask, is_causal=True)
        return self.head(h)

device = "cuda" if torch.cuda.is_available() else "cpu"
model = DemoLM().to(device)
n_params = sum(p.numel() for p in model.parameters())
print(f"Demo model: {n_params / 1e6:.1f}M parameters on {device}")
```

## Masked Cross-Entropy Loss

```python
def masked_cross_entropy(logits, targets, mask):
    """Compute CE loss only on masked (assistant) positions."""
    B, T, V = logits.shape
    loss = F.cross_entropy(logits.view(B * T, V), targets.view(B * T), reduction="none")
    loss = loss.view(B, T) * mask
    return loss.sum() / mask.sum().clamp(min=1)
```

## Fine-Tune Loop

```python
import os

ckpt_path = "checkpoints/phase1_80m.pt"
if os.path.exists(ckpt_path):
    ckpt = torch.load(ckpt_path, map_location=device)
    print("Loaded Phase 1 checkpoint — using pretrained weights")
else:
    print("Phase 1 checkpoint not found — training demo model from random init")

optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4, weight_decay=0.01)
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=len(loader))

loss_history = []
model.train()

for epoch in range(2):
    epoch_loss = 0.0
    n_steps = 0
    for step, (x_batch, y_batch, mask_batch) in enumerate(loader):
        x_batch = x_batch.to(device)
        y_batch = y_batch.to(device)
        mask_batch = mask_batch.to(device)

        logits = model(x_batch)
        loss = masked_cross_entropy(logits, y_batch, mask_batch)

        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        scheduler.step()

        loss_history.append(loss.item())
        epoch_loss += loss.item()
        n_steps += 1

        if step % 5 == 0:
            print(f"  epoch {epoch} step {step:3d} loss {loss.item():.4f}")

    print(f"Epoch {epoch} avg loss: {epoch_loss / max(n_steps, 1):.4f}")

print(f"\nTraining complete. Total steps: {len(loss_history)}")
print(f"Final loss: {loss_history[-1]:.4f}")
```

## Loss Curve Visualization

```python
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

fig, ax = plt.subplots(figsize=(10, 4))
ax.plot(loss_history, linewidth=0.8, alpha=0.7, label="Per-step loss")

window = min(10, len(loss_history))
if len(loss_history) >= window:
    smoothed = [
        sum(loss_history[max(0, i - window):i + 1]) / len(loss_history[max(0, i - window):i + 1])
        for i in range(len(loss_history))
    ]
    ax.plot(smoothed, linewidth=2, color="red", label=f"Moving avg (window={window})")

ax.set_xlabel("Training Step")
ax.set_ylabel("Masked Cross-Entropy Loss")
ax.set_title("GSM8K Fine-Tuning Loss Curve")
ax.legend()
ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig("gsm8k_loss_curve.png", dpi=100, bbox_inches="tight")
plt.close()
print("Saved gsm8k_loss_curve.png")
```

## Evaluation: Generate an Answer

Let's sample a question from the dataset, feed it to our model, and check whether the generated text contains the correct final answer.

```python
@torch.no_grad()
def generate(model, prompt_ids, max_new_tokens=100, temperature=0.8):
    """Simple autoregressive generation."""
    model.eval()
    input_ids = prompt_ids.unsqueeze(0).to(device)
    generated = input_ids

    for _ in range(max_new_tokens):
        if generated.shape[1] > 254:
            break
        logits = model(generated)
        next_logits = logits[0, -1, :] / temperature
        probs = F.softmax(next_logits, dim=-1)
        next_token = torch.multinomial(probs, 1)
        generated = torch.cat([generated, next_token.unsqueeze(0)], dim=1)

    model.train()
    return generated[0]

eval_row = gsm[0]
question = eval_row["question"]
_, correct_answer = parse_gsm8k_answer(eval_row["answer"])

prompt = f"{SPECIAL['user']}{question}{SPECIAL['end']}{SPECIAL['assistant']}"
prompt_ids = torch.tensor([(ord(c) % 8000) for c in prompt])

output_ids = generate(model, prompt_ids, max_new_tokens=80)
output_chars = "".join([chr(tid % 128 + 32) for tid in output_ids.tolist()])

print(f"Question: {question}")
print(f"Expected answer: {correct_answer}")
print(f"Generated (decoded): {output_chars[:200]}")
print(f"\nNote: With char-level pseudo-tokenization and a small model,")
print(f"generation quality is limited. The goal is demonstrating the loop.")
```

## Check Answer Correctness

```python
import re

def extract_number(text):
    """Extract the last number from generated text."""
    numbers = re.findall(r'-?\d+\.?\d*', text)
    return numbers[-1] if numbers else None

generated_answer = extract_number(output_chars)
print(f"Extracted number from generation: {generated_answer}")
print(f"Correct answer: {correct_answer}")
if generated_answer and correct_answer:
    try:
        is_correct = abs(float(generated_answer) - float(correct_answer.replace(",", ""))) < 0.01
        print(f"Correct: {is_correct}")
    except ValueError:
        print("Could not parse — expected with pseudo-tokenization")
```

## Exercise: Tune Hyperparameters and Compare Final Loss

Try different learning rates and epoch counts. Fill in the grid below and compare final training loss.

```python
configs = [
    {"lr": 5e-5, "epochs": 1},
    {"lr": 1e-4, "epochs": 2},
    {"lr": 3e-4, "epochs": 3},
]

results = []
for cfg in configs:
    test_model = DemoLM().to(device)
    test_opt = torch.optim.AdamW(test_model.parameters(), lr=cfg["lr"])
    test_model.train()

    final_loss = None
    for epoch in range(cfg["epochs"]):
        for x_batch, y_batch, mask_batch in loader:
            x_batch, y_batch, mask_batch = x_batch.to(device), y_batch.to(device), mask_batch.to(device)
            logits = test_model(x_batch)
            loss = masked_cross_entropy(logits, y_batch, mask_batch)
            test_opt.zero_grad()
            loss.backward()
            test_opt.step()
            final_loss = loss.item()

    results.append({"lr": cfg["lr"], "epochs": cfg["epochs"], "final_loss": final_loss})
    print(f"LR={cfg['lr']:.0e}, epochs={cfg['epochs']} → final loss={final_loss:.4f}")

print("\n=== Hyperparameter Comparison ===")
print(f"{'LR':<10} {'Epochs':<8} {'Final Loss':<12}")
print("-" * 30)
for r in results:
    print(f"{r['lr']:<10.0e} {r['epochs']:<8} {r['final_loss']:<12.4f}")

best = min(results, key=lambda r: r["final_loss"])
print(f"\nBest config: LR={best['lr']:.0e}, epochs={best['epochs']}")
```

## Where This Leads Next

With math reasoning working, Section 2.4 teaches the final instruction-tuning skill: emitting structured **JSON for tool use**, so the model can call a calculator or weather API instead of guessing. After that, Phase 3 shifts from *teaching* the model to *shrinking* it with quantization.

---

## Key Takeaway

A complete instruction-tuning pipeline has four components working together:

1. **Data formatting** — GSM8K's chain-of-thought is serialized into our chat template with `<|Thought|>` spans
2. **Mask construction** — `build_mask_from_template()` ensures only assistant tokens receive gradient signal
3. **Training loop** — standard autoregressive LM training with masked CE loss and gradient clipping
4. **Evaluation** — generate answers and extract final numbers to measure accuracy

The masked loss is critical: without it, the model wastes half its capacity learning to predict questions instead of learning to reason through answers. Even with a tiny model and pseudo-tokenization, you can observe the loss decreasing — confirming the training signal flows correctly through the masked objective.

## Further Reading (Optional)

**Optional — you do NOT need these to continue. They are for curious students who want the original sources.**

- Cobbe et al. (2021). *Training Verifiers to Solve Math Word Problems (GSM8K)*. arXiv:2110.14168.
- Wei et al. (2022). *Chain-of-Thought Prompting Elicits Reasoning in Large Language Models*. NeurIPS.
