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

# Section 7.3: The Dual-Stream Loss Function — Predicting Sound and System Tags

**Goal:** Joint heads for audio codebook logits and VAD/interrupt tags; combine losses.

## Why Two Heads?

A full-duplex audio model must predict two fundamentally different things at each time step:

1. **Audio tokens** — what sound to produce next (speech content, prosody, silence)
2. **Control tags** — what *state* the conversation is in (who's speaking, interruptions)

These require separate prediction heads because they operate on different vocabularies
and serve different purposes. The audio head produces speech; the tag head produces
control signals that drive the state machine from Section 7.1.

Think of it like driving: you simultaneously control the steering wheel (audio content)
and monitor the mirrors (conversation state). Both happen in parallel, both are essential.

---

## The Dual-Stream Head Architecture

```python
import torch
import torch.nn as nn
import torch.nn.functional as F

d_model = 512
audio_vocab = 1024
tag_vocab = 4  # SILENCE, USER_SPEAKING, AI_SPEAKING, INTERRUPT

class DualStreamHead(nn.Module):
    def __init__(self, d_model, audio_vocab, tag_vocab):
        super().__init__()
        self.audio_head = nn.Linear(d_model, audio_vocab)
        self.tag_head = nn.Linear(d_model, tag_vocab)

    def forward(self, h):
        return self.audio_head(h), self.tag_head(h)

head = DualStreamHead(d_model, audio_vocab, tag_vocab)
h = torch.randn(8, 32, d_model)  # (batch, seq_len, d_model)
audio_logits, tag_logits = head(h)
print(f"Audio logits: {audio_logits.shape}  (predicts 1 of {audio_vocab} codebook entries)")
print(f"Tag logits:   {tag_logits.shape}  (predicts 1 of {tag_vocab} control states)")
```

---

## Loss Weighting: Why Tag Loss Uses 0.5 Coefficient

The tag prediction task is *much easier* than audio prediction:
- Audio: predict 1 of 1024 possible next sounds (high entropy)
- Tags: predict 1 of 4 states (low entropy, long runs of same state)

Without down-weighting, the tag loss would dominate early training gradients (because
it drops to near-zero quickly), starving the audio head of learning signal. The 0.5
coefficient ensures balanced gradient contribution.

```python
audio_targets = torch.randint(0, audio_vocab, (8, 32))
tag_targets = torch.randint(0, tag_vocab, (8, 32))

loss_audio = F.cross_entropy(audio_logits.view(-1, audio_vocab), audio_targets.view(-1))
loss_tag = F.cross_entropy(tag_logits.view(-1, tag_vocab), tag_targets.view(-1))

alpha_tag = 0.5
loss = loss_audio + alpha_tag * loss_tag
print(f"Audio loss: {loss_audio.item():.4f} (CE over {audio_vocab} classes)")
print(f"Tag loss:   {loss_tag.item():.4f} (CE over {tag_vocab} classes)")
print(f"Combined:   {loss.item():.4f} (audio + {alpha_tag} * tag)")
print(f"\nNote: random baseline audio loss = ln({audio_vocab}) = {torch.log(torch.tensor(float(audio_vocab))).item():.2f}")
print(f"Note: random baseline tag loss = ln({tag_vocab}) = {torch.log(torch.tensor(float(tag_vocab))).item():.2f}")
```

---

## Tag Semantics

```python
TAG = ["SILENCE", "USER_SPEAKING", "AI_SPEAKING", "INTERRUPT"]
print("Tag vocabulary:")
for i, name in enumerate(TAG):
    print(f"  {i}: {name}")
```

---

## Sequence-Level Training Example with Annotated Labels

Let's create a realistic training sequence that shows what the labels look like for a
typical conversation turn:

```python
# Simulate a conversation: AI speaks -> user interrupts -> silence -> AI resumes
# Each frame represents ~80ms of audio

sequence_len = 40
audio_labels = torch.randint(0, audio_vocab, (sequence_len,))  # codec tokens
tag_labels = torch.zeros(sequence_len, dtype=torch.long)

# Annotate the conversation state
tag_labels[0:15] = 2   # AI_SPEAKING (frames 0-14)
tag_labels[15:16] = 3  # INTERRUPT (frame 15 - user barges in)
tag_labels[16:25] = 1  # USER_SPEAKING (frames 16-24)
tag_labels[25:30] = 0  # SILENCE (frames 25-29)
tag_labels[30:40] = 2  # AI_SPEAKING (frames 30-39 - AI resumes)

# During USER_SPEAKING and SILENCE, audio labels for AI output are masked
ai_audio_mask = torch.ones(sequence_len)
ai_audio_mask[16:30] = 0.0  # No AI audio loss during user turn

print("Sequence annotation (first 40 frames / ~3.2 seconds):")
print(f"{'Frame':<8} {'Tag':<15} {'Audio Masked?':<15}")
print("-" * 38)
for i in range(0, sequence_len, 5):
    tag_name = TAG[tag_labels[i].item()]
    masked = "YES (no AI audio)" if ai_audio_mask[i] == 0 else "no"
    print(f"{i:<8} {tag_name:<15} {masked:<15}")
```

---

## Tag Accuracy vs Audio Quality — Different Metrics

The two heads need different evaluation metrics because they solve different problems:

```python
def evaluate_dual_stream(model_head, hidden_states, audio_targets, tag_targets):
    """Compute separate metrics for each head."""
    audio_logits, tag_logits = model_head(hidden_states)
    B, T, _ = audio_logits.shape

    # Audio metric: perplexity (lower = better reconstruction)
    audio_loss = F.cross_entropy(
        audio_logits.view(-1, audio_vocab), audio_targets.view(-1))
    audio_perplexity = torch.exp(audio_loss)

    # Tag metric: accuracy (simpler classification task)
    tag_preds = tag_logits.argmax(dim=-1)
    tag_accuracy = (tag_preds == tag_targets).float().mean()

    # Tag metric: per-class precision (INTERRUPT is the critical class)
    interrupt_mask = tag_targets == 3
    if interrupt_mask.sum() > 0:
        interrupt_recall = (tag_preds[interrupt_mask] == 3).float().mean()
    else:
        interrupt_recall = torch.tensor(0.0)

    return {
        'audio_perplexity': audio_perplexity.item(),
        'tag_accuracy': tag_accuracy.item(),
        'interrupt_recall': interrupt_recall.item(),
    }

metrics = evaluate_dual_stream(head, h, audio_targets, tag_targets)
print("Evaluation Metrics:")
for k, v in metrics.items():
    print(f"  {k}: {v:.4f}")
print("\nGoals: audio_perplexity < 50, tag_accuracy > 0.95, interrupt_recall > 0.90")
```

---

## Training Loop: Train Both Heads for 50 Steps

```python
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

torch.manual_seed(42)

# Simple backbone + dual head
backbone = nn.Sequential(
    nn.Linear(d_model, d_model),
    nn.GELU(),
    nn.Linear(d_model, d_model),
)
head_train = DualStreamHead(d_model, audio_vocab, tag_vocab)
params = list(backbone.parameters()) + list(head_train.parameters())
optimizer = torch.optim.AdamW(params, lr=1e-3)

audio_losses = []
tag_losses = []
total_losses = []

n_steps = 50
for step in range(n_steps):
    # Synthetic batch
    x = torch.randn(4, 20, d_model)
    a_tgt = torch.randint(0, audio_vocab, (4, 20))
    t_tgt = torch.randint(0, tag_vocab, (4, 20))

    h_out = backbone(x)
    a_logits, t_logits = head_train(h_out)

    l_audio = F.cross_entropy(a_logits.view(-1, audio_vocab), a_tgt.view(-1))
    l_tag = F.cross_entropy(t_logits.view(-1, tag_vocab), t_tgt.view(-1))
    loss = l_audio + 0.5 * l_tag

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    audio_losses.append(l_audio.item())
    tag_losses.append(l_tag.item())
    total_losses.append(loss.item())

    if step % 10 == 0:
        print(f"Step {step:3d} | audio: {l_audio.item():.3f} | tag: {l_tag.item():.3f} | total: {loss.item():.3f}")

# Plot both loss curves
fig, ax = plt.subplots(figsize=(10, 5))
ax.plot(audio_losses, label='Audio Loss', color='steelblue', linewidth=2)
ax.plot(tag_losses, label='Tag Loss (×0.5 in combined)', color='coral', linewidth=2)
ax.plot(total_losses, label='Combined Loss', color='black', linewidth=2, linestyle='--')
ax.axhline(y=torch.log(torch.tensor(float(audio_vocab))).item(), color='steelblue',
           linestyle=':', alpha=0.5, label=f'Audio random baseline (ln {audio_vocab})')
ax.axhline(y=torch.log(torch.tensor(float(tag_vocab))).item(), color='coral',
           linestyle=':', alpha=0.5, label=f'Tag random baseline (ln {tag_vocab})')
ax.set_xlabel('Training Step')
ax.set_ylabel('Loss')
ax.set_title('Dual-Stream Training: Audio Loss vs Tag Loss')
ax.legend()
plt.tight_layout()
plt.savefig('dual_stream_loss.png', dpi=100, bbox_inches='tight')
plt.close()
print("Saved dual_stream_loss.png")
```

---

## Exercise: Experiment with Different Tag Loss Weights

```python
print("Experimenting with tag loss weights: alpha = [0.1, 0.5, 1.0]")
print("=" * 60)

results = {}
for alpha in [0.1, 0.5, 1.0]:
    torch.manual_seed(42)
    backbone_exp = nn.Sequential(
        nn.Linear(d_model, d_model), nn.GELU(), nn.Linear(d_model, d_model))
    head_exp = DualStreamHead(d_model, audio_vocab, tag_vocab)
    opt = torch.optim.AdamW(
        list(backbone_exp.parameters()) + list(head_exp.parameters()), lr=1e-3)

    final_audio = 0.0
    final_tag = 0.0
    for step in range(50):
        x = torch.randn(4, 20, d_model)
        a_tgt = torch.randint(0, audio_vocab, (4, 20))
        t_tgt = torch.randint(0, tag_vocab, (4, 20))
        h_out = backbone_exp(x)
        a_logits, t_logits = head_exp(h_out)
        l_audio = F.cross_entropy(a_logits.view(-1, audio_vocab), a_tgt.view(-1))
        l_tag = F.cross_entropy(t_logits.view(-1, tag_vocab), t_tgt.view(-1))
        loss = l_audio + alpha * l_tag
        opt.zero_grad()
        loss.backward()
        opt.step()
        final_audio = l_audio.item()
        final_tag = l_tag.item()

    results[alpha] = (final_audio, final_tag)
    print(f"  alpha={alpha:.1f} -> final audio_loss={final_audio:.4f}, tag_loss={final_tag:.4f}")

print("\nAnalysis:")
print("  alpha=0.1: Tag head under-trained (high tag loss), audio head benefits")
print("  alpha=0.5: Balanced — good compromise (default)")
print("  alpha=1.0: Tag head over-weighted, may hurt audio quality")
```

---

## Key Takeaway

The dual-stream loss is the architectural key to full-duplex audio: one head predicts
*what* to say (audio codebook tokens), another predicts the *conversational state*
(control tags). The tag loss coefficient (default 0.5) prevents the easier classification
task from dominating gradients. Train both heads jointly, but evaluate them with different
metrics — perplexity for audio quality, accuracy/recall for tag prediction. The INTERRUPT
tag recall is the safety-critical metric: missing an interrupt means talking over the user.
