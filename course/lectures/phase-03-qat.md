---
title: "Phase 3: Quantization-Aware Training"
subtitle: "Teaching the Model to Survive Rounding"
author: "LLMs From Scratch"
---

# Phase 3: Quantization-Aware Training (QAT)

## Shrink footprints without losing intelligence

Train the model to tolerate low-precision weights and activations.

<!-- notes: Phase 3 is about deployment efficiency. A model that only runs on a datacenter GPU isn't useful for most applications. Quantization lets us shrink model size by 4x (FP16 → INT4) while maintaining most of the model's capability. The key insight: if you quantize DURING training (QAT), the model learns to be robust to rounding errors. Post-training quantization (PTQ) is easier but loses more quality. -->

---

## Learning objectives

- Understand **number representations** (FP32, FP16, INT8, INT4)
- Implement **min-max** quantization with scale and zero-point
- Use **fake quantization** to simulate low precision during training
- Apply the **Straight-Through Estimator** (STE) for gradients through rounding
- Fine-tune with **mixed data** to prevent catastrophic forgetting

<!-- notes: The math in this phase is more concrete than Phase 1 — lots of worked numerical examples. Students who struggled with abstract attention math often find quantization more intuitive because it's fundamentally about rounding and scaling, which are familiar operations. -->

---

## Why quantize?

| Metric | FP16 model | INT4 model | Improvement |
|--------|-----------|-----------|-------------|
| Model size | 160 MB | **40 MB** | 4× smaller |
| Memory bandwidth | 160 MB/token | **40 MB/token** | 4× faster loads |
| Inference speed | 1× | **~2-3×** | Faster |
| Mobile deployable? | Barely | **Yes** | Edge-ready |

**Real-world motivation:**
- **Cloud cost:** 4× less VRAM = 4× more users per GPU = 4× cheaper
- **Inference speed:** memory bandwidth is the bottleneck; smaller weights = faster
- **Edge/mobile:** phones have 4-8 GB RAM total; a 160 MB model is fine, 1.6 GB is not
- **Energy:** less memory movement = less power consumption

<!-- notes: The key insight is that inference is almost always MEMORY-BOUND, not COMPUTE-BOUND. The GPU spends most of its time waiting for weights to be loaded from memory, not doing math. If weights are 4× smaller, they load 4× faster, so inference is roughly 4× faster. This is why quantization is the single most impactful optimization for deployment. Apple's on-device models, llama.cpp, and all mobile AI use heavy quantization. -->

---

## Number representations

**FP32 (32-bit floating point):**

```
[1 bit sign] [8 bits exponent] [23 bits mantissa]
 S  EEEEEEEE  MMMMMMMMMMMMMMMMMMMMMMM

Example: 3.14159 = 0 10000000 10010010000111111011011
Range: ±3.4 × 10³⁸,  Precision: ~7 decimal digits
```

**FP16 (16-bit floating point):**

```
[1 bit sign] [5 bits exponent] [10 bits mantissa]
 S  EEEEE  MMMMMMMMMM

Range: ±65,504,  Precision: ~3.3 decimal digits
```

**INT8 (8-bit integer):**

```
[8 bits]  →  range: -128 to +127  (256 values)
No fractional part — just whole numbers
```

**INT4 (4-bit integer):**

```
[4 bits]  →  range: -8 to +7  (16 values!)
Only 16 possible values to represent any weight
```

The fewer the bits, the **coarser** the representation — but also the **smaller** and **faster**.

<!-- notes: Draw these on the board. The dramatic reduction from FP32 (4 billion possible values) to INT4 (16 possible values) is what makes quantization both powerful and dangerous. The art of QAT is teaching the model to work with only 16 distinct weight values per group. It's like painting a masterpiece with only 16 colors instead of 16 million — you can still create great art, but you need to be strategic about which 16 colors you choose. -->

---

## Min-max quantization — worked example

**Goal:** Map floating-point values to integer range $[q_{\min}, q_{\max}]$

**Formulas:**

$$s = \frac{x_{\max} - x_{\min}}{q_{\max} - q_{\min}}$$

$$z = q_{\min} - \text{round}\left(\frac{x_{\min}}{s}\right)$$

$$x_{\text{int}} = \text{round}\left(\frac{x}{s} + z\right)$$

**Worked example** (INT4 unsigned, range 0–15):

```
Original floats:   [0.2, 0.5, 0.9, 0.1, 0.7]
x_min = 0.1,  x_max = 0.9

Step 1: Scale
  s = (0.9 - 0.1) / (15 - 0) = 0.8 / 15 = 0.0533

Step 2: Zero-point
  z = 0 - round(0.1 / 0.0533) = 0 - round(1.876) = -2

Step 3: Quantize each value
  0.2 → round(0.2/0.0533 + (-2)) = round(3.75 - 2) = round(1.75) = 2
  0.5 → round(0.5/0.0533 + (-2)) = round(9.38 - 2) = round(7.38) = 7
  0.9 → round(0.9/0.0533 + (-2)) = round(16.9 - 2) = round(14.9) = 15
  0.1 → round(0.1/0.0533 + (-2)) = round(1.88 - 2) = round(-0.12) = 0
  0.7 → round(0.7/0.0533 + (-2)) = round(13.1 - 2) = round(11.1) = 11

Quantized integers: [2, 7, 15, 0, 11]
```

**Step 4: Dequantize (reconstruct):**

$$\hat{x} = (x_{\text{int}} - z) \cdot s$$

```
  2  → (2 - (-2)) × 0.0533 = 4 × 0.0533  = 0.213   (was 0.2, error: 0.013)
  7  → (7 - (-2)) × 0.0533 = 9 × 0.0533  = 0.480   (was 0.5, error: 0.020)
  15 → (15 -(-2)) × 0.0533 = 17 × 0.0533 = 0.906   (was 0.9, error: 0.006)
  0  → (0 - (-2)) × 0.0533 = 2 × 0.0533  = 0.107   (was 0.1, error: 0.007)
  11 → (11 -(-2)) × 0.0533 = 13 × 0.0533 = 0.693   (was 0.7, error: 0.007)
```

Mean absolute error: **~0.01** — small price for 4× compression!

<!-- notes: Walk through EVERY step of this calculation in class. This is the core algorithm that students must understand completely. The scale s determines the "step size" between adjacent integer values. The zero-point z handles asymmetric ranges (when the float range doesn't start at zero). In practice, weights are often roughly symmetric around zero, so z is often 0 or close to it. -->

---

## Quantization error — what gets lost

```
Original:      0.1    0.2    0.3    0.4    0.5    0.6    0.7    0.8    0.9
               |      |      |      |      |      |      |      |      |
Quantized:     0.107  0.213  0.320  0.426  0.480  0.586  0.693  0.799  0.906
               |      |      |      |      |      |      |      |      |

Error:        +0.007 +0.013 +0.020 +0.026 -0.020 -0.014 -0.007 -0.001 +0.006
```

**Key observations:**

- Errors are bounded by $\pm s/2$ (half the step size)
- INT4 (16 levels): max error ≈ 0.03 per value
- INT8 (256 levels): max error ≈ 0.002 per value
- **Accumulation is the danger:** small per-weight errors compound across millions of multiply-adds in a forward pass

**QAT fixes this:** the model **sees** these errors during training and adapts its weights to be robust to them.

<!-- notes: The histogram analogy works well here. Imagine you have a histogram with fine bins (FP16) and you have to re-draw it with only 16 bins (INT4). The shape is preserved, but details are lost. The accumulation point is critical — a 0.01 error per weight, multiplied across a 512×2048 matrix multiply, can cause significant output drift. This is why post-training quantization (just rounding weights after training) often fails for aggressive quantization like INT4, while QAT succeeds. -->

---

## Fake quantization in the forward pass

**Fake quantization** = quantize and immediately dequantize during training:

```
               ┌──────────────────────────────────────┐
               │         Fake Quantize Block          │
               │                                      │
  float ──→   │  quantize ──→ int ──→ dequantize     │ ──→ float (with noise)
  weight       │  (round)              (scale back)    │
               └──────────────────────────────────────┘
```

**Why not use actual integers during training?**
- GPU matrix math requires floating point
- We need gradients, which require real-valued arithmetic
- Fake quant adds **quantization noise** to the forward pass without changing the data type

```python
def fake_quantize(x, scale, zero_point, qmin, qmax):
    x_int = torch.clamp(torch.round(x / scale + zero_point), qmin, qmax)
    x_dequant = (x_int - zero_point) * scale
    return x_dequant  # still a float, but with quantization noise
```

<!-- notes: Fake quantization is a simulation. The model never actually uses integers during training — it uses floats that have been rounded to the nearest representable integer value and then scaled back. The effect is that the model experiences the same rounding errors it would encounter during actual INT4 deployment. Think of it as "training with noise" — but the noise is structured (always rounds to discrete levels), not random Gaussian noise. -->

---

## Straight-Through Estimator (STE)

**The problem:** `round()` has zero derivative almost everywhere.

```
round(x):
                ┌───┐     ┌───┐     ┌───┐
           ─────┘   └─────┘   └─────┘   └─────
           0    1    2    3    4    5

d/dx round(x) = 0 everywhere (except at half-integers, where undefined)
```

No gradient → no learning → training is stuck.

**The STE trick:** use different functions for forward and backward:

```
Forward pass:   x_q = round(x)              ← quantized output
Backward pass:  ∂L/∂x = ∂L/∂x_q · 1        ← pretend round was identity

Computation graph:

  x ──→ [round] ──→ x_q ──→ ...  ──→ Loss
          │                              │
          │         BACKWARD             │
          └←───── ∂L/∂x = ∂L/∂x_q ←────┘
                  (gradient passes through
                   as if round() wasn't there)
```

**Intuition:** "I can't tell you exactly how rounding affected the loss, so I'll assume it had no effect and pass the gradient straight through." Crude but empirically effective.

<!-- notes: The STE was proposed by Bengio et al. (2013) and is one of those ideas that sounds like it shouldn't work but does. The mathematical justification is weak — we're literally lying about the derivative. But in practice, the STE provides a useful learning signal. The gradient tells each weight "you should get bigger" or "you should get smaller," and even though the round operation means the weight can only move in discrete jumps, the accumulated gradient from many batches eventually pushes it past a threshold. -->

---

## Per-tensor vs per-channel quantization

**Per-tensor:** one scale $s$ and zero-point $z$ for the entire weight matrix.

```
Weight matrix W (512 × 2048):
  All values share one scale → coarse approximation
  If row 1 ranges [−0.1, 0.1] but row 100 ranges [−2.0, 2.0],
  the small values in row 1 get crushed to zero.
```

**Per-channel:** separate scale and zero-point **for each output channel** (row).

```
Row 1:    s₁ = 0.013,  z₁ = 0    ← fine resolution for small weights
Row 100:  s₁₀₀ = 0.267, z₁₀₀ = 0  ← coarse resolution for large weights
```

**Why per-channel is better for weights:**

- Different neurons learn features at different scales
- Per-channel costs only $2 \times d_{\text{out}}$ extra parameters (negligible)
- Reduces quantization error by **2-5×** in practice

**Per-token quantization** is used for activations (which vary per input).

<!-- notes: Per-channel quantization is standard in all serious quantization frameworks (TensorRT, ONNX Runtime, etc.). The overhead is tiny — for a 512×2048 matrix, per-tensor needs 2 values (s, z), per-channel needs 2×512=1024 values, which is negligible compared to the 1M weights. The quality improvement is dramatic. In our labs, students compare per-tensor vs per-channel and see the accuracy difference firsthand. -->

---

## Catastrophic forgetting

**The risk:** fine-tuning on quantization-specific objectives can make the model forget its Phase 1 and Phase 2 skills.

```
Before QAT:  "What is 2+2?"  →  "The answer is 4."     ✓
After QAT:   "What is 2+2?"  →  "Once upon a time..."  ✗ (forgot instructions!)
```

**The solution:** continual data mixing during QAT.

```
QAT training data:
  ┌──────────────────────────────────────────┐
  │  40% TinyStories    (preserve language)  │
  │  30% GSM8K          (preserve reasoning) │
  │  30% Glaive         (preserve tools)     │
  └──────────────────────────────────────────┘
```

**Why this works:** by showing the model examples from ALL its previous training phases, it maintains those capabilities while simultaneously adapting to quantization noise.

**Analogy:** Learning to ride a bike with training wheels (quantization constraints) while still practicing everything you already know (language, math, tools).

<!-- notes: Catastrophic forgetting is one of the most important practical challenges in continual learning. It's not specific to quantization — any fine-tuning risks overwriting previous knowledge. The data mixing approach is simple but effective. More sophisticated approaches include EWC (Elastic Weight Consolidation) and progressive training, but data mixing works well enough for our purposes. Students should experiment with removing one data source and observing the degradation. -->

---

## QAT training setup

| Hyperparameter | Value | Notes |
|----------------|-------|-------|
| Base checkpoint | `phase2_instruct.pt` | Start from instruct model |
| Quantization | INT4 weights, INT8 activations | W4A8 scheme |
| Granularity | Per-channel (weights) | Per-token (activations) |
| Learning rate | 1e-5 | Very gentle — preserve knowledge |
| Epochs | 2 | Short — we're adapting, not retraining |
| Data mix | TinyStories + GSM8K + Glaive | Prevent forgetting |

**Training loop modification:**

```python
for batch in dataloader:
    with fake_quantize_context():     # <-- NEW: enables fake quant
        logits = model(batch.input_ids)
    loss = cross_entropy(logits, batch.targets)
    loss.backward()                   # STE handles round() gradients
    optimizer.step()
```

The only architectural change is inserting fake quantization nodes before each linear layer.

<!-- notes: The simplicity of QAT integration is the beauty of the approach. You don't change the model architecture, you don't change the loss function, you just wrap the existing forward pass with fake quantization. The STE handles the backward pass automatically. The lower learning rate (1e-5, down from 2e-5 in Phase 2) reflects that we're making even smaller adjustments — we just want the weights to shift slightly to be more robust to rounding, not to learn new capabilities. -->

---

## Lab map

| Lab | Topic | What you build |
|-----|-------|----------------|
| 3.1 | Min, max, rounding math | Quantize/dequantize functions from scratch |
| 3.2 | Fake quant in PyTorch | `FakeQuantize` module with per-channel scales |
| 3.3 | STE | Custom `torch.autograd.Function` with STE |
| 3.4 | QAT fine-tune | Full QAT training with data mixing |

**Deliverable:** `phase3_qat.pt` — a quantization-aware model ready for INT4 deployment.

<!-- notes: Lab 3.1 is pure math — students implement quantize and dequantize in plain Python first, then PyTorch. Lab 3.3 is the trickiest: writing a custom autograd function requires understanding PyTorch's extension mechanism. The STE forward/backward split is a clean example of this. Lab 3.4 brings it all together — students should compare generation quality before and after QAT to see that the model still works despite 4-bit weights. -->

---

## Quality check: before vs after QAT

Students should verify the quantized model still works:

```
Prompt: "What is 15 × 7?"

FP16 model:  "The answer is 105."        ✓
INT4 (no QAT): "The answer is 135."      ✗ (quantization error)
INT4 (with QAT): "The answer is 105."    ✓ (adapted to rounding)
```

**Metrics to track:**
- Perplexity on TinyStories validation (should increase by <10%)
- GSM8K accuracy (should decrease by <5 percentage points)
- JSON parse rate for tool calls (should remain >90%)

<!-- notes: The quality check is essential. Students should run the same evaluation suite on both the FP16 and QAT models to see the tradeoff quantitatively. The goal is not zero degradation — some quality loss is expected and acceptable. The question is whether the 4× compression is worth the small accuracy drop. For most deployment scenarios, it absolutely is. -->

---

## Next

**Phase 4:** Mixture of Experts — smarter, not slower.

We will replace the single FFN in each Transformer block with **multiple expert FFNs** and a **router** that selects which expert processes each token.

Preview:
- 4× total parameters, but only 1× active per token
- Load balancing to prevent expert collapse
- Specialization tracking: does the math expert handle math?

<!-- notes: Phase 4 is the architectural innovation phase. Phases 1-3 were about building, teaching, and compressing a standard dense Transformer. Phase 4 introduces conditional computation — the idea that not every parameter needs to be active for every token. This is how models like Mixtral achieve GPT-3.5-level performance at a fraction of the inference cost. -->
