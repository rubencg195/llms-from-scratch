---
title: "Phase 6: Encoder-Free Multimodal"
subtitle: "The Gemma 4 Architecture"
author: "LLMs From Scratch"
---

# Phase 6: Encoder-Free Multimodal

## Teach the LLM to see without a Vision Transformer

---

## Before You Begin (Prerequisites)

It all came from earlier phases — **no external knowledge required**.

- **From Phase 1 you need:** token *embeddings* (turning a token into a vector), *self-attention*, and the idea of a sequence of vectors flowing through the transformer.
- **From Phase 2 you need:** *loss masking* — only training on some positions and ignoring others. We reuse that exact trick here (mask the image, train on the text).
- **From Phase 5 you need:** the sense that **longer sequences cost memory** — images add tokens, so the KV-cache lessons still apply.
- **High-school algebra is enough:** the "projector" is just one matrix multiply plus a bias, the same $Wx + b$ you've seen all course.

A *pixel* is one colored dot; an *RGB* image stores three numbers (red, green, blue) per pixel. That's the only image background you need.

<!-- notes: Anchor students in what they already know. The big mental unlock for this phase is that an image patch is just another vector in the same space as a text embedding — and they already understand text embeddings from Phase 1. The loss-masking trick from Phase 2 returns almost verbatim. Reassure beginners who have never done computer vision: we define pixel and RGB right here, and nothing else about images is assumed. -->

---

## Learning objectives

- Explain why **separate encoders** waste VRAM
- **Patch projector:** flatten RGB squares → linear map to `d_model`
- Add **2D spatial coordinates** (X, Y)
- **Stitch** image and text tensors in one sequence
- Train with **text-only loss** (mask image patches)

<!-- notes: This phase is about architectural minimalism. Instead of bolting a 300M-parameter vision encoder onto our LLM, we'll teach the transformer to process raw image patches directly. This is the approach used by Gemma 4 and similar modern architectures. Students will implement the full pipeline from pixel patches to caption generation. -->

---

## The Multimodal Revolution

| Model | Architecture | Year |
|-------|-------------|------|
| CLIP + GPT | Frozen encoder → adapter → LLM | 2021–2023 |
| LLaVA | ViT-L + MLP adapter + Vicuna | 2023 |
| GPT-4V | Proprietary vision encoder + GPT-4 | 2023 |
| Gemini | Natively multimodal from pre-training | 2024 |
| Gemma 4 | Encoder-free patch projection | 2025 |

**Trend**: from bolted-on encoders → native multimodal → no encoder at all.

Why vision matters: **80% of human information intake is visual**. An LLM that can't see is fundamentally limited.

<!-- notes: The evolution here mirrors what happened with NLP. Early systems had separate modules for parsing, NER, sentiment, etc. Then transformers showed you could do everything with one model. The same unification is happening for vision — why have two separate models when one transformer can handle both modalities? -->

---

## Traditional Pipeline: CLIP + LLM

A *Vision Transformer (ViT)* is a transformer that reads images instead of text; an *encoder* is a network that turns raw input into feature vectors. *CLIP* is a popular pre-trained image-text encoder.

```
Image ──→ [ViT Encoder] ──→ [Adapter MLP] ──→ Visual tokens ─┐
                                                                ├──→ [LLM] ──→ Caption
Text  ──→ [Tokenizer]   ──→ [Embedding]   ──→ Text tokens   ─┘
```

**Problems**:

- ViT-L/14 = **304M parameters** just for vision
- Adapter MLP = another **10–50M parameters**
- Two separate training stages (pre-train encoder, then align)
- Vision encoder is **frozen** — can't adapt to your domain

<!-- notes: LLaVA and similar models use this architecture. The ViT encoder was pre-trained on image-text pairs (CLIP training), then frozen during LLM fine-tuning. The adapter MLP learns to translate ViT features into the LLM's embedding space. This works well but is expensive and inflexible. If your domain has unusual images (medical, satellite, microscopy), the frozen ViT may not have good features for them. -->

---

## Old vs New: parameter comparison

| Component | Traditional (LLaVA) | Encoder-Free (Ours) |
|-----------|-------------------|-------------------|
| Vision encoder | ViT-L: **304M params** | None: **0 params** |
| Adapter | MLP: **10M params** | Linear projector: **~400K params** |
| LLM backbone | 7B+ | 80M (our model) |
| Total vision overhead | **314M** | **0.4M** |
| VRAM for vision | ~600 MB (FP16) | ~0.8 MB (FP16) |

The encoder-free approach uses **785× fewer** vision parameters.

<!-- notes: This is a dramatic reduction. The traditional approach dedicates more parameters to vision than our entire LLM has in total. Of course, a 304M-param ViT produces richer features than a simple linear projection — but the question is whether the LLM's own transformer layers can learn to extract those features from raw patches. Spoiler: with enough data and the right training, yes. Gemma 4 proves this at scale. -->

---

## What is a patch?

Take a $48 \times 48$ RGB image. Divide it into a grid of $16 \times 16$ patches:

```
┌────────┬────────┬────────┐
│ (0,0)  │ (0,1)  │ (0,2)  │
│ 16×16  │ 16×16  │ 16×16  │
├────────┼────────┼────────┤
│ (1,0)  │ (1,1)  │ (1,2)  │
│ 16×16  │ 16×16  │ 16×16  │
├────────┼────────┼────────┤
│ (2,0)  │ (2,1)  │ (2,2)  │
│ 16×16  │ 16×16  │ 16×16  │
└────────┴────────┴────────┘
```

Each patch: $16 \times 16 \times 3 = 768$ raw pixel values.

$48 \times 48$ image → $3 \times 3 = 9$ patches → **9 visual tokens**.

<!-- notes: This is exactly how ViT works too — the difference is that ViT passes each patch through 24 transformer layers before the LLM ever sees it. We skip that entirely and hand the raw 768-dimensional vector to a single linear layer. Think of each patch as a "word" in the visual vocabulary. Just as a text token is a discrete symbol mapped to a vector, a patch is a continuous signal projected to the same vector space. -->

---

## The linear patch projector

The projector maps raw patch pixels to the LLM's embedding space:

$$h_{\text{patch}} = W_{\text{proj}} \cdot \text{flatten}(\text{patch}) + b$$

**Dimensions**:
- $\text{flatten}(\text{patch}) \in \mathbb{R}^{768}$ (16 × 16 × 3 RGB values)
- $W_{\text{proj}} \in \mathbb{R}^{d_{\text{model}} \times 768}$ — if $d_{\text{model}} = 512$, this is a $512 \times 768$ matrix
- $b \in \mathbb{R}^{d_{\text{model}}}$
- $h_{\text{patch}} \in \mathbb{R}^{d_{\text{model}}}$ — now lives in the same space as text embeddings

**Parameter count**: $512 \times 768 + 512 = 393{,}728 \approx 400\text{K}$

<!-- notes: This is literally a single nn.Linear layer in PyTorch. That's it. The magic is that the transformer layers above this projection learn to do the feature extraction that ViT would have done. The linear layer just gets the pixels into the right dimensionality. You could use an MLP (two layers with a nonlinearity) for slightly better features, but the linear version works surprisingly well and keeps the architecture minimal. -->

---

## Spatial Awareness: why position matters

Without position information, these two inputs look identical to the model:

- "The **cat on the left** and the dog on the right"
- "The cat on the right and the **dog on the left**"

The model sees 9 patch vectors but doesn't know which patch came from where.

**Solution**: add learned 2D position embeddings:

$$h_{\text{patch}}^{\text{pos}} = h_{\text{patch}} + E_{\text{row}}[r] + E_{\text{col}}[c]$$

Where $E_{\text{row}} \in \mathbb{R}^{3 \times d_{\text{model}}}$ and $E_{\text{col}} \in \mathbb{R}^{3 \times d_{\text{model}}}$ are learned embeddings for the 3×3 grid.

<!-- notes: This is analogous to positional embeddings in text transformers. Without them, attention is permutation-invariant — the model can't distinguish "cat left, dog right" from "dog left, cat right." The 2D factored approach (separate row and column embeddings, added together) is more parameter-efficient than learning a separate embedding for each (row, col) pair, and it generalizes better to different grid sizes. -->

---

## Interleaved sequences

The full input to the transformer is a single sequence mixing modalities:

```
[IMG_0,0] [IMG_0,1] [IMG_0,2] [IMG_1,0] ... [IMG_2,2] [TEXT_0] [TEXT_1] [TEXT_2] ...
 ←────── 9 visual tokens ──────→              ←──── text tokens ────→
```

Both share the **same** $d_{\text{model}}$-dimensional space. Self-attention operates over the full sequence — every text token can attend to every image patch and vice versa.

**No special fusion layer needed** — the transformer *is* the fusion mechanism.

<!-- notes: This is the elegance of the encoder-free approach. There's no cross-attention, no special fusion module, no architectural modification to the transformer at all. We just concatenate the projected patches with the text embeddings and let self-attention do its thing. The model learns which patches are relevant to which words through standard gradient descent on the language modeling objective. -->

---

## Training objective: text-only loss

$$\mathcal{L} = -\sum_{t \in \text{text positions}} \log P(x_t \mid x_{<t})$$

**Key**: we only compute loss on **text tokens**, not image patches.

Why? The image patches are *inputs*, not *predictions*. We don't ask the model to "predict the next pixel" — we ask it to describe what it sees.

This is the same principle as Phase 2 where we masked user turns in chat: **don't train the model to predict things it should treat as given**.

| Token Type | In Attention? | In Loss? |
|-----------|--------------|---------|
| Image patches | Yes | **No** |
| Text tokens | Yes | **Yes** |

<!-- notes: This masking is crucial. If you include image patches in the loss, the model wastes capacity trying to predict raw pixel values autoregressively — which is both difficult and unhelpful for the captioning task. The loss mask tells the model: "your job is to produce text that describes these patches, not to reproduce the patches themselves." This is exactly analogous to instruction tuning where you mask the instruction tokens and only train on the response. -->

---

## Resolution scaling

What happens with a $96 \times 96$ image instead of $48 \times 48$?

| Image Size | Patch Size | Grid | Patches | Visual Tokens |
|-----------|-----------|------|---------|--------------|
| 48×48 | 16×16 | 3×3 | 9 | 9 |
| 96×96 | 16×16 | 6×6 | 36 | 36 |
| 224×224 | 16×16 | 14×14 | 196 | 196 |

**Same projector** ($W_{\text{proj}}$ is the same $512 \times 768$ matrix) — just more patches.

Trade-off: more patches = more detail but **longer sequences** (more compute in attention).

Position embeddings need to be interpolated or re-learned for new grid sizes.

<!-- notes: This is a major advantage of the patch-based approach — resolution scaling is straightforward. You don't need to retrain the projector for different image sizes, just adjust the grid. In practice, you'd train at a fixed resolution and then interpolate position embeddings for inference at higher resolution. ViT-based models face the same position interpolation challenge, so this isn't a disadvantage of the encoder-free approach. -->

---

## Dataset: TinyImage-Stories

$48 \times 48$ RGB images + simple captions — fits 10 GB with Phase 1 backbone.

**Examples**:
- Image of a red circle → "A red circle in the center"
- Image of a blue square on the left → "A blue square on the left side"

Small enough to train on a single GPU, complex enough to test spatial understanding.

<!-- notes: We deliberately keep the images simple so students can verify the model is learning correctly. If the model says "red circle" when there's a blue square, that's an obvious failure. With photographic images, it's harder to tell if errors are due to the architecture or just ambiguity in the image. -->

---

## Lab map

| Lab | Topic |
|-----|-------|
| 6.1 | Encoder VRAM cost |
| 6.2 | Linear patch projector |
| 6.3 | Spatial coordinates |
| 6.4 | Modality mixing |

<!-- notes: Lab 6.1 profiles ViT-L memory usage to motivate the encoder-free approach. Lab 6.2 implements the linear projector and verifies patch extraction. Lab 6.3 adds 2D position embeddings and tests spatial discrimination. Lab 6.4 combines everything, trains on TinyImage-Stories, and evaluates captioning quality. -->

---

## Key takeaways

1. **Encoder-free** multimodal uses 785× fewer vision parameters than ViT-based pipelines
2. A **linear projector** maps raw $16 \times 16 \times 3$ patches into the LLM's embedding space
3. **2D position embeddings** give the model spatial awareness
4. The transformer's own self-attention is the **fusion mechanism** — no special cross-attention needed
5. Train on **text-only loss** — image patches are inputs, not targets

<!-- notes: The big lesson here is that transformers are general-purpose sequence processors. Once you project any modality into the right vector space, the same attention mechanism handles fusion, reasoning, and generation. This principle extends beyond vision to audio (Phase 7), 3D point clouds, and anything else you can tokenize. -->

---

## Bridge to the Next Phase

You just turned **images into tokens** with a simple projector and let the transformer fuse them with text. **Phase 7 (audio)** does the *same move* for sound.

- Vision: an image → patches → projected vectors → tokens in the sequence.
- Audio: a waveform → short time slices → **codec tokens** → tokens in the sequence.
- In both cases, **once a modality is tokenized, the transformer treats it like any other token** — no special fusion module.

The lesson carries straight over: **project any signal into the embedding space and self-attention handles the rest.** Next phase swaps pixels for sound waves.

<!-- notes: Reinforce the unifying principle: tokenize-then-attend. Phase 6 proved it for vision with a linear projector; Phase 7 proves it again for audio with a neural codec. The students should leave expecting audio to feel familiar, because the architectural pattern (signal -> tokens -> shared sequence -> self-attention) is identical. This continuity is the whole pedagogical point of the back half of the course. -->

---

## Further Reading (Optional)

**These papers are optional enrichment — you do NOT need to read any of them to continue the course.**

- Dosovitskiy et al. (2020). *An Image is Worth 16x16 Words: Transformers for Image Recognition at Scale (ViT)*. ICLR.
- Radford et al. (2021). *Learning Transferable Visual Models From Natural Language Supervision (CLIP)*. ICML.
- Alayrac et al. (2022). *Flamingo: a Visual Language Model for Few-Shot Learning*. NeurIPS.
- Bavishi et al. (2023). *Fuyu-8B: A Multimodal Architecture for AI Agents*. Adept AI. (encoder-free patch projection)
- Liu et al. (2023). *Visual Instruction Tuning (LLaVA)*. NeurIPS.
- Beyer et al. (2024). *PaliGemma: A versatile 3B VLM for transfer*. arXiv:2407.07726.

<!-- notes: Tie the reading list back to the slides. ViT and CLIP are the classic encoder-based approach we are simplifying away from. Flamingo and LLaVA are the adapter-based generation. Fuyu-8B is the real-world inspiration for our encoder-free patch projection — the exact idea students implemented in lab. PaliGemma is a strong modern open VLM if anyone wants to see the technique at scale. All optional. -->

---

## Next

**Phase 7:** Full-duplex audio and barge-in.
