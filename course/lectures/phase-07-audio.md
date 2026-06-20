---
title: "Phase 7: Full-Duplex Audio"
subtitle: "Continuous Conversational AI"
author: "LLMs From Scratch"
---

# Phase 7: Full-Duplex Audio & Interruptions

## Stream audio like a phone call, not a walkie-talkie

---

## Before You Begin (Prerequisites)

Built entirely on earlier phases — **no external knowledge required**.

- **From Phase 1 you need:** next-token prediction with a *softmax* (turning scores into probabilities over a vocabulary). Audio generation is literally next-token prediction over a *sound* vocabulary.
- **From Phase 6 you need:** the big idea that **any signal can become tokens** and join the same sequence. Last phase it was image patches; this phase it's sound.
- **From Phase 2 you need:** comfort with multiple training targets / masked losses — here we predict audio *and* control tags at once.
- **High-school algebra is enough:** no signal-processing background assumed; terms like *waveform* and *codec* are defined as they appear.

A *waveform* is just the up-and-down air-pressure signal of sound, sampled as a long list of numbers.

<!-- notes: Lower the intimidation factor — many students assume audio requires a DSP background. It does not. The only true prerequisite is the next-token-prediction loop from Phase 1, plus the tokenize-any-modality mindset from Phase 6. Stress the parallel: text predicts the next word, audio predicts the next codec token, and the math (softmax over a vocabulary) is identical. Define waveform and codec on the slides so nobody is lost. -->

---

## Learning objectives

- Contrast **half-duplex** vs **full-duplex**
- Discretize waveforms with a **neural codec** (Mimi / EnCodec)
- **Dual-stream loss:** audio tokens + system tags (VAD, interrupt)
- Implement **barge-in** control loop

<!-- notes: This phase brings our LLM into the real-time conversational world. By the end, students will understand how models like GPT-4o and Moshi process and generate speech as a native modality — not through a speech-to-text-to-speech pipeline, but as direct token prediction. This is the architecture that enables sub-200ms voice response. -->

---

## Voice AI in 2026

| System | Architecture | Latency | Duplex? |
|--------|-------------|---------|---------|
| Alexa / Siri (classic) | ASR → NLU → TTS pipeline | 1–3 sec | No |
| GPT-4o | Native audio tokens | ~300 ms | Partial |
| Gemini Live | Native multimodal | ~250 ms | Yes |
| Moshi (Kyutai) | Dual-stream LM | ~200 ms | **Full** |

The **full-duplex revolution**: AI that can listen *while* speaking, handle interruptions, and produce backchannels ("uh-huh", "right") — just like a human phone call.

<!-- notes: The pipeline approach (ASR → LLM → TTS) has a fundamental latency floor. Each step adds 100-500ms, and the system can't listen while generating a response. Moshi's key insight was that if you model audio as discrete tokens, the LLM can simultaneously predict the next audio token to speak AND process incoming audio tokens from the user. This is what full-duplex means — true simultaneous bidirectional communication. -->

---

## Half-duplex vs full-duplex: timing

**Half-duplex** (walkie-talkie):
```
User:  ████████████░░░░░░░░░░░░░░░░░░░░░░░
AI:    ░░░░░░░░░░░░░░░████████████░░░░░░░░░
       ←─ user speaks ─→← silence →←─ AI speaks ─→
```

**Full-duplex** (phone call):
```
User:  ████████████░░░████░░░░░░░░░░░░░░░░░
AI:    ░░░░░░░░████████████░░████████░░░░░░░
                ↑ overlap  ↑     ↑ resumes
              (backchannel) (interrupt)
```

Full-duplex enables:
- **Backchannels**: "uh-huh" while user is speaking
- **Interruptions**: user cuts in mid-response, AI stops gracefully
- **Turn-taking**: natural conversational rhythm

<!-- notes: Think about a real phone conversation. You don't wait for complete silence before responding. You overlap, you make encouraging noises, you interrupt when you have something urgent to say. Half-duplex systems feel robotic precisely because they lack this natural conversational flow. The technical challenge is that the model must process two streams simultaneously — incoming user audio and outgoing AI audio. -->

---

## From Waveforms to Tokens

A *neural codec* is a learned compressor (encode → tiny discrete codes → decode) — like MP3, but the codes are tokens an LLM can read.

The audio codec pipeline:

```
Raw PCM (24 kHz, 16-bit)
       │
       ▼
┌──────────────┐
│Mel Spectrogram│  Time-frequency representation
└──────────────┘   (128 mel bands × T frames)
       │
       ▼
┌──────────────┐
│Neural Encoder │  CNN + Transformer → latent vectors
└──────────────┘
       │
       ▼
┌──────────────┐
│  Quantizer   │  Residual Vector Quantization (RVQ)
└──────────────┘   → discrete codebook indices
       │
       ▼
  Discrete tokens at 12.5 Hz
  (1 token per 80ms of audio)
```

**24,000 samples/sec → 12.5 tokens/sec**: a 1,920× compression ratio.

<!-- notes: This compression is what makes audio modeling with LLMs feasible. Without it, one second of audio at 24kHz would be 24,000 tokens — autoregressive generation would be impossibly slow. The neural codec learns to compress audio into a small number of discrete tokens that capture the essential information: phonemes, pitch, timing, speaker identity. The 12.5 Hz rate means one token every 80ms, which aligns well with phoneme durations (typically 50-200ms). -->

---

## EnCodec / Mimi Architecture

```
┌─────────────────────────────────────────────┐
│              Neural Audio Codec              │
│                                             │
│  ┌─────────┐   ┌─────────┐   ┌─────────┐  │
│  │ Encoder │──→│   RVQ   │──→│ Decoder │  │
│  │ (CNN +  │   │(codebook│   │ (CNN +  │  │
│  │  LSTM)  │   │ lookup) │   │  LSTM)  │  │
│  └─────────┘   └─────────┘   └─────────┘  │
│       ↑              │             │        │
│   waveform     code indices    waveform     │
│    input        (tokens)       output       │
└─────────────────────────────────────────────┘
```

**Residual Vector Quantization (RVQ)**: multiple codebook layers, each encoding the *residual* error from the previous layer.

- Layer 1: coarse structure (phoneme identity)
- Layer 2: residual details (pitch contour)
- Layer 3+: fine details (speaker timbre, noise)

For LLM integration, we use only the **first codebook layer** — sufficient for intelligible speech.

<!-- notes: RVQ is like progressive JPEG for audio. The first layer gives you a rough but recognizable version. Each subsequent layer adds fidelity. Moshi uses 8 RVQ levels for high-quality audio, but for our purposes the first level captures the linguistic content we need. The encoder and decoder are pre-trained on large speech corpora — we freeze them and only use the tokenization in our pipeline. Think of the codec as the "tokenizer" for audio, just as BPE is the tokenizer for text. -->

---

## Why Discrete Audio?

Discrete tokens enable standard **autoregressive LLM generation** for audio:

$$P(\text{audio}_t \mid \text{audio}_{<t}, \text{text}_{<t}) = \text{softmax}(W_{\text{audio}} \cdot h_t)$$

**Alternative approaches and their drawbacks**:

| Approach | Method | Problem |
|----------|--------|---------|
| Continuous regression | Predict float waveform values | Blurry, averaged outputs |
| Diffusion | Iterative denoising | Slow (100+ steps), can't stream |
| Flow matching | Learned ODE | Complex, hard to condition on text |
| **Discrete tokens** | Codebook softmax | Fast, streamable, standard LLM |

Discrete tokens = **same architecture, same training loop, same inference** as text generation.

<!-- notes: This is the key insight that unlocked native audio LLMs. By converting audio to discrete tokens, we can treat speech generation exactly like text generation. No architectural changes needed, no special sampling procedures, no iterative refinement. The model just predicts the next codebook index from a 1024-class softmax, the same way it predicts the next text token from a 50,000-class softmax. This unification is beautiful — one model, one objective, multiple modalities. -->

---

## Dual-stream output: audio + tags

The model has **two parallel output heads** sharing the same hidden state:

```
                  ┌──────────────┐
Hidden state ────→│  Audio Head  │──→ softmax over 1024 codebook entries
     h_t     │   └──────────────┘
             │
             │   ┌──────────────┐
             └──→│   Tag Head   │──→ softmax over 4 system tags
                 └──────────────┘
```

**Audio head**: $P(\text{code}_t) = \text{softmax}(W_{\text{audio}} \cdot h_t) \in \mathbb{R}^{1024}$

**Tag head**: $P(\text{tag}_t) = \text{softmax}(W_{\text{tag}} \cdot h_t) \in \mathbb{R}^{4}$

**Joint loss**: $\mathcal{L} = \mathcal{L}_{\text{audio}} + \lambda \mathcal{L}_{\text{tag}}$

Both predicted at every time step (~12.5 Hz).

<!-- notes: The dual-head design is what enables real-time conversation management. The audio head generates speech while the tag head simultaneously monitors the conversation state. This is more efficient than having separate models for speech generation and voice activity detection. The shared hidden state means the tag predictions are conditioned on the full conversational context — the model knows what it's saying and what the user is saying, so it can make intelligent decisions about when to yield the floor. -->

---

## Voice Activity Detection Tags

*Voice Activity Detection (VAD)* = deciding, moment to moment, whether someone is speaking or silent. A *state machine* is a set of named states with rules for moving between them.

Four tags form a **state machine** for conversation control:

```
                 ┌──────────┐
          ┌─────→│ SILENCE  │←────┐
          │      └──────────┘     │
          │         │    │        │
          │    user │    │ AI     │
          │  starts │    │starts  │
          │         ▼    ▼        │
  ┌───────────────┐  ┌───────────────┐
  │USER_SPEAKING  │  │ AI_SPEAKING   │
  └───────────────┘  └───────────────┘
          │                │
          │  user speaks   │
          │  during AI     │
          │      ▼         │
          │  ┌───────────┐ │
          └──│ INTERRUPT  │─┘
             └───────────┘
```

| Tag | Meaning | Action |
|-----|---------|--------|
| `SILENCE` | Nobody speaking | Wait / generate filler |
| `USER_SPEAKING` | User has the floor | Listen, suppress AI output |
| `AI_SPEAKING` | AI has the floor | Continue generating |
| `INTERRUPT` | User cuts in during AI speech | Stop generation, yield floor |

<!-- notes: This state machine is the brain of the conversation manager. Traditional voice assistants use a separate VAD model to detect speech, then hand off to the LLM. Here, the LLM itself predicts these tags, which means the conversation control is context-aware. The model can distinguish between a user saying "uh-huh" (backchannel, don't stop) and "wait, actually—" (interrupt, stop and listen). This contextual understanding is what makes full-duplex feel natural. -->

---

## Barge-in control loop

The real-time inference loop with latency analysis:

```
Every 80ms (one codec frame):
┌──────────────────────────────────────────────┐
│ 1. Encode user audio → input token  (~5ms)   │
│ 2. Forward pass through LLM        (~20ms)   │
│ 3. Sample audio token + tag         (~1ms)   │
│ 4. Check tag:                                │
│    ├─ AI_SPEAKING → decode & play   (~5ms)   │
│    ├─ INTERRUPT → flush buffer, yield        │
│    ├─ USER_SPEAKING → suppress output        │
│    └─ SILENCE → optional backchannel         │
│ 5. Decode audio token → waveform    (~5ms)   │
│                                              │
│ Total per-frame budget: 80ms                 │
│ Typical compute:        ~36ms                │
│ Margin:                 ~44ms                │
└──────────────────────────────────────────────┘
```

**Critical path**: if compute exceeds 80ms, we drop frames → audible glitches.

<!-- notes: The 80ms frame budget is tight but achievable on modern GPUs for small models. Our 80M-parameter model comfortably fits within this budget. Larger models (7B+) need techniques like speculative decoding or model parallelism to hit real-time. The interrupt path is the most latency-sensitive — when a user cuts in, the AI should stop within 1-2 frames (80-160ms). Longer than 250ms and it feels unresponsive. The margin of ~44ms gives us room for garbage collection, OS scheduling, and audio buffering. -->

---

## Barge-in: the interrupt experience

**Without barge-in** (half-duplex):
```
AI:    "The answer to your question is that the primary reason
        for the economic downturn was—"
User:  [waits... waits... finally AI stops]
User:  "Actually, I meant a different question."
```

**With barge-in** (full-duplex):
```
AI:    "The answer to your question is that the—"
User:  "Wait, I meant—"     ← INTERRUPT detected at frame t
AI:    [stops within 80ms]   ← flush output buffer
AI:    [silence]             ← yields floor
User:  "I meant a different question."
AI:    "Sure, go ahead."    ← resumes after user finishes
```

Response feels **conversational**, not robotic.

<!-- notes: The key UX difference is responsiveness. When you interrupt a person mid-sentence, they stop within about 200ms. Our system targets 80-160ms, which feels instantaneous. The flush operation is important — we don't just stop generating, we also clear any audio that was generated but not yet played. Otherwise you'd hear a fragment of the AI's next word before it goes silent, which sounds unnatural. -->

---

## Challenges in production voice AI

| Challenge | Description | Mitigation |
|-----------|------------|------------|
| **Echo cancellation** | AI hears its own output through user's mic | Acoustic echo cancellation (AEC) filter |
| **Background noise** | Traffic, music, other speakers | Noise-robust codec training, denoising |
| **Latency spikes** | Network jitter, GPU contention | Jitter buffer, priority scheduling |
| **Emotional prosody** | Conveying empathy, excitement, concern | Prosody tokens or style conditioning |
| **Multilingual** | Code-switching mid-sentence | Shared multilingual codebook |
| **Hallucinated speech** | Model generates audio in silence periods | Tag-conditioned output gating |

<!-- notes: Echo cancellation is the biggest practical challenge. If the AI's audio leaks back through the user's microphone, the model hears its own speech and can get confused — it might think the user is speaking, or it might try to continue its own output in a feedback loop. Professional AEC is a signal processing problem that's been studied for decades in telephony, but neural codecs can also be trained to be echo-robust. In our lab, we sidestep this by using headphones, but production systems need proper AEC. -->

---

## Dataset

**SpeechInstruct** through pretrained codec weights (inference-only in lab).

We use pre-trained EnCodec/Mimi weights to tokenize audio — the codec training itself requires thousands of GPU hours on diverse speech data.

In lab, students will:
1. Tokenize pre-recorded audio with the frozen codec
2. Train the dual-head model on tokenized audio + tags
3. Run the barge-in control loop in simulated real-time

<!-- notes: We don't train the codec from scratch because it requires massive compute and diverse audio data. The codec is a tool, like the text tokenizer — we use it as-is. What students build is the LLM component: the dual-stream architecture that generates audio tokens and tags simultaneously. The simulated real-time environment lets us test interrupt handling without dealing with actual microphone input and speaker output. -->

---

## Lab map

| Lab | Topic |
|-----|-------|
| 7.1 | Duplex paradigm |
| 7.2 | Audio tokenization |
| 7.3 | Dual-stream loss |
| 7.4 | Barge-in loop |

<!-- notes: Lab 7.1 builds intuition by comparing half-duplex and full-duplex timing diagrams. Lab 7.2 tokenizes audio clips using the pre-trained codec and verifies reconstruction quality. Lab 7.3 implements the dual-head architecture and trains on SpeechInstruct. Lab 7.4 implements the real-time control loop with interrupt detection and floor management. -->

---

## Key takeaways

1. **Neural codecs** compress 24 kHz audio to 12.5 discrete tokens/sec — enabling LLM-based speech generation
2. **Dual-stream heads** (audio + tags) let one model handle generation *and* conversation management
3. **Barge-in** requires sub-80ms interrupt detection for natural interaction
4. Full-duplex voice AI is the **same autoregressive framework** as text — just different tokens

<!-- notes: The unification theme continues from Phase 6. Text, images, and now audio — all tokenized, all processed by the same transformer, all generated with next-token prediction. This is the power of the discrete token abstraction. Phase 8 will add the final piece: memory that persists across the entire conversation. -->

---

## Bridge to the Next Phase

Real conversations don't end — they **go on and on**. A long phone call produces a huge stream of audio tokens, and (from Phase 5) you know the **KV cache grows with every token**.

- Streaming audio makes contexts *effectively unbounded* — you can't keep every token forever.
- We need a way to **remember what matters** (the user's name, the request) while **forgetting filler** ("uh-huh", silence).
- That is exactly what **Phase 8 (Titans)** delivers: a memory that updates *while* you talk and stays a fixed size no matter how long the conversation runs.

So: full-duplex audio creates the *need* for endless memory — Phase 8 provides it.

<!-- notes: Motivate Phase 8 by making the pain concrete. Audio is the most token-hungry modality so far: a continuous conversation never stops producing tokens. Even with TurboQuant from Phase 5, a linear-growing cache eventually loses. This sets up the central promise of Titans — O(1) memory that learns at test time. The emotional beat: we've built something that can talk forever, but it can't yet remember forever. Phase 8 closes that gap. -->

---

## Further Reading (Optional)

**These papers are optional enrichment — you do NOT need to read any of them to continue the course.**

- van den Oord, Vinyals, & Kavukcuoglu (2017). *Neural Discrete Representation Learning (VQ-VAE)*. NeurIPS.
- Zeghidour et al. (2021). *SoundStream: An End-to-End Neural Audio Codec*. IEEE/ACM TASLP.
- Défossez et al. (2022). *High Fidelity Neural Audio Compression (EnCodec)*. arXiv:2210.13438.
- Borsos et al. (2022). *AudioLM: a Language Modeling Approach to Audio Generation*. arXiv:2209.03143.
- Wang et al. (2023). *Neural Codec Language Models are Zero-Shot Text to Speech Synthesizers (VALL-E)*. arXiv:2301.02111.
- Défossez et al. (2024). *Moshi: a speech-text foundation model for real-time dialogue*. arXiv:2410.00037.

<!-- notes: Map the readings to the lecture. VQ-VAE is the origin of discrete codes; SoundStream and EnCodec are the actual neural codecs whose tokens we use. AudioLM and VALL-E show language-modeling over audio tokens. Moshi is the full-duplex dual-stream system that directly inspired this phase's barge-in architecture. All optional — point curious students to Moshi first since it ties the whole phase together. -->

---

## Next

**Phase 8:** Google Titans — neural memory with test-time training.
