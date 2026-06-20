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

# Section 7.4: Writing the "Barge-In" Loop — Handling Interruptions

**Goal:** Streaming decode loop that stops AI audio generation when `<INTERRUPT>` tag probability spikes.

## The Critical 80ms Window

The codec frame rate determines interrupt latency. At 12.5 Hz (one frame every 80ms),
the model gets one chance per frame to detect that the user has started speaking. If it
misses the INTERRUPT signal in that frame, the next opportunity is 80ms later — and the
user perceives the AI as "talking over" them.

This means our interrupt detection must be:
- **High recall** — never miss a real interrupt (false negatives are unacceptable)
- **Moderate precision** — occasional false positives are tolerable (AI briefly pauses then resumes)
- **Zero latency within the frame** — decision must be made before the next frame starts generating

Compare this to half-duplex: 300-800ms to detect end-of-utterance, 200-500ms to process,
100-250ms to synthesize. Full-duplex interrupt: 80ms total.

---

## Probability Thresholding vs Argmax for Interrupt Detection

There are two strategies for deciding when to interrupt:

1. **Argmax** — interrupt if `argmax(tag_logits) == INTERRUPT`. Simple but brittle:
   requires the model to be very confident.

2. **Probability threshold** — interrupt if `P(INTERRUPT) > threshold`. More flexible:
   we can tune the threshold to trade off sensitivity vs false alarms.

```python
import torch
import torch.nn.functional as F

INTERRUPT_ID = 3
AI_SPEAKING_ID = 2
SILENCE_ID = 0
USER_SPEAKING_ID = 1

def detect_interrupt_argmax(tag_logits):
    """Simple argmax detection."""
    return tag_logits.argmax().item() == INTERRUPT_ID

def detect_interrupt_threshold(tag_logits, threshold=0.3):
    """Probability-based detection with tunable threshold."""
    probs = F.softmax(tag_logits, dim=-1)
    interrupt_prob = probs[INTERRUPT_ID].item()
    return interrupt_prob > threshold, interrupt_prob

# Compare on edge case: model is uncertain
uncertain_logits = torch.tensor([0.5, 1.0, 1.5, 1.2])  # slightly favoring AI_SPEAKING
print("Uncertain logits:", uncertain_logits.tolist())
print(f"  Argmax says interrupt: {detect_interrupt_argmax(uncertain_logits)}")
thresh_result, prob = detect_interrupt_threshold(uncertain_logits, threshold=0.2)
print(f"  Threshold(0.2) says interrupt: {thresh_result} (P(INTERRUPT)={prob:.3f})")
thresh_result2, prob2 = detect_interrupt_threshold(uncertain_logits, threshold=0.3)
print(f"  Threshold(0.3) says interrupt: {thresh_result2} (P(INTERRUPT)={prob2:.3f})")
```

---

## Core Streaming Session Implementation

```python
def sample_tag(logits, temp=1.0):
    probs = F.softmax(logits / temp, dim=-1)
    return torch.multinomial(probs, 1).item()

def streaming_session(generate_step, max_steps=100, interrupt_threshold=0.3):
    """
    Full streaming decode loop with probability-based interrupt detection.

    generate_step() -> (audio_token, tag_logits)
    Returns: list of emitted audio tokens, list of events
    """
    ai_active = True
    audio_out = []
    events = []

    for step in range(max_steps):
        audio_tok, tag_logits = generate_step()
        tag_probs = F.softmax(tag_logits, dim=-1)
        interrupt_prob = tag_probs[INTERRUPT_ID].item()

        if ai_active:
            audio_out.append(audio_tok)

            if interrupt_prob > interrupt_threshold:
                ai_active = False
                events.append((step, 'INTERRUPT', interrupt_prob))
                continue

        else:
            # AI is paused — check if we should resume
            if tag_probs[AI_SPEAKING_ID].item() > 0.5:
                ai_active = True
                events.append((step, 'RESUME', tag_probs[AI_SPEAKING_ID].item()))

    return audio_out, events

# Stub generator: interrupt spikes at step 15
step_counter = [0]
def fake_gen():
    step_counter[0] += 1
    s = step_counter[0]
    tag_logits = torch.tensor([0.1, 0.1, 2.0, 0.1])  # AI_SPEAKING dominant
    if s == 15:
        tag_logits = torch.tensor([0.1, 0.5, 0.2, 4.0])  # INTERRUPT spike
    if s > 15 and s < 25:
        tag_logits = torch.tensor([0.1, 2.0, 0.1, 0.1])  # USER_SPEAKING
    if s >= 25:
        tag_logits = torch.tensor([0.1, 0.1, 3.0, 0.1])  # AI can resume
    return s, tag_logits

out, events = streaming_session(fake_gen, max_steps=30)
print(f"Audio tokens emitted: {len(out)}")
print("Events:")
for step, event, prob in events:
    print(f"  Step {step}: {event} (prob={prob:.3f})")
```

---

## Full Streaming Session: Multiple Phases

A realistic session goes through several phases: AI speaks → user interrupts → AI listens
→ user finishes → AI resumes with updated context.

```python
class FullStreamingSession:
    def __init__(self, interrupt_threshold=0.3, resume_threshold=0.5):
        self.interrupt_threshold = interrupt_threshold
        self.resume_threshold = resume_threshold
        self.phase = 'ai_speaking'
        self.audio_buffer = []
        self.discarded_audio = []
        self.timeline = []

    def process_frame(self, step, audio_tok, tag_logits):
        """Process one codec frame (~80ms)."""
        probs = F.softmax(tag_logits, dim=-1)

        if self.phase == 'ai_speaking':
            self.audio_buffer.append(audio_tok)
            if probs[INTERRUPT_ID] > self.interrupt_threshold:
                self.phase = 'interrupted'
                n_discard = min(2, len(self.audio_buffer))
                self.discarded_audio = self.audio_buffer[-n_discard:]
                self.audio_buffer = self.audio_buffer[:-n_discard]
                self.timeline.append((step, 'INTERRUPT', f'discarded {n_discard} frames'))

        elif self.phase == 'interrupted':
            self.phase = 'listening'
            self.timeline.append((step, 'LISTENING', 'waiting for user to finish'))

        elif self.phase == 'listening':
            if probs[SILENCE_ID] > 0.5:
                self.phase = 'user_done'
                self.timeline.append((step, 'USER_DONE', 'silence detected'))

        elif self.phase == 'user_done':
            if probs[AI_SPEAKING_ID] > self.resume_threshold:
                self.phase = 'ai_speaking'
                self.timeline.append((step, 'RESUME', 'AI resumes with new context'))

    def run(self, generate_step, max_steps=50):
        for step in range(max_steps):
            audio_tok, tag_logits = generate_step()
            self.process_frame(step, audio_tok, tag_logits)
        return self

# Simulate multi-phase conversation
phase_step = [0]
def multi_phase_gen():
    phase_step[0] += 1
    s = phase_step[0]
    if s <= 12:
        return s, torch.tensor([0.1, 0.1, 3.0, 0.1])      # AI speaking
    elif s == 13:
        return s, torch.tensor([0.1, 0.5, 0.2, 4.0])      # INTERRUPT
    elif s <= 25:
        return s, torch.tensor([0.1, 2.5, 0.1, 0.1])      # User speaking
    elif s <= 30:
        return s, torch.tensor([3.0, 0.5, 0.2, 0.1])      # Silence
    else:
        return s, torch.tensor([0.1, 0.1, 3.5, 0.1])      # AI resumes

session = FullStreamingSession()
session.run(multi_phase_gen, max_steps=40)

print("Multi-Phase Session Timeline:")
print(f"{'Step':<6} {'Event':<15} {'Detail'}")
print("-" * 50)
for step, event, detail in session.timeline:
    latency_ms = step * 80
    print(f"{step:<6} {event:<15} {detail} ({latency_ms}ms)")
print(f"\nTotal audio frames emitted: {len(session.audio_buffer)}")
print(f"Frames discarded on interrupt: {len(session.discarded_audio)}")
```

---

## Graceful Degradation: Handling Partially Generated Audio

When interrupted, we face a choice about partially generated frames:

```python
def graceful_interrupt(audio_buffer, interrupt_point, fade_frames=2):
    """
    Apply a quick fade-out to partially generated audio at interrupt point.
    This prevents audible 'clicks' from sudden cutoff.
    """
    if len(audio_buffer) <= fade_frames:
        return []  # too short to fade, just discard

    # Keep everything before the fade zone
    keep = audio_buffer[:max(0, interrupt_point - fade_frames)]

    # The fade zone gets volume ramped down (in a real codec, this modifies the decoded audio)
    fade_zone = audio_buffer[max(0, interrupt_point - fade_frames):interrupt_point]
    for i, frame in enumerate(fade_zone):
        fade_factor = 1.0 - (i + 1) / (fade_frames + 1)
        fade_zone[i] = int(frame * fade_factor)  # simplified

    return keep + fade_zone

# Demonstrate
buffer = list(range(100, 120))  # 20 audio tokens
interrupted_at = 15
result = graceful_interrupt(buffer, interrupted_at)
print(f"Original buffer: {len(buffer)} frames")
print(f"After graceful interrupt at frame {interrupted_at}: {len(result)} frames kept")
print(f"Last 5 frames (faded): {result[-5:]}")
```

---

## Mic VAD Hook (Integration Point)

```python
def mic_vad_trigger(rms_energy, threshold=0.02):
    """Voice Activity Detection based on RMS energy."""
    return rms_energy > threshold

def compute_rms(audio_chunk):
    """Compute RMS energy of an audio chunk."""
    return (sum(x**2 for x in audio_chunk) / len(audio_chunk)) ** 0.5

# In a real loop: if mic_vad_trigger(...) while ai_active: inject INTERRUPT tag into state
print("Integration with real microphone:")
print("  1. sounddevice captures 80ms chunks")
print("  2. compute_rms(chunk) -> energy")
print("  3. if energy > threshold AND ai_active: force INTERRUPT tag")
print("  4. Model learns to predict INTERRUPT from context (self-supervised)")
```

---

## Exercise: Implement Interrupt Cooldown

Prevent false re-triggers by adding a cooldown period after an interrupt. The AI should
not re-interrupt within N frames of being interrupted itself.

```python
class StreamingWithCooldown:
    def __init__(self, interrupt_threshold=0.3, cooldown_frames=5):
        self.interrupt_threshold = interrupt_threshold
        self.cooldown_frames = cooldown_frames
        self.cooldown_counter = 0
        self.ai_active = True
        self.audio_out = []
        self.events = []

    def process_frame(self, step, audio_tok, tag_logits):
        probs = F.softmax(tag_logits, dim=-1)

        # Decrement cooldown
        if self.cooldown_counter > 0:
            self.cooldown_counter -= 1

        if self.ai_active:
            self.audio_out.append(audio_tok)

            # Only allow interrupt if not in cooldown
            if (probs[INTERRUPT_ID] > self.interrupt_threshold and
                self.cooldown_counter == 0):
                self.ai_active = False
                self.cooldown_counter = self.cooldown_frames
                self.events.append((step, 'INTERRUPT', 'cooldown started'))
        else:
            # Can resume only after cooldown expires
            if (probs[AI_SPEAKING_ID] > 0.5 and self.cooldown_counter == 0):
                self.ai_active = True
                self.events.append((step, 'RESUME', 'cooldown expired'))

# Test: rapid interrupt signals should be debounced
streamer = StreamingWithCooldown(cooldown_frames=5)
for s in range(30):
    if s in [5, 6, 7, 8]:  # rapid interrupt signals
        logits = torch.tensor([0.1, 0.5, 0.2, 4.0])
    elif s > 15:
        logits = torch.tensor([0.1, 0.1, 3.0, 0.1])
    else:
        logits = torch.tensor([0.1, 0.1, 2.0, 0.1])
    streamer.process_frame(s, s, logits)

print("Events with cooldown (only first interrupt should fire):")
for step, event, detail in streamer.events:
    print(f"  Step {step}: {event} - {detail}")
print(f"Audio tokens: {len(streamer.audio_out)} (vs 30 steps)")
print(f"\nWithout cooldown, steps 5,6,7,8 would all trigger interrupts.")
print(f"With cooldown=5 frames, only step 5 triggers; 6-10 are suppressed.")
```

---

## Key Takeaway

The barge-in loop is the real-time heart of full-duplex audio: at every 80ms codec frame,
the model simultaneously generates the next audio token *and* predicts whether the user
is interrupting. Probability thresholding (rather than argmax) gives us a tunable sensitivity
knob. Cooldown periods prevent rapid false re-triggers. Graceful degradation (fade-out of
partially generated audio) ensures clean transitions. The result: sub-100ms interrupt
latency — faster than human perception.
