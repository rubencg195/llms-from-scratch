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

# Section 7.1: Walkie-Talkies vs. Phone Calls — The Full-Duplex Paradigm

**Goal:** Model half-duplex vs full-duplex state machines and identify where `<INTERRUPT>` fires.

## What You Need to Know First

This section is mostly plain Python logic — if you can read a few `if` statements, you're ready. The only background you need:

- **Basic Python classes and `if`/`elif`** — we model conversations as objects that change a `state` variable based on what's happening.
- **A "state machine"** — just a system that is always in exactly one named state (like `IDLE` or `SPEAKING`) and follows rules for hopping between states. No math required.
- **Tokens and special tags** — from the text phases, you've seen special marker tokens (like end-of-text). Here `<INTERRUPT>` is the same idea: a control signal the model can emit.

That's everything — no audio or signal-processing knowledge is assumed.

## Why Current Voice Assistants Feel Robotic

Every voice assistant you've used (Siri, Alexa, Google Assistant) operates in **half-duplex**
mode — like a walkie-talkie. The rigid protocol: user speaks → system processes → system
responds → user waits. This creates three problems that make conversations feel unnatural:

1. **Turn-taking latency** — 300-800ms of silence while the system detects end-of-utterance,
   transcribes, generates, and synthesizes. In human conversation, gaps >200ms feel awkward.

2. **No backchannels** — Humans constantly signal engagement ("uh-huh," "mmm," "right")
   while the other person speaks. Half-duplex systems are completely silent during user input.

3. **No interruption** — If the system rambles or misunderstands, the user must wait for
   it to finish. In human conversation, we interrupt and course-correct in real-time.

Full-duplex solves all three by allowing **simultaneous input and output streams**.

---

## State Machine: Half-Duplex

```python
from enum import Enum, auto

class HalfDuplexState(Enum):
    IDLE = auto()
    LISTENING = auto()
    PROCESSING = auto()
    SPEAKING = auto()
    COOLDOWN = auto()

class HalfDuplexAgent:
    def __init__(self):
        self.state = HalfDuplexState.IDLE
        self.transcript = []

    def step(self, user_active, ai_has_response=False, ai_done_speaking=False):
        prev = self.state

        if self.state == HalfDuplexState.IDLE:
            if user_active:
                self.state = HalfDuplexState.LISTENING

        elif self.state == HalfDuplexState.LISTENING:
            if not user_active:
                self.state = HalfDuplexState.PROCESSING

        elif self.state == HalfDuplexState.PROCESSING:
            if ai_has_response:
                self.state = HalfDuplexState.SPEAKING

        elif self.state == HalfDuplexState.SPEAKING:
            if ai_done_speaking:
                self.state = HalfDuplexState.COOLDOWN

        elif self.state == HalfDuplexState.COOLDOWN:
            self.state = HalfDuplexState.IDLE

        if prev != self.state:
            self.transcript.append(f"{prev.name} -> {self.state.name}")
        return self.state

hd = HalfDuplexAgent()
# Simulate: user speaks for 3 steps, stops, AI processes, speaks, finishes
events = [
    (True, False, False),   # user speaking
    (True, False, False),   # user speaking
    (True, False, False),   # user speaking
    (False, False, False),  # user stops -> PROCESSING
    (False, True, False),   # AI has response -> SPEAKING
    (False, False, False),  # AI speaking...
    (False, False, True),   # AI done -> COOLDOWN
    (False, False, False),  # -> IDLE
]

print("Half-Duplex State Trace:")
for i, (ua, ar, ad) in enumerate(events):
    state = hd.step(ua, ar, ad)
    print(f"  t={i}: {state.name}")
```

---

## State Machine: Full-Duplex

```python
class FullDuplexState(Enum):
    IDLE = auto()
    AI_SPEAKING = auto()
    USER_SPEAKING = auto()
    BOTH_ACTIVE = auto()
    INTERRUPTING = auto()
    BACKCHANNEL = auto()

class FullDuplexAgent:
    def __init__(self, interrupt_threshold=0.7):
        self.state = FullDuplexState.IDLE
        self.ai_speaking = False
        self.user_speaking = False
        self.interrupt_threshold = interrupt_threshold
        self.transcript = []

    def step(self, user_active, ai_wants_to_speak=False, vad_energy=0.0):
        prev = self.state
        self.user_speaking = user_active

        if self.state == FullDuplexState.IDLE:
            if user_active:
                self.state = FullDuplexState.USER_SPEAKING
            elif ai_wants_to_speak:
                self.ai_speaking = True
                self.state = FullDuplexState.AI_SPEAKING

        elif self.state == FullDuplexState.AI_SPEAKING:
            if user_active and vad_energy > self.interrupt_threshold:
                self.ai_speaking = False
                self.state = FullDuplexState.INTERRUPTING
            elif user_active and vad_energy <= self.interrupt_threshold:
                self.state = FullDuplexState.BACKCHANNEL
            elif not self.ai_speaking:
                self.state = FullDuplexState.IDLE

        elif self.state == FullDuplexState.USER_SPEAKING:
            if not user_active:
                self.state = FullDuplexState.IDLE
            elif ai_wants_to_speak:
                self.ai_speaking = True
                self.state = FullDuplexState.BOTH_ACTIVE

        elif self.state == FullDuplexState.BOTH_ACTIVE:
            if not user_active:
                self.state = FullDuplexState.AI_SPEAKING
            elif vad_energy > self.interrupt_threshold:
                self.ai_speaking = False
                self.state = FullDuplexState.INTERRUPTING

        elif self.state == FullDuplexState.INTERRUPTING:
            self.ai_speaking = False
            self.state = FullDuplexState.USER_SPEAKING

        elif self.state == FullDuplexState.BACKCHANNEL:
            if not user_active:
                self.state = FullDuplexState.AI_SPEAKING
            elif vad_energy > self.interrupt_threshold:
                self.ai_speaking = False
                self.state = FullDuplexState.INTERRUPTING

        if prev != self.state:
            self.transcript.append(f"{prev.name} -> {self.state.name}")
        return {"state": self.state.name, "ai": self.ai_speaking, "user": self.user_speaking}

fd = FullDuplexAgent()
fd.ai_speaking = True
fd.state = FullDuplexState.AI_SPEAKING
result = fd.step(user_active=True, vad_energy=0.9)
print("Full-duplex barge-in:", result)
```

---

## Backchannel Signals

In human conversation, we constantly emit short signals to show we're listening:
"uh-huh," "mmm," "yeah," "right," "I see." These are **backchannels** — they don't
claim the floor but signal engagement. Full-duplex enables them because the AI can emit
short audio tokens while simultaneously processing user input.

```python
class BackchannelDetector:
    """Detect moments where a backchannel is appropriate."""
    def __init__(self, pause_threshold_ms=300, energy_drop=0.3):
        self.pause_threshold_ms = pause_threshold_ms
        self.energy_drop = energy_drop
        self.silence_counter = 0
        self.last_energy = 0.0

    def should_backchannel(self, user_energy, frame_ms=80):
        if user_energy < self.energy_drop and self.last_energy > self.energy_drop:
            self.silence_counter += frame_ms
        else:
            self.silence_counter = 0
        self.last_energy = user_energy

        # Brief pause in user speech -> good time for "uh-huh"
        if self.silence_counter >= self.pause_threshold_ms and user_energy > 0.01:
            self.silence_counter = 0
            return True
        return False

bc = BackchannelDetector()
energies = [0.8, 0.7, 0.1, 0.05, 0.1, 0.05, 0.6, 0.7]  # user speaks, pauses, resumes
for i, e in enumerate(energies):
    if bc.should_backchannel(e):
        print(f"  Frame {i}: emit backchannel (uh-huh)")
```

---

## Timing Analysis: Turn-Taking Latency

```python
import numpy as np

# Typical latencies in milliseconds
half_duplex_latencies = {
    'VAD endpoint detection': 400,
    'Audio streaming to server': 100,
    'Speech-to-text': 200,
    'LLM generation (first token)': 300,
    'Text-to-speech synthesis': 250,
    'Audio streaming to client': 100,
}

full_duplex_latencies = {
    'Codec frame (always listening)': 80,
    'Tag detection (inline)': 0,
    'Audio token generation': 40,
    'Codec decode (1 frame)': 80,
}

print("Half-Duplex Turn-Taking Latency:")
total_hd = 0
for step, ms in half_duplex_latencies.items():
    print(f"  {step:<35} {ms:>5} ms")
    total_hd += ms
print(f"  {'TOTAL':<35} {total_hd:>5} ms")

print(f"\nFull-Duplex Turn-Taking Latency:")
total_fd = 0
for step, ms in full_duplex_latencies.items():
    print(f"  {step:<35} {ms:>5} ms")
    total_fd += ms
print(f"  {'TOTAL':<35} {total_fd:>5} ms")

print(f"\nSpeedup: {total_hd / total_fd:.1f}x faster response")
print(f"Human perception threshold: ~200ms (full-duplex is below this)")
```

---

## Full Simulation: Run Both Machines Through a Conversation

```python
# Simulate a realistic conversation transcript
# Format: (time_ms, event)
conversation = [
    (0, 'user_start'),
    (1500, 'user_pause_brief'),  # brief pause
    (1800, 'user_resume'),
    (3000, 'user_end'),
    (3000 + total_hd, 'ai_start_hd'),  # half-duplex waits
    (3080, 'ai_start_fd'),             # full-duplex responds in 80ms
    (5000, 'user_interrupt'),           # user interrupts AI
    (5000 + 80, 'fd_interrupt'),        # full-duplex stops in 80ms
    (5000 + total_hd, 'hd_no_interrupt'),  # half-duplex: user must wait
]

print("Conversation Timeline Comparison:")
print("=" * 65)
print(f"{'Time (ms)':<12} {'Event':<30} {'Half-Duplex':<12} {'Full-Duplex'}")
print("-" * 65)

hd_state = "idle"
fd_state = "idle"

for time_ms, event in conversation:
    if event == 'user_start':
        hd_state = "listening"
        fd_state = "user_speaking"
    elif event == 'user_pause_brief':
        fd_state = "backchannel"
    elif event == 'user_resume':
        fd_state = "user_speaking"
    elif event == 'user_end':
        hd_state = "processing"
        fd_state = "processing"
    elif 'ai_start' in event:
        if 'hd' in event:
            hd_state = "speaking"
        else:
            fd_state = "speaking"
    elif event == 'user_interrupt':
        hd_state = "ignored"
        fd_state = "interrupting"
    elif event == 'fd_interrupt':
        fd_state = "listening"
    elif event == 'hd_no_interrupt':
        hd_state = "still speaking"

    print(f"{time_ms:<12} {event:<30} {hd_state:<12} {fd_state}")
```

---

## Exercise: Extend FullDuplexAgent with a "Backchannel" State

Add a proper backchannel emission mechanism to the full-duplex agent:

```python
class FullDuplexWithBackchannel(FullDuplexAgent):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.backchannel_tokens = ['<UH_HUH>', '<MMM>', '<YEAH>', '<RIGHT>']
        self.backchannel_cooldown = 0
        self.backchannel_interval = 5  # min frames between backchannels

    def step_with_backchannel(self, user_active, ai_wants_to_speak=False,
                               vad_energy=0.0, user_paused=False):
        result = self.step(user_active, ai_wants_to_speak, vad_energy)

        backchannel_emitted = None
        if self.backchannel_cooldown > 0:
            self.backchannel_cooldown -= 1

        # Emit backchannel during user pause if we're in the right state
        if (self.state == FullDuplexState.BACKCHANNEL and
            user_paused and self.backchannel_cooldown == 0):
            import random
            backchannel_emitted = random.choice(self.backchannel_tokens)
            self.backchannel_cooldown = self.backchannel_interval

        result['backchannel'] = backchannel_emitted
        return result

# Test the backchannel agent
import random
random.seed(42)
agent = FullDuplexWithBackchannel()
agent.ai_speaking = True
agent.state = FullDuplexState.AI_SPEAKING

# User starts speaking softly (not interrupting) -> backchannel opportunity
print("Backchannel Simulation:")
scenarios = [
    (True, False, 0.3, False),   # user speaks softly
    (True, False, 0.2, True),    # user pauses
    (True, False, 0.5, False),   # user continues
    (True, False, 0.2, True),    # user pauses again
    (True, False, 0.1, True),    # another pause
    (True, False, 0.9, False),   # user speaks loudly -> interrupt!
]

for i, (ua, aws, ve, up) in enumerate(scenarios):
    result = agent.step_with_backchannel(ua, aws, ve, up)
    bc_str = f" -> emit {result['backchannel']}" if result['backchannel'] else ""
    print(f"  Frame {i}: {result['state']}{bc_str}")
```

---

## Where This Leads Next

We now have the *control logic* for a real-time conversation, but the model still needs actual
sound to emit and consume. Section 7.2 tackles that: how to turn continuous audio waveforms
into discrete integer tokens — so sound can be modeled with the same next-token machinery as text.

## Key Takeaway

Full-duplex is not just "faster half-duplex" — it is a fundamentally different communication
paradigm. By maintaining parallel input and output streams, the model can: (1) respond within
one codec frame (~80ms vs ~1350ms), (2) emit backchannels while listening, and (3) be
interrupted mid-sentence. This requires the model to generate audio tokens *and* control tags
simultaneously — which is exactly what we build in the next labs.

## Further Reading (Optional)

**Optional — you do NOT need these to continue. They are for curious students who want the original sources.**

- Défossez et al. (2024). *Moshi: a speech-text foundation model for real-time dialogue*. arXiv:2410.00037.
- Skantze (2021). *Turn-taking in Conversational Systems and Human-Robot Interaction: A Review*. Computer Speech & Language.
