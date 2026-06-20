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

# Section 2.1: Formatting Chat Templates with Special `<|Thought|>` Tags

**Goal:** Serialize multi-turn chats with role markers and thought tags for instruction tuning.

## What You Need to Know First

Everything here builds on ideas from Phase 1 — no outside knowledge needed.

- **Tokens and tokenization** — text is chopped into small pieces (tokens), each with a number (an ID). We covered this in Phase 1.
- **Next-token prediction** — a base model is just trained to guess the next token over and over. That is all it knows how to do.
- **Special tokens** — extra "marker" tokens (like `<|user|>`) that we invent and add to the vocabulary to signal structure rather than ordinary words.
- Basic Python (functions, dictionaries, f-strings) is all the coding you need.

If any of these feel fuzzy, the examples below will make them concrete as you go.

## Why Chat Formatting Matters

A base language model is trained on next-token prediction over raw text — it has no concept of "user" or "assistant." If you prompt a base model with a question, it may continue writing more questions, finish the paragraph you started, or produce entirely unrelated text. It *babbles* because nothing in its training told it that a response is expected.

An **instruct model** solves this by training on formatted conversations where special tokens delineate who is speaking and when a response ends. The model learns: "When I see `<|assistant|>`, I should generate a helpful reply; when I see `<|end|>`, I should stop."

Chat templates are the **interface contract** between the user and the model. Getting them wrong during fine-tuning means the model cannot follow instructions at inference time — no matter how good the training data is.

## Comparing Chat Template Styles

Different model families use different formatting conventions. Here are the three most common:

| Style | Markers | Example |
|-------|---------|---------|
| **ChatML** (OpenAI) | `<\|im_start\|>role`, `<\|im_end\|>` | `<\|im_start\|>user\nHello<\|im_end\|>` |
| **Llama-style** (Meta) | `[INST]`, `[/INST]`, `<<SYS>>` | `[INST] Hello [/INST] Hi there!` |
| **Our Custom** | `<\|user\|>`, `<\|assistant\|>`, `<\|Thought\|>`, `<\|end\|>` | `<\|user\|>Hello<\|end\|><\|assistant\|>Hi!<\|end\|>` |

Our format adds an explicit **thought tag** so the model can perform chain-of-thought reasoning in a structured span that we can later hide from the user or use for evaluation.

## Token Stream Visualization

When the formatted text is tokenized, different regions serve different purposes during training:

| Position | Token(s) | Role Region | Train? | Purpose |
|----------|----------|-------------|--------|---------|
| 0 | `<\|user\|>` | User start | No | Role boundary marker |
| 1-5 | `What is 7 + 5 ?` | User body | No | Input context only |
| 6 | `<\|end\|>` | User end | No | Boundary |
| 7 | `<\|assistant\|>` | Assistant start | Yes | Teaches model to begin response |
| 8 | `<\|Thought\|>` | Thought start | Yes | Triggers reasoning mode |
| 9-14 | `I add seven and five...` | Thought body | Yes | Chain-of-thought reasoning |
| 15 | `<\|end\|>` | Thought end | Yes | Closes thought span |
| 16-19 | `7 + 5 = 12 .` | Answer body | Yes | Final answer tokens |
| 20 | `<\|end\|>` | Assistant end | Yes | EOS for generation |

The "Train?" column maps directly to the **loss mask** from Section 2.2 — we only compute gradients on assistant tokens.

## Defining Special Tokens

```python
SPECIAL = {
    "user": "<|user|>",
    "assistant": "<|assistant|>",
    "thought": "<|Thought|>",
    "end": "<|end|>",
}

def format_example(user_msg, thought, answer):
    return (
        f"{SPECIAL['user']}{user_msg}{SPECIAL['end']}"
        f"{SPECIAL['assistant']}{SPECIAL['thought']}{thought}{SPECIAL['end']}"
        f"{answer}{SPECIAL['end']}"
    )

example = format_example(
    "What is 7 + 5?",
    "I add seven and five to get twelve.",
    "7 + 5 = 12.",
)
print(example)
```

## Special Token Registration in the Tokenizer

For special tokens to work correctly, they must be **atomic** — the tokenizer should never split `<|user|>` into subword pieces like `<`, `|`, `user`, `|`, `>`. We achieve this by adding them explicitly to the tokenizer vocabulary.

```python
class SimpleTokenizerWithSpecial:
    """Demonstrates how special tokens are registered and handled."""

    def __init__(self, base_vocab_size=8000):
        self.base_vocab_size = base_vocab_size
        self.special_tokens = {}
        self.next_id = base_vocab_size

    def add_special_tokens(self, tokens: dict):
        """Register special tokens with dedicated IDs above base vocab."""
        for name, surface_form in tokens.items():
            self.special_tokens[surface_form] = self.next_id
            self.next_id += 1
        print(f"Vocab size after adding specials: {self.next_id}")
        return self.special_tokens

    def tokenize_with_special(self, text):
        """Split on special tokens first, then sub-tokenize body segments."""
        import re
        pattern = "(" + "|".join(re.escape(t) for t in self.special_tokens) + ")"
        segments = re.split(pattern, text)
        tokens = []
        for seg in segments:
            if seg in self.special_tokens:
                tokens.append({"text": seg, "id": self.special_tokens[seg], "is_special": True})
            elif seg:
                for ch in seg:
                    tokens.append({"text": ch, "id": ord(ch) % self.base_vocab_size, "is_special": False})
        return tokens

tokenizer = SimpleTokenizerWithSpecial()
id_map = tokenizer.add_special_tokens(SPECIAL)
print("Special token IDs:", id_map)

tokens = tokenizer.tokenize_with_special(example)
print(f"\nFirst 10 tokens:")
for t in tokens[:10]:
    marker = " [SPECIAL]" if t["is_special"] else ""
    print(f"  id={t['id']:5d}  '{t['text']}'{marker}")
```

## Tokenizer Alignment — Simplified Version

Special strings must be single tokens (or consistent subword splits). Register them in vocab when training BPE.

```python
def tokenize_with_special(text, special_tokens):
    """Simplified: split on markers then sub-tokenize body."""
    import re
    pattern = "(" + "|".join(re.escape(t) for t in special_tokens.values()) + ")"
    parts = re.split(pattern, text)
    result = []
    for part in parts:
        if part in special_tokens.values():
            result.append(part)
        else:
            result.extend(part.split())
    return result

print(tokenize_with_special(example, SPECIAL)[:15])
```

## Batch Formatting for GSM8K-Style Data

```python
def gsm8k_to_chat(question, reasoning, final_answer):
    thought = reasoning.strip()
    answer = f"The answer is {final_answer.strip()}."
    return format_example(question, thought, answer)

fake = gsm8k_to_chat(
    "Roger has 5 balls. He buys 2 more. How many?",
    "Start with 5. Add 2. Total 7.",
    "7",
)
print(fake)
```

## Multi-Turn Conversation Formatting

Real conversations have multiple exchanges. Our template must handle arbitrary numbers of turns while maintaining consistent structure. Each user-assistant pair forms a "turn."

```python
def apply_template(messages):
    """Format a multi-turn conversation.

    messages: list of dict(role, content, thought=None)
    """
    out = []
    for m in messages:
        if m["role"] == "user":
            out.append(f"{SPECIAL['user']}{m['content']}{SPECIAL['end']}")
        else:
            thought = m.get("thought", "")
            if thought:
                body = f"{SPECIAL['thought']}{thought}{SPECIAL['end']}{m['content']}"
            else:
                body = m["content"]
            out.append(f"{SPECIAL['assistant']}{body}{SPECIAL['end']}")
    return "".join(out)

multi_turn = apply_template([
    {"role": "user", "content": "What is the derivative of x^2?"},
    {"role": "assistant", "thought": "Power rule: d/dx x^n = n*x^(n-1)", "content": "The derivative of x² is 2x."},
    {"role": "user", "content": "What about x^3?"},
    {"role": "assistant", "thought": "Same rule: 3*x^2", "content": "The derivative of x³ is 3x²."},
    {"role": "user", "content": "Can you generalize?"},
    {"role": "assistant", "thought": "The general power rule applies for any n.", "content": "For any x^n, the derivative is n·x^(n-1). This is called the power rule."},
])
print("=== Multi-turn formatted ===")
print(multi_turn[:200], "...")
print(f"\nTotal characters: {len(multi_turn)}")
```

## Template Registry for Training Code

```python
CHAT_TEMPLATE = {
    "roles": ("user", "assistant"),
    "thought_tag": SPECIAL["thought"],
    "stop": SPECIAL["end"],
}

print(apply_template([
    {"role": "user", "content": "Hi"},
    {"role": "assistant", "thought": "Greet back.", "content": "Hello!"},
]))
```

## Extracting Role Spans for Loss Masking

A critical downstream use of chat templates is building the loss mask. We need to know which token positions belong to "assistant" regions so only those contribute to training loss.

```python
def get_role_spans(text, special_tokens):
    """Return a list of (start_char, end_char, role) tuples."""
    import re
    spans = []
    user_marker = special_tokens["user"]
    asst_marker = special_tokens["assistant"]
    end_marker = special_tokens["end"]

    current_role = None
    i = 0
    role_start = 0

    while i < len(text):
        if text[i:].startswith(user_marker):
            if current_role:
                spans.append((role_start, i, current_role))
            current_role = "user"
            role_start = i
            i += len(user_marker)
        elif text[i:].startswith(asst_marker):
            if current_role:
                spans.append((role_start, i, current_role))
            current_role = "assistant"
            role_start = i
            i += len(asst_marker)
        elif text[i:].startswith(end_marker) and current_role == "user":
            i += len(end_marker)
            spans.append((role_start, i, current_role))
            current_role = None
        else:
            i += 1

    if current_role:
        spans.append((role_start, len(text), current_role))

    return spans

spans = get_role_spans(example, SPECIAL)
for start, end, role in spans:
    preview = example[start:end][:40]
    print(f"  [{role:10s}] chars {start:3d}-{end:3d}: {preview}...")
```

## Exercise: Format a Multi-Turn Math Tutoring Conversation

Format the following 4-turn math tutoring session using `apply_template`. The assistant should use the thought tag to show intermediate reasoning before giving each answer.

**Conversation:**
1. Student: "What is 15% of 80?"
2. Tutor: (thinks: 15/100 * 80 = 12) "15% of 80 is 12."
3. Student: "And what is 20% of the result?"
4. Tutor: (thinks: 20/100 * 12 = 2.4) "20% of 12 is 2.4."

```python
tutoring_conversation = apply_template([
    {"role": "user", "content": "What is 15% of 80?"},
    {"role": "assistant", "thought": "15/100 * 80 = 0.15 * 80 = 12", "content": "15% of 80 is 12."},
    {"role": "user", "content": "And what is 20% of the result?"},
    {"role": "assistant", "thought": "20/100 * 12 = 0.20 * 12 = 2.4", "content": "20% of 12 is 2.4."},
])
print("=== Tutoring Session ===")
print(tutoring_conversation)

assert SPECIAL["thought"] in tutoring_conversation, "Missing thought tags!"
assert tutoring_conversation.count(SPECIAL["user"]) == 2, "Should have 2 user turns"
assert tutoring_conversation.count(SPECIAL["assistant"]) == 2, "Should have 2 assistant turns"
print("\nAll assertions passed.")
```

## Where This Leads Next

Now that you can serialize a conversation and mark which tokens belong to the assistant, the next question is how to *train* only on those assistant tokens. Section 2.2 builds the **masked loss function**, which uses exactly the "Train?" column from the token table above to ignore user tokens during learning.

---

## Key Takeaway

Chat templates are the **serialization protocol** between humans and language models. Without them, a model has no way to distinguish "what the user said" from "what I should generate." The template defines:

1. **Role boundaries** — the model knows when to start and stop generating
2. **Thought structure** — chain-of-thought reasoning is captured in a parseable span
3. **Training signals** — downstream loss masking depends on correct template formatting
4. **Multi-turn context** — conversation history is packed into a single token sequence with unambiguous structure

The template you choose at training time **must** match what you use at inference time. A mismatch (e.g., training with ChatML but prompting with Llama-style) will produce garbage outputs regardless of model quality.

## Further Reading (Optional)

**Optional — you do NOT need these to continue. They are for curious students who want the original sources.**

- Ouyang et al. (2022). *Training language models to follow instructions with human feedback (InstructGPT)*. NeurIPS.
- Taori et al. (2023). *Stanford Alpaca: An Instruction-following LLaMA model*. Stanford CRFM.
