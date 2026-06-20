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

# Section 2.4: Teaching the AI to Output Python Dictionaries (JSON) for Tool Use

**Goal:** Format Glaive-style function-calling examples and validate JSON tool payloads at decode time.

## Why Tool Use Matters for LLMs

Language models are powerful reasoners but have fundamental limitations:

1. **Knowledge cutoff** — training data has a fixed date; the model cannot know today's stock price or weather
2. **No computation** — despite knowing math rules, LLMs frequently make arithmetic errors on complex expressions
3. **No world interaction** — a model cannot send emails, query databases, or call APIs on its own

**Tool use** bridges this gap. Instead of hallucinating an answer, the model learns to emit a structured JSON payload requesting that an external system perform the operation. The result is fed back, and the model incorporates it into a natural language response.

This pattern (sometimes called "function calling" or "agent actions") is how ChatGPT plugins, Claude's tool use, and open-source agent frameworks operate under the hood.

## The Tool-Use Lifecycle

The complete lifecycle has 5 steps:

| Step | Actor | Action | Example |
|------|-------|--------|---------|
| 1 | User | Asks a question | "What is 144 / 12?" |
| 2 | Model | Emits JSON tool call | `{"name": "calculator", "arguments": {"expression": "144/12"}}` |
| 3 | Executor | Runs the tool | `eval("144/12")` → `12.0` |
| 4 | System | Feeds result back | `{"role": "tool", "content": "12.0"}` |
| 5 | Model | Gives final answer | "144 divided by 12 equals 12." |

The model must learn two distinct behaviors:
- **When** to call a tool (vs. answering directly)
- **How** to format the call (valid JSON matching the tool's schema)

## Defining Tool Schemas

```python
import json
import re

TOOL_SYSTEM = (
    "You have access to the following tools. When you need to use one, "
    "respond with ONLY a JSON object matching the tool's schema.\n\n"
    "Available tools:\n"
    "1. calculator: Evaluate a mathematical expression\n"
    '   Schema: {"name": "calculator", "arguments": {"expression": "<python expr>"}}\n\n'
    "2. weather_lookup: Get current weather for a city\n"
    '   Schema: {"name": "weather_lookup", "arguments": {"city": "<city name>", "unit": "celsius|fahrenheit"}}\n'
)

TOOL_SCHEMAS = {
    "calculator": {
        "name": "calculator",
        "description": "Evaluate a mathematical Python expression",
        "parameters": {
            "expression": {"type": "string", "description": "A valid Python math expression"}
        },
        "required": ["expression"],
    },
    "weather_lookup": {
        "name": "weather_lookup",
        "description": "Look up current weather for a city",
        "parameters": {
            "city": {"type": "string", "description": "City name"},
            "unit": {"type": "string", "enum": ["celsius", "fahrenheit"], "description": "Temperature unit"},
        },
        "required": ["city"],
    },
}

print("System prompt:")
print(TOOL_SYSTEM)
```

## Formatting Tool-Use Training Examples

```python
def format_tool_example(user_query, tool_call, tool_result, final_reply):
    return {
        "messages": [
            {"role": "system", "content": TOOL_SYSTEM},
            {"role": "user", "content": user_query},
            {"role": "assistant", "content": tool_call, "type": "tool_call"},
            {"role": "tool", "content": tool_result},
            {"role": "assistant", "content": final_reply},
        ]
    }

ex_calc = format_tool_example(
    "What is 144 divided by 12?",
    '{"name": "calculator", "arguments": {"expression": "144 / 12"}}',
    "12.0",
    "144 divided by 12 equals 12.",
)

ex_weather = format_tool_example(
    "What's the weather in Tokyo?",
    '{"name": "weather_lookup", "arguments": {"city": "Tokyo", "unit": "celsius"}}',
    '{"temperature": 22, "condition": "partly cloudy", "humidity": 65}',
    "It's currently 22°C and partly cloudy in Tokyo with 65% humidity.",
)

print("=== Calculator example ===")
print(json.dumps(ex_calc, indent=2))
print("\n=== Weather example ===")
print(json.dumps(ex_weather, indent=2))
```

## Load Glaive Function-Calling Subset

```python
from datasets import load_dataset

try:
    glaive = load_dataset("glaiveai/glaive-function-calling-v2", split="train[:1%]")
    print("columns:", glaive.column_names)
    print(glaive[0])
except Exception as e:
    print("Dataset fetch optional in air-gapped labs:", e)
```

## JSON Schema Validation

A robust parser must handle messy model output — the JSON might be embedded in surrounding text, or the model might produce invalid syntax.

```python
def parse_tool_json(text):
    """Extract and validate a tool-call JSON from model output."""
    text = text.strip()
    m = re.search(r"\{.*\}", text, re.DOTALL)
    if not m:
        return None
    try:
        obj = json.loads(m.group())
    except json.JSONDecodeError:
        return None
    if "name" not in obj or "arguments" not in obj:
        return None
    return obj

def validate_against_schema(tool_call, schemas):
    """Check that a parsed tool call matches its registered schema."""
    name = tool_call.get("name")
    if name not in schemas:
        return False, f"Unknown tool: {name}"
    schema = schemas[name]
    args = tool_call.get("arguments", {})
    for req_param in schema.get("required", []):
        if req_param not in args:
            return False, f"Missing required parameter: {req_param}"
    for param_name, param_val in args.items():
        if param_name not in schema["parameters"]:
            return False, f"Unknown parameter: {param_name}"
    return True, "Valid"

good = parse_tool_json('Sure! {"name": "calculator", "arguments": {"expression": "2+2"}}')
bad = parse_tool_json("I think it is 4")
weather = parse_tool_json('{"name": "weather_lookup", "arguments": {"city": "Paris", "unit": "celsius"}}')

print("good:", good, "→", validate_against_schema(good, TOOL_SCHEMAS) if good else "N/A")
print("bad:", bad)
print("weather:", weather, "→", validate_against_schema(weather, TOOL_SCHEMAS) if weather else "N/A")

invalid_schema = parse_tool_json('{"name": "calculator", "arguments": {}}')
print("missing param:", validate_against_schema(invalid_schema, TOOL_SCHEMAS) if invalid_schema else "N/A")
```

## Tool Executors

```python
def run_calculator(args):
    """Safely evaluate a math expression."""
    expr = args.get("expression", "0")
    allowed = set("0123456789+-*/(). ,eE")
    if not set(expr) <= allowed:
        raise ValueError(f"Unsafe expression: {expr}")
    result = eval(expr, {"__builtins__": {}}, {})
    return str(result)

def run_weather_lookup(args):
    """Simulated weather API (returns mock data for lab purposes)."""
    city = args.get("city", "Unknown")
    unit = args.get("unit", "celsius")
    import random
    random.seed(hash(city) % 2**32)
    temp = random.randint(-10, 40)
    if unit == "fahrenheit":
        temp = int(temp * 9 / 5 + 32)
    conditions = ["sunny", "partly cloudy", "overcast", "rainy", "snowy"]
    condition = conditions[random.randint(0, len(conditions) - 1)]
    return json.dumps({
        "city": city,
        "temperature": temp,
        "unit": unit,
        "condition": condition,
        "humidity": random.randint(20, 95),
    })

TOOL_EXECUTORS = {
    "calculator": run_calculator,
    "weather_lookup": run_weather_lookup,
}

print("Calculator:", run_calculator({"expression": "12 * 8"}))
print("Weather:", run_weather_lookup({"city": "London", "unit": "celsius"}))
```

## Constrained Decoding for Valid JSON

At generation time, we want to *encourage* the model to produce valid JSON. While full constrained decoding (e.g., grammar-based sampling) is complex, we can implement a simple logit bias approach:

```python
import torch
import torch.nn.functional as F

def json_logit_bias(logits, generated_so_far, vocab_size=256):
    """Apply soft bias toward JSON-structural characters.

    This is a simplified version — production systems use finite-state
    automata to guarantee valid JSON at every step.
    """
    biased = logits.clone()
    json_chars = set('{}[]":,0123456789.- trufalsen')
    json_ids = [ord(c) % vocab_size for c in json_chars if ord(c) < vocab_size]

    if generated_so_far.startswith("{"):
        for tid in json_ids:
            biased[tid] += 2.0

    open_braces = generated_so_far.count("{") - generated_so_far.count("}")
    if open_braces > 0:
        close_id = ord("}") % vocab_size
        biased[close_id] += 1.0 * open_braces

    return biased

dummy_logits = torch.randn(256)
biased = json_logit_bias(dummy_logits, '{"name": "calc')
print("Bias increased probability of JSON chars")
print(f"Original logit[ord('}}')]: {dummy_logits[ord('}') % 256]:.3f}")
print(f"Biased  logit[ord('}}')]: {biased[ord('}') % 256]:.3f}")
```

## Full End-to-End Agent Loop

This is the complete agent loop that processes a user query through potentially multiple tool calls before producing a final answer.

```python
def agent_loop(user_query, model_outputs, max_tool_calls=5):
    """Simulate an agent loop with pre-scripted model outputs.

    In production, `model_outputs` would come from actual model generation.
    Here we pass them as a list to demonstrate the control flow.
    """
    conversation = [
        {"role": "system", "content": TOOL_SYSTEM},
        {"role": "user", "content": user_query},
    ]
    tool_call_count = 0
    final_answer = None

    for model_output in model_outputs:
        tool_call = parse_tool_json(model_output)

        if tool_call is None:
            final_answer = model_output
            break

        valid, reason = validate_against_schema(tool_call, TOOL_SCHEMAS)
        if not valid:
            conversation.append({"role": "system", "content": f"Error: {reason}"})
            continue

        tool_name = tool_call["name"]
        executor = TOOL_EXECUTORS.get(tool_name)
        if executor is None:
            conversation.append({"role": "system", "content": f"No executor for {tool_name}"})
            continue

        try:
            result = executor(tool_call["arguments"])
        except Exception as e:
            result = f"Error: {str(e)}"

        conversation.append({"role": "assistant", "content": model_output, "type": "tool_call"})
        conversation.append({"role": "tool", "name": tool_name, "content": result})
        tool_call_count += 1

        if tool_call_count >= max_tool_calls:
            final_answer = "I've reached the maximum number of tool calls."
            break

    if final_answer:
        conversation.append({"role": "assistant", "content": final_answer})

    return conversation, final_answer

scripted_outputs = [
    '{"name": "calculator", "arguments": {"expression": "(15 * 24) + (3 * 8)"}}',
    "15 times 24 is 360, plus 3 times 8 is 24. The total is 384.",
]

conversation, answer = agent_loop(
    "What is 15*24 + 3*8?",
    scripted_outputs,
)

print("=== Agent Loop Trace ===")
for msg in conversation:
    role = msg["role"]
    content = msg["content"][:80]
    msg_type = msg.get("type", "")
    print(f"  [{role:10s}] {msg_type:12s} {content}")
print(f"\nFinal answer: {answer}")
```

## Multi-Tool Agent Example

Demonstrate an agent that needs to call multiple different tools to answer a compound question.

```python
multi_tool_outputs = [
    '{"name": "weather_lookup", "arguments": {"city": "New York", "unit": "fahrenheit"}}',
    '{"name": "calculator", "arguments": {"expression": "(72 - 32) * 5 / 9"}}',
    "The weather in New York is 72°F, which converts to approximately 22.2°C.",
]

conversation, answer = agent_loop(
    "What's the temperature in New York in both Fahrenheit and Celsius?",
    multi_tool_outputs,
)

print("=== Multi-Tool Agent Trace ===")
for msg in conversation:
    role = msg["role"]
    content = msg["content"][:100]
    msg_type = msg.get("type", "")
    print(f"  [{role:10s}] {msg_type:12s} {content}")
print(f"\nFinal answer: {answer}")
```

## Training Target

Assistant turn should be **only** valid JSON when `type=tool_call`. Use masked loss on that span; grade format with `parse_tool_json` during eval.

```python
def compute_tool_accuracy(examples):
    """Evaluate what fraction of model outputs parse as valid tool calls."""
    valid = 0
    total = len(examples)
    for ex in examples:
        parsed = parse_tool_json(ex)
        if parsed is not None:
            is_valid, _ = validate_against_schema(parsed, TOOL_SCHEMAS)
            if is_valid:
                valid += 1
    return valid / total if total > 0 else 0.0

test_outputs = [
    '{"name": "calculator", "arguments": {"expression": "2+2"}}',
    '{"name": "weather_lookup", "arguments": {"city": "Paris"}}',
    "I think the answer is 4",
    '{"name": "unknown_tool", "arguments": {}}',
    '{"name": "calculator", "arguments": {"expression": "10/3"}}',
    "not json at all",
]

accuracy = compute_tool_accuracy(test_outputs)
print(f"Tool-call accuracy: {accuracy:.1%} ({int(accuracy * len(test_outputs))}/{len(test_outputs)})")
```

## Exercise: Add a `unit_converter` Tool

Implement a `unit_converter` tool that converts between common units (km↔miles, kg↔lbs, °C↔°F). You need to:
1. Define the schema
2. Implement the executor
3. Test it through the agent loop

```python
TOOL_SCHEMAS["unit_converter"] = {
    "name": "unit_converter",
    "description": "Convert between common measurement units",
    "parameters": {
        "value": {"type": "number", "description": "The numeric value to convert"},
        "from_unit": {"type": "string", "description": "Source unit (km, miles, kg, lbs, celsius, fahrenheit)"},
        "to_unit": {"type": "string", "description": "Target unit"},
    },
    "required": ["value", "from_unit", "to_unit"],
}

CONVERSION_TABLE = {
    ("km", "miles"): lambda v: v * 0.621371,
    ("miles", "km"): lambda v: v * 1.60934,
    ("kg", "lbs"): lambda v: v * 2.20462,
    ("lbs", "kg"): lambda v: v * 0.453592,
    ("celsius", "fahrenheit"): lambda v: v * 9 / 5 + 32,
    ("fahrenheit", "celsius"): lambda v: (v - 32) * 5 / 9,
}

def run_unit_converter(args):
    """Execute unit conversion."""
    value = float(args["value"])
    from_unit = args["from_unit"].lower()
    to_unit = args["to_unit"].lower()
    key = (from_unit, to_unit)
    if key not in CONVERSION_TABLE:
        return json.dumps({"error": f"Unsupported conversion: {from_unit} → {to_unit}"})
    result = CONVERSION_TABLE[key](value)
    return json.dumps({
        "value": round(result, 4),
        "from": f"{value} {from_unit}",
        "to": f"{result:.4f} {to_unit}",
    })

TOOL_EXECUTORS["unit_converter"] = run_unit_converter

converter_outputs = [
    '{"name": "unit_converter", "arguments": {"value": 100, "from_unit": "km", "to_unit": "miles"}}',
    "100 kilometers is approximately 62.14 miles.",
]

conversation, answer = agent_loop("How many miles is 100 km?", converter_outputs)
print("=== Unit Converter Test ===")
for msg in conversation:
    print(f"  [{msg['role']:10s}] {msg['content'][:80]}")
print(f"\nFinal: {answer}")

direct_result = run_unit_converter({"value": 100, "from_unit": "celsius", "to_unit": "fahrenheit"})
print(f"\n100°C → °F: {direct_result}")
```

---

## Key Takeaway

Tool use transforms LLMs from pure text generators into **agents** that can interact with the world. The key architectural decisions are:

1. **Structured output** — the model must emit valid JSON matching a predefined schema; training on Glaive-style examples teaches this format
2. **Schema validation** — always validate tool calls before execution to catch malformed outputs early
3. **Executor isolation** — tool executors run in sandboxed environments with input validation (never `eval` arbitrary strings in production)
4. **Loop architecture** — the agent loop continues until the model produces a non-tool-call response, allowing multi-step reasoning with intermediate tool results
5. **Constrained decoding** — logit biasing or grammar-based sampling at generation time increases the probability of syntactically valid JSON

The model doesn't "understand" tools — it learns statistical patterns mapping questions to JSON structures. The quality depends entirely on training data diversity and format consistency.
