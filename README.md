# agent-context-trim

Trim LLM message arrays to fit within a token budget.

Removes oldest messages (excluding system prompts by default) until the estimated token count falls below a target budget. The token estimator is pluggable — the built-in default uses a word-based heuristic.

- Zero runtime dependencies (pure standard library).
- Ships type hints (PEP 561 `py.typed`).
- Works with OpenAI/Anthropic-style `{"role", "content"}` message dicts, including multi-modal content blocks and `content=None` tool-call messages.

## Install

```bash
pip install agent-context-trim
```

## Usage

```python
from agent_context_trim import AgentContextTrim

trimmer = AgentContextTrim(budget=2000)

messages = [
    {"role": "system",    "content": "Be helpful."},
    {"role": "user",      "content": "Hello"},
    {"role": "assistant", "content": "Hi!"},
    # ... many more turns ...
]

result = trimmer.trim(messages)
print(result.messages)         # trimmed list (deep copy)
print(result.dropped)          # number of messages removed
print(result.estimated_tokens) # estimated token count after trim
```

## API

### `AgentContextTrim(budget, *, estimator=None, keep_system=True, keep_first=0, keep_last=1)`

| Parameter | Description |
|-----------|-------------|
| `budget` | Target token budget. Messages are dropped until estimated count is at or below this value. |
| `estimator` | `(message: dict) -> int` callable. Defaults to a word-based heuristic. |
| `keep_system` | When `True` (default), `role="system"` messages are never dropped. |
| `keep_first` | Number of non-system messages to always keep from the front. Default `0`. |
| `keep_last` | Number of non-system messages to always keep from the tail. Default `1`. |

### Methods

| Method | Returns | Description |
|--------|---------|-------------|
| `trim(messages, *, budget=None)` | `TrimResult` | Trim messages to fit within budget. |
| `estimate(messages)` | `int` | Estimated token count (no trimming). |
| `fits(messages, *, budget=None)` | `bool` | Whether messages already fit within budget. |
| `would_drop(messages, *, budget=None)` | `int` | How many messages *would* be dropped. |

### `TrimResult`

| Attribute | Type | Description |
|-----------|------|-------------|
| `messages` | `list[dict]` | Trimmed message list (deep copy). |
| `dropped` | `int` | Number of messages removed. |
| `estimated_tokens` | `int` | Estimated token count after trim. |
| `original_count` | `int` | Number of messages before trimming. |

## Custom estimator

```python
import tiktoken

enc = tiktoken.encoding_for_model("gpt-4o")

def tiktoken_estimator(message: dict) -> int:
    content = message.get("content", "") or ""
    if isinstance(content, list):
        content = " ".join(b.get("text", "") for b in content if isinstance(b, dict))
    return len(enc.encode(content)) + 4  # +4 for role overhead

trimmer = AgentContextTrim(budget=4096, estimator=tiktoken_estimator)
```

## Behaviour notes

- Messages are dropped oldest-first from the conversation; the surviving
  messages keep their original relative order.
- System messages are pinned by default (`keep_system=True`) and are still
  counted toward the budget even though they are never dropped. If the pinned
  messages alone exceed the budget, nothing droppable is left and the result
  may still be over budget — `trim` never removes protected messages.
- The default estimator is intentionally tolerant of message shape: a plain
  string, a `None` content (assistant tool-call turns), or a list of
  multi-modal content blocks are all handled without raising.

## Running the tests

The test suite uses only the Python standard library (`unittest`), so no
third-party test runner is required:

```bash
python3 -m unittest discover -s tests
```

## License

MIT
