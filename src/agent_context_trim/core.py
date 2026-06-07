"""Trim LLM message arrays to fit within a token budget.

The trimmer removes the oldest messages (excluding system) until the
estimated token count falls below a target budget.  The token estimator
is pluggable — the default uses a word-based heuristic (1 token ~= 0.75 words).

Example::

    from agent_context_trim import AgentContextTrim

    trimmer = AgentContextTrim(budget=2000)

    messages = [
        {"role": "system",    "content": "Be helpful."},
        {"role": "user",      "content": "Hello"},
        {"role": "assistant", "content": "Hi!"},
        # ... many more turns ...
    ]

    result = trimmer.trim(messages)
    print(result.messages)        # trimmed list
    print(result.dropped)         # number of messages removed
    print(result.estimated_tokens)
"""

from __future__ import annotations

import copy
from dataclasses import dataclass
from typing import Any, Callable


def _default_estimator(message: dict[str, Any]) -> int:
    """Estimate the token count of a single message.

    Uses a simple heuristic: ``1 token ~= 0.75 words`` plus a flat
    overhead of ``4`` tokens to approximate the per-message role and
    delimiter tokens that most chat APIs add.

    The estimator is deliberately forgiving about message shape so it can
    be applied directly to messages from a variety of providers:

    * ``content`` may be a plain string.
    * ``content`` may be ``None`` (e.g. an assistant message that only
      carries ``tool_calls``); this contributes no text, only overhead.
    * ``content`` may be a list of multi-modal content blocks; only the
      ``text`` field of each ``dict`` block is counted, and non-string
      ``text`` values are ignored.

    Args:
        message: A single ``{"role": ..., "content": ...}`` mapping.

    Returns:
        The estimated number of tokens for *message* (always ``>= 4``).
    """
    content = message.get("content", "")
    if content is None:
        text = ""
    elif isinstance(content, str):
        text = content
    elif isinstance(content, list):
        # Multi-modal content block list — count text blocks only.
        parts: list[str] = []
        for block in content:
            if isinstance(block, dict):
                block_text = block.get("text", "")
                if isinstance(block_text, str):
                    parts.append(block_text)
        text = " ".join(parts)
    else:
        text = str(content)
    words = len(text.split())
    overhead = 4
    if words == 0:
        return overhead
    return int(words / 0.75) + overhead


@dataclass
class TrimResult:
    """Result of a trim operation.

    Attributes:
        messages:         The trimmed message list (deep copy).
        dropped:          Number of messages removed.
        estimated_tokens: Estimated token count of the trimmed list.
        original_count:   Number of messages before trimming.
    """

    messages: list[dict[str, Any]]
    dropped: int
    estimated_tokens: int
    original_count: int


class AgentContextTrim:
    """Trim a message list to fit within a token budget.

    Args:
        budget:    Target token budget.  Messages are dropped until the
                   estimated count is at or below this value.
        estimator: Callable ``(message: dict) -> int`` returning estimated
                   tokens for a single message.  Defaults to the built-in
                   word-based heuristic.
        keep_system: When ``True`` (default), messages with ``role="system"``
                     are never dropped.
        keep_first:  Number of non-system messages to always keep from the
                     front of the list, regardless of budget.  Defaults to 0.
        keep_last:   Number of non-system messages to always keep from the
                     tail of the list.  Defaults to 1 (keep the latest turn).
    """

    def __init__(
        self,
        budget: int,
        *,
        estimator: Callable[[dict[str, Any]], int] | None = None,
        keep_system: bool = True,
        keep_first: int = 0,
        keep_last: int = 1,
    ) -> None:
        if budget < 1:
            raise ValueError(f"budget must be >= 1, got {budget}")
        if keep_first < 0:
            raise ValueError(f"keep_first must be >= 0, got {keep_first}")
        if keep_last < 0:
            raise ValueError(f"keep_last must be >= 0, got {keep_last}")
        self._budget = budget
        self._estimator = estimator or _default_estimator
        self._keep_system = keep_system
        self._keep_first = keep_first
        self._keep_last = keep_last

    # ------------------------------------------------------------------
    # Core API
    # ------------------------------------------------------------------

    def trim(
        self,
        messages: list[dict[str, Any]],
        *,
        budget: int | None = None,
    ) -> TrimResult:
        """Trim *messages* to fit within the token budget.

        Args:
            messages: List of ``{"role": ..., "content": ...}`` dicts.
            budget:   Override the instance-level budget for this call.

        Returns:
            :class:`TrimResult` with the trimmed list and statistics.
        """
        target = budget if budget is not None else self._budget
        original_count = len(messages)

        # Separate pinned and droppable messages
        system_msgs: list[tuple[int, dict[str, Any]]] = []
        conv_msgs: list[tuple[int, dict[str, Any]]] = []
        for i, m in enumerate(messages):
            if self._keep_system and m.get("role") == "system":
                system_msgs.append((i, m))
            else:
                conv_msgs.append((i, m))

        # Identify protected conversation messages
        first_protected = set(range(min(self._keep_first, len(conv_msgs))))
        last_protected = set(
            range(max(0, len(conv_msgs) - self._keep_last), len(conv_msgs))
        )
        protected = first_protected | last_protected

        # Build a working list: (original_index, msg, droppable)
        droppable_indices: list[int] = []
        for j, (_, _) in enumerate(conv_msgs):
            if j not in protected:
                droppable_indices.append(j)

        # Drop from the front of droppable messages until under budget
        dropped_set: set[int] = set()  # indices into conv_msgs
        kept_msgs = list(messages)  # working copy

        def _estimate_all(msgs: list[dict[str, Any]]) -> int:
            return sum(self._estimator(m) for m in msgs)

        current = _estimate_all(kept_msgs)
        drop_ptr = 0
        while current > target and drop_ptr < len(droppable_indices):
            j = droppable_indices[drop_ptr]
            orig_idx = conv_msgs[j][0]
            dropped_set.add(orig_idx)
            removed = self._estimator(messages[orig_idx])
            current -= removed
            drop_ptr += 1

        # Rebuild the final list preserving original order
        final = [m for i, m in enumerate(messages) if i not in dropped_set]
        return TrimResult(
            messages=copy.deepcopy(final),
            dropped=len(dropped_set),
            estimated_tokens=_estimate_all(final),
            original_count=original_count,
        )

    def estimate(self, messages: list[dict[str, Any]]) -> int:
        """Return the estimated token count for *messages*."""
        return sum(self._estimator(m) for m in messages)

    def fits(
        self, messages: list[dict[str, Any]], *, budget: int | None = None
    ) -> bool:
        """Return ``True`` if *messages* already fit within the budget."""
        target = budget if budget is not None else self._budget
        return self.estimate(messages) <= target

    def would_drop(
        self, messages: list[dict[str, Any]], *, budget: int | None = None
    ) -> int:
        """Return how many messages would be dropped without actually trimming."""
        return self.trim(messages, budget=budget).dropped

    @property
    def budget(self) -> int:
        """The configured token budget."""
        return self._budget

    def __repr__(self) -> str:
        return (
            f"AgentContextTrim(budget={self._budget}, "
            f"keep_system={self._keep_system}, "
            f"keep_last={self._keep_last})"
        )
