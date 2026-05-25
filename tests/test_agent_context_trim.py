"""Tests for agent_context_trim."""

from __future__ import annotations

import pytest

from agent_context_trim import AgentContextTrim, TrimResult

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_msg(role: str, content: str) -> dict:
    return {"role": role, "content": content}


def _sys(content: str = "Be helpful.") -> dict:
    return _make_msg("system", content)


def _user(content: str = "Hello") -> dict:
    return _make_msg("user", content)


def _asst(content: str = "Hi!") -> dict:
    return _make_msg("assistant", content)


# A fixed-token estimator for deterministic tests: each message = 10 tokens
def _fixed_10(_msg: dict) -> int:
    return 10


# ---------------------------------------------------------------------------
# Constructor / repr
# ---------------------------------------------------------------------------


def test_repr():
    t = AgentContextTrim(budget=1000)
    assert "budget=1000" in repr(t)
    assert "keep_system=True" in repr(t)


def test_budget_property():
    assert AgentContextTrim(budget=500).budget == 500


def test_invalid_budget():
    with pytest.raises(ValueError):
        AgentContextTrim(budget=0)


def test_invalid_keep_first():
    with pytest.raises(ValueError):
        AgentContextTrim(budget=100, keep_first=-1)


def test_invalid_keep_last():
    with pytest.raises(ValueError):
        AgentContextTrim(budget=100, keep_last=-1)


# ---------------------------------------------------------------------------
# TrimResult structure
# ---------------------------------------------------------------------------


def test_trim_result_structure():
    t = AgentContextTrim(budget=999999, estimator=_fixed_10)
    result = t.trim([_user(), _asst()])
    assert isinstance(result, TrimResult)
    assert isinstance(result.messages, list)
    assert result.original_count == 2
    assert result.dropped == 0
    assert result.estimated_tokens > 0


# ---------------------------------------------------------------------------
# trim — no-op when already under budget
# ---------------------------------------------------------------------------


def test_no_trim_when_under_budget():
    t = AgentContextTrim(budget=999999, estimator=_fixed_10)
    msgs = [_user(), _asst()]
    result = t.trim(msgs)
    assert result.dropped == 0
    assert len(result.messages) == 2


def test_empty_messages():
    t = AgentContextTrim(budget=100, estimator=_fixed_10)
    result = t.trim([])
    assert result.messages == []
    assert result.dropped == 0
    assert result.estimated_tokens == 0


# ---------------------------------------------------------------------------
# trim — drops oldest non-system first
# ---------------------------------------------------------------------------


def test_drops_oldest_first():
    # budget = 30 tokens, each message = 10
    # 4 messages = 40 tokens; need to drop 1
    t = AgentContextTrim(budget=30, estimator=_fixed_10, keep_last=0)
    msgs = [_user("A"), _asst("B"), _user("C"), _asst("D")]
    result = t.trim(msgs)
    assert result.dropped == 1
    # First message (A) dropped
    contents = [m["content"] for m in result.messages]
    assert "A" not in contents
    assert "D" in contents


def test_drops_until_under_budget():
    # budget = 20 tokens, each = 10, 5 messages = 50 tokens
    t = AgentContextTrim(budget=20, estimator=_fixed_10, keep_last=0)
    msgs = [_user(str(i)) for i in range(5)]
    result = t.trim(msgs)
    assert result.estimated_tokens <= 20
    assert result.dropped == 3


# ---------------------------------------------------------------------------
# system messages preserved
# ---------------------------------------------------------------------------


def test_system_message_preserved():
    t = AgentContextTrim(budget=10, estimator=_fixed_10, keep_last=0)
    msgs = [_sys(), _user("A"), _user("B")]
    result = t.trim(msgs)
    roles = [m["role"] for m in result.messages]
    assert "system" in roles


def test_system_message_not_preserved_when_disabled():
    t = AgentContextTrim(budget=10, estimator=_fixed_10, keep_system=False, keep_last=0)
    msgs = [_sys("long system prompt"), _user("A"), _user("B")]
    result = t.trim(msgs)
    # System is now droppable
    assert result.dropped >= 1
    # With keep_system=False, system can be dropped
    assert len(result.messages) <= 3


# ---------------------------------------------------------------------------
# keep_last
# ---------------------------------------------------------------------------


def test_keep_last_protects_tail():
    # budget = 10, each msg = 10; 3 messages = 30 tokens
    # keep_last=1 protects the last message
    t = AgentContextTrim(budget=10, estimator=_fixed_10, keep_last=1)
    msgs = [_user("old1"), _user("old2"), _asst("keep me")]
    result = t.trim(msgs)
    assert result.messages[-1]["content"] == "keep me"


def test_keep_last_zero():
    # With keep_last=0, the last message is droppable too
    t = AgentContextTrim(budget=5, estimator=_fixed_10, keep_last=0)
    msgs = [_user("A"), _asst("B")]
    result = t.trim(msgs)
    assert result.dropped >= 1


# ---------------------------------------------------------------------------
# keep_first
# ---------------------------------------------------------------------------


def test_keep_first_protects_head():
    # budget = 10, each = 10; keep_first=1 protects first conv msg
    t = AgentContextTrim(budget=10, estimator=_fixed_10, keep_first=1, keep_last=0)
    msgs = [_user("first"), _asst("second"), _user("third")]
    result = t.trim(msgs)
    contents = [m["content"] for m in result.messages]
    assert "first" in contents


# ---------------------------------------------------------------------------
# Budget override per call
# ---------------------------------------------------------------------------


def test_budget_override():
    t = AgentContextTrim(budget=999, estimator=_fixed_10, keep_last=0)
    msgs = [_user("A"), _asst("B"), _user("C")]  # 30 tokens
    result = t.trim(msgs, budget=20)
    assert result.dropped == 1


# ---------------------------------------------------------------------------
# estimate / fits / would_drop
# ---------------------------------------------------------------------------


def test_estimate():
    t = AgentContextTrim(budget=100, estimator=_fixed_10)
    msgs = [_user(), _asst(), _user()]
    assert t.estimate(msgs) == 30


def test_fits_true():
    t = AgentContextTrim(budget=100, estimator=_fixed_10)
    assert t.fits([_user(), _asst()]) is True


def test_fits_false():
    t = AgentContextTrim(budget=5, estimator=_fixed_10)
    assert t.fits([_user()]) is False


def test_would_drop():
    t = AgentContextTrim(budget=20, estimator=_fixed_10, keep_last=0)
    msgs = [_user("A"), _asst("B"), _user("C")]
    assert t.would_drop(msgs) == 1


def test_would_drop_none():
    t = AgentContextTrim(budget=999, estimator=_fixed_10)
    assert t.would_drop([_user(), _asst()]) == 0


# ---------------------------------------------------------------------------
# Deep copy behaviour
# ---------------------------------------------------------------------------


def test_result_messages_deep_copied():
    t = AgentContextTrim(budget=999, estimator=_fixed_10)
    msgs = [_user("original")]
    result = t.trim(msgs)
    result.messages[0]["content"] = "mutated"
    assert t.trim(msgs).messages[0]["content"] == "original"


# ---------------------------------------------------------------------------
# Default estimator smoke tests
# ---------------------------------------------------------------------------


def test_default_estimator_non_zero():
    t = AgentContextTrim(budget=999999)
    msgs = [_user("Hello there, how are you today?")]
    assert t.estimate(msgs) > 0


def test_default_estimator_empty_content():
    t = AgentContextTrim(budget=999999)
    assert t.estimate([{"role": "user", "content": ""}]) > 0


def test_default_estimator_list_content():
    t = AgentContextTrim(budget=999999)
    msg = {"role": "user", "content": [{"type": "text", "text": "hello world"}]}
    assert t.estimate([msg]) > 0
