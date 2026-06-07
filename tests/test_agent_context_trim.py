"""Tests for :mod:`agent_context_trim`.

Written against the standard-library :mod:`unittest` framework so the suite
runs with zero third-party dependencies::

    python3 -m unittest discover -s tests
"""

from __future__ import annotations

import os
import sys
import unittest

# Allow running directly from a checkout (src layout) without installing the
# package first, e.g. ``python3 -m unittest discover -s tests``.
_SRC = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "src")
if os.path.isdir(_SRC) and _SRC not in sys.path:
    sys.path.insert(0, _SRC)

from agent_context_trim import AgentContextTrim, TrimResult  # noqa: E402
from agent_context_trim.core import _default_estimator  # noqa: E402

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_msg(role: str, content) -> dict:
    return {"role": role, "content": content}


def _sys(content: str = "Be helpful.") -> dict:
    return _make_msg("system", content)


def _user(content: str = "Hello") -> dict:
    return _make_msg("user", content)


def _asst(content: str = "Hi!") -> dict:
    return _make_msg("assistant", content)


# A fixed-token estimator for deterministic tests: each message = 10 tokens.
def _fixed_10(_msg: dict) -> int:
    return 10


# ---------------------------------------------------------------------------
# Constructor / repr
# ---------------------------------------------------------------------------


class ConstructorTests(unittest.TestCase):
    def test_repr(self):
        t = AgentContextTrim(budget=1000)
        self.assertIn("budget=1000", repr(t))
        self.assertIn("keep_system=True", repr(t))

    def test_budget_property(self):
        self.assertEqual(AgentContextTrim(budget=500).budget, 500)

    def test_invalid_budget(self):
        with self.assertRaises(ValueError):
            AgentContextTrim(budget=0)

    def test_invalid_budget_negative(self):
        with self.assertRaises(ValueError):
            AgentContextTrim(budget=-5)

    def test_invalid_keep_first(self):
        with self.assertRaises(ValueError):
            AgentContextTrim(budget=100, keep_first=-1)

    def test_invalid_keep_last(self):
        with self.assertRaises(ValueError):
            AgentContextTrim(budget=100, keep_last=-1)


# ---------------------------------------------------------------------------
# TrimResult structure
# ---------------------------------------------------------------------------


class TrimResultTests(unittest.TestCase):
    def test_trim_result_structure(self):
        t = AgentContextTrim(budget=999999, estimator=_fixed_10)
        result = t.trim([_user(), _asst()])
        self.assertIsInstance(result, TrimResult)
        self.assertIsInstance(result.messages, list)
        self.assertEqual(result.original_count, 2)
        self.assertEqual(result.dropped, 0)
        self.assertGreater(result.estimated_tokens, 0)


# ---------------------------------------------------------------------------
# trim — no-op when already under budget
# ---------------------------------------------------------------------------


class NoTrimTests(unittest.TestCase):
    def test_no_trim_when_under_budget(self):
        t = AgentContextTrim(budget=999999, estimator=_fixed_10)
        msgs = [_user(), _asst()]
        result = t.trim(msgs)
        self.assertEqual(result.dropped, 0)
        self.assertEqual(len(result.messages), 2)

    def test_at_budget_boundary_is_kept(self):
        # 3 messages * 10 = 30 tokens, budget exactly 30 -> nothing dropped.
        t = AgentContextTrim(budget=30, estimator=_fixed_10, keep_last=0)
        result = t.trim([_user("A"), _asst("B"), _user("C")])
        self.assertEqual(result.dropped, 0)
        self.assertEqual(result.estimated_tokens, 30)

    def test_empty_messages(self):
        t = AgentContextTrim(budget=100, estimator=_fixed_10)
        result = t.trim([])
        self.assertEqual(result.messages, [])
        self.assertEqual(result.dropped, 0)
        self.assertEqual(result.estimated_tokens, 0)
        self.assertEqual(result.original_count, 0)


# ---------------------------------------------------------------------------
# trim — drops oldest non-system first
# ---------------------------------------------------------------------------


class DropOrderTests(unittest.TestCase):
    def test_drops_oldest_first(self):
        # budget = 30 tokens, each message = 10; 4 messages = 40 -> drop 1.
        t = AgentContextTrim(budget=30, estimator=_fixed_10, keep_last=0)
        msgs = [_user("A"), _asst("B"), _user("C"), _asst("D")]
        result = t.trim(msgs)
        self.assertEqual(result.dropped, 1)
        contents = [m["content"] for m in result.messages]
        self.assertNotIn("A", contents)
        self.assertIn("D", contents)

    def test_drops_until_under_budget(self):
        # budget = 20 tokens, each = 10, 5 messages = 50 tokens.
        t = AgentContextTrim(budget=20, estimator=_fixed_10, keep_last=0)
        msgs = [_user(str(i)) for i in range(5)]
        result = t.trim(msgs)
        self.assertLessEqual(result.estimated_tokens, 20)
        self.assertEqual(result.dropped, 3)

    def test_remaining_order_preserved(self):
        t = AgentContextTrim(budget=20, estimator=_fixed_10, keep_last=0)
        msgs = [_user("A"), _user("B"), _user("C"), _user("D")]
        result = t.trim(msgs)
        contents = [m["content"] for m in result.messages]
        # Surviving messages keep their original relative order.
        self.assertEqual(contents, ["C", "D"])


# ---------------------------------------------------------------------------
# system messages preserved
# ---------------------------------------------------------------------------


class SystemMessageTests(unittest.TestCase):
    def test_system_message_preserved(self):
        t = AgentContextTrim(budget=10, estimator=_fixed_10, keep_last=0)
        msgs = [_sys(), _user("A"), _user("B")]
        result = t.trim(msgs)
        roles = [m["role"] for m in result.messages]
        self.assertIn("system", roles)

    def test_multiple_system_messages_preserved(self):
        t = AgentContextTrim(budget=10, estimator=_fixed_10, keep_last=0)
        msgs = [_sys("s1"), _user("A"), _sys("s2"), _user("B")]
        result = t.trim(msgs)
        sys_contents = [m["content"] for m in result.messages if m["role"] == "system"]
        self.assertEqual(sys_contents, ["s1", "s2"])

    def test_system_message_not_preserved_when_disabled(self):
        t = AgentContextTrim(
            budget=10, estimator=_fixed_10, keep_system=False, keep_last=0
        )
        msgs = [_sys("long system prompt"), _user("A"), _user("B")]
        result = t.trim(msgs)
        self.assertGreaterEqual(result.dropped, 1)
        self.assertLessEqual(len(result.messages), 3)


# ---------------------------------------------------------------------------
# keep_last
# ---------------------------------------------------------------------------


class KeepLastTests(unittest.TestCase):
    def test_keep_last_protects_tail(self):
        t = AgentContextTrim(budget=10, estimator=_fixed_10, keep_last=1)
        msgs = [_user("old1"), _user("old2"), _asst("keep me")]
        result = t.trim(msgs)
        self.assertEqual(result.messages[-1]["content"], "keep me")

    def test_keep_last_multiple(self):
        t = AgentContextTrim(budget=10, estimator=_fixed_10, keep_last=2)
        msgs = [_user("a"), _user("b"), _user("c"), _user("d")]
        result = t.trim(msgs)
        contents = [m["content"] for m in result.messages]
        self.assertIn("c", contents)
        self.assertIn("d", contents)

    def test_keep_last_zero(self):
        t = AgentContextTrim(budget=5, estimator=_fixed_10, keep_last=0)
        msgs = [_user("A"), _asst("B")]
        result = t.trim(msgs)
        self.assertGreaterEqual(result.dropped, 1)


# ---------------------------------------------------------------------------
# keep_first
# ---------------------------------------------------------------------------


class KeepFirstTests(unittest.TestCase):
    def test_keep_first_protects_head(self):
        t = AgentContextTrim(budget=10, estimator=_fixed_10, keep_first=1, keep_last=0)
        msgs = [_user("first"), _asst("second"), _user("third")]
        result = t.trim(msgs)
        contents = [m["content"] for m in result.messages]
        self.assertIn("first", contents)

    def test_keep_first_and_last_protect_both_ends(self):
        # Drop only from the middle.
        t = AgentContextTrim(budget=20, estimator=_fixed_10, keep_first=1, keep_last=1)
        msgs = [_user("first"), _user("mid1"), _user("mid2"), _user("last")]
        result = t.trim(msgs)
        contents = [m["content"] for m in result.messages]
        self.assertIn("first", contents)
        self.assertIn("last", contents)

    def test_protections_can_exceed_budget(self):
        # When everything is pinned, nothing is dropped even if over budget.
        t = AgentContextTrim(budget=5, estimator=_fixed_10, keep_first=2, keep_last=2)
        msgs = [_user("A"), _asst("B"), _user("C")]
        result = t.trim(msgs)
        self.assertEqual(result.dropped, 0)
        self.assertEqual(len(result.messages), 3)


# ---------------------------------------------------------------------------
# Budget override per call
# ---------------------------------------------------------------------------


class BudgetOverrideTests(unittest.TestCase):
    def test_budget_override(self):
        t = AgentContextTrim(budget=999, estimator=_fixed_10, keep_last=0)
        msgs = [_user("A"), _asst("B"), _user("C")]  # 30 tokens
        result = t.trim(msgs, budget=20)
        self.assertEqual(result.dropped, 1)

    def test_instance_budget_unchanged_after_override(self):
        t = AgentContextTrim(budget=999, estimator=_fixed_10)
        t.trim([_user("A")], budget=5)
        self.assertEqual(t.budget, 999)


# ---------------------------------------------------------------------------
# estimate / fits / would_drop
# ---------------------------------------------------------------------------


class HelperMethodTests(unittest.TestCase):
    def test_estimate(self):
        t = AgentContextTrim(budget=100, estimator=_fixed_10)
        msgs = [_user(), _asst(), _user()]
        self.assertEqual(t.estimate(msgs), 30)

    def test_fits_true(self):
        t = AgentContextTrim(budget=100, estimator=_fixed_10)
        self.assertTrue(t.fits([_user(), _asst()]))

    def test_fits_false(self):
        t = AgentContextTrim(budget=5, estimator=_fixed_10)
        self.assertFalse(t.fits([_user()]))

    def test_fits_budget_override(self):
        t = AgentContextTrim(budget=5, estimator=_fixed_10)
        self.assertTrue(t.fits([_user()], budget=10))

    def test_would_drop(self):
        t = AgentContextTrim(budget=20, estimator=_fixed_10, keep_last=0)
        msgs = [_user("A"), _asst("B"), _user("C")]
        self.assertEqual(t.would_drop(msgs), 1)

    def test_would_drop_none(self):
        t = AgentContextTrim(budget=999, estimator=_fixed_10)
        self.assertEqual(t.would_drop([_user(), _asst()]), 0)

    def test_would_drop_does_not_mutate_input(self):
        t = AgentContextTrim(budget=20, estimator=_fixed_10, keep_last=0)
        msgs = [_user("A"), _asst("B"), _user("C")]
        before = [dict(m) for m in msgs]
        t.would_drop(msgs)
        self.assertEqual(msgs, before)


# ---------------------------------------------------------------------------
# Deep copy behaviour
# ---------------------------------------------------------------------------


class DeepCopyTests(unittest.TestCase):
    def test_result_messages_deep_copied(self):
        t = AgentContextTrim(budget=999, estimator=_fixed_10)
        msgs = [_user("original")]
        result = t.trim(msgs)
        result.messages[0]["content"] = "mutated"
        self.assertEqual(t.trim(msgs).messages[0]["content"], "original")

    def test_trim_does_not_mutate_source_list(self):
        t = AgentContextTrim(budget=20, estimator=_fixed_10, keep_last=0)
        msgs = [_user("A"), _user("B"), _user("C")]
        t.trim(msgs)
        self.assertEqual(len(msgs), 3)


# ---------------------------------------------------------------------------
# Default estimator
# ---------------------------------------------------------------------------


class DefaultEstimatorTests(unittest.TestCase):
    def test_default_estimator_non_zero(self):
        t = AgentContextTrim(budget=999999)
        msgs = [_user("Hello there, how are you today?")]
        self.assertGreater(t.estimate(msgs), 0)

    def test_default_estimator_empty_content(self):
        t = AgentContextTrim(budget=999999)
        # Empty content contributes only the per-message overhead.
        self.assertEqual(t.estimate([{"role": "user", "content": ""}]), 4)

    def test_default_estimator_none_content(self):
        # Assistant messages that only carry tool_calls have content=None;
        # this must not raise and should count only the overhead.
        self.assertEqual(_default_estimator({"role": "assistant", "content": None}), 4)

    def test_default_estimator_list_content(self):
        t = AgentContextTrim(budget=999999)
        msg = {"role": "user", "content": [{"type": "text", "text": "hello world"}]}
        self.assertGreater(t.estimate([msg]), 4)

    def test_default_estimator_list_with_non_text_block(self):
        # Image blocks (no usable text) must not raise.
        msg = {
            "role": "user",
            "content": [
                {"type": "image", "source": {"data": "..."}},
                {"type": "text", "text": "describe this"},
            ],
        }
        self.assertGreater(_default_estimator(msg), 4)

    def test_default_estimator_list_with_none_text(self):
        # A block whose "text" is None must be ignored, not crash.
        msg = {"role": "user", "content": [{"type": "text", "text": None}]}
        self.assertEqual(_default_estimator(msg), 4)

    def test_default_estimator_missing_content_key(self):
        self.assertEqual(_default_estimator({"role": "user"}), 4)

    def test_default_estimator_word_scaling(self):
        # Longer text yields a larger estimate.
        short = _default_estimator({"role": "user", "content": "one two"})
        long = _default_estimator({"role": "user", "content": "one two three four five six"})
        self.assertGreater(long, short)


# ---------------------------------------------------------------------------
# Custom estimator wiring
# ---------------------------------------------------------------------------


class CustomEstimatorTests(unittest.TestCase):
    def test_custom_estimator_used(self):
        calls = []

        def estimator(msg):
            calls.append(msg)
            return 7

        t = AgentContextTrim(budget=100, estimator=estimator)
        self.assertEqual(t.estimate([_user(), _asst()]), 14)
        self.assertEqual(len(calls), 2)


if __name__ == "__main__":  # pragma: no cover
    unittest.main()
