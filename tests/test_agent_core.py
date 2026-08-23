"""Tests for the model-independent parts of the agent: tool-call parsing,
streamed tool-call suppression, and history truncation."""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from controlai_agent.agent import _StreamGate
from controlai_agent.toolcall import degenerate_reason, parse


class TestToolCallParsing(unittest.TestCase):
    def test_plain_call(self):
        calls, prose = parse(
            '<tool_call>\n{"name": "continuous_lqr", "arguments": {"A": [[0, 1], [-2, -3]]}}\n</tool_call>'
        )
        self.assertEqual(len(calls), 1)
        self.assertEqual(calls[0].name, "continuous_lqr")
        self.assertEqual(calls[0].arguments["A"], [[0, 1], [-2, -3]])
        self.assertEqual(prose, "")

    def test_narration_before_call_is_kept_separate(self):
        calls, prose = parse(
            'Let me compute that.\n<tool_call>\n{"name": "eigen_analysis", "arguments": {}}\n</tool_call>'
        )
        self.assertEqual([c.name for c in calls], ["eigen_analysis"])
        self.assertEqual(prose, "Let me compute that.")

    def test_truncated_call_is_recovered(self):
        """Hitting the token limit mid-call should not discard the call."""
        calls, _ = parse('<tool_call>\n{"name": "bode_analysis", "arguments": {"numerator": [10]')
        self.assertEqual(len(calls), 1)
        self.assertEqual(calls[0].arguments["numerator"], [10])

    def test_empty_tool_call_yields_no_call(self):
        """The failure mode of the old fine-tuned adapter: an empty call plus
        real prose. The prose must survive and no tool must run."""
        calls, prose = parse("<tool_call>\n\n</tool_call>\n\nThe phase margin is the answer.")
        self.assertEqual(calls, [])
        self.assertEqual(prose, "The phase margin is the answer.")

    def test_math_is_not_mangled(self):
        _, prose = parse(r"Gain margin is $6$ dB with $\zeta = 0.5$.")
        self.assertEqual(prose, r"Gain margin is $6$ dB with $\zeta = 0.5$.")

    def test_multiple_calls(self):
        calls, _ = parse(
            '<tool_call>\n{"name": "a", "arguments": {}}\n</tool_call>'
            '<tool_call>\n{"name": "b", "arguments": {}}\n</tool_call>'
        )
        self.assertEqual([c.name for c in calls], ["a", "b"])

    def test_thinking_block_stripped(self):
        _, prose = parse("<think>weighing options</think>\n\nThe answer is 3 dB.")
        self.assertEqual(prose, "The answer is 3 dB.")


class TestDegenerateGuard(unittest.TestCase):
    def test_repetition_loop_detected(self):
        self.assertIsNotNone(degenerate_reason([[0.5] * 40]))

    def test_runaway_length_detected(self):
        self.assertIsNotNone(degenerate_reason(list(range(500))))

    def test_ordinary_matrices_pass(self):
        for value in ([[0, 1], [-2, -3]], [[1.0]], [1, 6, 5, 0], [[0], [1]]):
            self.assertIsNone(degenerate_reason(value), value)

    def test_identity_matrix_passes(self):
        """A legitimate matrix full of repeated values must not be rejected."""
        self.assertIsNone(degenerate_reason([[1 if i == j else 0 for j in range(8)] for i in range(8)]))


class TestStreamGate(unittest.TestCase):
    def test_prose_passes_through(self):
        gate = _StreamGate()
        self.assertEqual(gate.feed("The gain margin "), "The gain margin ")
        self.assertEqual(gate.feed("is 6 dB."), "is 6 dB.")

    def test_tool_call_never_leaks(self):
        gate = _StreamGate()
        emitted = "".join(gate.feed(part) for part in ("Computing.", "<tool", "_call>", '{"name"', "}"))
        self.assertEqual(emitted, "Computing.")
        self.assertTrue(gate.suppressed)
        self.assertEqual(gate.flush(), "")

    def test_marker_split_across_chunks_is_held_back(self):
        gate = _StreamGate()
        self.assertEqual(gate.feed("done<too"), "done")
        self.assertEqual(gate.feed("l_call>x"), "")

    def test_partial_lookalike_is_released(self):
        """`<t` that turns out to be something else must not be swallowed."""
        gate = _StreamGate()
        gate.feed("value <t")
        self.assertEqual(gate.feed("hreshold>"), "<threshold>")


class TestHistoryTruncation(unittest.TestCase):
    def test_oldest_turns_dropped_first(self):
        from controlai_agent import agent as agent_module

        class FakeEngine:
            def count_tokens(self, text):
                return len(text.split())

        holder = object.__new__(agent_module.ControlAgent)
        holder.engine = FakeEngine()
        history = [
            {"role": "user", "content": "old " * 100},
            {"role": "assistant", "content": "reply " * 100},
            {"role": "user", "content": "recent question"},
        ]
        kept = holder._truncate(history)
        self.assertEqual(kept[-1]["content"], "recent question")
        self.assertLessEqual(len(kept), 3)

    def test_blank_and_tool_turns_dropped(self):
        from controlai_agent import agent as agent_module

        class FakeEngine:
            def count_tokens(self, text):
                return len(text.split())

        holder = object.__new__(agent_module.ControlAgent)
        holder.engine = FakeEngine()
        kept = holder._truncate(
            [{"role": "tool", "content": "x"}, {"role": "user", "content": "   "}, {"role": "user", "content": "hi"}]
        )
        self.assertEqual(kept, [{"role": "user", "content": "hi"}])


if __name__ == "__main__":
    unittest.main()
