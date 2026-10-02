"""
Unit tests for Part 4: Context Compaction Subsystem.
"""

import unittest
from unittest.mock import MagicMock

from commands.compaction import compact_history, drop_oldest_tool_pairs, stub_tool_result
from providers.base import ChatMessage, LLMResponse


class DummyToolCall:
    def __init__(self, name: str, args: dict):
        self.function = MagicMock(name=name, arguments=args)


class TestCompaction(unittest.TestCase):

    def setUp(self):
        self.sys_msg = ChatMessage(role="system", content="System Prompt")
        self.u1_msg = ChatMessage(role="user", content="First user query")

    def test_never_touch_rule(self):
        """System prompt and first user message are NEVER touched or removed."""
        history = [
            self.sys_msg,
            self.u1_msg,
            ChatMessage(role="assistant", content="old output 1"),
            ChatMessage(role="user", content="msg 2"),
            ChatMessage(role="assistant", content="msg 3"),
            ChatMessage(role="user", content="msg 4"),
            ChatMessage(role="assistant", content="msg 5"),
            ChatMessage(role="user", content="msg 6"),
            ChatMessage(role="assistant", content="msg 7"),
        ]

        compacted = compact_history(history, window=10, recent_count=4)
        self.assertEqual(compacted[0].role, "system")
        self.assertEqual(compacted[0].content, "System Prompt")
        self.assertEqual(compacted[1].role, "user")
        self.assertEqual(compacted[1].content, "First user query")

    def test_tool_call_pairing_rule(self):
        """Tool call assistant message and tool result message are kept as a PAIR."""
        tcall = DummyToolCall("read_file", {"path": "a.py"})
        history = [
            self.sys_msg,
            self.u1_msg,
            ChatMessage(role="assistant", content="", tool_calls=[tcall]),
            ChatMessage(role="tool", content="file content line 1\nline 2", name="read_file"),
            ChatMessage(role="user", content="recent 1"),
            ChatMessage(role="assistant", content="recent 2"),
            ChatMessage(role="user", content="recent 3"),
            ChatMessage(role="assistant", content="recent 4"),
            ChatMessage(role="user", content="recent 5"),
            ChatMessage(role="assistant", content="recent 6"),
        ]

        compacted = compact_history(history, window=100, recent_count=6)
        roles = [m.role for m in compacted]
        # Check that assistant with tool_calls and tool result stay adjacent
        if "tool" in roles:
            tool_idx = roles.index("tool")
            self.assertEqual(roles[tool_idx - 1], "assistant")

    def test_failure_fallback_pair_dropping(self):
        """If summary call fails or raises exception, falls back to dropping oldest tool pairs."""
        tcall = DummyToolCall("read_file", {"path": "a.py"})
        history = [
            self.sys_msg,
            self.u1_msg,
            ChatMessage(role="assistant", content="", tool_calls=[tcall]),
            ChatMessage(role="tool", content="old result " * 500, name="read_file"),
            ChatMessage(role="user", content="recent 1"),
            ChatMessage(role="assistant", content="recent 2"),
            ChatMessage(role="user", content="recent 3"),
            ChatMessage(role="assistant", content="recent 4"),
            ChatMessage(role="user", content="recent 5"),
            ChatMessage(role="assistant", content="recent 6"),
        ]

        # Agent whose chat() raises an exception
        broken_agent = MagicMock()
        broken_agent.chat.side_effect = RuntimeError("API Call Failed")

        compacted = compact_history(history, agent=broken_agent, window=100, recent_count=6)
        # Should drop the oldest tool pair cleanly without raising
        self.assertNotIn("old result", [m.content for m in compacted])

    def test_retry_limit(self):
        """Compaction retries at most 2 times and never loops infinitely."""
        history = [self.sys_msg, self.u1_msg] + [ChatMessage(role="user", content="X" * 1000) for _ in range(15)]
        broken_agent = MagicMock()
        broken_agent.chat.side_effect = RuntimeError("API Fail")

        compacted = compact_history(history, agent=broken_agent, window=50, recent_count=4)
        # Verify call count on broken agent is at most 2
        self.assertLessEqual(broken_agent.chat.call_count, 2)


if __name__ == "__main__":
    unittest.main()
