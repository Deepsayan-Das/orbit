"""
Unit tests for Part 2: Token measurement and compaction trigger.
"""

import unittest

from context.measure import RESERVE_TOKENS, estimate_tokens, should_compact
from providers.base import ChatMessage, LLMResponse


class TestMeasure(unittest.TestCase):

    def test_estimate_tokens_characters(self):
        """Estimate tokens falls back to character count divided by 4."""
        msgs = [ChatMessage(role="user", content="A" * 100)]
        self.assertEqual(estimate_tokens(msgs), 26)

    def test_estimate_tokens_llm_response_reported_usage(self):
        """Estimate tokens uses LLMResponse reported usage when present."""
        resp = LLMResponse(content="Hi", model="stub", metadata={"usage": {"prompt_tokens": 350}})
        self.assertEqual(estimate_tokens(resp), 350)

    def test_should_compact_trigger(self):
        """Compaction triggers at 75% usage or when remaining < RESERVE_TOKENS."""
        window = 4096

        # 50% usage -> False
        self.assertFalse(should_compact(2000, window))

        # 76% usage -> True
        self.assertTrue(should_compact(3100, window))

        # Remaining < 1000 tokens -> True (e.g. 3200 used out of 4096: remaining 896 < 1000)
        self.assertTrue(should_compact(3200, window))


if __name__ == "__main__":
    unittest.main()
