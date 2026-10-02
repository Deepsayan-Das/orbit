"""
Unit tests for Part 1: Context Window Resolver & Ollama num_ctx integration.
"""

import unittest
from unittest.mock import MagicMock

from config import OrbitConfig
from context.window import resolve_window
from providers.ollama_provider import OllamaProvider


class TestWindowResolver(unittest.TestCase):

    def test_table_resolution(self):
        """Known model in table resolves to table window and source 'table'."""
        cfg = OrbitConfig()
        win, source = resolve_window("openai", "gpt-4o", cfg)
        self.assertEqual(win, 128000)
        self.assertEqual(source, "table")

    def test_fallback_resolution(self):
        """Unknown model resolves to 4096 and source 'fallback'."""
        cfg = OrbitConfig()
        win, source = resolve_window("unknown_provider", "unknown_model_xyz", cfg)
        self.assertEqual(win, 4096)
        self.assertEqual(source, "fallback")

    def test_user_config_resolution_and_clamping(self):
        """User explicit config overrides window, but is clamped if in table."""
        # 1. Model NOT in table: uses user value directly
        cfg1 = OrbitConfig(context_windows={"my-custom-model": 16000})
        win1, source1 = resolve_window("custom", "my-custom-model", cfg1)
        self.assertEqual(win1, 16000)
        self.assertEqual(source1, "user")

        # 2. Model IN table with max 8192: user requests 32000 -> clamped to 8192
        cfg2 = OrbitConfig(context_windows={"gpt-4": 32000})
        win2, source2 = resolve_window("openai", "gpt-4", cfg2)
        self.assertEqual(win2, 8192)
        self.assertEqual(source2, "user")

    def test_ollama_provider_sends_num_ctx(self):
        """OllamaProvider includes num_ctx in chat options for every request."""
        provider = OllamaProvider(model="llama3.2", num_ctx=8192)
        provider.client = MagicMock()
        provider.client.chat.return_value = {"message": {"content": "ok"}}

        from providers.base import ChatMessage
        messages = [ChatMessage(role="user", content="hello")]

        provider.chat(messages, stream=False)

        provider.client.chat.assert_called_once()
        kwargs_called = provider.client.chat.call_args[1]
        self.assertIn("options", kwargs_called)
        self.assertEqual(kwargs_called["options"].get("num_ctx"), 8192)


if __name__ == "__main__":
    unittest.main()
