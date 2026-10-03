"""
Unit tests for AnthropicProvider implementation (interface conformance & mocked API calls).
"""

import unittest
from unittest.mock import MagicMock

from providers import AnthropicProvider, ChatMessage, LLMResponse, StreamChunk
from providers.base import BaseLLMProvider


class TestAnthropicProvider(unittest.TestCase):

    def test_provider_subclass(self):
        """Confirm AnthropicProvider is a subclass of BaseLLMProvider."""
        self.assertTrue(issubclass(AnthropicProvider, BaseLLMProvider))

    def test_is_available(self):
        """Confirm is_available returns True when API key is provided."""
        p1 = AnthropicProvider(api_key="test_key")
        self.assertTrue(p1.is_available())

        p2 = AnthropicProvider(api_key="")
        p2.api_key = None
        self.assertFalse(p2.is_available())

    def test_sync_chat_mocked(self):
        """Test AnthropicProvider.chat with mocked Anthropic client."""
        provider = AnthropicProvider(model="claude-3-5-sonnet-20241022", api_key="test_key")

        # Mock the client and Anthropic messages.create call
        mock_client = MagicMock()
        mock_block = MagicMock()
        mock_block.type = "text"
        mock_block.text = "Hello from Claude!"

        mock_response = MagicMock()
        mock_response.content = [mock_block]
        mock_response.usage = {"input_tokens": 10, "output_tokens": 5}

        mock_client.messages.create.return_value = mock_response
        provider._client = mock_client

        messages = [ChatMessage(role="user", content="Hello")]
        response = provider.chat(messages, stream=False)

        self.assertIsInstance(response, LLMResponse)
        self.assertEqual(response.content, "Hello from Claude!")
        self.assertEqual(response.model, "claude-3-5-sonnet-20241022")
        mock_client.messages.create.assert_called_once()

        # Check call arguments
        call_kwargs = mock_client.messages.create.call_args[1]
        self.assertEqual(call_kwargs["model"], "claude-3-5-sonnet-20241022")
        self.assertEqual(call_kwargs["messages"], [{"role": "user", "content": "Hello"}])


if __name__ == "__main__":
    unittest.main()
