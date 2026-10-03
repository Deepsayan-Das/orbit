"""
Unit tests for MistralProvider implementation (interface conformance & mocked API calls).
"""

import unittest
from unittest.mock import MagicMock

from providers import MistralProvider, ChatMessage, LLMResponse
from providers.base import BaseLLMProvider


class TestMistralProvider(unittest.TestCase):

    def test_provider_subclass(self):
        """Confirm MistralProvider is a subclass of BaseLLMProvider."""
        self.assertTrue(issubclass(MistralProvider, BaseLLMProvider))

    def test_is_available(self):
        """Confirm is_available returns True when API key is provided."""
        p1 = MistralProvider(api_key="test_key")
        self.assertTrue(p1.is_available())

        p2 = MistralProvider(api_key="")
        p2.api_key = None
        self.assertFalse(p2.is_available())

    def test_sync_chat_mocked(self):
        """Test MistralProvider.chat with mocked Mistral client."""
        provider = MistralProvider(model="mistral-small-latest", api_key="test_key")

        mock_client = MagicMock()
        mock_choice = MagicMock()
        mock_choice.message.content = "Hello from Mistral!"

        mock_response = MagicMock()
        mock_response.choices = [mock_choice]
        mock_response.usage = {"prompt_tokens": 10, "completion_tokens": 5}

        mock_client.chat.complete.return_value = mock_response
        provider._client = mock_client

        messages = [ChatMessage(role="user", content="Hello")]
        response = provider.chat(messages, stream=False)

        self.assertIsInstance(response, LLMResponse)
        self.assertEqual(response.content, "Hello from Mistral!")
        self.assertEqual(response.model, "mistral-small-latest")
        mock_client.chat.complete.assert_called_once()

        call_kwargs = mock_client.chat.complete.call_args[1]
        self.assertEqual(call_kwargs["model"], "mistral-small-latest")
        self.assertEqual(call_kwargs["messages"], [{"role": "user", "content": "Hello"}])


if __name__ == "__main__":
    unittest.main()
