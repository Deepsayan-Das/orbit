import io
import sys
import unittest
from unittest import mock
import ollama

from providers.base import ChatMessage
from providers.ollama_provider import OllamaProvider


class TestOllamaProvider(unittest.TestCase):
    """Test suite for OllamaProvider option handling and signature compliance."""

    @mock.patch.object(ollama.Client, "chat", autospec=True)
    def test_sync_chat_translates_generation_options(self, mock_chat):
        """Cover normal chat (stream=False) with max_tokens, temperature, and num_ctx."""
        mock_chat.return_value = {"message": {"content": "Hello!"}}
        provider = OllamaProvider(model="llama3.2:latest", host="http://localhost:11434")
        messages = [ChatMessage(role="user", content="Hi")]

        resp = provider.chat(
            messages,
            stream=False,
            max_tokens=2048,
            temperature=0.7,
            num_ctx=8192,
        )

        mock_chat.assert_called_once()
        called_kwargs = mock_chat.call_args[1]

        # Verify options dictionary has translated keys and values
        self.assertIn("options", called_kwargs)
        options = called_kwargs["options"]
        self.assertEqual(options.get("num_predict"), 2048)
        self.assertEqual(options.get("temperature"), 0.7)
        self.assertEqual(options.get("num_ctx"), 8192)

        # Assert no stray generation keywords were passed directly to Client.chat
        self.assertNotIn("max_tokens", called_kwargs)
        self.assertNotIn("temperature", called_kwargs)
        self.assertNotIn("num_ctx", called_kwargs)
        self.assertFalse(called_kwargs.get("stream"))
        self.assertEqual(resp.content, "Hello!")

    @mock.patch.object(ollama.Client, "chat", autospec=True)
    def test_stream_chat_translates_generation_options(self, mock_chat):
        """Cover streaming chat (stream=True) with max_tokens, temperature, and num_ctx."""
        mock_chat.return_value = iter([
            {"message": {"content": "Hello"}, "done": False},
            {"message": {"content": " world"}, "done": True},
        ])
        provider = OllamaProvider(model="llama3.2:latest", host="http://localhost:11434")
        messages = [ChatMessage(role="user", content="Hi")]

        chunk_iter = provider.chat(
            messages,
            stream=True,
            max_tokens=2048,
            temperature=0.7,
            num_ctx=8192,
        )
        chunks = list(chunk_iter)

        mock_chat.assert_called_once()
        called_kwargs = mock_chat.call_args[1]

        # Verify options dictionary has translated keys and values
        self.assertIn("options", called_kwargs)
        options = called_kwargs["options"]
        self.assertEqual(options.get("num_predict"), 2048)
        self.assertEqual(options.get("temperature"), 0.7)
        self.assertEqual(options.get("num_ctx"), 8192)

        # Assert no stray generation keywords were passed directly to Client.chat
        self.assertNotIn("max_tokens", called_kwargs)
        self.assertNotIn("temperature", called_kwargs)
        self.assertNotIn("num_ctx", called_kwargs)
        self.assertTrue(called_kwargs.get("stream"))
        self.assertEqual(len(chunks), 2)
        self.assertEqual(chunks[0].delta, "Hello")
        self.assertEqual(chunks[1].delta, " world")

    @mock.patch.object(ollama.Client, "chat", autospec=True)
    def test_existing_options_merged_without_duplicate(self, mock_chat):
        """Existing options (like GPU settings) are merged into one options dict without passing options twice."""
        mock_chat.return_value = {"message": {"content": "ok"}}
        provider = OllamaProvider(model="llama3.2:latest", host="http://localhost:11434", num_ctx=4096)
        messages = [ChatMessage(role="user", content="Hi")]

        provider.chat(
            messages,
            stream=False,
            options={"low_vram": True, "num_gpu": 1},
            max_tokens=1024,
            temperature=0.5,
        )

        mock_chat.assert_called_once()
        called_kwargs = mock_chat.call_args[1]

        self.assertIn("options", called_kwargs)
        options = called_kwargs["options"]
        self.assertEqual(options.get("low_vram"), True)
        self.assertEqual(options.get("num_gpu"), 1)
        self.assertEqual(options.get("num_predict"), 1024)
        self.assertEqual(options.get("temperature"), 0.5)
        self.assertEqual(options.get("num_ctx"), 4096)
        self.assertNotIn("max_tokens", called_kwargs)
        self.assertNotIn("temperature", called_kwargs)

    @mock.patch.object(ollama.Client, "chat", autospec=True)
    def test_unrecognized_key_warned_and_not_passed(self, mock_chat):
        """Unrecognized keys must not be passed to chat() and must print a warning naming the key."""
        mock_chat.return_value = {"message": {"content": "ok"}}
        provider = OllamaProvider(model="llama3.2:latest", host="http://localhost:11434")
        messages = [ChatMessage(role="user", content="Hi")]

        captured_stdout = io.StringIO()
        old_stdout = sys.stdout
        try:
            sys.stdout = captured_stdout
            provider.chat(
                messages,
                stream=False,
                unsupported_custom_setting="invalid_value",
            )
        finally:
            sys.stdout = old_stdout

        mock_chat.assert_called_once()
        called_kwargs = mock_chat.call_args[1]

        # Verify key was not forwarded as keyword argument or option
        self.assertNotIn("unsupported_custom_setting", called_kwargs)
        if "options" in called_kwargs:
            self.assertNotIn("unsupported_custom_setting", called_kwargs["options"])

        # Verify warning was printed naming the key
        warning_output = captured_stdout.getvalue()
        self.assertIn("unsupported_custom_setting", warning_output)


if __name__ == "__main__":
    unittest.main()
