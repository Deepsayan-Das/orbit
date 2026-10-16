"""
Unit tests for Phase 5: Cost & Latency Optimizations.
1. Ollama num_ctx default clamping (clamped to max 16384 unless user explicit).
2. Accurate token accounting from provider usage payloads.
3. Modern context windows in KNOWN_MODEL_WINDOWS.
4. Anthropic prompt caching structure (cache_control on system and tools).
5. Generation parameters (temperature, max_tokens) wired through OrbitLLM.
"""

import unittest
from unittest.mock import MagicMock, patch

from config import OrbitConfig
from context.measure import estimate_tokens
from context.window import resolve_window
from llm_client import OrbitLLM
from providers.anthropic_provider import AnthropicProvider
from providers.base import ChatMessage, LLMResponse


class TestPhase5CostLatency(unittest.TestCase):

    def test_ollama_num_ctx_clamping_default(self):
        """Ollama models with huge table windows (e.g. 131k) clamp default num_ctx to 16384."""
        cfg = OrbitConfig(provider="ollama", model="llama3.2")
        client = OrbitLLM.from_config(cfg)
        self.assertEqual(client.window_size, 16384)
        self.assertEqual(client.provider.num_ctx, 16384)

    def test_ollama_num_ctx_explicit_override_honored(self):
        """User explicit config overrides the 16384 clamp."""
        cfg = OrbitConfig(
            provider="ollama",
            model="llama3.2",
            context_windows={"llama3.2": 65536}
        )
        client = OrbitLLM.from_config(cfg)
        self.assertEqual(client.window_size, 65536)
        self.assertEqual(client.provider.num_ctx, 65536)

    def test_token_measurement_from_various_usages(self):
        """Accurately extract tokens from Ollama, Anthropic, Gemini, OpenAI usage styles."""
        # 1. Ollama prompt_eval_count
        resp_ollama = LLMResponse(
            content="ok",
            model="llama3.2",
            usage={"prompt_tokens": 512, "completion_tokens": 64, "total_tokens": 576},
            raw_response={"prompt_eval_count": 512, "eval_count": 64}
        )
        self.assertEqual(estimate_tokens(resp_ollama), 512)

        # 2. Anthropic input_tokens
        resp_anthropic = LLMResponse(
            content="ok",
            model="claude-3-5-sonnet",
            usage={"prompt_tokens": 1024, "completion_tokens": 256, "total_tokens": 1280}
        )
        self.assertEqual(estimate_tokens(resp_anthropic), 1024)

        # 3. Gemini usage_metadata directly on raw_response
        class MockGeminiMeta:
            prompt_token_count = 2048
            candidates_token_count = 128
            total_token_count = 2176

        class MockGeminiResp:
            usage_metadata = MockGeminiMeta()

        resp_gemini = LLMResponse(
            content="ok",
            model="gemini-2.0-flash",
            raw_response=MockGeminiResp()
        )
        self.assertEqual(estimate_tokens(resp_gemini), 2048)

    def test_modern_model_windows_table(self):
        """Verify new modern models resolve to accurate large windows."""
        win_c37, _ = resolve_window("anthropic", "claude-3-7-sonnet")
        self.assertEqual(win_c37, 200000)

        win_g25, _ = resolve_window("gemini", "gemini-2.5-flash")
        self.assertEqual(win_g25, 1000000)

        win_l33, _ = resolve_window("ollama", "llama-3.3-70b")
        self.assertEqual(win_l33, 131072)

    def test_anthropic_prompt_caching_format(self):
        """Anthropic provider attaches ephemeral cache_control to system and last tool."""
        provider = AnthropicProvider(api_key="test_key", model="claude-3-5-sonnet")
        mock_client = MagicMock()
        mock_response = MagicMock()
        mock_response.content = []
        mock_response.usage = MagicMock(input_tokens=100, output_tokens=50, cache_creation_input_tokens=0, cache_read_input_tokens=0)
        mock_client.messages.create.return_value = mock_response
        provider._client = mock_client

        messages = [ChatMessage(role="user", content="hello")]
        tools = [{"type": "function", "function": {"name": "tool1", "parameters": {}}}]

        provider.chat(messages, system_prompt="You are Orbit system prompt.", tools=tools)

        call_args = mock_client.messages.create.call_args[1]
        system_arg = call_args["system"]
        self.assertIsInstance(system_arg, list)
        self.assertEqual(system_arg[0]["cache_control"], {"type": "ephemeral"})

        tools_arg = call_args["tools"]
        self.assertEqual(tools_arg[-1]["cache_control"], {"type": "ephemeral"})

    def test_generation_parameters_wired(self):
        """Generation params (temperature, max_tokens) pass down through OrbitLLM."""
        cfg = OrbitConfig(
            provider="ollama",
            model="qwen:0.5b",
            generation={"temperature": 0.2, "max_tokens": 1024}
        )
        client = OrbitLLM.from_config(cfg)
        self.assertEqual(client.generation.get("temperature"), 0.2)
        self.assertEqual(client.generation.get("max_tokens"), 1024)

        client.provider.chat = MagicMock()
        client.chat("test query")
        called_kwargs = client.provider.chat.call_args[1]
        self.assertEqual(called_kwargs.get("temperature"), 0.2)
        self.assertEqual(called_kwargs.get("max_tokens"), 1024)


if __name__ == "__main__":
    unittest.main()
