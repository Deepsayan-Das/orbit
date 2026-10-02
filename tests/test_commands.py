"""
Unit tests for Orbit Slash Commands System (commands package).
"""

import os
import tempfile
import unittest
from typing import Iterator, Union

from commands import Session, dispatch, get_command, list_commands, clear_registry
from commands.compaction import compact_history
from commands.tokens import estimate_tokens
from config import OrbitConfig
from llm_client import OrbitLLM
from providers.base import BaseLLMProvider, ChatMessage, LLMResponse, StreamChunk
import tools.registry as tool_registry


class AvailableStubProvider(BaseLLMProvider):
    def __init__(self, model: str = "stub-model", **kwargs):
        self.model_name = model

    def chat(self, messages, stream=False, **kwargs) -> Union[LLMResponse, Iterator[StreamChunk]]:
        return LLMResponse(content="stub response", model=self.model_name)

    def is_available(self) -> bool:
        return True


class UnavailableStubProvider(BaseLLMProvider):
    def __init__(self, model: str = "unavailable-model", **kwargs):
        self.model_name = model

    def chat(self, messages, stream=False, **kwargs) -> Union[LLMResponse, Iterator[StreamChunk]]:
        raise RuntimeError("Provider is unavailable")

    def is_available(self) -> bool:
        return False


class TestCommandsSystem(unittest.TestCase):

    def setUp(self):
        OrbitLLM.register_provider("available_stub", AvailableStubProvider)
        OrbitLLM.register_provider("unavailable_stub", UnavailableStubProvider)

        self.cfg = OrbitConfig(
            provider="available_stub",
            model="test-model-1",
            providers={
                "openai": {"api_key": "sk-proj-secretkey1234567890abcdef"},
                "gemini": {"api_key": "AIzaSySecretKey98765"},
            },
        )
        self.agent = OrbitLLM.from_config(self.cfg)
        self.session = Session(
            agent=self.agent,
            config=self.cfg,
            history=[ChatMessage(role="user", content="Hello world")],
            max_steps=8,
        )

    def test_known_and_unknown_commands_dispatch(self):
        """Dispatch known commands returns result; unknown commands return friendly error pointing to /help."""
        res_help = dispatch("/help", self.session)
        self.assertIn("Available REPL Commands:", res_help)
        self.assertIn("/help", res_help)

        res_unknown = dispatch("/foo_bar_unknown", self.session)
        self.assertIn("[error: unknown command", res_unknown)
        self.assertIn("/help", res_unknown)

    def test_bare_word_aliases(self):
        """Bare word aliases ('quit', 'exit', 'reset', 'clear') dispatch correctly."""
        s = Session(agent=self.agent, config=self.cfg, history=[ChatMessage(role="user", content="Hi")])
        res_clear = dispatch("clear", s)
        self.assertEqual(res_clear, "[history cleared]")
        self.assertEqual(len(s.history), 0)

        s2 = Session(agent=self.agent, config=self.cfg)
        res_exit = dispatch("exit", s2)
        self.assertEqual(res_exit, "[exiting Orbit]")
        self.assertTrue(s2.should_exit)

    def test_handler_exceptions_contained(self):
        """Handler exceptions inside commands do not crash dispatch; they are returned as error text."""
        from commands.registry import command
        
        @command(name="buggy_cmd", help="Command that raises an exception")
        def buggy_handler(args, session):
            raise ValueError("Something exploded inside handler!")

        res = dispatch("/buggy_cmd", self.session)
        self.assertIn("[error executing /buggy_cmd: Something exploded inside handler!]", res)

    def test_api_key_masking_in_config(self):
        """The /config command masks API keys, showing only the last 4 characters."""
        res = dispatch("/config", self.session)
        self.assertIn("Active Orbit Configuration:", res)
        # Should contain masked keys
        self.assertIn("cdef", res)
        self.assertIn("8765", res)
        # MUST NOT contain full raw secret keys
        self.assertNotIn("sk-proj-secretkey1234567890abcdef", res)
        self.assertNotIn("AIzaSySecretKey98765", res)

    def test_steps_bounds(self):
        """The /steps command allows setting values between 1 and 20 only."""
        res_show = dispatch("/steps", self.session)
        self.assertIn("Current max_steps: 8", res_show)

        res_set_valid = dispatch("/steps 15", self.session)
        self.assertIn("max_steps updated to 15.", res_set_valid)
        self.assertEqual(self.session.max_steps, 15)

        res_set_low = dispatch("/steps 0", self.session)
        self.assertIn("[error: max_steps must be an integer between 1 and 20.]", res_set_low)
        self.assertEqual(self.session.max_steps, 15)

        res_set_high = dispatch("/steps 25", self.session)
        self.assertIn("[error: max_steps must be an integer between 1 and 20.]", res_set_high)
        self.assertEqual(self.session.max_steps, 15)

        res_set_invalid = dispatch("/steps abc", self.session)
        self.assertIn("[error: max_steps must be an integer between 1 and 20.]", res_set_invalid)

    def test_model_switching_with_unavailable_provider(self):
        """Switching model/provider to an unavailable provider keeps the old provider/model."""
        # 1. Test model switch with current available provider
        res_model_show = dispatch("/model", self.session)
        self.assertIn("Current model: test-model-1", res_model_show)

        res_model_switch = dispatch("/model test-model-2", self.session)
        self.assertIn("Model switched to 'test-model-2'.", res_model_switch)
        self.assertEqual(self.session.config.model, "test-model-2")

        # 2. Test provider switch to unavailable provider
        res_prov_unavailable = dispatch("/provider unavailable_stub", self.session)
        self.assertIn("[error: Provider 'unavailable_stub' is not available. Remaining on provider 'available_stub'.]", res_prov_unavailable)
        self.assertEqual(self.session.config.provider, "available_stub")

    def test_sources_and_context_and_compact(self):
        """Test /sources, /context, and /compact commands."""
        res_sources_on = dispatch("/sources on", self.session)
        self.assertTrue(self.session.show_sources)
        self.assertIn("Show sources enabled.", res_sources_on)

        res_context = dispatch("/context", self.session)
        self.assertIn("History context: 1 messages", res_context)

        res_compact = dispatch("/compact", self.session)
        self.assertIn("Compaction completed:", res_compact)

    def test_estimate_tokens_module(self):
        """Test pure estimate_tokens pure function."""
        msgs = [ChatMessage(role="user", content="Hello world! 1234567890")]
        tokens = estimate_tokens(msgs)
        self.assertGreater(tokens, 0)

    def test_compaction_module(self):
        """Test compaction module executes cleanly."""
        res = compact_history(self.session.history)
        self.assertEqual(len(res), len(self.session.history))


if __name__ == "__main__":
    unittest.main()
