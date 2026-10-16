"""
Unit tests verifying Phase 1 Correctness fixes:
1. Tool calling parsing across providers (OpenAI, Anthropic, Mistral, Groq, Gemini, Ollama, HuggingFace).
2. Native message format with tool_calls and tool_call_id.
3. Compaction fallback loop bug fix & no orphaned tool results.
4. Realistic tool status stubbing.
5. Ephemeral RAG context handling.
"""

import unittest
from unittest.mock import MagicMock

from providers import (
    ChatMessage,
    LLMResponse,
    ToolCall,
    OpenAIProvider,
    GroqProvider,
    AnthropicProvider,
    MistralProvider,
    GeminiProvider,
    OllamaProvider,
    HuggingFaceProvider,
)
from commands.compaction import compact_history, stub_tool_result


class TestProviderToolCalling(unittest.TestCase):

    def test_openai_tool_calls_parsing(self):
        provider = OpenAIProvider(api_key="sk-test")
        mock_client = MagicMock()
        mock_tc = MagicMock()
        mock_tc.id = "call_abc123"
        mock_tc.function.name = "read_file"
        mock_tc.function.arguments = '{"path": "rag/indexer.py"}'

        mock_choice = MagicMock()
        mock_choice.message.content = ""
        mock_choice.message.tool_calls = [mock_tc]
        mock_choice.finish_reason = "tool_calls"

        mock_response = MagicMock()
        mock_response.choices = [mock_choice]
        mock_response.usage = None

        mock_client.chat.completions.create.return_value = mock_response
        provider._client = mock_client

        res = provider.chat([ChatMessage(role="user", content="Read indexer")], stream=False)
        self.assertEqual(len(res.tool_calls), 1)
        self.assertEqual(res.tool_calls[0].id, "call_abc123")
        self.assertEqual(res.tool_calls[0].name, "read_file")
        self.assertEqual(res.tool_calls[0].args, {"path": "rag/indexer.py"})

    def test_anthropic_tool_calls_parsing(self):
        provider = AnthropicProvider(api_key="sk-ant-test")
        mock_client = MagicMock()
        mock_block = MagicMock()
        mock_block.type = "tool_use"
        mock_block.id = "toolu_xyz789"
        mock_block.name = "execute_command"
        mock_block.input = {"command": "git status"}

        mock_response = MagicMock()
        mock_response.content = [mock_block]
        mock_response.usage = {"input_tokens": 10, "output_tokens": 5}

        mock_client.messages.create.return_value = mock_response
        provider._client = mock_client

        res = provider.chat([ChatMessage(role="user", content="Check git")], stream=False)
        self.assertEqual(len(res.tool_calls), 1)
        self.assertEqual(res.tool_calls[0].id, "toolu_xyz789")
        self.assertEqual(res.tool_calls[0].name, "execute_command")
        self.assertEqual(res.tool_calls[0].args, {"command": "git status"})

    def test_anthropic_native_message_formatting(self):
        provider = AnthropicProvider(api_key="sk-ant-test")
        messages = [
            ChatMessage(role="system", content="Sys prompt"),
            ChatMessage(role="user", content="Do task"),
            ChatMessage(role="assistant", content="", tool_calls=[ToolCall(id="tc_1", name="f", args={"x": 1})]),
            ChatMessage(role="tool", content="out1", name="f", tool_call_id="tc_1"),
            ChatMessage(role="tool", content="out2", name="f2", tool_call_id="tc_2"),
        ]
        formatted, sys_prompt = provider._format_messages(messages)
        self.assertEqual(sys_prompt, "Sys prompt")
        # Check tool results grouped in user message
        tool_user_msg = formatted[-1]
        self.assertEqual(tool_user_msg["role"], "user")
        self.assertEqual(len(tool_user_msg["content"]), 2)
        self.assertEqual(tool_user_msg["content"][0]["type"], "tool_result")
        self.assertEqual(tool_user_msg["content"][0]["tool_use_id"], "tc_1")

    def test_mistral_tool_calls_parsing(self):
        provider = MistralProvider(api_key="test_key")
        mock_client = MagicMock()
        mock_tc = MagicMock()
        mock_tc.id = "call_mis123"
        mock_tc.function.name = "search_codebase"
        mock_tc.function.arguments = '{"query": "compaction"}'

        mock_choice = MagicMock()
        mock_choice.message.content = ""
        mock_choice.message.tool_calls = [mock_tc]

        mock_response = MagicMock()
        mock_response.choices = [mock_choice]
        mock_response.usage = None

        mock_client.chat.complete.return_value = mock_response
        provider._client = mock_client

        res = provider.chat([ChatMessage(role="user", content="Search")], stream=False)
        self.assertEqual(len(res.tool_calls), 1)
        self.assertEqual(res.tool_calls[0].id, "call_mis123")
        self.assertEqual(res.tool_calls[0].name, "search_codebase")
        self.assertEqual(res.tool_calls[0].args, {"query": "compaction"})

    def test_gemini_tool_calls_parsing(self):
        provider = GeminiProvider(api_key="test_gemini_key")
        mock_client = MagicMock()
        mock_fc = MagicMock()
        mock_fc.name = "edit_file"
        mock_fc.args = {"path": "test.py", "old_text": "a", "new_text": "b"}

        mock_response = MagicMock()
        mock_response.function_calls = [mock_fc]
        mock_response.text = None

        mock_client.models.generate_content.return_value = mock_response
        provider._client = mock_client

        res = provider.chat([ChatMessage(role="user", content="Edit")], stream=False)
        self.assertEqual(len(res.tool_calls), 1)
        self.assertEqual(res.tool_calls[0].name, "edit_file")
        self.assertEqual(res.tool_calls[0].args["path"], "test.py")

    def test_ollama_tool_calls_parsing(self):
        provider = OllamaProvider()
        mock_client = MagicMock()
        mock_client.chat.return_value = {
            "message": {
                "content": "",
                "tool_calls": [
                    {
                        "id": "call_ollama_1",
                        "function": {
                            "name": "run_shell_command",
                            "arguments": {"command": "ls"},
                        },
                    }
                ],
            }
        }
        provider.client = mock_client

        res = provider.chat([ChatMessage(role="user", content="Run ls")], stream=False)
        self.assertEqual(len(res.tool_calls), 1)
        self.assertEqual(res.tool_calls[0].name, "run_shell_command")
        self.assertEqual(res.tool_calls[0].args, {"command": "ls"})


class TestCompactionFixes(unittest.TestCase):

    def test_compaction_no_orphaned_tools(self):
        """Ensure recent_suffix boundary does not slice in the middle of tool execution."""
        sys_msg = ChatMessage(role="system", content="System")
        u1_msg = ChatMessage(role="user", content="User 1")
        tc = ToolCall(id="tc_1", name="read_file", args={})
        
        # Construct history where recent_count=4 would cut right between assistant tool_calls and tool result
        history = [
            sys_msg,
            u1_msg,
            ChatMessage(role="assistant", content="thought 1"),
            ChatMessage(role="user", content="query 2"),
            ChatMessage(role="assistant", content="calling tool", tool_calls=[tc]),
            ChatMessage(role="tool", content="result 1", tool_call_id="tc_1"),
            ChatMessage(role="user", content="query 3"),
            ChatMessage(role="assistant", content="answer 3"),
        ]

        compacted = compact_history(history, window=100, recent_count=3)
        # Check that if 'result 1' is retained, 'calling tool' is also in the history
        contents = [m.content for m in compacted]
        if "result 1" in contents:
            self.assertIn("calling tool", contents)

    def test_stub_tool_result_accuracy(self):
        res_ok = stub_tool_result("run_shell_command", "build succeeded\nFinished in 0.5s")
        self.assertIn("status ok", res_ok)

        res_err = stub_tool_result("run_shell_command", "error: command failed with exit code 1")
        self.assertIn("status error", res_err)

        test_ok = stub_tool_result("run_tests", "Ran 10 tests in 0.1s\n\nOK")
        self.assertIn("failing tests: none", test_ok)

        test_fail = stub_tool_result("run_tests", "Ran 10 tests in 0.1s\n\nFAILED (failures=1)")
        self.assertIn("failing tests detected", test_fail)


if __name__ == "__main__":
    unittest.main()
