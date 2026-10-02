"""
Unit tests for agent_loop.py: multi-step agentic loop.
"""

import unittest
from unittest.mock import MagicMock

from agent_loop import AgentTurnResult, run_agent_turn
from providers.base import ChatMessage, LLMResponse
import tools.registry as registry


class DummyToolCall:
    def __init__(self, name: str, arguments: dict):
        self.function = DummyFunction(name, arguments)


class DummyFunction:
    def __init__(self, name: str, arguments: dict):
        self.name = name
        self.arguments = arguments


class TestAgentLoop(unittest.TestCase):

    def setUp(self):
        registry.clear_registry()
        # Register a dummy tool for testing
        def mock_tool(val: str = "hello"):
            return f"executed: {val}"

        schema = {
            "type": "function",
            "function": {
                "name": "mock_tool",
                "description": "Mock tool for testing",
                "parameters": {
                    "type": "object",
                    "properties": {"val": {"type": "string"}},
                },
            },
        }

        registry.register_tool(
            "mock_tool",
            mock_tool,
            schema,
            risk_level=registry.ToolRiskLevel.SAFE,
        )

    def tearDown(self):
        registry.clear_registry()
        # Re-register default builtin tools so other tests are unaffected
        from tools.filesystem import register_filesystem_tools
        from tools.read_file import register_read_file_tool
        from tools.shell import register_shell_tools
        from tools.git_tools import register_git_tools
        from tools.container_tools import register_container_tools
        from tools.dev_tools import register_dev_tools
        from tools.code_search import register_code_search_tools
        register_filesystem_tools()
        register_read_file_tool()
        register_shell_tools()
        register_git_tools()
        register_container_tools()
        register_dev_tools()
        register_code_search_tools()

    def test_single_round_no_tool_calls(self):
        """When model returns plain content on round 1, loop stops and returns content."""
        agent = MagicMock()
        mock_response = LLMResponse(
            content="Hello! How can I help?",
            model="mock-model",
            raw_response={"message": {"content": "Hello! How can I help?", "tool_calls": []}},
        )
        agent.chat.return_value = mock_response

        messages = [ChatMessage(role="user", content="Hi")]
        tools_schema = registry.get_tools_schema()

        result = run_agent_turn(
            agent=agent,
            messages=messages,
            system_prompt="System Prompt",
            registered_tools=tools_schema,
            max_steps=8,
            print_status=False,
        )

        self.assertIsInstance(result, AgentTurnResult)
        self.assertEqual(result.content, "Hello! How can I help?")
        self.assertEqual(result.steps_taken, 1)
        self.assertFalse(result.hit_step_limit)
        self.assertEqual(len(result.tool_calls_log), 0)
        self.assertEqual(agent.chat.call_count, 1)

    def test_multi_step_tool_execution(self):
        """When model calls a tool, tool is executed and result is fed back to model until plain answer."""
        agent = MagicMock()

        tool_call_item = DummyToolCall("mock_tool", {"val": "test_input"})
        resp_step1 = LLMResponse(
            content="",
            model="mock-model",
            raw_response={"message": {"tool_calls": [tool_call_item]}},
        )
        resp_step2 = LLMResponse(
            content="The tool output was: executed: test_input",
            model="mock-model",
            raw_response={"message": {"content": "The tool output was: executed: test_input", "tool_calls": []}},
        )
        agent.chat.side_effect = [resp_step1, resp_step2]

        messages = [ChatMessage(role="user", content="Run mock tool")]
        tools_schema = registry.get_tools_schema()

        result = run_agent_turn(
            agent=agent,
            messages=messages,
            system_prompt="System Prompt",
            registered_tools=tools_schema,
            max_steps=8,
            print_status=False,
        )

        self.assertEqual(result.content, "The tool output was: executed: test_input")
        self.assertEqual(result.steps_taken, 2)
        self.assertFalse(result.hit_step_limit)
        self.assertEqual(len(result.tool_calls_log), 1)
        self.assertEqual(result.tool_calls_log[0]["tool"], "mock_tool")
        self.assertEqual(result.tool_calls_log[0]["args"], {"val": "test_input"})
        self.assertEqual(agent.chat.call_count, 2)

        # Verify messages history includes assistant + tool messages
        tool_msgs = [m for m in messages if m.role == "tool"]
        self.assertEqual(len(tool_msgs), 1)
        self.assertIn("executed: test_input", tool_msgs[0].content)

    def test_max_steps_termination(self):
        """When max_steps limit is reached, loop stops and makes a final non-tool call with ceiling note."""
        agent = MagicMock()

        tool_call_item = DummyToolCall("mock_tool", {"val": "loop"})
        loop_resp = LLMResponse(
            content="",
            model="mock-model",
            raw_response={"message": {"tool_calls": [tool_call_item]}},
        )
        final_resp = LLMResponse(
            content="Summary after reaching step limit.",
            model="mock-model",
            raw_response={"message": {"content": "Summary after reaching step limit."}},
        )

        # For max_steps=2: 2 tool-calling turns + 1 final non-tool call
        agent.chat.side_effect = [loop_resp, loop_resp, final_resp]

        messages = [ChatMessage(role="user", content="Keep calling tools")]
        tools_schema = registry.get_tools_schema()

        result = run_agent_turn(
            agent=agent,
            messages=messages,
            system_prompt="System Prompt",
            registered_tools=tools_schema,
            max_steps=2,
            print_status=False,
        )

        self.assertTrue(result.hit_step_limit)
        self.assertEqual(result.steps_taken, 2)
        self.assertIn("Summary after reaching step limit.", result.content)
        self.assertIn("[agent note: step limit (2) reached", result.content)
        self.assertEqual(len(result.tool_calls_log), 2)
        self.assertEqual(agent.chat.call_count, 3)


if __name__ == "__main__":
    unittest.main()
