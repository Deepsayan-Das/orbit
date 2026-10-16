"""
Unit tests for Phase 6: Polish, Subagent Integration & Tool Filtering.
1. tools.enabled filter on get_tools_schema.
2. delegate_task registered in tools registry and exposed to agent.
3. Subagent tool restriction and execution.
4. Nova companion commands documented in run_shell_command tool schema.
5. Per-turn telemetry tracking in AgentTurnResult.
"""

import unittest
from unittest.mock import MagicMock

from agent_loop import AgentTurnResult, run_agent_turn
from providers.base import ChatMessage, LLMResponse
import tools
from tools.registry import get_tool, get_tools_schema


class TestPhase6Polish(unittest.TestCase):

    def setUp(self):
        tools.reset_default_tools()

    def test_tools_enabled_filtering(self):
        """get_tools_schema filters tools when enabled_only is specified."""
        all_schemas = get_tools_schema()
        self.assertGreater(len(all_schemas), 5)

        # Filter to only read_file and write_file
        filtered = get_tools_schema(enabled_only=["read_file", "write_file"])
        filtered_names = [s["function"]["name"] for s in filtered]
        self.assertEqual(set(filtered_names), {"read_file", "write_file"})

    def test_delegate_task_registered_in_tools(self):
        """delegate_task is automatically registered in tool registry and exported in tools package."""
        self.assertIn("delegate_task", tools.__all__)
        tool_info = get_tool("delegate_task")
        self.assertIsNotNone(tool_info)
        self.assertEqual(tool_info["name"], "delegate_task")
        self.assertEqual(tool_info["risk_level"], tools.ToolRiskLevel.SENSITIVE)

    def test_shell_tool_schema_contains_nova(self):
        """run_shell_command schema documents Nova companion CLI capabilities."""
        tool_info = get_tool("run_shell_command")
        self.assertIsNotNone(tool_info)
        desc = tool_info["schema"]["description"]
        self.assertIn("nova", desc.lower())
        self.assertIn("container", desc.lower())
        self.assertIn("cluster", desc.lower())

    def test_per_turn_telemetry_in_agent_turn(self):
        """Agent turn records usage metadata in AgentTurnResult."""
        agent = MagicMock()
        resp = LLMResponse(
            content="Final answer",
            model="mock",
            usage={"prompt_tokens": 150, "completion_tokens": 50, "total_tokens": 200}
        )
        agent.chat.return_value = resp

        messages = [ChatMessage(role="user", content="help me")]
        result = run_agent_turn(
            agent=agent,
            messages=messages,
            system_prompt="Test system",
            registered_tools=[],
            print_status=False,
        )

        self.assertEqual(result.content, "Final answer")
        self.assertIsNotNone(result.usage)
        self.assertEqual(result.usage.get("total_tokens"), 200)


if __name__ == "__main__":
    unittest.main()
