"""
Unit tests for Part 5: Sub-agent mechanism (agent_sub.py).
"""

import os
import unittest
from unittest.mock import MagicMock

from agent_sub import delegate_task, register_subagent_tool, run_subagent
from providers.base import ChatMessage, LLMResponse
import tools.registry as registry


class TestSubagent(unittest.TestCase):

    def setUp(self):
        registry.clear_registry()
        # Register test safe and dangerous tools
        def safe_tool():
            return "safe result"

        def dangerous_tool():
            return "dangerous result"

        safe_schema = {
            "name": "safe_tool",
            "description": "Safe tool",
            "parameters": {"type": "object", "properties": {}},
        }
        dangerous_schema = {
            "name": "dangerous_tool",
            "description": "Dangerous tool",
            "parameters": {"type": "object", "properties": {}},
        }

        registry.register_tool("safe_tool", safe_tool, safe_schema, risk_level=registry.ToolRiskLevel.SAFE)
        registry.register_tool("dangerous_tool", dangerous_tool, dangerous_schema, risk_level=registry.ToolRiskLevel.DANGEROUS)
        register_subagent_tool()

    def tearDown(self):
        registry.clear_registry()
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

    def test_isolation_and_result_cap(self):
        """Sub-agent runs with isolated history and caps long outputs."""
        agent = MagicMock()
        agent.chat.return_value = LLMResponse(
            content="Sub-agent answer " * 200,
            model="mock",
            raw_response={"message": {"content": "Sub-agent answer " * 200, "tool_calls": []}},
        )

        res = run_subagent(task="Summarize repo", agent=agent)
        self.assertIn("Sub-agent answer", res)
        self.assertIn("[sub-agent result capped at 500 tokens]", res)

        # Verify agent.chat received isolated 1-message prompt
        called_messages = agent.chat.call_args[1]["messages"]
        self.assertEqual(len(called_messages), 1)
        self.assertEqual(called_messages[0].role, "user")
        self.assertEqual(called_messages[0].content, "Summarize repo")

    def test_tool_restriction_safe_only(self):
        """Default sub-agent receives only SAFE registered tools."""
        agent = MagicMock()
        agent.chat.return_value = LLMResponse(content="done", model="mock")

        run_subagent(task="Run task", tools="safe", agent=agent)

        passed_tools = agent.chat.call_args[1]["tools"]
        tool_names = [t["function"]["name"] for t in passed_tools]

        self.assertIn("safe_tool", tool_names)
        self.assertNotIn("dangerous_tool", tool_names)

    def test_depth_limit_enforcement(self):
        """Sub-agent cannot call run_subagent recursively (depth limit 1)."""
        agent = MagicMock()
        agent.chat.return_value = LLMResponse(content="done", model="mock")

        # Nested subagent call should fail with depth limit error
        def nested_call():
            return run_subagent(task="Nested task", agent=agent)

        # Simulate executing inside subagent loop
        import agent_sub
        agent_sub._SUBAGENT_DEPTH = 1
        try:
            res = run_subagent(task="Recursive task", agent=agent)
            self.assertEqual(res, "[error: sub-agent depth limit (1) exceeded]")
        finally:
            agent_sub._SUBAGENT_DEPTH = 0

    def test_error_handling_no_raise(self):
        """Exceptions inside sub-agent are caught and returned as error text without raising."""
        broken_agent = MagicMock()
        broken_agent.chat.side_effect = RuntimeError("Fatal LLM Connection Error")

        res = run_subagent(task="Broken task", agent=broken_agent)
        self.assertIn("[error executing sub-agent task: Fatal LLM Connection Error]", res)

    def test_delegate_task_logs_audit_agent_id(self):
        """delegate_task tool logs agent_id in audit log."""
        agent = MagicMock()
        agent.chat.return_value = LLMResponse(content="task finished", model="mock")

        with unittest.mock.patch("agent_sub.run_subagent", return_value="task finished"):
            res = delegate_task("Audit test task")
            self.assertEqual(res, "task finished")

        log_path = registry.AUDIT_LOG_PATH
        if os.path.exists(log_path):
            with open(log_path, "r", encoding="utf-8") as f:
                content = f.read()
            self.assertIn("delegate_task", content)
            self.assertIn("agent_id", content)


if __name__ == "__main__":
    unittest.main()
