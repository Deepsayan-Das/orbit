"""
Unit tests for Orbit Tool Registry system and builtin filesystem tools.
"""

import unittest
from tools import (
    clear_registry,
    execute_tool,
    get_tool,
    get_tools_schema,
    list_directory,
    list_registered_tools,
    register_tool,
)


class TestToolRegistry(unittest.TestCase):

    def setUp(self):
        clear_registry()

    def test_register_and_execute_tool(self):
        def add(a: int, b: int) -> int:
            return a + b

        schema = {
            "name": "add",
            "description": "Add two numbers",
            "parameters": {
                "type": "object",
                "properties": {
                    "a": {"type": "integer"},
                    "b": {"type": "integer"}
                },
                "required": ["a", "b"]
            }
        }

        register_tool("add", add, schema)
        self.assertIn("add", list_registered_tools())

        tool_info = get_tool("add")
        self.assertIsNotNone(tool_info)
        self.assertEqual(tool_info["name"], "add")

        result = execute_tool("add", {"a": 5, "b": 10})
        self.assertEqual(result, 15)

    def test_get_tools_schema_format(self):
        def dummy():
            pass

        schema = {
            "name": "dummy",
            "description": "Dummy tool",
            "parameters": {"type": "object", "properties": {}}
        }

        register_tool("dummy", dummy, schema)
        schemas = get_tools_schema()
        self.assertEqual(len(schemas), 1)
        self.assertEqual(schemas[0]["type"], "function")
        self.assertEqual(schemas[0]["function"]["name"], "dummy")

    def test_filesystem_list_directory_tool(self):
        from tools.filesystem import LIST_DIRECTORY_SCHEMA, list_directory
        register_tool("list_directory", list_directory, LIST_DIRECTORY_SCHEMA)
        self.assertIn("list_directory", list_registered_tools())

        res = execute_tool("list_directory", {"path": "."})
        self.assertIsInstance(res, str)
        self.assertNotIn("Error:", res)

    def test_tool_risk_levels(self):
        from tools import ToolRiskLevel

        def safe_func():
            return "safe"

        schema = {"name": "safe_func", "description": "safe", "parameters": {"type": "object", "properties": {}}}
        register_tool("safe_func", safe_func, schema, risk_level=ToolRiskLevel.SAFE)

        tool_info = get_tool("safe_func")
        self.assertEqual(tool_info["risk_level"], ToolRiskLevel.SAFE)
        res = execute_tool("safe_func", {})
        self.assertEqual(res, "safe")


if __name__ == "__main__":
    unittest.main()

