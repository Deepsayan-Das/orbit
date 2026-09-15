"""
Unit tests for new Orbit tools (run_shell_command, git_status, git_diff, search_codebase, read_file, write_file).
"""

import os
import tempfile
import unittest

from tools import (
    clear_registry,
    execute_tool,
    execute_tool_unsafe_for_testing_only,
    git_diff,
    git_status,
    is_safe_command,
    list_registered_tools,
    read_file,
    run_shell_command,
    write_file,
)
from tools.registry import AUDIT_LOG_PATH


class TestNewTools(unittest.TestCase):

    def test_registered_tools_presence(self):
        registered = list_registered_tools()
        self.assertIn("read_file", registered)
        self.assertIn("git_status", registered)
        self.assertIn("git_diff", registered)
        self.assertIn("search_codebase", registered)
        self.assertIn("list_directory", registered)
        self.assertIn("write_file", registered)
        self.assertIn("run_shell_command", registered)

    def test_read_file(self):
        with tempfile.NamedTemporaryFile("w", delete=False, encoding="utf-8") as f:
            f.write("hello orbit read test")
            temp_path = f.name

        try:
            content = read_file(temp_path)
            self.assertEqual(content, "hello orbit read test")
        finally:
            if os.path.exists(temp_path):
                os.remove(temp_path)

    def test_write_file(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            test_path = os.path.join(temp_dir, "sub", "test.txt")
            res = write_file(test_path, "sample content")
            self.assertIn("Successfully wrote", res)
            self.assertTrue(os.path.exists(test_path))

            with open(test_path, "r", encoding="utf-8") as f:
                self.assertEqual(f.read(), "sample content")

    def test_shell_command_safe_allowlist(self):
        self.assertTrue(is_safe_command("dir"))
        self.assertTrue(is_safe_command("git status"))
        self.assertTrue(is_safe_command("python --version"))
        self.assertFalse(is_safe_command("rm -rf /"))
        self.assertFalse(is_safe_command("del /f /q *"))
        self.assertFalse(is_safe_command("dir | grep test"))

    def test_run_shell_command_exec(self):
        res = run_shell_command("python --version")
        self.assertIn("Python", res)

    def test_git_status_and_diff(self):
        status_res = git_status()
        self.assertIsInstance(status_res, str)
        self.assertNotIn("Error executing", status_res)

        diff_res = git_diff()
        self.assertIsInstance(diff_res, str)
        self.assertNotIn("Error executing", diff_res)

    def test_audit_log_created(self):
        res = execute_tool_unsafe_for_testing_only("read_file", {"path": "README.md"})
        self.assertTrue(os.path.exists(AUDIT_LOG_PATH))


if __name__ == "__main__":
    unittest.main()
