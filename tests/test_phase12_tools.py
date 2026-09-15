"""
Unit tests for Phase 12 Deep Developer Integration tools.
"""

import os
import tempfile
import unittest

from tools import (
    container_logs,
    container_ps,
    git_blame,
    git_commit,
    git_diff,
    git_log,
    git_push,
    git_status,
    list_registered_tools,
    run_tests,
    search_logs,
)


class TestPhase12Tools(unittest.TestCase):

    def test_registered_tools_presence(self):
        registered = list_registered_tools()
        self.assertIn("git_log", registered)
        self.assertIn("git_blame", registered)
        self.assertIn("git_commit", registered)
        self.assertIn("git_push", registered)
        self.assertIn("container_ps", registered)
        self.assertIn("container_logs", registered)
        self.assertIn("run_tests", registered)
        self.assertIn("search_logs", registered)

    def test_git_log(self):
        res = git_log(n=3)
        self.assertIsInstance(res, str)
        self.assertNotIn("Error executing", res)

    def test_git_blame(self):
        res = git_blame(file="README.md")
        self.assertIsInstance(res, str)
        self.assertNotIn("Error executing", res)

    def test_container_ps(self):
        res = container_ps()
        self.assertIsInstance(res, str)
        # Either lists containers or reports no container runtime CLI found/no containers
        self.assertTrue(len(res) > 0)

    def test_container_logs(self):
        res = container_logs("non_existent_test_container")
        self.assertIsInstance(res, str)
        self.assertTrue(len(res) > 0)

    def test_run_tests(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            sample_test = os.path.join(temp_dir, "test_dummy.py")
            with open(sample_test, "w", encoding="utf-8") as f:
                f.write("import unittest\n\nclass DummyTest(unittest.TestCase):\n    def test_pass(self):\n        self.assertTrue(True)\n")

            res = run_tests(path=temp_dir)
            self.assertIsInstance(res, str)
            self.assertIn("Test Run", res)

    def test_search_logs(self):
        with tempfile.NamedTemporaryFile("w", delete=False, encoding="utf-8") as f:
            f.write("Line 1: INFO Starting server\nLine 2: ERROR Out of memory\nLine 3: INFO Shutting down\n")
            temp_path = f.name

        try:
            res = search_logs(pattern="ERROR", file=temp_path)
            self.assertIn("Out of memory", res)
            self.assertIn("L2:", res)

            res_empty = search_logs(pattern="WARNING", file=temp_path)
            self.assertIn("No matches found", res_empty)
        finally:
            if os.path.exists(temp_path):
                os.remove(temp_path)


if __name__ == "__main__":
    unittest.main()
