"""
Unit tests for edit_file tool, diff_utils pure logic, and preview_fn registry integration.
"""

import os
import tempfile
import unittest

from tools.diff_utils import compute_unified_diff, validate_and_replace
from tools.registry import (
    ToolRiskLevel,
    build_confirmation_display,
    clear_registry,
    get_tool,
    register_tool,
)
from tools.filesystem import edit_file, _edit_file_preview


# ---------------------------------------------------------------------------
# 1. Pure diff_utils tests
# ---------------------------------------------------------------------------

class TestValidateAndReplace(unittest.TestCase):
    """Test the pure validate_and_replace logic (no file I/O)."""

    def test_single_match_replaces(self):
        content = "hello world\nfoo bar\n"
        modified, error = validate_and_replace(content, "foo", "baz")
        self.assertIsNone(error)
        self.assertEqual(modified, "hello world\nbaz bar\n")

    def test_no_match_returns_error(self):
        content = "hello world\n"
        modified, error = validate_and_replace(content, "missing", "x")
        self.assertIsNone(modified)
        self.assertIn("not found", error)

    def test_ambiguous_multiple_match_returns_error(self):
        content = "aaa\naaa\n"
        modified, error = validate_and_replace(content, "aaa", "bbb")
        self.assertIsNone(modified)
        self.assertIn("2 times", error)
        self.assertIn("ambiguous", error)

    def test_three_occurrences_reports_count(self):
        content = "x\nx\nx\n"
        modified, error = validate_and_replace(content, "x", "y")
        self.assertIsNone(modified)
        self.assertIn("3 times", error)


class TestComputeUnifiedDiff(unittest.TestCase):
    """Test the pure unified-diff formatter."""

    def test_diff_contains_plus_minus_markers(self):
        original = "line1\nline2\nline3\n"
        modified = "line1\nchanged\nline3\n"
        diff = compute_unified_diff(original, modified, filepath="test.py")
        self.assertIn("-line2", diff)
        self.assertIn("+changed", diff)
        self.assertIn("a/test.py", diff)
        self.assertIn("b/test.py", diff)

    def test_identical_content_produces_empty_diff(self):
        content = "same\n"
        diff = compute_unified_diff(content, content)
        self.assertEqual(diff, "")


# ---------------------------------------------------------------------------
# 2. edit_file tool integration (file I/O)
# ---------------------------------------------------------------------------

class TestEditFileTool(unittest.TestCase):
    """Test edit_file end-to-end with real temp files."""

    def setUp(self):
        self.tmpdir = tempfile.mkdtemp()

    def _write_temp(self, name: str, content: str) -> str:
        path = os.path.join(self.tmpdir, name)
        with open(path, "w", encoding="utf-8") as f:
            f.write(content)
        return path

    def _read_temp(self, path: str) -> str:
        with open(path, "r", encoding="utf-8") as f:
            return f.read()

    def test_successful_single_match_edit(self):
        path = self._write_temp("a.txt", "alpha\nbeta\ngamma\n")
        result = edit_file(path, "beta", "BETA")
        self.assertIn("Successfully edited", result)
        self.assertEqual(self._read_temp(path), "alpha\nBETA\ngamma\n")
        # The result should include a diff
        self.assertIn("-beta", result)
        self.assertIn("+BETA", result)

    def test_no_match_returns_error_and_leaves_file_unchanged(self):
        original = "alpha\nbeta\n"
        path = self._write_temp("b.txt", original)
        result = edit_file(path, "missing_text", "x")
        self.assertIn("not found", result)
        # File must be untouched
        self.assertEqual(self._read_temp(path), original)

    def test_ambiguous_multiple_match_returns_error_and_leaves_file_unchanged(self):
        original = "foo\nbar\nfoo\n"
        path = self._write_temp("c.txt", original)
        result = edit_file(path, "foo", "baz")
        self.assertIn("ambiguous", result)
        self.assertIn("2 times", result)
        # File must be untouched
        self.assertEqual(self._read_temp(path), original)

    def test_nonexistent_file_returns_error(self):
        result = edit_file(os.path.join(self.tmpdir, "nope.txt"), "a", "b")
        self.assertIn("Error reading file", result)


# ---------------------------------------------------------------------------
# 3. preview_fn / build_confirmation_display registry tests
# ---------------------------------------------------------------------------

class TestPreviewFnRegistryIntegration(unittest.TestCase):
    """Test that build_confirmation_display uses preview_fn when present
    and falls back to raw-args display when absent."""

    def setUp(self):
        clear_registry()

    def tearDown(self):
        # Re-register all builtin tools so later test modules (which expect
        # them to be present) are not affected by our clear_registry() calls.
        clear_registry()
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

    def test_preview_fn_output_used_when_present(self):
        def my_preview(kwargs):
            return f"PREVIEW:{kwargs['x']}"

        register_tool(
            "with_preview",
            lambda x: x,
            {"name": "with_preview", "description": "", "parameters": {"type": "object", "properties": {}}},
            risk_level=ToolRiskLevel.DANGEROUS,
            preview_fn=my_preview,
        )

        tool_info = get_tool("with_preview")
        display = build_confirmation_display(tool_info, {"x": 42})
        self.assertEqual(display, "PREVIEW:42")

    def test_default_raw_args_used_when_no_preview_fn(self):
        register_tool(
            "no_preview",
            lambda: "ok",
            {"name": "no_preview", "description": "", "parameters": {"type": "object", "properties": {}}},
            risk_level=ToolRiskLevel.DANGEROUS,
        )

        tool_info = get_tool("no_preview")
        kwargs = {"a": 1, "b": "two"}
        display = build_confirmation_display(tool_info, kwargs)
        # Should be the default str(kwargs)
        self.assertEqual(display, str(kwargs))

    def test_broken_preview_fn_falls_back_to_raw_args(self):
        def broken_preview(kwargs):
            raise RuntimeError("boom")

        register_tool(
            "broken_preview",
            lambda: "ok",
            {"name": "broken_preview", "description": "", "parameters": {"type": "object", "properties": {}}},
            risk_level=ToolRiskLevel.DANGEROUS,
            preview_fn=broken_preview,
        )

        tool_info = get_tool("broken_preview")
        kwargs = {"key": "value"}
        display = build_confirmation_display(tool_info, kwargs)
        self.assertEqual(display, str(kwargs))

    def test_edit_file_preview_fn_returns_diff(self):
        """Verify that the actual _edit_file_preview function produces diff output."""
        tmpdir = tempfile.mkdtemp()
        path = os.path.join(tmpdir, "preview_test.txt")
        with open(path, "w") as f:
            f.write("line1\nline2\nline3\n")

        preview = _edit_file_preview({"path": path, "old_text": "line2", "new_text": "REPLACED"})
        self.assertIn("-line2", preview)
        self.assertIn("+REPLACED", preview)


if __name__ == "__main__":
    unittest.main()
