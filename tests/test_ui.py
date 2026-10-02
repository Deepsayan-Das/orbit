"""
Unit tests for Orbit Terminal UI module (ui.py).
"""

import unittest
from unittest.mock import patch
import io

import ui


class TestOrbitUI(unittest.TestCase):

    def test_print_banner_runs_without_error(self):
        """Verify banner rendering succeeds."""
        ui.print_banner(chunk_count=42, tool_count=10, provider="ollama", model="llama3.2")

    def test_print_orbit_response_renders_markdown(self):
        """Verify markdown response rendering."""
        sample_markdown = "# Heading\nHere is `code`: \n```python\nprint('hello')\n```"
        ui.print_orbit_response(sample_markdown)

    def test_print_step_start_and_result(self):
        """Verify step execution progress rendering."""
        ui.print_step_start(step=1, max_steps=8, tool_name="edit_file", fn_args={"path": "test.txt", "old_text": "a", "new_text": "b"})
        ui.print_step_result(summary="File updated successfully.")

    def test_print_diff_colorization(self):
        """Verify unified diff rendering."""
        sample_diff = "--- file.txt\n+++ file.txt\n@@ -1,3 +1,3 @@\n-old line\n+new line\n context line"
        ui.print_diff(sample_diff)

    def test_callout_helpers(self):
        """Verify info, warning, error, success callouts."""
        ui.print_info("Info message")
        ui.print_warning("Warning message")
        ui.print_error("Error message")
        ui.print_success("Success message")


if __name__ == "__main__":
    unittest.main()
