"""
Developer test and debugging tools for Orbit.

Provides run_tests and search_logs tools for automated test execution
and log file introspection.
"""

import os
import shutil
import subprocess
from typing import Any, Dict

from .registry import ToolRiskLevel, register_tool

RUN_TESTS_SCHEMA: Dict[str, Any] = {
    "name": "run_tests",
    "description": "Run project test suite (detects pytest or unittest) and return pass/fail output summary.",
    "parameters": {
        "type": "object",
        "properties": {
            "path": {
                "type": "string",
                "description": "Directory or specific test file path to execute (default: '.')."
            }
        }
    }
}

SEARCH_LOGS_SCHEMA: Dict[str, Any] = {
    "name": "search_logs",
    "description": "Search for a specific text pattern inside a log or text file (grep-style).",
    "parameters": {
        "type": "object",
        "properties": {
            "pattern": {
                "type": "string",
                "description": "Text pattern or substring to search for."
            },
            "file": {
                "type": "string",
                "description": "Path to the log/text file to search."
            }
        },
        "required": ["pattern", "file"]
    }
}


def run_tests(path: str = ".") -> str:
    """Execute project automated unit tests using pytest or unittest."""
    try:
        target_path = path.strip() if path and path.strip() else "."
        if shutil.which("pytest"):
            cmd = ["pytest", target_path]
        else:
            test_dir = target_path if os.path.isdir(target_path) else "tests"
            cmd = ["python", "-m", "unittest", "discover", "-s", test_dir]

        proc = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=60
        )

        output = proc.stdout
        if proc.stderr:
            output += ("\n" if output else "") + proc.stderr

        output = output.strip()
        if len(output) > 1500:
            output = output[:1450] + f"\n... [output truncated, total length {len(output)} chars]"

        status = "PASSED" if proc.returncode == 0 else f"FAILED (exit code {proc.returncode})"
        return f"Test Run {status}:\n{output}"
    except subprocess.TimeoutExpired:
        return f"Error: Test suite execution timed out after 60 seconds."
    except Exception as e:
        return f"Error running tests: {str(e)}"


def search_logs(pattern: str, file: str) -> str:
    """Grep-style pattern search within a log/text file."""
    if not pattern:
        return "Error: Search pattern cannot be empty."
    if not file or not os.path.exists(file):
        return f"Error: Target file '{file}' does not exist."

    try:
        matches = []
        pattern_lower = pattern.lower()
        with open(file, "r", encoding="utf-8", errors="replace") as f:
            for idx, line in enumerate(f, start=1):
                if pattern_lower in line.lower():
                    matches.append(f"L{idx}: {line.rstrip()}")
                    if len(matches) >= 50:
                        matches.append("... [capped at 50 matching lines]")
                        break

        if not matches:
            return f"No matches found for pattern '{pattern}' in '{file}'."
        return "\n".join(matches)
    except Exception as e:
        return f"Error searching logs in '{file}': {str(e)}"


def register_dev_tools():
    """Register developer test and debugging tools into global tool registry."""
    register_tool("run_tests", run_tests, RUN_TESTS_SCHEMA, risk_level=ToolRiskLevel.SENSITIVE)
    register_tool("search_logs", search_logs, SEARCH_LOGS_SCHEMA, risk_level=ToolRiskLevel.SAFE)


register_dev_tools()
