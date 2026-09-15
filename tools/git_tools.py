"""
Git Introspection tools for Orbit.

Provides read-only git_status and git_diff tools using native git CLI subprocess calls.
"""

import subprocess
from typing import Any, Dict

from .registry import ToolRiskLevel, register_tool

GIT_STATUS_SCHEMA: Dict[str, Any] = {
    "name": "git_status",
    "description": "Get the current Git repository status (modified files, untracked files, branch state).",
    "parameters": {
        "type": "object",
        "properties": {}
    }
}

GIT_DIFF_SCHEMA: Dict[str, Any] = {
    "name": "git_diff",
    "description": "Get the uncommitted or staged Git diff changes in the workspace repository.",
    "parameters": {
        "type": "object",
        "properties": {
            "staged": {
                "type": "boolean",
                "description": "If True, returns staged diff changes (--staged). If False, returns unstaged working directory diff."
            }
        }
    }
}


def git_status() -> str:
    """Get the current Git status overview for the repository."""
    try:
        proc = subprocess.run(
            ["git", "status"],
            capture_output=True,
            text=True,
            timeout=15
        )
        if proc.returncode != 0:
            return f"Git error: {proc.stderr.strip()}"
        return proc.stdout.strip() if proc.stdout.strip() else "Git workspace is clean."
    except Exception as e:
        return f"Error executing git status: {str(e)}"


def git_diff(staged: bool = False) -> str:
    """Get Git diff changes for unstaged or staged files."""
    try:
        cmd = ["git", "diff", "--staged"] if staged else ["git", "diff"]
        proc = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=15
        )
        if proc.returncode != 0:
            return f"Git error: {proc.stderr.strip()}"
        return proc.stdout.strip() if proc.stdout.strip() else f"No {'staged' if staged else 'unstaged'} git diff changes found."
    except Exception as e:
        return f"Error executing git diff: {str(e)}"


def register_git_tools():
    """Register git introspection tools into global tool registry."""
    register_tool("git_status", git_status, GIT_STATUS_SCHEMA, risk_level=ToolRiskLevel.SAFE)
    register_tool("git_diff", git_diff, GIT_DIFF_SCHEMA, risk_level=ToolRiskLevel.SAFE)


register_git_tools()
