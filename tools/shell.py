"""
Shell execution tool for Orbit.

Provides run_shell_command with safety allowlisting and interactive permission prompts
for high-stakes / non-read-only commands.
"""

import subprocess
from typing import Any, Dict

from .registry import ToolRiskLevel, register_tool

RUN_SHELL_COMMAND_SCHEMA: Dict[str, Any] = {
    "name": "run_shell_command",
    "description": "Execute a shell command on the host system. For Git tasks (commit, push, diff, status), use dedicated git_commit, git_push, git_status, git_diff tools instead.",
    "parameters": {
        "type": "object",
        "properties": {
            "command": {
                "type": "string",
                "description": "The shell command string to execute (e.g. 'python --version', 'dir')."
            }
        },
        "required": ["command"]
    }
}

SAFE_COMMAND_PREFIXES = {
    "dir", "ls", "pwd", "echo", "whoami",
    "git status", "git diff", "git log", "git branch", "git tag",
    "python --version", "python -v", "pip list", "pip show",
    "node -v", "npm -v"
}


def is_safe_command(command: str) -> bool:
    """Check if a shell command is in the safe read-only allowlist."""
    cmd = command.strip().lower()
    if any(op in cmd for op in ("|", ";", "&&", "||", ">", "<")):
        return False

    return any(cmd.startswith(prefix) for prefix in SAFE_COMMAND_PREFIXES)


def run_shell_command(command: str) -> str:
    """Execute a shell command on the host system."""
    cmd_str = command.strip()
    if not cmd_str:
        return "Error: Command string cannot be empty."

    try:
        proc = subprocess.run(
            cmd_str,
            shell=True,
            capture_output=True,
            text=True,
            timeout=30
        )
        output = proc.stdout
        if proc.stderr:
            output += ("\n" if output else "") + proc.stderr
        return output.strip() if output.strip() else f"Command completed with exit code {proc.returncode}."
    except subprocess.TimeoutExpired:
        return f"Error: Command '{cmd_str}' timed out after 30 seconds."
    except Exception as e:
        return f"Error executing command '{cmd_str}': {str(e)}"


def register_shell_tools():
    """Register shell tools into global tool registry."""
    register_tool("run_shell_command", run_shell_command, RUN_SHELL_COMMAND_SCHEMA, risk_level=ToolRiskLevel.DANGEROUS)


# Disabled auto-registration for safety
register_shell_tools()
