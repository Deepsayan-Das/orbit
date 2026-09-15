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


GIT_LOG_SCHEMA: Dict[str, Any] = {
    "name": "git_log",
    "description": "Get recent Git commit history (hash, author, date, message).",
    "parameters": {
        "type": "object",
        "properties": {
            "n": {
                "type": "integer",
                "description": "Number of recent commits to show (default: 10)."
            }
        }
    }
}

GIT_BLAME_SCHEMA: Dict[str, Any] = {
    "name": "git_blame",
    "description": "Show author and commit information for each line of a file.",
    "parameters": {
        "type": "object",
        "properties": {
            "file": {
                "type": "string",
                "description": "Path to the file to run git blame on."
            },
            "line": {
                "type": "integer",
                "description": "Optional specific line number to inspect."
            }
        },
        "required": ["file"]
    }
}

GIT_COMMIT_SCHEMA: Dict[str, Any] = {
    "name": "git_commit",
    "description": "Stage changes (git add .) and create a Git commit. Always use this tool for Git commits instead of run_shell_command. Requires explicit permission confirmation.",
    "parameters": {
        "type": "object",
        "properties": {
            "message": {
                "type": "string",
                "description": "The commit message string."
            },
            "stage_all": {
                "type": "boolean",
                "description": "Whether to stage all modified/untracked files (git add .) before committing (default: True)."
            }
        },
        "required": ["message"]
    }
}

GIT_PUSH_SCHEMA: Dict[str, Any] = {
    "name": "git_push",
    "description": "Push committed Git changes to remote repository. Always use this tool for Git push operations instead of run_shell_command. Requires explicit permission confirmation.",
    "parameters": {
        "type": "object",
        "properties": {
            "remote": {
                "type": "string",
                "description": "Remote name (default: 'origin')."
            },
            "branch": {
                "type": "string",
                "description": "Branch name to push (default: current branch)."
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


def git_log(n: int = 10) -> str:
    """Get recent Git commit history."""
    try:
        count = int(n) if n is not None else 10
        if count <= 0:
            count = 10
        cmd = ["git", "log", f"-n{count}", "--pretty=format:%h - %an (%cr): %s"]
        proc = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=15
        )
        if proc.returncode != 0:
            return f"Git error: {proc.stderr.strip()}"
        return proc.stdout.strip() if proc.stdout.strip() else "No commits found."
    except Exception as e:
        return f"Error executing git log: {str(e)}"


def git_blame(file: str, line: int = 0) -> str:
    """Show Git blame for a specific file and optional line."""
    try:
        if not file:
            return "Error: File path is required for git blame."
        cmd = ["git", "blame"]
        line_num = int(line) if line is not None else 0
        if line_num > 0:
            cmd.extend([f"-L{line_num},{line_num}"])
        cmd.append(file)

        proc = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=15
        )
        if proc.returncode != 0:
            return f"Git error: {proc.stderr.strip()}"

        output = proc.stdout.strip()
        lines = output.splitlines()
        if len(lines) > 100:
            output = "\n".join(lines[:100]) + f"\n... [truncated {len(lines) - 100} additional lines]"
        return output if output else f"No blame output for file '{file}'."
    except Exception as e:
        return f"Error executing git blame: {str(e)}"


def git_commit(message: str, stage_all: bool = True) -> str:
    """Stage changes and commit with the specified message."""
    try:
        if not message.strip():
            return "Error: Commit message cannot be empty."

        if stage_all:
            add_proc = subprocess.run(
                ["git", "add", "."],
                capture_output=True,
                text=True,
                timeout=15
            )
            if add_proc.returncode != 0:
                return f"Git error during staging: {add_proc.stderr.strip()}"

        commit_proc = subprocess.run(
            ["git", "commit", "-m", message.strip()],
            capture_output=True,
            text=True,
            timeout=15
        )
        if commit_proc.returncode != 0:
            return f"Git error during commit: {commit_proc.stderr.strip()}"

        return commit_proc.stdout.strip() if commit_proc.stdout.strip() else "Commit completed successfully."
    except Exception as e:
        return f"Error executing git commit: {str(e)}"


def git_push(remote: str = "origin", branch: str = "") -> str:
    """Push committed changes to a remote repository."""
    try:
        cmd = ["git", "push", remote]
        if branch.strip():
            cmd.append(branch.strip())

        proc = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=30
        )
        if proc.returncode != 0:
            return f"Git error during push: {proc.stderr.strip()}"

        return proc.stdout.strip() if proc.stdout.strip() else "Push completed successfully."
    except Exception as e:
        return f"Error executing git push: {str(e)}"


def register_git_tools():
    """Register git introspection and management tools into global tool registry."""
    register_tool("git_status", git_status, GIT_STATUS_SCHEMA, risk_level=ToolRiskLevel.SAFE)
    register_tool("git_diff", git_diff, GIT_DIFF_SCHEMA, risk_level=ToolRiskLevel.SAFE)
    register_tool("git_log", git_log, GIT_LOG_SCHEMA, risk_level=ToolRiskLevel.SAFE)
    register_tool("git_blame", git_blame, GIT_BLAME_SCHEMA, risk_level=ToolRiskLevel.SAFE)
    register_tool("git_commit", git_commit, GIT_COMMIT_SCHEMA, risk_level=ToolRiskLevel.DANGEROUS)
    register_tool("git_push", git_push, GIT_PUSH_SCHEMA, risk_level=ToolRiskLevel.DANGEROUS)


register_git_tools()

