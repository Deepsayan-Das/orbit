"""
Orbit Tools package.
Provides tool registry, risk levels (ToolRiskLevel), persistent audit logging, and builtin tools.
"""

from .registry import (
    ToolRiskLevel,
    clear_registry,
    execute_tool,
    execute_tool_unsafe_for_testing_only,
    get_tool,
    get_tools_schema,
    list_registered_tools,
    log_audit_event,
    register_tool,
)
from .filesystem import list_directory, register_filesystem_tools, write_file
from .read_file import read_file
from .shell import is_safe_command, run_shell_command
from .git_tools import git_blame, git_commit, git_diff, git_log, git_push, git_status
from .container_tools import container_logs, container_ps
from .dev_tools import run_tests, search_logs
from .code_search import search_codebase

__all__ = [
    "ToolRiskLevel",
    "register_tool",
    "get_tool",
    "get_tools_schema",
    "execute_tool",
    "execute_tool_unsafe_for_testing_only",
    "log_audit_event",
    "list_registered_tools",
    "clear_registry",
    "list_directory",
    "register_filesystem_tools",
    "read_file",
    "write_file",
    "run_shell_command",
    "is_safe_command",
    "git_status",
    "git_diff",
    "git_log",
    "git_blame",
    "git_commit",
    "git_push",
    "container_ps",
    "container_logs",
    "run_tests",
    "search_logs",
    "search_codebase",
]
