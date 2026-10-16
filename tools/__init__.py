"""
Orbit Tools package.
Provides tool registry, risk levels (ToolRiskLevel), persistent audit logging, and builtin tools.
"""

from .registry import (
    ToolRiskLevel,
    build_confirmation_display,
    clear_registry,
    execute_tool,
    execute_tool_unsafe_for_testing_only,
    get_tool,
    get_tools_schema,
    list_registered_tools,
    log_audit_event,
    register_tool,
)
from .diff_utils import compute_unified_diff, validate_and_replace
from .filesystem import edit_file, list_directory, register_filesystem_tools, write_file
from .read_file import read_file
from .shell import is_safe_command, run_shell_command
from .git_tools import git_blame, git_commit, git_diff, git_log, git_push, git_status
from .container_tools import container_logs, container_ps
from .dev_tools import run_tests, search_logs
from .code_search import search_codebase
from agent_sub import delegate_task, register_subagent_tool, run_subagent


def reset_default_tools() -> None:
    """Clear and re-register all standard builtin tools into the tool registry."""
    clear_registry()
    register_filesystem_tools()
    from .read_file import register_read_file_tool
    from .shell import register_shell_tools
    from .git_tools import register_git_tools
    from .container_tools import register_container_tools
    from .dev_tools import register_dev_tools
    from .code_search import register_code_search_tools

    register_read_file_tool()
    register_shell_tools()
    register_git_tools()
    register_container_tools()
    register_dev_tools()
    register_code_search_tools()
    register_subagent_tool()


__all__ = [
    "reset_default_tools",
    "ToolRiskLevel",
    "register_tool",
    "get_tool",
    "get_tools_schema",
    "execute_tool",
    "execute_tool_unsafe_for_testing_only",
    "build_confirmation_display",
    "log_audit_event",
    "list_registered_tools",
    "clear_registry",
    "compute_unified_diff",
    "validate_and_replace",
    "list_directory",
    "register_filesystem_tools",
    "read_file",
    "write_file",
    "edit_file",
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
    "delegate_task",
    "run_subagent",
]
