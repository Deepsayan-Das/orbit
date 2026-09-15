"""
Orbit Tool Registry & Audit Logging System.

Provides dynamic tool registration, schema aggregation for LLM provider chat calls,
mandatory risk-level permission enforcement (SAFE, SENSITIVE, DANGEROUS), tool execution routing,
and persistent local audit logging.
"""

from datetime import datetime
from enum import Enum
import os
from typing import Any, Callable, Dict, List, Optional

AUDIT_LOG_PATH = "./.orbit/audit.log"


class ToolRiskLevel(Enum):
    SAFE = "safe"           # read-only — auto-execute, no prompt
    SENSITIVE = "sensitive" # can change state — require user confirmation
    DANGEROUS = "dangerous" # broad/destructive potential — require explicit confirmation + show exact command/content before running


_TOOL_REGISTRY: Dict[str, Dict[str, Any]] = {}


def log_audit_event(
    tool_name: str, 
    risk_level: str, 
    status: str, 
    kwargs: Dict[str, Any], 
    result_summary: str
) -> None:
    """Append a structured line to local persistent audit log file."""
    try:
        os.makedirs(os.path.dirname(AUDIT_LOG_PATH), exist_ok=True)
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        single_line_result = result_summary.replace("\n", " ").strip()
        if len(single_line_result) > 150:
            single_line_result = single_line_result[:147] + "..."

        log_line = (
            f"[{timestamp}] TOOL: {tool_name} | RISK: {risk_level} | "
            f"STATUS: {status} | ARGS: {kwargs} | RESULT: {single_line_result}\n"
        )
        with open(AUDIT_LOG_PATH, "a", encoding="utf-8") as f:
            f.write(log_line)
    except Exception as e:
        print(f"[warn] Failed to write to audit log: {e}")


def register_tool(
    name: str, 
    function: Callable[..., Any], 
    schema: Dict[str, Any],
    risk_level: ToolRiskLevel = ToolRiskLevel.SAFE
) -> None:
    """Register a tool function along with its JSON schema definition and risk level."""
    _TOOL_REGISTRY[name] = {
        "name": name,
        "function": function,
        "schema": schema,
        "risk_level": risk_level,
    }


def get_tool(name: str) -> Optional[Dict[str, Any]]:
    """Retrieve a registered tool dictionary by name."""
    return _TOOL_REGISTRY.get(name)


def get_tools_schema() -> List[Dict[str, Any]]:
    """
    Return the full list of tool definitions in OpenAI/Ollama function-calling format
    ready to pass into LLM provider chat() calls (e.g. tools=[...]).
    """
    schemas = []
    for tool_info in _TOOL_REGISTRY.values():
        schemas.append({
            "type": "function",
            "function": tool_info["schema"]
        })
    return schemas


def execute_tool(name: str, kwargs: Dict[str, Any]) -> Any:
    """
    Execute a registered tool function with keyword arguments.
    Enforces mandatory risk-level permission checks (SENSITIVE / DANGEROUS) and logs audit events.
    """
    tool_info = _TOOL_REGISTRY.get(name)
    if not tool_info:
        raise ValueError(f"Tool '{name}' is not registered in the tool registry.")

    risk_level = tool_info.get("risk_level", ToolRiskLevel.SAFE)
    status = "AUTO_APPROVED"

    if risk_level == ToolRiskLevel.SENSITIVE:
        prompt_str = f"\n[permission check - SENSITIVE] Allow execution of '{name}' with args {kwargs}? (y/n): "
        response = input(prompt_str).strip().lower()
        if response not in ("y", "yes"):
            log_audit_event(name, risk_level.value, "DENIED", kwargs, "Cancelled by user permission check")
            return f"Permission denied: Execution of sensitive tool '{name}' was cancelled by user."
        status = "APPROVED"

    elif risk_level == ToolRiskLevel.DANGEROUS:
        prompt_str = f"\n[permission check - DANGEROUS] High-risk action! Allow '{name}' with args:\n{kwargs}\nExecute? (y/n): "
        response = input(prompt_str).strip().lower()
        if response not in ("y", "yes"):
            log_audit_event(name, risk_level.value, "DENIED", kwargs, "Cancelled by user permission check")
            return f"Permission denied: Execution of dangerous tool '{name}' was cancelled by user."
        status = "APPROVED"

    try:
        result = tool_info["function"](**kwargs)
        log_audit_event(name, risk_level.value, status, kwargs, str(result))
        return result
    except Exception as e:
        err_msg = f"Error executing tool '{name}': {str(e)}"
        log_audit_event(name, risk_level.value, f"{status}_ERROR", kwargs, err_msg)
        return err_msg


def execute_tool_unsafe_for_testing_only(name: str, kwargs: Dict[str, Any]) -> Any:
    """
    Explicit helper function bypassing interactive prompts STRICTLY for automated unit tests.
    Do NOT call this on main execution paths.
    """
    tool_info = _TOOL_REGISTRY.get(name)
    if not tool_info:
        raise ValueError(f"Tool '{name}' is not registered in the tool registry.")

    risk_level = tool_info.get("risk_level", ToolRiskLevel.SAFE)
    try:
        result = tool_info["function"](**kwargs)
        log_audit_event(name, risk_level.value, "TEST_UNSAFE", kwargs, str(result))
        return result
    except Exception as e:
        err_msg = f"Error executing tool '{name}': {str(e)}"
        log_audit_event(name, risk_level.value, "TEST_UNSAFE_ERROR", kwargs, err_msg)
        return err_msg


def list_registered_tools() -> List[str]:
    """Return names of all currently registered tools."""
    return list(_TOOL_REGISTRY.keys())


def clear_registry() -> None:
    """Clear all registered tools (primarily for testing)."""
    _TOOL_REGISTRY.clear()
