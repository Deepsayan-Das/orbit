"""
Orbit Sub-Agent Subsystem and delegate_task tool implementation.
"""

import uuid
from typing import Any, Dict, List, Optional

from agent_loop import run_agent_turn
from config import load_config
from llm_client import OrbitLLM
from providers.base import ChatMessage
import tools.registry as tool_registry

_SUBAGENT_DEPTH: int = 0

DELEGATE_TASK_SCHEMA = {
    "name": "delegate_task",
    "description": "Delegate a sub-task to an isolated sub-agent (depth-limited, safe-tools restricted).",
    "parameters": {
        "type": "object",
        "properties": {
            "task": {
                "type": "string",
                "description": "Task instruction string for the sub-agent to complete.",
            }
        },
        "required": ["task"],
    },
}


def run_subagent(
    task: str,
    tools: str = "safe",
    max_steps: int = 5,
    agent: Optional[OrbitLLM] = None,
    config: Optional[Any] = None,
    registered_tools: Optional[List[Dict[str, Any]]] = None,
) -> str:
    """Run a delegated task in an isolated sub-agent loop.

    Depth limit: 1 (sub-agents cannot spawn nested sub-agents).
    Failures are returned as error text, never raised.
    """
    global _SUBAGENT_DEPTH

    if _SUBAGENT_DEPTH >= 1:
        return "[error: sub-agent depth limit (1) exceeded]"

    try:
        _SUBAGENT_DEPTH += 1

        if agent is None:
            if config is None:
                config = load_config()
            agent = OrbitLLM.from_config(config)

        # Build isolated new history
        sub_messages = [ChatMessage(role="user", content=task)]

        # Dedicated subagent system prompt
        sub_system_prompt = (
            "You are a specialized sub-agent for Orbit. "
            "Complete the delegated task concisely and return only your final answer."
        )

        # Filter tool schemas based on allowed risk tier (default: SAFE only)
        if registered_tools is None:
            all_schemas = tool_registry.get_tools_schema()
        else:
            all_schemas = registered_tools

        sub_tools_schema = []
        for schema_item in all_schemas:
            fn = schema_item.get("function", {})
            name = fn.get("name", "")
            tinfo = tool_registry.get_tool(name)
            if tinfo:
                risk = tinfo.get("risk_level", tool_registry.ToolRiskLevel.SAFE)
                if tools == "safe" and risk == tool_registry.ToolRiskLevel.SAFE:
                    sub_tools_schema.append(schema_item)
                elif tools == "all":
                    sub_tools_schema.append(schema_item)
            elif tools == "safe":
                sub_tools_schema.append(schema_item)

        result = run_agent_turn(
            agent=agent,
            messages=sub_messages,
            system_prompt=sub_system_prompt,
            registered_tools=sub_tools_schema,
            max_steps=max_steps,
            print_status=False,
        )

        content = (result.content or "").strip()
        if len(content) > 2000:
            content = content[:1950] + "\n\n[sub-agent result capped at 500 tokens]"

        return content

    except Exception as e:
        return f"[error executing sub-agent task: {e}]"
    finally:
        _SUBAGENT_DEPTH -= 1


def delegate_task(task: str) -> str:
    """Tool function: delegate a task to an isolated sub-agent."""
    agent_id = f"subagent-{uuid.uuid4().hex[:6]}"
    tool_registry.log_audit_event(
        tool_name="delegate_task",
        risk_level="sensitive",
        status="SPAWNED",
        kwargs={"task": task, "agent_id": agent_id},
        result_summary=f"Spawned subagent {agent_id}",
    )
    return run_subagent(task)


def register_subagent_tool() -> None:
    """Register delegate_task in tool registry at SENSITIVE risk level."""
    tool_registry.register_tool(
        name="delegate_task",
        function=delegate_task,
        schema=DELEGATE_TASK_SCHEMA,
        risk_level=tool_registry.ToolRiskLevel.SENSITIVE,
    )


# Automatically register on module import
register_subagent_tool()
