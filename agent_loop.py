"""
Orbit Agentic Loop — multi-step tool-chaining orchestrator.

Implements the ReAct-style loop: send messages → if model returns tool_calls,
execute them via the existing registry/permission-gate and loop back; if the
model returns plain content, that's the final answer.

This module **orchestrates** — it never reimplements tool execution, permission
checks, or provider logic.  Those responsibilities stay in ``tools.registry``
and ``llm_client``.
"""

from __future__ import annotations

import sys
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from llm_client import OrbitLLM
from providers import ChatMessage
import tools as tool_registry


# ── Helpers (re-used from repl.py's existing pure extractors) ────────────

def extract_tool_calls(raw_response: Any) -> list:
    """Extract tool calls list from raw provider response object or dict."""
    if not raw_response:
        return []

    # 1. Gemini GenAI SDK function_calls attribute
    fn_calls = getattr(raw_response, "function_calls", None)
    if fn_calls:
        res = []
        for fc in fn_calls:
            name = getattr(fc, "name", "") or (fc.get("name") if isinstance(fc, dict) else "")
            args = getattr(fc, "args", {}) or (fc.get("args") if isinstance(fc, dict) else {})
            if isinstance(args, dict):
                args_dict = dict(args)
            else:
                try:
                    args_dict = dict(args)
                except Exception:
                    args_dict = {}
            res.append({
                "function": {
                    "name": name,
                    "arguments": args_dict
                }
            })
        return res

    # 2. OpenAI / Ollama standard tool calls format
    if isinstance(raw_response, dict):
        msg = raw_response.get("message", {})
        if isinstance(msg, dict):
            return msg.get("tool_calls") or []
        return getattr(msg, "tool_calls", None) or []
    else:
        msg = getattr(raw_response, "message", None)
        if msg:
            if isinstance(msg, dict):
                return msg.get("tool_calls") or []
            return getattr(msg, "tool_calls", None) or []
    return []


def parse_tool_call(call: Any) -> tuple[str, dict]:
    """Parse tool name and keyword arguments from a raw tool call item."""
    if isinstance(call, dict):
        fn = call.get("function", {})
        if isinstance(fn, dict):
            name = fn.get("name", "")
            args = fn.get("arguments", {})
            return name, args if isinstance(args, dict) else dict(args)
        name = getattr(fn, "name", "")
        args = getattr(fn, "arguments", {})
        return name, args if isinstance(args, dict) else dict(args)
    else:
        fn = getattr(call, "function", None)
        if fn:
            if isinstance(fn, dict):
                name = fn.get("name", "")
                args = fn.get("arguments", {})
                return name, args if isinstance(args, dict) else dict(args)
            name = getattr(fn, "name", "")
            args = getattr(fn, "arguments", {})
            return name, args if isinstance(args, dict) else dict(args)
    return "", {}


# ── Result container ─────────────────────────────────────────────────────

@dataclass
class AgentTurnResult:
    """Outcome of a single agent turn (one user message → final answer)."""
    content: str
    steps_taken: int
    hit_step_limit: bool = False
    tool_calls_log: List[Dict[str, Any]] = field(default_factory=list)


# ── Core loop ────────────────────────────────────────────────────────────

def run_agent_turn(
    agent: OrbitLLM,
    messages: List[ChatMessage],
    system_prompt: str,
    registered_tools: List[Dict[str, Any]],
    *,
    max_steps: int = 8,
    print_status: bool = True,
    stream_final: bool = False,
) -> AgentTurnResult:
    """Execute a multi-step agentic loop for one user turn."""
    tool_calls_log: List[Dict[str, Any]] = []
    step = 0

    while step < max_steps:
        step += 1

        # Check compaction trigger before model chat call
        try:
            from context.window import resolve_window
            from context.measure import estimate_tokens, should_compact
            from commands.compaction import compact_history

            w_size = getattr(agent, "window_size", None)
            if w_size is None:
                prov_name = getattr(agent, "provider_name", "ollama")
                mod_name = getattr(agent, "model", "")
                w_size, _ = resolve_window(prov_name, mod_name)

            used_tokens = estimate_tokens(messages)
            if should_compact(used_tokens, w_size):
                old_t = used_tokens
                compacted = compact_history(messages, focus=None, agent=agent, window=w_size)
                messages.clear()
                messages.extend(compacted)
                new_t = estimate_tokens(messages)
                if print_status:
                    try:
                        from ui import console
                        console.print(f"[bold yellow][compaction triggered: ~{old_t} tokens -> ~{new_t} tokens (window: {w_size})][/bold yellow]")
                    except Exception:
                        print(f"[compaction triggered: ~{old_t} tokens -> ~{new_t} tokens (window: {w_size})]")
        except Exception as e:
            pass

        # ── 1. Ask the model (non-streamed, tools attached) ──────────
        resp = agent.chat(
            messages=messages,
            system_prompt=system_prompt,
            stream=False,
            tools=registered_tools,
        )

        # ── 2. Check for tool calls ──────────────────────────────────
        raw_tool_calls = extract_tool_calls(resp.raw_response)
        if not raw_tool_calls:
            # Plain content — final answer.
            if stream_final and print_status:
                try:
                    from ui import print_orbit_response
                    print_orbit_response(resp.content)
                except Exception:
                    print(f"\nOrbit: {resp.content}")
            return AgentTurnResult(
                content=resp.content,
                steps_taken=step,
                hit_step_limit=False,
                tool_calls_log=tool_calls_log,
            )

        # ── 3. Execute each tool call in this response ───────────────
        messages.append(
            ChatMessage(role="assistant", content=resp.content or "", tool_calls=raw_tool_calls)
        )
        tool_executed_any = False
        for call in raw_tool_calls:
            fn_name, fn_args = parse_tool_call(call)
            if not fn_name or not tool_registry.get_tool(fn_name):
                continue

            if print_status:
                try:
                    from ui import print_step_start, print_step_result
                    print_step_start(step, max_steps, fn_name, fn_args)
                except Exception:
                    print(f"\n[step {step}/{max_steps}] calling {fn_name}...", flush=True)

            # Delegate to the existing registry — permission gate included.
            tool_result = tool_registry.execute_tool(fn_name, fn_args)

            if print_status:
                try:
                    from ui import print_step_result
                    print_step_result(str(tool_result))
                except Exception:
                    summary = str(tool_result)[:120]
                    print(f"  → {summary}{'...' if len(str(tool_result)) > 120 else ''}")

            tool_calls_log.append({
                "step": step,
                "tool": fn_name,
                "args": fn_args,
                "result_summary": str(tool_result)[:200],
            })

            messages.append(
                ChatMessage(
                    role="tool",
                    content=f"[Tool Result for {fn_name}]:\n{tool_result}",
                    name=fn_name,
                )
            )
            tool_executed_any = True

        if not tool_executed_any:
            # Model asked for tools that don't exist — treat as final answer.
            return AgentTurnResult(
                content=resp.content,
                steps_taken=step,
                hit_step_limit=False,
                tool_calls_log=tool_calls_log,
            )

    # ── 4. Max steps reached ─────────────────────────────────────────
    # Do one last plain call (no tools) to get whatever content the model
    # can produce, then annotate with the ceiling note.
    final_resp = agent.chat(
        messages=messages,
        system_prompt=system_prompt,
        stream=False,
    )
    ceiling_note = (
        f"\n\n[agent note: step limit ({max_steps}) reached — "
        "the plan may be incomplete]"
    )
    return AgentTurnResult(
        content=(final_resp.content or "") + ceiling_note,
        steps_taken=max_steps,
        hit_step_limit=True,
        tool_calls_log=tool_calls_log,
    )
