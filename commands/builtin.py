"""
Built-in slash command handlers for Orbit.
"""

import dataclasses
import os
from pathlib import Path
from typing import Any, Dict

from commands.compaction import compact_history
from commands.registry import command, list_commands
from commands.session import Session
from commands.tokens import estimate_tokens
from llm_client import OrbitLLM
import tools.registry as tool_registry


def mask_secret(value: str) -> str:
    """Mask a secret string, revealing only the last 4 characters."""
    if not value:
        return "[not set]"
    if len(value) <= 4:
        return "****"
    return "*" * (len(value) - 4) + value[-4:]


def mask_config_data(data: Any, parent_key: str = "") -> Any:
    """Recursively mask sensitive keys in a config dictionary or object."""
    if isinstance(data, dict):
        masked = {}
        for k, v in data.items():
            key_str = str(k).lower()
            if any(sec in key_str for sec in ["key", "token", "secret", "password", "auth", "credential"]):
                if isinstance(v, str):
                    masked[k] = mask_secret(v)
                else:
                    masked[k] = "*****"
            else:
                masked[k] = mask_config_data(v, key_str)
        return masked
    elif isinstance(data, list):
        return [mask_config_data(item, parent_key) for item in data]
    return data


@command(name="help", aliases=["?"], help="List all available REPL slash commands")
def handle_help(args: str, session: Session) -> str:
    cmds = list_commands()
    lines = ["Available REPL Commands:"]
    for info in sorted(cmds, key=lambda c: c.name):
        alias_str = f" (aliases: {', '.join('/' + a for a in info.aliases)})" if info.aliases else ""
        lines.append(f"  /{info.name}{alias_str} - {info.help}")
    return "\n".join(lines)


@command(name="clear", aliases=["reset"], help="Clear conversation history")
def handle_clear(args: str, session: Session) -> str:
    session.history.clear()
    return "[history cleared]"


@command(name="exit", aliases=["quit"], help="Exit the Orbit REPL")
def handle_exit(args: str, session: Session) -> str:
    session.should_exit = True
    return "[exiting Orbit]"


@command(name="status", help="Show Orbit system status")
def handle_status(args: str, session: Session) -> str:
    chunk_count = 0
    if session.collection is not None and hasattr(session.collection, "count"):
        try:
            chunk_count = session.collection.count()
        except Exception:
            chunk_count = 0

    win_size = getattr(session.agent, "window_size", 4096)
    win_source = getattr(session.agent, "window_source", "fallback")
    used_t = estimate_tokens(session.history)

    lines = [
        "Orbit Status:",
        f"  Provider:         {session.config.provider}",
        f"  Model:            {session.config.model}",
        f"  Context Window:   {win_size} (source: {win_source})",
        f"  Token Usage:      ~{used_t}/{win_size} tokens",
        f"  Max Steps:        {session.max_steps}",
        f"  Indexed Chunks:   {chunk_count}",
        f"  History Messages: {len(session.history)}",
        f"  Show Sources:     {'on' if session.show_sources else 'off'}",
    ]
    if win_source == "fallback":
        lines.append("  [warning: Context window size using fallback (4096). Consider configuring context_windows in config.yaml]")

    return "\n".join(lines)


@command(name="model", help="Show or switch active LLM model")
def handle_model(args: str, session: Session) -> str:
    if not args:
        return f"Current model: {session.config.model}"

    new_model = args.strip()
    old_model = session.config.model
    old_agent = session.agent

    test_cfg = dataclasses.replace(session.config, model=new_model)
    try:
        new_agent = OrbitLLM.from_config(test_cfg)
        if not new_agent.is_available():
            return f"[error: Provider/Model '{session.config.provider}/{new_model}' is not available. Remaining on model '{old_model}'.]"

        session.config.model = new_model
        session.agent = new_agent
        return f"Model switched to '{new_model}'."
    except Exception as e:
        session.config.model = old_model
        session.agent = old_agent
        return f"[error: Could not switch to model '{new_model}': {e}. Remaining on model '{old_model}'.]"


@command(name="provider", help="Show or switch active LLM provider")
def handle_provider(args: str, session: Session) -> str:
    if not args:
        return f"Current provider: {session.config.provider}"

    new_provider = args.strip().lower()
    old_provider = session.config.provider
    old_agent = session.agent

    if new_provider not in OrbitLLM._PROVIDER_REGISTRY:
        avail = ", ".join(sorted(OrbitLLM._PROVIDER_REGISTRY.keys()))
        return f"[error: Unknown provider '{args}'. Available providers: {avail}]"

    test_cfg = dataclasses.replace(session.config, provider=new_provider)
    try:
        new_agent = OrbitLLM.from_config(test_cfg)
        if not new_agent.is_available():
            return f"[error: Provider '{new_provider}' is not available. Remaining on provider '{old_provider}'.]"

        session.config.provider = new_provider
        session.agent = new_agent
        return f"Provider switched to '{new_provider}'."
    except Exception as e:
        session.config.provider = old_provider
        session.agent = old_agent
        return f"[error: Could not switch to provider '{new_provider}': {e}. Remaining on provider '{old_provider}'.]"


@command(name="config", help="Show active configuration (API keys masked)")
def handle_config(args: str, session: Session) -> str:
    raw_dict = dataclasses.asdict(session.config)
    masked_dict = mask_config_data(raw_dict)

    lines = ["Active Orbit Configuration:"]
    lines.append(f"  provider: {masked_dict.get('provider')}")
    lines.append(f"  model: {masked_dict.get('model')}")
    lines.append(f"  generation: {masked_dict.get('generation')}")
    lines.append(f"  tools: {masked_dict.get('tools')}")
    lines.append("  providers:")
    prov_dict = masked_dict.get("providers", {})
    if not prov_dict:
        lines.append("    (none configured)")
    else:
        for p_name, p_opts in prov_dict.items():
            lines.append(f"    {p_name}: {p_opts}")

    return "\n".join(lines)


@command(name="tools", help="List registered tools and their risk levels")
def handle_tools(args: str, session: Session) -> str:
    tool_names = tool_registry.list_registered_tools()
    if not tool_names:
        return "No tools currently registered."

    lines = ["Registered Tools:"]
    for tname in sorted(tool_names):
        info = tool_registry.get_tool(tname)
        if not info:
            continue
        risk = info.get("risk_level", tool_registry.ToolRiskLevel.SAFE)
        risk_str = risk.value if hasattr(risk, "value") else str(risk)
        desc = info.get("schema", {}).get("description", "No description")
        lines.append(f"  - {tname} [{risk_str.upper()}]: {desc}")

    return "\n".join(lines)


@command(name="audit", help="Show recent audit log entries")
def handle_audit(args: str, session: Session) -> str:
    n = 10
    if args:
        try:
            n = int(args)
            if n <= 0:
                n = 10
        except ValueError:
            return "[error: audit log count must be a positive integer]"

    log_path = Path(tool_registry.AUDIT_LOG_PATH)
    if not log_path.exists():
        return f"[no audit log found at {tool_registry.AUDIT_LOG_PATH}]"

    try:
        with open(log_path, "r", encoding="utf-8") as f:
            all_lines = [line.strip() for line in f if line.strip()]

        if not all_lines:
            return f"[audit log at {tool_registry.AUDIT_LOG_PATH} is empty]"

        recent = all_lines[-n:]
        lines = [f"Last {len(recent)} audit log entries:"]
        lines.extend(recent)
        return "\n".join(lines)
    except Exception as e:
        return f"[error reading audit log: {e}]"


@command(name="steps", help="Show or set max_steps ceiling (1 to 20)")
def handle_steps(args: str, session: Session) -> str:
    if not args:
        return f"Current max_steps: {session.max_steps}"

    try:
        val = int(args)
        if 1 <= val <= 20:
            session.max_steps = val
            return f"max_steps updated to {val}."
        return "[error: max_steps must be an integer between 1 and 20.]"
    except ValueError:
        return "[error: max_steps must be an integer between 1 and 20.]"


@command(name="reindex", help="Re-index workspace target path for RAG context")
def handle_reindex(args: str, session: Session) -> str:
    target = args.strip() if args.strip() else session.target_path
    if not target:
        target = "./"

    path = Path(target)
    if not path.exists():
        return f"[error: path '{target}' does not exist.]"

    try:
        from rag.indexer import index_directory, index_file
        if session.collection is None:
            from rag.indexer import get_orbit_collection
            session.collection = get_orbit_collection()

        if path.is_dir():
            index_directory(target, collection=session.collection)
            session.target_path = target
            count = session.collection.count()
            return f"Re-indexed directory '{target}' into persistent vector store ({count} total chunks)."
        else:
            index_file(target, collection=session.collection)
            session.target_path = target
            count = session.collection.count()
            return f"Re-indexed file '{target}' into persistent vector store ({count} total chunks)."
    except Exception as e:
        return f"[error re-indexing target '{target}': {e}]"


@command(name="sources", help="Toggle printing retrieved context sources (on|off)")
def handle_sources(args: str, session: Session) -> str:
    if not args:
        return f"Show sources is currently {'on' if session.show_sources else 'off'}."

    arg = args.strip().lower()
    if arg in ("on", "1", "true", "yes"):
        session.show_sources = True
        return "Show sources enabled."
    elif arg in ("off", "0", "false", "no"):
        session.show_sources = False
        return "Show sources disabled."
    else:
        return "[error: usage /sources on|off]"


@command(name="context", help="Show history message count and estimated token size")
def handle_context(args: str, session: Session) -> str:
    msg_count = len(session.history)
    tokens = estimate_tokens(session.history)
    win_size = getattr(session.agent, "window_size", 4096)
    win_source = getattr(session.agent, "window_source", "fallback")

    res = f"History context: {msg_count} messages (~{tokens}/{win_size} tokens, window source: {win_source})."
    if win_source == "fallback":
        res += "\n[warning: Context window size using fallback (4096). Consider configuring context_windows in config.yaml]"
    return res


@command(name="compact", help="Compact conversation history")
def handle_compact(args: str, session: Session) -> str:
    win_size = getattr(session.agent, "window_size", 4096)
    old_t = estimate_tokens(session.history)
    focus = args.strip() if args.strip() else None

    compacted = compact_history(session.history, focus=focus, agent=session.agent, window=win_size)
    session.history.clear()
    session.history.extend(compacted)
    new_t = estimate_tokens(session.history)
    return f"Compaction completed: ~{old_t} tokens -> ~{new_t} tokens (window: {win_size})."
