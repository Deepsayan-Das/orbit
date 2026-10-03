"""
Orbit REPL with persistent ChromaDB repo-context retrieval and Tool Registry integration.

Pipeline:
1. Index directory/file into persistent ChromaDB vector store (incremental via SHA-256).
2. Retrieve top-k context chunks via vector search & AST symbol typo correction.
3. Dynamically supply tool registry schemas to OrbitLLM.
4. Stream tokens to stdout.
"""

import os
import sys
from pathlib import Path

from config import load_config
from typing import Any, Optional


from agent_loop import run_agent_turn
from commands import Session, dispatch
from config import OrbitConfig, load_config
from llm_client import OrbitLLM
from providers import ChatMessage
from rag.indexer import get_orbit_collection, index_directory, index_file
from rag.retrieval import build_context_prompt, retrieve
import tools  # Ensures tool registration side-effects run (e.g. tools.filesystem)

SYSTEM_PROMPT = (
    "You are Orbit, a warm, enthusiastic, and joyful AI assistant for galactOS. "
    "Be friendly, helpful, and concise while maintaining a clean, professional tone. "
    "When a tool result is provided in the conversation, prioritize that tool output "
    "to directly answer the user's request."
)


EMBED_MODEL = "nomic-embed-text"


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


def stream_reply(agent: OrbitLLM, messages, system_prompt: str) -> str:
    """Streams a reply to stdout and returns the full text (for history)."""
    print("\nOrbit: ", end="", flush=True)
    full_response = ""
    for chunk in agent.chat(messages=messages, system_prompt=system_prompt, stream=True):
        sys.stdout.write(chunk.delta)
        sys.stdout.flush()
        full_response += chunk.delta
    print()
    return full_response


def repl(
    agent: OrbitLLM,
    collection,
    config: Optional[OrbitConfig] = None,
    target_path: str = "./",
):
    """Interactive REPL loop orchestrating context retrieval, tool integration, slash commands, and chat generation."""
    if config is None:
        config = load_config()

    session = Session(
        agent=agent,
        config=config,
        collection=collection,
        target_path=target_path,
        max_steps=8,
    )

    registered_tools = tools.get_tools_schema()
    tool_count = len(registered_tools)
    chunk_count = collection.count() if hasattr(collection, "count") else 0

    try:
        from ui import (
            print_banner,
            print_error,
            print_info,
            print_orbit_response,
            print_step_limit_warning,
            print_user_prompt_label,
            print_warning,
        )
        print_banner(
            chunk_count,
            tool_count,
            provider=getattr(session.config, "provider", ""),
            model=getattr(session.config, "model", ""),
        )
    except Exception:
        print(f"Orbit ready. Persistent collection indexed ({chunk_count} chunks, {tool_count} tools available). Type '/help' for commands.\n")
        print_info = print_warning = print_error = print

    while True:
        try:
            try:
                from ui import console
                user_input = console.input("[bold cyan]orbit > [/bold cyan]").strip()
            except Exception:
                user_input = input(">> ").strip()

            if not user_input:
                continue

            # Check if input is a slash command or a bare word command alias
            if user_input.startswith("/") or user_input.lower() in ("quit", "exit", "reset", "clear"):
                output = dispatch(user_input, session)
                if output:
                    try:
                        from ui import print_info
                        print_info(output)
                    except Exception:
                        print(output)
                if session.should_exit:
                    break
                continue

            # Retrieve fresh context for THIS turn only from persistent ChromaDB collection
            try:
                top_chunks = retrieve(user_input, collection=session.collection)
            except Exception:
                top_chunks = []
            if session.show_sources:
                try:
                    from ui import console
                    console.print(f"[dim][sources] Retrieved {len(top_chunks)} context chunks.[/dim]")
                    for i, chunk in enumerate(top_chunks, 1):
                        src = chunk.metadata.get("source", "unknown") if hasattr(chunk, "metadata") and isinstance(chunk.metadata, dict) else "unknown"
                        console.print(f"[dim]  Chunk {i}: {src}[/dim]")
                except Exception:
                    print(f"[sources: retrieved {len(top_chunks)} chunks]")

            grounded_prompt = build_context_prompt(user_input, top_chunks)

            # Send: clean history + this turn's grounded prompt
            current_turn_messages = session.history + [ChatMessage(role="user", content=grounded_prompt)]

            # ── Agentic loop: multi-step tool chaining ────────────────
            result = run_agent_turn(
                agent=session.agent,
                messages=current_turn_messages,
                system_prompt=SYSTEM_PROMPT,
                registered_tools=registered_tools,
                max_steps=session.max_steps,
                print_status=True,
                stream_final=True,
            )

            if result.hit_step_limit:
                try:
                    from ui import print_step_limit_warning
                    print_step_limit_warning(result.steps_taken)
                except Exception:
                    print(f"\nStep limit ({result.steps_taken}) reached. Type a message to continue this task, or /steps to raise the limit.")

            full_response = result.content
            if result.updated_messages:
                session.history = list(result.updated_messages)
            else:
                session.history.append(ChatMessage(role="user", content=user_input))
                session.history.append(ChatMessage(role="assistant", content=full_response))

        except KeyboardInterrupt:
            print("\n[interrupted, exiting]")
            break
        except EOFError:
            print("\n[EOF, exiting]")
            break
        except Exception as e:
            try:
                from ui import print_error
                print_error(f"Error: {e}")
            except Exception:
                print(f"\n[error: {e}]")
            break


if __name__ == "__main__":
    import argparse

    # Load persisted config as the base (provides defaults)
    cfg = load_config()

    parser = argparse.ArgumentParser(description="Orbit Interactive RAG REPL")
    parser.add_argument("target", nargs="?", default="./", help="Directory or file path to index (default: ./)")
    parser.add_argument("--provider", default=os.getenv("ORBIT_PROVIDER", cfg.provider), help="LLM Provider: ollama, huggingface, openai, gemini, groq")
    parser.add_argument("--model", default=os.getenv("ORBIT_MODEL", cfg.model), help="Model name (e.g. Qwen/Qwen2.5-Coder-7B-Instruct, llama3.2)")

    args = parser.parse_args()

    # CLI / env overrides take precedence over config file
    cfg.provider = args.provider.lower()
    cfg.model = args.model

    target = args.target
    path = Path(target)

    collection = get_orbit_collection()

    print(f"[Orbit] Indexing target path '{target}' into persistent vector store...")
    try:
        if path.is_dir():
            index_directory(target, collection=collection)
        else:
            index_file(target, collection=collection)
    except Exception as e:
        print(f"  [note: RAG indexing skipped ({e}). LLM engine active.]")

    print(f"[Orbit] Initializing LLM client (provider='{cfg.provider}', model='{cfg.model}')...")
    agent = OrbitLLM.from_config(cfg)
    repl(agent, collection, config=cfg, target_path=target)
