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
    "You are Orbit, an intelligent AI software engineering assistant for the galactOS platform. "
    "Be direct, helpful, and concise while maintaining a clean, professional tone. "
    "Leverage available tools to inspect repository files, search codebase context, run tests, and execute commands. "
    "When a tool result is provided in the conversation, prioritize that tool output to directly answer the user's request."
)


EMBED_MODEL = "nomic-embed-text"


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

    enabled_tools = config.tools.get("enabled", []) if isinstance(config.tools, dict) else []
    registered_tools = tools.get_tools_schema(enabled_only=enabled_tools)
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
                        src = chunk.get("source", "unknown") if isinstance(chunk, dict) else getattr(chunk, "source", "unknown")
                        console.print(f"[dim]  Chunk {i}: {src}[/dim]")
                except Exception:
                    print(f"[sources: retrieved {len(top_chunks)} chunks]")

            # Ephemeral repository context injection into turn system prompt (leaves chat history clean)
            turn_system_prompt = SYSTEM_PROMPT
            if top_chunks:
                context_block = "\n\n---\n\n".join(
                    f"# from {c.get('source', 'unknown')}\n{c.get('text', '')}"
                    for c in top_chunks
                )
                turn_system_prompt = (
                    f"{SYSTEM_PROMPT}\n\n"
                    f"Repository context for this request:\n\n{context_block}\n\n"
                    "Answer using this repository context when relevant. If the context does not contain the answer, rely on tools or general knowledge rather than guessing."
                )

            # Send: clean history + this turn's raw user query
            current_turn_messages = session.history + [ChatMessage(role="user", content=user_input)]

            # ── Agentic loop: multi-step tool chaining ────────────────
            result = run_agent_turn(
                agent=session.agent,
                messages=current_turn_messages,
                system_prompt=turn_system_prompt,
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

    collection = get_orbit_collection(repo_path=target)

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
