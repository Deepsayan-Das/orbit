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
from typing import Any, Optional


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
PROVIDER = "ollama"  # Options: "huggingface", "ollama", "openai", "gemini", "groq"
CHAT_MODEL = "llama3.2:latest"  # e.g. "Qwen/Qwen2.5-Coder-7B-Instruct", "llama3.2:latest"


def extract_tool_calls(raw_response: Any) -> list:
    """Extract tool calls list from raw provider response object or dict."""
    if not raw_response:
        return []
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


def repl(agent: OrbitLLM, collection):
    """Interactive REPL loop orchestrating context retrieval, tool integration, and chat generation."""
    history: list[ChatMessage] = []
    registered_tools = tools.get_tools_schema()
    tool_count = len(registered_tools)
    chunk_count = collection.count()

    print(f"Orbit ready. Persistent collection indexed ({chunk_count} chunks, {tool_count} tools available). Type 'quit' to exit.\n")

    while True:
        try:
            user_input = input(">> ").strip()
            if user_input in ("quit", "exit"):
                break
            if user_input in ("reset", "clear"):
                history = []
                print("[history cleared]")
                continue
            if not user_input:
                continue

            # Retrieve fresh context for THIS turn only from persistent ChromaDB collection
            top_chunks = retrieve(user_input, collection=collection)
            grounded_prompt = build_context_prompt(user_input, top_chunks)

            # Send: clean history + this turn's grounded prompt
            outgoing = history + [ChatMessage(role="user", content=grounded_prompt)]

            # Step 1: Decision call with tools attached (non-streamed)
            check_resp = agent.chat(
                messages=outgoing,
                system_prompt=SYSTEM_PROMPT,
                stream=False,
                tools=registered_tools
            )

            tool_calls = extract_tool_calls(check_resp.raw_response)

            if tool_calls:
                tool_executed = False
                for call in tool_calls:
                    fn_name, fn_args = parse_tool_call(call)
                    if fn_name and tools.get_tool(fn_name):
                        tool_result = tools.execute_tool(fn_name, fn_args)
                        print(f"\n[tool executed] {fn_name}({fn_args}) -> {str(tool_result)[:100]}...")

                        followup_messages = history + [
                            ChatMessage(role="user", content=grounded_prompt),
                            ChatMessage(role="assistant", content="", tool_calls=tool_calls),
                            ChatMessage(role="tool", content=str(tool_result), name=fn_name)
                        ]

                        full_response = stream_reply(agent, followup_messages, SYSTEM_PROMPT)
                        history.append(ChatMessage(role="user", content=user_input))
                        history.append(ChatMessage(role="assistant", content=full_response))
                        tool_executed = True
                        break


                if not tool_executed:
                    full_response = stream_reply(agent, outgoing, SYSTEM_PROMPT)
                    history.append(ChatMessage(role="user", content=user_input))
                    history.append(ChatMessage(role="assistant", content=full_response))

            else:
                # Step 2: No tool call requested — stream typing-effect output without tools attached
                full_response = stream_reply(agent, outgoing, SYSTEM_PROMPT)
                history.append(ChatMessage(role="user", content=user_input))
                history.append(ChatMessage(role="assistant", content=full_response))

        except KeyboardInterrupt:
            print("\n[interrupted, exiting]")
            break
        except EOFError:
            print("\n[EOF, exiting]")
            break
        except Exception as e:
            print(f"\n[error: {e}]")
            break



if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Orbit Interactive RAG REPL")
    parser.add_argument("target", nargs="?", default="./", help="Directory or file path to index (default: ./)")
    parser.add_argument("--provider", default=os.getenv("ORBIT_PROVIDER", PROVIDER), help="LLM Provider: ollama, huggingface, openai, gemini")
    parser.add_argument("--model", default=os.getenv("ORBIT_MODEL", CHAT_MODEL), help="Model name (e.g. Qwen/Qwen2.5-Coder-7B-Instruct, llama3.2)")

    args = parser.parse_args()

    provider_name = args.provider.lower()
    model_name = args.model

    target = args.target
    path = Path(target)

    collection = get_orbit_collection()

    print(f"[Orbit] Indexing target path '{target}' into persistent vector store...")
    if path.is_dir():
        index_directory(target, collection=collection)
    else:
        index_file(target, collection=collection)

    print(f"[Orbit] Initializing LLM client (provider='{provider_name}', model='{model_name}')...")
    agent = OrbitLLM(provider=provider_name, model=model_name)
    repl(agent, collection)
