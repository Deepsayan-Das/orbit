"""
Orbit REPL with repo-context retrieval (raw RAG, no framework).

Pipeline: chunk (AST-aware) -> embed -> store in-memory -> retrieve top-k by
cosine similarity -> inject into prompt -> generate.

This is deliberately "raw" (a Python list + linear scan), which is fine at
the scale of a single project's source files. See project notes for why
this doesn't scale to millions of chunks and where a real vector DB would
take over.
"""

import ast
import os
import sys
from pathlib import Path

import numpy as np
import ollama

from llm_client import OrbitLLM
from providers import ChatMessage

SYSTEM_PROMPT = (
    "You are Orbit, a warm, enthusiastic, and joyful AI assistant for galactOS. "
    "Be friendly, helpful, and concise while maintaining a clean, professional tone. "
    "When given repository context, ground your answer in it and say so if the "
    "context doesn't actually contain the answer."
)

EMBED_MODEL = "nomic-embed-text"
PROVIDER = "ollama"  # Options: "huggingface", "ollama", "openai", "gemini", "groq"
CHAT_MODEL = "llama3.2:latest"  # e.g. "Qwen/Qwen2.5-Coder-7B-Instruct", "llama3.2:latest"


# --------------------------------------------------------------------------
# Chunking
# --------------------------------------------------------------------------

def chunk_python_file(source: str, max_chunk_size: int = 1500) -> list[str]:
    """Split a Python source file into complete, semantically whole units
    (functions, classes). Classes larger than max_chunk_size are split
    per-method, with the class name/docstring prepended so each method
    chunk stays self-contained."""
    tree = ast.parse(source)
    chunks: list[str] = []

    for node in ast.iter_child_nodes(tree):
        if not isinstance(node, (ast.FunctionDef, ast.ClassDef, ast.AsyncFunctionDef)):
            continue

        segment = ast.get_source_segment(source, node)
        if not segment:
            continue

        if len(segment) <= max_chunk_size or not isinstance(node, ast.ClassDef):
            chunks.append(segment)
            continue

        # Class too big — split per-method, but keep class context attached
        # so a retrieved method chunk still says which class it belongs to.
        class_header = f"class {node.name}:\n"
        docstring = ast.get_docstring(node)
        if docstring:
            class_header += f'    """{docstring}"""\n'

        for sub_node in node.body:
            if isinstance(sub_node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                method_source = ast.get_source_segment(source, sub_node)
                if method_source:
                    chunks.append(f"{class_header}\n{method_source}")

    return chunks


# --------------------------------------------------------------------------
# Embedding + indexing
# --------------------------------------------------------------------------

def embed(text: str) -> list[float]:
    response = ollama.embeddings(model=EMBED_MODEL, prompt=text)
    return response["embedding"]


from chunker import build_module_summary, chunk_file

NOISE_DIRS = {
    ".git", ".github", "node_modules", "__pycache__", "venv", ".venv",
    "env", ".orbit", ".idea", ".vscode", "dist", "build", "site-packages"
}
IGNORED_FILES = {
    ".gitignore", ".gitattributes", ".ds_store", "package-lock.json",
    "yarn.lock", "pnpm-lock.yaml", "pipfile.lock", "poetry.lock"
}
SKIP_EXTENSIONS = {
    ".pyc", ".png", ".jpg", ".jpeg", ".gif", ".exe", ".dll", ".so",
    ".dylib", ".zip", ".tar", ".gz", ".lock", ".bin", ".pdf"
}


def is_ignored_path(path: Path) -> bool:
    """Check if a file or directory path should be skipped during indexing."""
    # Check parent directory names
    for part in path.parts[:-1]:
        if part in NOISE_DIRS or (part.startswith(".") and part not in {".", ".."}):
            return True

    # Check file name and extension
    filename = path.name.lower()
    if filename.startswith(".env") or filename in IGNORED_FILES:
        return True
    if path.suffix.lower() in SKIP_EXTENSIONS:
        return True

    return False


def index_file(filepath: str) -> list[dict]:
    """Chunk + embed a single file using chunk_file dispatcher, including a module summary record."""
    try:
        source = Path(filepath).read_text(encoding="utf-8", errors="replace")
    except Exception as e:
        print(f"[warn] could not read {filepath}: {e}")
        return []

    records: list[dict] = []

    # 1. Module-level summary record
    summary_text = build_module_summary(filepath, source)
    if summary_text:
        records.append({
            "source": filepath,
            "text": summary_text,
            "embedding": embed(summary_text),
            "level": "module",
            "kind": "summary"
        })

    # 2. Fine-grained chunk records
    file_chunks = chunk_file(filepath, source)
    for c in file_chunks:
        records.append({
            "source": filepath,
            "text": c["text"],
            "embedding": embed(c["text"]),
            "level": "chunk",
            "kind": c["kind"]
        })

    return records


def index_directory(dirpath: str) -> list[dict]:
    """Chunk + embed every code/prose file under a directory, skipping noise directories."""
    records: list[dict] = []
    base_path = Path(dirpath)

    for path in base_path.rglob("*"):
        if path.is_file() and not is_ignored_path(path):
            try:
                records.extend(index_file(str(path)))
            except Exception as e:
                print(f"[warn] skipping {path}: {e}")

    return records


# --------------------------------------------------------------------------
# Retrieval
# --------------------------------------------------------------------------

def cosine_similarity(a: list[float], b: list[float]) -> float:
    a_arr, b_arr = np.array(a), np.array(b)
    return float(np.dot(a_arr, b_arr) / (np.linalg.norm(a_arr) * np.linalg.norm(b_arr)))


def retrieve(query: str, records: list[dict], top_k: int = 3) -> list[dict]:
    query_vec = embed(query)
    scored = [
        {**r, "score": cosine_similarity(query_vec, r["embedding"])}
        for r in records
    ]
    scored.sort(key=lambda r: r["score"], reverse=True)
    return scored[:top_k]


def build_context_prompt(query: str, top_chunks: list[dict]) -> str:
    context = "\n\n---\n\n".join(
        f"# from {c['source']}\n{c['text']}" for c in top_chunks
    )
    return (
        f"Repository context:\n\n{context}\n\n"
        f"Question: {query}\n\n"
        "Answer using the context above. If the context doesn't contain "
        "the answer, say so plainly rather than guessing."
    )


# --------------------------------------------------------------------------
# REPL
# --------------------------------------------------------------------------

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


def repl(agent: OrbitLLM, records: list[dict]):
    history: list[ChatMessage] = []
    print(f"Orbit ready. Indexed {len(records)} chunks. Type 'quit' to exit.\n")

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

            # Retrieve fresh context for THIS turn only.
            top_chunks = retrieve(user_input, records)
            grounded_prompt = build_context_prompt(user_input, top_chunks)

            # Send: clean history + this turn's grounded prompt — but only
            # store the CLEAN question in history, not the injected version.
            outgoing = history + [ChatMessage(role="user", content=grounded_prompt)]
            full_response = stream_reply(agent, outgoing, SYSTEM_PROMPT)

            history.append(ChatMessage(role="user", content=user_input))       # clean
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

    if path.is_dir():
        records = index_directory(target)
    else:
        records = index_file(target)

    print(f"[Orbit] Initializing LLM client (provider='{provider_name}', model='{model_name}')...")
    agent = OrbitLLM(provider=provider_name, model=model_name)
    repl(agent, records)