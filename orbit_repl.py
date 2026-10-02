"""
Orbit REPL backwards compatibility entrypoint. Re-exports RAG functions and delegates REPL execution to repl.py.
"""

from rag.chunker import chunk_python_file
from rag.indexer import index_directory, index_file, is_ignored_path
from rag.retrieval import (
    EMBED_MODEL,
    build_context_prompt,
    correct_query_terms,
    cosine_similarity,
    embed,
    retrieve,
)
from repl import SYSTEM_PROMPT, repl, stream_reply

if __name__ == "__main__":
    import os
    import sys
    from pathlib import Path
    import argparse
    from config import load_config
    from llm_client import OrbitLLM
    from rag.indexer import get_orbit_collection

    cfg = load_config()

    parser = argparse.ArgumentParser(description="Orbit Interactive RAG REPL")
    parser.add_argument("target", nargs="?", default="./", help="Directory or file path to index (default: ./)")
    parser.add_argument("--provider", default=os.getenv("ORBIT_PROVIDER", cfg.provider), help="LLM Provider: ollama, huggingface, openai, gemini, groq")
    parser.add_argument("--model", default=os.getenv("ORBIT_MODEL", cfg.model), help="Model name")

    args = parser.parse_args()

    cfg.provider = args.provider.lower()
    cfg.model = args.model
    target = args.target

    collection = get_orbit_collection()
    path = Path(target)

    print(f"[Orbit] Indexing target path '{target}' into persistent vector store...")
    if path.is_dir():
        index_directory(target, collection=collection)
    else:
        index_file(target, collection=collection)

    print(f"[Orbit] Initializing LLM client (provider='{cfg.provider}', model='{cfg.model}')...")
    agent = OrbitLLM.from_config(cfg)
    repl(agent, collection)