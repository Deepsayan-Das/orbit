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
from repl import PROVIDER, SYSTEM_PROMPT, repl, stream_reply

if __name__ == "__main__":
    import os
    import sys
    from pathlib import Path
    import argparse
    from llm_client import OrbitLLM
    from rag.indexer import get_orbit_collection

    parser = argparse.ArgumentParser(description="Orbit Interactive RAG REPL")
    parser.add_argument("target", nargs="?", default="./", help="Directory or file path to index (default: ./)")
    parser.add_argument("--provider", default=os.getenv("ORBIT_PROVIDER", PROVIDER), help="LLM Provider: ollama, huggingface, openai, gemini")
    parser.add_argument("--model", default=os.getenv("ORBIT_MODEL", "llama3.2:latest"), help="Model name")

    args = parser.parse_args()

    provider_name = args.provider.lower()
    model_name = args.model
    target = args.target

    collection = get_orbit_collection()
    path = Path(target)

    print(f"[Orbit] Indexing target path '{target}' into persistent vector store...")
    if path.is_dir():
        index_directory(target, collection=collection)
    else:
        index_file(target, collection=collection)

    print(f"[Orbit] Initializing LLM client (provider='{provider_name}', model='{model_name}')...")
    agent = OrbitLLM(provider=provider_name, model=model_name)
    repl(agent, collection)