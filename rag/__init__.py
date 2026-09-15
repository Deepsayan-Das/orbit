"""
Orbit RAG (Retrieval-Augmented Generation) package.
Provides AST-aware chunking, persistent ChromaDB indexing, and vector retrieval.
"""

from .chunker import (
    build_module_summary,
    chunk_file,
    chunk_fallback,
    chunk_prose,
    chunk_python_file,
    chunk_with_treesitter,
    extract_symbols,
)
from .indexer import (
    get_chroma_client,
    get_orbit_collection,
    index_directory,
    index_file,
    is_ignored_path,
)
from .retrieval import (
    build_context_prompt,
    correct_query_terms,
    embed,
    retrieve,
)

__all__ = [
    "chunk_python_file",
    "chunk_with_treesitter",
    "chunk_prose",
    "chunk_fallback",
    "chunk_file",
    "build_module_summary",
    "extract_symbols",
    "get_chroma_client",
    "get_orbit_collection",
    "index_file",
    "index_directory",
    "is_ignored_path",
    "embed",
    "correct_query_terms",
    "retrieve",
    "build_context_prompt",
]
