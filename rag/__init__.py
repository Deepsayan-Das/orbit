"""
Orbit RAG (Retrieval-Augmented Generation) package.
Provides AST-aware chunking, persistent ChromaDB indexing, and vector retrieval.
"""

from .bm25 import (
    BM25,
    filter_and_deduplicate,
    reciprocal_rank_fusion,
    tokenize_code,
)
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
    DEFAULT_SYMBOL_CACHE_PATH,
    get_chroma_client,
    get_orbit_collection,
    get_repo_collection_name,
    index_directory,
    index_file,
    is_ignored_path,
    load_symbol_cache,
    save_symbol_cache,
)
from .retrieval import (
    build_context_prompt,
    correct_query_terms,
    embed,
    embed_batch,
    embed_document,
    embed_documents,
    embed_query,
    get_known_symbols,
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
    "get_repo_collection_name",
    "index_file",
    "index_directory",
    "is_ignored_path",
    "save_symbol_cache",
    "load_symbol_cache",
    "DEFAULT_SYMBOL_CACHE_PATH",
    "embed",
    "embed_batch",
    "embed_document",
    "embed_documents",
    "embed_query",
    "correct_query_terms",
    "get_known_symbols",
    "retrieve",
    "build_context_prompt",
    "BM25",
    "tokenize_code",
    "reciprocal_rank_fusion",
    "filter_and_deduplicate",
]
