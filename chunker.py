"""
Orbit Chunker compatibility module. Re-exports everything from rag.chunker.
"""

from rag.chunker import (
    CLASS_NODE_TYPES,
    CODE_EXTENSIONS,
    FUNCTION_NODE_TYPES,
    PROSE_EXTENSIONS,
    TREE_SITTER_LANGUAGES,
    build_module_summary,
    chunk_fallback,
    chunk_file,
    chunk_prose,
    chunk_python_file,
    chunk_with_treesitter,
    extract_symbols,
)

__all__ = [
    "CLASS_NODE_TYPES",
    "CODE_EXTENSIONS",
    "FUNCTION_NODE_TYPES",
    "PROSE_EXTENSIONS",
    "TREE_SITTER_LANGUAGES",
    "build_module_summary",
    "chunk_fallback",
    "chunk_file",
    "chunk_prose",
    "chunk_python_file",
    "chunk_with_treesitter",
    "extract_symbols",
]
