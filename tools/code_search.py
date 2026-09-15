"""
Codebase Search tool for Orbit.

Exposes Orbit's persistent ChromaDB RAG vector retrieval as a callable tool,
allowing the LLM runtime to search codebase context on demand.
"""

from typing import Any, Dict

from rag.indexer import get_orbit_collection
from rag.retrieval import retrieve
from .registry import ToolRiskLevel, register_tool

SEARCH_CODEBASE_SCHEMA: Dict[str, Any] = {
    "name": "search_codebase",
    "description": "Perform semantic vector search across the indexed repository code and documentation.",
    "parameters": {
        "type": "object",
        "properties": {
            "query": {
                "type": "string",
                "description": "The search query or concept to look up in the codebase."
            }
        },
        "required": ["query"]
    }
}


def search_codebase(query: str, top_k: int = 3) -> str:
    """Perform semantic vector search across the indexed codebase chunks."""
    try:
        collection = get_orbit_collection()
        top_chunks = retrieve(query, collection=collection, top_k=top_k)

        if not top_chunks:
            return f"No codebase chunks found matching query '{query}'."

        formatted_snippets = []
        for i, c in enumerate(top_chunks, 1):
            score_fmt = f"{c.get('score', 0.0):.2f}" if c.get('score') is not None else "N/A"
            snippet = f"--- Match {i} (from {c['source']}, kind: {c.get('kind', 'code')}, score: {score_fmt}) ---\n{c['text']}"
            formatted_snippets.append(snippet)

        return f"Codebase Search Results for '{query}':\n\n" + "\n\n".join(formatted_snippets)
    except Exception as e:
        return f"Error executing codebase search: {str(e)}"


def register_code_search_tools():
    """Register codebase search tools into global tool registry."""
    register_tool("search_codebase", search_codebase, SEARCH_CODEBASE_SCHEMA, risk_level=ToolRiskLevel.SAFE)


register_code_search_tools()
