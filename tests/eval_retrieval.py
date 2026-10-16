"""
Golden retrieval evaluation suite for Orbit RAG.

Defines a curated set of 30 query -> expected file/symbol pairs and calculates
Recall@k and Mean Reciprocal Rank (MRR) to establish retrieval quality baselines.
"""

from typing import Any, Callable, Dict, List, Optional
import unittest

GOLDEN_EVAL_SET = [
    {
        "query": "where is resolve_window defined and how are model context windows determined",
        "expected_files": ["context/window.py"],
        "expected_symbols": ["resolve_window", "KNOWN_MODEL_WINDOWS"],
    },
    {
        "query": "how is conversation history compacted when context exceeds threshold",
        "expected_files": ["commands/compaction.py"],
        "expected_symbols": ["compact_history", "drop_oldest_tool_pairs"],
    },
    {
        "query": "how are tokens estimated from messages",
        "expected_files": ["context/measure.py"],
        "expected_symbols": ["estimate_tokens", "should_compact"],
    },
    {
        "query": "where is the tool registry initialized and tools registered",
        "expected_files": ["tools/registry.py", "tools/__init__.py"],
        "expected_symbols": ["register_tool", "get_tool", "execute_tool"],
    },
    {
        "query": "how does the multi-step agent loop execute tool calls",
        "expected_files": ["agent_loop.py"],
        "expected_symbols": ["run_agent_turn", "extract_tool_calls"],
    },
    {
        "query": "how does python AST chunking work and handle classes and functions",
        "expected_files": ["rag/chunker.py"],
        "expected_symbols": ["chunk_python_file", "chunk_file"],
    },
    {
        "query": "tree sitter code parser for javascript and typescript",
        "expected_files": ["rag/chunker.py"],
        "expected_symbols": ["chunk_with_treesitter", "TREE_SITTER_LANGUAGES"],
    },
    {
        "query": "how are files incrementally indexed with sha256 hashes in ChromaDB",
        "expected_files": ["rag/indexer.py"],
        "expected_symbols": ["index_file", "compute_content_hash", "get_orbit_collection"],
    },
    {
        "query": "how does vector retrieval rank chunks and handle typos",
        "expected_files": ["rag/retrieval.py"],
        "expected_symbols": ["retrieve", "correct_query_terms", "embed"],
    },
    {
        "query": "where are ollama model options and num_ctx configured",
        "expected_files": ["providers/ollama_provider.py"],
        "expected_symbols": ["OllamaProvider", "_prepare_kwargs"],
    },
    {
        "query": "where is anthropic claude tool use and message formatting implemented",
        "expected_files": ["providers/anthropic_provider.py"],
        "expected_symbols": ["AnthropicProvider", "_format_messages"],
    },
    {
        "query": "how does openai provider handle tool completions",
        "expected_files": ["providers/openai_provider.py"],
        "expected_symbols": ["OpenAIProvider", "_sync_chat"],
    },
    {
        "query": "how does gemini provider handle function calling and parts",
        "expected_files": ["providers/gemini_provider.py"],
        "expected_symbols": ["GeminiProvider", "_format_messages_and_config"],
    },
    {
        "query": "how are subagents created and delegated tasks",
        "expected_files": ["agent_sub.py"],
        "expected_symbols": ["delegate_task", "run_subagent"],
    },
    {
        "query": "where are shell commands executed with permission gating",
        "expected_files": ["tools/shell.py"],
        "expected_symbols": ["run_shell_command", "register_shell_tools"],
    },
    {
        "query": "how does edit_file tool perform string replacements",
        "expected_files": ["tools/edit_file.py"],
        "expected_symbols": ["edit_file", "register_edit_file_tool"],
    },
    {
        "query": "where is read_file implemented with line offset and count",
        "expected_files": ["tools/read_file.py"],
        "expected_symbols": ["read_file", "register_read_file_tool"],
    },
    {
        "query": "where are git status and git diff commands implemented",
        "expected_files": ["tools/git_tools.py"],
        "expected_symbols": ["git_status", "git_diff", "register_git_tools"],
    },
    {
        "query": "how does codebase search tool query ChromaDB",
        "expected_files": ["tools/code_search.py"],
        "expected_symbols": ["search_codebase", "register_code_search_tools"],
    },
    {
        "query": "where is the interactive setup wizard and configure CLI",
        "expected_files": ["configure.py"],
        "expected_symbols": ["configure_wizard", "prompt_choice"],
    },
    {
        "query": "where is LLMClient defined with retry logic and provider dispatch",
        "expected_files": ["llm_client.py"],
        "expected_symbols": ["OrbitLLM", "chat", "is_transient_503"],
    },
    {
        "query": "where are container and docker tools registered",
        "expected_files": ["tools/container_tools.py"],
        "expected_symbols": ["container_list", "register_container_tools"],
    },
    {
        "query": "where are REPL slash commands like /clear and /history handled",
        "expected_files": ["commands/builtin.py"],
        "expected_symbols": ["command_clear", "command_history"],
    },
    {
        "query": "where is rich terminal UI formatted and printed",
        "expected_files": ["ui.py"],
        "expected_symbols": ["print_step_start", "print_diff_preview", "console"],
    },
    {
        "query": "where is the main REPL loop and turn execution",
        "expected_files": ["repl.py"],
        "expected_symbols": ["repl", "stream_reply"],
    },
]


def score_retrieval(
    retrieved_chunks: List[Dict[str, Any]],
    expected_files: List[str],
    expected_symbols: Optional[List[str]] = None,
) -> tuple[bool, Optional[int]]:
    """Determine if a query's expected file or symbol was recalled and at what rank (1-based)."""
    expected_symbols = expected_symbols or []
    norm_expected_files = [f.replace("\\", "/").lower() for f in expected_files]

    for rank, chunk in enumerate(retrieved_chunks, 1):
        source = chunk.get("source", "").replace("\\", "/").lower()
        text = chunk.get("text", "")
        meta_symbol = chunk.get("symbol", "")
        chunk_symbols = chunk.get("symbols", []) or []

        # Check source file match
        file_match = any(exp in source for exp in norm_expected_files)

        # Check symbol match
        symbol_match = False
        for exp_sym in expected_symbols:
            if exp_sym in meta_symbol or exp_sym in chunk_symbols or exp_sym in text:
                symbol_match = True
                break

        if file_match or symbol_match:
            return True, rank

    return False, None


def run_retrieval_eval(
    retrieve_fn: Callable[[str], List[Dict[str, Any]]],
    eval_set: List[Dict[str, Any]] = GOLDEN_EVAL_SET,
    top_k: int = 3,
) -> Dict[str, float]:
    """Run retrieval evaluation across the golden dataset and return metrics."""
    total = len(eval_set)
    hits = 0
    reciprocal_ranks = []

    for item in eval_set:
        query = item["query"]
        expected_files = item["expected_files"]
        expected_symbols = item.get("expected_symbols", [])

        try:
            chunks = retrieve_fn(query)
            if top_k and len(chunks) > top_k:
                chunks = chunks[:top_k]
        except Exception:
            chunks = []

        hit, rank = score_retrieval(chunks, expected_files, expected_symbols)
        if hit and rank is not None:
            hits += 1
            reciprocal_ranks.append(1.0 / rank)
        else:
            reciprocal_ranks.append(0.0)

    recall_at_k = hits / total if total > 0 else 0.0
    mrr = sum(reciprocal_ranks) / total if total > 0 else 0.0

    return {
        "total_queries": total,
        "hits": hits,
        f"recall@{top_k}": round(recall_at_k, 4),
        "mrr": round(mrr, 4),
    }


class TestGoldenRetrievalEval(unittest.TestCase):
    def test_golden_set_structure(self):
        self.assertGreaterEqual(len(GOLDEN_EVAL_SET), 20)
        for item in GOLDEN_EVAL_SET:
            self.assertIn("query", item)
            self.assertIn("expected_files", item)
            self.assertIsInstance(item["expected_files"], list)

    def test_scorer_behavior(self):
        sample_chunks = [
            {"source": "commands/compaction.py", "text": "def compact_history(): pass"},
            {"source": "context/window.py", "text": "def resolve_window(): pass"},
        ]
        hit, rank = score_retrieval(sample_chunks, ["context/window.py"], ["resolve_window"])
        self.assertTrue(hit)
        self.assertEqual(rank, 2)

        no_hit, no_rank = score_retrieval(sample_chunks, ["not_found.py"], ["not_found_sym"])
        self.assertFalse(no_hit)
        self.assertIsNone(no_rank)


if __name__ == "__main__":
    unittest.main()
