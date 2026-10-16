"""
Unit tests for Phase 4: Retrieval Upgrades.
1. BM25 keyword search & code identifier tokenization (snake_case, camelCase).
2. Reciprocal Rank Fusion (RRF) rank aggregation.
3. Relevance floor (min_score threshold) and per-file deduplication (max_per_file).
4. AST symbol cache persistence and retrieval without full collection scan.
5. Hybrid retrieval integration (Vector + BM25).
"""

import json
import os
import shutil
import tempfile
import unittest
from unittest.mock import MagicMock, patch

from rag.bm25 import (
    BM25,
    filter_and_deduplicate,
    reciprocal_rank_fusion,
    tokenize_code,
)
from rag.indexer import (
    DEFAULT_SYMBOL_CACHE_PATH,
    get_chroma_client,
    get_orbit_collection,
    load_symbol_cache,
    save_symbol_cache,
)
from rag.retrieval import (
    _KNOWN_SYMBOLS_CACHE,
    get_known_symbols,
    retrieve,
)


class TestPhase4RetrievalUpgrades(unittest.TestCase):

    def setUp(self):
        self.test_dir = tempfile.mkdtemp()
        self.db_path = os.path.join(self.test_dir, "chroma_db")
        self.client = get_chroma_client(db_path=self.db_path)

    def tearDown(self):
        shutil.rmtree(self.test_dir, ignore_errors=True)

    def test_tokenize_code_splits_identifiers(self):
        code = "def resolve_model_context_window(maxTokensLimit):"
        tokens = tokenize_code(code)
        self.assertIn("resolve_model_context_window", tokens)
        self.assertIn("resolve", tokens)
        self.assertIn("window", tokens)
        self.assertIn("context", tokens)
        self.assertIn("maxtokenslimit", tokens)
        self.assertIn("tokens", tokens)

    def test_bm25_exact_symbol_ranking(self):
        docs = [
            {"id": "doc1", "source": "a.py", "symbol": "foo_bar", "text": "def foo_bar(): pass"},
            {"id": "doc2", "source": "b.py", "symbol": "resolve_window", "text": "def resolve_window(): return 8192"},
            {"id": "doc3", "source": "c.py", "symbol": "other_func", "text": "def other_func(): pass"},
        ]
        bm25 = BM25().fit(docs)
        results = bm25.search("resolve_window", top_k=2)
        self.assertGreater(len(results), 0)
        top_doc, score = results[0]
        self.assertEqual(top_doc["symbol"], "resolve_window")
        self.assertGreater(score, 0.0)

    def test_reciprocal_rank_fusion(self):
        vector_results = [
            {"id": "doc_a", "source": "a.py", "symbol": "sym_a", "text": "aaa"},
            {"id": "doc_b", "source": "b.py", "symbol": "sym_b", "text": "bbb"},
        ]
        bm25_results = [
            {"id": "doc_b", "source": "b.py", "symbol": "sym_b", "text": "bbb"},
            {"id": "doc_c", "source": "c.py", "symbol": "sym_c", "text": "ccc"},
        ]
        fused = reciprocal_rank_fusion(vector_results, bm25_results, k=60, top_k=3)
        self.assertEqual(len(fused), 3)
        # doc_b appeared in both vector (#2) and BM25 (#1), so its RRF score must be highest
        self.assertEqual(fused[0]["id"], "doc_b")
        self.assertIn("rrf_score", fused[0])

    def test_relevance_floor_and_deduplication(self):
        chunks = [
            {"id": "1", "source": "file1.py", "score": 0.85, "text": "chunk 1"},
            {"id": "2", "source": "file1.py", "score": 0.80, "text": "chunk 2"},
            {"id": "3", "source": "file1.py", "score": 0.75, "text": "chunk 3"},
            {"id": "4", "source": "file2.py", "score": 0.70, "text": "chunk 4"},
            {"id": "5", "source": "file3.py", "score": 0.20, "text": "chunk 5"},  # Below floor (0.35)
        ]

        # max_per_file = 2, min_score = 0.35
        filtered = filter_and_deduplicate(chunks, min_score=0.35, max_per_file=2, top_k=5)
        sources = [c["source"] for c in filtered]
        ids = [c["id"] for c in filtered]

        # file1.py should only appear 2 times (chunk 3 dropped by deduplication)
        self.assertEqual(sources.count("file1.py"), 2)
        self.assertIn("4", ids)
        # chunk 5 (score 0.20) should be dropped by relevance floor
        self.assertNotIn("5", ids)

    def test_symbol_cache_persistence_and_lookup(self):
        cache_path = os.path.join(self.test_dir, "test_symbols.json")
        test_symbols = {"resolve_window", "compact_history", "execute_tool"}
        save_symbol_cache(test_symbols, cache_path)

        loaded = load_symbol_cache(cache_path)
        self.assertEqual(loaded, sorted(list(test_symbols)))

        # get_known_symbols loads from cache_path without calling Chroma collection
        mock_col = MagicMock()
        retrieved_symbols = get_known_symbols(collection=mock_col, cache_path=cache_path)
        self.assertIn("resolve_window", retrieved_symbols)
        mock_col.get.assert_not_called()

    def test_hybrid_retrieve_with_records(self):
        records = [
            {
                "id": "r1",
                "source": "context/window.py",
                "symbol": "resolve_window",
                "text": "def resolve_window(): return 128000",
                "embedding": [0.9, 0.1, 0.0],
            },
            {
                "id": "r2",
                "source": "commands/compaction.py",
                "symbol": "compact_history",
                "text": "def compact_history(): pass",
                "embedding": [0.1, 0.9, 0.0],
            },
        ]

        # Search for exact symbol "resolve_window"
        with patch("rag.retrieval.embed_query", return_value=[0.8, 0.2, 0.0]):
            res = retrieve("where is resolve_window defined", records=records, top_k=2, hybrid=True)
            self.assertGreater(len(res), 0)
            self.assertEqual(res[0]["source"], "context/window.py")


if __name__ == "__main__":
    unittest.main()
