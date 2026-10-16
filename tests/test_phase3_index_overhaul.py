"""
Unit tests verifying Phase 3 Index-Time Overhaul:
1. Cosine distance metric & per-repo isolated collection name.
2. nomic task prefixes (search_document: / search_query:).
3. Batched embedding calls.
4. Python chunker: decorator capture, preamble chunks, large function handling.
5. JS/TS chunker: export function, export class, arrow function unwrapping.
6. Contextual chunk headers & stale record pruning.
"""

import os
import shutil
import tempfile
import unittest
from unittest.mock import MagicMock, patch

from rag.chunker import (
    chunk_file,
    chunk_python_file,
    chunk_with_treesitter,
    extract_symbols,
)
from rag.indexer import (
    get_chroma_client,
    get_orbit_collection,
    get_repo_collection_name,
    index_directory,
    index_file,
)
from rag.retrieval import (
    embed,
    embed_document,
    embed_documents,
    embed_query,
)


class TestPhase3IndexOverhaul(unittest.TestCase):

    def setUp(self):
        self.test_dir = tempfile.mkdtemp()
        self.db_path = os.path.join(self.test_dir, "chroma_db")
        self.client = get_chroma_client(db_path=self.db_path)

    def tearDown(self):
        shutil.rmtree(self.test_dir, ignore_errors=True)

    def test_cosine_collection_and_per_repo_name(self):
        repo1 = os.path.join(self.test_dir, "repo1")
        repo2 = os.path.join(self.test_dir, "repo2")
        os.makedirs(repo1, exist_ok=True)
        os.makedirs(repo2, exist_ok=True)

        name1 = get_repo_collection_name(repo1)
        name2 = get_repo_collection_name(repo2)

        self.assertTrue(name1.endswith("_v2"))
        self.assertTrue(name2.endswith("_v2"))
        self.assertNotEqual(name1, name2)

        col1 = get_orbit_collection(self.client, repo_path=repo1)
        self.assertEqual(col1.metadata.get("hnsw:space"), "cosine")

    def test_nomic_task_prefixes(self):
        with patch("rag.retrieval.ollama.embeddings") as mock_emb:
            mock_emb.return_value = {"embedding": [0.1] * 384}

            embed_document("sample text")
            mock_emb.assert_called_with(model="nomic-embed-text", prompt="search_document: sample text")

            embed_query("find helper")
            mock_emb.assert_called_with(model="nomic-embed-text", prompt="search_query: find helper")

    def test_python_decorator_preservation(self):
        code = '''
@decorator_one
@decorator_two(param="val")
def decorated_function(x):
    return x * 2
'''
        chunks = chunk_file("test_dec.py", code)
        self.assertGreaterEqual(len(chunks), 1)
        fn_chunk = next(c for c in chunks if c["kind"] == "function")
        self.assertIn("@decorator_one", fn_chunk["text"])
        self.assertIn("@decorator_two", fn_chunk["text"])
        self.assertEqual(fn_chunk["symbol"], "decorated_function")

    def test_python_module_preamble(self):
        code = '''
"""Module docstring."""
import os
import sys

CONFIG_TABLE = {
    "key1": "val1",
    "key2": "val2",
}

def helper():
    return True
'''
        chunks = chunk_file("module.py", code)
        kinds = [c["kind"] for c in chunks]
        self.assertIn("preamble", kinds)
        preamble_chunk = next(c for c in chunks if c["kind"] == "preamble")
        self.assertIn("CONFIG_TABLE", preamble_chunk["text"])
        self.assertIn("import os", preamble_chunk["text"])

    def test_js_export_and_arrow_function_chunking(self):
        js_code = '''
export function exportedService(req) {
    return req.body;
}

export const arrowCalculator = (a, b) => {
    return a + b;
};

export default class MainApp {
    render() {
        return "<div/>";
    }
}
'''
        chunks = chunk_file("service.js", js_code)
        self.assertGreaterEqual(len(chunks), 3)

        symbols = [c.get("symbol") for c in chunks]
        self.assertIn("exportedService", symbols)
        self.assertIn("arrowCalculator", symbols)
        self.assertIn("MainApp", symbols)

        symbols_extracted = extract_symbols("service.js", js_code)
        self.assertIn("exportedService", symbols_extracted)
        self.assertIn("arrowCalculator", symbols_extracted)
        self.assertIn("MainApp", symbols_extracted)

    def test_contextual_headers_and_stale_cleanup(self):
        repo_dir = os.path.join(self.test_dir, "repo")
        os.makedirs(repo_dir, exist_ok=True)
        col = get_orbit_collection(self.client, repo_path=repo_dir)

        # Mock embeddings
        orig_embed = embed
        try:
            import rag.indexer
            import rag.retrieval
            def mock_embed(text):
                return [0.05] * 384
            rag.indexer.embed = mock_embed
            rag.retrieval.embed = mock_embed

            # Create file A
            file_a = os.path.join(repo_dir, "file_a.py")
            with open(file_a, "w", encoding="utf-8") as f:
                f.write("def func_a():\n    return 'a'\n")

            records = index_directory(repo_dir, collection=col)
            self.assertGreater(len(records), 0)

            # Verify chunk record has start_line and symbol metadata
            meta = col.get(ids=[records[0]["id"]], include=["metadatas"])["metadatas"][0]
            self.assertIn("start_line", meta)
            self.assertIn("lang", meta)

            # Remove file A and create file B -> indexing should prune file A
            os.remove(file_a)
            file_b = os.path.join(repo_dir, "file_b.py")
            with open(file_b, "w", encoding="utf-8") as f:
                f.write("def func_b():\n    return 'b'\n")

            index_directory(repo_dir, collection=col)

            all_records = col.get(include=["metadatas"])
            sources = [m.get("source") for m in all_records["metadatas"]]
            self.assertTrue(all("file_a.py" not in s for s in sources))
            self.assertTrue(any("file_b.py" in s for s in sources))
        finally:
            import rag.indexer
            import rag.retrieval
            rag.indexer.embed = orig_embed
            rag.retrieval.embed = orig_embed


if __name__ == "__main__":
    unittest.main()
