"""
Unit tests for Orbit RAG persistent indexing (ChromaDB), incremental hashing, and retrieval.
"""

import os
import shutil
import tempfile
import unittest

from rag.indexer import compute_content_hash, get_chroma_client, index_file
from rag.retrieval import correct_query_terms, cosine_similarity, retrieve


class TestOrbitRAG(unittest.TestCase):

    def setUp(self):
        self.test_dir = tempfile.mkdtemp()
        self.db_path = os.path.join(self.test_dir, "chroma_db")
        self.client = get_chroma_client(db_path=self.db_path)
        self.collection = self.client.get_or_create_collection("test_collection")

    def tearDown(self):
        shutil.rmtree(self.test_dir, ignore_errors=True)

    def test_compute_content_hash(self):
        h1 = compute_content_hash("def foo(): pass")
        h2 = compute_content_hash("def foo(): pass")
        h3 = compute_content_hash("def bar(): pass")
        self.assertEqual(h1, h2)
        self.assertNotEqual(h1, h3)

    def test_incremental_indexing(self):
        sample_file = os.path.join(self.test_dir, "sample.py")
        with open(sample_file, "w", encoding="utf-8") as f:
            f.write("def sample_func():\n    return 42\n")

        # Mock embedding function for testing without requiring Ollama server connection
        import rag.indexer
        import rag.retrieval
        orig_embed = rag.retrieval.embed

        def mock_embed(text: str):
            return [0.1] * 384

        rag.indexer.embed = mock_embed
        rag.retrieval.embed = mock_embed

        try:
            # First indexing pass
            records1 = index_file(sample_file, collection=self.collection)
            self.assertGreater(len(records1), 0)
            self.assertFalse(records1[0].get("skipped", False))

            # Second indexing pass on unchanged file -> should skip re-embedding
            records2 = index_file(sample_file, collection=self.collection)
            self.assertEqual(len(records1), len(records2))
            self.assertTrue(records2[0].get("skipped", False))

            # Modify file content -> should re-index
            with open(sample_file, "w", encoding="utf-8") as f:
                f.write("def sample_func_updated():\n    return 100\n")

            records3 = index_file(sample_file, collection=self.collection)
            self.assertFalse(records3[0].get("skipped", False))
        finally:
            rag.indexer.embed = orig_embed
            rag.retrieval.embed = orig_embed

    def test_cosine_similarity(self):
        v1 = [1.0, 0.0, 0.0]
        v2 = [1.0, 0.0, 0.0]
        v3 = [0.0, 1.0, 0.0]
        self.assertAlmostEqual(cosine_similarity(v1, v2), 1.0)
        self.assertAlmostEqual(cosine_similarity(v1, v3), 0.0)


if __name__ == "__main__":
    unittest.main()
