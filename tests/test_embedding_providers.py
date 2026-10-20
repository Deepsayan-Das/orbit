"""
Unit tests for multi-provider embedding resolution, fallbacks, and exit reporting.
"""

import os
import unittest
from unittest.mock import MagicMock, patch

from config import OrbitConfig, load_config
import rag.retrieval as retrieval


class TestEmbeddingProviders(unittest.TestCase):

    def test_default_embedding_config(self):
        """Test default embedding configuration defaults to ollama/nomic-embed-text."""
        with patch.dict(os.environ, {}, clear=True):
            prov, model = retrieval.get_embedding_provider_and_model()
            self.assertEqual(prov, "ollama")
            self.assertEqual(model, "nomic-embed-text")

    def test_env_embedding_override(self):
        """Test ORBIT_EMBEDDING_PROVIDER and ORBIT_EMBEDDING_MODEL overrides."""
        env = {
            "ORBIT_EMBEDDING_PROVIDER": "openai",
            "ORBIT_EMBEDDING_MODEL": "text-embedding-3-small",
        }
        with patch.dict(os.environ, env):
            prov, model = retrieval.get_embedding_provider_and_model()
            self.assertEqual(prov, "openai")
            self.assertEqual(model, "text-embedding-3-small")

    def test_openai_embedding_dispatch(self):
        """Test OpenAI embedding calls OpenAI API."""
        with patch("rag.retrieval.get_embedding_provider_and_model", return_value=("openai", "text-embedding-3-small")), \
             patch("rag.retrieval.get_provider_api_key", return_value="fake-sk"), \
             patch("openai.OpenAI") as mock_openai_cls:

            mock_client = MagicMock()
            mock_item = MagicMock()
            mock_item.embedding = [0.1, 0.2, 0.3]
            mock_client.embeddings.create.return_value = MagicMock(data=[mock_item])
            mock_openai_cls.return_value = mock_client

            emb = retrieval.embed("hello world")
            self.assertEqual(emb, [0.1, 0.2, 0.3])
            mock_client.embeddings.create.assert_called_once_with(input=["hello world"], model="text-embedding-3-small")

    def test_gemini_embedding_dispatch(self):
        """Test Gemini embedding calls google.genai Client."""
        with patch("rag.retrieval.get_embedding_provider_and_model", return_value=("gemini", "gemini-embedding-2")), \
             patch("rag.retrieval.get_provider_api_key", return_value="fake-gemini-key"), \
             patch("google.genai.Client") as mock_genai_cls:

            mock_client = MagicMock()
            mock_val = MagicMock()
            mock_val.values = [0.4, 0.5, 0.6]
            mock_resp = MagicMock(embedding=mock_val)
            mock_client.models.embed_content.return_value = mock_resp
            mock_genai_cls.return_value = mock_client

            emb = retrieval.embed("hello gemini")
            self.assertEqual(emb, [0.4, 0.5, 0.6])

    def test_mistral_embedding_dispatch(self):
        """Test Mistral embedding calls Mistral client."""
        with patch("rag.retrieval.get_embedding_provider_and_model", return_value=("mistral", "mistral-embed")), \
             patch("rag.retrieval.get_provider_api_key", return_value="fake-mistral-key"), \
             patch("mistralai.client.Mistral") as mock_mistral_cls:

            mock_client = MagicMock()
            mock_item = MagicMock()
            mock_item.embedding = [0.7, 0.8, 0.9]
            mock_client.embeddings.create.return_value = MagicMock(data=[mock_item])
            mock_mistral_cls.return_value = mock_client

            emb = retrieval.embed("hello mistral")
            self.assertEqual(emb, [0.7, 0.8, 0.9])

    def test_fallback_to_ollama_on_cloud_failure(self):
        """Test that cloud provider failure falls back to Ollama nomic-embed-text."""
        with patch("rag.retrieval.get_embedding_provider_and_model", return_value=("openai", "text-embedding-3-small")), \
             patch("rag.retrieval._embed_openai", side_effect=RuntimeError("API quota exceeded")), \
             patch("rag.retrieval._embed_ollama", return_value=[[0.1, 0.2, 0.3]]) as mock_ollama:

            emb = retrieval.embed("fallback test")
            self.assertEqual(emb, [0.1, 0.2, 0.3])
            mock_ollama.assert_called_once()
            args, kwargs = mock_ollama.call_args
            self.assertEqual(kwargs.get("model") or args[1], "nomic-embed-text")

    def test_exit_when_ollama_fallback_unavailable_strict(self):
        """Test that when ORBIT_STRICT_EMBEDDINGS is set, failure reports and exits with SystemExit."""
        with patch.dict(os.environ, {"ORBIT_STRICT_EMBEDDINGS": "1"}), \
             patch("rag.retrieval.get_embedding_provider_and_model", return_value=("openai", "text-embedding-3-small")), \
             patch("rag.retrieval._embed_openai", side_effect=RuntimeError("Network down")), \
             patch("rag.retrieval._embed_ollama", side_effect=RuntimeError("Ollama service not running")):

            with self.assertRaises(SystemExit) as cm:
                retrieval.embed("will fail completely")
            self.assertEqual(cm.exception.code, 1)

    def test_exit_when_primary_ollama_unavailable_strict(self):
        """Test that when primary Ollama is unavailable in strict mode, system exits with SystemExit."""
        with patch.dict(os.environ, {"ORBIT_STRICT_EMBEDDINGS": "1"}), \
             patch("rag.retrieval.get_embedding_provider_and_model", return_value=("ollama", "nomic-embed-text")), \
             patch("rag.retrieval._embed_ollama", side_effect=RuntimeError("Ollama not installed")):

            with self.assertRaises(SystemExit) as cm:
                retrieval.embed("will fail directly")
            self.assertEqual(cm.exception.code, 1)

    def test_degraded_bm25_mode_on_embedding_failure(self):
        """Test that when embeddings fail without strict flag, Orbit activates degraded BM25 mode."""
        with patch.dict(os.environ, {}, clear=True), \
             patch("rag.retrieval.get_embedding_provider_and_model", return_value=("openai", "text-embedding-3-small")), \
             patch("rag.retrieval._embed_openai", side_effect=RuntimeError("No cloud key")), \
             patch("rag.retrieval._embed_ollama", side_effect=RuntimeError("Ollama not installed")):

            # Reset degraded mode flags
            retrieval._DEGRADED_BM25_MODE = False
            retrieval._WARNED_DEGRADED_MODE = False

            emb = retrieval.embed("test text")
            self.assertEqual(emb, [0.0])
            self.assertTrue(retrieval.is_degraded_bm25_mode())


if __name__ == "__main__":
    unittest.main()
