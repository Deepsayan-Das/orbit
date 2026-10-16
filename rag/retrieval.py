"""
Orbit Retrieval & Query Processing module.

Generates embeddings via Ollama nomic-embed-text, fuzzy-matches query terms against
AST symbols to correct typos, and queries ChromaDB vector collections for top-k chunks.
"""

import difflib
import json
import os
import re
from typing import Any, Dict, List, Optional, Set

import numpy as np
import ollama

from .bm25 import BM25, filter_and_deduplicate, reciprocal_rank_fusion

EMBED_MODEL = "nomic-embed-text"

COMMON_ENGLISH_WORDS = {
    "a", "about", "above", "after", "again", "against", "all", "am", "an", "and",
    "any", "are", "aren't", "as", "at", "be", "because", "been", "before", "being",
    "below", "between", "both", "but", "by", "can", "can't", "cannot", "chat", "code",
    "could", "couldn't", "created", "current", "did", "didn't", "directory", "directories",
    "do", "does", "doesn't", "doing", "don't", "down", "during", "each", "explain",
    "few", "file", "files", "find", "folder", "folders", "for", "from", "further", "get",
    "had", "hadn't", "has", "hasn't", "have", "haven't", "having", "he", "help", "her",
    "here", "here's", "hers", "herself", "him", "himself", "his", "how", "i", "if",
    "in", "into", "is", "isn't", "it", "it's", "its", "itself", "list", "make",
    "me", "more", "most", "mustn't", "my", "myself", "no", "nor", "not", "of",
    "off", "on", "once", "only", "or", "other", "ought", "our", "ours", "ourselves",
    "out", "over", "own", "project", "purpose", "repo", "repository", "run", "running",
    "same", "shan't", "she", "should", "shouldn't", "show", "so", "some", "such",
    "than", "that", "that's", "the", "their", "theirs", "them", "themselves", "then",
    "there", "there's", "these", "they", "they'd", "they'll", "they're", "they've",
    "this", "those", "through", "to", "too", "under", "until", "up", "use", "used",
    "using", "very", "was", "wasn't", "we", "we'd", "we'll", "we're", "we've", "were",
    "weren't", "what", "what's", "when", "when's", "where", "where's", "which", "while",
    "who", "who's", "whom", "why", "why's", "will", "with", "won't", "work", "works",
    "would", "wouldn't", "you", "you'd", "you'll", "you're", "you've", "your", "yours",
    "yourself", "yourselves"
}

_KNOWN_SYMBOLS_CACHE: Set[str] = set()
_BM25_COLLECTION_CACHE: Dict[str, BM25] = {}


def get_known_symbols(
    collection: Optional[Any] = None,
    cache_path: str = "./.orbit/symbols.json"
) -> List[str]:
    """
    Get known AST symbols for typo correction.
    Checks memory cache first, then JSON cache file on disk, avoiding
    expensive whole-collection Chroma metadata scans at query time.
    """
    global _KNOWN_SYMBOLS_CACHE
    if _KNOWN_SYMBOLS_CACHE:
        return list(_KNOWN_SYMBOLS_CACHE)

    if os.path.exists(cache_path):
        try:
            with open(cache_path, "r", encoding="utf-8") as f:
                data = json.load(f)
                if data:
                    _KNOWN_SYMBOLS_CACHE = set(data)
                    return list(_KNOWN_SYMBOLS_CACHE)
        except Exception:
            pass

    if collection is not None:
        try:
            data = collection.get(include=["metadatas"])
            for meta in data.get("metadatas", []):
                syms_str = meta.get("symbols", "")
                if syms_str:
                    for s in syms_str.split(","):
                        if s.strip():
                            _KNOWN_SYMBOLS_CACHE.add(s.strip())
                if meta.get("symbol"):
                    _KNOWN_SYMBOLS_CACHE.add(meta["symbol"])
            return list(_KNOWN_SYMBOLS_CACHE)
        except Exception:
            pass

    return []


def _get_or_build_collection_bm25(collection: Any) -> Optional[BM25]:
    """Retrieve or build BM25 index over collection documents."""
    col_name = getattr(collection, "name", "default")
    if col_name in _BM25_COLLECTION_CACHE:
        return _BM25_COLLECTION_CACHE[col_name]

    try:
        data = collection.get(include=["documents", "metadatas"])
        if not data or not data.get("documents"):
            return None

        docs = []
        documents = data["documents"]
        metas = data.get("metadatas", [{}] * len(documents))
        ids = data.get("ids", [f"doc_{i}" for i in range(len(documents))])

        for i, doc in enumerate(documents):
            meta = metas[i] if i < len(metas) and metas[i] else {}
            docs.append({
                "id": ids[i],
                "source": meta.get("source", "unknown"),
                "text": doc,
                "symbol": meta.get("symbol", ""),
                "kind": meta.get("kind", "code"),
                "level": meta.get("level", "chunk"),
            })

        bm25 = BM25().fit(docs)
        _BM25_COLLECTION_CACHE[col_name] = bm25
        return bm25
    except Exception:
        return None


def embed(text: str, prefix: Optional[str] = None) -> List[float]:
    """Generate vector embedding for text using Ollama nomic-embed-text with optional task prefix."""
    full_text = f"{prefix}{text}" if prefix else text
    try:
        response = ollama.embeddings(model=EMBED_MODEL, prompt=full_text)
        return response["embedding"]
    except Exception as e:
        raise RuntimeError(f"Ollama embedding service unavailable: {e}") from e


def embed_document(text: str) -> List[float]:
    """Generate embedding for a code/document chunk with nomic search_document task prefix."""
    return embed(text, prefix="search_document: ")


def embed_query(text: str) -> List[float]:
    """Generate embedding for a search query with nomic search_query task prefix."""
    return embed(text, prefix="search_query: ")


def embed_batch(texts: List[str], prefix: Optional[str] = None) -> List[List[float]]:
    """Batch embed multiple text strings in a single call using ollama.embed where available."""
    if not texts:
        return []
    prefixed_texts = [f"{prefix}{t}" if prefix else t for t in texts]
    try:
        if hasattr(ollama, "embed"):
            resp = ollama.embed(model=EMBED_MODEL, input=prefixed_texts)
            if isinstance(resp, dict) and "embeddings" in resp:
                return resp["embeddings"]
            elif hasattr(resp, "embeddings"):
                return resp.embeddings
    except Exception:
        pass
    # Fallback to sequential embedding
    return [embed(t) for t in prefixed_texts]


def embed_documents(texts: List[str]) -> List[List[float]]:
    """Batch embed multiple document texts with search_document task prefix."""
    return embed_batch(texts, prefix="search_document: ")


def cosine_similarity(a: List[float], b: List[float]) -> float:
    """Compute cosine similarity between two float vectors."""
    a_arr, b_arr = np.array(a), np.array(b)
    norm_a = np.linalg.norm(a_arr)
    norm_b = np.linalg.norm(b_arr)
    if norm_a == 0 or norm_b == 0:
        return 0.0
    return float(np.dot(a_arr, b_arr) / (norm_a * norm_b))


def correct_query_terms(query: str, known_symbols: List[str]) -> str:
    """
    Fuzzy-match misspelled code symbol query terms against known extracted AST symbols.
    Ignores common English words and requires high similarity cutoff (0.85) to avoid false positives.
    """
    if not known_symbols:
        return query
    words = query.split()
    corrected = []
    for word in words:
        match_obj = re.search(r'^\W*(.*?)\W*$', word)
        core_word = match_obj.group(1) if match_obj else word
        if core_word and core_word.lower() not in COMMON_ENGLISH_WORDS and len(core_word) > 3:
            matches = difflib.get_close_matches(core_word, known_symbols, n=1, cutoff=0.85)
            if matches and abs(len(core_word) - len(matches[0])) <= 3:
                corrected_word = word.replace(core_word, matches[0], 1)
                corrected.append(corrected_word)
                continue
        corrected.append(word)
    return " ".join(corrected)


def _format_chroma_results(results: Dict[str, Any]) -> List[Dict[str, Any]]:
    retrieved = []
    if results and results.get("documents") and results["documents"][0]:
        docs = results["documents"][0]
        metas = results["metadatas"][0] if results.get("metadatas") else [{}] * len(docs)
        dists = results["distances"][0] if results.get("distances") else [0.0] * len(docs)
        ids = results["ids"][0] if results.get("ids") else ["" for _ in range(len(docs))]

        for i in range(len(docs)):
            dist = dists[i] if dists[i] is not None else 0.0
            # With Chroma hnsw:space = "cosine", distance is (1 - cosine_sim).
            # Therefore similarity score = 1.0 - distance, bounded to [0.0, 1.0].
            score = max(0.0, min(1.0, 1.0 - dist))
            retrieved.append({
                "id": ids[i],
                "source": metas[i].get("source", "unknown"),
                "text": docs[i],
                "level": metas[i].get("level", "chunk"),
                "kind": metas[i].get("kind", "code"),
                "symbol": metas[i].get("symbol", ""),
                "score": score,
            })
    return retrieved


def retrieve(
    query: str, 
    collection: Optional[Any] = None, 
    records: Optional[List[Dict[str, Any]]] = None,
    top_k: int = 3, 
    known_symbols: Optional[List[str]] = None,
    min_score: float = 0.35,
    max_per_file: int = 2,
    hybrid: bool = True,
    cache_path: str = "./.orbit/symbols.json"
) -> List[Dict[str, Any]]:
    """
    Retrieve top-k relevant code/prose chunks for a user query.
    1. Uses nomic search_query prefix for semantic vector retrieval.
    2. Combines with BM25 keyword search via Reciprocal Rank Fusion (RRF) for exact symbol precision.
    3. Employs fast cached AST symbol typo correction without scanning the whole collection.
    4. Enforces relevance floor (min_score) and per-file deduplication (max_per_file).
    """
    candidate_k = max(top_k * 3, 10)
    try:
        if collection is not None:
            retrieved_vector: List[Dict[str, Any]] = []
            try:
                try:
                    query_vec = embed_query(query)
                except Exception:
                    query_vec = embed(query)

                results = collection.query(
                    query_embeddings=[query_vec],
                    n_results=candidate_k,
                    include=["documents", "metadatas", "distances"]
                )
                retrieved_vector = _format_chroma_results(results)
            except Exception:
                retrieved_vector = []

            max_score = max([r["score"] for r in retrieved_vector], default=0.0)

            # Fallback typo correction if initial retrieval score is weak
            if max_score < 0.45:
                if known_symbols is None:
                    known_symbols = get_known_symbols(collection, cache_path=cache_path)

                corrected_query = correct_query_terms(query, known_symbols)
                if corrected_query != query:
                    print(f"[orbit] typo correction: '{query}' -> '{corrected_query}'")
                    try:
                        try:
                            query_vec = embed_query(corrected_query)
                        except Exception:
                            query_vec = embed(corrected_query)

                        results = collection.query(
                            query_embeddings=[query_vec],
                            n_results=candidate_k,
                            include=["documents", "metadatas", "distances"]
                        )
                        retrieved_vector = _format_chroma_results(results)
                    except Exception:
                        pass

            # Hybrid keyword matching with BM25
            if hybrid:
                bm25_idx = _get_or_build_collection_bm25(collection)
                if bm25_idx:
                    bm25_results = bm25_idx.search(query, top_k=candidate_k)
                    bm25_chunks = [item[0] for item in bm25_results]
                    if bm25_chunks and retrieved_vector:
                        candidates = reciprocal_rank_fusion(retrieved_vector, bm25_chunks, top_k=candidate_k)
                    elif bm25_chunks:
                        candidates = bm25_chunks
                    else:
                        candidates = retrieved_vector
                else:
                    candidates = retrieved_vector
            else:
                candidates = retrieved_vector

            return filter_and_deduplicate(candidates, min_score=min_score, max_per_file=max_per_file, top_k=top_k)

        elif records is not None:
            scored: List[Dict[str, Any]] = []
            try:
                try:
                    query_vec = embed_query(query)
                except Exception:
                    query_vec = embed(query)

                scored = [
                    {**r, "score": cosine_similarity(query_vec, r["embedding"])}
                    for r in records
                ]
                scored.sort(key=lambda r: r["score"], reverse=True)
            except Exception:
                scored = []

            max_score = scored[0]["score"] if scored else 0.0

            if max_score < 0.45:
                if known_symbols is None:
                    known_symbols_set = set()
                    for r in records:
                        for sym in r.get("symbols", []):
                            known_symbols_set.add(sym)
                        if "symbol" in r and r["symbol"]:
                            known_symbols_set.add(r["symbol"])
                    known_symbols = list(known_symbols_set)

                corrected_query = correct_query_terms(query, known_symbols)
                if corrected_query != query:
                    print(f"[orbit] typo correction: '{query}' -> '{corrected_query}'")
                    try:
                        try:
                            query_vec = embed_query(corrected_query)
                        except Exception:
                            query_vec = embed(corrected_query)
                        scored = [
                            {**r, "score": cosine_similarity(query_vec, r["embedding"])}
                            for r in records
                        ]
                        scored.sort(key=lambda r: r["score"], reverse=True)
                    except Exception:
                        pass

            if hybrid:
                bm25 = BM25().fit(records)
                bm25_res = bm25.search(query, top_k=candidate_k)
                bm25_chunks = [b[0] for b in bm25_res]
                if bm25_chunks and scored:
                    candidates = reciprocal_rank_fusion(scored, bm25_chunks, top_k=candidate_k)
                elif bm25_chunks:
                    candidates = bm25_chunks
                else:
                    candidates = scored
            else:
                candidates = scored

            return filter_and_deduplicate(candidates, min_score=min_score, max_per_file=max_per_file, top_k=top_k)

        else:
            return []
    except Exception:
        return []


def build_context_prompt(query: str, top_chunks: List[Dict[str, Any]]) -> str:
    """Format retrieved code chunks into prompt context for LLM generation."""
    if not top_chunks:
        return query

    context = "\n\n---\n\n".join(
        f"# from {c['source']}\n{c['text']}" for c in top_chunks
    )
    return (
        f"Repository context:\n\n{context}\n\n"
        f"Question: {query}\n\n"
        "Answer using the context above. If the context doesn't contain "
        "the answer, say so plainly rather than guessing."
    )
