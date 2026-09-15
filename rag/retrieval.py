"""
Orbit Retrieval & Query Processing module.

Generates embeddings via Ollama nomic-embed-text, fuzzy-matches query terms against
AST symbols to correct typos, and queries ChromaDB vector collections for top-k chunks.
"""

import difflib
import re
from typing import Any, Dict, List, Optional

import numpy as np
import ollama

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


def embed(text: str) -> List[float]:
    """Generate vector embedding for prompt text using Ollama nomic-embed-text."""
    response = ollama.embeddings(model=EMBED_MODEL, prompt=text)
    return response["embedding"]


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

        for i in range(len(docs)):
            retrieved.append({
                "source": metas[i].get("source", "unknown"),
                "text": docs[i],
                "level": metas[i].get("level", "chunk"),
                "kind": metas[i].get("kind", "code"),
                "score": 1.0 - dists[i] if dists[i] is not None else 1.0
            })
    return retrieved


def retrieve(
    query: str, 
    collection: Optional[Any] = None, 
    records: Optional[List[Dict[str, Any]]] = None,
    top_k: int = 3, 
    known_symbols: Optional[List[str]] = None
) -> List[Dict[str, Any]]:
    """
    Retrieve top-k relevant code/prose chunks for a user query.
    First queries with original natural language text; applies typo correction fallback only if top score is poor (< 0.45).
    """
    if collection is not None:
        query_vec = embed(query)
        results = collection.query(
            query_embeddings=[query_vec],
            n_results=top_k,
            include=["documents", "metadatas", "distances"]
        )

        retrieved = _format_chroma_results(results)
        max_score = max([r["score"] for r in retrieved], default=0.0)

        # Fallback typo correction if initial retrieval score is weak
        if max_score < 0.45:
            if known_symbols is None:
                known_symbols_set = set()
                try:
                    data = collection.get(include=["metadatas"])
                    for meta in data.get("metadatas", []):
                        syms_str = meta.get("symbols", "")
                        if syms_str:
                            for s in syms_str.split(","):
                                if s.strip():
                                    known_symbols_set.add(s.strip())
                        if meta.get("symbol"):
                            known_symbols_set.add(meta["symbol"])
                except Exception:
                    pass
                known_symbols = list(known_symbols_set)

            corrected_query = correct_query_terms(query, known_symbols)
            if corrected_query != query:
                print(f"[orbit] typo correction: '{query}' -> '{corrected_query}'")
                query_vec = embed(corrected_query)
                results = collection.query(
                    query_embeddings=[query_vec],
                    n_results=top_k,
                    include=["documents", "metadatas", "distances"]
                )
                retrieved = _format_chroma_results(results)

        return retrieved

    elif records is not None:
        query_vec = embed(query)
        scored = [
            {**r, "score": cosine_similarity(query_vec, r["embedding"])}
            for r in records
        ]
        scored.sort(key=lambda r: r["score"], reverse=True)
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
                query_vec = embed(corrected_query)
                scored = [
                    {**r, "score": cosine_similarity(query_vec, r["embedding"])}
                    for r in records
                ]
                scored.sort(key=lambda r: r["score"], reverse=True)

        return scored[:top_k]

    else:
        return []


def build_context_prompt(query: str, top_chunks: List[Dict[str, Any]]) -> str:
    """Format retrieved code chunks into prompt context for LLM generation."""
    context = "\n\n---\n\n".join(
        f"# from {c['source']}\n{c['text']}" for c in top_chunks
    )
    return (
        f"Repository context:\n\n{context}\n\n"
        f"Question: {query}\n\n"
        "Answer using the context above. If the context doesn't contain "
        "the answer, say so plainly rather than guessing."
    )
