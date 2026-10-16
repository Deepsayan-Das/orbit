"""
Orbit BM25 Keyword Search & Reciprocal Rank Fusion (RRF).

Provides token-level keyword search for code identifiers, function names, and exact terms
alongside vector embeddings to ensure precise retrieval for exact symbol queries.
"""

import math
import re
from typing import Any, Dict, List, Optional, Set, Tuple


def tokenize_code(text: str) -> List[str]:
    """
    Tokenize code/text into normalized search tokens.
    Splits compound identifiers across snake_case and camelCase.
    """
    tokens = re.findall(r'[a-zA-Z0-9_]+', text)
    result: List[str] = []
    for token in tokens:
        lower_token = token.lower()
        result.append(lower_token)

        # Split snake_case
        if "_" in token:
            for part in token.split("_"):
                part_low = part.lower()
                if part_low and part_low != lower_token:
                    result.append(part_low)

        # Split camelCase / PascalCase
        parts = re.findall(r'[A-Z]?[a-z]+|[A-Z]+(?=[A-Z][a-z]|\b)', token)
        if len(parts) > 1:
            for part in parts:
                part_low = part.lower()
                if part_low and part_low != lower_token:
                    result.append(part_low)

    return result


class BM25:
    """Okapi BM25 index tailored for code chunks and symbol lookups."""

    def __init__(self, k1: float = 1.5, b: float = 0.75):
        self.k1 = k1
        self.b = b
        self.doc_len: List[int] = []
        self.avgdl: float = 0.0
        self.doc_freqs: List[Dict[str, int]] = []
        self.idf: Dict[str, float] = {}
        self.corpus_size: int = 0
        self.docs: List[Dict[str, Any]] = []

    def fit(self, docs: List[Dict[str, Any]]) -> "BM25":
        """Index a list of chunk dictionaries containing 'text', 'source', 'symbol'."""
        self.docs = docs
        self.corpus_size = len(docs)
        if self.corpus_size == 0:
            return self

        self.doc_len = []
        self.doc_freqs = []
        df: Dict[str, int] = {}

        total_len = 0
        for doc in docs:
            # Emphasize symbol and source path in tokenization for precise code matching
            sym = doc.get("symbol", "")
            src = doc.get("source", "")
            txt = doc.get("text", "")
            boosted_text = f"{src} {sym} {sym} {txt}"

            tokens = tokenize_code(boosted_text)
            length = len(tokens)
            self.doc_len.append(length)
            total_len += length

            freqs: Dict[str, int] = {}
            for t in tokens:
                freqs[t] = freqs.get(t, 0) + 1
            self.doc_freqs.append(freqs)

            for term in freqs:
                df[term] = df.get(term, 0) + 1

        self.avgdl = total_len / self.corpus_size if self.corpus_size > 0 else 0.0

        self.idf = {}
        for term, freq in df.items():
            # Okapi smoothed IDF
            self.idf[term] = math.log(1.0 + (self.corpus_size - freq + 0.5) / (freq + 0.5))

        return self

    def search(self, query: str, top_k: int = 10) -> List[Tuple[Dict[str, Any], float]]:
        """Score all documents against query tokens and return top_k (doc, score) pairs."""
        if not self.corpus_size or not self.docs:
            return []
        tokens = tokenize_code(query)
        if not tokens:
            return []

        scores: List[float] = [0.0] * self.corpus_size
        for token in tokens:
            if token not in self.idf:
                continue
            idf_val = self.idf[token]
            for idx, freqs in enumerate(self.doc_freqs):
                if token not in freqs:
                    continue
                tf = freqs[token]
                doc_len = self.doc_len[idx]
                denom = tf + self.k1 * (1.0 - self.b + self.b * (doc_len / self.avgdl if self.avgdl > 0 else 1.0))
                scores[idx] += idf_val * (tf * (self.k1 + 1.0) / (denom if denom > 0 else 1.0))

        ranked_indices = sorted(
            [i for i in range(self.corpus_size) if scores[i] > 0.0],
            key=lambda i: scores[i],
            reverse=True
        )
        return [(self.docs[i], scores[i]) for i in ranked_indices[:top_k]]


def reciprocal_rank_fusion(
    vector_results: List[Dict[str, Any]],
    bm25_results: List[Dict[str, Any]],
    k: int = 60,
    top_k: int = 5
) -> List[Dict[str, Any]]:
    """
    Merge vector search results and BM25 keyword results using Reciprocal Rank Fusion (RRF).
    RRF score = sum(1.0 / (k + rank)).
    """
    rrf_scores: Dict[str, float] = {}
    doc_map: Dict[str, Dict[str, Any]] = {}

    for rank, doc in enumerate(vector_results, 1):
        key = doc.get("id") or f"{doc.get('source')}::{doc.get('symbol')}::{doc.get('text', '')[:60]}"
        doc_map[key] = doc
        rrf_scores[key] = rrf_scores.get(key, 0.0) + (1.0 / (k + rank))

    for rank, doc in enumerate(bm25_results, 1):
        key = doc.get("id") or f"{doc.get('source')}::{doc.get('symbol')}::{doc.get('text', '')[:60]}"
        if key not in doc_map:
            doc_map[key] = doc
        rrf_scores[key] = rrf_scores.get(key, 0.0) + (1.0 / (k + rank))

    ranked = sorted(rrf_scores.items(), key=lambda x: x[1], reverse=True)
    fused: List[Dict[str, Any]] = []
    for key, score in ranked:
        item = dict(doc_map[key])
        item["rrf_score"] = score
        fused.append(item)

    return fused[:top_k]


def filter_and_deduplicate(
    chunks: List[Dict[str, Any]],
    min_score: float = 0.35,
    max_per_file: int = 2,
    top_k: int = 3,
) -> List[Dict[str, Any]]:
    """
    Filter chunks below the relevance floor and deduplicate so at most
    max_per_file chunks originate from the same source file.
    """
    filtered: List[Dict[str, Any]] = []
    file_counts: Dict[str, int] = {}

    for chunk in chunks:
        # Check relevance floor for vector scores
        score = chunk.get("score")
        # If pure vector score is present and below min_score (and no strong keyword/RRF rank), drop it
        if score is not None and score < min_score and "rrf_score" not in chunk:
            continue

        source = chunk.get("source", "unknown")
        if file_counts.get(source, 0) >= max_per_file:
            continue

        file_counts[source] = file_counts.get(source, 0) + 1
        filtered.append(chunk)

        if len(filtered) >= top_k:
            break

    return filtered
