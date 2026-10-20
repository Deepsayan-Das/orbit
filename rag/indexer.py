"""
Orbit Persistent Vector Indexer powered by ChromaDB.

Provides persistent collection management, document chunking + precomputed Ollama
embeddings, and incremental SHA-256 hashing to skip re-indexing unchanged files.
"""

import hashlib
import json
import os
from pathlib import Path
from typing import Any, Dict, List, Optional, Set

import chromadb

from .chunker import build_module_summary, chunk_file, extract_symbols
from .retrieval import embed, embed_documents, embed_query

DEFAULT_DB_PATH = "./.orbit/chroma_data"
DEFAULT_COLLECTION_NAME = "orbit_codebase_v2"
DEFAULT_SYMBOL_CACHE_PATH = "./.orbit/symbols.json"


def save_symbol_cache(symbols: Any, cache_path: str = DEFAULT_SYMBOL_CACHE_PATH) -> None:
    """Save extracted AST symbol list/set to a JSON cache file to eliminate collection scanning at query time."""
    try:
        os.makedirs(os.path.dirname(os.path.abspath(cache_path)), exist_ok=True)
        sym_list = sorted(list(set(symbols)))
        with open(cache_path, "w", encoding="utf-8") as f:
            json.dump(sym_list, f, indent=2)
    except Exception:
        pass


def load_symbol_cache(cache_path: str = DEFAULT_SYMBOL_CACHE_PATH) -> List[str]:
    """Load cached AST symbols from disk."""
    if os.path.exists(cache_path):
        try:
            with open(cache_path, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception:
            pass
    return []

NOISE_DIRS = {
    ".git", ".github", "node_modules", "__pycache__", "venv", ".venv",
    "env", ".orbit", ".idea", ".vscode", "dist", "build", "site-packages",
    "chroma_db", "chroma_data"
}
IGNORED_FILES = {
    ".gitignore", ".gitattributes", ".ds_store", "package-lock.json",
    "yarn.lock", "pnpm-lock.yaml", "pipfile.lock", "poetry.lock"
}
SKIP_EXTENSIONS = {
    ".pyc", ".png", ".jpg", ".jpeg", ".gif", ".exe", ".dll", ".so",
    ".dylib", ".zip", ".tar", ".gz", ".lock", ".bin", ".pdf",
    ".sqlite3", ".sqlite", ".db"
}


def get_chroma_client(db_path: str = DEFAULT_DB_PATH) -> chromadb.PersistentClient:
    """Initialize and return a persistent local ChromaDB client."""
    os.makedirs(db_path, exist_ok=True)
    return chromadb.PersistentClient(path=db_path)


def get_repo_collection_name(repo_path: str = "./") -> str:
    """Generate deterministic, isolated ChromaDB collection name keyed by repository path."""
    try:
        resolved = str(Path(repo_path).resolve()).lower().replace("\\", "/")
        repo_hash = hashlib.sha256(resolved.encode("utf-8")).hexdigest()[:8]
        return f"orbit_codebase_{repo_hash}_v2"
    except Exception:
        return DEFAULT_COLLECTION_NAME


def get_orbit_collection(
    client: Optional[chromadb.PersistentClient] = None,
    collection_name: Optional[str] = None,
    repo_path: str = "./",
) -> chromadb.Collection:
    """Retrieve or create the Orbit ChromaDB vector collection using cosine distance.
    Automatically detects embedding model changes in metadata and resets collection to prevent dimension mismatch."""
    if client is None:
        client = get_chroma_client()
    if collection_name is None:
        collection_name = get_repo_collection_name(repo_path)

    try:
        from .retrieval import get_embedding_provider_and_model
        emb_prov, emb_model = get_embedding_provider_and_model()
    except Exception:
        emb_prov, emb_model = "ollama", "nomic-embed-text"

    target_model_tag = f"{emb_prov}:{emb_model}"

    # Check if collection exists and whether its embedding model changed
    if hasattr(client, "get_collection"):
        try:
            existing_col = client.get_collection(name=collection_name)
            meta = existing_col.metadata or {}
            existing_model_tag = meta.get("embedding_model_tag")
            if existing_model_tag and existing_model_tag != target_model_tag:
                print(
                    f"[orbit] Embedding model changed ('{existing_model_tag}' -> '{target_model_tag}'). "
                    f"Rebuilding vector index to prevent dimension mismatch..."
                )
                client.delete_collection(name=collection_name)
        except Exception:
            pass

    return client.get_or_create_collection(
        name=collection_name,
        metadata={
            "hnsw:space": "cosine",
            "description": "Orbit AST-aware code & document embeddings (v2)",
            "embedding_model_tag": target_model_tag,
            "embedding_provider": emb_prov,
            "embedding_model": emb_model,
        }
    )


def is_ignored_path(path: Path) -> bool:
    """Check if a file or directory path should be skipped during indexing."""
    for part in path.parts[:-1]:
        if part in NOISE_DIRS or (part.startswith(".") and part not in {".", ".."}):
            return True

    filename = path.name.lower()
    if filename.startswith(".env") or filename in IGNORED_FILES:
        return True
    if path.suffix.lower() in SKIP_EXTENSIONS:
        return True

    return False


def compute_content_hash(source: str) -> str:
    """Compute SHA-256 content hash for change detection / incremental indexing."""
    return hashlib.sha256(source.encode("utf-8")).hexdigest()


def _embed_documents_safe(texts: List[str]) -> List[List[float]]:
    """Embed documents, using batched nomic search_document prefix while respecting test mocks."""
    import rag.indexer
    if rag.indexer.embed != embed:
        return [rag.indexer.embed(t) for t in texts]
    try:
        return embed_documents(texts)
    except Exception:
        return [embed(t) for t in texts]


def index_file(
    filepath: str, 
    collection: Optional[chromadb.Collection] = None
) -> List[Dict[str, Any]]:
    """
    Chunk, embed, and store records for a single file into ChromaDB.
    Uses SHA-256 content hashes to make indexing idempotent (skips unchanged files).
    Embeds contextual headers with file, symbol, kind, and language context.
    """
    if collection is None:
        collection = get_orbit_collection()

    norm_path = os.path.normpath(filepath).replace("\\", "/")
    try:
        source = Path(filepath).read_text(encoding="utf-8", errors="replace")
    except Exception as e:
        print(f"[warn] could not read {filepath}: {e}")
        return []

    content_hash = compute_content_hash(source)

    # Incremental check: see if file is already indexed with identical content hash
    try:
        existing = collection.get(where={"source": norm_path})
        if existing and existing.get("metadatas"):
            metas = existing["metadatas"]
            if metas and all(m.get("content_hash") == content_hash for m in metas):
                # Unchanged file — return existing records without re-embedding
                records = []
                for i, doc in enumerate(existing["documents"]):
                    m = metas[i]
                    records.append({
                        "id": existing["ids"][i],
                        "source": norm_path,
                        "text": doc,
                        "level": m.get("level", "chunk"),
                        "kind": m.get("kind", "code"),
                        "symbol": m.get("symbol", ""),
                        "symbols": m.get("symbols", "").split(",") if m.get("symbols") else [],
                        "start_line": m.get("start_line", 1),
                        "end_line": m.get("end_line", 1),
                        "skipped": True
                    })
                return records

        # File changed or not yet indexed: delete outdated records for this file
        if existing and existing.get("ids"):
            collection.delete(ids=existing["ids"])
    except Exception:
        pass

    file_symbols = extract_symbols(filepath, source)
    symbols_str = ",".join(file_symbols) if file_symbols else ""
    lang = Path(filepath).suffix.lstrip(".").lower() or "text"

    texts_to_embed: List[str] = []
    chunk_descriptors: List[Dict[str, Any]] = []

    # 1. Module-level summary record
    summary_text = build_module_summary(filepath, source)
    if summary_text:
        doc_id = f"{norm_path}::summary"
        summary_header = f"File: {norm_path} | Level: module | Kind: summary | Lang: {lang}\n"
        texts_to_embed.append(f"{summary_header}{summary_text}")
        chunk_descriptors.append({
            "id": doc_id,
            "text": summary_text,
            "level": "module",
            "kind": "summary",
            "symbol": "summary",
            "symbols_str": symbols_str,
            "start_line": 1,
            "end_line": len(source.splitlines()),
        })

    # 2. Fine-grained chunk records
    file_chunks = chunk_file(filepath, source)
    for idx, c in enumerate(file_chunks):
        doc_id = f"{norm_path}::chunk_{idx}"
        chunk_text = c.get("text", "")
        kind = c.get("kind", "code")
        symbol = c.get("symbol", "")
        start_line = c.get("start_line", 1)
        end_line = c.get("end_line", start_line + len(chunk_text.splitlines()))

        context_header = f"File: {norm_path} | Symbol: {symbol} | Kind: {kind} | Lang: {lang}\n"
        texts_to_embed.append(f"{context_header}{chunk_text}")
        chunk_descriptors.append({
            "id": doc_id,
            "text": chunk_text,
            "level": "chunk",
            "kind": kind,
            "symbol": symbol,
            "symbols_str": "",  # Store full symbols on module summary record to keep chunk metadata compact
            "start_line": start_line,
            "end_line": end_line,
        })

    # Batched embedding call for all chunks of this file
    if texts_to_embed:
        embeddings = _embed_documents_safe(texts_to_embed)
    else:
        embeddings = []

    ids: List[str] = []
    documents: List[str] = []
    metadatas: List[Dict[str, Any]] = []
    records: List[Dict[str, Any]] = []

    for desc, emb in zip(chunk_descriptors, embeddings):
        meta = {
            "source": norm_path,
            "level": desc["level"],
            "kind": desc["kind"],
            "symbol": desc["symbol"],
            "lang": lang,
            "start_line": desc["start_line"],
            "end_line": desc["end_line"],
            "content_hash": content_hash,
        }
        if desc["symbols_str"]:
            meta["symbols"] = desc["symbols_str"]

        ids.append(desc["id"])
        documents.append(desc["text"])
        metadatas.append(meta)
        records.append({
            "id": desc["id"],
            "source": norm_path,
            "text": desc["text"],
            "embedding": emb,
            "level": desc["level"],
            "kind": desc["kind"],
            "symbol": desc["symbol"],
            "symbols": file_symbols if desc["level"] == "module" else [],
            "start_line": desc["start_line"],
            "end_line": desc["end_line"],
        })

    if ids:
        try:
            collection.upsert(
                ids=ids,
                documents=documents,
                embeddings=embeddings,
                metadatas=metadatas
            )
        except Exception as e:
            if "dimension" in str(e).lower():
                print(f"[orbit] Dimension mismatch detected during index ({e}). Rebuilding collection...")
                client = get_chroma_client()
                col_name = getattr(collection, "name", get_repo_collection_name(norm_path))
                try:
                    client.delete_collection(name=col_name)
                except Exception:
                    pass
                new_col = get_orbit_collection(client=client, collection_name=col_name, repo_path=norm_path)
                new_col.upsert(
                    ids=ids,
                    documents=documents,
                    embeddings=embeddings,
                    metadatas=metadatas
                )
            else:
                raise

    return records


def index_directory(
    dirpath: str, 
    collection: Optional[chromadb.Collection] = None
) -> List[Dict[str, Any]]:
    """Chunk + embed every code/prose file under a directory into ChromaDB.
    Prunes stale chunks from deleted or renamed files."""
    if collection is None:
        collection = get_orbit_collection(repo_path=dirpath)

    records: List[Dict[str, Any]] = []
    base_path = Path(dirpath)

    if base_path.is_file():
        try:
            return index_file(str(base_path), collection)
        except Exception:
            return []

    current_files = set()
    warned_ollama = False
    for path in base_path.rglob("*"):
        if path.is_file() and not is_ignored_path(path):
            norm_p = os.path.normpath(str(path)).replace("\\", "/")
            current_files.add(norm_p)
            try:
                records.extend(index_file(str(path), collection))
            except Exception as e:
                if not warned_ollama and "Ollama embedding service unavailable" in str(e):
                    print("  [note: Ollama embedding service (nomic-embed-text) not detected. Vector RAG indexing skipped, but Orbit LLM client remains fully functional.]")
                    warned_ollama = True
                    break
                elif not warned_ollama:
                    print(f"[warn] skipping {path}: {e}")

    # Prune stale records from deleted or renamed files
    try:
        existing = collection.get(include=["metadatas"])
        if existing and existing.get("ids") and existing.get("metadatas"):
            stale_ids = []
            for doc_id, meta in zip(existing["ids"], existing["metadatas"]):
                src = meta.get("source", "")
                if src and src not in current_files and not os.path.exists(src):
                    stale_ids.append(doc_id)
            if stale_ids:
                collection.delete(ids=stale_ids)
    except Exception:
        pass

    # Persist extracted AST symbols to cache to accelerate typo correction and avoid whole-collection scans
    extracted_symbols: Set[str] = set()
    for r in records:
        sym = r.get("symbol")
        if sym:
            extracted_symbols.add(sym)
        for s in r.get("symbols", []):
            if s:
                extracted_symbols.add(s)
    if extracted_symbols:
        cache_path = os.path.join(dirpath, ".orbit", "symbols.json") if os.path.isdir(dirpath) else DEFAULT_SYMBOL_CACHE_PATH
        save_symbol_cache(extracted_symbols, cache_path)

    return records
