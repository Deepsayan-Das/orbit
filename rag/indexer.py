"""
Orbit Persistent Vector Indexer powered by ChromaDB.

Provides persistent collection management, document chunking + precomputed Ollama
embeddings, and incremental SHA-256 hashing to skip re-indexing unchanged files.
"""

import hashlib
import os
from pathlib import Path
from typing import Any, Dict, List, Optional

import chromadb

from .chunker import build_module_summary, chunk_file, extract_symbols
from .retrieval import embed

DEFAULT_DB_PATH = "./.orbit/chroma_data"
DEFAULT_COLLECTION_NAME = "orbit_codebase"

NOISE_DIRS = {
    ".git", ".github", "node_modules", "__pycache__", "venv", ".venv",
    "env", ".orbit", ".idea", ".vscode", "dist", "build", "site-packages"
}
IGNORED_FILES = {
    ".gitignore", ".gitattributes", ".ds_store", "package-lock.json",
    "yarn.lock", "pnpm-lock.yaml", "pipfile.lock", "poetry.lock"
}
SKIP_EXTENSIONS = {
    ".pyc", ".png", ".jpg", ".jpeg", ".gif", ".exe", ".dll", ".so",
    ".dylib", ".zip", ".tar", ".gz", ".lock", ".bin", ".pdf"
}


def get_chroma_client(db_path: str = DEFAULT_DB_PATH) -> chromadb.PersistentClient:
    """Initialize and return a persistent local ChromaDB client."""
    os.makedirs(db_path, exist_ok=True)
    return chromadb.PersistentClient(path=db_path)


def get_orbit_collection(
    client: Optional[chromadb.PersistentClient] = None,
    collection_name: str = DEFAULT_COLLECTION_NAME
) -> chromadb.Collection:
    """Retrieve or create the Orbit ChromaDB vector collection."""
    if client is None:
        client = get_chroma_client()
    return client.get_or_create_collection(
        name=collection_name,
        metadata={"description": "Orbit AST-aware code & document embeddings"}
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


def index_file(
    filepath: str, 
    collection: Optional[chromadb.Collection] = None
) -> List[Dict[str, Any]]:
    """
    Chunk, embed, and store records for a single file into ChromaDB.
    Uses SHA-256 content hashes to make indexing idempotent (skips unchanged files).
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
                        "symbols": m.get("symbols", "").split(",") if m.get("symbols") else [],
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

    ids: List[str] = []
    documents: List[str] = []
    embeddings: List[List[float]] = []
    metadatas: List[Dict[str, Any]] = []
    records: List[Dict[str, Any]] = []

    # 1. Module-level summary record
    summary_text = build_module_summary(filepath, source)
    if summary_text:
        doc_id = f"{norm_path}::summary"
        emb = embed(summary_text)
        meta = {
            "source": norm_path,
            "level": "module",
            "kind": "summary",
            "symbols": symbols_str,
            "content_hash": content_hash
        }

        ids.append(doc_id)
        documents.append(summary_text)
        embeddings.append(emb)
        metadatas.append(meta)
        records.append({
            "id": doc_id,
            "source": norm_path,
            "text": summary_text,
            "embedding": emb,
            "level": "module",
            "kind": "summary",
            "symbols": file_symbols
        })

    # 2. Fine-grained chunk records
    file_chunks = chunk_file(filepath, source)
    for idx, c in enumerate(file_chunks):
        doc_id = f"{norm_path}::chunk_{idx}"
        chunk_text = c["text"]
        emb = embed(chunk_text)
        meta = {
            "source": norm_path,
            "level": "chunk",
            "kind": c["kind"],
            "symbols": symbols_str,
            "content_hash": content_hash
        }
        if "symbol" in c:
            meta["symbol"] = c["symbol"]

        ids.append(doc_id)
        documents.append(chunk_text)
        embeddings.append(emb)
        metadatas.append(meta)
        records.append({
            "id": doc_id,
            "source": norm_path,
            "text": chunk_text,
            "embedding": emb,
            "level": "chunk",
            "kind": c["kind"],
            "symbols": file_symbols,
            "symbol": c.get("symbol")
        })

    if ids:
        collection.upsert(
            ids=ids,
            documents=documents,
            embeddings=embeddings,
            metadatas=metadatas
        )

    return records


def index_directory(
    dirpath: str, 
    collection: Optional[chromadb.Collection] = None
) -> List[Dict[str, Any]]:
    """Chunk + embed every code/prose file under a directory into ChromaDB."""
    if collection is None:
        collection = get_orbit_collection()

    records: List[Dict[str, Any]] = []
    base_path = Path(dirpath)

    if base_path.is_file():
        return index_file(str(base_path), collection)

    for path in base_path.rglob("*"):
        if path.is_file() and not is_ignored_path(path):
            try:
                records.extend(index_file(str(path), collection))
            except Exception as e:
                print(f"[warn] skipping {path}: {e}")

    return records
