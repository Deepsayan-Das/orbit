# Orbit Architecture Overview

Orbit is a developer-first AI assistant for galactOS. This document outlines the modular package structure, Retrieval-Augmented Generation (RAG) vector pipeline, and Tool Registry system.

---

## 1. Directory & Package Structure

```
orbit/
├── providers/           # Provider implementations (Ollama, OpenAI, Gemini, Groq, HuggingFace)
│   ├── base.py          # BaseLLMProvider, ChatMessage, LLMResponse, StreamChunk
│   ├── ollama_provider.py
│   ├── openai_provider.py
│   ├── gemini_provider.py
│   ├── groq_provider.py
│   └── huggingface_provider.py
├── llm_client.py        # OrbitLLM unified facade with provider registry & streaming chat
├── rag/                 # Retrieval-Augmented Generation subsystem
│   ├── __init__.py
│   ├── chunker.py       # AST Python chunker, Tree-Sitter chunker (Go, JS, TS, Rust, C, C++, Java), Prose & Fallback
│   ├── indexer.py       # ChromaDB PersistentClient vector store, SHA-256 incremental hashing
│   └── retrieval.py     # Ollama nomic-embed-text embeddings, common word filtering & difflib AST symbol typo correction
├── tools/               # Tool Registry & Builtin Tools subsystem
│   ├── __init__.py
│   ├── registry.py      # Tool registration, schema generation, and execution router
│   ├── filesystem.py    # list_directory and write_file (with permission confirmation)
│   ├── read_file.py     # read_file tool
│   ├── shell.py         # run_shell_command (with SAFE_COMMAND_PREFIXES allowlisting & permission check)
│   ├── git_tools.py     # git_status and git_diff read-only git introspection
│   └── code_search.py   # search_codebase vector search tool wrapper
├── repl.py              # Interactive REPL orchestrating RAG context retrieval and LLM chat
├── orbit_repl.py        # Legacy entrypoint stub delegating to repl.py
├── chunker.py           # Legacy import stub delegating to rag.chunker
├── docs/                # Architecture and technical documentation
│   └── ARCHITECTURE.md
└── tests/               # Automated unit test suite
```

---

## 2. RAG Subsystem (`rag/`)

The RAG pipeline provides context grounding for codebases and documents without external AI frameworks:

1. **AST-Aware Chunking (`rag/chunker.py`)**:
   - Parses Python files via standard `ast` into functions, classes, and method units.
   - Parses Go, JS, TS, Rust, Java, C, and C++ files via Tree-Sitter.
   - Parses prose/markdown files by headings and paragraphs.

2. **Persistent Vector Store & Incremental Indexing (`rag/indexer.py`)**:
   - Uses `chromadb.PersistentClient` stored locally at `./.orbit/chroma_data` (gitignored).
   - Computes SHA-256 content hashes (`content_hash`) for indexed files.
   - Skips unchanged files during indexing runs to eliminate unnecessary re-embedding.

3. **Query Typo Correction Safeguards & Vector Retrieval (`rag/retrieval.py`)**:
   - Pre-computes query embeddings using `nomic-embed-text` via Ollama.
   - Queries ChromaDB collection (`collection.query()`) with original prompt text first.
   - Runs typo correction strictly as a fallback if initial max retrieval score is weak ($< 0.45$).
   - Ignores standard English stop words (`COMMON_ENGLISH_WORDS`) to prevent query mangling.

---

## 3. Tool Registry Subsystem (`tools/`)

The Tool Registry decouples tool implementations from the interactive REPL:

1. **Registry Manager (`tools/registry.py`)**:
   - Stores tool definitions with `register_tool(name, function, schema)`.
   - Aggregates tool schemas via `get_tools_schema()`, formatted for OpenAI/Ollama function calling (`tools=[...]`).
   - Routes execution via `execute_tool(name, kwargs)`.

2. **Builtin Tools Suite**:
   - `list_directory(path)`: List directory contents.
   - `read_file(path)`: Read text content of a file from disk.
   - `write_file(path, content)`: Write/update a file on disk. Prompts for explicit user permission check before writing.
   - `run_shell_command(command)`: Execute a host shell command. Automatically executes safe read-only commands (`SAFE_COMMAND_PREFIXES`), and prompts for permission before executing non-allowlisted shell commands.
   - `git_status()` / `git_diff(staged)`: Native Git repository status and diff introspection.
   - `search_codebase(query)`: Semantic vector search callable tool exposing RAG retrieval on demand.
