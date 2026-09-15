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
│   ├── git_tools.py     # Git tools: git_status, git_diff, git_log, git_blame (SAFE), git_commit, git_push (DANGEROUS)
│   ├── container_tools.py # Container tools: container_ps, container_logs (SAFE) shelling out to podman/docker
│   ├── dev_tools.py     # Developer tools: run_tests, search_logs (SAFE)
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

2. **Builtin Tools Suite & Risk Levels**:
   - **Filesystem Tools** ([tools/filesystem.py](file:///d:/web%20development%20projects/ongoing/orbit/tools/filesystem.py)):
     - `list_directory(path)`: List directory contents (`SAFE`).
     - `write_file(path, content)`: Write/update a file on disk (`DANGEROUS` – requires user prompt showing exact path and content).
   - **Shell Execution** ([tools/shell.py](file:///d:/web%20development%20projects/ongoing/orbit/tools/shell.py)):
     - `run_shell_command(command)`: Host shell command execution (`DANGEROUS` – requires user prompt with exact command).
   - **Git Management** ([tools/git_tools.py](file:///d:/web%20development%20projects/ongoing/orbit/tools/git_tools.py)):
     - `git_status()` / `git_diff(staged)`: Repository status and diff inspection (`SAFE`).
     - `git_log(n)` / `git_blame(file, line)`: Commit history and line blame (`SAFE`).
     - `git_commit(message, stage_all)`: Stage changes and create commit (`DANGEROUS` – requires user prompt).
     - `git_push(remote, branch)`: Push committed changes to remote repository (`DANGEROUS` – requires user prompt).
   - **Container Management** ([tools/container_tools.py](file:///d:/web%20development%20projects/ongoing/orbit/tools/container_tools.py)):
     - `container_ps()` / `container_logs(name_or_id, tail)`: Inspect running containers and logs via host CLI (`SAFE`).
   - **Dev & Test Infrastructure** ([tools/dev_tools.py](file:///d:/web%20development%20projects/ongoing/orbit/tools/dev_tools.py)):
     - `run_tests(path)`: Execute test suite via `pytest` or `unittest` with output truncation (`SENSITIVE` – requires user prompt).
     - `search_logs(pattern, file)`: Grep-style log pattern matching (`SAFE`).
   - **Code Search** ([tools/code_search.py](file:///d:/web%20development%20projects/ongoing/orbit/tools/code_search.py)):
     - `search_codebase(query)`: Semantic vector search callable tool (`SAFE`).

