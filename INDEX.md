# Orbit Project Index & Architecture Overview

Welcome to **Orbit**, the official AI assistant and multi-provider LLM client system built for **galactOS** (a developer-first operating system).

This document serves as the high-level architectural index of the Orbit codebase. It maps the repository components, RAG retrieval engine, provider ecosystem, tool registry, and interactive REPL system.

---

## 🚀 Core Features

- **Unified LLM Facade (`OrbitLLM`)**: A single interface to switch seamlessly between Ollama (local), OpenAI, HuggingFace, Google Gemini, and Groq providers.
- **Token-by-Token Streaming**: Real-time stdout token streaming for low-latency responses.
- **AST & Tree-Sitter RAG Pipeline**: Language-aware codebase chunking (Python AST, Tree-Sitter for Go, JS/TS, Rust, C++, Java) with parent context retention.
- **Persistent ChromaDB Vector Store & Incremental Indexing**: Local persistent vector index (`./.orbit/chroma_data`) with SHA-256 content hashing to avoid re-indexing unchanged files.
- **Query Typo Correction**: `difflib`-based AST symbol fuzzy matching fallback when vector retrieval scores are low ($< 0.45$).
- **Tool Registry Subsystem (`tools/`)**: Builtin tools (filesystem, shell, git, containers, dev tests, codebase vector search) with risk levels (`ToolRiskLevel`), user permission confirmation, and audit logging (`.orbit/audit.log`).
- **Interactive REPL (`repl.py`)**: Terminal user interface orchestrating context retrieval, multi-turn tool calling, and token streaming.

---

## 📁 Repository Directory Structure

```text
orbit/
├── repl.py                    # Primary interactive REPL CLI with ChromaDB RAG & Tool calling
├── orbit_repl.py              # Legacy REPL entrypoint stub (delegates to repl.py)
├── llm_client.py              # OrbitLLM unified facade & provider registry
├── chunker.py                 # Legacy import stub (delegates to rag.chunker)
├── PROVIDERS.md               # Detailed configuration guide for LLM providers
├── INDEX.md                   # Repository architecture index (this file)
├── requirements.txt           # Python dependencies
│
├── providers/                 # Provider plugin architecture
│   ├── __init__.py            # Provider package exports
│   ├── base.py                # BaseLLMProvider abstract interface & data models
│   ├── ollama_provider.py     # Local Ollama integration & embeddings
│   ├── openai_provider.py     # OpenAI API integration
│   ├── huggingface_provider.py# HuggingFace Inference API integration
│   ├── gemini_provider.py     # Google Gemini API integration
│   └── groq_provider.py       # Groq LPU API integration
│
├── rag/                       # RAG Subsystem
│   ├── __init__.py
│   ├── chunker.py             # AST Python chunker, Tree-Sitter chunker, Prose & Fallback splitters
│   ├── indexer.py             # Persistent ChromaDB client & SHA-256 incremental indexer
│   └── retrieval.py           # Embeddings generator, cosine similarity & typo correction
│
├── tools/                     # Tool Registry & Builtin Tools Subsystem
│   ├── __init__.py            # Package exports
│   ├── registry.py            # Tool registration, schema generation, risk levels & execution router
│   ├── filesystem.py          # list_directory (SAFE) and write_file (DANGEROUS)
│   ├── read_file.py           # read_file tool (SAFE)
│   ├── shell.py               # run_shell_command (DANGEROUS with permission check)
│   ├── git_tools.py           # git_status, git_diff, git_log, git_blame (SAFE), git_commit, git_push (DANGEROUS)
│   ├── container_tools.py     # container_ps, container_logs (SAFE)
│   ├── dev_tools.py           # run_tests (SENSITIVE), search_logs (SAFE)
│   └── code_search.py         # search_codebase vector search tool wrapper (SAFE)
│
├── docs/                      # Technical Documentation
│   └── ARCHITECTURE.md        # Comprehensive system architecture specification
│
└── tests/                     # Automated Test Suite
    ├── test_chunker.py        # Tests for AST chunking and module summary generators
    ├── test_rag.py            # Tests for RAG indexing, SHA-256 incremental hashing & retrieval
    ├── test_tools.py          # Core tests for tool registry and basic builtin tools
    ├── test_new_tools.py      # Unit tests for read_file, shell, git, container & dev tools
    └── test_phase12_tools.py  # Validation for filesystem tool registration & risk enforcement
```

---

## 🏗️ Architecture & Component Details

### 1. Unified Client Facade (`llm_client.py`)
`OrbitLLM` serves as the primary entry point for AI interactions in Orbit. It wraps provider instances and normalizes requests.
- **Class**: `OrbitLLM(provider="ollama", model="llama3.2:latest", **kwargs)`
- **Methods**:
  - `chat(messages, system_prompt, stream=True)`: Process multi-turn chat messages.
  - `generate(prompt, system_prompt, stream=True)`: One-shot prompt completion alias.
  - `register_provider(name, provider_cls)`: Dynamically register custom LLM providers.
- **Supported Providers**: `"ollama"`, `"openai"`, `"huggingface"`, `"gemini"`, `"groq"`.

### 2. Multi-Language RAG Subsystem (`rag/`)
- **AST Chunker (`rag/chunker.py`)**: Splits `.py` files using `ast`, multi-language code files (`.go`, `.js`, `.ts`, `.rs`, `.java`, `.c`, `.cpp`) using `tree-sitter`, prose (`.md`, `.txt`, `.rst`) on headings/paragraphs, and fallback sliding window for others. Generates deterministic module summaries.
- **Persistent Vector Store & Incremental Indexer (`rag/indexer.py`)**: Manages ChromaDB persistent client at `./.orbit/chroma_data`. Computes SHA-256 hashes (`content_hash`) to skip unchanged files during re-indexing.
- **Retrieval Engine & Typo Correction (`rag/retrieval.py`)**: Pre-computes query embeddings via Ollama `nomic-embed-text`. If the initial vector search score is below $0.45$, applies `difflib` fuzzy matching against extracted AST symbols to correct misspelled search terms.

### 3. Tool Registry Subsystem (`tools/`)
Decouples tool implementations from the interactive REPL with clear authorization boundaries.
- **Registry & Schema Generator (`tools/registry.py`)**: Manages tool registration, schema formatting for LLM function calling, and execution routing.
- **Risk Levels**: `SAFE` (auto-execute), `SENSITIVE` / `DANGEROUS` (require explicit user confirmation). Logs all actions to `.orbit/audit.log`.
- **Builtin Suite**: Filesystem, Shell execution, Git operations, Container inspection, Test execution, and Codebase semantic search.

### 4. Interactive RAG REPL (`repl.py`)
- **Initialization**: Automatically indexes target path into persistent ChromaDB store.
- **Decision & Execution Loop**: Retrieves top-k chunks, injects grounded context prompt, formats tool schemas, handles multi-turn tool calling, and streams tokens to stdout.

---

## 🛠️ Quickstart Usage

### Running the Interactive REPL
To index the current repository and launch Orbit REPL:
```bash
python repl.py ./
```

### Running Unit Tests
To run the full automated test suite:
```bash
python -m unittest discover -s tests -p "test_*.py"
```

---

## 📌 Summary for RAG Context
Orbit combines a unified multi-provider LLM interface with a persistent AST-aware RAG vector store and a secure tool execution registry. It enables local, syntax-aware code reasoning and safe automation for galactOS.
