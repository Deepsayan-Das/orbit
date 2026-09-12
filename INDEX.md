# Orbit Project Index & Architecture Overview

Welcome to **Orbit**, the official AI assistant and multi-provider LLM client system built for **galactOS** (a developer-first operating system).

This document serves as the high-level architectural index of the Orbit codebase. It maps the repository components, RAG retrieval engine, provider ecosystem, and interactive REPL system.

---

## 🚀 Core Features

- **Unified LLM Facade (`OrbitLLM`)**: A single interface to switch seamlessly between Ollama (local), OpenAI, HuggingFace, and Google Gemini providers.
- **Token-by-Token Streaming**: Real-time stdout token streaming for low-latency responses.
- **AST & Tree-Sitter RAG Pipeline**: Language-aware codebase chunking (Python AST, Tree-Sitter for Go, JS/TS, Rust, C++, Java) with parent context retention.
- **Module-Level Summarization**: Deterministic, LLM-free module and file summary extraction embedded alongside fine-grained code chunks.
- **Interactive REPL (`orbit_repl.py`)**: A command-line terminal interface with grounded repository context retrieval, noise filtering, and turn-by-turn conversation state.

---

## 📁 Repository Directory Structure

```text
orbit/
├── orbit_repl.py              # Interactive RAG REPL CLI & embedding indexer
├── orbit-core.py              # Core usage demonstrations (streaming & chat history)
├── llm_client.py              # OrbitLLM unified facade & provider registry
├── chunker.py                 # Multi-language AST dispatcher & module summarizer
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
│   └── gemini_provider.py     # Google Gemini API integration
│
└── tests/                     # Test suite
    └── test_chunker.py        # Unit tests for chunkers and summary generators
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

### 2. Multi-Language RAG Chunker (`chunker.py`)
Extracts semantic chunks from source files while preserving context boundaries.
- **Python AST Chunker (`chunk_python_file`)**: Splits Python code on function and class nodes using Python's native `ast` module.
- **Tree-Sitter Chunker (`chunk_with_treesitter`)**: Uses `tree-sitter` for `.go`, `.js`, `.ts`, `.rs`, `.java`, `.c`, `.cpp`. Over-sized classes or structs are split per-method with parent header context prepended.
- **Prose Chunker (`chunk_prose`)**: Splits Markdown (`.md`), `.txt`, and `.rst` documents on heading boundaries (`#`, `##`) or paragraph breaks (`\n\n`).
- **Fallback Chunker (`chunk_fallback`)**: Character-count sliding window (size ~500, overlap ~50) for unrecognized extensions.
- **Module Summarizer (`build_module_summary`)**: Cheap, deterministic summary builder extracting docstrings/comment blocks + top-level symbol signatures for code, or heading + paragraph for prose.

### 3. Interactive RAG REPL (`orbit_repl.py`)
- **Index Generation (`index_directory`)**: Recursively scans directory source files, skipping noise directories (`.git`, `node_modules`, `__pycache__`, `venv`, `.env`) and embedding fine-grained chunks (`level="chunk"`) plus module summaries (`level="module"`).
- **Cosine Retrieval (`retrieve`)**: Calculates vector cosine similarity between user queries and stored chunk embeddings using `nomic-embed-text`.
- **Grounded Chat Loop (`repl`)**: Injects retrieved context into temporary prompt turns without polluting long-term chat history.

### 4. Provider Subsystem (`providers/`)
All providers inherit from `BaseLLMProvider` in `providers/base.py`:
- `OllamaProvider`: Local LLM execution & embedding generation (`ollama.embeddings`).
- `OpenAIProvider`: Calls OpenAI's `chat.completions` API.
- `HuggingFaceProvider`: Calls HuggingFace Inference API endpoints.
- `GeminiProvider`: Integration with Google Gemini API models.

---

## 🛠️ Quickstart Usage

### Running the Interactive REPL
To index the current repository and launch Orbit:
```bash
python3 orbit_repl.py ./
```

### Running Unit Tests
To run the automated unit test suite:
```bash
python3 -m unittest discover -s tests -p "test_*.py"
```

---

## 📌 Summary for RAG Context
Orbit combines a unified multi-provider LLM interface with a lightweight, framework-free RAG indexer. It enables fast, local, syntax-aware code reasoning for galactOS.
