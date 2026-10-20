# Orbit Project Index & Architecture Overview

Welcome to **Orbit**, the official AI assistant and multi-provider LLM runtime built for **galactOS** (a developer-first operating system).

This document serves as the high-level architectural index of the Orbit codebase. It maps the repository components, hybrid RAG retrieval engine, provider ecosystem, context compaction subsystem, tool registry, and interactive REPL system.

---

## 🚀 Core Features

- **Unified Multi-Provider LLM Facade (`OrbitLLM`)**: A standardized client interface supporting 7 backends: **Ollama** (local), **OpenAI**, **Anthropic Claude**, **Google Gemini**, **Mistral AI**, **Groq LPU**, and **HuggingFace** Serverless Inference.
- **Resilient API Calls**: Built-in exponential backoff retry handler for transient HTTP 503 / capacity overload errors.
- **Token-by-Token Streaming**: Real-time token streaming to stdout for low-latency terminal interaction.
- **Hybrid RAG Pipeline (`rag/`)**: Combines dense semantic vector retrieval (`nomic-embed-text`) with sparse **BM25 keyword search** via **Reciprocal Rank Fusion (RRF)** for exact symbol precision, relevance floor gating (`min_score=0.35`), and candidate deduplication.
- **Multi-Language AST & Tree-Sitter Chunking**: Language-aware codebase parsing using Python `ast` alongside Tree-Sitter grammars for Go, JavaScript, TypeScript, Rust, Java, C, and C++, plus semantic prose chunking.
- **Persistent ChromaDB & Incremental Indexing**: Local persistent vector index (`./.orbit/chroma_data`) with SHA-256 content hashing to eliminate redundant re-embedding.
- **Cached AST Symbol Typo Correction**: Disk-cached symbols (`./.orbit/symbols.json`) for fast `difflib` fuzzy query correction without scanning the whole collection.
- **ReAct Multi-Step Agentic Loop (`agent_loop.py`)**: Autonomous tool-chaining cycle with configurable step ceiling (1 to 20), loop-prevention, and graceful summary degradation when step limits are reached.
- **Context Management & Compaction Subsystem (`context/`, `commands/compaction.py`)**: Automatic context window resolver (40+ models), token estimation, and proactive compaction (pairing rule, outcome stubbing, LLM summarization) triggered at $\ge 75\%$ capacity.
- **Sub-Agent Delegation (`agent_sub.py`)**: Isolated task execution via `delegate_task` restricted to `SAFE` tools with a recursion depth ceiling of 1 and unique audit tracking.
- **Risk-Tiered Tool Registry (`tools/`)**: Builtin tools (filesystem, surgical diff-based editing, shell execution, Git, Podman-native containers, test runner, semantic code search) with `SAFE`, `SENSITIVE`, and `DANGEROUS` authorization levels and persistent audit logging (`./.orbit/audit.log`).
- **Companion Integration with Nova**: First-class support for GalactOS Nova CLI operations (`nova container`, `nova cluster`, `nova project doctor`, `nova system`).
- **Interactive REPL & Slash Commands (`repl.py`, `commands/`)**: Terminal UI with Rich markdown rendering, unified diff previews, and 14 built-in slash commands.

---

## 📁 Repository Directory Structure

```text
orbit/
├── repl.py                    # Primary interactive REPL CLI with ChromaDB RAG & agentic loop
├── orbit_repl.py              # Legacy REPL entrypoint stub (delegates to repl.py)
├── orbit-core.py              # Minimal programmatic streaming and multi-turn chat demo
├── llm_client.py              # OrbitLLM unified facade, provider registry & 503 retry handler
├── agent_loop.py              # ReAct multi-step agentic loop orchestrator
├── agent_sub.py               # Isolated sub-agent execution engine & delegate_task tool
├── config.py                  # OrbitConfig dataclass, model window mappings & config.yaml persistence
├── configure.py               # Interactive CLI configuration wizard (survey-based)
├── ui.py                      # Rich terminal UI, banner, markdown styling & unified diff previews
├── chunker.py                 # Legacy import stub (delegates to rag.chunker)
├── requirements.txt           # Python dependencies
├── PROVIDERS.md               # Detailed configuration guide for LLM providers
├── INDEX.md                   # Repository architecture index (this file)
├── README.md                  # Comprehensive user & developer guide
│
├── providers/                 # Multi-provider plugin architecture
│   ├── __init__.py            # Provider package exports & normalization utilities
│   ├── base.py                # BaseLLMProvider abstract interface, ChatMessage, LLMResponse, ToolCall
│   ├── ollama_provider.py     # Local Ollama integration (custom num_ctx & GPU offload)
│   ├── openai_provider.py     # OpenAI API integration (GPT-4o, o1, o3-mini)
│   ├── anthropic_provider.py  # Anthropic Claude API integration (Claude 3.5/3.7 Sonnet, Haiku)
│   ├── gemini_provider.py     # Google Gemini API integration via google-genai SDK
│   ├── mistral_provider.py    # Mistral AI API integration (Mistral Small/Large, Codestral)
│   ├── groq_provider.py       # Groq LPU API integration (ultra-fast inference)
│   └── huggingface_provider.py# HuggingFace Serverless Inference API integration
│
├── context/                   # Context Window & Estimation Subsystem
│   ├── window.py              # Model window resolver (40+ models, user config, 4096 fallback)
│   └── measure.py             # Token estimator (reported usage or chars/4) & 75% trigger
│
├── commands/                  # REPL Slash Command Subsystem
│   ├── __init__.py            # Package exports & dispatch wrapper
│   ├── session.py             # Session dataclass (agent, config, history, REPL state)
│   ├── registry.py            # Command registry, @command decorator & exception-safe dispatch
│   ├── builtin.py             # Handlers for 14 slash commands (/help, /status, /model, /tools, etc.)
│   ├── compaction.py          # Intelligent context history compaction engine
│   └── tokens.py              # Token estimation utility delegate
│
├── rag/                       # Hybrid RAG Subsystem
│   ├── __init__.py            # Package exports
│   ├── chunker.py             # Python AST chunker, Tree-Sitter multi-language chunker, prose chunker
│   ├── bm25.py                # Sparse BM25 keyword indexer & Reciprocal Rank Fusion (RRF)
│   ├── indexer.py             # Persistent ChromaDB client, SHA-256 incremental indexer & symbol cache
│   └── retrieval.py           # Hybrid dense+sparse retrieval, task prefixes & fuzzy typo correction
│
├── tools/                     # Tool Registry & Builtin Tools Subsystem
│   ├── __init__.py            # Package exports & tool registration bootstrapping
│   ├── registry.py            # Tool registration, schema generation, risk levels & audit logger
│   ├── diff_utils.py          # Unified diff preview generator & atomic validate-and-replace algorithm
│   ├── filesystem.py          # list_directory (SAFE), write_file (DANGEROUS), edit_file (DANGEROUS)
│   ├── read_file.py           # read_file (SAFE)
│   ├── shell.py               # run_shell_command (DANGEROUS with Nova allowlisted prefixes)
│   ├── git_tools.py           # git_status, git_diff, git_log, git_blame (SAFE), git_commit, git_push (DANGEROUS)
│   ├── container_tools.py     # container_ps, container_logs (SAFE, Podman-native)
│   ├── dev_tools.py           # run_tests (SENSITIVE), search_logs (SAFE)
│   └── code_search.py         # search_codebase vector search tool wrapper (SAFE)
│
├── docs/                      # Technical Documentation
│   └── ARCHITECTURE.md        # Deep architectural design specification
│
└── tests/                     # Automated Test Suite (119 automated tests)
    ├── eval_retrieval.py      # Golden retrieval benchmark (30 queries, Recall@k & MRR)
    ├── test_agent_loop.py     # ReAct loop, step limit & ceiling degradation tests
    ├── test_anthropic.py      # Anthropic provider unit tests
    ├── test_mistral.py        # Mistral provider unit tests
    ├── test_chunker.py        # AST and Tree-Sitter chunker tests
    ├── test_commands.py       # REPL slash command tests
    ├── test_compaction.py     # Context compaction tests
    ├── test_edit_file.py      # Surgical file edit & diff validation tests
    ├── test_measure.py        # Token measurement & compaction trigger tests
    ├── test_rag.py            # Vector store, hashing & retrieval tests
    ├── test_subagent.py       # Sub-agent isolation & delegation tests
    ├── test_tools.py          # Core tool registry tests
    ├── test_new_tools.py      # Shell, git, container & dev tools tests
    ├── test_ui.py             # Rich UI rendering tests
    └── test_window.py         # Context window resolver tests
```

---

## 🏗️ Architecture & Component Details

### 1. Unified Client Facade (`llm_client.py`)
`OrbitLLM` serves as the primary entry point for AI interactions in Orbit. It wraps provider instances, standardizes parameters, translates tool calls, and handles transient errors.
- **Class**: `OrbitLLM(provider="ollama", model="llama3.2:latest", **kwargs)`
- **Key Methods**:
  - `chat(messages, system_prompt, stream=False, tools=None)`: Process multi-turn conversations with automatic 503 exponential backoff.
  - `generate(prompt, system_prompt, stream=False)`: One-shot prompt completion helper.
  - `from_config(cfg)`: Construct an instance from loaded `OrbitConfig` with per-provider credential resolution.
  - `register_provider(name, provider_cls)`: Dynamically register third-party LLM providers.
- **Supported Providers**: `"ollama"`, `"openai"`, `"anthropic"`, `"gemini"`, `"mistral"`, `"groq"`, `"huggingface"`.

### 2. Hybrid RAG Subsystem (`rag/`)
- **Multi-Language Chunker (`rag/chunker.py`)**:
  - Parses `.py` files using Python `ast` into functions, classes, and methods.
  - Parses `.go`, `.js`, `.jsx`, `.ts`, `.tsx`, `.rs`, `.java`, `.c`, and `.cpp` files using Tree-Sitter grammars.
  - Splits prose (`.md`, `.txt`, `.rst`) on headings and paragraphs.
  - Produces deterministic module-level summaries for high-level repository context.
- **Persistent ChromaDB & Incremental Indexer (`rag/indexer.py`)**:
  - Manages persistent ChromaDB collections in `./.orbit/chroma_data`.
  - Calculates SHA-256 hashes (`content_hash`) to avoid re-embedding unchanged files.
  - Caches extracted AST symbols in `./.orbit/symbols.json` during indexing.
- **Hybrid Retrieval & RRF Fusion (`rag/bm25.py`, `rag/retrieval.py`)**:
  - Dense embeddings generated via Ollama `nomic-embed-text` with task prefixes (`search_document: `, `search_query: `) and batched execution.
  - Sparse BM25 keyword matching runs in tandem to capture exact function/variable identifiers.
  - Combines results via Reciprocal Rank Fusion (RRF), filtering with a relevance floor (`min_score=0.35`) and per-file candidate caps (`max_per_file=2`).
  - Fallback fuzzy typo correction matches misspelled query terms against cached AST symbols if vector retrieval score is low ($< 0.45$).
- **Benchmark Evaluation (`tests/eval_retrieval.py`)**:
  - 30 curated queries evaluate Recall@k and Mean Reciprocal Rank (MRR) across codebase symbols.

### 3. ReAct Multi-Step Agentic Loop (`agent_loop.py`)
- Coordinates iterative reasoning and tool execution:
  1. Supplies message history and tool schemas to `OrbitLLM`.
  2. Parses model-emitted `ToolCall` objects.
  3. Executes tools via `tools.execute_tool`, respecting risk tiers and 100-line output truncation.
  4. Appends tool results as `role="tool"` messages and loops back to the model.
  5. Enforces a configurable step ceiling (`max_steps`, default 8), prompting the model for a concise final summary if the limit is reached.
  6. Automatically checks context token usage before each LLM turn, triggering proactive context compaction if $\ge 75\%$ threshold is reached.

### 4. Context Management & Compaction (`context/`, `commands/compaction.py`)
- **Window Resolver (`context/window.py`)**: Maps 40+ known models across providers to their native context limits, prioritizing user config over defaults.
- **Token Estimator (`context/measure.py`)**: Calculates token counts via provider response usage or character ratio (`chars // 4`).
- **Compaction Engine (`commands/compaction.py`)**:
  - Preserves the system prompt and the initial user request (*never-touch rule*).
  - Preserves the 6 most recent messages.
  - Atomically preserves assistant tool-call and tool-result message pairs.
  - Stubs older tool outputs (`read_file`, `edit_file`, `run_tests`, `run_shell_command`).
  - Summarizes intermediate turns using the active LLM, falling back to dropping the oldest tool pairs if still over threshold.

### 5. Risk-Tiered Tool Registry (`tools/`)
- **Authorization Tiers**:
  - `SAFE`: Runs automatically (`list_directory`, `read_file`, `git_status`, `git_diff`, `git_log`, `git_blame`, `container_ps`, `container_logs`, `search_logs`, `search_codebase`).
  - `SENSITIVE`: Prompts user before execution (`run_tests`).
  - `DANGEROUS`: Displays full payload/command preview and requires explicit user confirmation (`write_file`, `edit_file`, `run_shell_command`, `git_commit`, `git_push`).
- **Surgical File Editing (`tools/diff_utils.py`, `tools/filesystem.py`)**:
  - `edit_file(path, old_text, new_text)` validates targeted text replacements, displays a colorized terminal unified diff preview, and executes atomic file updates.
- **Audit Logging**: All tool invocations and user approvals are recorded in `./.orbit/audit.log`.

### 6. Interactive REPL & Slash Commands (`repl.py`, `commands/`)
- Features 14 slash commands (`/help`, `/status`, `/model`, `/provider`, `/config`, `/tools`, `/audit`, `/steps`, `/reindex`, `/sources`, `/context`, `/compact`, `/clear`, `/exit`) and bare-word aliases (`/?`, `/reset`, `/quit`).
- Formats outputs with Rich terminal panels, syntax-highlighted code blocks, step execution indicators, and unified diff previews.

---

## 🛠️ Quickstart Usage

### 1. Interactive Setup
Configure provider, model, context size, and tool permissions:
```bash
python configure.py
```

### 2. Launching the Interactive REPL
Index the target repository path and start the Orbit REPL:
```bash
python repl.py ./
```

Launch with runtime overrides:
```bash
python repl.py ./ --provider anthropic --model claude-3-5-sonnet-20241022
python repl.py ./ --provider gemini --model gemini-2.5-flash
```

### 3. Programmatic Demo
Run the standalone streaming and multi-turn chat demo:
```bash
python orbit-core.py
```

### 4. Running Tests
Run the complete automated test suite (119 tests):
```bash
python -m unittest discover -s tests -p "test_*.py"
```

Run retrieval evaluation benchmarks:
```bash
python -m unittest tests/eval_retrieval.py
```

---

## 📌 Summary for RAG Context
Orbit combines a unified 7-provider LLM runtime with a Hybrid AST + BM25 RAG vector pipeline, multi-step ReAct agent loop, proactive context compaction, sub-agent delegation, and a risk-tiered tool registry. It delivers fast, secure, syntax-aware code reasoning and system automation for galactOS.
