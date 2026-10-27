# Orbit

[![CI](https://github.com/Deepsayan-Das/orbit/actions/workflows/ci.yml/badge.svg)](https://github.com/Deepsayan-Das/orbit/actions/workflows/ci.yml)

Orbit is the AI assistant layer for **GalactOS** — a provider-agnostic LLM runtime coupled with a high-performance Hybrid RAG (retrieval-augmented generation) pipeline, multi-step ReAct agentic loop, automated context compaction engine, isolated sub-agent delegation, and a risk-tiered tool registry.

Orbit works natively with GalactOS and its companion tool **Nova** (`nova container`, `nova project doctor`, `nova cluster`, `nova system`), featuring Podman-native container execution and shell safety allowlisting.

Built from scratch without bloated orchestration frameworks — every component (AST & Tree-Sitter chunking, BM25 + dense vector hybrid retrieval, ChromaDB persistent indexing, symbol typo correction, multi-provider tool calling, context compaction, and sub-agents) is implemented directly for complete transparency, speed, and maintainability. See [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md) for detailed architectural specifications.

---

## What's in Here

```text
providers/                 # Multi-provider LLM abstraction layer
  base.py                  # BaseLLMProvider (ABC), ChatMessage, LLMResponse, StreamChunk, ToolCall
  anthropic_provider.py    # Anthropic Claude API (tool calling & streaming)
  gemini_provider.py       # Google Gemini API via google-genai SDK
  groq_provider.py         # Groq LPU API (ultra-fast inference)
  huggingface_provider.py  # HuggingFace Serverless Inference API
  mistral_provider.py      # Mistral AI API (Mistral Small/Large, Codestral)
  ollama_provider.py       # Local Ollama integration (custom num_ctx & GPU offloading)
  openai_provider.py       # OpenAI Cloud API (GPT-4o, o-series)
llm_client.py              # OrbitLLM unified facade with 503 exponential backoff retries
config.py                  # OrbitConfig dataclass, context window maps & ~/.orbit/config.yaml
configure.py               # Interactive CLI configuration wizard (survey-based)
agent_loop.py              # ReAct multi-step agentic tool-chaining loop with streaming
agent_sub.py               # Isolated sub-agent execution engine & delegate_task tool
context/                   # Context Window & Compaction Subsystem
  window.py                # Context window resolver (40+ models, user config, 4096 fallback)
  measure.py               # Token estimator (reported usage or chars/4) & 75% trigger
ui.py                      # Rich terminal UI, markdown rendering, and diff colorization
commands/                  # Slash command subsystem
  session.py               # Session dataclass (agent, config, history, REPL state)
  registry.py              # Command registry, @command decorator, exception-safe dispatch
  builtin.py               # Handlers for 14 slash commands (/help, /status, /model, etc.)
  compaction.py            # Context compaction engine (pairing, stubbing, LLM summarization)
  tokens.py                # Token estimation utility delegate
rag/                       # Hybrid Retrieval-Augmented Generation subsystem
  chunker.py               # AST (Python) & Tree-Sitter (Go, JS/TS, Rust, Java, C/C++) chunker
  bm25.py                  # Sparse BM25 keyword index & Reciprocal Rank Fusion (RRF)
  indexer.py               # Persistent ChromaDB vector store with SHA-256 incremental hashing
  retrieval.py             # Hybrid dense+sparse retrieval, nomic embeddings, AST typo correction
tools/                     # Builtin Tool Registry & risk-level authorization system
  registry.py              # Tool schema generator, audit logger, output capping, execution router
  diff_utils.py            # Unified diff generator & atomic validate-and-replace algorithm
  filesystem.py            # list_directory (SAFE), write_file (DANGEROUS), edit_file (DANGEROUS)
  read_file.py             # read_file (SAFE)
  shell.py                 # run_shell_command (DANGEROUS, with Nova CLI safe prefixes)
  git_tools.py             # git_status, git_diff, git_log, git_blame (SAFE), git_commit, git_push (DANGEROUS)
  container_tools.py       # container_ps, container_logs (SAFE, Podman-native)
  dev_tools.py             # run_tests (SENSITIVE), search_logs (SAFE)
  code_search.py           # search_codebase (SAFE)
repl.py                    # Interactive RAG REPL with agentic loop and slash commands
orbit_repl.py              # Legacy backwards-compatibility entrypoint stub
orbit-core.py              # Minimal programmatic streaming and multi-turn chat demo
tests/                     # Automated unit & evaluation test suite (119 tests)
```

---

## Core Capabilities

### 1. Multi-Provider LLM Abstraction (`providers/`, `llm_client.py`)
Orbit normalizes 7 LLM backends behind `BaseLLMProvider` and the `OrbitLLM` facade:
- **Local & Cloud**: Ollama, OpenAI, Anthropic, Google Gemini, Mistral, Groq, Hugging Face.
- **Unified Function Calling**: Standardized `ToolCall` and schema translation across all providers.
- **Resilience**: Built-in exponential backoff retry handler for transient HTTP 503 / capacity errors.
- **Token Streaming**: Real-time token streaming to stdout for low latency interaction.

```python
from config import load_config
from llm_client import OrbitLLM

cfg = load_config()
llm = OrbitLLM.from_config(cfg)
response = llm.chat("Explain the GalactOS process scheduler", stream=True)
for chunk in response:
    print(chunk.delta, end="", flush=True)
```

### 2. Hybrid RAG Pipeline (`rag/`)
Orbit features a hybrid retrieval engine combining dense semantics with exact-symbol keyword search:
- **Language-Aware AST Chunking (`rag/chunker.py`)**:
  - Python parsed via standard library `ast` into functions, classes, and methods with docstrings.
  - Go, JavaScript, TypeScript, Rust, Java, C, and C++ parsed via `tree-sitter`.
  - Markdown/prose split on semantic headings and paragraphs.
  - Deterministic module-level summary generation for top-level repo indexing.
- **Persistent ChromaDB & Incremental Hashing (`rag/indexer.py`)**:
  - Local persistent vector storage in `./.orbit/chroma_data`.
  - SHA-256 content hashing (`content_hash`) skips unchanged files to eliminate redundant re-embedding.
- **Multi-Provider Embeddings & Degraded Fallback (`rag/retrieval.py`)**:
  - Supports embedding models across **Ollama** (`nomic-embed-text`), **OpenAI** (`text-embedding-3-small`), **Google Gemini** (`gemini-embedding-2`), **Mistral AI** (`mistral-embed`), and **Hugging Face** (`sentence-transformers/all-MiniLM-L6-v2`). Note: Google shut down `text-embedding-004` on January 14, 2026; Orbit defaults to `gemini-embedding-2`.
  - **Vector Dimension Mismatch Auto-Rebuild (`rag/indexer.py`)**: Stores `embedding_model_tag` (`<provider>:<model>`) in collection metadata. If a user switches embedding models (e.g. 768-dim `nomic-embed-text` to 3072-dim `gemini-embedding-2`), Orbit automatically detects the change, resets the collection, and rebuilds the vector index to prevent dimension mismatch exceptions.
  - **Anthropic Embedding Guidance**: Anthropic does not provide an embedding model API. Users whose primary LLM is Anthropic Claude configure a secondary provider (OpenAI, Gemini, Mistral, HuggingFace) or local Ollama for embeddings.
  - **Degraded BM25 Keyword-Only Fallback**: If no embedding provider is available (no cloud keys and Ollama is not installed/running, such as on a fresh OS ISO), Orbit does not crash. Instead, it enters a degraded **keyword-only BM25 mode** with an informative warning, ensuring full text search, AST symbol lookup, and agent operations remain fully functional offline.
- **Hybrid Search via Reciprocal Rank Fusion (`rag/bm25.py`, `rag/retrieval.py`)**:
  - Dense vector retrieval combined with sparse BM25 keyword matching for exact symbol and identifier precision.
  - Results merged via Reciprocal Rank Fusion (RRF) with relevance floor (`min_score=0.35`) and per-file deduplication (`max_per_file=2`).
- **Cached AST Symbol Typo Correction**:
  - Extracts and caches known code symbols in `./.orbit/symbols.json` during indexing.
  - If vector retrieval score is low ($< 0.45$), applies `difflib` fuzzy matching against cached AST symbols without expensive collection-wide scans.
- **Evaluation Suite (`tests/eval_retrieval.py`)**:
  - 30 curated golden query-symbol pairs testing Recall@k and Mean Reciprocal Rank (MRR).

### 3. ReAct Multi-Step Agentic Loop (`agent_loop.py`)
- **Iterative Tool Chaining**: Sends message history and tool schemas to the model; executes requested tools via the permission-gated registry and loops back.
- **Configurable Step Ceiling**: Defaults to 8 steps (adjustable from 1 to 20 via `/steps <n>`) with loop-prevention and graceful degradation (provides summary when step limit is hit).
- **Proactive Context Compaction**: Automatically triggers context compaction during the turn if estimated token usage reaches $\ge 75\%$ or reserve drops below 1,000 tokens.

### 4. Context Management & Intelligent Compaction (`context/`, `commands/compaction.py`)
- **Window Resolver (`context/window.py`)**: Lookups for 40+ models across OpenAI, Anthropic, Gemini, LLaMA 3, Mistral, DeepSeek, and Qwen, with user config overrides and safe 4096 fallback.
- **Token Estimator (`context/measure.py`)**: Accurate tracking using provider-reported usage or character ratio estimation.
- **Compaction Engine (`commands/compaction.py`)**:
  1. *Never-touch rule*: Preserves system prompt and the initial user request.
  2. *Recent window*: Preserves the most recent 6 messages intact.
  3. *Atomic Tool Pairing*: Keeps tool calls and results strictly paired.
  4. *Outcome Stubbing*: Condenses historical tool outputs (`read_file`, `edit_file`, `run_tests`, `run_shell_command`) into concise result stubs.
  5. *LLM Summarization & Fallback*: Condenses middle messages using the active LLM, falling back to dropping the oldest tool pairs if needed.

### 5. Risk-Tiered Tool Registry & Surgical Editing (`tools/`)
Tools enforce clear security boundaries and audit tracking:
- **Risk Tiers**:
  - `SAFE`: Automatically executes without prompting (`list_directory`, `read_file`, `git_status`, `git_diff`, `git_log`, `git_blame`, `container_ps`, `container_logs`, `search_logs`, `search_codebase`).
  - `SENSITIVE`: Prompts user before execution (`run_tests`).
  - `DANGEROUS`: Requires explicit confirmation with full preview (`write_file`, `edit_file`, `run_shell_command`, `git_commit`, `git_push`).
- **Surgical File Editing (`tools/diff_utils.py`, `tools/filesystem.py`)**:
  - `edit_file(path, old_text, new_text)` executes exact replacements with whitespace-forgiving fallback and displays an interactive terminal unified diff preview before confirmation.
- **Audit Logging**: All tool executions and permission approvals are appended to `./.orbit/audit.log`.
- **Output Capping**: Tool outputs are capped at 100 lines to prevent context blowing.

### 6. Sub-Agent Delegation (`agent_sub.py`)
- Tasks can be partitioned to isolated sub-agents via `delegate_task(task)`.
- Runs with an isolated message history, restricts access to `SAFE` tools only, enforces a recursion depth ceiling of 1, and tags audit log events with unique sub-agent IDs.

### 7. GalactOS & Nova Companion Integration
- **Container Management**: Podman-native execution (`podman ps`, `nova container ps`).
- **Safe Command Allowlist**: Recognizes read-only commands and GalactOS Nova commands (`nova version`, `nova system`, `nova project doctor`, `nova container ps`) as safe.

---

## REPL Slash Commands

Any command prefixed with `/` (or supported bare-word aliases) is intercepted by the command dispatcher:

| Command | Aliases | Description |
|---|---|---|
| `/help` | `/?` | List all registered REPL slash commands and aliases |
| `/status` | | Display active provider, model, context window size, token usage, max steps, and chunk count |
| `/model [name]` | | Show or switch the active LLM model at runtime (verifies provider availability) |
| `/provider [name]` | | Show or switch the active LLM provider at runtime |
| `/config` | | Display active configuration with API keys masked (`****cdef`) |
| `/tools` | | List all registered tools and their associated risk tiers |
| `/audit [n]` | | View the last `n` entries from `./.orbit/audit.log` (default 10) |
| `/steps [n]` | | Show or set the `max_steps` tool chaining limit (1 to 20) |
| `/reindex [path]` | | Re-index workspace path into the ChromaDB vector store |
| `/sources [on\|off]` | | Toggle displaying retrieved RAG context sources for each query |
| `/context` | | Show history message count, estimated token usage, and context window source |
| `/compact [focus]` | | Trigger manual context compaction with optional focus prompt |
| `/clear` | `/reset` | Clear current conversation history |
| `/exit` | `/quit` | Exit the Orbit REPL |

---

## Quickstart & Usage

### 1. Interactive Configuration Wizard
Run the interactive setup wizard to configure your default LLM provider, chat model, embedding provider, context window, and tool permissions:
```bash
python configure.py
```

### 2. Launching the Orbit REPL
Index the current project directory and enter the interactive REPL:
```bash
python repl.py ./
```

You can override LLM or embedding parameters on launch:
```bash
python repl.py ./ --provider anthropic --model claude-sonnet-5-5
python repl.py ./ --provider gemini --model gemini-2.5-flash
python repl.py ./ --embedding-provider openai --embedding-model text-embedding-3-small
python repl.py ./ --provider ollama --model llama3.2:latest
```

### 3. Programmatic Demo
Run the programmatic streaming and multi-turn chat demonstration script:
```bash
python orbit-core.py
```

---

## Running Automated Tests

Orbit includes an automated test suite of **127 unit and integration tests** covering all providers, embedding fallbacks, RAG pipelines, chunkers, tools, context compaction, and sub-agents:

```bash
python -m unittest discover -s tests -p "test_*.py"
```

To run retrieval evaluation benchmarks:
```bash
python -m unittest tests/eval_retrieval.py
```

---

## Configuration & Environment Variables

Orbit stores configuration in `~/.orbit/config.yaml`. Provider API keys can be defined in `~/.orbit/config.yaml` or exported in your environment / `.env`:

| Provider | Key | Environment Variable | Default Model |
|---|---|---|---|
| Ollama | `ollama` | *(None required - local)* | `llama3.2:latest` |
| OpenAI | `openai` | `OPENAI_API_KEY` | `gpt-4o-mini` |
| Anthropic | `anthropic` | `ANTHROPIC_API_KEY` | `claude-sonnet-5-5` |
| Google Gemini | `gemini` | `GEMINI_API_KEY` | `gemini-2.5-flash` |
| Mistral AI | `mistral` | `MISTRAL_API_KEY` | `mistral-small-latest` |
| Groq | `groq` | `GROQ_API_KEY` | `llama-3.3-70b-versatile` |
| HuggingFace | `huggingface` | `HF_TOKEN` (or `HUGGINGFACE_API_KEY`) | `Qwen/Qwen2.5-Coder-7B-Instruct` |

> **Security Note:** Sensitive files (`.env`, credentials, `.git`) are automatically excluded from indexing. All destructive or sensitive operations require interactive confirmation and are recorded in `./.orbit/audit.log`.