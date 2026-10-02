# Orbit

Orbit is the AI assistant layer for **GalactOS** — a provider-agnostic LLM
runtime plus a raw, from-scratch RAG (retrieval-augmented generation)
pipeline, multi-step agentic loop, context compaction subsystem, sub-agent delegation engine, and tool registry system that lets Orbit answer questions grounded
in a real codebase and perform tool-assisted actions safely.

Orbit works natively with GalactOS and its companion tool **Nova** (`nova container`, `nova project doctor`, `nova system`), with Podman-native container execution.

No heavy AI orchestration frameworks — every component (AST chunking, ChromaDB
vector indexing, retrieval, symbol typo correction, multi-step tool chaining, slash commands, context compaction, sub-agents) is built directly so the mechanics are fully understood and maintainable. See [docs/ARCHITECTURE.md](file:///d:/web%20development%20projects/ongoing/orbit/docs/ARCHITECTURE.md)
for the complete design breakdown.

## What's in here

```text
providers/             # LLM Provider abstraction layer
  base.py              # BaseLLMProvider (ABC), ChatMessage, LLMResponse, StreamChunk
  ollama_provider.py   # Ollama local model integration (with num_ctx context window options)
  openai_provider.py   # OpenAI Cloud API
  huggingface_provider.py # HuggingFace Inference API
  gemini_provider.py   # Google Gemini API
  groq_provider.py     # Groq LPU API
llm_client.py          # OrbitLLM — provider-agnostic facade/factory
config.py              # OrbitConfig dataclass (with context_windows model map) & ~/.orbit/config.yaml
configure.py           # Interactive configuration wizard (with Ollama num_ctx VRAM prompts)
agent_loop.py          # ReAct multi-step agentic tool chaining orchestrator with streaming
agent_sub.py           # Isolated sub-agent execution engine & delegate_task tool
context/               # Context Management Subsystem
  window.py            # Window resolver (model context table lookup, user config, 4096 fallback)
  measure.py           # Token estimator (reported LLMResponse usage or chars/4) & 75%/reserve trigger
ui.py                  # Rich terminal UI, markdown rendering, and diff colorization
commands/              # Slash command system
  session.py           # Session dataclass holding agent, config, history, and REPL state
  registry.py          # Command registry, @command decorator, and exception-safe dispatch
  builtin.py           # Handlers for /help, /status, /model, /provider, /config, /tools, etc.
  compaction.py        # Compaction engine (pairing rule, stubbing, LLM summary, fallback)
  tokens.py            # Token estimation utility delegate
rag/                   # Retrieval-Augmented Generation subsystem
  chunker.py           # AST & Tree-sitter chunker + module summarizer
  indexer.py           # Persistent ChromaDB vector store & SHA-256 incremental hashing
  retrieval.py         # Vector similarity search & difflib AST symbol typo correction
tools/                 # Builtin Tool Registry & risk-level authorization system
  registry.py          # Tool schema generator, audit logger, output capping (100 lines), router
  filesystem.py        # list_directory (SAFE), write_file (DANGEROUS), edit_file (DANGEROUS with unified diffs)
  read_file.py         # read_file (SAFE)
  shell.py             # run_shell_command (DANGEROUS, with Nova CLI safe prefixes)
  git_tools.py         # git_status, git_diff, git_log, git_blame (SAFE), git_commit, git_push (DANGEROUS)
  container_tools.py   # container_ps, container_logs (SAFE, Podman-native)
  dev_tools.py         # run_tests (SENSITIVE), search_logs (SAFE)
  code_search.py       # search_codebase (SAFE)
repl.py                # Interactive RAG REPL with agentic loop and slash commands
tests/                 # Automated unit test suite (80 unit tests)
```

## Architecture

### 1. Provider Abstraction & Config (`providers/`, `llm_client.py`, `config.py`)

Every LLM backend (Ollama, OpenAI, HuggingFace, Gemini, Groq) implements a unified `BaseLLMProvider` interface.

Configuration is persisted to `~/.orbit/config.yaml` via `config.py` and can be set interactively using `python configure.py`.

```python
from config import load_config
from llm_client import OrbitLLM

cfg = load_config()
llm = OrbitLLM.from_config(cfg)
response = llm.chat(messages="Hello, Orbit")
print(response.content)
```

### 2. Context Management & Compaction (`context/`, `commands/compaction.py`)

Orbit features automated context window management:
- **Window Resolver** (`context/window.py`): Resolves maximum tokens per model (user config $\rightarrow$ model table $\rightarrow$ 4096 fallback).
- **Measurement & Trigger** (`context/measure.py`): Uses provider-reported token usage or character length estimation; triggers compaction when token usage reaches $\ge 75\%$ or remaining tokens $< 1000$.
- **Compaction Engine** (`commands/compaction.py`):
  1. *Never-touch rule*: System prompt and 1st user query are preserved.
  2. *Recent window*: Keeps most recent 6 messages untouched.
  3. *Atomic Tool Pairing*: Tool call assistant messages and tool result messages are paired atomically (never orphaned).
  4. *Outcome Stubbing*: Replaces old tool outputs with short outcome summaries (`read_file`, `edit_file`, `run_tests`, `run_shell_command`).
  5. *LLM Summarization & Fallback*: Summarizes middle history; falls back to oldest tool pair dropping if summary fails or remains over limit.

### 3. Sub-Agents (`agent_sub.py`)

Tasks can be delegated to isolated sub-agents via the `delegate_task(task)` tool:
- **Isolated History**: Runs in a fresh, isolated message history.
- **Tool Restriction**: Restricted to `SAFE` risk tier tools by default.
- **Depth Limit 1**: Sub-agents cannot spawn nested sub-agents.
- **Audit Logging**: Logs unique `agent_id` in `.orbit/audit.log`.

### 4. Multi-Step Agentic Loop & Real-Time Streaming (`agent_loop.py`)

Orbit runs a multi-step tool-chaining cycle (`run_agent_turn`):
1. Sends conversation history + tool schemas to the active LLM.
2. If the model emits tool calls, executes them via `tools.execute_tool` (enforcing risk tier permissions and 100-line output caps), appends results, and loops back.
3. If the model outputs plain text, renders final answer tokens directly to standard output.
4. Enforces a `max_steps` ceiling (default 8) to prevent infinite loops.

### 5. GalactOS & Nova Companion Tool Integration

Orbit is built for **GalactOS** and integrates with **Nova** CLI utilities:
- **Container Management**: Podman-native execution (`podman ps`, `nova container ps`).
- **Command Safety Allowlist**: Recognizes `nova version`, `nova system`, `nova project doctor`, and `nova container ps` as safe read-only operations.

---

## REPL Slash Commands

Any line starting with `/` (or standard bare-word aliases) is intercepted by the command dispatcher:

| Command | Description |
|---|---|
| `/help` (or `?`) | List all registered REPL slash commands and aliases |
| `/status` | Display provider, model, context window size (with source), token usage, max steps, and chunk count |
| `/model [name]` | Show or switch current model at runtime (verifies provider availability) |
| `/provider [name]` | Show or switch current provider at runtime |
| `/config` | Display active configuration with API keys masked (`****cdef`) |
| `/tools` | List registered tools and their risk tiers |
| `/audit [n]` | View last `n` entries from `.orbit/audit.log` (default 10) |
| `/steps [n]` | Show or set `max_steps` tool execution limit (1 to 20) |
| `/reindex [path]` | Re-index workspace target path into vector store |
| `/sources [on\|off]` | Toggle printing retrieved RAG context sources for each query |
| `/context` | Show message history count, token usage, window size, and source |
| `/compact [focus]` | Trigger intelligent context history compaction |
| `/clear` (alias `/reset`) | Clear conversation history |
| `/exit` (alias `/quit`) | Exit the Orbit REPL |

---

## Usage

```bash
# Interactively configure provider, model, context size, and tool defaults
python configure.py

# Index current directory and start Orbit REPL
python repl.py ./

# Specify provider and model via CLI override
python repl.py ./ --provider gemini --model gemini-2.0-flash
```

### Running Unit Tests

```bash
python -m unittest discover -s tests -p "test_*.py"
```

## Configuration & Security

Set provider API keys in environment variables or `~/.orbit/config.yaml`:
- `GEMINI_API_KEY`
- `OPENAI_API_KEY`
- `GROQ_API_KEY`
- `HF_TOKEN`

**Security note:** `.env` and credential files are excluded from indexing. Sensitive and dangerous operations enforce explicit confirmation prompts and append log entries to `.orbit/audit.log`.