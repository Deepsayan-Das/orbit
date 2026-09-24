# Orbit

Orbit is the AI assistant layer for **GalactOS** — a provider-agnostic LLM
runtime plus a raw, from-scratch RAG (retrieval-augmented generation)
pipeline and tool registry system that lets Orbit answer questions grounded
in a real codebase and perform tool-assisted actions safely.

No heavy AI orchestration frameworks — every component (AST chunking, ChromaDB
vector indexing, retrieval, symbol typo correction, tool dispatching) is built directly so the mechanics are fully understood and maintainable. See [docs/ARCHITECTURE.md](file:///d:/web%20development%20projects/ongoing/orbit/docs/ARCHITECTURE.md)
for the complete design breakdown.

## What's in here

```text
providers/             # LLM Provider abstraction layer
  base.py              # BaseLLMProvider (ABC), ChatMessage, LLMResponse, StreamChunk
  ollama_provider.py   # Ollama local model integration & embeddings
  openai_provider.py   # OpenAI Cloud API
  huggingface_provider.py # HuggingFace Inference API
  gemini_provider.py   # Google Gemini API
  groq_provider.py     # Groq LPU API
llm_client.py          # OrbitLLM — provider-agnostic facade/factory
rag/                   # Retrieval-Augmented Generation subsystem
  chunker.py           # AST & Tree-sitter chunker + module summarizer
  indexer.py           # Persistent ChromaDB vector store & SHA-256 incremental hashing
  retrieval.py         # Vector similarity search & difflib AST symbol typo correction
tools/                 # Builtin Tool Registry & risk-level authorization system
  registry.py          # Tool schema generator, audit logger, and execution router
  filesystem.py        # list_directory (SAFE), write_file (DANGEROUS)
  read_file.py         # read_file (SAFE)
  shell.py             # run_shell_command (DANGEROUS)
  git_tools.py         # git_status, git_diff, git_log, git_blame (SAFE), git_commit, git_push (DANGEROUS)
  container_tools.py   # container_ps, container_logs (SAFE)
  dev_tools.py         # run_tests (SENSITIVE), search_logs (SAFE)
  code_search.py       # search_codebase (SAFE)
repl.py                # Interactive RAG REPL with tool execution and streaming
orbit_repl.py          # Legacy entrypoint stub delegating to repl.py
chunker.py             # Legacy import stub delegating to rag.chunker
tests/                 # Automated unit test suite
```

## Architecture

### 1. Provider abstraction (`providers/`, `llm_client.py`)

Every LLM backend (Ollama, OpenAI, HuggingFace, Gemini, Groq, and custom
registered providers) implements a unified interface:

```python
class BaseLLMProvider(ABC):
    @abstractmethod
    def chat(self, messages: List[ChatMessage], stream: bool = False, **kwargs) \
            -> Union[LLMResponse, Iterator[StreamChunk]]: ...

    @abstractmethod
    def is_available(self) -> bool: ...
```

`OrbitLLM` is the facade application code talks to:

```python
from llm_client import OrbitLLM

llm = OrbitLLM(provider="ollama", model="llama3.2:latest")
response = llm.chat(messages="Hello, Orbit")
print(response.content)
```

Switching providers is a simple one-line change — no call-site modifications needed:

```python
llm = OrbitLLM(provider="groq", model="llama-3.3-70b-versatile")
```

Custom providers can be registered at runtime:

```python
OrbitLLM.register_provider("custom", MyCustomProvider)
```

Input is normalized centrally (`normalize_messages`) so callers can pass a
bare string, a dict, a `ChatMessage`, or a list of any of those — every
provider always receives a clean `List[ChatMessage]`.

### 2. AST-Aware Chunking (`rag/chunker.py`)

Chunking strategy is dispatched by file type, avoiding destructive mid-statement cuts:

| File type | Strategy |
|---|---|
| `.py` | Python `ast` — one chunk per top-level function/class |
| `.go`, `.js`, `.ts`, `.rs`, `.java`, `.c`, `.cpp` | `tree-sitter` — language-aware grammar rules |
| `.md`, `.txt`, `.rst` | Heading-based (falls back to paragraph breaks) |
| anything else | Character-count sliding window (logged fallback) |

Oversized classes/structs are split per-member with parent header context prepended. Each file also receives a deterministic, LLM-free **module-level summary** record (docstrings/comments + structural symbol listing) so whole-file queries retrieve effectively.

### 3. Persistent Retrieval & Tool Registry (`rag/`, `tools/`, `repl.py`)

```text
index_directory(path)
    → chunk_file() per file (dispatched by extension)
    → compute SHA-256 content_hash (skip unchanged files)
    → embed() each chunk (Ollama nomic-embed-text)
    → store in persistent ChromaDB collection (./.orbit/chroma_data)

retrieve(query, collection, top_k=3)
    → embed query & vector search ChromaDB collection
    → if top retrieval score < 0.45: apply difflib typo correction on AST symbols

repl(agent, collection)
    → inject retrieved context per turn
    → provide tool schemas from tools/ registry to LLM decision call
    → execute requested tools (requesting confirmation for DANGEROUS/SENSITIVE operations)
    → stream final token response to stdout
```

## Usage

```bash
# Index current directory and start Orbit REPL
python repl.py ./

# Specify provider and model
python repl.py ./ --provider groq --model llama-3.3-70b-versatile
```

REPL commands:
- `quit` / `exit` — exit REPL
- `reset` / `clear` — clear conversation state (indexed collection remains persistent)

### Running Unit Tests

```bash
python -m unittest discover -s tests -p "test_*.py"
```

## Configuration

| Env var | Used by |
|---|---|
| `OPENAI_API_KEY` | `OpenAIProvider` |
| `HF_TOKEN` | `HuggingFaceProvider` |
| `GEMINI_API_KEY` | `GeminiProvider` |
| `GROQ_API_KEY` | `GroqProvider` |

Set these in a local `.env` file (never commit it — see `.gitignore`).
Ollama requires no key but must be running locally with the relevant
models pulled (e.g. `ollama pull llama3.2`, `ollama pull nomic-embed-text`).

**Security note:** the indexer explicitly excludes `.env` and other
credential-shaped files/extensions from indexing. Dangerous tool operations
(`write_file`, `run_shell_command`, `git_commit`, `git_push`) enforce explicit user confirmation prompts and audit logging (`.orbit/audit.log`).

## Known Limitations

- Retrieval relies on local Ollama service for embedding generation (`nomic-embed-text`).
- Grounding reduces but does not eliminate hallucination.
- Tree-sitter grammar support requires language pack availability.