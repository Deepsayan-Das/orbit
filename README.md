# Orbit

Orbit is the AI assistant layer for **GalactOS** — a provider-agnostic LLM
runtime plus a raw, from-scratch RAG (retrieval-augmented generation)
pipeline that lets Orbit answer questions grounded in a real codebase,
rather than guessing from a single prompt or the model's training data
alone.

No LangChain, no LlamaIndex — every piece (chunking, embedding, retrieval,
injection) is hand-built, so the mechanics are fully understood rather than
hidden behind a framework. See `docs/` (or the project's learning notes)
for the reasoning behind each design decision.

## What's in here

```
providers/
  base.py          # BaseLLMProvider (ABC), ChatMessage, LLMResponse, StreamChunk
  ollama_provider.py
  openai_provider.py
  huggingface_provider.py
  gemini_provider.py
  groq_provider.py
llm_client.py      # OrbitLLM — provider-agnostic facade/factory
chunker.py         # Multi-language chunking dispatcher (AST, tree-sitter, prose)
orbit_repl.py       # Indexing pipeline + interactive REPL with RAG
tests/
```

## Architecture

### 1. Provider abstraction (`providers/`, `llm_client.py`)

Every LLM backend (Ollama, OpenAI, HuggingFace, Gemini, Groq, and any future
provider) implements a single interface:

```python
class BaseLLMProvider(ABC):
    @abstractmethod
    def chat(self, messages: List[ChatMessage], stream: bool = False, **kwargs) \
            -> Union[LLMResponse, Iterator[StreamChunk]]: ...

    @abstractmethod
    def is_available(self) -> bool: ...
```

`OrbitLLM` is the facade application code actually talks to:

```python
from llm_client import OrbitLLM

llm = OrbitLLM(provider="ollama", model="llama3.2:latest")
response = llm.chat(messages="Hello, Orbit")
print(response.content)
```

Switching providers is a one-line change — no call-site changes needed:

```python
llm = OrbitLLM(provider="openai", model="gpt-4o-mini")
```

Custom providers can be registered at runtime:

```python
OrbitLLM.register_provider("custom", MyCustomProvider)
```

Input is normalized centrally (`normalize_messages`) so callers can pass a
bare string, a dict, a `ChatMessage`, or a list of any of those — every
provider always receives a clean `List[ChatMessage]`.

### 2. Chunking (`chunker.py`)

Chunking strategy is dispatched by file type, because a single universal
splitter produces meaningless boundaries (proven the hard way — naive
character-count and blank-line splitting both cut through the middle of
functions and statements):

| File type | Strategy |
|---|---|
| `.py` | Python `ast` — one chunk per top-level function/class |
| `.go`, `.js`, `.ts`, `.rs`, `.java`, `.c`, `.cpp`, ... | `tree-sitter` — same principle, language-aware grammar |
| `.md`, `.txt`, `.rst` | Heading-based (falls back to paragraph breaks) |
| anything else | Character-count sliding window (last resort, logged) |

Oversized classes/structs are split per-member, with the parent's name and
docstring prepended, so a retrieved method chunk is still self-contained
and knows what it belongs to.

Each indexed file also gets one **module-level summary** record (deterministic,
no LLM call — built from docstrings/comments + a structural listing of
top-level symbols). This exists specifically so "what does this file do"
style questions have something to retrieve — fine-grained chunks alone
can't answer whole-file questions, since that answer isn't localized to
any single chunk.

### 3. Retrieval + generation (`orbit_repl.py`)

```
index_directory(path)
    → chunk_file() per file (dispatched by extension)
    → embed() each chunk (Ollama nomic-embed-text)
    → in-memory records: {source, text, embedding, level}

retrieve(query, records, top_k=3)
    → embed the query
    → cosine similarity against every stored record
    → top-k highest-scoring chunks

ask_with_context(query, records)
    → build_context_prompt(query, top_chunks)
    → OrbitLLM.chat(...)
```

Retrieval is a plain Python list + linear scan — genuinely fine at the
scale of a single project's source (hundreds to low thousands of chunks).
It does **not** scale to large monorepos or company-wide search; a real
vector index (e.g. Chroma) would replace the linear scan at that point
without changing anything upstream of it.

## Usage

```bash
# Index and chat against a single file
python orbit_repl.py providers/base.py

# Index and chat against an entire directory
python orbit_repl.py ./
```

REPL commands:
- `quit` / `exit` — leave
- `reset` / `clear` — clear conversation memory (index stays loaded)

Conversation history stores only the clean question/answer pairs — the
retrieved context is injected fresh per turn and is not carried forward,
to avoid unbounded context growth across a long session.

## Configuration

| Env var | Used by |
|---|---|
| `OPENAI_API_KEY` | `OpenAIProvider` |
| `HF_TOKEN` | `HuggingFaceProvider` |
| `GEMINI_API_KEY` | `GeminiProvider` |

Set these in a local `.env` file (never commit it — see `.gitignore`).
Ollama requires no key but must be running locally with the relevant
models pulled (e.g. `ollama pull llama3.2`, `ollama pull nomic-embed-text`).

**Security note:** the indexer explicitly excludes `.env` and other
credential-shaped files/extensions from indexing — secrets must never end
up embedded in the retrieval index.

## Known limitations

- Retrieval is a linear scan (fine at project scale, not at scale-out).
- Grounding reduces but does not eliminate hallucination — a model can
  retrieve real symbol names correctly and still invent a plausible-looking
  but incorrect method signature or argument order around them. Larger
  models (e.g. `llama3.2` vs `qwen:0.5b`) noticeably improve faithfulness
  to retrieved context, but don't eliminate the issue.
- `is_available()` semantics differ slightly per provider (Ollama checks
  live server reachability for free; API-based providers trade off
  cost-per-check against certainty — see provider docstrings).
- Tree-sitter chunking has not been validated against every language in
  the support matrix on real-world files — treat exotic language edge
  cases as untested until exercised.