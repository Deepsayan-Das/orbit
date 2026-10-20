# Orbit LLM Provider Guide (`ai-runtime`)

Orbit provides an extensible, provider-agnostic LLM abstraction layer (`BaseLLMProvider`) designed for seamless switching between local and cloud backends. It features unified function/tool calling, multi-turn message history, token-by-token streaming, and automated exponential backoff retries for transient service errors.

---

## 1. Supported Providers & Required Environment Variables

Configure environment variables in a `.env` file at the root of your project or in your system environment:

| Provider Key | Class | Default Model | Required Environment Variable(s) | Description |
| :--- | :--- | :--- | :--- | :--- |
| `"ollama"` | `OllamaProvider` | `llama3.2:latest` | *(None required)* | Local model inference via Ollama service (`http://localhost:11434`). Supports custom `num_ctx` window sizes. |
| `"openai"` | `OpenAIProvider` | `gpt-4o-mini` | `OPENAI_API_KEY` | OpenAI Cloud API models (GPT-4o, GPT-4o-mini, o1, o3-mini). |
| `"anthropic"` | `AnthropicProvider` | `claude-3-5-sonnet-20241022` | `ANTHROPIC_API_KEY` | Anthropic Claude models via official SDK (Claude 3.5 Sonnet, Claude 3.7 Sonnet, Claude 3.5 Haiku, Claude 3 Opus). |
| `"gemini"` | `GeminiProvider` | `gemini-2.5-flash` | `GEMINI_API_KEY` | Google Gemini models via official `google-genai` SDK with native tool call formatting. |
| `"mistral"` | `MistralProvider` | `mistral-small-latest` | `MISTRAL_API_KEY` | Mistral AI API models (Mistral Small, Mistral Large, Codestral). |
| `"groq"` | `GroqProvider` | `llama-3.3-70b-versatile` | `GROQ_API_KEY` | Ultra-fast Groq LPU inference via official SDK or OpenAI-compatible client. |
| `"huggingface"` | `HuggingFaceProvider` | `Qwen/Qwen2.5-Coder-7B-Instruct` | `HF_TOKEN` (or `HUGGINGFACE_API_KEY`) | Serverless Inference API models on HuggingFace Hub. |

### Example `.env` File

```env
# Cloud API keys
OPENAI_API_KEY=sk-proj-...
ANTHROPIC_API_KEY=sk-ant-...
GEMINI_API_KEY=AIzaSy...
MISTRAL_API_KEY=...
GROQ_API_KEY=gsk_...
HF_TOKEN=hf_...

# Optional provider overrides
# OLLAMA_HOST=http://localhost:11434
```

---

## 2. How to Switch Providers

Instantiate `OrbitLLM` passing the provider key string, or load settings directly from `~/.orbit/config.yaml`:

### Option A: Direct Instantiation via Key

```python
from llm_client import OrbitLLM

# 1. Local Ollama
llm_ollama = OrbitLLM(provider="ollama", model="llama3.2:latest")

# 2. OpenAI
llm_openai = OrbitLLM(provider="openai", model="gpt-4o-mini")

# 3. Anthropic Claude
llm_anthropic = OrbitLLM(provider="anthropic", model="claude-3-5-sonnet-20241022")

# 4. Google Gemini
llm_gemini = OrbitLLM(provider="gemini", model="gemini-2.5-flash")

# 5. Mistral AI
llm_mistral = OrbitLLM(provider="mistral", model="mistral-small-latest")

# 6. Groq LPU
llm_groq = OrbitLLM(provider="groq", model="llama-3.3-70b-versatile")

# 7. Hugging Face Serverless Inference
llm_hf = OrbitLLM(provider="huggingface", model="Qwen/Qwen2.5-Coder-7B-Instruct")
```

### Option B: Recommended Startup via Configuration (`config.yaml`)

```python
from config import load_config
from llm_client import OrbitLLM

# Merges provider selection, model choice, credentials, and context windows
cfg = load_config()
llm = OrbitLLM.from_config(cfg)
```

---

## 3. Streaming & Multi-Turn Generation

Orbit supports both token-by-token streaming and multi-turn message histories:

```python
from providers import ChatMessage
from llm_client import OrbitLLM

llm = OrbitLLM(provider="anthropic", model="claude-3-5-sonnet-20241022")

# 1. One-Shot Prompt Streaming
stream = llm.generate(prompt="Explain the GalactOS process manager", stream=True)
for chunk in stream:
    print(chunk.delta, end="", flush=True)

# 2. Multi-Turn Conversation
messages = [
    ChatMessage(role="system", content="You are Orbit, an AI assistant for GalactOS."),
    ChatMessage(role="user", content="What is Podman?"),
    ChatMessage(role="assistant", content="Podman is a daemonless container engine..."),
    ChatMessage(role="user", content="How do I view running containers with Nova?"),
]

response = llm.chat(messages=messages, stream=False)
print(response.content)
```

---

## 4. Unified Function / Tool Calling

All 7 providers normalize tool calling schemas. Pass standard tool schemas into `chat()` to receive parsed `ToolCall` objects:

```python
from llm_client import OrbitLLM
from tools import get_tools_schema

llm = OrbitLLM(provider="gemini", model="gemini-2.5-flash")

# Fetch registered tool schemas formatted for LLMs
tools_schema = get_tools_schema()

response = llm.chat(
    messages="List the files in the current directory",
    tools=tools_schema,
    stream=False,
)

if response.tool_calls:
    for call in response.tool_calls:
        print(f"Tool to execute: {call.name} with args {call.args}")
```

---

## 5. Resilience: 503 Exponential Backoff

`OrbitLLM.chat` automatically intercepts transient HTTP 503 / `Service Unavailable` / capacity overload exceptions from cloud APIs, backing off exponentially up to 3 retries before raising an exception:

```text
(!) 503 Service Unavailable (503 Service Unavailable). Retrying attempt 2/3 in 2.0s...
(!) 503 Service Unavailable (503 Service Unavailable). Retrying attempt 3/3 in 4.0s...
```

---

## 6. How to Register a Custom Provider

Extend `BaseLLMProvider` and register it dynamically via `OrbitLLM.register_provider()`:

```python
from typing import Any, Iterator, List, Union
from providers import BaseLLMProvider, ChatMessage, LLMResponse, StreamChunk
from llm_client import OrbitLLM


class CustomLLMProvider(BaseLLMProvider):
    def __init__(self, model: str = "custom-v1", api_url: str = "https://api.custom.ai"):
        self.model = model
        self.api_url = api_url

    def chat(
        self,
        messages: List[ChatMessage],
        stream: bool = False,
        **kwargs: Any
    ) -> Union[LLMResponse, Iterator[StreamChunk]]:
        if stream:
            yield StreamChunk(delta="Hello from custom provider!")
        else:
            return LLMResponse(content="Response from custom provider", model=self.model)

    def is_available(self) -> bool:
        return True


# Register custom provider
OrbitLLM.register_provider("custom_ai", CustomLLMProvider)

# Use custom provider seamlessly
llm = OrbitLLM(provider="custom_ai", model="custom-v1")
response = llm.generate(prompt="Hello Orbit")
print(response.content)
```
