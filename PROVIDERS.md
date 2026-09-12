# Orbit LLM Provider Guide (`ai-runtime`)

Orbit provides an extensible provider abstraction layer (`BaseLLMProvider`) designed for seamless switching between local and cloud LLM backends, complete with multi-turn message history and real-time token streaming.

---

## 1. Supported Providers & Required Environment Variables

Configure environment variables in a `.env` file at the root of your project:

| Provider Key | Class | Default Model | Required Environment Variable(s) | Description |
| :--- | :--- | :--- | :--- | :--- |
| `"ollama"` | `OllamaProvider` | `qwen:0.5b` | *(None required)* | Local model inference via Ollama service (`http://localhost:11434`). |
| `"openai"` | `OpenAIProvider` | `gpt-4o-mini` | `OPENAI_API_KEY` | OpenAI Cloud API models. |
| `"huggingface"` | `HuggingFaceProvider` | `Qwen/Qwen2.5-7B-Instruct` | `HF_TOKEN` (or `HUGGINGFACE_API_KEY`) | Free-tier accessible Serverless Inference API models on HuggingFace Hub. |
| `"gemini"` | `GeminiProvider` | `gemini-2.5-flash` | `GEMINI_API_KEY` | Google Gemini models via official `google-genai` SDK. |

### Example `.env` File
```env
OPENAI_API_KEY=sk-proj-...
HF_TOKEN=hf_...
GEMINI_API_KEY=AIzaSy...
```

---

## 2. How to Switch Providers

Instantiate `OrbitLLM` passing the provider key string (or passing custom parameters):

```python
from llm_client import OrbitLLM

# 1. Use local Ollama
llm_ollama = OrbitLLM(provider="ollama", model="qwen:0.5b")

# 2. Use Hugging Face Serverless Inference API
llm_hf = OrbitLLM(provider="huggingface", model="Qwen/Qwen2.5-7B-Instruct")

# 3. Use Google Gemini API
llm_gemini = OrbitLLM(provider="gemini", model="gemini-2.5-flash")

# 4. Use OpenAI API
llm_openai = OrbitLLM(provider="openai", model="gpt-4o-mini")

# Generate completion with streaming (token-by-token)
stream = llm_gemini.generate(prompt="Explain galactOS architecture", stream=True)
for chunk in stream:
    print(chunk.delta, end="", flush=True)
```

---

## 3. How to Register a Fully Custom Provider

You can implement a custom provider by subclassing `BaseLLMProvider` and registering it dynamically via `OrbitLLM.register_provider()`:

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
        formatted_messages = [msg.to_dict() for msg in messages]
        
        if stream:
            # Yield StreamChunk(delta="...") objects
            yield StreamChunk(delta="Hello from custom provider!")
        else:
            # Return LLMResponse object
            return LLMResponse(content="Response from custom provider", model=self.model)

    def is_available(self) -> bool:
        # Perform lightweight connectivity / authorization check
        return True


# Register the custom provider under a new key name
OrbitLLM.register_provider("custom_ai", CustomLLMProvider)

# Instantiate and use custom provider seamlessly
llm = OrbitLLM(provider="custom_ai", model="custom-v1")
response = llm.generate(prompt="Hello Orbit")
print(response)
```
