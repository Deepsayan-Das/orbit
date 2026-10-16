from __future__ import annotations

from typing import Any, Dict, Iterator, List, Optional, Type, Union, TYPE_CHECKING

if TYPE_CHECKING:
    from config import OrbitConfig

from providers import (
    AnthropicProvider,
    BaseLLMProvider,
    ChatMessage,
    GeminiProvider,
    GroqProvider,
    HuggingFaceProvider,
    LLMResponse,
    MistralProvider,
    OllamaProvider,
    OpenAIProvider,
    StreamChunk,
    normalize_messages,
)


import time


def is_transient_503(exception: Exception) -> bool:
    """Check if exception is a 503 / Service Unavailable / overloaded error."""
    err_str = str(exception).lower()
    if any(keyword in err_str for keyword in ("503", "unavailable", "overloaded", "resource_exhausted", "service unavailable", "capacity")):
        return True
    status_code = getattr(exception, "status_code", None) or getattr(exception, "code", None)
    if status_code in (503, "503"):
        return True
    return False


class OrbitLLM:
    """
    Unified LLM Client facade for Orbit.
    Supports multi-turn message histories, token-by-token streaming, and provider switching.
    """

    _PROVIDER_REGISTRY: Dict[str, Type[BaseLLMProvider]] = {
        "ollama": OllamaProvider,
        "openai": OpenAIProvider,
        "huggingface": HuggingFaceProvider,
        "gemini": GeminiProvider,
        "groq": GroqProvider,
        "anthropic": AnthropicProvider,
        "mistral": MistralProvider,
    }

    MAX_OLLAMA_DEFAULT_NUM_CTX: int = 16384

    def __init__(
        self, 
        provider: Union[str, BaseLLMProvider] = "ollama", 
        model: Optional[str] = None,
        **provider_kwargs: Any
    ):
        from context.window import resolve_window
        num_ctx = provider_kwargs.pop("num_ctx", None)
        self.generation: Dict[str, Any] = provider_kwargs.pop("generation", {}) or {}

        if isinstance(provider, BaseLLMProvider):
            self.provider = provider
            prov_name = getattr(provider, "__class__", {}).__name__.lower().replace("provider", "")
            model_name = getattr(provider, "model", getattr(provider, "model_name", "unknown"))
            w_size, w_source = resolve_window(prov_name, model_name)
            self.window_size = num_ctx if num_ctx is not None else w_size
            self.window_source = "user" if num_ctx is not None else w_source
        elif isinstance(provider, str):
            provider_key = provider.lower()
            if provider_key not in self._PROVIDER_REGISTRY:
                available = ", ".join(self._PROVIDER_REGISTRY.keys())
                raise ValueError(f"Unknown provider '{provider}'. Available providers: {available}")
            
            provider_cls = self._PROVIDER_REGISTRY[provider_key]
            if model:
                provider_kwargs["model"] = model
            w_size, w_source = resolve_window(provider_key, model or "")
            if provider_key == "ollama" and num_ctx is None:
                w_size = min(w_size, self.MAX_OLLAMA_DEFAULT_NUM_CTX)
            self.window_size = num_ctx if num_ctx is not None else w_size
            self.window_source = "user" if num_ctx is not None else w_source
            if provider_key == "ollama":
                provider_kwargs["num_ctx"] = self.window_size
            self.provider = provider_cls(**provider_kwargs)
        else:
            raise TypeError("provider must be a string key or BaseLLMProvider instance")

    @classmethod
    def from_config(cls, cfg: "OrbitConfig") -> "OrbitLLM":
        """
        Construct an OrbitLLM from a loaded OrbitConfig.

        Merges per-provider settings (e.g. api_key) from cfg.providers[<name>]
        into the provider constructor kwargs.  This is the recommended startup
        path: ``OrbitLLM.from_config(load_config())``.
        """
        from context.window import resolve_window
        provider_key = cfg.provider.lower()
        provider_kwargs: Dict[str, Any] = {}

        # Merge per-provider overrides (api_key, host, etc.)
        per_provider = cfg.providers.get(provider_key, {})
        if isinstance(per_provider, dict):
            provider_kwargs.update(per_provider)

        win_size, win_source = resolve_window(provider_key, cfg.model, cfg)
        if provider_key == "ollama" and win_source != "user":
            win_size = min(win_size, cls.MAX_OLLAMA_DEFAULT_NUM_CTX)

        instance = cls(
            provider=provider_key, 
            model=cfg.model, 
            num_ctx=win_size, 
            generation=cfg.generation,
            **provider_kwargs
        )
        instance.window_size = win_size
        instance.window_source = win_source
        instance.generation = cfg.generation or {}
        return instance

    @classmethod
    def register_provider(cls, name: str, provider_cls: Type[BaseLLMProvider]) -> None:
        """Register a custom provider class dynamically."""
        cls._PROVIDER_REGISTRY[name.lower()] = provider_cls

    def chat(
        self,
        messages: Union[str, ChatMessage, Dict[str, str], List[Union[ChatMessage, Dict[str, str]]]],
        system_prompt: Optional[str] = None,
        stream: bool = False,
        **kwargs: Any
    ) -> Union[LLMResponse, Iterator[StreamChunk]]:
        """
        Process multi-turn messages or single prompt strings with automatic 503 retries.
        Incorporates user configured generation options (temperature, max_tokens).
        """
        normalized = normalize_messages(messages, system_prompt=system_prompt)
        call_kwargs = dict(self.generation or {})
        call_kwargs.update(kwargs)

        max_retries = 3
        for attempt in range(1, max_retries + 1):
            try:
                return self.provider.chat(messages=normalized, stream=stream, **call_kwargs)
            except Exception as e:
                if is_transient_503(e) and attempt < max_retries:
                    wait_time = attempt * 2.0
                    try:
                        from ui import print_warning
                        print_warning(f"503 Service Unavailable ({e}). Retrying attempt {attempt + 1}/{max_retries} in {wait_time:.1f}s...")
                    except Exception:
                        print(f"[orbit: 503 API error ({e}). Retrying attempt {attempt + 1}/{max_retries} in {wait_time:.1f}s...]")
                    time.sleep(wait_time)
                else:
                    raise

    def generate(
        self, 
        prompt: str, 
        system_prompt: Optional[str] = None, 
        stream: bool = False,
        **kwargs: Any
    ) -> Union[LLMResponse, Iterator[StreamChunk]]:
        """Convenience alias for single-prompt generation using chat under the hood."""
        return self.chat(messages=prompt, system_prompt=system_prompt, stream=stream, **kwargs)

    def is_available(self) -> bool:
        """Check if the active provider is available."""
        return self.provider.is_available()
