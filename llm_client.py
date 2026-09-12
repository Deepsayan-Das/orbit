from typing import Any, Dict, Iterator, List, Optional, Type, Union

from providers import (
    BaseLLMProvider,
    ChatMessage,
    GeminiProvider,
    GroqProvider,
    HuggingFaceProvider,
    LLMResponse,
    OllamaProvider,
    OpenAIProvider,
    StreamChunk,
    normalize_messages,
)


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
    }

    def __init__(
        self, 
        provider: Union[str, BaseLLMProvider] = "ollama", 
        model: Optional[str] = None,
        **provider_kwargs: Any
    ):
        if isinstance(provider, BaseLLMProvider):
            self.provider = provider
        elif isinstance(provider, str):
            provider_key = provider.lower()
            if provider_key not in self._PROVIDER_REGISTRY:
                available = ", ".join(self._PROVIDER_REGISTRY.keys())
                raise ValueError(f"Unknown provider '{provider}'. Available providers: {available}")
            
            provider_cls = self._PROVIDER_REGISTRY[provider_key]
            if model:
                provider_kwargs["model"] = model
            self.provider = provider_cls(**provider_kwargs)
        else:
            raise TypeError("provider must be a string key or BaseLLMProvider instance")

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
        Process multi-turn messages or single prompt strings.
        
        Args:
            messages: Single prompt string, ChatMessage object, message dict, or list of messages.
            system_prompt: Optional system prompt to prepend.
            stream: If True, returns an Iterator[StreamChunk] yielding tokens in real-time.
                    If False, returns complete LLMResponse.
        """
        normalized = normalize_messages(messages, system_prompt=system_prompt)
        return self.provider.chat(messages=normalized, stream=stream, **kwargs)

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
