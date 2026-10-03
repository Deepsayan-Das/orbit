import os
from typing import Any, Dict, Iterator, List, Optional, Union
from dotenv import load_dotenv

try:
    from mistralai.client import Mistral
except ImportError:
    Mistral = None

from .base import BaseLLMProvider, ChatMessage, LLMResponse, StreamChunk

load_dotenv()


class MistralProvider(BaseLLMProvider):
    """Mistral AI Provider implementation using official mistralai SDK."""

    def __init__(
        self, 
        model: str = "mistral-small-latest", 
        api_key: Optional[str] = None, 
        **kwargs: Any
    ):
        self.model = model
        self.api_key = api_key or os.getenv("MISTRAL_API_KEY")
        self._client: Optional[Any] = None

    @property
    def client(self) -> Any:
        if self._client is None:
            if not self.api_key:
                raise ValueError("MISTRAL_API_KEY environment variable or api_key parameter is required.")
            if Mistral is None:
                raise ImportError("mistralai package is not installed. Install via `pip install mistralai`.")
            self._client = Mistral(api_key=self.api_key)
        return self._client

    def chat(
        self,
        messages: List[ChatMessage],
        stream: bool = False,
        **kwargs: Any
    ) -> Union[LLMResponse, Iterator[StreamChunk]]:
        kwargs.pop("tools", None)
        formatted_messages = self._format_messages(messages)

        if stream:
            return self._stream_chat(formatted_messages, **kwargs)
        else:
            return self._sync_chat(formatted_messages, **kwargs)

    def _format_messages(self, messages: List[ChatMessage]) -> List[Dict[str, Any]]:
        formatted: List[Dict[str, Any]] = []

        for msg in messages:
            if msg.role == "tool":
                tool_name = msg.name or "tool"
                formatted.append({
                    "role": "user",
                    "content": f"[Tool Output for '{tool_name}']:\n{msg.content}"
                })
            else:
                role = "assistant" if msg.role in ("assistant", "model") else msg.role
                content = msg.content or ""
                if not content and role == "assistant":
                    content = "[Executing tools...]"
                formatted.append({"role": role, "content": content})

        return formatted

    def _sync_chat(
        self, 
        messages: List[Dict[str, Any]], 
        **kwargs: Any
    ) -> LLMResponse:
        kwargs.pop("num_ctx", None)
        response = self.client.chat.complete(
            model=self.model,
            messages=messages,
            **kwargs
        )
        choice = response.choices[0]
        content = choice.message.content or ""

        return LLMResponse(
            content=content,
            model=self.model,
            metadata={"usage": getattr(response, "usage", None)},
            raw_response=response,
        )

    def _stream_chat(
        self, 
        messages: List[Dict[str, Any]], 
        **kwargs: Any
    ) -> Iterator[StreamChunk]:
        kwargs.pop("num_ctx", None)
        response_stream = self.client.chat.stream(
            model=self.model,
            messages=messages,
            **kwargs
        )

        for event in response_stream:
            if hasattr(event, "data") and hasattr(event.data, "choices"):
                delta = event.data.choices[0].delta.content or ""
                yield StreamChunk(delta=delta)
            elif hasattr(event, "choices"):
                delta = event.choices[0].delta.content or ""
                yield StreamChunk(delta=delta)

    def is_available(self) -> bool:
        """Check if MISTRAL_API_KEY environment variable or parameter is present."""
        return bool(self.api_key)
