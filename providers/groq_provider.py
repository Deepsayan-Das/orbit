import os
from typing import Any, Dict, Iterator, List, Optional, Union
from dotenv import load_dotenv

from .base import BaseLLMProvider, ChatMessage, LLMResponse, StreamChunk

load_dotenv()

# Try importing the official groq SDK first, fall back to OpenAI SDK with Groq base URL
try:
    from groq import Groq
except ImportError:
    Groq = None

try:
    from openai import OpenAI
except ImportError:
    OpenAI = None


class GroqProvider(BaseLLMProvider):
    """Groq API Provider supporting ultra-fast LLM inference and streaming."""

    def __init__(self, model: str = "llama-3.3-70b-versatile", api_key: Optional[str] = None):
        self.model = model
        self.api_key = api_key or os.getenv("GROQ_API_KEY")
        self._client: Optional[Any] = None

    @property
    def client(self) -> Any:
        if self._client is None:
            if not self.api_key:
                raise ValueError("GROQ_API_KEY environment variable or api_key parameter is required.")
            
            if Groq is not None:
                self._client = Groq(api_key=self.api_key)
            elif OpenAI is not None:
                self._client = OpenAI(
                    api_key=self.api_key,
                    base_url="https://api.groq.com/openai/v1"
                )
            else:
                raise ImportError(
                    "Neither 'groq' nor 'openai' package is installed. "
                    "Please install either 'groq' or 'openai' via pip."
                )
        return self._client

    def chat(
        self,
        messages: List[ChatMessage],
        stream: bool = False,
        **kwargs: Any
    ) -> Union[LLMResponse, Iterator[StreamChunk]]:
        formatted_messages = [msg.to_dict() for msg in messages]

        if stream:
            return self._stream_chat(formatted_messages, **kwargs)
        else:
            return self._sync_chat(formatted_messages, **kwargs)

    def _sync_chat(self, messages: List[Dict[str, str]], **kwargs: Any) -> LLMResponse:
        response = self.client.chat.completions.create(
            model=self.model,
            messages=messages,
            stream=False,
            **kwargs
        )
        choice = response.choices[0]
        content = choice.message.content or ""

        usage = {}
        if hasattr(response, "usage") and response.usage:
            usage = response.usage.model_dump() if hasattr(response.usage, "model_dump") else dict(response.usage)

        return LLMResponse(
            content=content,
            model=self.model,
            metadata={
                "finish_reason": getattr(choice, "finish_reason", None),
                "usage": usage,
            },
            raw_response=response,
        )

    def _stream_chat(self, messages: List[Dict[str, str]], **kwargs: Any) -> Iterator[StreamChunk]:
        stream_response = self.client.chat.completions.create(
            model=self.model,
            messages=messages,
            stream=True,
            **kwargs
        )

        for chunk in stream_response:
            if chunk.choices:
                choice = chunk.choices[0]
                delta = choice.delta.content or ""
                finish_reason = getattr(choice, "finish_reason", None)
                yield StreamChunk(
                    delta=delta,
                    finish_reason=finish_reason,
                    raw_chunk=chunk
                )

    def is_available(self) -> bool:
        """
        Check if the Groq API key is set and valid.
        Executes models.list() to verify authentication without consuming tokens.
        """
        if not self.api_key:
            return False
        try:
            self.client.models.list()
            return True
        except Exception:
            return False
