import os
from typing import Any, Dict, Iterator, List, Optional, Union
from dotenv import load_dotenv
from openai import OpenAI

from .base import BaseLLMProvider, ChatMessage, LLMResponse, StreamChunk

load_dotenv()


class OpenAIProvider(BaseLLMProvider):
    """OpenAI API Provider supporting multi-turn chat and real-time streaming."""

    def __init__(self, model: str = "gpt-4o-mini", api_key: Optional[str] = None):
        self.model = model
        self.api_key = api_key or os.getenv("OPENAI_API_KEY")
        self._client: Optional[OpenAI] = None

    @property
    def client(self) -> OpenAI:
        if self._client is None:
            if not self.api_key:
                raise ValueError("OPENAI_API_KEY environment variable or api_key parameter is required.")
            self._client = OpenAI(api_key=self.api_key)
        return self._client

    def chat(
        self,
        messages: List[ChatMessage],
        stream: bool = False,
        **kwargs: Any
    ) -> Union[LLMResponse, Iterator[StreamChunk]]:
        # Convert List[ChatMessage] -> List[Dict[str, str]]
        # System messages (ChatMessage(role="system", content="...")) are automatically
        # formatted as {"role": "system", "content": "..."} which OpenAI natively expects.
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
        content = response.choices[0].message.content or ""

        return LLMResponse(
            content=content,
            model=self.model,
            metadata={
                "finish_reason": response.choices[0].finish_reason,
                "usage": response.usage.model_dump() if getattr(response, "usage", None) else {},
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
                delta = chunk.choices[0].delta.content or ""
                finish_reason = chunk.choices[0].finish_reason
                yield StreamChunk(
                    delta=delta,
                    finish_reason=finish_reason,
                    raw_chunk=chunk
                )

    def is_available(self) -> bool:
        """
        Check if the OpenAI API key is valid and network connectivity is active.
        Uses client.models.list() which is 100% FREE (uses zero tokens).
        """
        if not self.api_key:
            return False
        try:
            # Free API call: validates authentication & connectivity without spending tokens
            self.client.models.list()
            return True
        except Exception:
            return False
