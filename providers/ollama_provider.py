from typing import Any, Dict, Iterator, List, Optional, Union
import ollama

from .base import BaseLLMProvider, ChatMessage, LLMResponse, StreamChunk


class OllamaProvider(BaseLLMProvider):
    """Ollama Local Model Provider supporting multi-turn chat and token streaming."""

    def __init__(self, model: str = "qwen:0.5b", host: Optional[str] = None):
        self.model = model
        self.host = host
        if host:
            self.client = ollama.Client(host=host)
        else:
            self.client = ollama

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
        raw_resp = self.client.chat(
            model=self.model,
            messages=messages,
            stream=False,
            **kwargs
        )

        msg_obj = raw_resp.get("message", {}) if isinstance(raw_resp, dict) else getattr(raw_resp, "message", {})
        content = msg_obj.get("content", "") if isinstance(msg_obj, dict) else getattr(msg_obj, "content", "")

        return LLMResponse(
            content=content,
            model=self.model,
            metadata={
                "done": raw_resp.get("done") if isinstance(raw_resp, dict) else getattr(raw_resp, "done", None),
                "created_at": raw_resp.get("created_at") if isinstance(raw_resp, dict) else getattr(raw_resp, "created_at", None),
            },
            raw_response=raw_resp,
        )

    def _stream_chat(self, messages: List[Dict[str, str]], **kwargs: Any) -> Iterator[StreamChunk]:
        stream_response = self.client.chat(
            model=self.model,
            messages=messages,
            stream=True,
            **kwargs
        )

        for chunk in stream_response:
            msg_obj = chunk.get("message", {}) if isinstance(chunk, dict) else getattr(chunk, "message", {})
            delta = msg_obj.get("content", "") if isinstance(msg_obj, dict) else getattr(msg_obj, "content", "")
            done = chunk.get("done", False) if isinstance(chunk, dict) else getattr(chunk, "done", False)
            finish_reason = "stop" if done else None

            yield StreamChunk(
                delta=delta,
                finish_reason=finish_reason,
                raw_chunk=chunk
            )

    def is_available(self) -> bool:
        try:
            self.client.list()
            return True
        except Exception:
            return False
