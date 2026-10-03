import os
from typing import Any, Dict, Iterator, List, Optional, Union
from dotenv import load_dotenv

try:
    from anthropic import Anthropic
except ImportError:
    Anthropic = None

from .base import BaseLLMProvider, ChatMessage, LLMResponse, StreamChunk

load_dotenv()


class AnthropicProvider(BaseLLMProvider):
    """Anthropic Claude API Provider implementation using official anthropic SDK."""

    def __init__(
        self, 
        model: str = "claude-3-5-sonnet-20241022", 
        api_key: Optional[str] = None, 
        **kwargs: Any
    ):
        self.model = model
        self.api_key = api_key or os.getenv("ANTHROPIC_API_KEY")
        self._client: Optional[Any] = None

    @property
    def client(self) -> Any:
        if self._client is None:
            if not self.api_key:
                raise ValueError("ANTHROPIC_API_KEY environment variable or api_key parameter is required.")
            if Anthropic is None:
                raise ImportError("anthropic package is not installed. Install via `pip install anthropic`.")
            self._client = Anthropic(api_key=self.api_key)
        return self._client

    def chat(
        self,
        messages: List[ChatMessage],
        stream: bool = False,
        **kwargs: Any
    ) -> Union[LLMResponse, Iterator[StreamChunk]]:
        raw_tools = kwargs.pop("tools", None)
        formatted_messages, system_prompt = self._format_messages(messages)
        anthropic_tools = self._format_tools(raw_tools) if raw_tools else None

        if stream:
            return self._stream_chat(formatted_messages, system_prompt=system_prompt, tools=anthropic_tools, **kwargs)
        else:
            return self._sync_chat(formatted_messages, system_prompt=system_prompt, tools=anthropic_tools, **kwargs)

    def _format_messages(self, messages: List[ChatMessage]) -> tuple[List[Dict[str, Any]], Optional[str]]:
        formatted: List[Dict[str, Any]] = []
        system_parts: List[str] = []

        for msg in messages:
            if msg.role == "system":
                system_parts.append(msg.content)
            elif msg.role == "tool":
                tool_name = msg.name or "tool"
                formatted.append({
                    "role": "user",
                    "content": f"[Tool Output for '{tool_name}']:\n{msg.content}"
                })
            else:
                role = "assistant" if msg.role in ("assistant", "model") else "user"
                content = msg.content or ""
                if not content and role == "assistant":
                    content = "[Executing tools...]"
                formatted.append({"role": role, "content": content})

        system_prompt = "\n\n".join(system_parts) if system_parts else None
        return formatted, system_prompt

    def _format_tools(self, raw_tools: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        formatted_tools = []
        for t in raw_tools:
            if isinstance(t, dict) and t.get("type") == "function":
                fn = t.get("function", {})
                formatted_tools.append({
                    "name": fn.get("name"),
                    "description": fn.get("description", ""),
                    "input_schema": fn.get("parameters", {"type": "object", "properties": {}})
                })
            elif isinstance(t, dict) and "name" in t:
                formatted_tools.append({
                    "name": t.get("name"),
                    "description": t.get("description", ""),
                    "input_schema": t.get("parameters", {"type": "object", "properties": {}})
                })
        return formatted_tools

    def _sync_chat(
        self, 
        messages: List[Dict[str, Any]], 
        system_prompt: Optional[str] = None, 
        tools: Optional[List[Dict[str, Any]]] = None,
        **kwargs: Any
    ) -> LLMResponse:
        kwargs.pop("num_ctx", None)
        max_tokens = kwargs.pop("max_tokens", 4096)
        call_kwargs: Dict[str, Any] = {
            "model": self.model,
            "messages": messages,
            "max_tokens": max_tokens,
        }
        if system_prompt:
            call_kwargs["system"] = system_prompt
        if tools:
            call_kwargs["tools"] = tools

        call_kwargs.update(kwargs)

        response = self.client.messages.create(**call_kwargs)

        content_parts = []
        for block in getattr(response, "content", []):
            if getattr(block, "type", None) == "text":
                content_parts.append(getattr(block, "text", ""))

        content = "".join(content_parts)

        return LLMResponse(
            content=content,
            model=self.model,
            metadata={"usage": getattr(response, "usage", None)},
            raw_response=response,
        )

    def _stream_chat(
        self, 
        messages: List[Dict[str, Any]], 
        system_prompt: Optional[str] = None, 
        tools: Optional[List[Dict[str, Any]]] = None,
        **kwargs: Any
    ) -> Iterator[StreamChunk]:
        kwargs.pop("num_ctx", None)
        max_tokens = kwargs.pop("max_tokens", 4096)
        call_kwargs: Dict[str, Any] = {
            "model": self.model,
            "messages": messages,
            "max_tokens": max_tokens,
        }
        if system_prompt:
            call_kwargs["system"] = system_prompt
        if tools:
            call_kwargs["tools"] = tools

        call_kwargs.update(kwargs)

        with self.client.messages.stream(**call_kwargs) as stream:
            for text in stream.text_stream:
                yield StreamChunk(delta=text)

    def is_available(self) -> bool:
        """Check if ANTHROPIC_API_KEY environment variable or parameter is present."""
        return bool(self.api_key)
