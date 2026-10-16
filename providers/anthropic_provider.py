import os
from typing import Any, Dict, Iterator, List, Optional, Union
from dotenv import load_dotenv

try:
    from anthropic import Anthropic
except ImportError:
    Anthropic = None

from .base import BaseLLMProvider, ChatMessage, LLMResponse, StreamChunk, ToolCall

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
        sys_kw = kwargs.pop("system_prompt", None)
        formatted_messages, system_prompt = self._format_messages(messages)
        if sys_kw:
            system_prompt = f"{sys_kw}\n\n{system_prompt}" if system_prompt else sys_kw
        anthropic_tools = self._format_tools(raw_tools) if raw_tools else None

        if stream:
            return self._stream_chat(formatted_messages, system_prompt=system_prompt, tools=anthropic_tools, **kwargs)
        else:
            return self._sync_chat(formatted_messages, system_prompt=system_prompt, tools=anthropic_tools, **kwargs)

    def _format_messages(self, messages: List[ChatMessage]) -> tuple[List[Dict[str, Any]], Optional[str]]:
        formatted: List[Dict[str, Any]] = []
        system_parts: List[str] = []

        i = 0
        while i < len(messages):
            msg = messages[i]
            if msg.role == "system":
                system_parts.append(msg.content)
                i += 1
            elif msg.role == "tool":
                # Group consecutive tool outputs into a single user message with tool_result blocks
                tool_results = []
                while i < len(messages) and messages[i].role == "tool":
                    tmsg = messages[i]
                    tool_results.append({
                        "type": "tool_result",
                        "tool_use_id": tmsg.tool_call_id or f"call_{tmsg.name or 'default'}",
                        "content": str(tmsg.content or ""),
                    })
                    i += 1
                formatted.append({
                    "role": "user",
                    "content": tool_results,
                })
            elif msg.role in ("assistant", "model"):
                if msg.tool_calls:
                    content_blocks: List[Dict[str, Any]] = []
                    if msg.content:
                        content_blocks.append({"type": "text", "text": msg.content})
                    for tc in msg.tool_calls:
                        if isinstance(tc, ToolCall):
                            content_blocks.append({
                                "type": "tool_use",
                                "id": tc.id,
                                "name": tc.name,
                                "input": tc.args,
                            })
                        elif isinstance(tc, dict):
                            fn = tc.get("function", {})
                            content_blocks.append({
                                "type": "tool_use",
                                "id": tc.get("id", f"call_{fn.get('name', 'default')}"),
                                "name": fn.get("name", ""),
                                "input": fn.get("arguments", {}),
                            })
                        elif hasattr(tc, "function"):
                            fn = getattr(tc, "function")
                            content_blocks.append({
                                "type": "tool_use",
                                "id": getattr(tc, "id", f"call_{getattr(fn, 'name', 'default')}"),
                                "name": getattr(fn, "name", ""),
                                "input": getattr(fn, "arguments", {}),
                            })
                    formatted.append({"role": "assistant", "content": content_blocks})
                else:
                    formatted.append({"role": "assistant", "content": msg.content or ""})
                i += 1
            else:
                formatted.append({"role": "user", "content": msg.content or ""})
                i += 1

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
            if isinstance(system_prompt, str):
                call_kwargs["system"] = [
                    {
                        "type": "text",
                        "text": system_prompt,
                        "cache_control": {"type": "ephemeral"}
                    }
                ]
            else:
                call_kwargs["system"] = system_prompt

        if tools:
            cached_tools = [dict(t) for t in tools]
            if cached_tools:
                cached_tools[-1]["cache_control"] = {"type": "ephemeral"}
            call_kwargs["tools"] = cached_tools

        call_kwargs.update(kwargs)

        response = self.client.messages.create(**call_kwargs)

        content_parts = []
        tool_calls: List[ToolCall] = []
        for block in getattr(response, "content", []):
            block_type = getattr(block, "type", None)
            if block_type == "text":
                content_parts.append(getattr(block, "text", ""))
            elif block_type == "tool_use":
                tool_calls.append(
                    ToolCall(
                        id=getattr(block, "id", ""),
                        name=getattr(block, "name", ""),
                        args=getattr(block, "input", {}) or {},
                    )
                )

        content = "".join(content_parts)

        usage_dict = {}
        raw_usage = getattr(response, "usage", None)
        if raw_usage:
            usage_dict = {
                "prompt_tokens": getattr(raw_usage, "input_tokens", 0),
                "completion_tokens": getattr(raw_usage, "output_tokens", 0),
                "total_tokens": getattr(raw_usage, "input_tokens", 0) + getattr(raw_usage, "output_tokens", 0),
                "cache_creation_input_tokens": getattr(raw_usage, "cache_creation_input_tokens", 0),
                "cache_read_input_tokens": getattr(raw_usage, "cache_read_input_tokens", 0),
            }

        return LLMResponse(
            content=content,
            model=self.model,
            tool_calls=tool_calls,
            metadata={"usage": usage_dict or raw_usage},
            raw_response=response,
            usage=usage_dict or None,
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
            if isinstance(system_prompt, str):
                call_kwargs["system"] = [
                    {
                        "type": "text",
                        "text": system_prompt,
                        "cache_control": {"type": "ephemeral"}
                    }
                ]
            else:
                call_kwargs["system"] = system_prompt

        if tools:
            cached_tools = [dict(t) for t in tools]
            if cached_tools:
                cached_tools[-1]["cache_control"] = {"type": "ephemeral"}
            call_kwargs["tools"] = cached_tools

        call_kwargs.update(kwargs)

        with self.client.messages.stream(**call_kwargs) as stream:
            for text in stream.text_stream:
                yield StreamChunk(delta=text)

    def is_available(self) -> bool:
        """Check if ANTHROPIC_API_KEY environment variable or parameter is present."""
        return bool(self.api_key)
