import os
from typing import Any, Dict, Iterator, List, Optional, Union
from dotenv import load_dotenv

try:
    from mistralai.client import Mistral
except ImportError:
    Mistral = None

import json
from .base import BaseLLMProvider, ChatMessage, LLMResponse, StreamChunk, ToolCall

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

    def _format_messages(self, messages: List[ChatMessage]) -> List[Dict[str, Any]]:
        formatted: List[Dict[str, Any]] = []

        for msg in messages:
            d = msg.to_dict()
            if msg.role in ("assistant", "model") and d.get("tool_calls"):
                clean_tcs = []
                for tc in d["tool_calls"]:
                    fn = tc.get("function", {})
                    args = fn.get("arguments", {})
                    if not isinstance(args, str):
                        args_str = json.dumps(args)
                    else:
                        args_str = args
                    clean_tcs.append({
                        "id": tc.get("id", f"call_{fn.get('name', 'default')}"),
                        "type": "function",
                        "function": {
                            "name": fn.get("name", ""),
                            "arguments": args_str,
                        }
                    })
                d["tool_calls"] = clean_tcs
                if not d.get("content"):
                    d["content"] = None
            elif msg.role == "tool":
                if not d.get("tool_call_id"):
                    d["tool_call_id"] = f"call_{msg.name or 'default'}"
            formatted.append(d)

        return formatted

    def chat(
        self,
        messages: List[ChatMessage],
        stream: bool = False,
        **kwargs: Any
    ) -> Union[LLMResponse, Iterator[StreamChunk]]:
        formatted_messages = self._format_messages(messages)

        if stream:
            return self._stream_chat(formatted_messages, **kwargs)
        else:
            return self._sync_chat(formatted_messages, **kwargs)

    def _sync_chat(
        self, 
        messages: List[Dict[str, Any]], 
        **kwargs: Any
    ) -> LLMResponse:
        call_kwargs = dict(kwargs)
        call_kwargs.pop("num_ctx", None)
        response = self.client.chat.complete(
            model=self.model,
            messages=messages,
            **call_kwargs
        )
        choice = response.choices[0]
        content = choice.message.content or ""

        tool_calls: List[ToolCall] = []
        raw_tcs = getattr(choice.message, "tool_calls", None)
        if raw_tcs:
            for idx, tc in enumerate(raw_tcs):
                tc_id = getattr(tc, "id", f"call_{idx}")
                fn = getattr(tc, "function", None)
                fn_name = getattr(fn, "name", "") if fn else ""
                fn_args_raw = getattr(fn, "arguments", "{}") if fn else "{}"
                if isinstance(fn_args_raw, str):
                    try:
                        fn_args = json.loads(fn_args_raw)
                    except Exception:
                        fn_args = {}
                elif isinstance(fn_args_raw, dict):
                    fn_args = fn_args_raw
                else:
                    fn_args = {}
                tool_calls.append(ToolCall(id=tc_id, name=fn_name, args=fn_args))

        usage_dict = {}
        raw_usage = getattr(response, "usage", None)
        if raw_usage:
            if hasattr(raw_usage, "prompt_tokens"):
                usage_dict = {
                    "prompt_tokens": getattr(raw_usage, "prompt_tokens", 0),
                    "completion_tokens": getattr(raw_usage, "completion_tokens", 0),
                    "total_tokens": getattr(raw_usage, "total_tokens", 0),
                }
            elif isinstance(raw_usage, dict):
                usage_dict = raw_usage

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
