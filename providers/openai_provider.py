import os
from typing import Any, Dict, Iterator, List, Optional, Union
from dotenv import load_dotenv
from openai import OpenAI

import json
from .base import BaseLLMProvider, ChatMessage, LLMResponse, StreamChunk, ToolCall

load_dotenv()


class OpenAIProvider(BaseLLMProvider):
    """OpenAI API Provider supporting multi-turn chat, tool calling, and real-time streaming."""

    def __init__(self, model: str = "gpt-4o-mini", api_key: Optional[str] = None, **kwargs: Any):
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

    def _format_messages(self, messages: List[ChatMessage]) -> List[Dict[str, Any]]:
        formatted = []
        for msg in messages:
            d = msg.to_dict()
            # If assistant with tool calls, ensure arguments is JSON-serialized string
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

    def _sync_chat(self, messages: List[Dict[str, Any]], **kwargs: Any) -> LLMResponse:
        call_kwargs = dict(kwargs)
        call_kwargs.pop("num_ctx", None)
        response = self.client.chat.completions.create(
            model=self.model,
            messages=messages,
            stream=False,
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

        usage_dict = response.usage.model_dump() if getattr(response, "usage", None) else {}

        return LLMResponse(
            content=content,
            model=self.model,
            tool_calls=tool_calls,
            metadata={
                "finish_reason": choice.finish_reason,
                "usage": usage_dict,
            },
            raw_response=response,
            usage=usage_dict or None,
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
