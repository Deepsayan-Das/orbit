from typing import Any, Dict, Iterator, List, Optional, Union
import ollama

from .base import BaseLLMProvider, ChatMessage, LLMResponse, StreamChunk, ToolCall


class OllamaProvider(BaseLLMProvider):
    """Ollama Local Model Provider supporting multi-turn chat and token streaming."""

    GENERIC_OPTION_MAP: Dict[str, str] = {
        "max_tokens": "num_predict",
        "max_completion_tokens": "num_predict",
    }

    OLLAMA_OPTIONS: set = {
        "numa", "num_ctx", "num_batch", "num_gpu", "main_gpu", "low_vram",
        "f16_kv", "logits_all", "vocab_only", "use_mmap", "use_mlock",
        "embedding_only", "num_thread", "num_keep", "seed", "num_predict",
        "top_k", "top_p", "tfs_z", "typical_p", "repeat_last_n",
        "temperature", "repeat_penalty", "presence_penalty",
        "frequency_penalty", "mirostat", "mirostat_tau", "mirostat_eta",
        "penalize_newline", "stop",
    }

    VALID_CHAT_KWARGS: set = {
        "tools", "format", "keep_alive", "think", "logprobs", "top_logprobs"
    }

    def __init__(
        self,
        model: str = "qwen:0.5b",
        host: Optional[str] = None,
        num_ctx: Optional[int] = None,
        **kwargs: Any
    ):
        self.model = model
        self.host = host
        self.num_ctx = num_ctx
        if host:
            self.client = ollama.Client(host=host)
        else:
            self.client = ollama.Client()

    def _prepare_kwargs(self, kwargs: Dict[str, Any]) -> Dict[str, Any]:
        kw = dict(kwargs)
        options = dict(kw.pop("options", None) or {})

        prepared: Dict[str, Any] = {}

        for key, val in kw.items():
            if key in ("model", "messages", "stream"):
                continue

            if key in self.GENERIC_OPTION_MAP:
                target_key = self.GENERIC_OPTION_MAP[key]
                if target_key not in options:
                    options[target_key] = val
            elif key in self.OLLAMA_OPTIONS:
                if key not in options:
                    options[key] = val
            elif key in self.VALID_CHAT_KWARGS:
                prepared[key] = val
            else:
                print(f"[warning] Unrecognized parameter '{key}' for Ollama provider; ignoring.")

        if self.num_ctx is not None and "num_ctx" not in options:
            options["num_ctx"] = self.num_ctx

        if options:
            prepared["options"] = options
        return prepared

    def chat(
        self,
        messages: List[ChatMessage],
        stream: bool = False,
        **kwargs: Any
    ) -> Union[LLMResponse, Iterator[StreamChunk]]:
        formatted_messages = [msg.to_dict() for msg in messages]
        prepared_kwargs = self._prepare_kwargs(kwargs)

        if stream:
            return self._stream_chat(formatted_messages, **prepared_kwargs)
        else:
            return self._sync_chat(formatted_messages, **prepared_kwargs)

    def _sync_chat(self, messages: List[Dict[str, str]], **kwargs: Any) -> LLMResponse:
        raw_resp = self.client.chat(
            model=self.model,
            messages=messages,
            stream=False,
            **kwargs
        )

        msg_obj = raw_resp.get("message", {}) if isinstance(raw_resp, dict) else getattr(raw_resp, "message", {})
        content = msg_obj.get("content", "") if isinstance(msg_obj, dict) else getattr(msg_obj, "content", "")

        tool_calls: List[ToolCall] = []
        raw_tcs = msg_obj.get("tool_calls") if isinstance(msg_obj, dict) else getattr(msg_obj, "tool_calls", None)
        if raw_tcs:
            for idx, tc in enumerate(raw_tcs):
                fn = tc.get("function", {}) if isinstance(tc, dict) else getattr(tc, "function", None)
                fn_name = fn.get("name", "") if isinstance(fn, dict) else getattr(fn, "name", "")
                fn_args = fn.get("arguments", {}) if isinstance(fn, dict) else getattr(fn, "arguments", {})
                if isinstance(fn_args, str):
                    import json
                    try:
                        fn_args = json.loads(fn_args)
                    except Exception:
                        fn_args = {}
                tc_id = tc.get("id", f"call_{idx}_{fn_name}") if isinstance(tc, dict) else getattr(tc, "id", f"call_{idx}_{fn_name}")
                tool_calls.append(ToolCall(id=tc_id, name=fn_name, args=dict(fn_args) if isinstance(fn_args, dict) else {}))

        usage = {}
        if isinstance(raw_resp, dict):
            if "prompt_eval_count" in raw_resp:
                usage["prompt_tokens"] = raw_resp["prompt_eval_count"]
            if "eval_count" in raw_resp:
                usage["completion_tokens"] = raw_resp["eval_count"]
            if "prompt_tokens" in usage and "completion_tokens" in usage:
                usage["total_tokens"] = usage["prompt_tokens"] + usage["completion_tokens"]
        elif hasattr(raw_resp, "prompt_eval_count"):
            usage["prompt_tokens"] = getattr(raw_resp, "prompt_eval_count", 0)
            usage["completion_tokens"] = getattr(raw_resp, "eval_count", 0)
            usage["total_tokens"] = usage["prompt_tokens"] + usage["completion_tokens"]

        return LLMResponse(
            content=content,
            model=self.model,
            tool_calls=tool_calls,
            metadata={
                "done": raw_resp.get("done") if isinstance(raw_resp, dict) else getattr(raw_resp, "done", None),
                "created_at": raw_resp.get("created_at") if isinstance(raw_resp, dict) else getattr(raw_resp, "created_at", None),
                "usage": usage or None,
            },
            raw_response=raw_resp,
            usage=usage or None,
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
