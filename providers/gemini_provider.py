import os
from typing import Any, Dict, Iterator, List, Optional, Union
from dotenv import load_dotenv
from google import genai
from google.genai import types

from .base import BaseLLMProvider, ChatMessage, LLMResponse, StreamChunk, ToolCall

load_dotenv()


class GeminiProvider(BaseLLMProvider):
    """Google Gemini API Provider implementation using official google-genai SDK."""

    def __init__(self, model: str = "gemini-2.5-flash", api_key: Optional[str] = None, **kwargs: Any):
        self.model = model
        self.api_key = api_key or os.getenv("GEMINI_API_KEY")
        self._client: Optional[genai.Client] = None

    @property
    def client(self) -> genai.Client:
        if self._client is None:
            if not self.api_key:
                raise ValueError("GEMINI_API_KEY environment variable is required.")
            self._client = genai.Client(api_key=self.api_key)
        return self._client

    def chat(
        self,
        messages: List[ChatMessage],
        stream: bool = False,
        **kwargs: Any
    ) -> Union[LLMResponse, Iterator[StreamChunk]]:
        raw_tools = kwargs.pop("tools", None)
        contents, config = self._format_messages_and_config(messages, raw_tools=raw_tools)

        if stream:
            return self._stream_chat(contents, config=config, **kwargs)
        else:
            return self._sync_chat(contents, config=config, **kwargs)

    def _format_messages_and_config(
        self, 
        messages: List[ChatMessage],
        raw_tools: Optional[List[Dict[str, Any]]] = None,
    ) -> tuple[List[types.Content], Optional[types.GenerateContentConfig]]:
        contents: List[types.Content] = []
        system_instructions: List[str] = []

        for msg in messages:
            if msg.role == "system":
                system_instructions.append(msg.content)
            elif msg.role == "tool":
                tool_name = msg.name or "tool"
                try:
                    part = types.Part.from_function_response(
                        name=tool_name,
                        response={"result": msg.content or ""}
                    )
                    contents.append(types.Content(role="user", parts=[part]))
                except Exception:
                    text = f"[Tool Output for '{tool_name}']:\n{msg.content}"
                    contents.append(
                        types.Content(
                            role="user",
                            parts=[types.Part.from_text(text=text)]
                        )
                    )
            else:
                role = "model" if msg.role in ("assistant", "model") else "user"
                parts = []
                if msg.content:
                    parts.append(types.Part.from_text(text=msg.content))
                if msg.role in ("assistant", "model") and msg.tool_calls:
                    for tc in msg.tool_calls:
                        tc_name = getattr(tc, "name", "") if hasattr(tc, "name") else tc.get("function", {}).get("name", "")
                        tc_args = getattr(tc, "args", {}) if hasattr(tc, "args") else tc.get("function", {}).get("arguments", {})
                        if isinstance(tc_args, str):
                            import json
                            try:
                                tc_args = json.loads(tc_args)
                            except Exception:
                                tc_args = {}
                        try:
                            parts.append(types.Part.from_function_call(name=tc_name, args=tc_args))
                        except Exception:
                            pass
                if not parts:
                    parts.append(types.Part.from_text(text=""))
                contents.append(types.Content(role=role, parts=parts))

        tools_param = None
        if raw_tools:
            function_declarations = []
            for t in raw_tools:
                if isinstance(t, dict) and t.get("type") == "function":
                    fn = t.get("function", {})
                    function_declarations.append(
                        types.FunctionDeclaration(
                            name=fn.get("name"),
                            description=fn.get("description", ""),
                            parameters=fn.get("parameters"),
                        )
                    )
                elif isinstance(t, dict) and "name" in t:
                    function_declarations.append(
                        types.FunctionDeclaration(
                            name=t.get("name"),
                            description=t.get("description", ""),
                            parameters=t.get("parameters"),
                        )
                    )
            if function_declarations:
                tools_param = [types.Tool(function_declarations=function_declarations)]

        config: Optional[types.GenerateContentConfig] = None
        if system_instructions or tools_param:
            config = types.GenerateContentConfig(
                system_instruction="\n\n".join(system_instructions) if system_instructions else None,
                tools=tools_param,
            )

        return contents, config

    def _sync_chat(
        self, 
        contents: List[types.Content], 
        config: Optional[types.GenerateContentConfig] = None, 
        **kwargs: Any
    ) -> LLMResponse:
        kwargs.pop("tools", None)
        response = self.client.models.generate_content(
            model=self.model,
            contents=contents,
            config=config,
            **kwargs
        )
        try:
            content = response.text or ""
        except Exception:
            content = ""

        tool_calls: List[ToolCall] = []
        raw_fcs = getattr(response, "function_calls", None)
        if raw_fcs:
            for idx, fc in enumerate(raw_fcs):
                fc_name = getattr(fc, "name", "") or (fc.get("name") if isinstance(fc, dict) else "")
                fc_args = getattr(fc, "args", {}) or (fc.get("args") if isinstance(fc, dict) else {})
                tool_calls.append(ToolCall(id=f"call_{idx}_{fc_name}", name=fc_name, args=dict(fc_args) if isinstance(fc_args, dict) else {}))

        usage_dict = {}
        um = getattr(response, "usage_metadata", None)
        if um:
            usage_dict = {
                "prompt_tokens": getattr(um, "prompt_token_count", 0),
                "completion_tokens": getattr(um, "candidates_token_count", 0),
                "total_tokens": getattr(um, "total_token_count", 0),
            }

        return LLMResponse(
            content=content,
            model=self.model,
            tool_calls=tool_calls,
            metadata={"usage": usage_dict},
            raw_response=response,
            usage=usage_dict or None,
        )

    def _stream_chat(
        self, 
        contents: List[types.Content], 
        config: Optional[types.GenerateContentConfig] = None, 
        **kwargs: Any
    ) -> Iterator[StreamChunk]:
        kwargs.pop("tools", None)
        stream_response = self.client.models.generate_content_stream(
            model=self.model,
            contents=contents,
            config=config,
            **kwargs
        )

        for chunk in stream_response:
            delta = chunk.text or ""
            yield StreamChunk(
                delta=delta,
                raw_chunk=chunk
            )

    def is_available(self) -> bool:
        """
        Check if GEMINI_API_KEY is provided, valid, and the model is reachable.
        Uses client.models.get() as a lightweight model metadata validation.
        """
        if not self.api_key:
            return False
        try:
            self.client.models.get(model=self.model)
            return True
        except Exception:
            return False
