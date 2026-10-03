import os
from typing import Any, Dict, Iterator, List, Optional, Union
from dotenv import load_dotenv
from google import genai
from google.genai import types

from .base import BaseLLMProvider, ChatMessage, LLMResponse, StreamChunk

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
                text = f"[Tool Output for '{tool_name}']:\n{msg.content}"
                contents.append(
                    types.Content(
                        role="user",
                        parts=[types.Part.from_text(text=text)]
                    )
                )
            else:
                role = "model" if msg.role == "assistant" else "user"
                text = msg.content or ""
                if not text and msg.role == "assistant":
                    text = "[Executing tools...]"
                contents.append(
                    types.Content(
                        role=role,
                        parts=[types.Part.from_text(text=text)]
                    )
                )

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
        content = response.text or ""

        return LLMResponse(
            content=content,
            model=self.model,
            metadata={},
            raw_response=response,
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
