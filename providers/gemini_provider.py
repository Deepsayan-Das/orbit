import os
from typing import Any, Dict, Iterator, List, Optional, Union
from dotenv import load_dotenv
from google import genai
from google.genai import types

from .base import BaseLLMProvider, ChatMessage, LLMResponse, StreamChunk

load_dotenv()


class GeminiProvider(BaseLLMProvider):
    """Google Gemini API Provider implementation using official google-genai SDK."""

    def __init__(self, model: str = "gemini-2.5-flash", api_key: Optional[str] = None):
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
        contents, config = self._format_messages_and_config(messages)

        if stream:
            return self._stream_chat(contents, config=config, **kwargs)
        else:
            return self._sync_chat(contents, config=config, **kwargs)

    def _format_messages_and_config(
        self, 
        messages: List[ChatMessage]
    ) -> tuple[List[types.Content], Optional[types.GenerateContentConfig]]:
        contents: List[types.Content] = []
        system_instructions: List[str] = []

        for msg in messages:
            if msg.role == "system":
                system_instructions.append(msg.content)
            else:
                role = "model" if msg.role == "assistant" else "user"
                contents.append(
                    types.Content(
                        role=role,
                        parts=[types.Part.from_text(text=msg.content)]
                    )
                )

        config: Optional[types.GenerateContentConfig] = None
        if system_instructions:
            config = types.GenerateContentConfig(
                system_instruction="\n\n".join(system_instructions)
            )

        return contents, config

    def _sync_chat(
        self, 
        contents: List[types.Content], 
        config: Optional[types.GenerateContentConfig] = None, 
        **kwargs: Any
    ) -> LLMResponse:
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
