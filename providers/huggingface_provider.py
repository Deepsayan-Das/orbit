import os
from typing import Any, Dict, Iterator, List, Optional, Union
from dotenv import load_dotenv
from huggingface_hub import HfApi, InferenceClient

from .base import BaseLLMProvider, ChatMessage, LLMResponse, StreamChunk

load_dotenv()


class HuggingFaceProvider(BaseLLMProvider):
    """Hugging Face Inference API Provider implementation."""

    def __init__(self, model: str = "Qwen/Qwen2.5-7B-Instruct", token: Optional[str] = None):
        self.model = model
        self.token = token or os.getenv("HF_TOKEN") or os.getenv("HUGGINGFACE_API_KEY")
        self._client: Optional[InferenceClient] = None

    @property
    def client(self) -> InferenceClient:
        if self._client is None:
            if not self.token:
                raise ValueError("HF_TOKEN or HUGGINGFACE_API_KEY environment variable is required.")
            self._client = InferenceClient(model=self.model, token=self.token)
        return self._client

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
        response = self.client.chat_completion(
            messages=messages,
            stream=False,
            **kwargs
        )
        choice = response.choices[0]
        content = choice.message.content or ""

        return LLMResponse(
            content=content,
            model=self.model,
            metadata={
                "finish_reason": getattr(choice, "finish_reason", None),
            },
            raw_response=response,
        )

    def _stream_chat(self, messages: List[Dict[str, str]], **kwargs: Any) -> Iterator[StreamChunk]:
        stream_response = self.client.chat_completion(
            messages=messages,
            stream=True,
            **kwargs
        )

        for chunk in stream_response:
            if chunk.choices:
                choice = chunk.choices[0]
                delta = choice.delta.content or ""
                finish_reason = getattr(choice, "finish_reason", None)
                yield StreamChunk(
                    delta=delta,
                    finish_reason=finish_reason,
                    raw_chunk=chunk
                )

    def is_available(self) -> bool:
        """
        Check if the HF_TOKEN is provided, valid, and the model is reachable on HuggingFace.
        Uses HfApi.model_info() as a lightweight, zero-generation-cost metadata check.
        """
        if not self.token:
            return False
        try:
            api = HfApi(token=self.token)
            api.model_info(self.model)
            return True
        except Exception:
            return False
