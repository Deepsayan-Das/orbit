from abc import ABC, abstractmethod
from typing import Any, Dict, Iterator, List, Optional, Union
from dataclasses import dataclass, field


@dataclass
class ChatMessage:
    """Represents a single message in a multi-turn conversation."""
    role: str  # "system", "user", or "assistant"
    content: str

    def to_dict(self) -> Dict[str, str]:
        return {"role": self.role, "content": self.content}


@dataclass
class StreamChunk:
    """Represents a single streamed token/chunk from the LLM."""
    delta: str
    finish_reason: Optional[str] = None
    raw_chunk: Optional[Any] = None


@dataclass
class LLMResponse:
    """Represents a complete non-streamed LLM response."""
    content: str
    model: str
    metadata: Dict[str, Any] = field(default_factory=dict)
    raw_response: Optional[Any] = None

    def __str__(self) -> str:
        return self.content


def normalize_messages(
    messages: Union[str, ChatMessage, Dict[str, str], List[Union[ChatMessage, Dict[str, str]]]],
    system_prompt: Optional[str] = None
) -> List[ChatMessage]:
    """Helper utility to normalize strings, dicts, or ChatMessage lists into List[ChatMessage]."""
    result: List[ChatMessage] = []

    if system_prompt:
        result.append(ChatMessage(role="system", content=system_prompt))

    if isinstance(messages, str):
        result.append(ChatMessage(role="user", content=messages))
    elif isinstance(messages, ChatMessage):
        result.append(messages)
    elif isinstance(messages, dict):
        result.append(ChatMessage(role=messages.get("role", "user"), content=messages.get("content", "")))
    elif isinstance(messages, list):
        for msg in messages:
            if isinstance(msg, ChatMessage):
                result.append(msg)
            elif isinstance(msg, dict):
                result.append(ChatMessage(role=msg.get("role", "user"), content=msg.get("content", "")))
            elif isinstance(msg, str):
                result.append(ChatMessage(role="user", content=msg))
    
    return result


class BaseLLMProvider(ABC):
    """Abstract Base Class for all LLM Providers in Orbit."""

    @abstractmethod
    def chat(
        self,
        messages: List[ChatMessage],
        stream: bool = False,
        **kwargs: Any
    ) -> Union[LLMResponse, Iterator[StreamChunk]]:
        """
        Process multi-turn messages.
        Returns an LLMResponse when stream=False, or an Iterator[StreamChunk] when stream=True.
        """
        pass

    @abstractmethod
    def is_available(self) -> bool:
        """Check if the provider service or API is accessible."""
        pass
