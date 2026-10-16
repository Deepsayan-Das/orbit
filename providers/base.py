from abc import ABC, abstractmethod
from typing import Any, Dict, Iterator, List, Optional, Union
from dataclasses import dataclass, field


@dataclass
class ToolCall:
    """Represents a structured tool call returned by an LLM."""
    id: str
    name: str
    args: Dict[str, Any]


@dataclass
class ChatMessage:
    """Represents a single message in a multi-turn conversation."""
    role: str  # "system", "user", "assistant", or "tool"
    content: str
    name: Optional[str] = None
    tool_calls: Optional[Any] = None
    tool_call_id: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        d: Dict[str, Any] = {"role": self.role, "content": self.content or ""}
        if self.name is not None:
            d["name"] = self.name
        if self.tool_calls is not None:
            formatted_tc = []
            for tc in self.tool_calls:
                if isinstance(tc, ToolCall):
                    formatted_tc.append({
                        "id": tc.id,
                        "type": "function",
                        "function": {"name": tc.name, "arguments": tc.args}
                    })
                elif isinstance(tc, dict):
                    formatted_tc.append(tc)
                elif hasattr(tc, "function"):
                    fn = getattr(tc, "function")
                    fn_name = getattr(fn, "name", "")
                    fn_args = getattr(fn, "arguments", {})
                    tc_id = getattr(tc, "id", f"call_{fn_name}")
                    formatted_tc.append({
                        "id": tc_id,
                        "type": "function",
                        "function": {"name": fn_name, "arguments": fn_args}
                    })
                else:
                    formatted_tc.append(tc)
            d["tool_calls"] = formatted_tc
        if self.tool_call_id is not None:
            d["tool_call_id"] = self.tool_call_id
        return d



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
    tool_calls: List[ToolCall] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)
    raw_response: Optional[Any] = None
    usage: Optional[Dict[str, Any]] = None

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
        result.append(ChatMessage(
            role=messages.get("role", "user"),
            content=messages.get("content", ""),
            name=messages.get("name"),
            tool_calls=messages.get("tool_calls"),
            tool_call_id=messages.get("tool_call_id"),
        ))
    elif isinstance(messages, list):
        for msg in messages:
            if isinstance(msg, ChatMessage):
                result.append(msg)
            elif isinstance(msg, dict):
                result.append(ChatMessage(
                    role=msg.get("role", "user"),
                    content=msg.get("content", ""),
                    name=msg.get("name"),
                    tool_calls=msg.get("tool_calls"),
                    tool_call_id=msg.get("tool_call_id"),
                ))
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
