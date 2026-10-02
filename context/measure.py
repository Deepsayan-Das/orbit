"""
Context Measurement and Compaction Trigger Module.

Pure logic for measuring token usage and determining compaction triggers.
"""

from typing import Any, Dict, List, Union
from providers.base import ChatMessage, LLMResponse

RESERVE_TOKENS: int = 1000


def estimate_tokens(messages: Union[List[ChatMessage], LLMResponse, Any]) -> int:
    """Estimate token count for a message history list or LLMResponse object.

    If an LLMResponse is passed carrying provider-reported usage, uses reported prompt/total tokens.
    Otherwise, computes character count divided by 4 across all messages.
    """
    if isinstance(messages, LLMResponse):
        # 1. Check direct usage field if present
        usage = getattr(messages, "usage", None) or messages.metadata.get("usage")
        if isinstance(usage, dict):
            prompt_t = usage.get("prompt_tokens") or usage.get("input_tokens") or usage.get("total_tokens")
            if prompt_t is not None:
                return int(prompt_t)
        elif isinstance(usage, int):
            return usage

        # Fallback to content length estimation
        return max(0, len(messages.content or "") // 4)

    if not messages:
        return 0

    if not isinstance(messages, list):
        messages = [messages]

    total_chars = 0
    for msg in messages:
        if isinstance(msg, ChatMessage):
            total_chars += len(msg.role or "")
            total_chars += len(msg.content or "")
            if msg.name:
                total_chars += len(msg.name)
            if msg.tool_calls:
                total_chars += len(str(msg.tool_calls))
        elif isinstance(msg, dict):
            total_chars += len(msg.get("role", ""))
            total_chars += len(msg.get("content", ""))
            if msg.get("name"):
                total_chars += len(msg.get("name"))
            if msg.get("tool_calls"):
                total_chars += len(str(msg.get("tool_calls")))
        elif isinstance(msg, str):
            total_chars += len(msg)

    return max(0, total_chars // 4)


def should_compact(used: int, window: int) -> bool:
    """Determine whether history should be compacted based on token usage and window.

    Triggers when used >= 75% of window OR (window - used) < RESERVE_TOKENS.
    """
    if window <= 0:
        return False

    trigger_75 = used >= (window * 0.75)
    trigger_reserve = (window - used) < RESERVE_TOKENS
    return trigger_75 or trigger_reserve
