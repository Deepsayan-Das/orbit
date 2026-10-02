"""
Context Compaction Subsystem for Orbit.

Implements intelligent history compaction preserving system prompt & 1st user message,
enforcing strict atomic tool call/result pairing, per-tool outcome stubbing, LLM summarization,
and failure fallback to pair dropping.
"""

from typing import Any, List, Optional
from context.measure import estimate_tokens, should_compact
from providers.base import ChatMessage


def stub_tool_result(fn_name: str, content: str) -> str:
    """Generate a short stub summary for an old tool result string."""
    clean = (content or "").strip()
    lines = [l for l in clean.split("\n") if l.strip()]

    if fn_name == "read_file":
        path_hint = lines[0] if lines else "file"
        symbol_hint = "functions/classes" if "def " in clean or "class " in clean else "content"
        return f"[read_file outcome for {path_hint}: {len(lines)} lines, main symbols: {symbol_hint} (may be outdated)]"

    elif fn_name == "run_tests":
        failing = "none" if "OK" in clean or "PASSED" in clean else "failing tests detected"
        reason = lines[0] if lines else "test run completed"
        return f"[run_tests outcome: failing tests: {failing}, reason: {reason}]"

    elif fn_name == "edit_file":
        path_hint = lines[0] if lines else "file"
        return f"[edit_file outcome for {path_hint}: modified content with diff]"

    elif fn_name == "run_shell_command":
        cmd_hint = "shell command"
        last_line = lines[-1] if lines else "ok"
        return f"[run_shell_command outcome for {cmd_hint}: exit code 0/status ok, last lines: {last_line[:60]}]"

    else:
        first_line = lines[0][:60] if lines else "completed"
        return f"[{fn_name} outcome: {first_line} ({len(clean)} chars)]"


def stub_middle_tool_results(messages: List[ChatMessage]) -> List[ChatMessage]:
    """Return a new list of messages where tool result messages have stubbed content."""
    stubbed = []
    for msg in messages:
        if isinstance(msg, ChatMessage) and msg.role == "tool":
            fn_name = msg.name or "tool"
            stub_content = stub_tool_result(fn_name, msg.content)
            stubbed.append(ChatMessage(role="tool", content=stub_content, name=msg.name))
        else:
            stubbed.append(msg)
    return stubbed


def find_tool_pairs(messages: List[ChatMessage]) -> List[List[int]]:
    """Find indices of atomic tool call/result pairs in a message list."""
    pairs = []
    i = 0
    while i < len(messages):
        msg = messages[i]
        if isinstance(msg, ChatMessage) and msg.role == "assistant" and msg.tool_calls:
            # Look for subsequent tool messages matching this call
            pair_indices = [i]
            j = i + 1
            while j < len(messages) and isinstance(messages[j], ChatMessage) and messages[j].role == "tool":
                pair_indices.append(j)
                j += 1
            if len(pair_indices) > 1:
                pairs.append(pair_indices)
                i = j
                continue
        i += 1
    return pairs


def drop_oldest_tool_pairs(middle: List[ChatMessage]) -> List[ChatMessage]:
    """Drop the oldest tool call/result pair from middle messages."""
    pairs = find_tool_pairs(middle)
    if not pairs:
        return middle[1:] if middle else []

    oldest_pair_indices = set(pairs[0])
    return [msg for idx, msg in enumerate(middle) if idx not in oldest_pair_indices]


def compact_history(
    history: List[ChatMessage],
    focus: Optional[str] = None,
    agent: Optional[Any] = None,
    window: int = 4096,
    recent_count: int = 6,
) -> List[ChatMessage]:
    """Compact conversation history while enforcing all safety & pairing rules.

    Rules:
    1. Never touch: System prompt (index 0 if role='system') and 1st user message (role='user').
    2. Keep recent N messages (default 6) unchanged.
    3. Tool Call Pairing Rule: Tool call assistant message + tool result message form an atomic pair.
    4. Replace old tool results with short outcome stubs.
    5. Summarize middle history if still above trigger; fallback to pair dropping if summary fails.
    6. Retry limit: max 2 retries.
    """
    if not history:
        return history

    # Identify protected prefix (system prompt + 1st user message)
    protected_count = 0
    if history[0].role == "system":
        protected_count = 1
        if len(history) > 1 and history[1].role == "user":
            protected_count = 2
    elif history[0].role == "user":
        protected_count = 1

    if len(history) <= protected_count + recent_count:
        return history

    protected_prefix = history[:protected_count]
    recent_suffix = history[-recent_count:]
    middle = history[protected_count:-recent_count]

    if not middle:
        return history

    retry = 0
    candidate = history

    while retry < 2:
        retry += 1

        # Step 1: Stub old tool results in middle
        stubbed_middle = stub_middle_tool_results(middle)
        candidate = protected_prefix + stubbed_middle + recent_suffix

        if not should_compact(estimate_tokens(candidate), window):
            return candidate

        # Step 2: Try LLM summarization if agent is available
        summary_succeeded = False
        if agent is not None and hasattr(agent, "chat"):
            try:
                middle_text = []
                for m in stubbed_middle:
                    role_str = m.role.upper() if isinstance(m, ChatMessage) else "MSG"
                    content_str = m.content if isinstance(m, ChatMessage) else str(m)
                    middle_text.append(f"[{role_str}]: {content_str}")

                prompt = (
                    "Summarize the preceding conversation middle history into a concise summary.\n"
                    "Include: key decisions made, work completed, failures/errors encountered, and current state."
                )
                if focus:
                    prompt += f"\nFocus specifically on: {focus}"

                summary_turn = [ChatMessage(role="user", content="\n".join(middle_text) + "\n\n" + prompt)]
                resp = agent.chat(messages=summary_turn, stream=False)
                summary_content = getattr(resp, "content", str(resp)).strip()

                if summary_content:
                    summary_msg = ChatMessage(role="user", content=f"[Context Summary]:\n{summary_content}")
                    candidate = protected_prefix + [summary_msg] + recent_suffix
                    summary_succeeded = True
                    if not should_compact(estimate_tokens(candidate), window):
                        return candidate
            except Exception:
                summary_succeeded = False

        # Step 3: Fallback — drop oldest tool pair if summary failed or still above trigger
        if not summary_succeeded or should_compact(estimate_tokens(candidate), window):
            middle = drop_oldest_tool_pairs(middle)
            candidate = protected_prefix + stubbed_middle + recent_suffix

    return candidate
