"""
Orbit Slash Commands Package.
"""

from commands.session import Session
from commands.registry import (
    command,
    dispatch,
    get_command,
    list_commands,
    clear_registry,
)
from commands.tokens import estimate_tokens
from commands.compaction import compact_history

# Import builtin to trigger decorator registrations
import commands.builtin

__all__ = [
    "Session",
    "command",
    "dispatch",
    "get_command",
    "list_commands",
    "clear_registry",
    "estimate_tokens",
    "compact_history",
]
