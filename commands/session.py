"""
Session state container passed to command handlers.
"""

from dataclasses import dataclass, field
from typing import Any, List, Optional

from config import OrbitConfig
from llm_client import OrbitLLM
from providers.base import ChatMessage


@dataclass
class Session:
    """Dataclass holding all state needed by slash command handlers."""
    agent: OrbitLLM
    config: OrbitConfig
    history: List[ChatMessage] = field(default_factory=list)
    collection: Any = None
    target_path: str = "./"
    max_steps: int = 8
    show_sources: bool = False
    should_exit: bool = False
