"""
Orbit Command Registry & Dispatcher.
"""

from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional
from commands.session import Session


@dataclass
class CommandInfo:
    name: str
    handler: Callable[[str, Session], str]
    aliases: List[str] = field(default_factory=list)
    help: str = ""


_COMMAND_REGISTRY: Dict[str, CommandInfo] = {}
_ALIAS_MAP: Dict[str, str] = {}


def command(name: str, aliases: Optional[List[str]] = None, help: str = ""):
    """Decorator to register a slash command handler function."""
    aliases_list = aliases or []

    def decorator(fn: Callable[[str, Session], str]):
        cmd_info = CommandInfo(
            name=name.lower(),
            handler=fn,
            aliases=[a.lower() for a in aliases_list],
            help=help,
        )
        _COMMAND_REGISTRY[name.lower()] = cmd_info
        for alias in aliases_list:
            _ALIAS_MAP[alias.lower()] = name.lower()
        return fn

    return decorator


def get_command(name_or_alias: str) -> Optional[CommandInfo]:
    """Lookup command info by name or alias."""
    key = name_or_alias.lower()
    if key in _COMMAND_REGISTRY:
        return _COMMAND_REGISTRY[key]
    if key in _ALIAS_MAP:
        primary_name = _ALIAS_MAP[key]
        return _COMMAND_REGISTRY.get(primary_name)
    return None


def list_commands() -> List[CommandInfo]:
    """Return list of all registered primary commands."""
    return list(_COMMAND_REGISTRY.values())


def clear_registry() -> None:
    """Clear command registry and alias mapping (useful for testing)."""
    _COMMAND_REGISTRY.clear()
    _ALIAS_MAP.clear()


def dispatch(line: str, session: Session) -> str:
    """Parse and dispatch a slash command line.

    Never raises an exception — handler errors are caught and returned as error text.
    """
    try:
        raw_line = line.strip()
        if not raw_line:
            return ""

        if raw_line.startswith("/"):
            content = raw_line[1:].strip()
        else:
            content = raw_line

        parts = content.split(maxsplit=1)
        cmd_name = parts[0].lower() if parts else ""
        args = parts[1].strip() if len(parts) > 1 else ""

        cmd_info = get_command(cmd_name)
        if not cmd_info:
            display_cmd = raw_line if raw_line.startswith("/") else f"/{raw_line}"
            return f"[error: unknown command '{display_cmd}'. Type /help for available commands.]"

        try:
            return cmd_info.handler(args, session)
        except Exception as e:
            return f"[error executing /{cmd_info.name}: {e}]"

    except Exception as e:
        return f"[error dispatching command: {e}]"
