"""
Context Window Resolver Module.

Pure logic for resolving maximum context window sizes per model and provider.
"""

from typing import Dict, Tuple
from config import OrbitConfig

# Known model maximum context windows (tokens)
KNOWN_MODEL_WINDOWS: Dict[str, int] = {
    "gpt-4o": 128000,
    "gpt-4o-mini": 128000,
    "gpt-4-turbo": 128000,
    "gpt-4": 8192,
    "gpt-3.5-turbo": 16384,
    "claude-3-5-sonnet": 200000,
    "claude-3-opus": 200000,
    "gemini-1.5-pro": 1000000,
    "gemini-1.5-flash": 1000000,
    "gemini-2.0-flash": 1000000,
    "llama3.2": 131072,
    "llama3.1": 131072,
    "llama3": 8192,
    "qwen2.5-coder": 32768,
    "qwen2.5": 32768,
    "qwen:0.5b": 32768,
    "deepseek-r1": 64000,
    "deepseek-coder": 64000,
    "mistral": 32768,
    "mixtral": 32768,
    "codellama": 16384,
}


def find_table_window(model: str) -> Tuple[int, bool]:
    """Find maximum window in KNOWN_MODEL_WINDOWS by exact or prefix/substring match.

    Returns (max_window, found_in_table).
    """
    model_lower = model.strip().lower()
    
    # 1. Exact match
    if model_lower in KNOWN_MODEL_WINDOWS:
        return KNOWN_MODEL_WINDOWS[model_lower], True

    # 2. Prefix / substring match
    for k, win in KNOWN_MODEL_WINDOWS.items():
        if k in model_lower or model_lower in k:
            return win, True

    return 4096, False


def resolve_window(provider: str, model: str, config: OrbitConfig = None) -> Tuple[int, str]:
    """Resolve maximum context window tokens and source.

    Returns:
        (tokens, source) where source is "user", "table", or "fallback".
    """
    model_name = (model or "").strip()
    table_val, found_in_table = find_table_window(model_name)

    # 1. Check user explicit config keyed by model
    if config is not None and getattr(config, "context_windows", None):
        user_windows = config.context_windows or {}
        # Match case-insensitively
        user_val = None
        for k, v in user_windows.items():
            if str(k).strip().lower() == model_name.lower():
                try:
                    user_val = int(v)
                    break
                except (ValueError, TypeError):
                    pass

        if user_val is not None:
            # Clamp to table maximum if model is in table
            if found_in_table:
                return min(user_val, table_val), "user"
            return user_val, "user"

    # 2. Check built-in table
    if found_in_table:
        return table_val, "table"

    # 3. Fixed fallback
    return 4096, "fallback"
