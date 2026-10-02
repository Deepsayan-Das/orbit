"""
Orbit Interactive Configuration Wizard.

Uses `survey` for interactive prompts (consistent with Orbit's CLI pattern).
All I/O lives here — config persistence is delegated to config.save_config().

Usage:
    python configure.py
"""

import survey

from config import OrbitConfig, load_config, save_config
from llm_client import OrbitLLM
from tools.registry import list_registered_tools

# Ensure tool registration side-effects run so the registry is populated
import tools  # noqa: F401

# ── Provider metadata ────────────────────────────────────────────────────────

# Providers that run locally / don't need an API key
_NO_KEY_PROVIDERS = {"ollama"}

# Suggested default model per provider (user can always override)
_DEFAULT_MODELS = {
    "ollama":      "llama3.2:latest",
    "openai":      "gpt-4o-mini",
    "huggingface": "Qwen/Qwen2.5-Coder-7B-Instruct",
    "gemini":      "gemini-2.5-flash",
    "groq":        "llama-3.3-70b-versatile",
}

# Environment-variable name each provider typically reads
_API_KEY_ENV_NAMES = {
    "openai":      "OPENAI_API_KEY",
    "huggingface": "HF_TOKEN",
    "gemini":      "GEMINI_API_KEY",
    "groq":        "GROQ_API_KEY",
}


def _pick_provider(providers: list[str], current: str) -> str:
    """Let the user choose a provider from the registry."""
    # Pre-select the current value if present
    default_idx = providers.index(current) if current in providers else 0
    idx = survey.routines.select(
        "Select LLM provider: ",
        options=providers,
        index=default_idx,
    )
    return providers[idx]


def _input_model(provider: str, current: str) -> str:
    """Prompt for the model name with a sensible default pre-filled."""
    default = current or _DEFAULT_MODELS.get(provider, "")
    value = survey.routines.input(
        f"Model name [{default}]: ",
    )
    return value.strip() or default


def _input_api_key(provider: str) -> str:
    """Ask for an API key (password-masked) — skipped for local providers."""
    if provider in _NO_KEY_PROVIDERS:
        return ""
    env_name = _API_KEY_ENV_NAMES.get(provider, f"{provider.upper()}_API_KEY")
    key = survey.routines.input(
        f"API key (env: {env_name}) — leave blank to use env var: ",
    )
    return key.strip()


def _input_temperature(current: float) -> float:
    """Prompt for temperature with the current value as default."""
    raw = survey.routines.input(
        f"Temperature [{current}]: ",
    )
    if not raw.strip():
        return current
    try:
        return float(raw.strip())
    except ValueError:
        print(f"  Invalid number, keeping {current}")
        return current


def _input_max_tokens(current: int) -> int:
    """Prompt for max_tokens with the current value as default."""
    raw = survey.routines.input(
        f"Max tokens [{current}]: ",
    )
    if not raw.strip():
        return current
    try:
        return int(raw.strip())
    except ValueError:
        print(f"  Invalid number, keeping {current}")
        return current


def _pick_tools(available: list[str], currently_enabled: list[str]) -> list[str]:
    """Multi-select from the tool registry; pre-check currently enabled tools."""
    if not available:
        print("  No tools registered — skipping.")
        return []

    # Build pre-selection indices
    pre_selected = [i for i, name in enumerate(available) if name in currently_enabled]

    indices = survey.routines.basket(
        "Enable tools (space to toggle, enter to confirm): ",
        options=available,
        checked=pre_selected if pre_selected else None,
    )
    return [available[i] for i in indices] if indices else []


def _input_ollama_context_size(model: str, current_win: int = 8192) -> int:
    """Prompt for Ollama context window size (num_ctx), suggesting modest default and note about ollama ps."""
    print("  [note: Raising Ollama num_ctx increases VRAM usage. Check 'ollama ps' after raising.]")
    raw = survey.routines.input(
        f"Ollama Context Window num_ctx [{current_win}]: ",
    )
    if not raw.strip():
        return current_win
    try:
        val = int(raw.strip())
        return val if val > 0 else current_win
    except ValueError:
        print(f"  Invalid number, keeping {current_win}")
        return current_win


# ── Main wizard ──────────────────────────────────────────────────────────────

def run_configure_wizard() -> OrbitConfig:
    """
    Walk the user through every configurable field and return a populated OrbitConfig.
    Starts from the current persisted config so re-running is a "patch" experience.
    """
    cfg = load_config()

    print("\n╭──────────────────────────────────────╮")
    print("│   🪐  Orbit Configuration Wizard     │")
    print("╰──────────────────────────────────────╯\n")

    # 1 — Provider
    providers = sorted(OrbitLLM._PROVIDER_REGISTRY.keys())
    cfg.provider = _pick_provider(providers, cfg.provider)

    # 2 — Model
    cfg.model = _input_model(cfg.provider, cfg.model)

    # 3 — API key (stored under providers.<name>.api_key)
    api_key = _input_api_key(cfg.provider)
    if api_key:
        cfg.providers.setdefault(cfg.provider, {})["api_key"] = api_key

    # 3b — Ollama Context Window Size prompt
    if cfg.provider == "ollama":
        print()
        curr_win = cfg.context_windows.get(cfg.model, 8192)
        new_win = _input_ollama_context_size(cfg.model, current_win=curr_win)
        cfg.context_windows[cfg.model] = new_win

    # 4 — Generation parameters
    print()
    cfg.generation["temperature"] = _input_temperature(cfg.generation.get("temperature", 0.7))
    cfg.generation["max_tokens"] = _input_max_tokens(cfg.generation.get("max_tokens", 2048))

    # 5 — Tool selection
    print()
    available_tools = list_registered_tools()
    cfg.tools["enabled"] = _pick_tools(available_tools, cfg.tools.get("enabled", []))

    return cfg


def main() -> None:
    """Entry point for `orbit configure`."""
    try:
        cfg = run_configure_wizard()
        save_config(cfg)
        print(f"\n✔ Configuration saved to {save_config.__module__}.CONFIG_PATH")
        print(f"  Provider : {cfg.provider}")
        print(f"  Model    : {cfg.model}")
        print(f"  Temp     : {cfg.generation['temperature']}")
        print(f"  MaxTok   : {cfg.generation['max_tokens']}")
        print(f"  Tools    : {', '.join(cfg.tools['enabled']) or '(none)'}")
    except KeyboardInterrupt:
        print("\n\n[cancelled] No changes saved.")


if __name__ == "__main__":
    main()
