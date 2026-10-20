"""
Orbit Configuration — pure data layer.

Handles reading/writing ~/.orbit/config.yaml into the OrbitConfig dataclass.
No interactive I/O lives here; see configure.py for the wizard.
"""

import yaml
import os
from pathlib import Path
from dataclasses import dataclass, field, asdict

CONFIG_PATH = Path.home() / ".orbit" / "config.yaml"


@dataclass
class OrbitConfig:
    provider: str = "ollama"
    model: str = "llama3.2:latest"
    providers: dict = field(default_factory=dict)
    generation: dict = field(default_factory=lambda: {"temperature": 0.7, "max_tokens": 2048})
    tools: dict = field(default_factory=lambda: {"enabled": [], "risk_overrides": {}})
    context_windows: dict = field(default_factory=dict)
    embedding: dict = field(default_factory=lambda: {"provider": "ollama", "model": "nomic-embed-text"})


def load_config() -> OrbitConfig:
    """Load OrbitConfig from ~/.orbit/config.yaml, falling back to defaults."""
    if not CONFIG_PATH.exists():
        return OrbitConfig()

    try:
        with open(CONFIG_PATH, "r") as f:
            data = yaml.safe_load(f) or {}

        # Normalise provider name to lowercase
        if "provider" in data and isinstance(data["provider"], str):
            data["provider"] = data["provider"].lower()

        # Normalise provider keys inside the per-provider dict
        if "providers" in data and isinstance(data["providers"], dict):
            data["providers"] = {k.lower(): v for k, v in data["providers"].items()}

        # Normalise embedding configuration
        if "embedding" in data and isinstance(data["embedding"], dict):
            if "provider" in data["embedding"] and isinstance(data["embedding"]["provider"], str):
                data["embedding"]["provider"] = data["embedding"]["provider"].lower()
        else:
            data["embedding"] = {"provider": "ollama", "model": "nomic-embed-text"}

        # Filter only recognized OrbitConfig fields
        valid_fields = {f.name for f in OrbitConfig.__dataclass_fields__.values()}
        filtered_data = {k: v for k, v in data.items() if k in valid_fields}

        return OrbitConfig(**filtered_data)
    except Exception as e:
        print(f"Error loading config: {e}")
        return OrbitConfig()


def save_config(cfg: OrbitConfig) -> None:
    """Persist an OrbitConfig back to ~/.orbit/config.yaml, creating the directory if needed."""
    CONFIG_PATH.parent.mkdir(parents=True, exist_ok=True)

    data = asdict(cfg)
    with open(CONFIG_PATH, "w") as f:
        yaml.dump(data, f, default_flow_style=False, sort_keys=False)


if __name__ == "__main__":
    cfg = load_config()
    print(cfg)