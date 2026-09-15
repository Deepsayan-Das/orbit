"""
Container management tools for Orbit.

Provides container_ps and container_logs tools by shelling out directly
to host container runtimes (podman or docker CLI).
"""

import shutil
import subprocess
from typing import Any, Dict, Optional

from .registry import ToolRiskLevel, register_tool

CONTAINER_PS_SCHEMA: Dict[str, Any] = {
    "name": "container_ps",
    "description": "List active / running containers using host container CLI (podman or docker).",
    "parameters": {
        "type": "object",
        "properties": {
            "all": {
                "type": "boolean",
                "description": "If True, shows all containers including stopped ones (ps -a). Default: False."
            }
        }
    }
}

CONTAINER_LOGS_SCHEMA: Dict[str, Any] = {
    "name": "container_logs",
    "description": "Fetch recent logs from a container by name or ID using host container CLI (podman or docker).",
    "parameters": {
        "type": "object",
        "properties": {
            "name_or_id": {
                "type": "string",
                "description": "The name or container ID to fetch logs for."
            },
            "tail": {
                "type": "integer",
                "description": "Number of recent log lines to fetch (default: 50)."
            }
        },
        "required": ["name_or_id"]
    }
}


def _get_container_cli() -> Optional[str]:
    """Detect available container CLI executable (podman or docker)."""
    for cli in ("podman", "docker"):
        if shutil.which(cli):
            return cli
    return None


def container_ps(all: bool = False) -> str:
    """List containers using detected container runtime (podman or docker)."""
    cli = _get_container_cli()
    if not cli:
        return "Error: No container runtime CLI ('podman' or 'docker') found in system PATH."

    try:
        cmd = [cli, "ps", "-a"] if all else [cli, "ps"]
        proc = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=15
        )
        if proc.returncode != 0:
            return f"Container CLI ({cli}) error: {proc.stderr.strip()}"
        return proc.stdout.strip() if proc.stdout.strip() else f"No active containers found ({cli})."
    except Exception as e:
        return f"Error executing container_ps: {str(e)}"


def container_logs(name_or_id: str, tail: int = 50) -> str:
    """Fetch recent container logs using detected container runtime."""
    cli = _get_container_cli()
    if not cli:
        return "Error: No container runtime CLI ('podman' or 'docker') found in system PATH."

    if not name_or_id.strip():
        return "Error: Container name_or_id parameter cannot be empty."

    try:
        lines_count = int(tail) if tail is not None else 50
        if lines_count <= 0:
            lines_count = 50
        cmd = [cli, "logs", f"--tail={lines_count}", name_or_id.strip()]
        proc = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=20
        )
        output = proc.stdout
        if proc.stderr:
            output += ("\n" if output else "") + proc.stderr

        return output.strip() if output.strip() else f"No logs retrieved for container '{name_or_id}'."
    except Exception as e:
        return f"Error executing container_logs: {str(e)}"


def register_container_tools():
    """Register container management tools into global tool registry."""
    register_tool("container_ps", container_ps, CONTAINER_PS_SCHEMA, risk_level=ToolRiskLevel.SAFE)
    register_tool("container_logs", container_logs, CONTAINER_LOGS_SCHEMA, risk_level=ToolRiskLevel.SAFE)


register_container_tools()
