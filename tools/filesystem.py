"""
Filesystem tools for Orbit.

Provides filesystem navigation, directory listing, and file writing tools registered with the tool registry.
"""

import os
from typing import Any, Dict

from .registry import ToolRiskLevel, register_tool

LIST_DIRECTORY_SCHEMA: Dict[str, Any] = {
    "name": "list_directory",
    "description": "List the files and folders in a given directory path.",
    "parameters": {
        "type": "object",
        "properties": {
            "path": {
                "type": "string",
                "description": "The directory path to list, e.g. '.' for current directory"
            }
        },
        "required": ["path"]
    }
}

WRITE_FILE_SCHEMA: Dict[str, Any] = {
    "name": "write_file",
    "description": "Write text content to a file on disk. Requires user permission confirmation.",
    "parameters": {
        "type": "object",
        "properties": {
            "path": {
                "type": "string",
                "description": "The path to the file to write or update."
            },
            "content": {
                "type": "string",
                "description": "The text content to write into the file."
            }
        },
        "required": ["path", "content"]
    }
}


def list_directory(path: str = ".") -> str:
    """List the files and folders in a given directory path."""
    try:
        entries = os.listdir(path)
        return "\n".join(entries) if entries else "Empty directory"
    except Exception as e:
        return f"Error: {str(e)}"


def write_file(path: str, content: str) -> str:
    """Write text content to a file on disk."""
    try:
        norm_path = os.path.normpath(path)
        parent = os.path.dirname(norm_path)
        if parent:
            os.makedirs(parent, exist_ok=True)

        with open(norm_path, "w", encoding="utf-8") as f:
            f.write(content)

        return f"Successfully wrote {len(content)} bytes to '{norm_path}'."
    except Exception as e:
        return f"Error writing file '{path}': {str(e)}"


def register_filesystem_tools():
    """Register all filesystem tools into the global tool registry."""
    register_tool("list_directory", list_directory, LIST_DIRECTORY_SCHEMA, risk_level=ToolRiskLevel.SAFE)
    register_tool("write_file", write_file, WRITE_FILE_SCHEMA, risk_level=ToolRiskLevel.SENSITIVE)


# Disabled auto-registration for safety
register_filesystem_tools()
