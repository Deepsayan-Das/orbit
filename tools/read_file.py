"""
Read File tool for Orbit.

Provides safe file reading capabilities registered with the tool registry.
"""

import os
from typing import Any, Dict
from .registry import ToolRiskLevel, register_tool

READ_FILE_SCHEMA: Dict[str, Any] = {
    "name": "read_file",
    "description": "Read the text content of a file from disk.",
    "parameters": {
        "type": "object",
        "properties": {
            "path": {
                "type": "string",
                "description": "The path to the file to read."
            }
        },
        "required": ["path"]
    }
}


def read_file(path: str) -> str:
    """Read the text content of a file from disk."""
    try:
        if not os.path.exists(path):
            return f"Error: File '{path}' does not exist."
        with open(path, "r", encoding="utf-8", errors="replace") as f:
            return f.read()
    except Exception as e:
        return f"Error reading file '{path}': {str(e)}"


def register_read_file_tool():
    """Register read_file tool into global tool registry."""
    register_tool("read_file", read_file, READ_FILE_SCHEMA, risk_level=ToolRiskLevel.SAFE)


register_read_file_tool()