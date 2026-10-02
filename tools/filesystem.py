"""
Filesystem tools for Orbit.

Provides filesystem navigation, directory listing, file writing, and
targeted file editing tools registered with the tool registry.
"""

import os
from typing import Any, Dict

from .registry import ToolRiskLevel, register_tool
from .diff_utils import compute_unified_diff, validate_and_replace

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

EDIT_FILE_SCHEMA: Dict[str, Any] = {
    "name": "edit_file",
    "description": (
        "Apply a precise, targeted text edit to an existing file. "
        "Replaces exactly one occurrence of old_text with new_text. "
        "Fails safely if old_text is not found or is ambiguous (multiple matches). "
        "Requires user permission confirmation — the confirmation prompt shows a unified diff."
    ),
    "parameters": {
        "type": "object",
        "properties": {
            "path": {
                "type": "string",
                "description": "Path to the file to edit."
            },
            "old_text": {
                "type": "string",
                "description": "The exact text snippet to find (must appear exactly once)."
            },
            "new_text": {
                "type": "string",
                "description": "The replacement text."
            }
        },
        "required": ["path", "old_text", "new_text"]
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


def edit_file(path: str, old_text: str, new_text: str) -> str:
    """Apply a targeted text replacement to an existing file.

    Reads *path*, replaces exactly one occurrence of *old_text* with
    *new_text*, and writes the result back.  Returns an error message
    (without writing) if the match is missing or ambiguous.
    """
    norm_path = os.path.normpath(path)

    # ── Read ──────────────────────────────────────────────────────
    try:
        with open(norm_path, "r", encoding="utf-8") as f:
            content = f.read()
    except Exception as e:
        return f"Error reading file '{path}': {e}"

    # ── Validate & compute replacement (pure) ─────────────────────
    modified, error = validate_and_replace(content, old_text, new_text)
    if error is not None:
        return error

    # ── Write ─────────────────────────────────────────────────────
    try:
        with open(norm_path, "w", encoding="utf-8") as f:
            f.write(modified)
    except Exception as e:
        return f"Error writing file '{path}': {e}"

    diff = compute_unified_diff(content, modified, filepath=norm_path)
    return f"Successfully edited '{norm_path}'.\n\n{diff}"


def _edit_file_preview(kwargs: Dict[str, Any]) -> str:
    """preview_fn for edit_file — shows a unified diff in the confirmation prompt."""
    path = kwargs.get("path", "<unknown>")
    old_text = kwargs.get("old_text", "")
    new_text = kwargs.get("new_text", "")

    norm_path = os.path.normpath(path)

    try:
        with open(norm_path, "r", encoding="utf-8") as f:
            content = f.read()
    except Exception as e:
        return f"(cannot preview — error reading '{path}': {e})\nRaw args: {kwargs}"

    modified, error = validate_and_replace(content, old_text, new_text)
    if error is not None:
        return f"{error}\nRaw args: {kwargs}"

    diff = compute_unified_diff(content, modified, filepath=norm_path)
    return diff if diff else "(no changes)"


def register_filesystem_tools():
    """Register all filesystem tools into the global tool registry."""
    register_tool("list_directory", list_directory, LIST_DIRECTORY_SCHEMA, risk_level=ToolRiskLevel.SAFE)
    register_tool("write_file", write_file, WRITE_FILE_SCHEMA, risk_level=ToolRiskLevel.DANGEROUS)
    register_tool(
        "edit_file",
        edit_file,
        EDIT_FILE_SCHEMA,
        risk_level=ToolRiskLevel.DANGEROUS,
        preview_fn=_edit_file_preview,
    )


# Disabled auto-registration for safety
register_filesystem_tools()

