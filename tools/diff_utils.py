"""
Pure diff-computation utilities for Orbit's edit tools.

All functions here are pure (no file I/O, no side effects) — they accept
strings and return strings.  File reading/writing is the caller's concern.
"""

import difflib
from typing import Optional


def compute_unified_diff(
    original: str,
    modified: str,
    filepath: str = "file",
    context_lines: int = 3,
) -> str:
    """
    Return a unified-diff string between *original* and *modified* content.

    Uses ``difflib.unified_diff`` with ``fromfile``/``tofile`` headers so the
    output looks like a standard ``diff -u``.
    """
    original_lines = original.splitlines(keepends=True)
    modified_lines = modified.splitlines(keepends=True)

    diff_lines = difflib.unified_diff(
        original_lines,
        modified_lines,
        fromfile=f"a/{filepath}",
        tofile=f"b/{filepath}",
        n=context_lines,
    )
    return "".join(diff_lines)


def validate_and_replace(
    content: str,
    old_text: str,
    new_text: str,
) -> tuple[Optional[str], Optional[str]]:
    """
    Attempt to replace *old_text* with *new_text* inside *content*.

    Returns ``(modified_content, None)`` on success, or
    ``(None, error_message)`` when the edit cannot be applied safely.

    Rules:
    * *old_text* must appear **exactly once** in *content*.
    * Zero occurrences  → error (no match).
    * Two-or-more       → error (ambiguous).
    """
    count = content.count(old_text)

    if count == 0:
        return None, (
            "Edit rejected: the provided old_text was not found in the file.  "
            "Make sure the text (including whitespace) matches exactly."
        )

    if count > 1:
        return None, (
            f"Edit rejected: old_text appears {count} times in the file — "
            "the match is ambiguous.  Provide a larger, unique snippet so "
            "exactly one location is matched."
        )

    modified = content.replace(old_text, new_text, 1)
    return modified, None
