"""
Orbit Multi-Language Chunker & Summary Generator.

Provides AST-aware chunking for Python, tree-sitter AST chunking for major
programming languages (Go, JS, TS, Rust, Java, C, C++), heading/paragraph
prose chunking for Markdown/text, sliding window fallback chunking, and
deterministic module-level summary generation.
"""

import ast
import os
import re
from pathlib import Path
from typing import Dict, List, Optional

# Tree-sitter imports & language binding setup
import tree_sitter

TREE_SITTER_LANGUAGES = {}

try:
    import tree_sitter_go
    TREE_SITTER_LANGUAGES[".go"] = tree_sitter.Language(tree_sitter_go.language())
except ImportError:
    pass

try:
    import tree_sitter_javascript
    js_lang = tree_sitter.Language(tree_sitter_javascript.language())
    TREE_SITTER_LANGUAGES[".js"] = js_lang
    TREE_SITTER_LANGUAGES[".jsx"] = js_lang
except ImportError:
    pass

try:
    import tree_sitter_typescript
    TREE_SITTER_LANGUAGES[".ts"] = tree_sitter.Language(tree_sitter_typescript.language_typescript())
    TREE_SITTER_LANGUAGES[".tsx"] = tree_sitter.Language(tree_sitter_typescript.language_tsx())
except ImportError:
    pass

try:
    import tree_sitter_rust
    TREE_SITTER_LANGUAGES[".rs"] = tree_sitter.Language(tree_sitter_rust.language())
except ImportError:
    pass

try:
    import tree_sitter_java
    TREE_SITTER_LANGUAGES[".java"] = tree_sitter.Language(tree_sitter_java.language())
except ImportError:
    pass

try:
    import tree_sitter_c
    TREE_SITTER_LANGUAGES[".c"] = tree_sitter.Language(tree_sitter_c.language())
except ImportError:
    pass

try:
    import tree_sitter_cpp
    cpp_lang = tree_sitter.Language(tree_sitter_cpp.language())
    TREE_SITTER_LANGUAGES[".cpp"] = cpp_lang
    TREE_SITTER_LANGUAGES[".hpp"] = cpp_lang
    TREE_SITTER_LANGUAGES[".h"] = cpp_lang
    TREE_SITTER_LANGUAGES[".cc"] = cpp_lang
except ImportError:
    pass


# --------------------------------------------------------------------------
# Tree-sitter Chunking
# --------------------------------------------------------------------------

# Language-specific node type classifications
FUNCTION_NODE_TYPES = {
    "function_declaration", "function_definition", "method_declaration",
    "method_definition", "function_item", "arrow_function"
}

CLASS_NODE_TYPES = {
    "class_declaration", "class_specifier", "struct_specifier",
    "struct_item", "type_declaration", "type_spec", "interface_declaration",
    "enum_declaration", "impl_item", "trait_item"
}


def _get_node_name(node, source_bytes: bytes) -> Optional[str]:
    """Helper to extract identifier/name from a tree-sitter AST node."""
    for child in node.children:
        if child.type in ("identifier", "type_identifier", "field_identifier", "name"):
            return source_bytes[child.start_byte:child.end_byte].decode("utf-8", errors="replace")
    return None


def chunk_with_treesitter(source: str, ext: str, max_chunk_size: int = 1500) -> List[Dict[str, str]]:
    """Parse source code with tree-sitter and extract complete top-level functions,
    classes, or structs. If a class/struct exceeds max_chunk_size, split per-method
    with parent header attached."""
    lang = TREE_SITTER_LANGUAGES.get(ext.lower())
    if not lang:
        return chunk_fallback(source, filepath=ext)

    parser = tree_sitter.Parser(lang)
    source_bytes = source.encode("utf-8")
    tree = parser.parse(source_bytes)
    root = tree.root_node

    chunks: List[Dict[str, str]] = []

    for node in root.children:
        node_type = node.type
        node_text = source_bytes[node.start_byte:node.end_byte].decode("utf-8", errors="replace").strip()

        if not node_text:
            continue

        if node_type in FUNCTION_NODE_TYPES:
            chunks.append({"text": node_text, "kind": "function"})

        elif node_type in CLASS_NODE_TYPES:
            if len(node_text) <= max_chunk_size:
                chunks.append({"text": node_text, "kind": "class"})
            else:
                # Class/struct too big: split members with parent context
                node_name = _get_node_name(node, source_bytes) or "Container"
                header_line = node_text.splitlines()[0] if node_text.splitlines() else f"{node_type} {node_name}"
                parent_context = f"{header_line}\n"

                member_found = False
                # Inspect children for methods/functions
                body_node = node
                for child in node.children:
                    if child.type in ("class_body", "declaration_list", "field_declaration_list", "interface_body"):
                        body_node = child
                        break

                for child in body_node.children:
                    if child.type in FUNCTION_NODE_TYPES or child.type in ("method_definition", "method_declaration", "field_declaration"):
                        member_text = source_bytes[child.start_byte:child.end_byte].decode("utf-8", errors="replace").strip()
                        if member_text:
                            chunks.append({
                                "text": f"{parent_context}    {member_text}",
                                "kind": "method"
                            })
                            member_found = True

                if not member_found:
                    chunks.append({"text": node_text, "kind": "class"})

    if not chunks and source.strip():
        # Fallback if no top-level functions/classes detected in tree-sitter AST
        return chunk_fallback(source, filepath=f"source{ext}")

    return chunks


# --------------------------------------------------------------------------
# Prose Chunking
# --------------------------------------------------------------------------

def chunk_prose(source: str) -> List[Dict[str, str]]:
    """Split markdown / prose documents by headings (#, ##, etc.) or paragraph breaks."""
    lines = source.splitlines()
    sections: List[str] = []
    current_section: List[str] = []

    has_headings = any(re.match(r"^#{1,6}\s+", line.strip()) for line in lines)

    if has_headings:
        for line in lines:
            if re.match(r"^#{1,6}\s+", line.strip()) and current_section:
                sections.append("\n".join(current_section).strip())
                current_section = [line]
            else:
                current_section.append(line)
        if current_section:
            sections.append("\n".join(current_section).strip())
    else:
        # Fallback to paragraph splitting (\n\n)
        paragraphs = source.split("\n\n")
        sections = [p.strip() for p in paragraphs if p.strip()]

    return [{"text": sec, "kind": "prose_section"} for sec in sections if sec]


# --------------------------------------------------------------------------
# Fallback Chunking
# --------------------------------------------------------------------------

def chunk_fallback(source: str, chunk_size: int = 500, overlap: int = 50, filepath: str = "") -> List[Dict[str, str]]:
    """Character-count sliding window fallback for unrecognized file extensions."""
    if filepath:
        print(f"[warn] using fallback chunker for {filepath}")

    source = source.strip()
    if not source:
        return []

    chunks: List[Dict[str, str]] = []
    start = 0
    length = len(source)

    while start < length:
        end = min(start + chunk_size, length)
        chunk_str = source[start:end].strip()
        if chunk_str:
            chunks.append({"text": chunk_str, "kind": "fallback"})
        if end == length:
            break
        start += (chunk_size - overlap)

    return chunks


# --------------------------------------------------------------------------
# Dispatcher
# --------------------------------------------------------------------------

PROSE_EXTENSIONS = {".md", ".txt", ".rst"}
CODE_EXTENSIONS = {".go", ".js", ".jsx", ".ts", ".tsx", ".rs", ".java", ".c", ".cpp", ".hpp", ".h", ".cc"}


def chunk_file(filepath: str, source: str) -> List[Dict[str, str]]:
    """Dispatcher routing files by extension to Python AST, Tree-Sitter AST, Prose, or Fallback."""
    ext = Path(filepath).suffix.lower()

    if ext == ".py":
        # Import Python AST chunker from orbit_repl or local definition
        from orbit_repl import chunk_python_file
        raw_chunks = chunk_python_file(source)
        result: List[Dict[str, str]] = []
        for c in raw_chunks:
            if c.startswith("class ") and "\n    def " in c and len(c) > 1500:
                kind = "method"
            elif c.startswith("class "):
                kind = "class"
            else:
                kind = "function"
            result.append({"text": c, "kind": kind})
        return result

    elif ext in CODE_EXTENSIONS:
        return chunk_with_treesitter(source, ext)

    elif ext in PROSE_EXTENSIONS:
        return chunk_prose(source)

    else:
        return chunk_fallback(source, filepath=filepath)


# --------------------------------------------------------------------------
# Module Summary Generator
# --------------------------------------------------------------------------

def build_module_summary(filepath: str, source: str) -> str:
    """Generate a cheap, deterministic module/file-level summary without LLM calls.
    Includes top-level docstrings/comments + top-level function/class symbols."""
    ext = Path(filepath).suffix.lower()
    filename = Path(filepath).name

    if ext == ".py":
        try:
            tree = ast.parse(source)
            docstring = ast.get_docstring(tree) or "No module docstring."
            top_symbols = []
            for node in ast.iter_child_nodes(tree):
                if isinstance(node, ast.FunctionDef):
                    top_symbols.append(f"def {node.name}")
                elif isinstance(node, ast.AsyncFunctionDef):
                    top_symbols.append(f"async def {node.name}")
                elif isinstance(node, ast.ClassDef):
                    top_symbols.append(f"class {node.name}")
            symbols_str = ", ".join(top_symbols) if top_symbols else "None"
            return f"Module Summary: {filename}\nDocstring: {docstring}\nTop-level symbols: {symbols_str}"
        except Exception:
            pass

    elif ext in CODE_EXTENSIONS:
        # Extract leading comments
        comment_lines = []
        for line in source.splitlines():
            line_str = line.strip()
            if line_str.startswith("//") or line_str.startswith("/*") or line_str.startswith("*") or line_str.startswith("#"):
                comment_lines.append(line_str)
            elif not line_str:
                continue
            else:
                break
        leading_comments = "\n".join(comment_lines) if comment_lines else "No leading comments."

        # Extract top-level symbols from chunk_file
        chunks = chunk_file(filepath, source)
        symbol_names = []
        for c in chunks:
            first_line = c["text"].splitlines()[0] if c["text"].splitlines() else ""
            if first_line:
                symbol_names.append(first_line[:60])
        symbols_str = "; ".join(symbol_names) if symbol_names else "None"
        return f"File Summary: {filename}\nComments:\n{leading_comments}\nTop-level symbols: {symbols_str}"

    elif ext in PROSE_EXTENSIONS:
        lines = [line.strip() for line in source.splitlines() if line.strip()]
        first_heading = lines[0] if lines else f"# {filename}"
        first_paragraph = ""
        for line in lines[1:]:
            if not line.startswith("#"):
                first_paragraph = line
                break
        return f"Document Summary: {filename}\nHeading: {first_heading}\nPreview: {first_paragraph}"

    # Fallback summary for unrecognized files
    preview = source[:300].strip()
    return f"File Summary: {filename}\nPreview:\n{preview}"
