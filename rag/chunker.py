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
from typing import Any, Dict, List, Optional

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


_PARSER_CACHE: Dict[str, Any] = {}

def get_treesitter_parser(ext: str) -> Optional[Any]:
    """Retrieve or cache a reusable tree_sitter.Parser instance per language."""
    ext_clean = ext.lower()
    lang = TREE_SITTER_LANGUAGES.get(ext_clean)
    if not lang:
        return None
    if ext_clean not in _PARSER_CACHE:
        _PARSER_CACHE[ext_clean] = tree_sitter.Parser(lang)
    return _PARSER_CACHE[ext_clean]


# --------------------------------------------------------------------------
# Python AST Chunking
# --------------------------------------------------------------------------

def chunk_python_file(
    source: str, 
    max_chunk_size: int = 1500,
    return_dicts: bool = False
) -> List[Any]:
    """Split a Python source file into semantically whole units (preamble, functions, classes).
    Includes decorators, captures module-level preambles, and splits oversized functions."""
    try:
        tree = ast.parse(source)
    except Exception:
        return []

    lines = source.splitlines()
    total_lines = len(lines)
    chunks: List[Dict[str, Any]] = []
    covered_line_indices = set()

    for node in ast.iter_child_nodes(tree):
        if not isinstance(node, (ast.FunctionDef, ast.ClassDef, ast.AsyncFunctionDef)):
            continue

        # Include all decorators preceding the definition
        start_line = node.decorator_list[0].lineno if getattr(node, "decorator_list", None) else node.lineno
        end_line = getattr(node, "end_lineno", node.lineno)

        for lno in range(start_line, end_line + 1):
            covered_line_indices.add(lno)

        segment = "\n".join(lines[start_line - 1 : end_line])
        if not segment.strip():
            continue

        if isinstance(node, ast.ClassDef):
            if len(segment) <= max_chunk_size:
                chunks.append({
                    "text": segment,
                    "kind": "class",
                    "symbol": node.name,
                    "start_line": start_line,
                    "end_line": end_line,
                })
            else:
                # Class too big — split per-method with class header context
                class_header = f"class {node.name}:\n"
                docstring = ast.get_docstring(node)
                if docstring:
                    class_header += f'    """{docstring}"""\n'

                method_found = False
                for sub_node in node.body:
                    if isinstance(sub_node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                        sub_start = sub_node.decorator_list[0].lineno if getattr(sub_node, "decorator_list", None) else sub_node.lineno
                        sub_end = getattr(sub_node, "end_lineno", sub_node.lineno)
                        sub_segment = "\n".join(lines[sub_start - 1 : sub_end])
                        if sub_segment.strip():
                            chunks.append({
                                "text": f"{class_header}\n{sub_segment}",
                                "kind": "method",
                                "symbol": f"{node.name}.{sub_node.name}",
                                "start_line": sub_start,
                                "end_line": sub_end,
                            })
                            method_found = True
                if not method_found:
                    chunks.append({
                        "text": segment,
                        "kind": "class",
                        "symbol": node.name,
                        "start_line": start_line,
                        "end_line": end_line,
                    })
        else:
            # Function or AsyncFunction
            if len(segment) <= max_chunk_size:
                chunks.append({
                    "text": segment,
                    "kind": "function",
                    "symbol": node.name,
                    "start_line": start_line,
                    "end_line": end_line,
                })
            else:
                # Split large functions with signature header preserved
                first_line = segment.splitlines()[0] if segment.splitlines() else f"def {node.name}(...):"
                fn_header = f"# in function {node.name}:\n{first_line}\n"
                body_lines = segment.splitlines()[1:]
                step = 30
                overlap = 5
                for i in range(0, len(body_lines), step - overlap):
                    window = body_lines[i : i + step]
                    if not window:
                        break
                    piece_text = fn_header + "\n".join(window)
                    chunks.append({
                        "text": piece_text,
                        "kind": "function",
                        "symbol": node.name,
                        "start_line": start_line + i,
                        "end_line": min(end_line, start_line + i + len(window)),
                    })

    # Module preamble: gather top-level statements outside def/class (imports, constants, registries)
    preamble_lines = []
    preamble_start = None
    for idx, line in enumerate(lines, 1):
        if idx not in covered_line_indices:
            if preamble_start is None:
                preamble_start = idx
            preamble_lines.append(line)
        else:
            if preamble_lines and any(l.strip() and not l.strip().startswith("#") for l in preamble_lines):
                p_text = "\n".join(preamble_lines).strip()
                if len(p_text) > 40:
                    chunks.insert(0, {
                        "text": p_text,
                        "kind": "preamble",
                        "symbol": "module_preamble",
                        "start_line": preamble_start,
                        "end_line": idx - 1,
                    })
            preamble_lines = []
            preamble_start = None

    if preamble_lines and any(l.strip() and not l.strip().startswith("#") for l in preamble_lines):
        p_text = "\n".join(preamble_lines).strip()
        if len(p_text) > 40:
            chunks.append({
                "text": p_text,
                "kind": "preamble",
                "symbol": "module_preamble",
                "start_line": preamble_start or 1,
                "end_line": total_lines,
            })

    if return_dicts:
        return chunks
    return [c["text"] for c in chunks]


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


def chunk_with_treesitter(source: str, ext: str, max_chunk_size: int = 1500) -> List[Dict[str, Any]]:
    """Parse source code with cached tree-sitter parser and extract functions, classes,
    unwrapping export statements, arrow function declarations, and structs."""
    parser = get_treesitter_parser(ext)
    if not parser:
        return chunk_fallback(source, filepath=ext)

    source_bytes = source.encode("utf-8")
    tree = parser.parse(source_bytes)
    root = tree.root_node

    chunks: List[Dict[str, Any]] = []

    for node in root.children:
        target_node = node
        node_type = node.type

        # Unwrap export statements (e.g. export function ..., export class ..., export const ... = () => {})
        if node_type == "export_statement":
            for ch in node.children:
                if ch.type in FUNCTION_NODE_TYPES or ch.type in CLASS_NODE_TYPES or ch.type in ("lexical_declaration", "variable_declaration"):
                    target_node = ch
                    node_type = ch.type
                    break

        node_text = source_bytes[node.start_byte:node.end_byte].decode("utf-8", errors="replace").strip()
        if not node_text:
            continue

        start_line = node.start_point[0] + 1
        end_line = node.end_point[0] + 1

        if node_type in FUNCTION_NODE_TYPES:
            fn_name = _get_node_name(target_node, source_bytes) or "function"
            chunks.append({
                "text": node_text,
                "kind": "function",
                "symbol": fn_name,
                "start_line": start_line,
                "end_line": end_line,
            })

        elif node_type in ("lexical_declaration", "variable_declaration"):
            # Check for arrow functions: const foo = () => { ... }
            arrow_found = False
            for ch in target_node.children:
                if ch.type == "variable_declarator":
                    var_name = _get_node_name(ch, source_bytes)
                    for sub in ch.children:
                        if sub.type in ("arrow_function", "function_expression"):
                            chunks.append({
                                "text": node_text,
                                "kind": "function",
                                "symbol": var_name or "arrow_func",
                                "start_line": start_line,
                                "end_line": end_line,
                            })
                            arrow_found = True
                            break
            if not arrow_found and len(node_text) > 30:
                chunks.append({
                    "text": node_text,
                    "kind": "preamble",
                    "symbol": "declaration",
                    "start_line": start_line,
                    "end_line": end_line,
                })

        elif node_type in ("const_declaration", "var_declaration"):
            # Go top-level const/var blocks
            chunks.append({
                "text": node_text,
                "kind": "preamble",
                "symbol": "module_constants",
                "start_line": start_line,
                "end_line": end_line,
            })

        elif node_type in CLASS_NODE_TYPES:
            class_name = _get_node_name(target_node, source_bytes) or "Container"
            if len(node_text) <= max_chunk_size:
                chunks.append({
                    "text": node_text,
                    "kind": "class",
                    "symbol": class_name,
                    "start_line": start_line,
                    "end_line": end_line,
                })
            else:
                header_line = node_text.splitlines()[0] if node_text.splitlines() else f"{node_type} {class_name}"
                parent_context = f"{header_line}\n"

                member_found = False
                body_node = target_node
                for child in target_node.children:
                    if child.type in ("class_body", "declaration_list", "field_declaration_list", "interface_body"):
                        body_node = child
                        break

                for child in body_node.children:
                    if child.type in FUNCTION_NODE_TYPES or child.type in ("method_definition", "method_declaration", "field_declaration"):
                        member_text = source_bytes[child.start_byte:child.end_byte].decode("utf-8", errors="replace").strip()
                        member_name = _get_node_name(child, source_bytes) or "method"
                        if member_text:
                            chunks.append({
                                "text": f"{parent_context}    {member_text}",
                                "kind": "method",
                                "symbol": f"{class_name}.{member_name}",
                                "start_line": child.start_point[0] + 1,
                                "end_line": child.end_point[0] + 1,
                            })
                            member_found = True

                if not member_found:
                    chunks.append({
                        "text": node_text,
                        "kind": "class",
                        "symbol": class_name,
                        "start_line": start_line,
                        "end_line": end_line,
                    })

    if not chunks and source.strip():
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


def chunk_file(filepath: str, source: str) -> List[Dict[str, Any]]:
    """Dispatcher routing files by extension to Python AST, Tree-Sitter AST, Prose, or Fallback."""
    ext = Path(filepath).suffix.lower()

    if ext == ".py":
        return chunk_python_file(source, return_dicts=True)

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


# --------------------------------------------------------------------------
# Known Symbol Extractor
# --------------------------------------------------------------------------

def extract_symbols(filepath: str, source: str) -> List[str]:
    """Extract function, class, and method symbol names from source code using AST parsing."""
    ext = Path(filepath).suffix.lower()
    symbols: List[str] = []

    if ext == ".py":
        try:
            tree = ast.parse(source)
            for node in ast.walk(tree):
                if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
                    if node.name and node.name not in symbols:
                        symbols.append(node.name)
        except Exception:
            pass

    elif ext in CODE_EXTENSIONS:
        parser = get_treesitter_parser(ext)
        if parser:
            try:
                source_bytes = source.encode("utf-8")
                tree = parser.parse(source_bytes)
                stack = [tree.root_node]
                while stack:
                    curr = stack.pop()
                    if curr.type in FUNCTION_NODE_TYPES or curr.type in CLASS_NODE_TYPES:
                        name = _get_node_name(curr, source_bytes)
                        if name and name not in symbols:
                            symbols.append(name)
                    elif curr.type == "variable_declarator":
                        for sub in curr.children:
                            if sub.type in ("arrow_function", "function_expression"):
                                name = _get_node_name(curr, source_bytes)
                                if name and name not in symbols:
                                    symbols.append(name)
                    stack.extend(reversed(curr.children))
            except Exception:
                pass

    return symbols
