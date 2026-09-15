"""
Unit tests for Orbit chunker dispatcher, Tree-Sitter parsing, prose chunking,
fallback chunking, and module summary generation.
"""

import unittest
from chunker import (
    build_module_summary,
    chunk_file,
    chunk_fallback,
    chunk_prose,
    chunk_with_treesitter,
    extract_symbols,
)
from orbit_repl import correct_query_terms


class TestOrbitChunker(unittest.TestCase):


    def test_python_chunking(self):
        py_code = '''
def add(a, b):
    return a + b

class MathUtils:
    """Math helper class."""
    def multiply(self, a, b):
        return a * b
'''
        chunks = chunk_file("utils.py", py_code)
        self.assertGreaterEqual(len(chunks), 2)
        kinds = [c["kind"] for c in chunks]
        self.assertIn("function", kinds)
        self.assertIn("class", kinds)

    def test_go_treesitter_chunking(self):
        go_code = '''
package main

import "fmt"

func Greet(name string) string {
    return fmt.Sprintf("Hello, %s", name)
}

type User struct {
    Name string
    Age  int
}
'''
        chunks = chunk_file("main.go", go_code)
        self.assertGreaterEqual(len(chunks), 1)
        texts = " ".join([c["text"] for c in chunks])
        self.assertIn("func Greet", texts)

    def test_js_treesitter_chunking(self):
        js_code = '''
function calculateTotal(items) {
    return items.reduce((a, b) => a + b, 0);
}

class Invoice {
    constructor(id) {
        this.id = id;
    }
}
'''
        chunks = chunk_file("invoice.js", js_code)
        self.assertGreaterEqual(len(chunks), 1)
        texts = " ".join([c["text"] for c in chunks])
        self.assertIn("function calculateTotal", texts)

    def test_rust_treesitter_chunking(self):
        rs_code = '''
fn process_data(input: &str) -> String {
    input.to_uppercase()
}

struct Config {
    port: u16,
}
'''
        chunks = chunk_file("config.rs", rs_code)
        self.assertGreaterEqual(len(chunks), 1)
        texts = " ".join([c["text"] for c in chunks])
        self.assertIn("fn process_data", texts)

    def test_prose_chunking_with_headings(self):
        md_text = '''# Title

This is the introductory paragraph.

## Section 1

Details for section 1.

## Section 2

Details for section 2.
'''
        chunks = chunk_prose(md_text)
        self.assertGreaterEqual(len(chunks), 2)
        for c in chunks:
            self.assertEqual(c["kind"], "prose_section")

    def test_prose_chunking_paragraphs(self):
        txt_text = '''Paragraph 1 with some information.

Paragraph 2 with more details.

Paragraph 3 concluding the note.'''
        chunks = chunk_prose(txt_text)
        self.assertEqual(len(chunks), 3)

    def test_fallback_chunking(self):
        raw_text = "A" * 1200
        chunks = chunk_file("data.unknown", raw_text)
        self.assertGreaterEqual(len(chunks), 2)
        for c in chunks:
            self.assertEqual(c["kind"], "fallback")

    def test_build_module_summary_python(self):
        py_code = '''"""Module for managing database connections."""

def connect():
    pass

class DatabasePool:
    pass
'''
        summary = build_module_summary("db.py", py_code)
        self.assertIn("Module for managing database connections.", summary)
        self.assertIn("def connect", summary)
        self.assertIn("class DatabasePool", summary)

    def test_build_module_summary_markdown(self):
        md_code = '''# Architecture Overview

Orbit is a developer-first AI assistant built for galactOS.
'''
        summary = build_module_summary("README.md", md_code)
        self.assertIn("Heading: # Architecture Overview", summary)
        self.assertIn("Orbit is a developer-first AI assistant", summary)

    def test_extract_symbols(self):
        py_code = '''
def calculate_total(prices):
    pass

class InvoiceManager:
    def create_invoice(self):
        pass
'''
        symbols = extract_symbols("invoice.py", py_code)
        self.assertIn("calculate_total", symbols)
        self.assertIn("InvoiceManager", symbols)

    def test_correct_query_terms(self):
        known_symbols = ["calculate_total", "InvoiceManager", "chunk_python_file", "build_module_summary"]

        query1 = "how does calculte_total work?"
        corrected1 = correct_query_terms(query1, known_symbols)
        self.assertEqual(corrected1, "how does calculate_total work?")

        query2 = "explain chukn_python_file and build_module_summary"
        corrected2 = correct_query_terms(query2, known_symbols)
        self.assertEqual(corrected2, "explain chunk_python_file and build_module_summary")

    def test_correct_query_terms_does_not_mangle_english(self):
        known_symbols = ["chat", "list_directory", "ChatMessage", "stream_chat", "calculate_total"]
        query = "what files are in the current directory?"
        corrected = correct_query_terms(query, known_symbols)
        self.assertEqual(corrected, "what files are in the current directory?")


if __name__ == "__main__":
    unittest.main()


