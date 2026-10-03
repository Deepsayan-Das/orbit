"""
Orbit Terminal UI Module — Powered by Rich.

Provides rich markdown rendering, syntax highlighting, colorized diffs,
sleek status panels, and styled prompts for the Orbit CLI.
Handles legacy Windows UTF-8 stdout encoding cleanly.
"""

import sys
from typing import Any, Dict, Optional

# Force UTF-8 encoding on stdout/stderr if possible on Windows
if hasattr(sys.stdout, "reconfigure"):
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass

if hasattr(sys.stderr, "reconfigure"):
    try:
        sys.stderr.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass

from rich.console import Console
from rich.markdown import Markdown
from rich.panel import Panel
from rich.syntax import Syntax
from rich.text import Text
from rich.theme import Theme

# Custom Orbit Theme
ORBIT_THEME = Theme({
    "info": "cyan",
    "warning": "yellow",
    "error": "bold red",
    "success": "bold green",
    "banner": "bold cyan",
    "step": "bold magenta",
    "tool_name": "bold yellow",
    "orbit_prefix": "bold bright_blue",
    "user_prefix": "bold cyan",
})

console = Console(theme=ORBIT_THEME, safe_box=True)


def print_banner(
    chunk_count: int,
    tool_count: int,
    provider: str = "",
    model: str = "",
) -> None:
    """Prints the sleek Orbit welcome banner with system metadata."""
    title = "[bold bright_cyan]Orbit AI[/bold bright_cyan] [dim]- galactOS CLI Agent[/dim]"
    
    meta_parts = []
    if provider and model:
        meta_parts.append(f"[bold white]Provider:[/bold white] [cyan]{provider}[/cyan] ([dim]{model}[/dim])")
    meta_parts.append(f"[bold white]RAG Context:[/bold white] [green]{chunk_count}[/green] chunks indexed")
    meta_parts.append(f"[bold white]Tools:[/bold white] [yellow]{tool_count}[/yellow] active")
    
    meta_text = " | ".join(meta_parts)
    banner_content = f"{title}\n[dim]{meta_text}[/dim]\n\n[dim]Type 'quit' to exit, 'clear' to reset history.[/dim]"

    console.print()
    console.print(
        Panel(
            banner_content,
            border_style="cyan",
            padding=(0, 2),
            expand=False,
        )
    )
    console.print()


def print_user_prompt_label() -> str:
    """Returns the formatted prompt string for input()."""
    return "orbit > "


def print_orbit_response(content: str) -> None:
    """Renders Orbit's response formatted as Markdown with syntax highlighting."""
    console.print()
    console.print("[bold bright_blue]Orbit[/bold bright_blue]")
    if not content:
        return
    md = Markdown(content.strip())
    console.print(md)
    console.print()


def print_step_start(step: int, max_steps: int, tool_name: str, fn_args: Dict[str, Any]) -> None:
    """Renders a styled header when a tool execution step begins."""
    args_preview = ", ".join(f"{k}={repr(v)[:30]}" for k, v in fn_args.items())
    if len(args_preview) > 80:
        args_preview = args_preview[:77] + "..."
        
    step_str = f"[bold magenta]Step {step}/{max_steps}[/bold magenta]"
    tool_str = f"[bold yellow]{tool_name}[/bold yellow]({args_preview})"
    
    console.print(f" {step_str}  {tool_str}")


def print_step_result(summary: str) -> None:
    """Renders a styled summary of a tool execution result."""
    clean_summary = summary.strip().replace("\r\n", "\n")
    lines = clean_summary.split("\n")
    first_line = lines[0][:100] + ("..." if len(lines[0]) > 100 or len(lines) > 1 else "")
    
    console.print(f"   [dim]-> {first_line}[/dim]")


def print_diff(diff_text: str) -> None:
    """Colorizes and prints a unified diff (plus green, minus red, headers cyan)."""
    if not diff_text.strip():
        console.print("[dim](No changes)[/dim]")
        return

    text = Text()
    for line in diff_text.splitlines():
        if line.startswith("+++") or line.startswith("---"):
            text.append(line + "\n", style="bold cyan")
        elif line.startswith("@@"):
            text.append(line + "\n", style="bold yellow")
        elif line.startswith("+"):
            text.append(line + "\n", style="green")
        elif line.startswith("-"):
            text.append(line + "\n", style="red")
        else:
            text.append(line + "\n", style="dim")

    console.print(Panel(text, title="[bold cyan]Unified Diff Preview[/bold cyan]", border_style="cyan"))


def print_info(msg: str) -> None:
    """Print an informational message."""
    console.print(f"[cyan](i) {msg}[/cyan]")


def print_success(msg: str) -> None:
    """Print a success message."""
    console.print(f"[bold green](+) {msg}[/bold green]")


def print_warning(msg: str) -> None:
    """Print a warning message."""
    console.print(f"[bold yellow](!) {msg}[/bold yellow]")


def print_error(msg: str) -> None:
    """Print an error message."""
    console.print(f"[bold red](x) {msg}[/bold red]")


def print_step_limit_warning(steps_taken: int) -> None:
    """Print step limit reached notice."""
    console.print(
        f"\n[bold yellow](!) Step limit ({steps_taken}) reached. Type a message to continue this task, or /steps to raise the limit.[/bold yellow]\n"
    )
