# console (terminal) user feedback, the rich counterpart of gui.py
import warnings
from functools import partial
from rich.console import Console
from rich.table import Table
from rich.panel import Panel
from rich.pretty import Pretty
from rich.progress import track
from rich.traceback import install as install_traceback
from rich.markup import escape
from pandas import DataFrame

console = Console()

# spinner: `with status("Loading nch LUT..."):`
status = console.status
# progress bar: `for x in progress(items, description=...):`
progress = partial(track, console=console, transient=True)


def install():
    """Route warnings and tracebacks through rich."""
    install_traceback(console=console, show_locals=False)
    warnings.showwarning = lambda msg, cat, *_: console.print(
        f"[yellow]⚠ {cat.__name__}:[/] {escape(str(msg))}"
    )


def show_mapping(title: str, data: dict):
    console.print(Panel(Pretty(data), title=title, expand=False))


def show_results(df: DataFrame, title: str):
    """Transposed table: one row per parameter, one column per instance."""
    table = Table(title=title, header_style="bold cyan", row_styles=["", "dim"])
    table.add_column("Parameter", style="bold")
    for id in df["ID"]:
        table.add_column(escape(str(id)), justify="right")
    for col in df.columns.drop("ID"):
        # units like "[fF]" would otherwise be parsed as rich markup
        cells = ["—" if v is None else escape(str(getattr(v, "value", v))) for v in df[col]]
        table.add_row(escape(col), *cells)
    console.print(table)


def error(msg: str):
    console.print(f"[bold red]✗[/] {escape(msg)}")


def success(msg: str):
    console.print(f"[bold green]✓[/] {escape(msg)}")
