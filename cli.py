"""
Intelligent Analytics Query Engine — CLI

Usage:
    python cli.py query "Total sales in India for March"
    python cli.py batch dataset/nl_queries.json --output outputs/results.json
    python cli.py review
    python cli.py feedback-mark 0 --correct --note "Verified in BI tool"
"""

import json
import sys
from pathlib import Path

# Ensure UTF-8 output on Windows consoles
if sys.platform == "win32":
    try:
        if hasattr(sys.stdout, "reconfigure"):
            sys.stdout.reconfigure(encoding="utf-8")
        if hasattr(sys.stderr, "reconfigure"):
            sys.stderr.reconfigure(encoding="utf-8")
    except Exception:
        pass

import typer
from rich.console import Console
from rich.table import Table
from rich.panel import Panel
import pandas as pd

from IAQE.engine import QueryEngine

app = typer.Typer(
    name="iaqe",
    help="Intelligent Analytics Query Engine — translate natural language to verified analytics.",
)
console = Console(legacy_windows=False)

# Default paths (relative to project root)
DEFAULT_CSV_PATHS = ["dataset/sales_data.csv", "dataset/targets.csv"]
DEFAULT_DATA_DICT = "dataset/data_dictionary.json"
DEFAULT_FEEDBACK_LOG = "feedback_log.csv"


def _get_engine() -> QueryEngine:
    """Initialize the QueryEngine with default dataset paths."""
    return QueryEngine(
        csv_paths=DEFAULT_CSV_PATHS,
        data_dict_path=DEFAULT_DATA_DICT,
        feedback_log_path=DEFAULT_FEEDBACK_LOG,
    )


def _load_json_robust(path: str):
    """Load JSON file, handling the quirky quoted-line format."""
    with open(path, "r", encoding="utf-8-sig") as f:
        raw = f.read()

    # Try standard JSON first
    try:
        return json.loads(raw)
    except json.JSONDecodeError:
        pass

    # Fallback: strip outer quotes, un-double internal quotes per line
    lines = []
    for line in raw.splitlines():
        stripped = line.strip()
        if stripped.startswith('"') and stripped.endswith('"') and len(stripped) > 2:
            inner = stripped[1:-1].replace('""', '"')
            lines.append(inner)
        else:
            lines.append(stripped)

    return json.loads("\n".join(lines))


def _display_result(result: dict):
    """Pretty-print a query result using Rich."""
    # Header
    console.print(Panel(
        f"[bold cyan]{result['query']}[/bold cyan]",
        title="Query",
    ))

    # Confidence badge
    score = result["confidence_score"]
    if score >= 0.85:
        badge = f"[bold green][HIGH] ({score})[/bold green]"
    elif score >= 0.65:
        badge = f"[bold yellow][MEDIUM] ({score})[/bold yellow]"
    elif score >= 0.40:
        badge = f"[bold orange1][LOW] ({score})[/bold orange1]"
    else:
        badge = f"[bold red][UNRELIABLE] ({score})[/bold red]"

    console.print(f"\nConfidence: {badge}")
    console.print(
        f"Track: [dim]{result['metadata']['track_used']}[/dim]  |  "
        f"Complexity: [dim]{result['metadata']['complexity']}[/dim]  |  "
        f"Time: [dim]{result['metadata']['processing_time_ms']}ms[/dim]\n"
    )

    # Generated Logic
    console.print(Panel(
        result["generated_logic"] or "[dim]No code generated[/dim]",
        title="Generated Logic",
    ))

    # Result Table
    if result["result"]:
        df = pd.DataFrame(result["result"])
        table = Table(title="Result", show_lines=True)
        for col in df.columns:
            table.add_column(str(col), style="cyan")
        for _, row in df.head(20).iterrows():
            table.add_row(*[str(v) for v in row.values])
        console.print(table)
    else:
        console.print("[dim]No results returned.[/dim]")

    # Explanation
    explanation = result["explanation"]
    console.print(Panel(
        f"[bold]Understood:[/bold] {explanation['understood']}\n\n"
        f"[bold]Approach:[/bold] {explanation['approach']}\n\n"
        f"[bold]Caveats:[/bold] {explanation['caveats']}",
        title="Explanation",
    ))


# ═════════════════════════════════════════════════════════════════════════
#  Commands
# ═════════════════════════════════════════════════════════════════════════


@app.command()
def query(
    question: str = typer.Argument(..., help="Natural language analytics question"),
    output: str = typer.Option(None, "--output", "-o", help="Save JSON output to file"),
):
    """Run a single natural language query."""
    engine = _get_engine()

    with console.status("[bold green]Processing query..."):
        result = engine.run(question)

    _display_result(result)

    if output:
        Path(output).parent.mkdir(parents=True, exist_ok=True)
        with open(output, "w") as f:
            json.dump(result, f, indent=2, default=str)
        console.print(f"\n[green]Result saved to {output}[/green]")


@app.command()
def batch(
    queries_file: str = typer.Argument(..., help="Path to nl_queries.json"),
    output: str = typer.Option(
        "outputs/results.json", "--output", "-o", help="Output file path"
    ),
):
    """Batch process all queries from a JSON file."""
    queries_data = _load_json_robust(queries_file)
    engine = _get_engine()
    results = []

    for i, q in enumerate(queries_data):
        query_text = q["query"]
        console.print(f"\n[bold]Query {i + 1}/{len(queries_data)}:[/bold] {query_text}")

        with console.status("[green]Processing..."):
            result = engine.run(query_text)

        results.append(result)

        # Brief inline summary
        score = result["confidence_score"]
        track = result["metadata"]["track_used"]
        console.print(f"  → Confidence: {score} | Track: {track}")

    # Save results
    Path(output).parent.mkdir(parents=True, exist_ok=True)
    with open(output, "w") as f:
        json.dump(results, f, indent=2, default=str)

    # Summary statistics
    console.print(f"\n[bold green][SUCCESS] {len(results)} results saved to {output}[/bold green]")
    scores = [r["confidence_score"] for r in results]
    avg = sum(scores) / len(scores) if scores else 0
    console.print(f"  Average confidence: {avg:.3f}")
    console.print(f"  High (>=0.85):       {sum(1 for s in scores if s >= 0.85)}")
    console.print(f"  Medium (0.65-0.84):  {sum(1 for s in scores if 0.65 <= s < 0.85)}")
    console.print(f"  Low (<0.65):         {sum(1 for s in scores if s < 0.65)}")


@app.command()
def review():
    """Review pending feedback entries."""
    log_path = Path(DEFAULT_FEEDBACK_LOG)
    if not log_path.exists():
        console.print("[yellow]No feedback log found. Run some queries first.[/yellow]")
        return

    df = pd.read_csv(log_path)
    pending = df[df["feedback"] == "pending"]

    if pending.empty:
        console.print("[green]No pending entries to review.[/green]")
        return

    table = Table(title=f"Pending Reviews ({len(pending)})", show_lines=True)
    table.add_column("ID", style="dim")
    table.add_column("Query", style="cyan")
    table.add_column("Confidence", style="yellow")
    table.add_column("Track", style="dim")

    for idx, row in pending.iterrows():
        table.add_row(
            str(idx),
            str(row["query"])[:60],
            str(row["confidence_score"]),
            str(row["track_used"]),
        )

    console.print(table)


@app.command("feedback-mark")
def feedback_mark(
    entry_id: int = typer.Argument(..., help="Row ID of the entry to mark"),
    correct: bool = typer.Option(False, "--correct", help="Mark as correct"),
    incorrect: bool = typer.Option(False, "--incorrect", help="Mark as incorrect"),
    note: str = typer.Option("", "--note", "-n", help="Optional feedback note"),
):
    """Mark a feedback entry as correct or incorrect."""
    log_path = Path(DEFAULT_FEEDBACK_LOG)
    if not log_path.exists():
        console.print("[red]No feedback log found.[/red]")
        return

    df = pd.read_csv(log_path)

    if entry_id >= len(df):
        console.print(f"[red]Entry ID {entry_id} not found.[/red]")
        return

    if correct:
        df.at[entry_id, "feedback"] = "correct"
    elif incorrect:
        df.at[entry_id, "feedback"] = "incorrect"
    else:
        console.print("[yellow]Specify --correct or --incorrect[/yellow]")
        return

    if note:
        df.at[entry_id, "feedback_note"] = note

    df.to_csv(log_path, index=False)
    status = "correct" if correct else "incorrect"
    console.print(f"[green][OK] Entry {entry_id} marked as {status}[/green]")


@app.command()
def stats(
    results_file: str = typer.Argument(..., help="Path to batch results JSON"),
):
    """Show confidence distribution across a batch run."""
    with open(results_file, "r") as f:
        results = json.load(f)

    scores = [r["confidence_score"] for r in results]

    table = Table(title="Confidence Distribution", show_lines=True)
    table.add_column("Band", style="bold")
    table.add_column("Count", style="cyan")
    table.add_column("Queries", style="dim")

    bands = [
        ("[bold green]High (>=0.85)[/bold green]", [r for r in results if r["confidence_score"] >= 0.85]),
        ("[bold yellow]Medium (0.65-0.84)[/bold yellow]", [r for r in results if 0.65 <= r["confidence_score"] < 0.85]),
        ("[bold orange1]Low (0.40-0.64)[/bold orange1]", [r for r in results if 0.40 <= r["confidence_score"] < 0.65]),
        ("[bold red]Unreliable (<0.40)[/bold red]", [r for r in results if r["confidence_score"] < 0.40]),
    ]

    for label, group in bands:
        queries = ", ".join(r["query"][:40] for r in group) if group else "-"
        table.add_row(label, str(len(group)), queries)

    console.print(table)
    console.print(f"\n  Total: {len(scores)}  |  Average: {sum(scores)/len(scores):.3f}")


if __name__ == "__main__":
    app()
