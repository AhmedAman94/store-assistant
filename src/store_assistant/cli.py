"""Command-line demo UI.

uv run store-assistant            # chat with the configured model
uv run store-assistant --demo     # chat with the rule-based demo model, no API key needed
uv run store-assistant stores     # list saved stores
uv run store-assistant conversations
uv run store-assistant show 3     # transcript and summary for conversation 3
"""

from __future__ import annotations

import json
from collections.abc import Iterable
from pathlib import Path
from typing import Annotated

import typer
from rich.console import Console
from rich.markup import escape
from rich.panel import Panel
from rich.table import Table

from store_assistant.config import Settings
from store_assistant.db import Repository, ToolCallRecord
from store_assistant.demo_model import make_demo_model, make_demo_summary_model
from store_assistant.runner import ConversationRunner, DailyBudgetExceeded
from store_assistant.tracing import configure_tracing
from store_assistant.validation import mask_phone

app = typer.Typer(help="Store Assistant: save and retrieve store phone numbers by chatting.")
console = Console()

DemoOpt = Annotated[
    bool, typer.Option("--demo", help="Use the rule-based demo model (no API key).")
]
ModelOpt = Annotated[
    str | None, typer.Option("--model", help="Override LLM_MODEL, e.g. anthropic:claude-haiku-5-5.")
]
DbOpt = Annotated[Path | None, typer.Option("--db", help="SQLite file (default from DB_PATH).")]
VerboseOpt = Annotated[
    bool, typer.Option("--verbose", "-v", help="Show the tool calls behind each reply.")
]


def _print_tool_calls(calls: Iterable[ToolCallRecord]) -> None:
    for call in calls:
        args = escape(json.dumps(call.args))
        status = call.result.get("status") if isinstance(call.result, dict) else call.result
        console.print(f"   [magenta]tool[/] {call.tool_name}({args}) -> {escape(str(status))}")


def _settings(model: str | None, db: Path | None) -> Settings:
    overrides: dict[str, object] = {}
    if model:
        overrides["llm_model"] = model
    if db:
        overrides["db_path"] = db
    return Settings(**overrides)  # type: ignore[arg-type]


@app.callback(invoke_without_command=True)
def main(
    ctx: typer.Context,
    demo: DemoOpt = False,
    model: ModelOpt = None,
    db: DbOpt = None,
    verbose: VerboseOpt = False,
) -> None:
    if ctx.invoked_subcommand is None:
        chat(demo=demo, model=model, db=db, verbose=verbose)


@app.command()
def chat(
    demo: DemoOpt = False, model: ModelOpt = None, db: DbOpt = None, verbose: VerboseOpt = False
) -> None:
    """Start an interactive conversation."""
    settings = _settings(model, db)
    tracing_on = configure_tracing(settings)

    if not demo and settings.uses_anthropic and not settings.anthropic_api_key:
        console.print(
            "[red]ANTHROPIC_API_KEY is not set.[/] Add it to a .env file (see .env.example), "
            "or run with [bold]--demo[/] to try the assistant without an API key."
        )
        raise typer.Exit(code=1)

    repo = Repository(settings.db_path)
    runner = (
        ConversationRunner(
            settings, repo, model=make_demo_model(), summary_model=make_demo_summary_model()
        )
        if demo
        else ConversationRunner(settings, repo)
    )

    try:
        greeting = runner.start()
    except DailyBudgetExceeded as exc:
        console.print(f"[red]{exc}[/] Raise DAILY_COST_LIMIT_USD or try again tomorrow.")
        raise typer.Exit(code=1) from None

    mode = "demo model (no API calls)" if demo else runner.model_name
    trace = "Logfire on" if tracing_on else "Logfire off (set LOGFIRE_TOKEN to enable)"
    console.print(
        Panel(
            f"Model: [bold]{mode}[/]  |  {trace}  |  DB: {settings.db_path}\n"
            f"Demo passphrase for lookups: [bold]{escape(settings.store_passphrase)}[/]  "
            "(set STORE_PASSPHRASE in .env to change)",
            title="Store Assistant",
        )
    )
    console.print(Panel(escape(greeting), title="Assistant", border_style="green"))

    try:
        while not runner.ended:
            try:
                user_text = console.input("[bold cyan]You[/] > ").strip()
                if not user_text:
                    continue
                with console.status("thinking..."):
                    result = runner.send(user_text)
            except (EOFError, KeyboardInterrupt):
                console.print()
                summary = runner.finish("aborted")
                console.print(Panel(escape(summary), title="Summary saved", border_style="yellow"))
                break
            if verbose:
                _print_tool_calls(result.tool_calls)
            console.print(Panel(escape(result.reply), title="Assistant", border_style="green"))
            console.print(
                f"[dim]turn {result.turn}/{settings.max_turns_per_session} | "
                f"tokens in {result.input_tokens:,} out {result.output_tokens:,} | "
                f"est. cost ${result.cost_usd:.4f} "
                f"(session cap ${settings.session_cost_limit_usd})[/]"
            )
            if result.ended:
                console.print(
                    Panel(
                        escape(result.summary or ""),
                        title=f"Summary saved (reason: {result.end_reason})",
                        border_style="yellow",
                    )
                )
    finally:
        repo.close()


@app.command()
def stores(
    db: DbOpt = None,
    show_phones: Annotated[
        bool, typer.Option("--show-phones", help="Print full phone numbers (admin view).")
    ] = False,
) -> None:
    """List saved stores. Phone numbers are masked unless --show-phones is given."""
    repo = Repository(_settings(None, db).db_path)
    try:
        table = Table("ID", "Name", "Phone (E.164)", "Updated")
        for store in repo.list_stores():
            phone = store.phone_e164 if show_phones else mask_phone(store.phone_e164)
            table.add_row(str(store.id), escape(store.name), phone, store.updated_at)
        console.print(table)
    finally:
        repo.close()


@app.command()
def conversations(db: DbOpt = None, limit: int = 20) -> None:
    """List past conversations with their summaries."""
    repo = Repository(_settings(None, db).db_path)
    try:
        table = Table("ID", "Started", "Ended", "Reason", "Cost", "Summary")
        for conv in repo.list_conversations(limit):
            table.add_row(
                str(conv.id),
                conv.started_at,
                conv.ended_at or "-",
                conv.end_reason or "-",
                f"${conv.estimated_cost_usd:.4f}",
                escape((conv.summary or "")[:120]),
            )
        console.print(table)
    finally:
        repo.close()


@app.command()
def show(conversation_id: int, db: DbOpt = None) -> None:
    """Print one conversation's transcript and summary."""
    repo = Repository(_settings(None, db).db_path)
    try:
        conv = repo.get_conversation(conversation_id)
        if conv is None:
            console.print(f"[red]No conversation with id {conversation_id}[/]")
            raise typer.Exit(code=1)
        calls = repo.get_tool_calls(conversation_id)
        for message in repo.get_transcript(conversation_id):
            style = "cyan" if message.role == "user" else "green"
            if message.role == "assistant" and message.turn > 0:
                _print_tool_calls(c for c in calls if c.turn == message.turn)
            console.print(f"[{style}]{message.role:>9}[/] | {escape(message.content)}")
        console.print(
            Panel(
                escape(conv.summary or "(no summary)"),
                title=f"Summary (reason: {conv.end_reason})",
            )
        )
    finally:
        repo.close()


if __name__ == "__main__":
    app()
