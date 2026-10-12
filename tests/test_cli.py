"""CLI tests through Typer's runner, using the demo model so no API key is needed."""

from __future__ import annotations

from pathlib import Path

from typer.testing import CliRunner

from store_assistant.cli import app

runner = CliRunner()


def test_demo_chat_survives_markup_in_user_text_and_saves_summary(tmp_path: Path) -> None:
    db = tmp_path / "cli.db"
    result = runner.invoke(
        app,
        ["chat", "--demo", "--db", str(db)],
        input="save [/b]Mart 415-555-0102\nI'm done\n",
    )
    assert result.exit_code == 0, result.output
    assert "[/b]Mart" in result.output
    assert "Summary saved" in result.output

    shown = runner.invoke(app, ["show", "1", "--db", str(db)])
    assert shown.exit_code == 0, shown.output
    assert "[/b]Mart" in shown.output


def test_stores_masks_phones_unless_asked(tmp_path: Path) -> None:
    db = tmp_path / "cli.db"
    runner.invoke(
        app, ["chat", "--demo", "--db", str(db)], input="save Costco 415-555-0101\ndone\n"
    )
    masked = runner.invoke(app, ["stores", "--db", str(db)])
    assert masked.exit_code == 0 and "+14155550101" not in masked.output
    assert "+*********01" in masked.output
    shown = runner.invoke(app, ["stores", "--db", str(db), "--show-phones"])
    assert "+14155550101" in shown.output


def test_verbose_shows_tool_calls_and_redacts_passphrase(tmp_path: Path) -> None:
    db = tmp_path / "cli.db"
    result = runner.invoke(
        app,
        ["chat", "--demo", "--db", str(db), "--verbose"],
        input="save Costco 415-555-0101\nlook up Costco\nopen sesame\nI'm done\n",
    )
    assert result.exit_code == 0, result.output
    assert "tool save_store" in result.output and "-> saved" in result.output
    assert "[passphrase redacted]" in result.output
    assert "open sesame" not in result.output


def test_missing_key_without_demo_explains_and_exits(tmp_path: Path) -> None:
    result = runner.invoke(app, ["chat", "--db", str(tmp_path / "x.db")])
    assert result.exit_code == 1
    assert "ANTHROPIC_API_KEY" in result.output and "--demo" in result.output
