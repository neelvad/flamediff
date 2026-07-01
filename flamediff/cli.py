"""flamediff CLI -- operate the drift monitor over a run's checkpoint stream.

Heavy imports (torch, the pipeline) are deferred into the command so `--help` stays instant.
`watch` (incremental polling) is a planned fast-follow; `report` is the batch core.
"""
from __future__ import annotations

import glob
from pathlib import Path

import typer

app = typer.Typer(add_completion=False, help="Structural drift monitoring for recsys checkpoints.")


@app.callback()
def _root() -> None:
    """flamediff: detect + attribute structural drift across a run's checkpoints."""


@app.command()
def report(
    run_dir: str = typer.Argument(..., help="A run directory containing ckpt_* checkpoints."),
    json_out: bool = typer.Option(False, "--json", help="Emit JSON instead of text."),
    md: Path | None = typer.Option(None, "--md", help="Also write a markdown report here."),
    html_out: Path | None = typer.Option(
        None, "--html", help="Also write a self-contained HTML report viewer here."),
    table: str | None = typer.Option(None, "--table", help="Restrict to one embedding table."),
    min_severity: float = typer.Option(
        1.0, "--min-severity", help="Hide events below this calibrated severity."),
    fail_on: float | None = typer.Option(
        None, "--fail-on", help="Exit nonzero if any event reaches this severity (CI gate)."),
) -> None:
    """Fuse calibrated anomalies with their attribution over all present checkpoints."""
    from flamediff import load_checkpoint
    from flamediff.report import build_report

    paths = sorted(glob.glob(f"{run_dir}/ckpt_*"))
    if len(paths) < 2:
        typer.echo(f"need >=2 checkpoints under {run_dir!r} (found {len(paths)})", err=True)
        raise typer.Exit(2)

    rep = build_report([load_checkpoint(p) for p in paths], run=run_dir,
                       table=table, min_severity=min_severity)
    typer.echo(rep.to_json() if json_out else rep.to_text())
    if md is not None:
        md.write_text(rep.to_markdown())
        typer.echo(f"wrote {md}", err=True)
    if html_out is not None:
        html_out.write_text(rep.to_html())
        typer.echo(f"wrote {html_out}", err=True)
    if fail_on is not None and rep.worst_severity() >= fail_on:
        raise typer.Exit(1)


@app.command()
def watch(
    run_dir: str = typer.Argument(..., help="Run dir; new ckpt_* are picked up as they land."),
    interval: float = typer.Option(
        60.0, "--interval", help="Seconds between polls (a cheap glob — raise it freely)."),
    min_severity: float = typer.Option(
        1.0, "--min-severity", help="Only surface events at/above this calibrated severity."),
    table: str | None = typer.Option(None, "--table", help="Restrict to one embedding table."),
    fail_on: float | None = typer.Option(
        None, "--fail-on", help="Exit nonzero when an event reaches this severity (guard a run)."),
    max_polls: int = typer.Option(0, "--max-polls", help="Stop after N polls (0 = forever)."),
) -> None:
    """Stream NEW calibrated anomalies (with their why) as checkpoints are dropped."""
    import time

    from flamediff.report import Watcher

    watcher = Watcher(run_dir, table=table, min_severity=min_severity)
    typer.echo(f"watching {run_dir} — polling every {interval:g}s (Ctrl-C to stop)", err=True)
    polls = 0
    while True:
        for ee in watcher.poll():
            typer.echo(ee.to_line())
            if fail_on is not None and ee.severity >= fail_on:
                typer.echo(f"severity {ee.severity:.1f} ≥ {fail_on:g} — failing.", err=True)
                raise typer.Exit(1)
        polls += 1
        if max_polls and polls >= max_polls:
            break
        time.sleep(interval)


def main() -> None:
    app()


if __name__ == "__main__":
    main()
