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
    if fail_on is not None and rep.worst_severity() >= fail_on:
        raise typer.Exit(1)


def main() -> None:
    app()


if __name__ == "__main__":
    main()
