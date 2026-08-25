from __future__ import annotations

import json
from typing import Optional

import typer
from rich.console import Console
from rich.table import Table

from quantum_earth.orchestration.loop import OperatingLoop

app = typer.Typer(
    name="quantum-earth",
    help="QUANTUM EARTH — Autonomous Earth-System Prediction Intelligence Platform",
    no_args_is_help=True,
)
console = Console()
loop = OperatingLoop()


@app.command()
def status() -> None:
    """Show platform discovery + health."""
    payload = loop.status()
    console.print_json(data=payload)


@app.command()
def forecast(
    location: str = typer.Argument("Birmingham"),
    hours: int = typer.Option(48, "--hours"),
    variable: str = typer.Option("temperature_2m", "--variable", "-v"),
) -> None:
    """Probabilistic forecast for a location."""
    fc = loop.forecast(location, variable=variable, hours=hours)
    table = Table(title=f"QUANTUM EARTH · {fc['location_name']} · {variable} · {hours}h")
    table.add_column("Lead")
    table.add_column("Mean")
    table.add_column("Q10")
    table.add_column("Q90")
    for i in range(min(12, len(fc["mean"]))):
        table.add_row(
            str(fc["timestamps"][i]),
            _fmt(fc["mean"][i]),
            _fmt(fc["q10"][i]),
            _fmt(fc["q90"][i]),
        )
    console.print(table)
    console.print(f"Assurance: [bold]{fc['assurance']}[/bold] — {', '.join(fc['assurance_reasons'])}")
    console.print(f"Models: {', '.join(fc['model_ids'])}")
    console.print(f"Integrity: {fc['integrity']}")
    if len(fc["mean"]) > 12:
        console.print(f"... {len(fc['mean']) - 12} additional hours omitted from table")


@app.command("event")
def event_cmd(
    event_type: str = typer.Argument(...),
    region: Optional[str] = typer.Option(None, "--region"),
) -> None:
    """Event engine — Stage 1 returns explicit UNKNOWN / not implemented."""
    console.print(
        {
            "event_type": event_type,
            "region": region,
            "integrity": "UNKNOWN",
            "status": "NOT_IMPLEMENTED",
            "message": "Event catalogue scaffolding deferred beyond continuous-variable slice.",
        }
    )


@app.command()
def season(
    location: str = typer.Argument("London"),
    year: int = typer.Option(2027, "--year"),
) -> None:
    console.print(
        {
            "location": location,
            "year": year,
            "integrity": "UNKNOWN",
            "status": "NOT_IMPLEMENTED",
            "message": "Season engine planned for Stage 6; will be threshold-based.",
        }
    )


@app.command()
def phenology(
    location: str = typer.Argument("London"),
    event: str = typer.Option("leaf-fall", "--event"),
) -> None:
    console.print(
        {
            "location": location,
            "event": event,
            "integrity": "UNKNOWN",
            "status": "NOT_IMPLEMENTED",
            "message": "Phenology engine planned for Stage 6.",
        }
    )


@app.command()
def extremes(
    location: str = typer.Argument("London"),
    variable: str = typer.Option("snowfall", "--variable"),
) -> None:
    console.print(
        {
            "location": location,
            "variable": variable,
            "integrity": "UNKNOWN",
            "status": "NOT_IMPLEMENTED",
            "message": "Extreme-value engine not yet operational.",
        }
    )


@app.command()
def verify(
    location: str = typer.Option("Birmingham", "--location"),
    model: Optional[str] = typer.Option(None, "--model"),
    hours: int = typer.Option(24, "--hours"),
    variable: str = typer.Option("temperature_2m", "--variable"),
) -> None:
    """Verify baseline forecasts against archive ground truth."""
    report = loop.verify_recent(location, variable=variable, hours=hours)
    if model:
        scores = report.get("scores", {})
        # show specific if present
        console.print_json(data={model: scores.get(model, scores)})
    else:
        console.print_json(data=report)


models_app = typer.Typer(help="Model registry commands")
app.add_typer(models_app, name="models")


@models_app.callback(invoke_without_command=True)
def models_root(ctx: typer.Context) -> None:
    if ctx.invoked_subcommand is not None:
        return
    _print_models(leaderboard=False)


@models_app.command("leaderboard")
def models_leaderboard() -> None:
    _print_models(leaderboard=True)


def _print_models(leaderboard: bool) -> None:
    from quantum_earth.models.baselines import ModelRegistry

    reg = ModelRegistry()
    cards = reg.leaderboard("mae") if leaderboard else reg.list_models()
    table = Table(title="Model Registry")
    table.add_column("ID")
    table.add_column("Status")
    table.add_column("Family")
    table.add_column("MAE")
    for m in cards:
        table.add_row(
            m.model_id,
            m.status.value,
            m.family,
            f"{m.metrics.get('mae', float('nan')):.3f}" if m.metrics.get("mae") is not None else "—",
        )
    console.print(table)


@app.command()
def health() -> None:
    status()


@app.command()
def current(location: str = typer.Argument("Birmingham")) -> None:
    payload = loop.ingest_and_state(location)
    console.print_json(data=payload)


def _fmt(v: float | None) -> str:
    if v is None:
        return "—"
    return f"{v:.2f}"


# Typer needs a callable for setuptools script entry; expose `app` directly.
# Support: python -m quantum_earth.cli.main
if __name__ == "__main__":
    app()
