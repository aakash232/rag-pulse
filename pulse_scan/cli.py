"""pulse-scan CLI entry point."""

import os
import uuid
from pathlib import Path

import structlog
import typer

from pulse_scan.config import load_config
from pulse_scan.db.schema import open_db

app = typer.Typer(
    name="pulse",
    help="Read-only RAG corpus health scanner.",
    no_args_is_help=True,
)
log = structlog.get_logger()


def _check_gpu() -> None:
    """Exit if no GPU is available (unless dev bypass is set)."""
    if os.getenv("PULSE_SKIP_GPU_CHECK"):
        log.warning("gpu_check_bypassed", reason="PULSE_SKIP_GPU_CHECK env var set")
        return
    try:
        import torch
        if not torch.cuda.is_available():
            typer.echo(
                "ERROR: No CUDA-capable GPU detected. pulse-scan requires a GPU.\n"
                "Set up a GPU or, for local dev only, set PULSE_SKIP_GPU_CHECK=1.\n"
                "See: https://github.com/kickdrumtech/pulse-scan/docs/gpu-setup",
                err=True,
            )
            raise typer.Exit(code=1)
    except ImportError:
        typer.echo("ERROR: torch is not installed. Run: pip install pulse-scan", err=True)
        raise typer.Exit(code=1)


@app.command()
def scan(
    config: Path = typer.Option(
        Path("pulse.config.yaml"),
        "--config", "-c",
        help="Path to pulse.config.yaml",
        exists=True,
        file_okay=True,
    ),
    data_dir: Path = typer.Option(
        Path(".pulse"),
        "--data-dir",
        help="Directory for DuckDB control plane and embeddings",
    ),
) -> None:
    """Run a full corpus scan."""
    _check_gpu()

    cfg = load_config(config)

    from pulse_scan.adapters.fixture import LocalFixtureAdapter
    from pulse_scan.stages.stage0_ingest import IngestStage

    if cfg.store.type == "fixture":
        fixture_dir = cfg.fixture_dir or cfg.store.connection.get("path")
        if not fixture_dir:
            typer.echo("ERROR: fixture_dir must be set for store type 'fixture'", err=True)
            raise typer.Exit(code=1)
        adapter = LocalFixtureAdapter(fixture_dir)
    else:
        typer.echo(f"ERROR: Unsupported store type '{cfg.store.type}' in v1", err=True)
        raise typer.Exit(code=1)

    conn = open_db(data_dir)
    run_id = str(uuid.uuid4())

    log.info("scan_started", run_id=run_id, store_type=cfg.store.type)

    # Stage 0: Ingest
    stage0 = IngestStage(conn=conn, adapter=adapter, config=cfg, data_dir=data_dir)
    stats = stage0.run(run_id=run_id)

    # Stage 0.5: Calibrate (if needed)
    from pulse_scan.stages.stage05_calibrate import CalibrateStage
    calibrator = CalibrateStage(conn=conn, data_dir=data_dir)
    thresholds = None
    if calibrator.should_run():
        thresholds = calibrator.run(scan_run_id=run_id)
    else:
        from pulse_scan.stages.stage05_calibrate import load_latest_calibration
        thresholds = load_latest_calibration(conn)
        log.info("calibration_skipped_using_cached", thresholds=thresholds)

    typer.echo(
        f"Scan complete [{run_id[:8]}]\n"
        f"  new:       {stats['chunks_new']}\n"
        f"  unchanged: {stats['chunks_unchanged']}\n"
        f"  updated:   {stats['chunks_updated']}\n"
        f"  deleted:   {stats['chunks_deleted']}\n"
        f"  elapsed:   {stats['elapsed_secs']:.1f}s\n"
        f"\nCalibration thresholds:\n"
        f"  dedup cosine:          {thresholds['dedup_cosine_threshold']:.4f}\n"
        f"  contradiction cosine:  {thresholds['contradiction_candidate_threshold']:.4f}\n"
        f"  cluster density:       {thresholds['cluster_min_density']:.4f}"
    )
    conn.close()


@app.command()
def dashboard(
    data_dir: Path = typer.Option(
        Path(".pulse"),
        "--data-dir",
        help="Directory containing the DuckDB control plane",
    ),
) -> None:
    """Launch the Streamlit dashboard (Step 11)."""
    typer.echo("Dashboard not yet implemented (Step 11).")
    raise typer.Exit(code=1)


@app.command()
def calibrate(
    data_dir: Path = typer.Option(
        Path(".pulse"),
        "--data-dir",
        help="Directory containing the DuckDB control plane",
    ),
) -> None:
    """Force re-run calibration against the current corpus."""
    from pulse_scan.db.schema import open_db
    from pulse_scan.stages.stage05_calibrate import CalibrateStage

    conn = open_db(data_dir)
    run_id = str(uuid.uuid4())
    calibrator = CalibrateStage(conn=conn, data_dir=data_dir)
    thresholds = calibrator.run(scan_run_id=run_id)
    typer.echo(
        "Calibration complete:\n"
        f"  dedup cosine:          {thresholds['dedup_cosine_threshold']:.4f}\n"
        f"  contradiction cosine:  {thresholds['contradiction_candidate_threshold']:.4f}\n"
        f"  cluster density:       {thresholds['cluster_min_density']:.4f}"
    )
    conn.close()
