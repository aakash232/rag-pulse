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
        if not torch.cuda.is_available() and not torch.backends.mps.is_available():
            typer.echo(
                "ERROR: No GPU detected (CUDA or MPS). pulse-scan requires a GPU.\n"
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
    elif cfg.store.type == "chroma":
        from pulse_scan.adapters.chroma import ChromaAdapter
        adapter = ChromaAdapter(
            host=cfg.store.connection.get("host", "localhost"),
            port=int(cfg.store.connection.get("port", 8000)),
            collection_prefix=cfg.store.connection.get("collection_prefix", ""),
        )
    else:
        typer.echo(f"ERROR: Unsupported store type '{cfg.store.type}'. Supported: fixture, chroma", err=True)
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

    # Stage 1: Embedding-channel dedup
    from pulse_scan.stages.stage1_dedup import DeduplicateStage, TextDeduplicateStage
    dedup_stats = DeduplicateStage(conn=conn, data_dir=data_dir).run()
    text_dedup_stats = TextDeduplicateStage(conn=conn).run()

    # Stage 2: Clustering (UMAP + HDBSCAN)
    from pulse_scan.stages.stage2_cluster import ClusterStage
    cluster_stats = ClusterStage(
        conn=conn, data_dir=data_dir, clustering_config=cfg.clustering
    ).run()

    # Stage 3: Triage — select which chunks get NLI budget
    from pulse_scan.stages.stage3_triage import TriageStage
    allowed_chunk_ids, triage_stats = TriageStage(
        conn=conn, scan_run_id=run_id,
        scan_config=cfg.scan,
        collection_configs=cfg.collections,
    ).run()

    # Stage 4: NLI contradiction detection (Detector A)
    from pulse_scan.stages.stage4_nli import NLIContradictionStage
    nli_stats = NLIContradictionStage(
        conn=conn, data_dir=data_dir, scan_run_id=run_id,
        inference_config=cfg.inference, scan_config=cfg.scan,
    ).run(allowed_chunk_ids=allowed_chunk_ids)

    # Stage 9 detectors: numeric + version
    from pulse_scan.stages.stage9_detectors import (
        NumericContradictionDetector, VersionContradictionDetector,
    )
    numeric_stats = {"pairs_checked": 0, "contradictions_found": 0}
    version_stats = {"pairs_checked": 0, "contradictions_found": 0}
    if cfg.scan.enable_numeric_detector:
        numeric_stats = NumericContradictionDetector(conn=conn, scan_run_id=run_id).run()
    if cfg.scan.enable_version_detector:
        version_stats = VersionContradictionDetector(conn=conn, scan_run_id=run_id).run()

    # Stage 5: Staleness scoring
    from pulse_scan.stages.stage5_staleness import StalenessStage
    staleness_stats = StalenessStage(
        conn=conn, data_dir=data_dir,
        collection_configs=cfg.collections,
    ).run()

    # Stage 6: JSON report
    from pulse_scan.stages.stage6_report import ReportStage
    report_path = ReportStage(
        conn=conn, data_dir=data_dir, run_id=run_id,
        store_type=cfg.store.type,
    ).run()

    label_lines = "".join(
        f"  {label}:{'':>{20 - len(label)}}{count}\n"
        for label, count in staleness_stats.get("label_counts", {}).items()
    )
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
        f"  cluster density:       {thresholds['cluster_min_density']:.4f}\n"
        f"\nDeduplication:\n"
        f"  embedding groups:      {dedup_stats['groups_found']}\n"
        f"  chunks in emb groups:  {dedup_stats['chunks_in_groups']}\n"
        f"  text groups added:     {text_dedup_stats['groups_added']}\n"
        f"  emb groups +text:      {text_dedup_stats['channels_updated']}\n"
        f"\nClustering:\n"
        f"  clusters found:        {cluster_stats['clusters_found']}\n"
        f"  noise chunks:          {cluster_stats['noise_chunks']}\n"
        f"\nTriage:\n"
        f"  chunks scored:         {triage_stats['chunks_scored']}\n"
        f"  chunks allowed (NLI):  {triage_stats['chunks_allowed']}\n"
        f"  budget used:           {triage_stats['budget_used']}\n"
        f"\nContradictions:\n"
        f"  NLI pairs checked:     {nli_stats['pairs_checked']}\n"
        f"  NLI found:             {nli_stats['contradictions_found']}\n"
        f"  numeric found:         {numeric_stats['contradictions_found']}\n"
        f"  version found:         {version_stats['contradictions_found']}\n"
        f"\nStaleness scoring:\n"
        f"  chunks scored:         {staleness_stats['chunks_scored']}\n"
        + label_lines
        + f"\nReport: {report_path}"
    )
    conn.close()


@app.command()
def dashboard(
    data_dir: Path = typer.Option(
        Path(".pulse"),
        "--data-dir",
        help="Directory containing the DuckDB control plane",
    ),
    port: int = typer.Option(8501, "--port", "-p", help="Port to serve dashboard on"),
) -> None:
    """Launch the Streamlit dashboard."""
    import subprocess
    import sys
    from pulse_scan.dashboard import app as _dashboard_app

    app_path = str(Path(_dashboard_app.__file__).resolve())
    cmd = [
        sys.executable, "-m", "streamlit", "run", app_path,
        "--server.port", str(port),
        "--server.headless", "true",
        "--", "--data-dir", str(data_dir.resolve()),
    ]
    typer.echo(f"Starting dashboard on http://localhost:{port}")
    raise typer.Exit(code=subprocess.run(cmd).returncode)


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
