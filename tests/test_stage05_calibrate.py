"""Tests for Stage 0.5: Calibration."""

import json
import numpy as np
import pytest
from pathlib import Path

from pulse_scan.adapters.fixture import LocalFixtureAdapter
from pulse_scan.config import CollectionConfig, PulseConfig, StoreConfig
from pulse_scan.db.schema import open_db
from pulse_scan.stages.stage05_calibrate import (
    CalibrateStage,
    load_latest_calibration,
    _model_defaults,
)
from pulse_scan.stages.stage0_ingest import IngestStage, EmbeddingStore


def _make_config(corpus_dir: Path) -> PulseConfig:
    return PulseConfig(
        store=StoreConfig(type="fixture"),
        collections=[CollectionConfig(name="docs")],
        fixture_dir=str(corpus_dir),
    )


def _ingest(conn, corpus_dir, data_dir):
    adapter = LocalFixtureAdapter(corpus_dir)
    cfg = _make_config(corpus_dir)
    IngestStage(conn=conn, adapter=adapter, config=cfg, data_dir=data_dir).run(run_id="run-001")


# ---------------------------------------------------------------------------
# Small corpus path (model defaults)
# ---------------------------------------------------------------------------

def test_small_corpus_uses_model_defaults(corpus_dir, data_dir):
    """50 chunks < default threshold → use model defaults for 384d."""
    conn = open_db(data_dir)
    _ingest(conn, corpus_dir, data_dir)

    calibrator = CalibrateStage(conn=conn, data_dir=data_dir)
    thresholds = calibrator.run(scan_run_id="cal-001")

    expected_dedup, expected_contradiction, expected_density = _model_defaults(4)
    assert thresholds["dedup_cosine_threshold"] == pytest.approx(expected_dedup)
    assert thresholds["contradiction_candidate_threshold"] == pytest.approx(expected_contradiction)
    assert thresholds["cluster_min_density"] == pytest.approx(expected_density)
    conn.close()


def test_calibration_persisted_to_db(corpus_dir, data_dir):
    conn = open_db(data_dir)
    _ingest(conn, corpus_dir, data_dir)

    CalibrateStage(conn=conn, data_dir=data_dir).run(scan_run_id="cal-001")

    row = conn.execute(
        "SELECT scan_run_id, dedup_threshold, contradiction_candidate_threshold, cluster_min_density, distributions "
        "FROM calibration WHERE scan_run_id = 'cal-001'"
    ).fetchone()
    assert row is not None
    assert row[0] == "cal-001"
    assert 0 < row[1] <= 1.0
    assert 0 < row[2] <= 1.0
    assert 0 < row[3] <= 1.0
    dist = json.loads(row[4])
    assert "method" in dist
    assert "corpus_size" in dist
    conn.close()


def test_threshold_ordering(corpus_dir, data_dir):
    """dedup_threshold > contradiction_threshold > cluster_density (always)."""
    conn = open_db(data_dir)
    _ingest(conn, corpus_dir, data_dir)

    thresholds = CalibrateStage(conn=conn, data_dir=data_dir).run(scan_run_id="cal-001")

    assert thresholds["dedup_cosine_threshold"] > thresholds["contradiction_candidate_threshold"]
    assert thresholds["contradiction_candidate_threshold"] > thresholds["cluster_min_density"]
    conn.close()


# ---------------------------------------------------------------------------
# should_run() logic
# ---------------------------------------------------------------------------

def test_should_run_true_on_first_scan(corpus_dir, data_dir):
    conn = open_db(data_dir)
    _ingest(conn, corpus_dir, data_dir)
    calibrator = CalibrateStage(conn=conn, data_dir=data_dir)
    assert calibrator.should_run() is True
    conn.close()


def test_should_run_false_after_calibration(corpus_dir, data_dir):
    conn = open_db(data_dir)
    _ingest(conn, corpus_dir, data_dir)
    cal = CalibrateStage(conn=conn, data_dir=data_dir)
    cal.run(scan_run_id="cal-001")
    # Corpus unchanged → should not re-run
    assert cal.should_run() is False
    conn.close()


def test_should_run_true_after_50pct_growth(corpus_dir, data_dir):
    """Add enough chunks to exceed 50% growth → recalibration triggered."""
    conn = open_db(data_dir)
    _ingest(conn, corpus_dir, data_dir)
    cal = CalibrateStage(conn=conn, data_dir=data_dir)
    cal.run(scan_run_id="cal-001")

    # Directly insert extra chunk rows to simulate growth beyond 50%
    # 5 existing active chunks → need >2.5 more → insert 4 more
    from datetime import datetime, timezone
    now = datetime.now(timezone.utc).replace(tzinfo=None)
    for i in range(4):
        conn.execute(
            "INSERT INTO chunks (store_id, collection, chunk_id, text, content_hash, "
            "embedding_offset, first_seen_by_pulse, last_seen_by_pulse, version) "
            "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)",
            ["fixture:test", "docs", f"extra-{i}", f"extra chunk {i}", f"hash{i}",
             -1, now, now, 1],
        )
    assert cal.should_run() is True
    conn.close()


# ---------------------------------------------------------------------------
# load_latest_calibration helper
# ---------------------------------------------------------------------------

def test_load_latest_calibration_returns_none_when_none(data_dir):
    conn = open_db(data_dir)
    result = load_latest_calibration(conn)
    assert result is None
    conn.close()


def test_load_latest_calibration_returns_thresholds(corpus_dir, data_dir):
    conn = open_db(data_dir)
    _ingest(conn, corpus_dir, data_dir)
    CalibrateStage(conn=conn, data_dir=data_dir).run(scan_run_id="cal-001")

    result = load_latest_calibration(conn)
    assert result is not None
    assert "dedup_cosine_threshold" in result
    assert "contradiction_candidate_threshold" in result
    assert "cluster_min_density" in result
    conn.close()


# ---------------------------------------------------------------------------
# HNSW path (large corpus)
# ---------------------------------------------------------------------------

def _make_large_fixture(corpus_dir: Path, n_chunks: int = 600, dim: int = 4) -> None:
    """Write a single-collection fixture with n_chunks random chunks."""
    rng = np.random.default_rng(99)
    chunks = []
    for i in range(n_chunks):
        v = rng.standard_normal(dim).astype(float)
        v /= float(np.linalg.norm(v))
        chunks.append({
            "id": f"big-{i:04d}",
            "text": f"Large corpus chunk number {i}.",
            "embedding": v.tolist(),
            "metadata": {"created_at": "2024-01-01T00:00:00Z"},
        })
    import json
    (corpus_dir / "docs.json").write_text(json.dumps(chunks))


def test_hnsw_path_runs_and_produces_valid_thresholds(tmp_path):
    """600-chunk corpus exceeds threshold → HNSW calibration used."""
    corpus_dir = tmp_path / "corpus"
    corpus_dir.mkdir()
    data_dir = tmp_path / "data"
    data_dir.mkdir()

    _make_large_fixture(corpus_dir, n_chunks=600, dim=4)

    conn = open_db(data_dir)
    adapter = LocalFixtureAdapter(corpus_dir)
    cfg = PulseConfig(
        store=StoreConfig(type="fixture"),
        collections=[CollectionConfig(name="docs")],
        fixture_dir=str(corpus_dir),
    )
    IngestStage(conn=conn, adapter=adapter, config=cfg, data_dir=data_dir).run(run_id="r-001")

    cal = CalibrateStage(conn=conn, data_dir=data_dir, small_corpus_threshold=500)
    thresholds = cal.run(scan_run_id="cal-001")

    # All thresholds in valid range
    for key, val in thresholds.items():
        assert -1.0 <= val <= 1.0, f"{key}={val} out of range"

    # Ordering preserved
    assert thresholds["dedup_cosine_threshold"] >= thresholds["contradiction_candidate_threshold"]

    # Method stored in distributions
    row = conn.execute("SELECT distributions FROM calibration").fetchone()
    dist = json.loads(row[0])
    assert dist["method"] == "hnsw"
    assert dist["sample_size"] == 600  # N <= SAMPLE_SIZE and corpus=600
    conn.close()


def test_hnsw_sample_capped_at_2000(tmp_path):
    """Corpus > 2000 → sample_size == 2000, not full corpus."""
    corpus_dir = tmp_path / "corpus"
    corpus_dir.mkdir()
    data_dir = tmp_path / "data"
    data_dir.mkdir()

    _make_large_fixture(corpus_dir, n_chunks=2500, dim=4)

    conn = open_db(data_dir)
    adapter = LocalFixtureAdapter(corpus_dir)
    cfg = PulseConfig(
        store=StoreConfig(type="fixture"),
        collections=[CollectionConfig(name="docs")],
        fixture_dir=str(corpus_dir),
    )
    IngestStage(conn=conn, adapter=adapter, config=cfg, data_dir=data_dir).run(run_id="r-001")

    CalibrateStage(conn=conn, data_dir=data_dir, small_corpus_threshold=500).run("cal-001")

    row = conn.execute("SELECT distributions FROM calibration").fetchone()
    dist = json.loads(row[0])
    assert dist["sample_size"] == 2000
    conn.close()
