"""Tests for Stage 0: Ingestion + Cache."""

import json
import numpy as np
import pytest
from pathlib import Path

from pulse_scan.adapters.fixture import LocalFixtureAdapter
from pulse_scan.config import CollectionConfig, PulseConfig, StoreConfig
from pulse_scan.db.schema import open_db
from pulse_scan.stages.stage0_ingest import IngestStage

DIM = 4  # matches conftest


def _make_config(corpus_dir: Path) -> PulseConfig:
    return PulseConfig(
        store=StoreConfig(type="fixture"),
        collections=[CollectionConfig(name="docs", timestamp_field="created_at")],
        fixture_dir=str(corpus_dir),
    )


def _make_stage(conn, adapter, cfg, data_dir) -> IngestStage:
    return IngestStage(conn=conn, adapter=adapter, config=cfg, data_dir=data_dir)


# ---------------------------------------------------------------------------
# Basic ingestion
# ---------------------------------------------------------------------------

def test_first_scan_inserts_all_chunks(corpus_dir, data_dir):
    conn = open_db(data_dir)
    adapter = LocalFixtureAdapter(corpus_dir)
    cfg = _make_config(corpus_dir)
    stage = _make_stage(conn, adapter, cfg, data_dir)

    stats = stage.run(run_id="run-001")

    assert stats["chunks_new"] == 5
    assert stats["chunks_unchanged"] == 0
    assert stats["chunks_updated"] == 0
    assert stats["chunks_deleted"] == 0

    rows = conn.execute("SELECT chunk_id FROM chunks ORDER BY chunk_id").fetchall()
    assert [r[0] for r in rows] == [f"chunk-{i:03d}" for i in range(5)]
    conn.close()


def test_second_scan_is_idempotent(corpus_dir, data_dir):
    conn = open_db(data_dir)
    adapter = LocalFixtureAdapter(corpus_dir)
    cfg = _make_config(corpus_dir)

    _make_stage(conn, adapter, cfg, data_dir).run(run_id="run-001")
    stats = _make_stage(conn, adapter, cfg, data_dir).run(run_id="run-002")

    assert stats["chunks_new"] == 0
    assert stats["chunks_unchanged"] == 5
    assert stats["chunks_updated"] == 0
    conn.close()


def test_updated_chunk_increments_version(corpus_dir, data_dir):
    conn = open_db(data_dir)
    adapter = LocalFixtureAdapter(corpus_dir)
    cfg = _make_config(corpus_dir)

    _make_stage(conn, adapter, cfg, data_dir).run(run_id="run-001")

    # Modify chunk-002's text in the fixture
    chunks = json.loads((corpus_dir / "docs.json").read_text())
    for c in chunks:
        if c["id"] == "chunk-002":
            c["text"] = "Updated text for chunk 002."
    (corpus_dir / "docs.json").write_text(json.dumps(chunks))

    stats = _make_stage(conn, adapter, cfg, data_dir).run(run_id="run-002")

    assert stats["chunks_updated"] == 1
    assert stats["chunks_unchanged"] == 4

    row = conn.execute(
        "SELECT version, text FROM chunks WHERE chunk_id = 'chunk-002'"
    ).fetchone()
    assert row[0] == 2
    assert row[1] == "Updated text for chunk 002."
    conn.close()


def test_deleted_chunk_is_marked(corpus_dir, data_dir):
    conn = open_db(data_dir)
    adapter = LocalFixtureAdapter(corpus_dir)
    cfg = _make_config(corpus_dir)

    _make_stage(conn, adapter, cfg, data_dir).run(run_id="run-001")

    # Remove chunk-004 from the fixture
    chunks = json.loads((corpus_dir / "docs.json").read_text())
    chunks = [c for c in chunks if c["id"] != "chunk-004"]
    (corpus_dir / "docs.json").write_text(json.dumps(chunks))

    stats = _make_stage(conn, adapter, cfg, data_dir).run(run_id="run-002")

    assert stats["chunks_deleted"] == 1

    row = conn.execute(
        "SELECT deleted_at FROM chunks WHERE chunk_id = 'chunk-004'"
    ).fetchone()
    assert row[0] is not None  # deleted_at set
    conn.close()


# ---------------------------------------------------------------------------
# scan_runs table
# ---------------------------------------------------------------------------

def test_scan_run_recorded(corpus_dir, data_dir):
    conn = open_db(data_dir)
    adapter = LocalFixtureAdapter(corpus_dir)
    cfg = _make_config(corpus_dir)

    _make_stage(conn, adapter, cfg, data_dir).run(run_id="run-abc")

    row = conn.execute(
        "SELECT run_id, stats FROM scan_runs WHERE run_id = 'run-abc'"
    ).fetchone()
    assert row is not None
    assert row[0] == "run-abc"
    scan_stats = json.loads(row[1])
    assert scan_stats["chunks_new"] == 5
    conn.close()


# ---------------------------------------------------------------------------
# Embedding memmap
# ---------------------------------------------------------------------------

def test_memmap_shape(corpus_dir, data_dir):
    conn = open_db(data_dir)
    adapter = LocalFixtureAdapter(corpus_dir)
    cfg = _make_config(corpus_dir)

    _make_stage(conn, adapter, cfg, data_dir).run(run_id="run-001")

    # memmap should exist and meta should record dim + n_used
    import json as _json
    meta = _json.loads((data_dir / "embeddings.meta.json").read_text())
    assert meta["dim"] == DIM
    assert meta["n_used"] == 5

    # Check file size matches allocated capacity
    arr = np.memmap(
        data_dir / "embeddings.f32.npy",
        dtype=np.float32,
        mode="r",
        shape=(meta["n_allocated"], meta["dim"]),
    )
    assert arr.shape[1] == DIM
    # First 5 rows should contain the written embeddings
    for i in range(5):
        expected = np.full(DIM, float(i), dtype=np.float32)
        np.testing.assert_array_equal(arr[i], expected)


def test_memmap_embedding_offset_stored(corpus_dir, data_dir):
    conn = open_db(data_dir)
    adapter = LocalFixtureAdapter(corpus_dir)
    cfg = _make_config(corpus_dir)

    _make_stage(conn, adapter, cfg, data_dir).run(run_id="run-001")

    rows = conn.execute(
        "SELECT chunk_id, embedding_offset FROM chunks ORDER BY chunk_id"
    ).fetchall()
    offsets = [r[1] for r in rows]
    # Offsets should be 0..4 (sequential)
    assert sorted(offsets) == list(range(5))
    conn.close()


# ---------------------------------------------------------------------------
# Dimension mismatch detection
# ---------------------------------------------------------------------------

def test_dimension_mismatch_raises(corpus_dir, data_dir):
    conn = open_db(data_dir)
    adapter = LocalFixtureAdapter(corpus_dir)
    cfg = _make_config(corpus_dir)

    _make_stage(conn, adapter, cfg, data_dir).run(run_id="run-001")

    # Replace corpus with different-dimensional embeddings
    wrong_dim = DIM + 2
    chunks = [
        {
            "id": f"chunk-{i:03d}",
            "text": f"chunk {i}",
            "embedding": [float(i)] * wrong_dim,
            "metadata": {},
        }
        for i in range(5)
    ]
    (corpus_dir / "docs.json").write_text(json.dumps(chunks))

    with pytest.raises(RuntimeError, match="dimension"):
        _make_stage(conn, adapter, cfg, data_dir).run(run_id="run-002")
    conn.close()


# ---------------------------------------------------------------------------
# Timestamp resolution
# ---------------------------------------------------------------------------

def test_timestamp_resolved_from_metadata(corpus_dir, data_dir):
    conn = open_db(data_dir)
    adapter = LocalFixtureAdapter(corpus_dir)
    cfg = _make_config(corpus_dir)

    _make_stage(conn, adapter, cfg, data_dir).run(run_id="run-001")

    row = conn.execute(
        "SELECT timestamp_source FROM chunks WHERE chunk_id = 'chunk-000'"
    ).fetchone()
    assert row[0] == "metadata_field"
    conn.close()


def test_timestamp_fallback_when_no_field(corpus_dir, data_dir):
    # Config with no timestamp_field
    conn = open_db(data_dir)
    adapter = LocalFixtureAdapter(corpus_dir)
    cfg = PulseConfig(
        store=StoreConfig(type="fixture"),
        collections=[CollectionConfig(name="docs", timestamp_field=None)],
        fixture_dir=str(corpus_dir),
    )

    _make_stage(conn, adapter, cfg, data_dir).run(run_id="run-001")

    row = conn.execute(
        "SELECT timestamp_source FROM chunks WHERE chunk_id = 'chunk-000'"
    ).fetchone()
    # No timestamp_field, no UUIDv1/ULID → first_seen fallback
    assert row[0] == "first_seen"
    conn.close()
