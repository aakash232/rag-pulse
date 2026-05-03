"""Tests for Stage 2: Embedding-channel deduplication."""

import json
from pathlib import Path

import numpy as np
import pytest

from pulse_scan.adapters.fixture import LocalFixtureAdapter
from pulse_scan.config import CollectionConfig, PulseConfig, StoreConfig
from pulse_scan.db.schema import open_db
from pulse_scan.stages.stage0_ingest import IngestStage
from pulse_scan.stages.stage1_calibrate import CalibrateStage
from pulse_scan.stages.stage2_dedup import DeduplicateStage

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


def _u(v: list) -> list:
    """Return normalized list[float] for a DIM=4 embedding."""
    a = np.array(v, dtype=np.float32)
    return (a / np.linalg.norm(a)).tolist()


def _make_dedup_fixture(corpus_dir: Path) -> None:
    """
    6-chunk DIM=4 fixture with two near-dup pairs and two isolated chunks.

    Pair 1 — near1a / near1b:
      cosine ≈ 0.995  (near1b has newer timestamp → canonical)
    Pair 2 — near2a / near2b:
      cosine ≈ 0.995  (same timestamp; near2b has longer text → canonical)
    Isolated — solo1 / solo2:
      orthogonal to everything → never grouped
    """
    chunks = [
        {
            "id": "near1a",
            "text": "near dup cluster one version one",
            "embedding": _u([1, 0, 0, 0]),
            "metadata": {"created_at": "2023-01-01T00:00:00Z"},
        },
        {
            "id": "near1b",
            "text": "near dup cluster one version two",
            "embedding": _u([1, 0.1, 0, 0]),
            "metadata": {"created_at": "2023-02-01T00:00:00Z"},  # newer
        },
        {
            "id": "near2a",
            "text": "near dup cluster two short text",
            "embedding": _u([0, 1, 0, 0]),
            "metadata": {"created_at": "2023-01-01T00:00:00Z"},
        },
        {
            "id": "near2b",
            "text": "near dup cluster two much longer text makes this the canonical tiebreaker",
            "embedding": _u([0, 1, 0.1, 0]),
            "metadata": {"created_at": "2023-01-01T00:00:00Z"},  # same ts, longer text
        },
        {
            "id": "solo1",
            "text": "completely isolated chunk one",
            "embedding": _u([0, 0, 1, 0]),
            "metadata": {"created_at": "2023-01-01T00:00:00Z"},
        },
        {
            "id": "solo2",
            "text": "completely isolated chunk two",
            "embedding": _u([0, 0, 0, 1]),
            "metadata": {"created_at": "2023-01-01T00:00:00Z"},
        },
    ]
    (corpus_dir / "test.json").write_text(json.dumps(chunks))


def _cfg(corpus_dir: Path) -> PulseConfig:
    return PulseConfig(
        store=StoreConfig(type="fixture"),
        collections=[CollectionConfig(name="test", timestamp_field="created_at")],
        fixture_dir=str(corpus_dir),
    )


def _ingest_and_calibrate(conn, corpus_dir: Path, data_dir: Path) -> None:
    adapter = LocalFixtureAdapter(corpus_dir)
    IngestStage(conn=conn, adapter=adapter, config=_cfg(corpus_dir), data_dir=data_dir).run(run_id="run-001")
    CalibrateStage(conn=conn, data_dir=data_dir).run(scan_run_id="cal-001")


# ---------------------------------------------------------------------------
# Core grouping
# ---------------------------------------------------------------------------


def test_dedup_finds_two_groups(tmp_path):
    """Two near-dup pairs are detected; solo chunks are not grouped."""
    corpus_dir = tmp_path / "corpus"
    corpus_dir.mkdir()
    data_dir = tmp_path / "data"
    data_dir.mkdir()
    _make_dedup_fixture(corpus_dir)

    conn = open_db(data_dir)
    _ingest_and_calibrate(conn, corpus_dir, data_dir)

    result = DeduplicateStage(conn=conn, data_dir=data_dir, threshold=0.95).run()

    assert result["groups_found"] == 2
    assert result["chunks_in_groups"] == 4
    conn.close()


def test_dedup_canonical_by_timestamp(tmp_path):
    """Canonical is the newest chunk (near1b, created_at 2023-02-01)."""
    corpus_dir = tmp_path / "corpus"
    corpus_dir.mkdir()
    data_dir = tmp_path / "data"
    data_dir.mkdir()
    _make_dedup_fixture(corpus_dir)

    conn = open_db(data_dir)
    _ingest_and_calibrate(conn, corpus_dir, data_dir)
    DeduplicateStage(conn=conn, data_dir=data_dir, threshold=0.95).run()

    rows = conn.execute("SELECT canonical_chunk_id, member_chunk_ids FROM dedup_groups").fetchall()
    members_by_canonical = {r[0]: set(json.loads(r[1])) for r in rows}

    assert "near1b" in members_by_canonical
    assert members_by_canonical["near1b"] == {"near1a", "near1b"}
    conn.close()


def test_dedup_canonical_by_text_length_tiebreaker(tmp_path):
    """When timestamps tie, the chunk with longer text is canonical (near2b)."""
    corpus_dir = tmp_path / "corpus"
    corpus_dir.mkdir()
    data_dir = tmp_path / "data"
    data_dir.mkdir()
    _make_dedup_fixture(corpus_dir)

    conn = open_db(data_dir)
    _ingest_and_calibrate(conn, corpus_dir, data_dir)
    DeduplicateStage(conn=conn, data_dir=data_dir, threshold=0.95).run()

    rows = conn.execute("SELECT canonical_chunk_id, member_chunk_ids FROM dedup_groups").fetchall()
    members_by_canonical = {r[0]: set(json.loads(r[1])) for r in rows}

    assert "near2b" in members_by_canonical
    assert members_by_canonical["near2b"] == {"near2a", "near2b"}
    conn.close()


def test_dedup_detection_channel_is_embedding(tmp_path):
    """detection_channels must be ['embedding'] for Stage 1 groups."""
    corpus_dir = tmp_path / "corpus"
    corpus_dir.mkdir()
    data_dir = tmp_path / "data"
    data_dir.mkdir()
    _make_dedup_fixture(corpus_dir)

    conn = open_db(data_dir)
    _ingest_and_calibrate(conn, corpus_dir, data_dir)
    DeduplicateStage(conn=conn, data_dir=data_dir, threshold=0.95).run()

    rows = conn.execute("SELECT detection_channels FROM dedup_groups").fetchall()
    assert len(rows) == 2
    for (ch_json,) in rows:
        assert json.loads(ch_json) == ["embedding"]
    conn.close()


# ---------------------------------------------------------------------------
# Threshold sensitivity
# ---------------------------------------------------------------------------


def test_dedup_no_groups_when_threshold_above_all_cosines(tmp_path):
    """threshold=0.9999 → no pair qualifies → zero groups."""
    corpus_dir = tmp_path / "corpus"
    corpus_dir.mkdir()
    data_dir = tmp_path / "data"
    data_dir.mkdir()
    _make_dedup_fixture(corpus_dir)

    conn = open_db(data_dir)
    _ingest_and_calibrate(conn, corpus_dir, data_dir)
    result = DeduplicateStage(conn=conn, data_dir=data_dir, threshold=0.9999).run()

    assert result["groups_found"] == 0
    assert result["chunks_in_groups"] == 0
    conn.close()


# ---------------------------------------------------------------------------
# Transitive grouping
# ---------------------------------------------------------------------------


def test_dedup_transitive_three_chunk_group(tmp_path):
    """A-B and B-C above threshold → union-find merges all three into one group."""
    corpus_dir = tmp_path / "corpus"
    corpus_dir.mkdir()
    data_dir = tmp_path / "data"
    data_dir.mkdir()

    # All pairwise cosines > 0.99 at these vectors
    chunks = [
        {
            "id": "tri-a",
            "text": "triple a",
            "embedding": _u([1, 0, 0, 0]),
            "metadata": {"created_at": "2023-01-01T00:00:00Z"},
        },
        {
            "id": "tri-b",
            "text": "triple b",
            "embedding": _u([1, 0.05, 0, 0]),
            "metadata": {"created_at": "2023-01-01T00:00:00Z"},
        },
        {
            "id": "tri-c",
            "text": "triple c with the longest text here",
            "embedding": _u([1, 0.10, 0, 0]),
            "metadata": {"created_at": "2023-01-01T00:00:00Z"},
        },
    ]
    (corpus_dir / "test.json").write_text(json.dumps(chunks))

    conn = open_db(data_dir)
    adapter = LocalFixtureAdapter(corpus_dir)
    IngestStage(conn=conn, adapter=adapter, config=_cfg(corpus_dir), data_dir=data_dir).run(run_id="run-001")
    CalibrateStage(conn=conn, data_dir=data_dir).run(scan_run_id="cal-001")

    result = DeduplicateStage(conn=conn, data_dir=data_dir, threshold=0.99).run()

    assert result["groups_found"] == 1
    assert result["chunks_in_groups"] == 3

    row = conn.execute("SELECT canonical_chunk_id, member_chunk_ids FROM dedup_groups").fetchone()
    assert set(json.loads(row[1])) == {"tri-a", "tri-b", "tri-c"}
    assert row[0] == "tri-c"  # longest text
    conn.close()


# ---------------------------------------------------------------------------
# Edge cases
# ---------------------------------------------------------------------------


def test_dedup_skips_when_fewer_than_two_chunks(tmp_path):
    """Single-chunk corpus returns zeros without error."""
    corpus_dir = tmp_path / "corpus"
    corpus_dir.mkdir()
    data_dir = tmp_path / "data"
    data_dir.mkdir()
    (corpus_dir / "test.json").write_text(
        json.dumps(
            [
                {
                    "id": "only",
                    "text": "the only chunk",
                    "embedding": [1.0, 0.0, 0.0, 0.0],
                    "metadata": {"created_at": "2023-01-01T00:00:00Z"},
                },
            ]
        )
    )

    conn = open_db(data_dir)
    adapter = LocalFixtureAdapter(corpus_dir)
    IngestStage(conn=conn, adapter=adapter, config=_cfg(corpus_dir), data_dir=data_dir).run(run_id="run-001")
    CalibrateStage(conn=conn, data_dir=data_dir).run(scan_run_id="cal-001")

    result = DeduplicateStage(conn=conn, data_dir=data_dir, threshold=0.95).run()
    assert result == {"groups_found": 0, "chunks_in_groups": 0}
    conn.close()


def test_dedup_idempotent(tmp_path):
    """Running dedup twice produces the same result (DELETE+INSERT is idempotent)."""
    corpus_dir = tmp_path / "corpus"
    corpus_dir.mkdir()
    data_dir = tmp_path / "data"
    data_dir.mkdir()
    _make_dedup_fixture(corpus_dir)

    conn = open_db(data_dir)
    _ingest_and_calibrate(conn, corpus_dir, data_dir)

    stage = DeduplicateStage(conn=conn, data_dir=data_dir, threshold=0.95)
    r1 = stage.run()
    r2 = stage.run()

    assert r1 == r2
    n = conn.execute("SELECT COUNT(*) FROM dedup_groups").fetchone()[0]
    assert n == 2
    conn.close()


def test_dedup_reads_threshold_from_calibration(tmp_path):
    """Without threshold override, dedup reads from the calibration table (no raise)."""
    corpus_dir = tmp_path / "corpus"
    corpus_dir.mkdir()
    data_dir = tmp_path / "data"
    data_dir.mkdir()
    _make_dedup_fixture(corpus_dir)

    conn = open_db(data_dir)
    _ingest_and_calibrate(conn, corpus_dir, data_dir)

    result = DeduplicateStage(conn=conn, data_dir=data_dir).run()
    assert isinstance(result["groups_found"], int)
    conn.close()


def test_dedup_raises_without_calibration(tmp_path):
    """No calibration row and no override threshold → RuntimeError."""
    corpus_dir = tmp_path / "corpus"
    corpus_dir.mkdir()
    data_dir = tmp_path / "data"
    data_dir.mkdir()
    _make_dedup_fixture(corpus_dir)

    conn = open_db(data_dir)
    adapter = LocalFixtureAdapter(corpus_dir)
    IngestStage(conn=conn, adapter=adapter, config=_cfg(corpus_dir), data_dir=data_dir).run(run_id="run-001")
    # Deliberately skip calibration

    with pytest.raises(RuntimeError, match="No calibration found"):
        DeduplicateStage(conn=conn, data_dir=data_dir).run()
    conn.close()
