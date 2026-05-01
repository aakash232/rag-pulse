"""Tests for Stage 9 text-channel dedup (TextDeduplicateStage)."""

import json
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np
import pytest

from pulse_scan.adapters.fixture import LocalFixtureAdapter
from pulse_scan.config import CollectionConfig, PulseConfig, StoreConfig
from pulse_scan.db.schema import open_db
from pulse_scan.stages.stage0_ingest import IngestStage
from pulse_scan.stages.stage05_calibrate import CalibrateStage
from pulse_scan.stages.stage1_dedup import DeduplicateStage, TextDeduplicateStage

DIM = 8
RUN_ID = "run-text-dedup"


def _cfg(corpus_dir: Path) -> PulseConfig:
    return PulseConfig(
        store=StoreConfig(type="fixture"),
        collections=[CollectionConfig(name="docs", timestamp_field="created_at")],
        fixture_dir=str(corpus_dir),
    )


def _rand_unit(dim: int, seed: int) -> list[float]:
    rng = np.random.default_rng(seed)
    v = rng.standard_normal(dim).astype(np.float32)
    return (v / np.linalg.norm(v)).tolist()


def _ingest(conn, corpus_dir, data_dir):
    cfg = _cfg(corpus_dir)
    IngestStage(conn=conn, adapter=LocalFixtureAdapter(corpus_dir), config=cfg, data_dir=data_dir).run(RUN_ID)
    CalibrateStage(conn=conn, data_dir=data_dir).run(scan_run_id=RUN_ID)


# ---------------------------------------------------------------------------
# Tests for purely text-similar pairs (embeddings are orthogonal)
# ---------------------------------------------------------------------------

def test_text_dedup_finds_near_duplicate_text(tmp_path):
    """Two chunks with nearly identical text but orthogonal embeddings → text-only group."""
    corpus_dir = tmp_path / "corpus"; corpus_dir.mkdir()
    data_dir = tmp_path / "data"; data_dir.mkdir()

    text = "The quick brown fox jumps over the lazy dog every single day"
    chunks = [
        {"id": "a", "text": text, "embedding": _rand_unit(DIM, 0),
         "metadata": {"created_at": "2024-01-01T00:00:00Z"}},
        {"id": "b", "text": text + " and night",  # minimal change, high Jaccard
         "embedding": _rand_unit(DIM, 99),          # orthogonal to a
         "metadata": {"created_at": "2024-02-01T00:00:00Z"}},
        {"id": "c", "text": "Completely unrelated content about databases and SQL queries",
         "embedding": _rand_unit(DIM, 50),
         "metadata": {"created_at": "2024-01-15T00:00:00Z"}},
    ]
    (corpus_dir / "docs.json").write_text(json.dumps(chunks))

    conn = open_db(data_dir)
    _ingest(conn, corpus_dir, data_dir)
    # Embedding dedup at high threshold → no embedding groups
    DeduplicateStage(conn=conn, data_dir=data_dir, threshold=0.999).run()
    assert conn.execute("SELECT COUNT(*) FROM dedup_groups").fetchone()[0] == 0

    result = TextDeduplicateStage(conn=conn, threshold=0.7).run()
    assert result["groups_added"] == 1
    groups = conn.execute("SELECT * FROM dedup_groups").fetchall()
    assert len(groups) == 1
    members = set(json.loads(groups[0][2]))
    assert members == {"a", "b"}
    channels = json.loads(groups[0][3])
    assert channels == ["text"]
    conn.close()


def test_text_dedup_adds_channel_to_existing_embedding_group(tmp_path):
    """Two chunks already in same embedding group → 'text' added to detection_channels."""
    corpus_dir = tmp_path / "corpus"; corpus_dir.mkdir()
    data_dir = tmp_path / "data"; data_dir.mkdir()

    base = np.array([1, 0, 0, 0, 0, 0, 0, 0], dtype=np.float32)
    near = base.copy(); near[0] = 0.9999; near[1] = 0.014  # cosine ≈ 0.9999

    text = "The quick brown fox jumps over the lazy dog every single day of the week"
    chunks = [
        {"id": "a", "text": text, "embedding": (base / np.linalg.norm(base)).tolist(),
         "metadata": {"created_at": "2024-01-01T00:00:00Z"}},
        {"id": "b", "text": text + " and night",
         "embedding": (near / np.linalg.norm(near)).tolist(),
         "metadata": {"created_at": "2024-02-01T00:00:00Z"}},
    ]
    (corpus_dir / "docs.json").write_text(json.dumps(chunks))

    conn = open_db(data_dir)
    _ingest(conn, corpus_dir, data_dir)
    # Embedding dedup with low threshold captures the near-dup pair
    DeduplicateStage(conn=conn, data_dir=data_dir, threshold=0.99).run()
    assert conn.execute("SELECT COUNT(*) FROM dedup_groups").fetchone()[0] == 1
    channels_before = json.loads(
        conn.execute("SELECT detection_channels FROM dedup_groups").fetchone()[0]
    )
    assert channels_before == ["embedding"]

    result = TextDeduplicateStage(conn=conn, threshold=0.7).run()
    assert result["groups_added"] == 0
    assert result["channels_updated"] == 1

    channels_after = json.loads(
        conn.execute("SELECT detection_channels FROM dedup_groups").fetchone()[0]
    )
    assert set(channels_after) == {"embedding", "text"}
    conn.close()


def test_text_dedup_no_groups_for_dissimilar_text(tmp_path):
    """Chunks with low text similarity → no text groups added."""
    corpus_dir = tmp_path / "corpus"; corpus_dir.mkdir()
    data_dir = tmp_path / "data"; data_dir.mkdir()

    chunks = [
        {"id": "a", "text": "Python is a general purpose programming language",
         "embedding": _rand_unit(DIM, 1), "metadata": {"created_at": "2024-01-01T00:00:00Z"}},
        {"id": "b", "text": "The capital of France is Paris and it is beautiful",
         "embedding": _rand_unit(DIM, 2), "metadata": {"created_at": "2024-01-01T00:00:00Z"}},
    ]
    (corpus_dir / "docs.json").write_text(json.dumps(chunks))

    conn = open_db(data_dir)
    _ingest(conn, corpus_dir, data_dir)
    DeduplicateStage(conn=conn, data_dir=data_dir, threshold=0.999).run()

    result = TextDeduplicateStage(conn=conn, threshold=0.8).run()
    assert result["groups_added"] == 0
    assert conn.execute("SELECT COUNT(*) FROM dedup_groups").fetchone()[0] == 0
    conn.close()


def test_text_dedup_canonical_is_newest(tmp_path):
    """In a text-only group, canonical chunk is the one with the newest timestamp."""
    corpus_dir = tmp_path / "corpus"; corpus_dir.mkdir()
    data_dir = tmp_path / "data"; data_dir.mkdir()

    text = "The quick brown fox jumps over the lazy dog every single day of the week"
    chunks = [
        {"id": "old", "text": text, "embedding": _rand_unit(DIM, 10),
         "metadata": {"created_at": "2022-01-01T00:00:00Z"}},
        {"id": "new", "text": text + " of the year",
         "embedding": _rand_unit(DIM, 11),
         "metadata": {"created_at": "2024-01-01T00:00:00Z"}},
    ]
    (corpus_dir / "docs.json").write_text(json.dumps(chunks))

    conn = open_db(data_dir)
    _ingest(conn, corpus_dir, data_dir)
    DeduplicateStage(conn=conn, data_dir=data_dir, threshold=0.999).run()
    TextDeduplicateStage(conn=conn, threshold=0.6).run()

    canonical = conn.execute("SELECT canonical_chunk_id FROM dedup_groups").fetchone()[0]
    assert canonical == "new"
    conn.close()


def test_text_dedup_idempotent(tmp_path):
    """Running TextDeduplicateStage twice produces the same groups."""
    corpus_dir = tmp_path / "corpus"; corpus_dir.mkdir()
    data_dir = tmp_path / "data"; data_dir.mkdir()

    text = "The quick brown fox jumps over the lazy dog every single day of the week"
    chunks = [
        {"id": "a", "text": text, "embedding": _rand_unit(DIM, 20),
         "metadata": {"created_at": "2024-01-01T00:00:00Z"}},
        {"id": "b", "text": text + " and night and morning",
         "embedding": _rand_unit(DIM, 21),
         "metadata": {"created_at": "2024-02-01T00:00:00Z"}},
    ]
    (corpus_dir / "docs.json").write_text(json.dumps(chunks))

    conn = open_db(data_dir)
    _ingest(conn, corpus_dir, data_dir)
    DeduplicateStage(conn=conn, data_dir=data_dir, threshold=0.999).run()

    stage = TextDeduplicateStage(conn=conn, threshold=0.6)
    stage.run()
    groups_1 = conn.execute(
        "SELECT group_id, canonical_chunk_id, member_chunk_ids, detection_channels "
        "FROM dedup_groups ORDER BY group_id"
    ).fetchall()

    stage.run()
    groups_2 = conn.execute(
        "SELECT group_id, canonical_chunk_id, member_chunk_ids, detection_channels "
        "FROM dedup_groups ORDER BY group_id"
    ).fetchall()

    assert groups_1 == groups_2
    conn.close()


def test_text_dedup_merges_two_different_groups(tmp_path):
    """When text similarity bridges two embedding groups, they are merged."""
    corpus_dir = tmp_path / "corpus"; corpus_dir.mkdir()
    data_dir = tmp_path / "data"; data_dir.mkdir()

    long_text = "The quick brown fox jumps over the lazy dog every single wonderful day"
    chunks = [
        # Group A: a1 + a2 (embedding near-dups)
        {"id": "a1", "text": long_text, "embedding": _rand_unit(DIM, 30),
         "metadata": {"created_at": "2024-01-01T00:00:00Z"}},
        {"id": "a2", "text": long_text + " indeed", "embedding": _rand_unit(DIM, 31),
         "metadata": {"created_at": "2024-01-02T00:00:00Z"}},
        # Group B: b1 + b2 (embedding near-dups)
        {"id": "b1", "text": long_text + " and night", "embedding": _rand_unit(DIM, 40),
         "metadata": {"created_at": "2024-02-01T00:00:00Z"}},
        {"id": "b2", "text": long_text + " and night too", "embedding": _rand_unit(DIM, 41),
         "metadata": {"created_at": "2024-02-02T00:00:00Z"}},
    ]
    (corpus_dir / "docs.json").write_text(json.dumps(chunks))

    conn = open_db(data_dir)
    _ingest(conn, corpus_dir, data_dir)
    # Manually create two embedding groups (bypass HNSW since embeddings are random/orthogonal)
    conn.execute("DELETE FROM dedup_groups")
    conn.execute(
        "INSERT INTO dedup_groups (group_id, canonical_chunk_id, member_chunk_ids, detection_channels) "
        "VALUES (0, 'a2', ?, ?)",
        [json.dumps(["a1", "a2"]), json.dumps(["embedding"])],
    )
    conn.execute(
        "INSERT INTO dedup_groups (group_id, canonical_chunk_id, member_chunk_ids, detection_channels) "
        "VALUES (1, 'b2', ?, ?)",
        [json.dumps(["b1", "b2"]), json.dumps(["embedding"])],
    )

    # Text dedup at low threshold will see all four as similar → merge the two groups
    TextDeduplicateStage(conn=conn, threshold=0.5).run()

    # All four chunks should now be in one group
    groups = conn.execute("SELECT member_chunk_ids FROM dedup_groups").fetchall()
    assert len(groups) == 1
    members = set(json.loads(groups[0][0]))
    assert members == {"a1", "a2", "b1", "b2"}
    channels = set(json.loads(
        conn.execute("SELECT detection_channels FROM dedup_groups").fetchone()[0]
    ))
    assert "text" in channels
    assert "embedding" in channels
    conn.close()


def test_text_dedup_skips_single_chunk(tmp_path):
    """With fewer than 2 active chunks, stage returns zeros without error."""
    corpus_dir = tmp_path / "corpus"; corpus_dir.mkdir()
    data_dir = tmp_path / "data"; data_dir.mkdir()

    chunks = [
        {"id": "only", "text": "lonely chunk", "embedding": _rand_unit(DIM, 0),
         "metadata": {"created_at": "2024-01-01T00:00:00Z"}},
    ]
    (corpus_dir / "docs.json").write_text(json.dumps(chunks))

    conn = open_db(data_dir)
    _ingest(conn, corpus_dir, data_dir)
    result = TextDeduplicateStage(conn=conn).run()
    assert result["groups_added"] == 0
    conn.close()
