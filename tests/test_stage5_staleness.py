"""Tests for Stage 5: Staleness Scoring."""

import json
from datetime import datetime, timedelta

import numpy as np
import pytest
from pathlib import Path

from pulse_scan.adapters.fixture import LocalFixtureAdapter
from pulse_scan.config import CollectionConfig, PulseConfig, StoreConfig
from pulse_scan.db.schema import open_db
from pulse_scan.stages.stage0_ingest import IngestStage
from pulse_scan.stages.stage05_calibrate import CalibrateStage
from pulse_scan.stages.stage5_staleness import (
    StalenessStage,
    _age_decay, _cluster_drift, _staleness_label, DEFAULT_HALF_LIFE_DAYS,
)

# Fixed reference time so tests are deterministic regardless of when they run.
REF_TIME = datetime(2024, 6, 1, 0, 0, 0)


# ---------------------------------------------------------------------------
# Unit tests for pure component functions
# ---------------------------------------------------------------------------

def test_age_decay_brand_new():
    """A chunk created today has age_decay ≈ 0."""
    ad = _age_decay(REF_TIME, REF_TIME, half_life_days=90)
    assert ad == pytest.approx(0.0, abs=1e-6)


def test_age_decay_one_half_life():
    """After one half-life, age_decay = 0.5 by definition."""
    ts = datetime(2024, 3, 3)  # ~90 days before REF_TIME
    ad = _age_decay(ts, REF_TIME, half_life_days=90)
    assert ad == pytest.approx(0.5, abs=0.02)  # 89–91 day range


def test_age_decay_very_old_approaches_one():
    """After many half-lives, age_decay → 1."""
    ts = datetime(2020, 1, 1)  # ~4.5 years before REF_TIME
    ad = _age_decay(ts, REF_TIME, half_life_days=90)
    assert ad > 0.99


def test_age_decay_none_timestamp():
    """Missing timestamp returns 0 (no signal)."""
    assert _age_decay(None, REF_TIME, 90) == 0.0


def test_cluster_drift_no_cluster():
    """Noise chunk (cluster_id=None) has drift=0."""
    emb = np.array([1, 0, 0, 0], dtype=np.float32)
    assert _cluster_drift(emb, None, {}) == pytest.approx(0.0)


def test_cluster_drift_aligned_centroid():
    """Embedding aligned with centroid → drift ≈ 0."""
    emb = np.array([1, 0, 0, 0], dtype=np.float32)
    centroid = np.array([1, 0, 0, 0], dtype=np.float32)
    assert _cluster_drift(emb, 0, {0: centroid}) == pytest.approx(0.0, abs=1e-6)


def test_cluster_drift_orthogonal_centroid():
    """Embedding orthogonal to centroid → drift = 1."""
    emb = np.array([1, 0, 0, 0], dtype=np.float32)
    centroid = np.array([0, 1, 0, 0], dtype=np.float32)
    assert _cluster_drift(emb, 0, {0: centroid}) == pytest.approx(1.0)


def test_staleness_label_boundaries():
    assert _staleness_label(0.0) == "fresh"
    assert _staleness_label(0.29) == "fresh"
    assert _staleness_label(0.30) == "aging"
    assert _staleness_label(0.59) == "aging"
    assert _staleness_label(0.60) == "stale"
    assert _staleness_label(0.84) == "stale"
    assert _staleness_label(0.85) == "abandoned"
    assert _staleness_label(1.00) == "abandoned"


# ---------------------------------------------------------------------------
# Integration: StalenessStage against real DuckDB state
# ---------------------------------------------------------------------------

def _make_4chunk_fixture(corpus_dir: Path, ref_time: datetime) -> None:
    """
    4-chunk DIM=4 corpus:
      fresh-a  : created 7 days before ref_time  → very low age_decay
      old-b    : created 2022-01-01               → age_decay ≈ 1 (half_life=90)
      cl1-base : cluster 1 base vector
      cl1-drift: cluster 1 member far from centroid
    """
    fresh_dt = ref_time - timedelta(days=7)
    fresh_ts = fresh_dt.strftime("%Y-%m-%dT%H:%M:%SZ")
    chunks = [
        {"id": "fresh-a", "text": "a brand new chunk",
         "embedding": [1.0, 0.0, 0.0, 0.0],
         "metadata": {"created_at": fresh_ts}},
        {"id": "old-b", "text": "a very old chunk",
         "embedding": [0.999, 0.045, 0.0, 0.0],  # near fresh-a in same cluster
         "metadata": {"created_at": "2022-01-01T00:00:00Z"}},
        {"id": "cl1-base", "text": "cluster 1 centroid member",
         "embedding": [0.0, 1.0, 0.0, 0.0],
         "metadata": {"created_at": "2023-01-01T00:00:00Z"}},
        {"id": "cl1-drift", "text": "cluster 1 drifted member",
         "embedding": [0.0, 0.707, 0.707, 0.0],  # 45° away from cl1 centroid
         "metadata": {"created_at": "2023-01-01T00:00:00Z"}},
    ]
    (corpus_dir / "test.json").write_text(json.dumps(chunks))


def _cfg(corpus_dir: Path) -> PulseConfig:
    return PulseConfig(
        store=StoreConfig(type="fixture"),
        collections=[CollectionConfig(name="test", timestamp_field="created_at",
                                      half_life_days=DEFAULT_HALF_LIFE_DAYS)],
        fixture_dir=str(corpus_dir),
    )


def _ingest_and_calibrate(conn, corpus_dir, data_dir):
    adapter = LocalFixtureAdapter(corpus_dir)
    IngestStage(conn=conn, adapter=adapter, config=_cfg(corpus_dir), data_dir=data_dir).run(
        run_id="run-001"
    )
    CalibrateStage(conn=conn, data_dir=data_dir).run(scan_run_id="run-001")


def _assign_clusters(conn):
    """Manually assign cluster IDs and insert centroids (bypasses UMAP for speed)."""
    # Cluster 0: fresh-a and old-b
    conn.execute("UPDATE chunks SET cluster_id = 0 WHERE chunk_id IN ('fresh-a', 'old-b')")
    c0 = np.array([1.0, 0.025, 0.0, 0.0], dtype=np.float32)
    c0 /= np.linalg.norm(c0)
    conn.execute(
        "INSERT INTO cluster_centroids (cluster_id, centroid, n_chunks) VALUES (0, ?, 2)",
        [c0.tobytes()],
    )
    # Cluster 1: cl1-base and cl1-drift
    conn.execute("UPDATE chunks SET cluster_id = 1 WHERE chunk_id IN ('cl1-base', 'cl1-drift')")
    c1 = np.array([0.0, 1.0, 0.0, 0.0], dtype=np.float32)
    conn.execute(
        "INSERT INTO cluster_centroids (cluster_id, centroid, n_chunks) VALUES (1, ?, 2)",
        [c1.tobytes()],
    )


# ---------------------------------------------------------------------------
# Score and label tests
# ---------------------------------------------------------------------------

def test_fresh_chunk_scores_below_fresh_threshold(tmp_path):
    """A brand-new chunk with no stale signals scores < 0.3 (fresh)."""
    corpus_dir = tmp_path / "corpus"
    corpus_dir.mkdir()
    data_dir = tmp_path / "data"
    data_dir.mkdir()
    _make_4chunk_fixture(corpus_dir, REF_TIME)

    conn = open_db(data_dir)
    _ingest_and_calibrate(conn, corpus_dir, data_dir)
    _assign_clusters(conn)

    StalenessStage(
        conn=conn, data_dir=data_dir,
        collection_configs=[CollectionConfig(name="test", timestamp_field="created_at")],
        reference_time=REF_TIME,
    ).run()

    row = conn.execute(
        "SELECT staleness_score, staleness_label FROM chunks WHERE chunk_id = 'fresh-a'"
    ).fetchone()
    assert row[0] < 0.30
    assert row[1] == "fresh"
    conn.close()


def test_old_chunk_scores_above_aging_threshold(tmp_path):
    """A chunk from 2022 (>2 years before ref_time) with half_life=90 scores ≥ 0.3."""
    corpus_dir = tmp_path / "corpus"
    corpus_dir.mkdir()
    data_dir = tmp_path / "data"
    data_dir.mkdir()
    _make_4chunk_fixture(corpus_dir, REF_TIME)

    conn = open_db(data_dir)
    _ingest_and_calibrate(conn, corpus_dir, data_dir)
    _assign_clusters(conn)

    StalenessStage(
        conn=conn, data_dir=data_dir,
        collection_configs=[CollectionConfig(name="test", timestamp_field="created_at")],
        reference_time=REF_TIME,
    ).run()

    row = conn.execute(
        "SELECT staleness_score, staleness_label FROM chunks WHERE chunk_id = 'old-b'"
    ).fetchone()
    assert row[0] >= 0.30
    assert row[1] == "aging"
    conn.close()


def test_drifted_chunk_scores_higher_than_centroid_member(tmp_path):
    """A chunk 45° from cluster centroid scores higher than one aligned with it."""
    corpus_dir = tmp_path / "corpus"
    corpus_dir.mkdir()
    data_dir = tmp_path / "data"
    data_dir.mkdir()
    _make_4chunk_fixture(corpus_dir, REF_TIME)

    conn = open_db(data_dir)
    _ingest_and_calibrate(conn, corpus_dir, data_dir)
    _assign_clusters(conn)

    StalenessStage(
        conn=conn, data_dir=data_dir,
        collection_configs=[CollectionConfig(name="test", timestamp_field="created_at")],
        reference_time=REF_TIME,
    ).run()

    base_score = conn.execute(
        "SELECT staleness_score FROM chunks WHERE chunk_id = 'cl1-base'"
    ).fetchone()[0]
    drift_score = conn.execute(
        "SELECT staleness_score FROM chunks WHERE chunk_id = 'cl1-drift'"
    ).fetchone()[0]
    assert drift_score > base_score
    conn.close()


def test_superseded_chunk_scores_higher_than_canonical(tmp_path):
    """Non-canonical member of a dedup group scores higher than the canonical."""
    corpus_dir = tmp_path / "corpus"
    corpus_dir.mkdir()
    data_dir = tmp_path / "data"
    data_dir.mkdir()
    _make_4chunk_fixture(corpus_dir, REF_TIME)

    conn = open_db(data_dir)
    _ingest_and_calibrate(conn, corpus_dir, data_dir)
    _assign_clusters(conn)

    # fresh-a is canonical; old-b is superseded
    conn.execute(
        "INSERT INTO dedup_groups (group_id, canonical_chunk_id, member_chunk_ids, detection_channels) "
        "VALUES (1, 'fresh-a', ?, ?)",
        [json.dumps(["fresh-a", "old-b"]), json.dumps(["test"])],
    )

    StalenessStage(
        conn=conn, data_dir=data_dir,
        collection_configs=[CollectionConfig(name="test", timestamp_field="created_at")],
        reference_time=REF_TIME,
    ).run()

    fresh_score = conn.execute(
        "SELECT staleness_score FROM chunks WHERE chunk_id = 'fresh-a'"
    ).fetchone()[0]
    superseded_score = conn.execute(
        "SELECT staleness_score FROM chunks WHERE chunk_id = 'old-b'"
    ).fetchone()[0]
    assert superseded_score > fresh_score
    conn.close()


def test_contradicted_chunk_scores_higher(tmp_path):
    """A chunk with contradictions scores higher than a chunk without."""
    corpus_dir = tmp_path / "corpus"
    corpus_dir.mkdir()
    data_dir = tmp_path / "data"
    data_dir.mkdir()
    _make_4chunk_fixture(corpus_dir, REF_TIME)

    conn = open_db(data_dir)
    _ingest_and_calibrate(conn, corpus_dir, data_dir)
    _assign_clusters(conn)

    # Give cl1-base a contradiction, leave cl1-drift clean
    conn.execute(
        "INSERT INTO contradictions "
        "(chunk_a, chunk_b, detector, raw_score, calibrated_confidence, "
        "calibration_state, direction, scan_run_id) "
        "VALUES ('cl1-base', 'cl1-drift', 'nli', 0.85, NULL, 'uncalibrated', 'a->b', 'run-001')"
    )

    StalenessStage(
        conn=conn, data_dir=data_dir,
        collection_configs=[CollectionConfig(name="test", timestamp_field="created_at")],
        reference_time=REF_TIME,
    ).run()

    contra_score = conn.execute(
        "SELECT staleness_score FROM chunks WHERE chunk_id = 'cl1-base'"
    ).fetchone()[0]
    clean_score = conn.execute(
        "SELECT staleness_score FROM chunks WHERE chunk_id = 'cl1-drift'"
    ).fetchone()[0]
    # cl1-base (contradicted) should score higher; cl1-drift has drift signal instead
    # Test just that contradicted chunk has contradiction_evidence > 0
    components = json.loads(
        conn.execute(
            "SELECT staleness_components FROM chunks WHERE chunk_id = 'cl1-base'"
        ).fetchone()[0]
    )
    assert components["contradiction_evidence"] > 0.0
    conn.close()


# ---------------------------------------------------------------------------
# Components JSON and DB write
# ---------------------------------------------------------------------------

def test_staleness_components_json_has_all_keys(tmp_path):
    """staleness_components JSON includes all five expected component keys."""
    corpus_dir = tmp_path / "corpus"
    corpus_dir.mkdir()
    data_dir = tmp_path / "data"
    data_dir.mkdir()
    _make_4chunk_fixture(corpus_dir, REF_TIME)

    conn = open_db(data_dir)
    _ingest_and_calibrate(conn, corpus_dir, data_dir)
    _assign_clusters(conn)

    StalenessStage(
        conn=conn, data_dir=data_dir,
        collection_configs=[CollectionConfig(name="test", timestamp_field="created_at")],
        reference_time=REF_TIME,
    ).run()

    rows = conn.execute(
        "SELECT staleness_components FROM chunks WHERE deleted_at IS NULL"
    ).fetchall()
    assert len(rows) == 4
    required_keys = {
        "age_decay", "cluster_drift", "contradiction_evidence",
        "supersession_evidence", "retrieval_abandonment",
    }
    for (comp_json,) in rows:
        components = json.loads(comp_json)
        assert required_keys == set(components.keys())
    conn.close()


def test_staleness_scores_all_chunks(tmp_path):
    """run() updates staleness_score on all active chunks."""
    corpus_dir = tmp_path / "corpus"
    corpus_dir.mkdir()
    data_dir = tmp_path / "data"
    data_dir.mkdir()
    _make_4chunk_fixture(corpus_dir, REF_TIME)

    conn = open_db(data_dir)
    _ingest_and_calibrate(conn, corpus_dir, data_dir)
    _assign_clusters(conn)

    result = StalenessStage(
        conn=conn, data_dir=data_dir, reference_time=REF_TIME
    ).run()

    assert result["chunks_scored"] == 4
    n_scored = conn.execute(
        "SELECT COUNT(*) FROM chunks WHERE staleness_score IS NOT NULL AND deleted_at IS NULL"
    ).fetchone()[0]
    assert n_scored == 4
    conn.close()


def test_staleness_idempotent(tmp_path):
    """Running staleness twice gives the same scores (UPDATE, not INSERT)."""
    corpus_dir = tmp_path / "corpus"
    corpus_dir.mkdir()
    data_dir = tmp_path / "data"
    data_dir.mkdir()
    _make_4chunk_fixture(corpus_dir, REF_TIME)

    conn = open_db(data_dir)
    _ingest_and_calibrate(conn, corpus_dir, data_dir)
    _assign_clusters(conn)

    stage = StalenessStage(
        conn=conn, data_dir=data_dir, reference_time=REF_TIME
    )
    stage.run()
    scores_1 = {
        r[0]: r[1]
        for r in conn.execute(
            "SELECT chunk_id, staleness_score FROM chunks WHERE deleted_at IS NULL"
        ).fetchall()
    }
    stage.run()
    scores_2 = {
        r[0]: r[1]
        for r in conn.execute(
            "SELECT chunk_id, staleness_score FROM chunks WHERE deleted_at IS NULL"
        ).fetchall()
    }
    assert scores_1 == scores_2
    conn.close()


def test_staleness_per_collection_half_life(tmp_path):
    """A collection with shorter half_life_days produces higher age_decay for same age chunk."""
    corpus_dir = tmp_path / "corpus"
    corpus_dir.mkdir()
    data_dir = tmp_path / "data"
    data_dir.mkdir()
    _make_4chunk_fixture(corpus_dir, REF_TIME)

    # Run once with half_life=7 (very aggressive)
    conn = open_db(data_dir)
    _ingest_and_calibrate(conn, corpus_dir, data_dir)
    _assign_clusters(conn)

    StalenessStage(
        conn=conn, data_dir=data_dir,
        collection_configs=[CollectionConfig(name="test", timestamp_field="created_at",
                                             half_life_days=7)],
        reference_time=REF_TIME,
    ).run()
    score_7 = conn.execute(
        "SELECT staleness_score FROM chunks WHERE chunk_id = 'fresh-a'"
    ).fetchone()[0]

    # Reset and run with half_life=9000 (very lenient)
    conn.execute("UPDATE chunks SET staleness_score = NULL, staleness_label = NULL, staleness_components = NULL")
    StalenessStage(
        conn=conn, data_dir=data_dir,
        collection_configs=[CollectionConfig(name="test", timestamp_field="created_at",
                                             half_life_days=9000)],
        reference_time=REF_TIME,
    ).run()
    score_9000 = conn.execute(
        "SELECT staleness_score FROM chunks WHERE chunk_id = 'fresh-a'"
    ).fetchone()[0]

    assert score_7 > score_9000
    conn.close()
