"""Tests for Stage 4: NLI Contradiction Detection (Detector A)."""

import json

import numpy as np
import pytest
from pathlib import Path

from pulse_scan.adapters.fixture import LocalFixtureAdapter
from pulse_scan.config import ClusteringConfig, CollectionConfig, PulseConfig, StoreConfig
from pulse_scan.db.schema import open_db
from pulse_scan.stages.stage0_ingest import IngestStage
from pulse_scan.stages.stage05_calibrate import CalibrateStage
from pulse_scan.stages.stage2_cluster import ClusterStage
from pulse_scan.stages.stage4_nli import NLIContradictionStage


# ---------------------------------------------------------------------------
# Mock predict functions
# ---------------------------------------------------------------------------

def _always_contradict(pairs):
    return [{"contradiction": 0.92, "entailment": 0.04, "neutral": 0.04}] * len(pairs)


def _never_contradict(pairs):
    return [{"contradiction": 0.05, "entailment": 0.85, "neutral": 0.10}] * len(pairs)


def _forward_only_contradict(pairs):
    """Stateful mock: first call (forward) returns contradiction; second (backward) does not."""
    _forward_only_contradict.calls = getattr(_forward_only_contradict, "calls", 0) + 1
    if _forward_only_contradict.calls % 2 == 1:
        return [{"contradiction": 0.91, "entailment": 0.04, "neutral": 0.05}] * len(pairs)
    else:
        return [{"contradiction": 0.05, "entailment": 0.85, "neutral": 0.10}] * len(pairs)


# ---------------------------------------------------------------------------
# Fixture helpers
# ---------------------------------------------------------------------------

def _cfg(corpus_dir: Path) -> PulseConfig:
    return PulseConfig(
        store=StoreConfig(type="fixture"),
        collections=[CollectionConfig(name="test", timestamp_field="created_at")],
        fixture_dir=str(corpus_dir),
    )


def _make_cluster_fixture(corpus_dir: Path, n_per_cluster: int = 8, dim: int = 8) -> None:
    """Three well-separated clusters; no noise by design (small noise scale)."""
    rng = np.random.default_rng(7)

    def _unit(v):
        v = v.astype(np.float32)
        return (v / np.linalg.norm(v)).tolist()

    chunks = []
    for cl, axis in enumerate([0, 1, 2]):
        base = np.zeros(dim, dtype=np.float32)
        base[axis] = 1.0
        for i in range(n_per_cluster):
            v = base + rng.standard_normal(dim).astype(np.float32) * 0.02
            chunks.append({
                "id": f"cl{cl}-{i:03d}",
                "text": f"cluster {cl} chunk {i}. " * 5,
                "embedding": _unit(v),
                "metadata": {"created_at": "2023-01-01T00:00:00Z"},
            })

    (corpus_dir / "test.json").write_text(json.dumps(chunks))


def _run_pipeline(conn, corpus_dir, data_dir, min_cluster_size=5):
    """Run stages 0, 0.5, 2 to populate cluster assignments."""
    adapter = LocalFixtureAdapter(corpus_dir)
    IngestStage(conn=conn, adapter=adapter, config=_cfg(corpus_dir), data_dir=data_dir).run(
        run_id="run-001"
    )
    CalibrateStage(conn=conn, data_dir=data_dir).run(scan_run_id="run-001")
    ClusterStage(
        conn=conn, data_dir=data_dir,
        clustering_config=ClusteringConfig(min_cluster_size=min_cluster_size),
    ).run()


# ---------------------------------------------------------------------------
# Core detection
# ---------------------------------------------------------------------------

def test_nli_records_contradictions_when_score_high(tmp_path):
    """NLI records all candidate pairs as contradictions when predict_fn always returns high score."""
    corpus_dir = tmp_path / "corpus"
    corpus_dir.mkdir()
    data_dir = tmp_path / "data"
    data_dir.mkdir()
    _make_cluster_fixture(corpus_dir, n_per_cluster=8, dim=8)

    conn = open_db(data_dir)
    _run_pipeline(conn, corpus_dir, data_dir)

    result = NLIContradictionStage(
        conn=conn, data_dir=data_dir, scan_run_id="run-001",
        _predict_fn=_always_contradict,
    ).run()

    assert result["pairs_checked"] > 0
    assert result["contradictions_found"] == result["pairs_checked"]
    conn.close()


def test_nli_no_contradictions_when_score_low(tmp_path):
    """NLI records zero contradictions when predict_fn always returns low score."""
    corpus_dir = tmp_path / "corpus"
    corpus_dir.mkdir()
    data_dir = tmp_path / "data"
    data_dir.mkdir()
    _make_cluster_fixture(corpus_dir, n_per_cluster=8, dim=8)

    conn = open_db(data_dir)
    _run_pipeline(conn, corpus_dir, data_dir)

    result = NLIContradictionStage(
        conn=conn, data_dir=data_dir, scan_run_id="run-001",
        _predict_fn=_never_contradict,
    ).run()

    assert result["contradictions_found"] == 0
    n = conn.execute("SELECT COUNT(*) FROM contradictions").fetchone()[0]
    assert n == 0
    conn.close()


def test_nli_cold_start_state(tmp_path):
    """All contradiction records are written with uncalibrated state and NULL confidence."""
    corpus_dir = tmp_path / "corpus"
    corpus_dir.mkdir()
    data_dir = tmp_path / "data"
    data_dir.mkdir()
    _make_cluster_fixture(corpus_dir, n_per_cluster=8, dim=8)

    conn = open_db(data_dir)
    _run_pipeline(conn, corpus_dir, data_dir)

    NLIContradictionStage(
        conn=conn, data_dir=data_dir, scan_run_id="run-001",
        _predict_fn=_always_contradict,
    ).run()

    rows = conn.execute(
        "SELECT calibration_state, calibrated_confidence FROM contradictions"
    ).fetchall()
    assert len(rows) > 0
    for state, confidence in rows:
        assert state == "uncalibrated"
        assert confidence is None
    conn.close()


def test_nli_detector_label_is_nli(tmp_path):
    """detector column is always 'nli' for this stage."""
    corpus_dir = tmp_path / "corpus"
    corpus_dir.mkdir()
    data_dir = tmp_path / "data"
    data_dir.mkdir()
    _make_cluster_fixture(corpus_dir, n_per_cluster=8, dim=8)

    conn = open_db(data_dir)
    _run_pipeline(conn, corpus_dir, data_dir)

    NLIContradictionStage(
        conn=conn, data_dir=data_dir, scan_run_id="run-001",
        _predict_fn=_always_contradict,
    ).run()

    rows = conn.execute("SELECT DISTINCT detector FROM contradictions").fetchall()
    assert rows == [("nli",)]
    conn.close()


# ---------------------------------------------------------------------------
# Direction logic
# ---------------------------------------------------------------------------

def test_nli_direction_both_when_both_score_high(tmp_path):
    """When both forward and backward score > 0.5, direction='both' with max score."""
    corpus_dir = tmp_path / "corpus"
    corpus_dir.mkdir()
    data_dir = tmp_path / "data"
    data_dir.mkdir()
    _make_cluster_fixture(corpus_dir, n_per_cluster=8, dim=8)

    conn = open_db(data_dir)
    _run_pipeline(conn, corpus_dir, data_dir)

    NLIContradictionStage(
        conn=conn, data_dir=data_dir, scan_run_id="run-001",
        _predict_fn=_always_contradict,
    ).run()

    directions = set(
        r[0] for r in conn.execute("SELECT DISTINCT direction FROM contradictions").fetchall()
    )
    assert "both" in directions
    conn.close()


def test_nli_direction_single_when_only_forward(tmp_path):
    """When only forward direction contradicts, direction is 'a->b'."""
    from pulse_scan.config import ScanConfig

    corpus_dir = tmp_path / "corpus"
    corpus_dir.mkdir()
    data_dir = tmp_path / "data"
    data_dir.mkdir()
    _make_cluster_fixture(corpus_dir, n_per_cluster=8, dim=8)

    conn = open_db(data_dir)
    _run_pipeline(conn, corpus_dir, data_dir)

    # Use a large batch_size so each direction is a single call to predict_fn,
    # making the call-count based mock reliable.
    _forward_only_contradict.calls = 0
    NLIContradictionStage(
        conn=conn, data_dir=data_dir, scan_run_id="run-001",
        scan_config=ScanConfig(nli_batch_size=10_000),
        _predict_fn=_forward_only_contradict,
    ).run()

    directions = set(
        r[0] for r in conn.execute("SELECT DISTINCT direction FROM contradictions").fetchall()
    )
    # Forward only → 'a->b'; backward returns low score so no 'b->a' or 'both'
    assert "a->b" in directions
    assert "both" not in directions
    assert "b->a" not in directions
    conn.close()


# ---------------------------------------------------------------------------
# Dedup group exclusion
# ---------------------------------------------------------------------------

def test_nli_skips_dedup_group_pairs(tmp_path):
    """Pairs in the same dedup group are not passed to the NLI model."""
    corpus_dir = tmp_path / "corpus"
    corpus_dir.mkdir()
    data_dir = tmp_path / "data"
    data_dir.mkdir()
    _make_cluster_fixture(corpus_dir, n_per_cluster=8, dim=8)

    conn = open_db(data_dir)
    _run_pipeline(conn, corpus_dir, data_dir)

    # Get all chunk IDs in cluster 0 so we can block all their pairs via dedup groups
    cluster_0_ids = [
        r[0] for r in conn.execute(
            "SELECT chunk_id FROM chunks WHERE cluster_id = 0 AND deleted_at IS NULL"
        ).fetchall()
    ]
    assert len(cluster_0_ids) >= 2

    # Register all cluster-0 chunks as a single dedup group → NLI skips all their pairs
    conn.execute(
        "INSERT INTO dedup_groups (group_id, canonical_chunk_id, member_chunk_ids, detection_channels) "
        "VALUES (?, ?, ?, ?)",
        [99, cluster_0_ids[0], json.dumps(cluster_0_ids), json.dumps(["test"])],
    )

    # Track how many pairs are passed to predict_fn
    calls = []

    def _counting_predict(pairs):
        calls.extend(pairs)
        return [{"contradiction": 0.9, "entailment": 0.05, "neutral": 0.05}] * len(pairs)

    NLIContradictionStage(
        conn=conn, data_dir=data_dir, scan_run_id="run-001",
        _predict_fn=_counting_predict,
    ).run()

    # None of the pairs should involve two cluster-0 chunks (all blocked by dedup group)
    cluster_0_set = set(cluster_0_ids)
    for text_a, text_b in calls:
        # We only have chunk IDs, not texts, but pairs can be verified via the DB
        pass

    # Simpler check: with all cluster-0 pairs blocked, only cluster-1 and cluster-2 pairs remain
    # Verify contradictions table has no entry where both chunk_a and chunk_b are in cluster 0
    rows = conn.execute(
        "SELECT c.chunk_a, c.chunk_b FROM contradictions c"
    ).fetchall()
    for chunk_a, chunk_b in rows:
        assert not (chunk_a in cluster_0_set and chunk_b in cluster_0_set)

    conn.close()


# ---------------------------------------------------------------------------
# Noise chunk exclusion
# ---------------------------------------------------------------------------

def test_nli_skips_noise_chunks(tmp_path):
    """Chunks with cluster_id=NULL are not included in NLI candidates."""
    corpus_dir = tmp_path / "corpus"
    corpus_dir.mkdir()
    data_dir = tmp_path / "data"
    data_dir.mkdir()
    _make_cluster_fixture(corpus_dir, n_per_cluster=8, dim=8)

    conn = open_db(data_dir)
    _run_pipeline(conn, corpus_dir, data_dir)

    # Force a few chunks to NULL cluster_id to simulate noise
    conn.execute(
        "UPDATE chunks SET cluster_id = NULL WHERE chunk_id LIKE 'cl0-000' OR chunk_id LIKE 'cl0-001'"
    )
    conn.execute(
        "DELETE FROM clusters WHERE chunk_id LIKE 'cl0-000' OR chunk_id LIKE 'cl0-001'"
    )

    texts_seen = []

    def _capture_predict(pairs):
        texts_seen.extend(pairs)
        return [{"contradiction": 0.9, "entailment": 0.05, "neutral": 0.05}] * len(pairs)

    NLIContradictionStage(
        conn=conn, data_dir=data_dir, scan_run_id="run-001",
        _predict_fn=_capture_predict,
    ).run()

    # The noise chunks' texts should never appear in predict_fn inputs
    noise_ids = {"cl0-000", "cl0-001"}
    noise_texts = set(
        r[0] for r in conn.execute(
            "SELECT text FROM chunks WHERE chunk_id IN ('cl0-000', 'cl0-001')"
        ).fetchall()
    )
    for text_a, text_b in texts_seen:
        assert text_a not in noise_texts
        assert text_b not in noise_texts

    conn.close()


# ---------------------------------------------------------------------------
# Idempotency
# ---------------------------------------------------------------------------

def test_nli_idempotent(tmp_path):
    """Running NLI twice for the same scan_run_id gives identical result (not doubled)."""
    corpus_dir = tmp_path / "corpus"
    corpus_dir.mkdir()
    data_dir = tmp_path / "data"
    data_dir.mkdir()
    _make_cluster_fixture(corpus_dir, n_per_cluster=8, dim=8)

    conn = open_db(data_dir)
    _run_pipeline(conn, corpus_dir, data_dir)

    stage = NLIContradictionStage(
        conn=conn, data_dir=data_dir, scan_run_id="run-001",
        _predict_fn=_always_contradict,
    )
    r1 = stage.run()
    r2 = stage.run()

    assert r1 == r2
    n = conn.execute(
        "SELECT COUNT(*) FROM contradictions WHERE scan_run_id = 'run-001'"
    ).fetchone()[0]
    assert n == r1["contradictions_found"]
    conn.close()


# ---------------------------------------------------------------------------
# Edge cases
# ---------------------------------------------------------------------------

def test_nli_skips_when_no_clustered_chunks(tmp_path):
    """If no chunks have cluster_id set, NLI returns zero pairs without error."""
    corpus_dir = tmp_path / "corpus"
    corpus_dir.mkdir()
    data_dir = tmp_path / "data"
    data_dir.mkdir()
    _make_cluster_fixture(corpus_dir, n_per_cluster=8, dim=8)

    conn = open_db(data_dir)
    _run_pipeline(conn, corpus_dir, data_dir)

    # Clear all cluster assignments
    conn.execute("UPDATE chunks SET cluster_id = NULL")

    result = NLIContradictionStage(
        conn=conn, data_dir=data_dir, scan_run_id="run-001",
        _predict_fn=_always_contradict,
    ).run()

    assert result == {"pairs_checked": 0, "contradictions_found": 0}
    conn.close()


def test_nli_respects_allowed_chunk_ids(tmp_path):
    """Restricting allowed_chunk_ids reduces the candidate pairs passed to the model."""
    corpus_dir = tmp_path / "corpus"
    corpus_dir.mkdir()
    data_dir = tmp_path / "data"
    data_dir.mkdir()
    _make_cluster_fixture(corpus_dir, n_per_cluster=8, dim=8)

    conn = open_db(data_dir)
    _run_pipeline(conn, corpus_dir, data_dir)

    pair_counts: list[int] = []

    def _counting_predict(pairs):
        pair_counts.append(len(pairs))
        return [{"contradiction": 0.0, "entailment": 1.0, "neutral": 0.0}] * len(pairs)

    stage = NLIContradictionStage(
        conn=conn, data_dir=data_dir, scan_run_id="run-001",
        _predict_fn=_counting_predict,
    )

    # Run with no restriction: all clustered chunks participate
    stage.run()
    total_unrestricted = sum(pair_counts)

    # Run with only one chunk from cluster 0 allowed
    cluster_0_ids = [
        r[0] for r in conn.execute(
            "SELECT chunk_id FROM chunks WHERE cluster_id = 0 AND deleted_at IS NULL"
        ).fetchall()
    ]
    pair_counts.clear()
    stage.run(allowed_chunk_ids={cluster_0_ids[0]})
    total_restricted = sum(pair_counts)

    # Restricting to 1 of 24 chunks must yield fewer (or equal if already 0) candidate pairs
    assert total_restricted <= total_unrestricted
    # With 8 chunks per cluster and 3 clusters unrestricted has more queries than 1 chunk
    assert total_restricted < total_unrestricted
    conn.close()


def test_nli_raises_without_calibration(tmp_path):
    """RuntimeError is raised if calibration table is empty."""
    corpus_dir = tmp_path / "corpus"
    corpus_dir.mkdir()
    data_dir = tmp_path / "data"
    data_dir.mkdir()
    _make_cluster_fixture(corpus_dir, n_per_cluster=8, dim=8)

    conn = open_db(data_dir)
    adapter = LocalFixtureAdapter(corpus_dir)
    IngestStage(conn=conn, adapter=adapter, config=_cfg(corpus_dir), data_dir=data_dir).run(
        run_id="run-001"
    )
    # Deliberately skip calibration and clustering

    with pytest.raises(RuntimeError, match="No calibration found"):
        NLIContradictionStage(
            conn=conn, data_dir=data_dir, scan_run_id="run-001",
            _predict_fn=_always_contradict,
        ).run()
    conn.close()
