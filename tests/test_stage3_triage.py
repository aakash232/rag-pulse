"""Tests for Stage 3: Triage with cost budget."""

import json
from datetime import datetime

import pytest

from pulse_scan.config import CollectionConfig, ScanConfig
from pulse_scan.db.schema import open_db
from pulse_scan.stages.stage3_triage import TriageStage, _age_factor

REF_TIME = datetime(2024, 6, 1, 0, 0, 0)
DUMMY_CENTROID = b"\x00" * 16  # placeholder blob for cluster_centroids.centroid


# ---------------------------------------------------------------------------
# Unit tests for age_factor
# ---------------------------------------------------------------------------


def test_age_factor_brand_new():
    assert _age_factor(REF_TIME, REF_TIME, 90) == pytest.approx(0.0, abs=1e-9)


def test_age_factor_old():
    ts = datetime(2020, 1, 1)
    assert _age_factor(ts, REF_TIME, 90) > 0.99


def test_age_factor_none():
    assert _age_factor(None, REF_TIME, 90) == 0.0


# ---------------------------------------------------------------------------
# Fixture helpers — create DB rows directly (no ingest needed for triage tests)
# ---------------------------------------------------------------------------


def _insert_chunk(conn, chunk_id, collection="docs", resolved_ts=None, cluster_id=0):
    conn.execute(
        "INSERT INTO chunks (store_id, collection, chunk_id, text, content_hash, "
        "resolved_timestamp, timestamp_source, embedding_offset, cluster_id, "
        "first_seen_by_pulse, last_seen_by_pulse, version) "
        "VALUES ('test', ?, ?, 'text', 'hash', ?, 'metadata', 0, ?, "
        "        CURRENT_TIMESTAMP, CURRENT_TIMESTAMP, 1)",
        [collection, chunk_id, resolved_ts, cluster_id],
    )


def _insert_centroid(conn, cluster_id, n_chunks):
    conn.execute(
        "INSERT INTO cluster_centroids (cluster_id, centroid, n_chunks) VALUES (?, ?, ?)",
        [cluster_id, DUMMY_CENTROID, n_chunks],
    )


def _cfg(cost_budget=50_000, k=5):
    return ScanConfig(cost_budget=cost_budget, contradiction_candidates_per_chunk=k)


# ---------------------------------------------------------------------------
# Integration tests
# ---------------------------------------------------------------------------


def test_triage_scores_all_clustered_chunks(tmp_path):
    conn = open_db(tmp_path)
    for i in range(4):
        _insert_chunk(conn, f"c{i}", cluster_id=0, resolved_ts=datetime(2023, 1, i + 1))
    _insert_centroid(conn, 0, 4)

    _, stats = TriageStage(conn=conn, scan_run_id="r1", scan_config=_cfg(), reference_time=REF_TIME).run()
    assert stats["chunks_scored"] == 4
    conn.close()


def test_triage_excludes_noise_chunks(tmp_path):
    conn = open_db(tmp_path)
    _insert_chunk(conn, "clustered", cluster_id=0, resolved_ts=datetime(2023, 1, 1))
    # Noise chunk: cluster_id IS NULL
    conn.execute(
        "INSERT INTO chunks (store_id, collection, chunk_id, text, content_hash, "
        "first_seen_by_pulse, last_seen_by_pulse, version) "
        "VALUES ('test', 'docs', 'noise', 'text', 'hash', "
        "        CURRENT_TIMESTAMP, CURRENT_TIMESTAMP, 1)"
    )
    _insert_centroid(conn, 0, 1)

    _, stats = TriageStage(conn=conn, scan_run_id="r1", scan_config=_cfg(), reference_time=REF_TIME).run()
    assert stats["chunks_scored"] == 1  # only the clustered chunk
    conn.close()


def test_triage_older_chunk_higher_priority(tmp_path):
    conn = open_db(tmp_path)
    _insert_chunk(conn, "old", cluster_id=0, resolved_ts=datetime(2020, 1, 1))
    _insert_chunk(conn, "new", cluster_id=0, resolved_ts=datetime(2024, 5, 31))
    _insert_centroid(conn, 0, 2)

    TriageStage(conn=conn, scan_run_id="r1", scan_config=_cfg(), reference_time=REF_TIME).run()

    rows = {r[0]: r[1] for r in conn.execute("SELECT chunk_id, priority FROM triage_log ORDER BY chunk_id").fetchall()}
    assert rows["old"] > rows["new"]
    conn.close()


def test_triage_larger_cluster_higher_priority(tmp_path):
    conn = open_db(tmp_path)
    # Same timestamp, different cluster sizes
    ts = datetime(2023, 6, 1)
    _insert_chunk(conn, "small-cluster", cluster_id=0, resolved_ts=ts)
    _insert_chunk(conn, "large-cluster", cluster_id=1, resolved_ts=ts)
    _insert_centroid(conn, 0, n_chunks=2)  # small
    _insert_centroid(conn, 1, n_chunks=10)  # large

    TriageStage(conn=conn, scan_run_id="r1", scan_config=_cfg(), reference_time=REF_TIME).run()

    rows = {r[0]: r[1] for r in conn.execute("SELECT chunk_id, priority FROM triage_log").fetchall()}
    assert rows["large-cluster"] > rows["small-cluster"]
    conn.close()


def test_triage_budget_limits_allowed_chunks(tmp_path):
    conn = open_db(tmp_path)
    for i in range(10):
        _insert_chunk(conn, f"c{i}", cluster_id=0, resolved_ts=datetime(2023, 1, i + 1))
    _insert_centroid(conn, 0, 10)

    # budget=10, k=5 → max_allowed = 10 // 5 = 2
    allowed, stats = TriageStage(
        conn=conn, scan_run_id="r1", scan_config=_cfg(cost_budget=10, k=5), reference_time=REF_TIME
    ).run()
    assert len(allowed) == 2
    assert stats["chunks_allowed"] == 2
    conn.close()


def test_triage_all_fit_within_large_budget(tmp_path):
    conn = open_db(tmp_path)
    for i in range(5):
        _insert_chunk(conn, f"c{i}", cluster_id=0, resolved_ts=datetime(2023, 1, i + 1))
    _insert_centroid(conn, 0, 5)

    allowed, stats = TriageStage(
        conn=conn,
        scan_run_id="r1",
        scan_config=_cfg(cost_budget=1_000_000, k=5),
        reference_time=REF_TIME,
    ).run()
    assert len(allowed) == 5
    assert stats["chunks_allowed"] == 5
    conn.close()


def test_triage_writes_triage_log(tmp_path):
    conn = open_db(tmp_path)
    for i in range(3):
        _insert_chunk(conn, f"c{i}", cluster_id=0, resolved_ts=datetime(2023, 1, i + 1))
    _insert_centroid(conn, 0, 3)

    TriageStage(conn=conn, scan_run_id="r1", scan_config=_cfg(), reference_time=REF_TIME).run()

    rows = conn.execute("SELECT chunk_id, priority, was_scanned FROM triage_log").fetchall()
    assert len(rows) == 3
    for _, priority, _ in rows:
        assert 0.0 <= priority <= 1.0
    conn.close()


def test_triage_was_scanned_matches_allowed(tmp_path):
    conn = open_db(tmp_path)
    for i in range(6):
        _insert_chunk(conn, f"c{i}", cluster_id=0, resolved_ts=datetime(2023, 1, i + 1))
    _insert_centroid(conn, 0, 6)

    # budget=10, k=5 → only 2 allowed
    allowed, _ = TriageStage(
        conn=conn, scan_run_id="r1", scan_config=_cfg(cost_budget=10, k=5), reference_time=REF_TIME
    ).run()

    scanned = {r[0] for r in conn.execute("SELECT chunk_id FROM triage_log WHERE was_scanned = true").fetchall()}
    assert scanned == allowed
    conn.close()


def test_triage_components_json_present(tmp_path):
    conn = open_db(tmp_path)
    _insert_chunk(conn, "c0", cluster_id=0, resolved_ts=datetime(2023, 1, 1))
    _insert_centroid(conn, 0, 1)

    TriageStage(conn=conn, scan_run_id="r1", scan_config=_cfg(), reference_time=REF_TIME).run()

    components_raw = conn.execute("SELECT components FROM triage_log").fetchone()[0]
    comp = json.loads(components_raw)
    assert "age_factor" in comp
    assert "cluster_factor" in comp
    conn.close()


def test_triage_idempotent(tmp_path):
    conn = open_db(tmp_path)
    for i in range(4):
        _insert_chunk(conn, f"c{i}", cluster_id=0, resolved_ts=datetime(2023, 1, i + 1))
    _insert_centroid(conn, 0, 4)

    stage = TriageStage(conn=conn, scan_run_id="r1", scan_config=_cfg(), reference_time=REF_TIME)
    allowed_1, stats_1 = stage.run()
    allowed_2, stats_2 = stage.run()

    assert allowed_1 == allowed_2
    assert stats_1 == stats_2
    log_count = conn.execute("SELECT COUNT(*) FROM triage_log").fetchone()[0]
    assert log_count == 4  # not doubled
    conn.close()


def test_triage_no_clustered_chunks_returns_empty(tmp_path):
    conn = open_db(tmp_path)
    # Insert a noise chunk (no cluster_id)
    conn.execute(
        "INSERT INTO chunks (store_id, collection, chunk_id, text, content_hash, "
        "first_seen_by_pulse, last_seen_by_pulse, version) "
        "VALUES ('test', 'docs', 'noise', 'text', 'hash', "
        "        CURRENT_TIMESTAMP, CURRENT_TIMESTAMP, 1)"
    )

    allowed, stats = TriageStage(conn=conn, scan_run_id="r1", scan_config=_cfg(), reference_time=REF_TIME).run()
    assert allowed == set()
    assert stats["chunks_scored"] == 0
    conn.close()


def test_triage_collection_half_life_respected(tmp_path):
    """A collection with shorter half_life → higher age priority for same-age chunk."""
    conn = open_db(tmp_path)
    ts = datetime(2023, 6, 1)  # ~1 year before REF_TIME
    _insert_chunk(conn, "short", collection="fast", cluster_id=0, resolved_ts=ts)
    _insert_chunk(conn, "long", collection="slow", cluster_id=0, resolved_ts=ts)
    _insert_centroid(conn, 0, 2)

    TriageStage(
        conn=conn,
        scan_run_id="r1",
        scan_config=_cfg(),
        collection_configs=[
            CollectionConfig(name="fast", half_life_days=7),
            CollectionConfig(name="slow", half_life_days=9000),
        ],
        reference_time=REF_TIME,
    ).run()

    rows = {r[0]: r[1] for r in conn.execute("SELECT chunk_id, priority FROM triage_log").fetchall()}
    # short half-life → age_factor closer to 1 → higher priority
    assert rows["short"] > rows["long"]
    conn.close()
