"""Tests for dashboard query functions (DB layer, no Streamlit needed)."""

import json
from datetime import datetime, timezone
from pathlib import Path

import pytest

from pulse_scan.db.schema import open_db
from pulse_scan.dashboard.queries import (
    get_available_runs,
    get_corpus_overview,
    get_contradictions,
    get_dedup_groups,
    get_resolution_summary,
    get_staleness_df,
    get_staleness_label_counts,
    get_triage_summary,
    open_connection,
    resolve_contradiction,
)


# ---------------------------------------------------------------------------
# Fixture helpers
# ---------------------------------------------------------------------------

def _insert_chunk(conn, chunk_id, collection="docs", staleness_score=None,
                  staleness_label=None, staleness_components=None, deleted=False):
    conn.execute(
        "INSERT INTO chunks (store_id, collection, chunk_id, text, content_hash, "
        "embedding_offset, staleness_score, staleness_label, staleness_components, "
        "first_seen_by_pulse, last_seen_by_pulse, deleted_at, version) "
        "VALUES ('t', ?, ?, 'hello', 'h', 0, ?, ?, ?, "
        "CURRENT_TIMESTAMP, CURRENT_TIMESTAMP, ?, 1)",
        [collection, chunk_id,
         staleness_score, staleness_label,
         json.dumps(staleness_components) if staleness_components else None,
         datetime.now(timezone.utc).replace(tzinfo=None) if deleted else None],
    )


def _insert_run(conn, run_id):
    conn.execute(
        "INSERT INTO scan_runs (run_id, started_at, finished_at, config, stats) "
        "VALUES (?, CURRENT_TIMESTAMP, CURRENT_TIMESTAMP, '{}', '{}')",
        [run_id],
    )


def _insert_contradiction(conn, chunk_a, chunk_b, run_id="r1",
                           detector="nli", score=0.9, resolution=None):
    conn.execute(
        "INSERT INTO contradictions "
        "(chunk_a, chunk_b, detector, raw_score, calibrated_confidence, "
        " calibration_state, direction, scan_run_id, user_resolution, resolved_at) "
        "VALUES (?, ?, ?, ?, NULL, 'uncalibrated', 'both', ?, ?, NULL)",
        [chunk_a, chunk_b, detector, score, run_id, resolution],
    )


# ---------------------------------------------------------------------------
# open_connection
# ---------------------------------------------------------------------------

def test_open_connection_returns_none_when_missing(tmp_path):
    assert open_connection(tmp_path) is None


def test_open_connection_returns_conn_when_db_exists(tmp_path):
    open_db(tmp_path)  # creates pulse.db
    conn = open_connection(tmp_path)
    assert conn is not None
    conn.close()


# ---------------------------------------------------------------------------
# get_available_runs
# ---------------------------------------------------------------------------

def test_get_available_runs_empty(tmp_path):
    conn = open_db(tmp_path)
    assert get_available_runs(conn) == []
    conn.close()


def test_get_available_runs_returns_newest_first(tmp_path):
    conn = open_db(tmp_path)
    conn.execute(
        "INSERT INTO scan_runs (run_id, started_at) VALUES ('old', '2024-01-01'), ('new', '2024-06-01')"
    )
    runs = get_available_runs(conn)
    assert runs[0] == "new"
    assert runs[1] == "old"
    conn.close()


# ---------------------------------------------------------------------------
# get_corpus_overview
# ---------------------------------------------------------------------------

def test_get_corpus_overview_counts(tmp_path):
    conn = open_db(tmp_path)
    _insert_chunk(conn, "a")
    _insert_chunk(conn, "b")
    _insert_chunk(conn, "c", deleted=True)

    overview = get_corpus_overview(conn)
    assert overview["total_chunks"] == 3
    assert overview["active_chunks"] == 2
    assert overview["deleted_chunks"] == 1
    conn.close()


def test_get_corpus_overview_dedup_and_contradiction_counts(tmp_path):
    conn = open_db(tmp_path)
    _insert_chunk(conn, "a")
    _insert_chunk(conn, "b")
    conn.execute(
        "INSERT INTO dedup_groups (group_id, canonical_chunk_id, member_chunk_ids, detection_channels) "
        "VALUES (0, 'a', '[\"a\",\"b\"]', '[\"embedding\"]')"
    )
    _insert_contradiction(conn, "a", "b", resolution=None)
    overview = get_corpus_overview(conn)
    assert overview["dedup_groups"] == 1
    assert overview["open_contradictions"] == 1
    conn.close()


# ---------------------------------------------------------------------------
# get_staleness_label_counts
# ---------------------------------------------------------------------------

def test_staleness_label_counts(tmp_path):
    conn = open_db(tmp_path)
    _insert_chunk(conn, "f1", staleness_score=0.1, staleness_label="fresh",
                  staleness_components={"age_decay": 0.1, "cluster_drift": 0,
                                        "contradiction_evidence": 0, "supersession_evidence": 0,
                                        "retrieval_abandonment": 0})
    _insert_chunk(conn, "f2", staleness_score=0.2, staleness_label="fresh",
                  staleness_components={"age_decay": 0.2, "cluster_drift": 0,
                                        "contradiction_evidence": 0, "supersession_evidence": 0,
                                        "retrieval_abandonment": 0})
    _insert_chunk(conn, "s1", staleness_score=0.7, staleness_label="stale",
                  staleness_components={"age_decay": 0.7, "cluster_drift": 0,
                                        "contradiction_evidence": 0, "supersession_evidence": 0,
                                        "retrieval_abandonment": 0})
    counts = get_staleness_label_counts(conn)
    assert counts["fresh"] == 2
    assert counts["stale"] == 1
    assert counts["aging"] == 0
    assert counts["abandoned"] == 0
    conn.close()


# ---------------------------------------------------------------------------
# get_staleness_df
# ---------------------------------------------------------------------------

def test_staleness_df_returns_dataframe(tmp_path):
    conn = open_db(tmp_path)
    comp = {"age_decay": 0.5, "cluster_drift": 0, "contradiction_evidence": 0,
            "supersession_evidence": 0, "retrieval_abandonment": 0}
    _insert_chunk(conn, "c1", staleness_score=0.5, staleness_label="aging",
                  staleness_components=comp)
    df = get_staleness_df(conn)
    assert len(df) == 1
    assert df.iloc[0]["chunk_id"] == "c1"
    conn.close()


def test_staleness_df_label_filter(tmp_path):
    conn = open_db(tmp_path)
    comp = {"age_decay": 0.1, "cluster_drift": 0, "contradiction_evidence": 0,
            "supersession_evidence": 0, "retrieval_abandonment": 0}
    _insert_chunk(conn, "fresh-one", staleness_score=0.1, staleness_label="fresh",
                  staleness_components=comp)
    _insert_chunk(conn, "stale-one", staleness_score=0.7, staleness_label="stale",
                  staleness_components=comp)
    df = get_staleness_df(conn, label_filter="fresh")
    assert len(df) == 1
    assert df.iloc[0]["chunk_id"] == "fresh-one"
    conn.close()


# ---------------------------------------------------------------------------
# get_dedup_groups
# ---------------------------------------------------------------------------

def test_get_dedup_groups_empty(tmp_path):
    conn = open_db(tmp_path)
    assert get_dedup_groups(conn) == []
    conn.close()


def test_get_dedup_groups_returns_members(tmp_path):
    conn = open_db(tmp_path)
    _insert_chunk(conn, "a")
    _insert_chunk(conn, "b")
    conn.execute(
        "INSERT INTO dedup_groups (group_id, canonical_chunk_id, member_chunk_ids, detection_channels) "
        "VALUES (0, 'a', '[\"a\",\"b\"]', '[\"embedding\"]')"
    )
    groups = get_dedup_groups(conn)
    assert len(groups) == 1
    assert groups[0]["canonical_chunk_id"] == "a"
    assert groups[0]["n_members"] == 2
    member_ids = {m["chunk_id"] for m in groups[0]["members"]}
    assert member_ids == {"a", "b"}
    canonical_member = next(m for m in groups[0]["members"] if m["chunk_id"] == "a")
    assert canonical_member["is_canonical"] is True
    conn.close()


# ---------------------------------------------------------------------------
# get_contradictions
# ---------------------------------------------------------------------------

def test_get_contradictions_unresolved_only(tmp_path):
    conn = open_db(tmp_path)
    _insert_chunk(conn, "a")
    _insert_chunk(conn, "b")
    _insert_chunk(conn, "c")
    _insert_contradiction(conn, "a", "b", resolution=None)
    _insert_contradiction(conn, "a", "c", resolution="confirmed")

    result = get_contradictions(conn, unresolved_only=True)
    assert len(result) == 1
    assert result[0]["chunk_a"] == "a"
    assert result[0]["chunk_b"] == "b"
    conn.close()


def test_get_contradictions_all_including_resolved(tmp_path):
    conn = open_db(tmp_path)
    _insert_chunk(conn, "a")
    _insert_chunk(conn, "b")
    _insert_contradiction(conn, "a", "b", resolution="confirmed")

    result = get_contradictions(conn, unresolved_only=False)
    assert len(result) == 1
    conn.close()


def test_get_contradictions_detector_filter(tmp_path):
    conn = open_db(tmp_path)
    _insert_chunk(conn, "a")
    _insert_chunk(conn, "b")
    _insert_contradiction(conn, "a", "b", detector="nli")
    # Insert a second pair with different detector (need different chunk pair)
    _insert_chunk(conn, "c")
    _insert_contradiction(conn, "a", "c", detector="numeric")

    nli_only = get_contradictions(conn, unresolved_only=False, detector="nli")
    assert all(r["detector"] == "nli" for r in nli_only)
    assert len(nli_only) == 1
    conn.close()


def test_get_contradictions_run_id_filter(tmp_path):
    conn = open_db(tmp_path)
    _insert_chunk(conn, "a")
    _insert_chunk(conn, "b")
    _insert_chunk(conn, "c")
    _insert_contradiction(conn, "a", "b", run_id="run-A")
    _insert_contradiction(conn, "a", "c", run_id="run-B")

    result = get_contradictions(conn, run_id="run-A", unresolved_only=False)
    assert len(result) == 1
    assert result[0]["scan_run_id"] == "run-A"
    conn.close()


# ---------------------------------------------------------------------------
# resolve_contradiction
# ---------------------------------------------------------------------------

def test_resolve_contradiction_sets_resolution(tmp_path):
    conn = open_db(tmp_path)
    _insert_chunk(conn, "a")
    _insert_chunk(conn, "b")
    _insert_contradiction(conn, "a", "b")
    conn.close()

    resolve_contradiction(tmp_path, "a", "b", "confirmed")

    conn = open_db(tmp_path)
    row = conn.execute(
        "SELECT user_resolution FROM contradictions WHERE chunk_a = 'a' AND chunk_b = 'b'"
    ).fetchone()
    assert row[0] == "confirmed"
    conn.close()


def test_resolve_contradiction_works_both_orders(tmp_path):
    """Resolving (b, a) should also update the row stored as (a, b)."""
    conn = open_db(tmp_path)
    _insert_chunk(conn, "a")
    _insert_chunk(conn, "b")
    _insert_contradiction(conn, "a", "b")
    conn.close()

    resolve_contradiction(tmp_path, "b", "a", "false_positive")

    conn = open_db(tmp_path)
    row = conn.execute(
        "SELECT user_resolution FROM contradictions"
    ).fetchone()
    assert row[0] == "false_positive"
    conn.close()


# ---------------------------------------------------------------------------
# get_resolution_summary
# ---------------------------------------------------------------------------

def test_resolution_summary(tmp_path):
    conn = open_db(tmp_path)
    _insert_chunk(conn, "a")
    _insert_chunk(conn, "b")
    _insert_chunk(conn, "c")
    _insert_contradiction(conn, "a", "b", resolution="confirmed")
    _insert_contradiction(conn, "a", "c", resolution=None)

    summary = get_resolution_summary(conn)
    assert summary["confirmed"] == 1
    assert summary["unresolved"] == 1
    assert summary["false_positive"] == 0
    conn.close()


# ---------------------------------------------------------------------------
# get_triage_summary
# ---------------------------------------------------------------------------

def test_triage_summary(tmp_path):
    conn = open_db(tmp_path)
    conn.execute(
        "INSERT INTO triage_log (scan_run_id, chunk_id, priority, components, was_scanned) "
        "VALUES ('r1', 'a', 0.8, '{}', true), ('r1', 'b', 0.3, '{}', false)"
    )
    summary = get_triage_summary(conn, "r1")
    assert summary["chunks_scored"] == 2
    assert summary["chunks_scanned"] == 1
    conn.close()
