"""Tests for Stage 6: JSON Report."""

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
from pulse_scan.stages.stage5_staleness import StalenessStage
from pulse_scan.stages.stage6_report import ReportStage

REF_TIME = datetime(2024, 6, 1, 0, 0, 0)
RUN_ID = "test-run-0001"


# ---------------------------------------------------------------------------
# Fixture helpers
# ---------------------------------------------------------------------------

def _make_corpus(corpus_dir: Path) -> None:
    fresh_ts = (REF_TIME - timedelta(days=7)).strftime("%Y-%m-%dT%H:%M:%SZ")
    chunks = [
        {"id": "chunk-a", "text": "alpha content",
         "embedding": [1.0, 0.0, 0.0, 0.0],
         "metadata": {"created_at": fresh_ts}},
        {"id": "chunk-b", "text": "beta content different",
         "embedding": [0.999, 0.045, 0.0, 0.0],
         "metadata": {"created_at": "2022-01-01T00:00:00Z"}},
        {"id": "chunk-c", "text": "gamma standalone",
         "embedding": [0.0, 1.0, 0.0, 0.0],
         "metadata": {"created_at": "2023-06-01T00:00:00Z"}},
    ]
    (corpus_dir / "docs.json").write_text(json.dumps(chunks))


def _cfg(corpus_dir: Path) -> PulseConfig:
    return PulseConfig(
        store=StoreConfig(type="fixture"),
        collections=[CollectionConfig(name="docs", timestamp_field="created_at",
                                      half_life_days=90)],
        fixture_dir=str(corpus_dir),
    )


def _setup(tmp_path: Path):
    """Returns (conn, data_dir) with ingestion + calibration + staleness done."""
    corpus_dir = tmp_path / "corpus"
    corpus_dir.mkdir()
    data_dir = tmp_path / "data"
    data_dir.mkdir()
    _make_corpus(corpus_dir)

    conn = open_db(data_dir)
    cfg = _cfg(corpus_dir)
    adapter = LocalFixtureAdapter(corpus_dir)
    IngestStage(conn=conn, adapter=adapter, config=cfg, data_dir=data_dir).run(run_id=RUN_ID)
    CalibrateStage(conn=conn, data_dir=data_dir).run(scan_run_id=RUN_ID)

    # Manually assign clusters + centroids (bypass UMAP)
    conn.execute("UPDATE chunks SET cluster_id = 0 WHERE chunk_id IN ('chunk-a', 'chunk-b')")
    c0 = np.array([1.0, 0.025, 0.0, 0.0], dtype=np.float32)
    c0 /= np.linalg.norm(c0)
    conn.execute("INSERT INTO cluster_centroids (cluster_id, centroid, n_chunks) VALUES (0, ?, 2)",
                 [c0.tobytes()])
    conn.execute("UPDATE chunks SET cluster_id = 1 WHERE chunk_id = 'chunk-c'")
    c1 = np.array([0.0, 1.0, 0.0, 0.0], dtype=np.float32)
    conn.execute("INSERT INTO cluster_centroids (cluster_id, centroid, n_chunks) VALUES (1, ?, 1)",
                 [c1.tobytes()])

    StalenessStage(conn=conn, data_dir=data_dir,
                   collection_configs=cfg.collections,
                   reference_time=REF_TIME).run()

    return conn, data_dir


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

def test_report_file_created(tmp_path):
    conn, data_dir = _setup(tmp_path)
    out = ReportStage(conn=conn, data_dir=data_dir, run_id=RUN_ID).run()
    assert out.exists()
    assert out.name == f"report-{RUN_ID}.json"
    conn.close()


def test_report_is_valid_json(tmp_path):
    conn, data_dir = _setup(tmp_path)
    out = ReportStage(conn=conn, data_dir=data_dir, run_id=RUN_ID).run()
    doc = json.loads(out.read_text())
    assert isinstance(doc, dict)
    conn.close()


def test_report_top_level_keys(tmp_path):
    conn, data_dir = _setup(tmp_path)
    out = ReportStage(conn=conn, data_dir=data_dir, run_id=RUN_ID).run()
    doc = json.loads(out.read_text())
    for key in ("run_id", "generated_at", "corpus_info", "calibration",
                 "summary", "dedup_groups", "contradictions", "staleness"):
        assert key in doc, f"missing key: {key}"
    conn.close()


def test_report_run_id_matches(tmp_path):
    conn, data_dir = _setup(tmp_path)
    out = ReportStage(conn=conn, data_dir=data_dir, run_id=RUN_ID).run()
    doc = json.loads(out.read_text())
    assert doc["run_id"] == RUN_ID
    conn.close()


def test_corpus_info_counts(tmp_path):
    conn, data_dir = _setup(tmp_path)
    out = ReportStage(conn=conn, data_dir=data_dir, run_id=RUN_ID).run()
    doc = json.loads(out.read_text())
    ci = doc["corpus_info"]
    assert ci["total_chunks"] == 3
    assert ci["active_chunks"] == 3
    assert ci["store_type"] == "fixture"
    assert len(ci["collections"]) == 1
    assert ci["collections"][0]["name"] == "docs"
    assert ci["collections"][0]["chunk_count"] == 3
    conn.close()


def test_calibration_section_present(tmp_path):
    conn, data_dir = _setup(tmp_path)
    out = ReportStage(conn=conn, data_dir=data_dir, run_id=RUN_ID).run()
    doc = json.loads(out.read_text())
    cal = doc["calibration"]
    assert cal is not None
    assert "dedup_cosine_threshold" in cal
    assert "contradiction_candidate_threshold" in cal
    assert "cluster_min_density" in cal
    conn.close()


def test_summary_counts(tmp_path):
    conn, data_dir = _setup(tmp_path)
    out = ReportStage(conn=conn, data_dir=data_dir, run_id=RUN_ID).run()
    doc = json.loads(out.read_text())
    s = doc["summary"]
    assert s["dedup_groups"] == 0  # no dedup groups set up
    assert s["chunks_in_dedup_groups"] == 0
    assert s["contradictions_unresolved"] == 0
    label_sum = sum(s["staleness_labels"].values())
    assert label_sum == 3  # all 3 chunks scored
    conn.close()


def test_staleness_section_all_chunks(tmp_path):
    conn, data_dir = _setup(tmp_path)
    out = ReportStage(conn=conn, data_dir=data_dir, run_id=RUN_ID).run()
    doc = json.loads(out.read_text())
    ids = {e["chunk_id"] for e in doc["staleness"]}
    assert ids == {"chunk-a", "chunk-b", "chunk-c"}
    conn.close()


def test_staleness_sorted_by_score_desc(tmp_path):
    conn, data_dir = _setup(tmp_path)
    out = ReportStage(conn=conn, data_dir=data_dir, run_id=RUN_ID).run()
    doc = json.loads(out.read_text())
    scores = [e["staleness_score"] for e in doc["staleness"]]
    assert scores == sorted(scores, reverse=True)
    conn.close()


def test_staleness_entry_has_required_fields(tmp_path):
    conn, data_dir = _setup(tmp_path)
    out = ReportStage(conn=conn, data_dir=data_dir, run_id=RUN_ID).run()
    doc = json.loads(out.read_text())
    for entry in doc["staleness"]:
        for field in ("chunk_id", "collection", "text", "staleness_score",
                      "staleness_label", "staleness_components", "is_superseded"):
            assert field in entry, f"missing field {field!r} in staleness entry"
    conn.close()


def test_staleness_components_has_all_keys(tmp_path):
    conn, data_dir = _setup(tmp_path)
    out = ReportStage(conn=conn, data_dir=data_dir, run_id=RUN_ID).run()
    doc = json.loads(out.read_text())
    for entry in doc["staleness"]:
        comp = entry["staleness_components"]
        for key in ("age_decay", "cluster_drift", "contradiction_evidence",
                    "supersession_evidence", "retrieval_abandonment"):
            assert key in comp, f"missing component {key!r}"
    conn.close()


def test_dedup_groups_with_manual_group(tmp_path):
    conn, data_dir = _setup(tmp_path)
    # Manually insert a dedup group
    conn.execute(
        "INSERT INTO dedup_groups (group_id, canonical_chunk_id, member_chunk_ids, detection_channels) "
        "VALUES (1, 'chunk-a', ?, ?)",
        [json.dumps(["chunk-a", "chunk-b"]), json.dumps(["embedding"])],
    )
    out = ReportStage(conn=conn, data_dir=data_dir, run_id=RUN_ID).run()
    doc = json.loads(out.read_text())
    assert len(doc["dedup_groups"]) == 1
    g = doc["dedup_groups"][0]
    assert g["group_id"] == 1
    assert g["canonical_chunk_id"] == "chunk-a"
    assert g["detection_channels"] == ["embedding"]
    member_ids = {m["chunk_id"] for m in g["members"]}
    assert member_ids == {"chunk-a", "chunk-b"}
    canonical_member = next(m for m in g["members"] if m["chunk_id"] == "chunk-a")
    assert canonical_member["is_canonical"] is True
    non_canonical = next(m for m in g["members"] if m["chunk_id"] == "chunk-b")
    assert non_canonical["is_canonical"] is False
    conn.close()


def test_contradictions_with_manual_entry(tmp_path):
    conn, data_dir = _setup(tmp_path)
    conn.execute(
        "INSERT INTO contradictions "
        "(chunk_a, chunk_b, detector, raw_score, calibrated_confidence, "
        " calibration_state, direction, scan_run_id, user_resolution, resolved_at) "
        "VALUES ('chunk-a', 'chunk-b', 'nli', 0.87, NULL, 'uncalibrated', 'both', ?, NULL, NULL)",
        [RUN_ID],
    )
    out = ReportStage(conn=conn, data_dir=data_dir, run_id=RUN_ID).run()
    doc = json.loads(out.read_text())
    assert len(doc["contradictions"]) == 1
    c = doc["contradictions"][0]
    assert c["chunk_a_id"] == "chunk-a"
    assert c["chunk_b_id"] == "chunk-b"
    assert c["detector"] == "nli"
    assert c["raw_score"] == pytest.approx(0.87)
    assert c["calibration_state"] == "uncalibrated"
    assert c["direction"] == "both"
    assert c["chunk_a_text"] == "alpha content"
    assert c["chunk_b_text"] == "beta content different"
    conn.close()


def test_contradictions_excludes_resolved(tmp_path):
    conn, data_dir = _setup(tmp_path)
    conn.execute(
        "INSERT INTO contradictions "
        "(chunk_a, chunk_b, detector, raw_score, calibrated_confidence, "
        " calibration_state, direction, scan_run_id, user_resolution, resolved_at) "
        "VALUES ('chunk-a', 'chunk-b', 'nli', 0.87, NULL, 'uncalibrated', 'both', ?, 'not_a_contradiction', NULL)",
        [RUN_ID],
    )
    out = ReportStage(conn=conn, data_dir=data_dir, run_id=RUN_ID).run()
    doc = json.loads(out.read_text())
    assert doc["contradictions"] == []
    conn.close()


def test_contradictions_scoped_to_run_id(tmp_path):
    conn, data_dir = _setup(tmp_path)
    conn.execute(
        "INSERT INTO contradictions "
        "(chunk_a, chunk_b, detector, raw_score, calibrated_confidence, "
        " calibration_state, direction, scan_run_id, user_resolution, resolved_at) "
        "VALUES ('chunk-a', 'chunk-c', 'nli', 0.91, NULL, 'uncalibrated', 'a->b', 'other-run', NULL, NULL)",
    )
    out = ReportStage(conn=conn, data_dir=data_dir, run_id=RUN_ID).run()
    doc = json.loads(out.read_text())
    assert doc["contradictions"] == []
    conn.close()


def test_report_idempotent(tmp_path):
    conn, data_dir = _setup(tmp_path)
    stage = ReportStage(conn=conn, data_dir=data_dir, run_id=RUN_ID)
    out1 = stage.run()
    doc1 = json.loads(out1.read_text())
    out2 = stage.run()
    doc2 = json.loads(out2.read_text())
    # Everything except generated_at should be identical
    doc1.pop("generated_at")
    doc2.pop("generated_at")
    assert doc1 == doc2
    conn.close()
