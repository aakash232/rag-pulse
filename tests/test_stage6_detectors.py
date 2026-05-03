"""Tests for Stage 6: Numeric and Version contradiction detectors."""

import json
from pathlib import Path

import numpy as np
import pytest

from pulse_scan.adapters.fixture import LocalFixtureAdapter
from pulse_scan.config import CollectionConfig, PulseConfig, StoreConfig
from pulse_scan.db.schema import open_db
from pulse_scan.stages.stage0_ingest import IngestStage
from pulse_scan.stages.stage1_calibrate import CalibrateStage
from pulse_scan.stages.stage6_detectors import (
    NumericContradictionDetector,
    VersionContradictionDetector,
    is_numeric_contradiction,
    is_version_contradiction,
)

DIM = 4
RUN_ID = "run-detectors"


# ---------------------------------------------------------------------------
# Unit tests for pure detector functions
# ---------------------------------------------------------------------------


def test_numeric_contradiction_different_numbers():
    text_a = "The price of the widget is 100 dollars per unit"
    text_b = "The price of the widget is 200 dollars per unit"
    ok, score = is_numeric_contradiction(text_a, text_b)
    assert ok is True
    assert score > 0.0


def test_numeric_no_contradiction_same_numbers():
    text_a = "The price of the widget is 100 dollars per unit"
    text_b = "The price of the widget is 100 dollars per unit in stock"
    ok, _ = is_numeric_contradiction(text_a, text_b)
    assert ok is False


def test_numeric_no_contradiction_no_numbers():
    text_a = "The product is available in red and blue"
    text_b = "The product is available in green and yellow"
    ok, _ = is_numeric_contradiction(text_a, text_b)
    assert ok is False


def test_numeric_no_contradiction_low_context_similarity():
    text_a = "The dosage is 500 mg per tablet for patients"
    text_b = "There are 3 lanes on the highway to the city center"
    ok, _ = is_numeric_contradiction(text_a, text_b)
    assert ok is False


def test_version_contradiction_different_versions():
    text_a = "The software requires Python 3.9 to run properly"
    text_b = "The software requires Python 3.11 to run properly"
    ok, score = is_version_contradiction(text_a, text_b)
    assert ok is True
    assert score == 1.0


def test_version_no_contradiction_same_version():
    text_a = "Install version 2.3.1 of the library for this feature"
    text_b = "Install version 2.3.1 of the library using pip"
    ok, _ = is_version_contradiction(text_a, text_b)
    assert ok is False


def test_version_no_contradiction_no_version():
    text_a = "This document describes the installation process"
    text_b = "This document describes the configuration options"
    ok, _ = is_version_contradiction(text_a, text_b)
    assert ok is False


def test_version_no_contradiction_low_context_similarity():
    text_a = "Use Django 4.2 to build web applications"
    text_b = "The car model 3.5 comes with leather seats and a sunroof"
    ok, _ = is_version_contradiction(text_a, text_b)
    assert ok is False


# ---------------------------------------------------------------------------
# Integration tests against DuckDB
# ---------------------------------------------------------------------------


def _make_corpus(corpus_dir: Path, chunks: list) -> None:
    (corpus_dir / "docs.json").write_text(json.dumps(chunks))


def _cfg(corpus_dir: Path) -> PulseConfig:
    return PulseConfig(
        store=StoreConfig(type="fixture"),
        collections=[CollectionConfig(name="docs", timestamp_field="created_at")],
        fixture_dir=str(corpus_dir),
    )


def _base_vec(dim: int = DIM) -> list[float]:
    v = np.zeros(dim, dtype=np.float32)
    v[0] = 1.0
    return v.tolist()


def _ingest_and_cluster(conn, corpus_dir, data_dir, chunks):
    _make_corpus(corpus_dir, chunks)
    cfg = _cfg(corpus_dir)
    IngestStage(conn=conn, adapter=LocalFixtureAdapter(corpus_dir), config=cfg, data_dir=data_dir).run(RUN_ID)
    CalibrateStage(conn=conn, data_dir=data_dir).run(scan_run_id=RUN_ID)
    # Assign all chunks to cluster 0
    conn.execute("UPDATE chunks SET cluster_id = 0 WHERE deleted_at IS NULL")


def test_numeric_detector_finds_contradiction(tmp_path):
    corpus_dir = tmp_path / "corpus"
    corpus_dir.mkdir()
    data_dir = tmp_path / "data"
    data_dir.mkdir()

    chunks = [
        {
            "id": "a",
            "text": "The price of the widget is 100 dollars per unit",
            "embedding": _base_vec(),
            "metadata": {"created_at": "2024-01-01T00:00:00Z"},
        },
        {
            "id": "b",
            "text": "The price of the widget is 200 dollars per unit",
            "embedding": _base_vec(),
            "metadata": {"created_at": "2024-01-01T00:00:00Z"},
        },
    ]
    conn = open_db(data_dir)
    _ingest_and_cluster(conn, corpus_dir, data_dir, chunks)

    result = NumericContradictionDetector(conn=conn, scan_run_id=RUN_ID).run()
    assert result["contradictions_found"] == 1
    row = conn.execute("SELECT detector, direction FROM contradictions").fetchone()
    assert row[0] == "numeric"
    assert row[1] == "both"
    conn.close()


def test_numeric_detector_no_contradictions_same_numbers(tmp_path):
    corpus_dir = tmp_path / "corpus"
    corpus_dir.mkdir()
    data_dir = tmp_path / "data"
    data_dir.mkdir()

    chunks = [
        {
            "id": "a",
            "text": "The product costs 50 euros at checkout",
            "embedding": _base_vec(),
            "metadata": {"created_at": "2024-01-01T00:00:00Z"},
        },
        {
            "id": "b",
            "text": "The product costs 50 euros including VAT",
            "embedding": _base_vec(),
            "metadata": {"created_at": "2024-01-01T00:00:00Z"},
        },
    ]
    conn = open_db(data_dir)
    _ingest_and_cluster(conn, corpus_dir, data_dir, chunks)

    result = NumericContradictionDetector(conn=conn, scan_run_id=RUN_ID).run()
    assert result["contradictions_found"] == 0
    conn.close()


def test_numeric_detector_skips_noise_chunks(tmp_path):
    """Chunks without cluster_id (noise) are excluded from candidate pairs."""
    corpus_dir = tmp_path / "corpus"
    corpus_dir.mkdir()
    data_dir = tmp_path / "data"
    data_dir.mkdir()

    chunks = [
        {
            "id": "a",
            "text": "The price is 100 dollars per item right here",
            "embedding": _base_vec(),
            "metadata": {"created_at": "2024-01-01T00:00:00Z"},
        },
        {
            "id": "b",
            "text": "The price is 200 dollars per item right here",
            "embedding": _base_vec(),
            "metadata": {"created_at": "2024-01-01T00:00:00Z"},
        },
    ]
    conn = open_db(data_dir)
    _make_corpus(corpus_dir, chunks)
    cfg = _cfg(corpus_dir)
    IngestStage(conn=conn, adapter=LocalFixtureAdapter(corpus_dir), config=cfg, data_dir=data_dir).run(RUN_ID)
    CalibrateStage(conn=conn, data_dir=data_dir).run(scan_run_id=RUN_ID)
    # Leave cluster_id = NULL (noise)

    result = NumericContradictionDetector(conn=conn, scan_run_id=RUN_ID).run()
    assert result["pairs_checked"] == 0
    assert result["contradictions_found"] == 0
    conn.close()


def test_numeric_detector_idempotent(tmp_path):
    corpus_dir = tmp_path / "corpus"
    corpus_dir.mkdir()
    data_dir = tmp_path / "data"
    data_dir.mkdir()

    chunks = [
        {
            "id": "a",
            "text": "The price of the widget is 100 dollars per unit",
            "embedding": _base_vec(),
            "metadata": {"created_at": "2024-01-01T00:00:00Z"},
        },
        {
            "id": "b",
            "text": "The price of the widget is 200 dollars per unit",
            "embedding": _base_vec(),
            "metadata": {"created_at": "2024-01-01T00:00:00Z"},
        },
    ]
    conn = open_db(data_dir)
    _ingest_and_cluster(conn, corpus_dir, data_dir, chunks)

    det = NumericContradictionDetector(conn=conn, scan_run_id=RUN_ID)
    det.run()
    count_1 = conn.execute("SELECT COUNT(*) FROM contradictions WHERE detector = 'numeric'").fetchone()[0]
    det.run()
    count_2 = conn.execute("SELECT COUNT(*) FROM contradictions WHERE detector = 'numeric'").fetchone()[0]
    assert count_1 == count_2 == 1
    conn.close()


def test_version_detector_finds_contradiction(tmp_path):
    corpus_dir = tmp_path / "corpus"
    corpus_dir.mkdir()
    data_dir = tmp_path / "data"
    data_dir.mkdir()

    chunks = [
        {
            "id": "a",
            "text": "The software requires Python 3.9 to run properly on this system",
            "embedding": _base_vec(),
            "metadata": {"created_at": "2024-01-01T00:00:00Z"},
        },
        {
            "id": "b",
            "text": "The software requires Python 3.11 to run properly on this system",
            "embedding": _base_vec(),
            "metadata": {"created_at": "2024-01-01T00:00:00Z"},
        },
    ]
    conn = open_db(data_dir)
    _ingest_and_cluster(conn, corpus_dir, data_dir, chunks)

    result = VersionContradictionDetector(conn=conn, scan_run_id=RUN_ID).run()
    assert result["contradictions_found"] == 1
    row = conn.execute("SELECT detector, raw_score FROM contradictions").fetchone()
    assert row[0] == "version"
    assert row[1] == pytest.approx(1.0)
    conn.close()


def test_version_detector_no_contradictions_no_versions(tmp_path):
    corpus_dir = tmp_path / "corpus"
    corpus_dir.mkdir()
    data_dir = tmp_path / "data"
    data_dir.mkdir()

    chunks = [
        {
            "id": "a",
            "text": "Install the library using pip install library",
            "embedding": _base_vec(),
            "metadata": {"created_at": "2024-01-01T00:00:00Z"},
        },
        {
            "id": "b",
            "text": "Install the library using conda install library",
            "embedding": _base_vec(),
            "metadata": {"created_at": "2024-01-01T00:00:00Z"},
        },
    ]
    conn = open_db(data_dir)
    _ingest_and_cluster(conn, corpus_dir, data_dir, chunks)

    result = VersionContradictionDetector(conn=conn, scan_run_id=RUN_ID).run()
    assert result["contradictions_found"] == 0
    conn.close()


def test_version_detector_idempotent(tmp_path):
    corpus_dir = tmp_path / "corpus"
    corpus_dir.mkdir()
    data_dir = tmp_path / "data"
    data_dir.mkdir()

    chunks = [
        {
            "id": "a",
            "text": "The software requires Python 3.9 to run on this system",
            "embedding": _base_vec(),
            "metadata": {"created_at": "2024-01-01T00:00:00Z"},
        },
        {
            "id": "b",
            "text": "The software requires Python 3.11 to run on this system",
            "embedding": _base_vec(),
            "metadata": {"created_at": "2024-01-01T00:00:00Z"},
        },
    ]
    conn = open_db(data_dir)
    _ingest_and_cluster(conn, corpus_dir, data_dir, chunks)

    det = VersionContradictionDetector(conn=conn, scan_run_id=RUN_ID)
    det.run()
    count_1 = conn.execute("SELECT COUNT(*) FROM contradictions WHERE detector = 'version'").fetchone()[0]
    det.run()
    count_2 = conn.execute("SELECT COUNT(*) FROM contradictions WHERE detector = 'version'").fetchone()[0]
    assert count_1 == count_2 == 1
    conn.close()
