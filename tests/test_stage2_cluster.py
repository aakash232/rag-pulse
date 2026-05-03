"""Tests for Stage 2: UMAP + HDBSCAN clustering."""

import json
from pathlib import Path

import numpy as np

from pulse_scan.adapters.fixture import LocalFixtureAdapter
from pulse_scan.config import ClusteringConfig, CollectionConfig, PulseConfig, StoreConfig
from pulse_scan.db.schema import open_db
from pulse_scan.stages.stage0_ingest import IngestStage
from pulse_scan.stages.stage05_calibrate import CalibrateStage
from pulse_scan.stages.stage2_cluster import ClusterStage, _min_cluster_size_auto

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _cfg(corpus_dir: Path) -> PulseConfig:
    return PulseConfig(
        store=StoreConfig(type="fixture"),
        collections=[CollectionConfig(name="test", timestamp_field="created_at")],
        fixture_dir=str(corpus_dir),
    )


def _make_clustered_fixture(corpus_dir: Path, n_per_cluster: int = 8, dim: int = 8) -> None:
    """
    Create a fixture with 3 well-separated clusters + a few noise points.

    Each cluster occupies a different axis corner in `dim`-dimensional space.
    Noise points are random unit vectors far from all clusters.
    """
    rng = np.random.default_rng(0)

    def _unit(v):
        v = v.astype(np.float32)
        return (v / np.linalg.norm(v)).tolist()

    chunks = []
    # Cluster 0: near axis 0
    base0 = np.zeros(dim, dtype=np.float32)
    base0[0] = 1.0
    for i in range(n_per_cluster):
        v = base0 + rng.standard_normal(dim).astype(np.float32) * 0.05
        chunks.append(
            {
                "id": f"cl0-{i:03d}",
                "text": f"cluster zero member {i}",
                "embedding": _unit(v),
                "metadata": {"created_at": "2023-01-01T00:00:00Z"},
            }
        )

    # Cluster 1: near axis 1
    base1 = np.zeros(dim, dtype=np.float32)
    base1[1] = 1.0
    for i in range(n_per_cluster):
        v = base1 + rng.standard_normal(dim).astype(np.float32) * 0.05
        chunks.append(
            {
                "id": f"cl1-{i:03d}",
                "text": f"cluster one member {i}",
                "embedding": _unit(v),
                "metadata": {"created_at": "2023-01-01T00:00:00Z"},
            }
        )

    # Cluster 2: near axis 2
    base2 = np.zeros(dim, dtype=np.float32)
    base2[2] = 1.0
    for i in range(n_per_cluster):
        v = base2 + rng.standard_normal(dim).astype(np.float32) * 0.05
        chunks.append(
            {
                "id": f"cl2-{i:03d}",
                "text": f"cluster two member {i}",
                "embedding": _unit(v),
                "metadata": {"created_at": "2023-01-01T00:00:00Z"},
            }
        )

    # 3 random noise points (orthogonal to all clusters)
    for i in range(3):
        v = rng.standard_normal(dim).astype(np.float32)
        v[:3] = 0.0  # zero out the cluster axes
        chunks.append(
            {
                "id": f"noise-{i:03d}",
                "text": f"isolated noise chunk {i}",
                "embedding": _unit(v),
                "metadata": {"created_at": "2023-01-01T00:00:00Z"},
            }
        )

    (corpus_dir / "test.json").write_text(json.dumps(chunks))


def _ingest(conn, corpus_dir: Path, data_dir: Path) -> None:
    adapter = LocalFixtureAdapter(corpus_dir)
    IngestStage(conn=conn, adapter=adapter, config=_cfg(corpus_dir), data_dir=data_dir).run(run_id="run-001")
    CalibrateStage(conn=conn, data_dir=data_dir).run(scan_run_id="cal-001")


# ---------------------------------------------------------------------------
# min_cluster_size_auto formula
# ---------------------------------------------------------------------------


def test_min_cluster_size_auto_small_corpus():
    assert _min_cluster_size_auto(50) == 5  # sqrt(50)/4 ≈ 1.77 → 1; max(5,1) = 5


def test_min_cluster_size_auto_grows_with_n():
    # sqrt(10000)/4 = 25 → max(5,25) = 25
    assert _min_cluster_size_auto(10_000) == 25


# ---------------------------------------------------------------------------
# Core clustering
# ---------------------------------------------------------------------------


def test_cluster_finds_three_clusters(tmp_path):
    """Three well-separated clusters are detected."""
    corpus_dir = tmp_path / "corpus"
    corpus_dir.mkdir()
    data_dir = tmp_path / "data"
    data_dir.mkdir()
    _make_clustered_fixture(corpus_dir, n_per_cluster=8, dim=8)

    conn = open_db(data_dir)
    _ingest(conn, corpus_dir, data_dir)

    cfg = ClusteringConfig(min_cluster_size=5)
    result = ClusterStage(conn=conn, data_dir=data_dir, clustering_config=cfg).run()

    assert result["clusters_found"] == 3
    assert result["n_chunks"] == 27  # 3*8 + 3 noise
    conn.close()


def test_cluster_assigns_chunk_ids_correctly(tmp_path):
    """clustered chunks have a non-NULL cluster_id; noise chunks have NULL; counts are consistent."""
    corpus_dir = tmp_path / "corpus"
    corpus_dir.mkdir()
    data_dir = tmp_path / "data"
    data_dir.mkdir()
    _make_clustered_fixture(corpus_dir, n_per_cluster=8, dim=8)

    conn = open_db(data_dir)
    _ingest(conn, corpus_dir, data_dir)

    result = ClusterStage(conn=conn, data_dir=data_dir, clustering_config=ClusteringConfig(min_cluster_size=5)).run()

    N = result["n_chunks"]
    expected_in_clusters = N - result["noise_chunks"]

    # clusters table matches non-noise count
    n_in_clusters = conn.execute("SELECT COUNT(*) FROM clusters").fetchone()[0]
    assert n_in_clusters == expected_in_clusters

    # chunks.cluster_id matches clusters table and noise count
    n_with_id = conn.execute(
        "SELECT COUNT(*) FROM chunks WHERE cluster_id IS NOT NULL AND deleted_at IS NULL"
    ).fetchone()[0]
    n_without_id = conn.execute(
        "SELECT COUNT(*) FROM chunks WHERE cluster_id IS NULL AND deleted_at IS NULL"
    ).fetchone()[0]
    assert n_with_id == expected_in_clusters
    assert n_without_id == result["noise_chunks"]
    conn.close()


def test_cluster_centroids_stored(tmp_path):
    """cluster_centroids table has one row per cluster with correct n_chunks."""
    corpus_dir = tmp_path / "corpus"
    corpus_dir.mkdir()
    data_dir = tmp_path / "data"
    data_dir.mkdir()
    _make_clustered_fixture(corpus_dir, n_per_cluster=8, dim=8)

    conn = open_db(data_dir)
    _ingest(conn, corpus_dir, data_dir)
    ClusterStage(conn=conn, data_dir=data_dir, clustering_config=ClusteringConfig(min_cluster_size=5)).run()

    rows = conn.execute(
        "SELECT cluster_id, n_chunks, octet_length(centroid) FROM cluster_centroids ORDER BY cluster_id"
    ).fetchall()
    assert len(rows) == 3
    for cluster_id, n_chunks, centroid_len in rows:
        assert n_chunks >= 5  # at least min_cluster_size members
        assert centroid_len == 8 * 4  # dim=8 float32 values → 32 bytes
    conn.close()


def test_cluster_centroid_recoverable_as_ndarray(tmp_path):
    """Centroid blob round-trips through np.frombuffer correctly."""
    corpus_dir = tmp_path / "corpus"
    corpus_dir.mkdir()
    data_dir = tmp_path / "data"
    data_dir.mkdir()
    _make_clustered_fixture(corpus_dir, n_per_cluster=8, dim=8)

    conn = open_db(data_dir)
    _ingest(conn, corpus_dir, data_dir)
    ClusterStage(conn=conn, data_dir=data_dir, clustering_config=ClusteringConfig(min_cluster_size=5)).run()

    row = conn.execute("SELECT centroid FROM cluster_centroids LIMIT 1").fetchone()
    centroid = np.frombuffer(row[0], dtype=np.float32)
    assert centroid.shape == (8,)
    assert np.all(np.isfinite(centroid))
    conn.close()


# ---------------------------------------------------------------------------
# UMAP model caching
# ---------------------------------------------------------------------------


def test_umap_model_cached_after_run(tmp_path):
    """UMAP model is written to disk after first run."""
    corpus_dir = tmp_path / "corpus"
    corpus_dir.mkdir()
    data_dir = tmp_path / "data"
    data_dir.mkdir()
    _make_clustered_fixture(corpus_dir, n_per_cluster=8, dim=8)

    conn = open_db(data_dir)
    _ingest(conn, corpus_dir, data_dir)
    ClusterStage(conn=conn, data_dir=data_dir, clustering_config=ClusteringConfig(min_cluster_size=5)).run()

    assert (data_dir / "umap_model.joblib").exists()
    assert (data_dir / "umap_meta.json").exists()
    conn.close()


def test_should_refit_true_before_first_run(tmp_path):
    """should_refit() returns True when no cached model exists."""
    data_dir = tmp_path / "data"
    data_dir.mkdir()
    conn = open_db(data_dir)
    stage = ClusterStage(conn=conn, data_dir=data_dir)
    assert stage.should_refit() is True
    conn.close()


def test_should_refit_false_after_run(tmp_path):
    """should_refit() returns False immediately after a successful run."""
    corpus_dir = tmp_path / "corpus"
    corpus_dir.mkdir()
    data_dir = tmp_path / "data"
    data_dir.mkdir()
    _make_clustered_fixture(corpus_dir, n_per_cluster=8, dim=8)

    conn = open_db(data_dir)
    _ingest(conn, corpus_dir, data_dir)
    stage = ClusterStage(conn=conn, data_dir=data_dir, clustering_config=ClusteringConfig(min_cluster_size=5))
    stage.run()
    assert stage.should_refit() is False
    conn.close()


# ---------------------------------------------------------------------------
# Idempotency
# ---------------------------------------------------------------------------


def test_clustering_idempotent(tmp_path):
    """Running clustering twice produces identical results."""
    corpus_dir = tmp_path / "corpus"
    corpus_dir.mkdir()
    data_dir = tmp_path / "data"
    data_dir.mkdir()
    _make_clustered_fixture(corpus_dir, n_per_cluster=8, dim=8)

    conn = open_db(data_dir)
    _ingest(conn, corpus_dir, data_dir)
    stage = ClusterStage(conn=conn, data_dir=data_dir, clustering_config=ClusteringConfig(min_cluster_size=5))
    r1 = stage.run()
    r2 = stage.run()
    assert r1 == r2

    # Second run must not double-insert: total rows must equal first run's count
    n = conn.execute("SELECT COUNT(*) FROM clusters").fetchone()[0]
    assert n == r1["n_chunks"] - r1["noise_chunks"]
    conn.close()


# ---------------------------------------------------------------------------
# Edge cases
# ---------------------------------------------------------------------------


def test_clustering_skips_fewer_than_two_chunks(tmp_path):
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
                    "embedding": [1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                    "metadata": {"created_at": "2023-01-01T00:00:00Z"},
                }
            ]
        )
    )

    conn = open_db(data_dir)
    adapter = LocalFixtureAdapter(corpus_dir)
    IngestStage(conn=conn, adapter=adapter, config=_cfg(corpus_dir), data_dir=data_dir).run(run_id="run-001")
    CalibrateStage(conn=conn, data_dir=data_dir).run(scan_run_id="cal-001")

    result = ClusterStage(conn=conn, data_dir=data_dir).run()
    assert result["clusters_found"] == 0
    conn.close()
