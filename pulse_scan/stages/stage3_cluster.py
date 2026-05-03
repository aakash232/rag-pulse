"""Stage 3: Clustering — UMAP dimensionality reduction + HDBSCAN.

Reduces chunk embeddings to 50d via UMAP, then clusters with HDBSCAN.
Chunks labeled -1 (noise) get cluster_id = NULL in the chunks table and
are treated as orphan candidates downstream.

The fitted UMAP model is cached to disk. It is re-fitted only when >20%
new chunks have appeared since the last fit, ensuring stable cluster
assignments across incremental scans.
"""

import json
from pathlib import Path

import duckdb
import hdbscan
import joblib
import numpy as np
import structlog
from umap import UMAP

log = structlog.get_logger()

UMAP_COMPONENTS = 50
UMAP_N_NEIGHBORS = 15
UMAP_MIN_DIST = 0.1
UMAP_RANDOM_STATE = 42
HDBSCAN_MIN_SAMPLES = 3
REFIT_THRESHOLD = 0.20  # re-fit UMAP if corpus grew >20% since last fit


def _min_cluster_size_auto(N: int) -> int:
    return max(5, int(np.sqrt(N) / 4))


class ClusterStage:
    def __init__(
        self,
        conn: duckdb.DuckDBPyConnection,
        data_dir: Path,
        clustering_config=None,  # ClusteringConfig or None
    ):
        self.conn = conn
        self.data_dir = data_dir
        self._cfg = clustering_config

    # ------------------------------------------------------------------
    # Public interface
    # ------------------------------------------------------------------

    def should_refit(self) -> bool:
        """True if no cached UMAP model, or corpus grew >20% since last fit."""
        model_path = self.data_dir / "umap_model.joblib"
        meta_path = self.data_dir / "umap_meta.json"
        if not model_path.exists() or not meta_path.exists():
            return True
        meta = json.loads(meta_path.read_text())
        last_n = meta.get("n_chunks_at_fit", 0)
        if last_n == 0:
            return True
        current_n = self._count_active_chunks()
        return (current_n - last_n) / last_n > REFIT_THRESHOLD

    def run(self) -> dict:
        rows = self.conn.execute(
            "SELECT chunk_id, embedding_offset "
            "FROM chunks "
            "WHERE deleted_at IS NULL AND embedding_offset >= 0 "
            "ORDER BY chunk_id"
        ).fetchall()

        N = len(rows)
        if N < 2:
            log.info("clustering_skipped_too_few_chunks", n_chunks=N)
            return {"clusters_found": 0, "noise_chunks": N, "n_chunks": N}

        meta_path = self.data_dir / "embeddings.meta.json"
        meta = json.loads(meta_path.read_text())
        dim: int = meta["dim"]
        n_allocated: int = meta["n_allocated"]

        chunk_ids = [r[0] for r in rows]
        offsets = [r[1] for r in rows]

        mmap = np.memmap(
            self.data_dir / "embeddings.f32.npy",
            dtype=np.float32,
            mode="r",
            shape=(n_allocated, dim),
        )
        embeddings = np.array(mmap[offsets], dtype=np.float32)
        del mmap

        norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
        norms[norms == 0] = 1.0
        embeddings /= norms

        # UMAP reduction
        # Spectral init requires n_components + 1 < N → cap at N-2
        n_components = min(UMAP_COMPONENTS, max(2, N - 2))
        n_neighbors = min(UMAP_N_NEIGHBORS, N - 1)

        model_path = self.data_dir / "umap_model.joblib"
        if self.should_refit():
            reducer = self._make_umap(n_components, n_neighbors)
            reduced = reducer.fit_transform(embeddings)
            joblib.dump(reducer, model_path)
            (self.data_dir / "umap_meta.json").write_text(
                json.dumps({"n_chunks_at_fit": N, "n_components": n_components})
            )
            log.info("umap_fitted", n_chunks=N, n_components=n_components, n_neighbors=n_neighbors)
        else:
            reducer = joblib.load(model_path)
            reduced = reducer.transform(embeddings)
            log.info("umap_cached_transform", n_chunks=N)

        # HDBSCAN clustering
        min_cs = self._resolve_min_cluster_size(N, reduced)
        clusterer = hdbscan.HDBSCAN(
            min_cluster_size=min_cs,
            min_samples=HDBSCAN_MIN_SAMPLES,
            metric="euclidean",
        )
        labels: np.ndarray = clusterer.fit_predict(reduced)

        n_clusters, noise_count = self._persist(chunk_ids, labels, embeddings)

        log.info(
            "clustering_complete",
            n_chunks=N,
            clusters_found=n_clusters,
            noise_chunks=noise_count,
            min_cluster_size=min_cs,
        )
        return {"clusters_found": n_clusters, "noise_chunks": noise_count, "n_chunks": N}

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _make_umap(self, n_components: int, n_neighbors: int) -> UMAP:
        use_gpu = self._cfg.use_gpu if self._cfg else False
        if use_gpu:
            try:
                from cuml.manifold import UMAP as cuMLUMAP  # type: ignore[import]

                log.info("umap_using_gpu_cuml")
                return cuMLUMAP(
                    n_components=n_components,
                    n_neighbors=n_neighbors,
                    min_dist=UMAP_MIN_DIST,
                    random_state=UMAP_RANDOM_STATE,
                )
            except ImportError:
                log.warning("cuml_not_available_falling_back_to_cpu_umap")
        return UMAP(
            n_components=n_components,
            n_neighbors=n_neighbors,
            min_dist=UMAP_MIN_DIST,
            random_state=UMAP_RANDOM_STATE,
        )

    def _resolve_min_cluster_size(self, N: int, reduced: np.ndarray) -> int:
        cfg_val = self._cfg.min_cluster_size if self._cfg else "auto"
        auto_tune = self._cfg.auto_tune_clustering if self._cfg else False

        if cfg_val != "auto":
            return int(cfg_val)

        base = _min_cluster_size_auto(N)

        if auto_tune:
            return self._auto_tune(reduced, base)
        return base

    def _auto_tune(self, reduced: np.ndarray, base: int) -> int:
        from sklearn.metrics import silhouette_score  # imported lazily: only used when opted in

        candidates = sorted({max(2, base // 2), base, base * 2})
        best_score = -2.0
        best_size = base

        for size in candidates:
            labels = hdbscan.HDBSCAN(
                min_cluster_size=size,
                min_samples=HDBSCAN_MIN_SAMPLES,
                metric="euclidean",
            ).fit_predict(reduced)
            mask = labels != -1
            n_non_noise = int(mask.sum())
            n_clusters = len(set(labels[mask]))
            if n_non_noise < 2 or n_clusters < 2:
                continue
            score = float(silhouette_score(reduced[mask], labels[mask]))
            log.info("auto_tune_candidate", min_cluster_size=size, silhouette=round(score, 4))
            if score > best_score:
                best_score = score
                best_size = size

        log.info("auto_tune_selected", min_cluster_size=best_size, silhouette=round(best_score, 4))
        return best_size

    def _persist(
        self,
        chunk_ids: list[str],
        labels: np.ndarray,
        embeddings: np.ndarray,
    ) -> tuple[int, int]:
        # Clear previous results (idempotent)
        self.conn.execute("DELETE FROM clusters")
        self.conn.execute("DELETE FROM cluster_centroids")
        self.conn.execute("UPDATE chunks SET cluster_id = NULL")

        unique_labels = sorted(set(labels.tolist()))
        cluster_labels = [lb for lb in unique_labels if lb != -1]

        for cluster_id in cluster_labels:
            mask = labels == cluster_id
            member_ids = [chunk_ids[i] for i, m in enumerate(mask) if m]
            centroid = embeddings[mask].mean(axis=0).astype(np.float32)

            for cid in member_ids:
                self.conn.execute(
                    "INSERT INTO clusters (cluster_id, chunk_id) VALUES (?, ?)",
                    [cluster_id, cid],
                )
                self.conn.execute(
                    "UPDATE chunks SET cluster_id = ? WHERE chunk_id = ? AND deleted_at IS NULL",
                    [cluster_id, cid],
                )

            self.conn.execute(
                "INSERT INTO cluster_centroids (cluster_id, centroid, n_chunks) VALUES (?, ?, ?)",
                [cluster_id, centroid.tobytes(), int(mask.sum())],
            )

        noise_count = int((labels == -1).sum())
        return len(cluster_labels), noise_count

    def _count_active_chunks(self) -> int:
        row = self.conn.execute(
            "SELECT COUNT(*) FROM chunks WHERE deleted_at IS NULL AND embedding_offset >= 0"
        ).fetchone()
        return row[0] if row else 0
