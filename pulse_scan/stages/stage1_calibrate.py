"""Stage 1: Calibration.

Computes per-corpus cosine similarity thresholds before any thresholded
operation runs. Without this, dedup and contradiction detection use
hardcoded numbers that break across different embedding models.

Two paths:
  Small corpus (<500 chunks): model-specific defaults from dimension lookup.
  Large corpus (>=500 chunks): HNSW-based empirical distribution sampling.

HNSW threshold formulas (formula_version=2):

  contradiction = baseline_p95 + 0.5 * gap
    where gap = neighbor_p50 - baseline_p95
    → midpoint between "clearly random" and "clearly topically related";
      robust across compressed (OpenAI) and spread (MiniLM) distributions.

  dedup = neighbor_p99 - 0.1 * spread
    where spread = neighbor_p99 - neighbor_p50
    → just below the upper tail of the neighbor distribution; wide spread
      (MiniLM) yields a threshold well below 1.0, compressed spread (OpenAI)
      yields one near 0.999 — both correct for their model family.

  density = neighbor_p50
    → unchanged; median neighbor similarity describes typical cluster tightness.

If the resulting thresholds violate baseline_p95 < contradiction < dedup < 1.0,
the distribution is degenerate and the stage falls back to model_defaults with
a warning log.
"""

import json
from pathlib import Path
from typing import Optional

import duckdb
import hnswlib
import numpy as np
import structlog
import xxhash

log = structlog.get_logger()

# From LLD §3 Stage 0
KNOWN_MODELS: dict[int, list[str]] = {
    1536: ["openai-ada-002", "openai-text-embedding-3-small"],
    3072: ["openai-text-embedding-3-large"],
    768: ["sentence-transformers/all-mpnet-base-v2", "BGE-base", "E5-base"],
    1024: ["BGE-large", "E5-large", "Voyage-2", "Cohere-embed-v3"],
    384: ["sentence-transformers/all-MiniLM-L6-v2"],
}

# Conservative per-model defaults (for small corpora where distributions are noisy)
_MODEL_DEFAULTS: dict[int, tuple[float, float, float]] = {
    # dim: (dedup_threshold, contradiction_candidate_threshold, cluster_density)
    384: (0.950, 0.850, 0.750),
    768: (0.930, 0.820, 0.720),
    1024: (0.920, 0.800, 0.700),
    1536: (0.900, 0.780, 0.680),
    3072: (0.880, 0.760, 0.650),
}
_GENERIC_DEFAULTS: tuple[float, float, float] = (0.920, 0.820, 0.720)

SAMPLE_SIZE = 2000
K_NEIGHBORS = 10
GROWTH_THRESHOLD = 0.50  # re-calibrate if corpus grows by >50%
NLI_FEEDBACK_MIN_LABELS = 5  # minimum labeled contradictions to attempt re-tuning
NLI_DEFAULT_THRESHOLD = 0.50


def _model_defaults(dim: Optional[int]) -> tuple[float, float, float]:
    if dim is None:
        return _GENERIC_DEFAULTS
    if dim in _MODEL_DEFAULTS:
        return _MODEL_DEFAULTS[dim]
    # Multiple known models for this dim → use most conservative (lowest thresholds)
    closest = min(_MODEL_DEFAULTS, key=lambda d: abs(d - dim))
    return _MODEL_DEFAULTS[closest]


def _sample_hash(chunk_ids: list[str]) -> str:
    h = xxhash.xxh64()
    for cid in sorted(chunk_ids):
        h.update(cid.encode())
    return h.hexdigest()


class CalibrateStage:
    def __init__(
        self,
        conn: duckdb.DuckDBPyConnection,
        data_dir: Path,
        small_corpus_threshold: int = 500,
    ):
        self.conn = conn
        self.data_dir = data_dir
        self.small_corpus_threshold = small_corpus_threshold

    # ------------------------------------------------------------------
    # Public interface
    # ------------------------------------------------------------------

    def should_run(self) -> bool:
        """True on first scan, or if corpus grew >50% since last calibration."""
        last_dist = self._last_distributions()
        if last_dist is None:
            return True
        last_size = last_dist.get("corpus_size", 0)
        if last_size == 0:
            return True
        current_size = self._count_active_chunks()
        return (current_size - last_size) / last_size > GROWTH_THRESHOLD

    def run(self, scan_run_id: str) -> dict:
        """Run calibration. Returns threshold dict."""
        rows = self.conn.execute(
            "SELECT chunk_id, embedding_offset FROM chunks WHERE deleted_at IS NULL AND embedding_offset >= 0"
        ).fetchall()

        meta_path = self.data_dir / "embeddings.meta.json"

        if not rows or not meta_path.exists():
            log.warning("calibration_no_data", n_rows=len(rows), meta_exists=meta_path.exists())
            dedup, contradiction, density = _model_defaults(None)
            return self._as_dict(dedup, contradiction, density)

        meta = json.loads(meta_path.read_text())
        dim: int = meta["dim"]
        corpus_size = len(rows)

        if corpus_size < self.small_corpus_threshold:
            return self._small_corpus_path(dim, corpus_size, rows, scan_run_id)
        else:
            return self._hnsw_path(rows, meta, scan_run_id)

    # ------------------------------------------------------------------
    # Internal: small corpus path
    # ------------------------------------------------------------------

    def _small_corpus_path(
        self,
        dim: int,
        corpus_size: int,
        rows: list[tuple],
        scan_run_id: str,
    ) -> dict:
        dedup, contradiction, density = _model_defaults(dim)
        chunk_ids = [r[0] for r in rows]
        distributions = {
            "method": "model_defaults",
            "dim": dim,
            "corpus_size": corpus_size,
            "model_family": KNOWN_MODELS.get(dim, ["unknown"]),
        }
        self._persist(
            scan_run_id,
            _sample_hash(chunk_ids[:SAMPLE_SIZE]),
            dedup,
            contradiction,
            density,
            distributions,
        )
        log.info(
            "calibration_complete",
            method="model_defaults",
            dim=dim,
            corpus_size=corpus_size,
            dedup_cosine_threshold=dedup,
            contradiction_candidate_threshold=contradiction,
            cluster_min_density=density,
        )
        return self._as_dict(dedup, contradiction, density)

    # ------------------------------------------------------------------
    # Internal: HNSW distribution-based path
    # ------------------------------------------------------------------

    def _hnsw_path(
        self,
        rows: list[tuple],
        meta: dict,
        scan_run_id: str,
    ) -> dict:
        dim: int = meta["dim"]
        n_allocated: int = meta["n_allocated"]
        chunk_ids = [r[0] for r in rows]
        offsets = [r[1] for r in rows]
        corpus_size = len(rows)

        # Load and normalize embeddings from memmap
        mmap = np.memmap(
            self.data_dir / "embeddings.f32.npy",
            dtype=np.float32,
            mode="r",
            shape=(n_allocated, dim),
        )
        all_embs = np.array(mmap[offsets], dtype=np.float32)
        del mmap

        norms = np.linalg.norm(all_embs, axis=1, keepdims=True)
        norms[norms == 0] = 1.0
        all_embs /= norms

        # Sample
        N = min(SAMPLE_SIZE, corpus_size)
        rng = np.random.default_rng(42)
        sample_idx = rng.choice(corpus_size, size=N, replace=False) if N < corpus_size else np.arange(N)
        sample_embs = all_embs[sample_idx].astype(np.float32)
        sample_ids = [chunk_ids[i] for i in sample_idx]

        # Baseline distribution: random pairs (approximates expected cosine)
        n_pairs = min(10_000, N * (N - 1) // 2)
        ia = rng.integers(0, N, size=n_pairs * 2)
        ib = rng.integers(0, N, size=n_pairs * 2)
        valid = ia != ib
        ia, ib = ia[valid][:n_pairs], ib[valid][:n_pairs]
        baseline_cosines = (sample_embs[ia] * sample_embs[ib]).sum(axis=1)

        # Neighbor distribution: HNSW nearest-neighbor cosines
        k = min(K_NEIGHBORS + 1, N)  # +1 because first result is self
        index = hnswlib.Index(space="cosine", dim=dim)
        index.init_index(max_elements=N, ef_construction=200, M=16)
        index.set_ef(max(50, k * 2))
        index.add_items(sample_embs, np.arange(N))

        _, distances = index.knn_query(sample_embs, k=k)
        # hnswlib cosine space: distance = 1 - cosine_similarity
        # Exclude first column (self-match, distance ≈ 0).
        # Clip to [-1, 1]: float32 rounding on near-identical vectors can produce
        # distances slightly below 0, yielding cosines just above 1.0.
        neighbor_cosines = np.clip(1.0 - distances[:, 1:], -1.0, 1.0).flatten()

        # Derive thresholds using gap-based formulas (formula_version=2).
        # See module docstring for the full rationale.
        baseline_p95 = float(np.percentile(baseline_cosines, 95))
        nbr_p50 = float(np.percentile(neighbor_cosines, 50))
        nbr_p99 = float(np.percentile(neighbor_cosines, 99))

        gap = nbr_p50 - baseline_p95
        spread = nbr_p99 - nbr_p50
        contradiction = baseline_p95 + 0.5 * gap
        dedup = nbr_p99 - 0.1 * spread
        density = nbr_p50

        # Sanity check: thresholds must be strictly ordered and below 1.0.
        # A failure means the distribution is degenerate (all-identical embeddings,
        # un-normalised vectors, etc.). Fall back to model_defaults so downstream
        # stages are not silenced.
        if not (baseline_p95 < contradiction < dedup < 1.0):
            log.warning(
                "calibration_hnsw_degenerate_fallback",
                baseline_p95=round(baseline_p95, 4),
                contradiction_computed=round(contradiction, 4),
                dedup_computed=round(dedup, 4),
                action="falling_back_to_model_defaults",
            )
            dedup, contradiction, density = _model_defaults(dim)

        distributions = {
            "method": "hnsw",
            "formula_version": 2,
            "dim": dim,
            "corpus_size": corpus_size,
            "sample_size": N,
            "baseline_p50": float(np.median(baseline_cosines)),
            "baseline_p95": baseline_p95,
            "neighbor_p50": nbr_p50,
            "neighbor_p99": nbr_p99,
            "gap": round(gap, 4),
            "spread": round(spread, 4),
        }
        self._persist(
            scan_run_id,
            _sample_hash(sample_ids),
            dedup,
            contradiction,
            density,
            distributions,
        )
        log.info(
            "calibration_complete",
            method="hnsw",
            corpus_size=corpus_size,
            sample_size=N,
            baseline_p95=round(baseline_p95, 4),
            gap=round(gap, 4),
            spread=round(spread, 4),
            dedup_cosine_threshold=round(dedup, 4),
            contradiction_candidate_threshold=round(contradiction, 4),
            cluster_min_density=round(density, 4),
        )
        return self._as_dict(dedup, contradiction, density)

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _persist(
        self,
        scan_run_id: str,
        sample_hash: str,
        dedup: float,
        contradiction: float,
        density: float,
        distributions: dict,
    ) -> None:
        self.conn.execute(
            """INSERT INTO calibration (
                scan_run_id, sample_hash,
                dedup_threshold, contradiction_candidate_threshold, cluster_min_density,
                distributions
            ) VALUES (?, ?, ?, ?, ?, ?)""",
            [scan_run_id, sample_hash, dedup, contradiction, density, json.dumps(distributions)],
        )

    def _last_distributions(self) -> Optional[dict]:
        row = self.conn.execute("SELECT distributions FROM calibration ORDER BY rowid DESC LIMIT 1").fetchone()
        if row is None or row[0] is None:
            return None
        return json.loads(row[0])

    def _count_active_chunks(self) -> int:
        row = self.conn.execute("SELECT COUNT(*) FROM chunks WHERE deleted_at IS NULL").fetchone()
        return row[0] if row else 0

    @staticmethod
    def _as_dict(dedup: float, contradiction: float, density: float) -> dict:
        return {
            "dedup_cosine_threshold": dedup,
            "contradiction_candidate_threshold": contradiction,
            "cluster_min_density": density,
        }


def retune_nli_threshold(conn: duckdb.DuckDBPyConnection) -> float:
    """Re-tune the NLI contradiction score threshold using human-labeled feedback.

    Reads confirmed/false_positive rows from the contradictions table, sweeps
    candidate thresholds, picks the one maximising F0.5 (precision-weighted),
    then writes the result back into the latest calibration row's distributions
    JSON.  Also back-fills calibrated_confidence and calibration_state on all
    NLI contradiction rows.

    Returns the chosen threshold (or NLI_DEFAULT_THRESHOLD if too few labels).
    """
    rows = conn.execute(
        "SELECT raw_score, user_resolution "
        "FROM contradictions "
        "WHERE detector = 'nli' AND user_resolution IN ('confirmed', 'false_positive')"
    ).fetchall()

    confirmed = [float(r[0]) for r in rows if r[1] == "confirmed"]
    false_pos = [float(r[0]) for r in rows if r[1] == "false_positive"]
    n_labeled = len(confirmed) + len(false_pos)

    if n_labeled < NLI_FEEDBACK_MIN_LABELS:
        log.info(
            "nli_retune_skipped_too_few_labels",
            n_labeled=n_labeled,
            min_required=NLI_FEEDBACK_MIN_LABELS,
        )
        return NLI_DEFAULT_THRESHOLD

    all_scores = confirmed + false_pos
    all_labels = [1] * len(confirmed) + [0] * len(false_pos)

    best_t = NLI_DEFAULT_THRESHOLD
    best_f = 0.0
    beta = 0.5  # F0.5 — precision-weighted

    for t in sorted(set(all_scores)):
        tp = sum(1 for s, lbl in zip(all_scores, all_labels) if s >= t and lbl == 1)
        fp = sum(1 for s, lbl in zip(all_scores, all_labels) if s >= t and lbl == 0)
        if tp + fp == 0:
            continue
        precision = tp / (tp + fp)
        recall = tp / len(confirmed) if confirmed else 0.0
        if precision + recall == 0:
            continue
        f = (1 + beta**2) * precision * recall / (beta**2 * precision + recall)
        if f > best_f:
            best_f = f
            best_t = float(t)

    # Persist threshold into the latest calibration row's distributions JSON
    cal_row = conn.execute("SELECT rowid, distributions FROM calibration ORDER BY rowid DESC LIMIT 1").fetchone()
    if cal_row is not None:
        rowid, dist_json = cal_row
        dist = json.loads(dist_json) if dist_json else {}
        dist["nli_score_threshold"] = best_t
        dist["nli_retune_n_confirmed"] = len(confirmed)
        dist["nli_retune_n_false_pos"] = len(false_pos)
        conn.execute(
            "UPDATE calibration SET distributions = ? WHERE rowid = ?",
            [json.dumps(dist), rowid],
        )

    # Back-fill calibrated_confidence + calibration_state on NLI rows
    conn.execute(
        "UPDATE contradictions "
        "SET calibrated_confidence = raw_score, "
        "    calibration_state = CASE WHEN raw_score >= ? THEN 'calibrated' "
        "                            ELSE 'below_threshold' END "
        "WHERE detector = 'nli'",
        [best_t],
    )

    log.info(
        "nli_retune_complete",
        threshold=round(best_t, 4),
        n_confirmed=len(confirmed),
        n_false_pos=len(false_pos),
        f_score=round(best_f, 4),
    )
    return best_t


def load_nli_score_threshold(conn: duckdb.DuckDBPyConnection) -> float:
    """Return the calibrated NLI score threshold, or 0.5 if not yet tuned."""
    row = conn.execute("SELECT distributions FROM calibration ORDER BY rowid DESC LIMIT 1").fetchone()
    if row is None or row[0] is None:
        return NLI_DEFAULT_THRESHOLD
    dist = json.loads(row[0])
    return float(dist.get("nli_score_threshold", NLI_DEFAULT_THRESHOLD))


def load_latest_calibration(conn: duckdb.DuckDBPyConnection) -> Optional[dict]:
    """Load the most recent calibration thresholds from DuckDB"""
    row = conn.execute(
        "SELECT dedup_threshold, contradiction_candidate_threshold, cluster_min_density, distributions "
        "FROM calibration ORDER BY rowid DESC LIMIT 1"
    ).fetchone()
    if row is None:
        return None
    dist = json.loads(row[3]) if row[3] else {}
    return {
        "dedup_cosine_threshold": row[0],
        "contradiction_candidate_threshold": row[1],
        "cluster_min_density": row[2],
        "nli_score_threshold": float(dist.get("nli_score_threshold", NLI_DEFAULT_THRESHOLD)),
    }
