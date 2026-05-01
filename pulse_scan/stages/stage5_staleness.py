"""Stage 5: Staleness Scoring.

Computes a continuous staleness score in [0, 1] for every active chunk by
combining four available signals:

  staleness = w_age  * age_decay
            + w_drift * cluster_drift
            + w_contra * contradiction_evidence
            + w_super  * supersession_evidence

(retrieval_abandonment is stubbed at 0.0 — retrieval logs are out of scope for v1)

Labels:  fresh < 0.3  |  aging 0.3–0.6  |  stale 0.6–0.85  |  abandoned > 0.85

Outputs written to chunks.staleness_score, chunks.staleness_label,
chunks.staleness_components (JSON for per-component explainability).
"""

import json
import math
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

import duckdb
import numpy as np
import structlog

log = structlog.get_logger()

# Component weights (initial heuristic; sum = 1.0).
# age and contradiction are the primary drivers; cluster_drift and supersession
# act as amplifiers when stale signals accumulate.
_W_AGE   = 0.35
_W_DRIFT = 0.20
_W_CONTRA = 0.30
_W_SUPER  = 0.15

DEFAULT_HALF_LIFE_DAYS = 90


# ---------------------------------------------------------------------------
# Component functions
# ---------------------------------------------------------------------------

def _age_decay(resolved_ts: Optional[datetime], now: datetime, half_life_days: int) -> float:
    """1 − 2^(−age / half_life).  Returns 0 if no timestamp available."""
    if resolved_ts is None:
        return 0.0
    age_days = max(0.0, (now - resolved_ts).total_seconds() / 86400.0)
    return 1.0 - 2.0 ** (-age_days / half_life_days)


def _cluster_drift(embedding: np.ndarray, cluster_id, centroids: dict) -> float:
    """1 − cosine(embedding, centroid).  Returns 0.0 for noise chunks."""
    if cluster_id is None or cluster_id not in centroids:
        return 0.0
    centroid = centroids[cluster_id]
    cosine = float(np.dot(embedding, centroid))
    return 1.0 - max(-1.0, min(1.0, cosine))


def _contradiction_evidence(chunk_id: str, counts: dict) -> float:
    """min(1, count / 3) — saturates at 3 unresolved contradictions."""
    n = counts.get(chunk_id, 0)
    return min(1.0, n / 3.0)


def _staleness_label(score: float) -> str:
    if score < 0.30:
        return "fresh"
    if score < 0.60:
        return "aging"
    if score < 0.85:
        return "stale"
    return "abandoned"


def _utc_naive_now() -> datetime:
    return datetime.now(timezone.utc).replace(tzinfo=None)


# ---------------------------------------------------------------------------
# Stage
# ---------------------------------------------------------------------------

class StalenessStage:
    def __init__(
        self,
        conn: duckdb.DuckDBPyConnection,
        data_dir: Path,
        collection_configs=None,       # list[CollectionConfig] or None
        reference_time: Optional[datetime] = None,  # inject for testing
    ):
        self.conn = conn
        self.data_dir = data_dir
        self._collection_configs = collection_configs or []
        self._now = reference_time or _utc_naive_now()

    def run(self) -> dict:
        rows = self.conn.execute(
            "SELECT chunk_id, collection, resolved_timestamp, cluster_id, embedding_offset "
            "FROM chunks "
            "WHERE deleted_at IS NULL AND embedding_offset >= 0 "
            "ORDER BY chunk_id"
        ).fetchall()

        if not rows:
            return {"chunks_scored": 0}

        meta = json.loads((self.data_dir / "embeddings.meta.json").read_text())
        dim: int = meta["dim"]
        n_allocated: int = meta["n_allocated"]

        mmap = np.memmap(
            self.data_dir / "embeddings.f32.npy",
            dtype=np.float32,
            mode="r",
            shape=(n_allocated, dim),
        )
        embeddings = np.array(mmap[[r[4] for r in rows]], dtype=np.float32)
        del mmap

        norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
        norms[norms == 0] = 1.0
        embeddings /= norms

        centroids = self._load_centroids()
        contra_counts = self._load_contradiction_counts()
        superseded = self._load_superseded_ids()
        half_lives = {c.name: c.half_life_days for c in self._collection_configs}

        n_scored = 0
        label_counts: dict[str, int] = {}

        for i, (chunk_id, collection, resolved_ts, cluster_id, _) in enumerate(rows):
            half_life = half_lives.get(collection, DEFAULT_HALF_LIFE_DAYS)

            ad = _age_decay(resolved_ts, self._now, half_life)
            cd = _cluster_drift(embeddings[i], cluster_id, centroids)
            ce = _contradiction_evidence(chunk_id, contra_counts)
            se = 1.0 if chunk_id in superseded else 0.0

            raw = _W_AGE * ad + _W_DRIFT * cd + _W_CONTRA * ce + _W_SUPER * se
            score = min(1.0, max(0.0, raw))
            label = _staleness_label(score)

            components = {
                "age_decay": round(ad, 4),
                "cluster_drift": round(cd, 4),
                "contradiction_evidence": round(ce, 4),
                "supersession_evidence": round(se, 4),
                "retrieval_abandonment": 0.0,  # v1 stub: no retrieval logs available
            }

            self.conn.execute(
                "UPDATE chunks "
                "SET staleness_score = ?, staleness_label = ?, staleness_components = ? "
                "WHERE chunk_id = ? AND deleted_at IS NULL",
                [score, label, json.dumps(components), chunk_id],
            )
            label_counts[label] = label_counts.get(label, 0) + 1
            n_scored += 1

        log.info("staleness_complete", chunks_scored=n_scored, **label_counts)
        return {"chunks_scored": n_scored, "label_counts": label_counts}

    # ------------------------------------------------------------------
    # Internal loaders
    # ------------------------------------------------------------------

    def _load_centroids(self) -> dict:
        rows = self.conn.execute(
            "SELECT cluster_id, centroid FROM cluster_centroids"
        ).fetchall()
        centroids: dict[int, np.ndarray] = {}
        for cluster_id, blob in rows:
            c = np.frombuffer(blob, dtype=np.float32).copy()
            n = float(np.linalg.norm(c))
            if n > 0:
                c /= n
            centroids[cluster_id] = c
        return centroids

    def _load_contradiction_counts(self) -> dict:
        rows = self.conn.execute(
            "SELECT chunk_id, COUNT(*) AS n "
            "FROM ("
            "  SELECT chunk_a AS chunk_id FROM contradictions WHERE user_resolution IS NULL "
            "  UNION ALL "
            "  SELECT chunk_b AS chunk_id FROM contradictions WHERE user_resolution IS NULL "
            ") GROUP BY chunk_id"
        ).fetchall()
        return {chunk_id: int(n) for chunk_id, n in rows}

    def _load_superseded_ids(self) -> set:
        rows = self.conn.execute(
            "SELECT canonical_chunk_id, member_chunk_ids FROM dedup_groups"
        ).fetchall()
        superseded: set[str] = set()
        for canonical, members_json in rows:
            for member in json.loads(members_json):
                if member != canonical:
                    superseded.add(member)
        return superseded
