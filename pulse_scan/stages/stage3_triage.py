"""Stage 3: Triage with cost budget.

Scores every clustered chunk by priority and selects the top-N chunks whose
NLI candidate pairs fit within the cost_budget (1 NLI pair = 1 cost unit,
each chunk generates at most k pairs → N ≤ cost_budget / k).

Priority formula:
  priority = 0.5 * age_factor + 0.5 * cluster_factor
  age_factor    = 1 − 2^(−age_days / half_life)   [0, 1]
  cluster_factor = cluster_size / max_cluster_size  [0, 1]

Older chunks and members of larger clusters receive higher priority.
For cold start (first scan), there is no prior contradiction history to
incorporate — that signal is reserved for a future calibration loop.

Writes one row per clustered chunk to triage_log, with was_scanned=True
for chunks within the budget.  Returns the allowed set for NLIContradictionStage.
"""

import json
from datetime import datetime, timezone
from typing import Optional

import duckdb
import structlog

log = structlog.get_logger()

_W_AGE = 0.5
_W_CLUSTER = 0.5
DEFAULT_HALF_LIFE_DAYS = 90


def _age_factor(resolved_ts: Optional[datetime], now: datetime, half_life_days: int) -> float:
    """1 − 2^(−age / half_life).  Older → higher.  0 if no timestamp."""
    if resolved_ts is None:
        return 0.0
    age_days = max(0.0, (now - resolved_ts).total_seconds() / 86400.0)
    return 1.0 - 2.0 ** (-age_days / half_life_days)


def _utc_naive_now() -> datetime:
    return datetime.now(timezone.utc).replace(tzinfo=None)


class TriageStage:
    def __init__(
        self,
        conn: duckdb.DuckDBPyConnection,
        scan_run_id: str,
        scan_config=None,  # ScanConfig or None
        collection_configs=None,  # list[CollectionConfig] or None
        reference_time: Optional[datetime] = None,
    ):
        self.conn = conn
        self.scan_run_id = scan_run_id
        self._cost_budget = scan_config.cost_budget if scan_config else 50_000
        self._k = scan_config.contradiction_candidates_per_chunk if scan_config else 5
        self._collection_configs = collection_configs or []
        self._now = reference_time or _utc_naive_now()

    def run(self) -> tuple[set, dict]:
        """Score all clustered chunks, apply budget, write triage_log.

        Returns (allowed_chunk_ids, stats).
        """
        rows = self.conn.execute(
            "SELECT chunk_id, collection, resolved_timestamp, cluster_id "
            "FROM chunks "
            "WHERE deleted_at IS NULL AND cluster_id IS NOT NULL "
            "ORDER BY chunk_id"
        ).fetchall()

        if not rows:
            log.info("triage_no_clustered_chunks")
            return set(), {"chunks_scored": 0, "chunks_allowed": 0, "budget_used": 0}

        half_lives = {c.name: c.half_life_days for c in self._collection_configs}

        # cluster_id → n_chunks (from stored centroids)
        cluster_rows = self.conn.execute("SELECT cluster_id, n_chunks FROM cluster_centroids").fetchall()
        cluster_sizes: dict[int, int] = {cid: n for cid, n in cluster_rows}
        max_cluster_size = max(cluster_sizes.values(), default=1)

        # Score each chunk
        scored: list[tuple[str, float, float, float]] = []
        for chunk_id, collection, resolved_ts, cluster_id in rows:
            half_life = half_lives.get(collection, DEFAULT_HALF_LIFE_DAYS)
            age = _age_factor(resolved_ts, self._now, half_life)
            cluster_factor = cluster_sizes.get(cluster_id, 1) / max_cluster_size
            priority = _W_AGE * age + _W_CLUSTER * cluster_factor
            scored.append((chunk_id, priority, age, cluster_factor))

        scored.sort(key=lambda x: -x[1])

        # Budget: each allowed chunk queries at most k pairs
        k = max(self._k, 1)
        max_allowed = self._cost_budget // k
        allowed_set: set[str] = {cid for cid, _, _, _ in scored[:max_allowed]}

        # Idempotent: clear previous triage log for this run
        self.conn.execute("DELETE FROM triage_log WHERE scan_run_id = ?", [self.scan_run_id])

        for chunk_id, priority, age, cluster_factor in scored:
            components = {
                "age_factor": round(age, 4),
                "cluster_factor": round(cluster_factor, 4),
            }
            self.conn.execute(
                "INSERT INTO triage_log (scan_run_id, chunk_id, priority, components, was_scanned) "
                "VALUES (?, ?, ?, ?, ?)",
                [
                    self.scan_run_id,
                    chunk_id,
                    round(priority, 6),
                    json.dumps(components),
                    chunk_id in allowed_set,
                ],
            )

        budget_used = min(len(allowed_set) * k, self._cost_budget)
        log.info(
            "triage_complete",
            chunks_scored=len(scored),
            chunks_allowed=len(allowed_set),
            budget_used=budget_used,
        )
        return allowed_set, {
            "chunks_scored": len(scored),
            "chunks_allowed": len(allowed_set),
            "budget_used": budget_used,
        }
