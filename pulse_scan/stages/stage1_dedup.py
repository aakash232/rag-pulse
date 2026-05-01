"""Stage 1: Deduplication — embedding channel.

Builds an in-memory HNSW index over all active chunk embeddings, queries
each chunk for k=10 nearest neighbors, and groups pairs with cosine
similarity above the dedup_cosine_threshold (from calibration) via
union-find.

Text-channel dedup (MinHash + LSH) is added in Step 9.
"""

import json
from pathlib import Path
from typing import Optional

import duckdb
import hnswlib
import numpy as np
import structlog

log = structlog.get_logger()

K_NEIGHBORS = 10


# ---------------------------------------------------------------------------
# Union-Find (integer keys, 0..N-1)
# ---------------------------------------------------------------------------

class UnionFind:
    def __init__(self, n: int):
        self._parent = list(range(n))

    def find(self, x: int) -> int:
        if self._parent[x] != x:
            self._parent[x] = self.find(self._parent[x])  # path compression
        return self._parent[x]

    def union(self, x: int, y: int) -> None:
        px, py = self.find(x), self.find(y)
        if px != py:
            self._parent[px] = py

    def groups(self) -> dict[int, list[int]]:
        """Return {root_idx → [member_idxs]} for all connected components."""
        result: dict[int, list[int]] = {}
        for x in range(len(self._parent)):
            result.setdefault(self.find(x), []).append(x)
        return result


# ---------------------------------------------------------------------------
# Stage runner
# ---------------------------------------------------------------------------

class DeduplicateStage:
    def __init__(
        self,
        conn: duckdb.DuckDBPyConnection,
        data_dir: Path,
        threshold: Optional[float] = None,  # None = read from calibration table
    ):
        self.conn = conn
        self.data_dir = data_dir
        self._override_threshold = threshold

    def run(self) -> dict:
        threshold = self._get_threshold()

        rows = self.conn.execute(
            "SELECT chunk_id, embedding_offset, resolved_timestamp, text "
            "FROM chunks "
            "WHERE deleted_at IS NULL AND embedding_offset >= 0 "
            "ORDER BY chunk_id"  # stable ordering for reproducibility
        ).fetchall()

        if len(rows) < 2:
            log.info("dedup_skipped_too_few_chunks", n_chunks=len(rows))
            return {"groups_found": 0, "chunks_in_groups": 0}

        meta_path = self.data_dir / "embeddings.meta.json"
        meta = json.loads(meta_path.read_text())
        dim: int = meta["dim"]
        n_allocated: int = meta["n_allocated"]

        chunk_ids: list[str] = []
        chunk_meta: list[tuple] = []  # (resolved_timestamp_str, text_len)
        offsets: list[int] = []

        for chunk_id, offset, ts, text in rows:
            chunk_ids.append(chunk_id)
            chunk_meta.append((str(ts) if ts is not None else "0000-01-01", len(text or "")))
            offsets.append(offset)

        N = len(rows)

        # Load and normalize embeddings
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

        # Build HNSW index and query neighbors
        k = min(K_NEIGHBORS + 1, N)  # +1 to absorb the self-match
        index = hnswlib.Index(space="cosine", dim=dim)
        index.init_index(max_elements=N, ef_construction=200, M=16)
        index.set_ef(max(50, k * 2))
        index.add_items(embeddings, np.arange(N))

        labels, distances = index.knn_query(embeddings, k=k)
        # hnswlib cosine space: distance = 1 - cosine_similarity

        # Build candidate pairs and union-find
        uf = UnionFind(N)
        for i in range(N):
            for j_pos in range(labels.shape[1]):
                j = int(labels[i, j_pos])
                if j == i:
                    continue  # self-match
                cosine = 1.0 - float(distances[i, j_pos])
                if cosine > threshold:
                    uf.union(i, j)

        # Resolve groups — only multi-member groups are dedup groups
        raw_groups = uf.groups()
        dedup_groups = {
            root: members
            for root, members in raw_groups.items()
            if len(members) > 1
        }

        # Clear previous results (idempotent)
        self.conn.execute("DELETE FROM dedup_groups")

        groups_found = 0
        chunks_in_groups = 0

        for group_id, (_, members) in enumerate(dedup_groups.items()):
            # Canonical: newest timestamp, longest text as tiebreaker
            canonical_idx = max(
                members,
                key=lambda i: (chunk_meta[i][0], chunk_meta[i][1]),
            )
            canonical_chunk_id = chunk_ids[canonical_idx]
            member_chunk_ids = [chunk_ids[i] for i in members]

            self.conn.execute(
                "INSERT INTO dedup_groups "
                "(group_id, canonical_chunk_id, member_chunk_ids, detection_channels) "
                "VALUES (?, ?, ?, ?)",
                [
                    group_id,
                    canonical_chunk_id,
                    json.dumps(member_chunk_ids),
                    json.dumps(["embedding"]),
                ],
            )
            groups_found += 1
            chunks_in_groups += len(members)

        log.info(
            "dedup_embedding_complete",
            n_chunks=N,
            threshold=round(threshold, 4),
            groups_found=groups_found,
            chunks_in_groups=chunks_in_groups,
        )
        return {"groups_found": groups_found, "chunks_in_groups": chunks_in_groups}

    def _get_threshold(self) -> float:
        if self._override_threshold is not None:
            return self._override_threshold
        row = self.conn.execute(
            "SELECT dedup_threshold FROM calibration ORDER BY rowid DESC LIMIT 1"
        ).fetchone()
        if row is None:
            raise RuntimeError(
                "No calibration found. Run 'pulse scan' or 'pulse calibrate' first."
            )
        return float(row[0])
