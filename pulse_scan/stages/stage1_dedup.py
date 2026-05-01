"""Stage 1: Deduplication — embedding channel + text channel.

Embedding channel: builds an in-memory HNSW index, queries k=10 nearest
neighbors, groups pairs above the calibrated dedup_cosine_threshold via
union-find.

Text channel (TextDeduplicateStage): runs MinHash + LSH over 3-word shingles,
finds pairs with Jaccard similarity ≥ TEXT_JACCARD_THRESHOLD (default 0.8),
then merges/annotates the dedup_groups table produced by the embedding pass.
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


# ---------------------------------------------------------------------------
# Text-channel dedup (MinHash + LSH)
# ---------------------------------------------------------------------------

class TextDeduplicateStage:
    """Augments dedup_groups produced by the embedding pass using text similarity.

    Runs MinHash LSH over 3-word shingles. For each text-similar pair it:
    - adds 'text' to detection_channels when both members are already in the
      same embedding group;
    - adds the un-grouped member to an existing group when only one is grouped;
    - merges two different embedding groups when both members are already grouped;
    - creates a new text-only group when neither member is in any group.

    Must be called AFTER DeduplicateStage.run() because it reads and rewrites
    the dedup_groups table.
    """

    TEXT_JACCARD_THRESHOLD = 0.8
    NUM_PERM = 128
    SHINGLE_K = 3

    def __init__(
        self,
        conn: duckdb.DuckDBPyConnection,
        threshold: float | None = None,
    ):
        self.conn = conn
        self._threshold = threshold if threshold is not None else self.TEXT_JACCARD_THRESHOLD

    def run(self) -> dict:
        from datasketch import MinHash, MinHashLSH

        rows = self.conn.execute(
            "SELECT chunk_id, text, resolved_timestamp "
            "FROM chunks WHERE deleted_at IS NULL ORDER BY chunk_id"
        ).fetchall()

        if len(rows) < 2:
            return {"groups_added": 0, "channels_updated": 0, "total_groups": 0}

        chunk_ids = [r[0] for r in rows]
        texts = {r[0]: r[1] or "" for r in rows}
        timestamps = {r[0]: str(r[2]) if r[2] else "0000" for r in rows}
        text_lens = {r[0]: len(r[1] or "") for r in rows}

        # Build MinHash index
        lsh = MinHashLSH(threshold=self._threshold, num_perm=self.NUM_PERM)
        minhashes: dict[str, MinHash] = {}
        for cid in chunk_ids:
            m = MinHash(num_perm=self.NUM_PERM)
            for s in self._shingles(texts[cid]):
                m.update(s)
            minhashes[cid] = m
            lsh.insert(cid, m)

        # Find all candidate pairs
        text_pairs: set[frozenset] = set()
        for cid in chunk_ids:
            for other in lsh.query(minhashes[cid]):
                if other != cid:
                    text_pairs.add(frozenset([cid, other]))

        if not text_pairs:
            log.info("dedup_text_complete", pairs_found=0, groups_added=0, channels_updated=0)
            return {"groups_added": 0, "channels_updated": 0, "total_groups": 0}

        # Load existing embedding-channel groups
        group_rows = self.conn.execute(
            "SELECT group_id, canonical_chunk_id, member_chunk_ids, detection_channels "
            "FROM dedup_groups ORDER BY group_id"
        ).fetchall()

        chunk_to_group: dict[str, int] = {}
        groups: dict[int, dict] = {}

        for gid, canonical, members_json, channels_json in group_rows:
            members = set(json.loads(members_json))
            channels = set(json.loads(channels_json))
            groups[gid] = {"members": members, "channels": channels}
            for m in members:
                chunk_to_group[m] = gid

        next_gid = max(groups.keys(), default=-1) + 1
        groups_added = 0
        channels_updated = 0

        for pair in text_pairs:
            a, b = sorted(pair)
            ga = chunk_to_group.get(a)
            gb = chunk_to_group.get(b)

            if ga is not None and gb is not None and ga == gb:
                if "text" not in groups[ga]["channels"]:
                    groups[ga]["channels"].add("text")
                    channels_updated += 1

            elif ga is None and gb is None:
                gid = next_gid
                next_gid += 1
                groups[gid] = {"members": {a, b}, "channels": {"text"}}
                chunk_to_group[a] = gid
                chunk_to_group[b] = gid
                groups_added += 1

            elif ga is not None and gb is None:
                groups[ga]["members"].add(b)
                groups[ga]["channels"].add("text")
                chunk_to_group[b] = ga

            elif gb is not None and ga is None:
                groups[gb]["members"].add(a)
                groups[gb]["channels"].add("text")
                chunk_to_group[a] = gb

            else:
                # Merge two different groups: absorb smaller into larger
                keep = ga if len(groups[ga]["members"]) >= len(groups[gb]["members"]) else gb
                drop = gb if keep == ga else ga
                groups[keep]["members"] |= groups[drop]["members"]
                groups[keep]["channels"] |= groups[drop]["channels"]
                groups[keep]["channels"].add("text")
                for m in groups[drop]["members"]:
                    chunk_to_group[m] = keep
                del groups[drop]

        # Rewrite dedup_groups with final consolidated state
        self.conn.execute("DELETE FROM dedup_groups")
        for gid, info in sorted(groups.items()):
            canonical = max(
                info["members"],
                key=lambda x: (timestamps.get(x, "0000"), text_lens.get(x, 0)),
            )
            self.conn.execute(
                "INSERT INTO dedup_groups "
                "(group_id, canonical_chunk_id, member_chunk_ids, detection_channels) "
                "VALUES (?, ?, ?, ?)",
                [gid, canonical, json.dumps(sorted(info["members"])),
                 json.dumps(sorted(info["channels"]))],
            )

        total_groups = len(groups)
        log.info(
            "dedup_text_complete",
            pairs_found=len(text_pairs),
            groups_added=groups_added,
            channels_updated=channels_updated,
            total_groups=total_groups,
        )
        return {"groups_added": groups_added, "channels_updated": channels_updated,
                "total_groups": total_groups}

    def _shingles(self, text: str) -> list[bytes]:
        words = text.lower().split()
        k = self.SHINGLE_K
        if not words:
            return [b""]
        if len(words) < k:
            return [" ".join(words).encode("utf-8")]
        return [" ".join(words[i:i+k]).encode("utf-8") for i in range(len(words) - k + 1)]
