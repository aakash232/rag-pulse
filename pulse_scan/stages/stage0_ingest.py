"""Stage 0: Ingestion + Cache.

Connects to the vector store adapter, fetches chunks, and populates or
refreshes the local DuckDB + numpy-memmap cache.
"""

import json
import time
import uuid
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Optional

import duckdb
import numpy as np
import structlog
import xxhash

from pulse_scan.adapters.base import VectorStoreAdapter
from pulse_scan.config import CollectionConfig, PulseConfig
from pulse_scan.models import ChunkRecord

log = structlog.get_logger()

# Embedding dimension → known model names (from LLD §3)
KNOWN_MODELS: dict[int, list[str]] = {
    1536: ["openai-ada-002", "openai-text-embedding-3-small"],
    3072: ["openai-text-embedding-3-large"],
    768: ["sentence-transformers/all-mpnet-base-v2", "BGE-base", "E5-base"],
    1024: ["BGE-large", "E5-large", "Voyage-2", "Cohere-embed-v3"],
    384: ["sentence-transformers/all-MiniLM-L6-v2"],
}

FETCH_BUDGET_SECS = 30 * 60  # 30-minute default (LLD §3 Stage 0)
MEMMAP_INITIAL_CAPACITY = 10_000
MEMMAP_GROWTH_FACTOR = 2


# ---------------------------------------------------------------------------
# Embedding store (numpy memmap with JSON metadata sidecar)
# ---------------------------------------------------------------------------

class EmbeddingStore:
    def __init__(self, data_dir: Path):
        self._arr_path = data_dir / "embeddings.f32.npy"
        self._meta_path = data_dir / "embeddings.meta.json"
        self._mmap: Optional[np.memmap] = None
        self._meta: Optional[dict] = None

    # -- meta helpers --------------------------------------------------------

    def _load_meta(self) -> Optional[dict]:
        if self._meta_path.exists():
            return json.loads(self._meta_path.read_text())
        return None

    def _save_meta(self) -> None:
        assert self._meta is not None
        self._meta_path.write_text(json.dumps(self._meta))

    # -- public interface ----------------------------------------------------

    @property
    def dim(self) -> Optional[int]:
        return self._meta["dim"] if self._meta else None

    @property
    def n_used(self) -> int:
        return self._meta["n_used"] if self._meta else 0

    def open(self) -> None:
        """Open existing store (no-op if store does not yet exist)."""
        self._meta = self._load_meta()
        if self._meta is not None:
            self._mmap = np.memmap(
                self._arr_path,
                dtype=np.float32,
                mode="r+",
                shape=(self._meta["n_allocated"], self._meta["dim"]),
            )

    def initialize_dim(self, dim: int) -> None:
        """Call once when the first embedding is seen in a scan."""
        if self._meta is None:
            capacity = MEMMAP_INITIAL_CAPACITY
            self._meta = {"dim": dim, "n_allocated": capacity, "n_used": 0}
            self._mmap = np.memmap(
                self._arr_path, dtype=np.float32, mode="w+",
                shape=(capacity, dim),
            )
            self._save_meta()
            log.info(
                "embedding_store_created",
                dim=dim,
                known_models=KNOWN_MODELS.get(dim, ["unknown"]),
            )
        elif self._meta["dim"] != dim:
            raise RuntimeError(
                f"Embedding dimension changed: stored={self._meta['dim']}, "
                f"received={dim}. This indicates an embedding model change. "
                "Explicit migration is required. See docs."
            )

    def _grow(self) -> None:
        old_used = self._meta["n_used"]
        old_data = np.array(self._mmap[:old_used])
        del self._mmap

        new_cap = max(
            self._meta["n_allocated"] * MEMMAP_GROWTH_FACTOR,
            old_used + 1_000,
        )
        self._mmap = np.memmap(
            self._arr_path, dtype=np.float32, mode="w+",
            shape=(new_cap, self._meta["dim"]),
        )
        if old_used > 0:
            self._mmap[:old_used] = old_data
        self._meta["n_allocated"] = new_cap
        self._save_meta()
        log.info("embedding_store_grown", new_capacity=new_cap)

    def append(self, embedding: np.ndarray) -> int:
        """Write embedding and return its row offset."""
        if self._meta is None:
            raise RuntimeError("Call initialize_dim before append.")
        if self._meta["n_used"] >= self._meta["n_allocated"]:
            self._grow()
        offset = self._meta["n_used"]
        self._mmap[offset] = embedding
        self._meta["n_used"] += 1
        return offset

    def close(self) -> None:
        if self._mmap is not None:
            self._mmap.flush()
            del self._mmap
            self._mmap = None
        if self._meta is not None:
            self._save_meta()


# ---------------------------------------------------------------------------
# Timestamp resolution
# ---------------------------------------------------------------------------

_UUID_EPOCH = datetime(1582, 10, 15, tzinfo=timezone.utc)
_CROCKFORD = "0123456789ABCDEFGHJKMNPQRSTVWXYZ"


def _try_uuid_v1(chunk_id: str) -> Optional[datetime]:
    try:
        uid = uuid.UUID(str(chunk_id))
        if uid.version == 1:
            ts_us = uid.time // 10  # 100ns → µs
            return _UUID_EPOCH + timedelta(microseconds=ts_us)
    except (ValueError, AttributeError):
        pass
    return None


def _try_ulid(chunk_id: str) -> Optional[datetime]:
    if len(chunk_id) != 26:
        return None
    try:
        ts_chars = chunk_id.upper()[:10]
        ts_ms = 0
        for ch in ts_chars:
            ts_ms = ts_ms * 32 + _CROCKFORD.index(ch)
        return datetime.fromtimestamp(ts_ms / 1000.0, tz=timezone.utc)
    except (ValueError, IndexError):
        return None


def _utc_naive(dt: datetime) -> datetime:
    """Convert tz-aware datetime to UTC naive (for DuckDB TIMESTAMP storage)."""
    if dt.tzinfo is not None:
        dt = dt.astimezone(timezone.utc).replace(tzinfo=None)
    return dt


def resolve_timestamp(
    chunk_id: str,
    metadata: dict,
    timestamp_field: Optional[str],
    fallback: datetime,
) -> tuple[datetime, str]:
    """Returns (resolved_timestamp_utc_naive, timestamp_source)."""
    # 1. User-declared field
    if timestamp_field:
        raw = metadata.get(timestamp_field)
        if raw:
            try:
                ts = datetime.fromisoformat(str(raw).replace("Z", "+00:00"))
                if ts.tzinfo is None:
                    ts = ts.replace(tzinfo=timezone.utc)
                return _utc_naive(ts), "metadata_field"
            except (ValueError, TypeError):
                pass

    # 2. UUIDv1
    ts = _try_uuid_v1(chunk_id)
    if ts is not None:
        return _utc_naive(ts), "uuid_v1"

    # 3. ULID
    ts = _try_ulid(chunk_id)
    if ts is not None:
        return _utc_naive(ts), "ulid"

    # 4. Fallback
    return _utc_naive(fallback), "first_seen"


# ---------------------------------------------------------------------------
# Stage 0 runner
# ---------------------------------------------------------------------------

def _content_hash(text: str) -> str:
    return xxhash.xxh64(text.encode()).hexdigest()


class IngestStage:
    def __init__(
        self,
        conn: duckdb.DuckDBPyConnection,
        adapter: VectorStoreAdapter,
        config: PulseConfig,
        data_dir: Path,
    ):
        self.conn = conn
        self.adapter = adapter
        self.config = config
        self.emb_store = EmbeddingStore(data_dir)

    def run(self, run_id: str) -> dict:
        store_id = getattr(self.adapter, "store_id", "unknown")
        started_at = _utc_naive(datetime.now(timezone.utc))
        deadline = time.monotonic() + FETCH_BUDGET_SECS

        stats = {
            "chunks_new": 0,
            "chunks_unchanged": 0,
            "chunks_updated": 0,
            "chunks_deleted": 0,
        }

        col_cfg_map = {c.name: c for c in self.config.collections}
        collections = self.adapter.list_collections()

        self.emb_store.open()
        seen_ids: dict[str, set[str]] = {}

        try:
            for cinfo in collections:
                col_name = cinfo.name
                col_cfg = col_cfg_map.get(col_name, CollectionConfig(name=col_name))
                seen_ids[col_name] = set()

                log.info("ingesting_collection", collection=col_name, store_id=store_id)

                batch_size = 500
                for batch in self.adapter.fetch_chunks(col_name, batch_size=batch_size):
                    if time.monotonic() > deadline:
                        log.warning("fetch_budget_exceeded", elapsed_secs=FETCH_BUDGET_SECS)
                        break

                    now = _utc_naive(datetime.now(timezone.utc))

                    # Initialize embedding dim from first embedding seen
                    if self.emb_store.dim is None:
                        for chunk in batch.chunks:
                            if chunk.embedding is not None:
                                self.emb_store.initialize_dim(len(chunk.embedding))
                                break

                    # Validate dim consistency
                    for chunk in batch.chunks:
                        if chunk.embedding is not None and self.emb_store.dim is not None:
                            if len(chunk.embedding) != self.emb_store.dim:
                                raise RuntimeError(
                                    f"Embedding dimension mismatch on chunk {chunk.chunk_id}: "
                                    f"expected {self.emb_store.dim}, got {len(chunk.embedding)}. "
                                    "Model change detected. Migration required."
                                )

                    # Bulk-look up existing chunks for this batch
                    chunk_ids = [c.chunk_id for c in batch.chunks]
                    rows = self.conn.execute(
                        "SELECT chunk_id, content_hash, version FROM chunks "
                        "WHERE store_id = ? AND collection = ? AND chunk_id IN "
                        f"({','.join('?' * len(chunk_ids))})",
                        [store_id, col_name] + chunk_ids,
                    ).fetchall()
                    existing: dict[str, tuple[str, int]] = {
                        row[0]: (row[1], row[2]) for row in rows
                    }

                    for chunk in batch.chunks:
                        seen_ids[col_name].add(chunk.chunk_id)
                        h = _content_hash(chunk.text)
                        resolved_ts, ts_src = resolve_timestamp(
                            chunk.chunk_id, chunk.metadata,
                            col_cfg.timestamp_field, now,
                        )

                        if chunk.chunk_id not in existing:
                            emb_offset = self._write_embedding(chunk)
                            self.conn.execute(
                                """INSERT INTO chunks (
                                    store_id, collection, chunk_id, text, content_hash,
                                    resolved_timestamp, timestamp_source, embedding_offset,
                                    first_seen_by_pulse, last_seen_by_pulse, version
                                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                                [store_id, col_name, chunk.chunk_id, chunk.text, h,
                                 resolved_ts, ts_src, emb_offset, now, now, 1],
                            )
                            stats["chunks_new"] += 1

                        else:
                            old_hash, old_version = existing[chunk.chunk_id]
                            if old_hash == h:
                                self.conn.execute(
                                    "UPDATE chunks SET last_seen_by_pulse = ? "
                                    "WHERE store_id = ? AND collection = ? AND chunk_id = ?",
                                    [now, store_id, col_name, chunk.chunk_id],
                                )
                                stats["chunks_unchanged"] += 1
                            else:
                                emb_offset = self._write_embedding(chunk)
                                self.conn.execute(
                                    """UPDATE chunks SET
                                        text = ?, content_hash = ?,
                                        resolved_timestamp = ?, timestamp_source = ?,
                                        embedding_offset = ?,
                                        last_seen_by_pulse = ?,
                                        version = ?,
                                        deleted_at = NULL
                                    WHERE store_id = ? AND collection = ? AND chunk_id = ?""",
                                    [chunk.text, h, resolved_ts, ts_src, emb_offset,
                                     now, old_version + 1,
                                     store_id, col_name, chunk.chunk_id],
                                )
                                stats["chunks_updated"] += 1

            # Mark absent chunks as deleted
            now = _utc_naive(datetime.now(timezone.utc))
            for col_name, present in seen_ids.items():
                if not present:
                    continue
                stored_rows = self.conn.execute(
                    "SELECT chunk_id FROM chunks "
                    "WHERE store_id = ? AND collection = ? AND deleted_at IS NULL",
                    [store_id, col_name],
                ).fetchall()
                for (cid,) in stored_rows:
                    if cid not in present:
                        self.conn.execute(
                            "UPDATE chunks SET deleted_at = ? "
                            "WHERE store_id = ? AND collection = ? AND chunk_id = ?",
                            [now, store_id, col_name, cid],
                        )
                        stats["chunks_deleted"] += 1

        finally:
            self.emb_store.close()

        finished_at = _utc_naive(datetime.now(timezone.utc))
        stats["elapsed_secs"] = (finished_at - started_at).total_seconds()

        self.conn.execute(
            "INSERT INTO scan_runs (run_id, started_at, finished_at, config, stats) "
            "VALUES (?, ?, ?, ?, ?)",
            [
                run_id,
                started_at,
                finished_at,
                json.dumps({"store_type": self.config.store.type}),
                json.dumps(stats),
            ],
        )
        log.info("stage0_complete", **stats)
        return stats

    def _write_embedding(self, chunk: ChunkRecord) -> int:
        if chunk.embedding is not None and self.emb_store.dim is not None:
            return self.emb_store.append(chunk.embedding)
        return -1
