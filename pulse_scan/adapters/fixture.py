import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterator, Optional

import numpy as np

from pulse_scan.adapters.base import VectorStoreAdapter
from pulse_scan.models import ChunkBatch, ChunkRecord, CollectionInfo


class LocalFixtureAdapter(VectorStoreAdapter):
    """Reads chunks from a directory of per-collection JSON files.

    Each file is named <collection>.json and contains an array of objects:
      [{"id": "...", "text": "...", "embedding": [...], "metadata": {...}}, ...]
    """

    def __init__(self, corpus_dir: str | Path):
        self.corpus_dir = Path(corpus_dir)
        self.store_id = f"fixture:{self.corpus_dir.resolve()}"

    @property
    def supports_embeddings_in_fetch(self) -> bool:
        return True

    @property
    def supports_metadata_filtering(self) -> bool:
        return False

    def list_collections(self) -> list[CollectionInfo]:
        collections = []
        for json_file in sorted(self.corpus_dir.glob("*.json")):
            data = json.loads(json_file.read_text())
            collections.append(
                CollectionInfo(
                    name=json_file.stem,
                    store_id=self.store_id,
                    count=len(data),
                )
            )
        return collections

    def fetch_chunks(
        self,
        collection: str,
        batch_size: int,
        since: Optional[datetime] = None,
    ) -> Iterator[ChunkBatch]:
        json_file = self.corpus_dir / f"{collection}.json"
        if not json_file.exists():
            raise FileNotFoundError(f"Collection file not found: {json_file}")

        data = json.loads(json_file.read_text())

        if since is not None:
            filtered = []
            for item in data:
                raw_ts = item.get("metadata", {}).get("created_at")
                if raw_ts:
                    try:
                        item_ts = datetime.fromisoformat(raw_ts.replace("Z", "+00:00"))
                        if item_ts.replace(tzinfo=timezone.utc) < since.replace(tzinfo=timezone.utc):
                            continue
                    except (ValueError, TypeError):
                        pass
                filtered.append(item)
            data = filtered

        batch: list[ChunkRecord] = []
        for item in data:
            raw_emb = item.get("embedding")
            chunk = ChunkRecord(
                chunk_id=item["id"],
                text=item["text"],
                embedding=np.array(raw_emb, dtype=np.float32) if raw_emb is not None else None,
                metadata=item.get("metadata", {}),
            )
            batch.append(chunk)
            if len(batch) >= batch_size:
                yield ChunkBatch(collection=collection, chunks=batch)
                batch = []
        if batch:
            yield ChunkBatch(collection=collection, chunks=batch)

    def get_embeddings(
        self,
        collection: str,
        chunk_ids: list[str],
    ) -> np.ndarray:
        json_file = self.corpus_dir / f"{collection}.json"
        data = json.loads(json_file.read_text())
        id_to_item = {item["id"]: item for item in data}
        return np.array(
            [id_to_item[cid]["embedding"] for cid in chunk_ids],
            dtype=np.float32,
        )
