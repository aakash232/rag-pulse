"""Chroma vector store adapter (v1 production target).

Connects to a Chroma server via the HTTP client, or to an in-process
EphemeralClient when a client is injected (used in tests).

Collection prefix:
  When collection_prefix is set, only Chroma collections whose names begin
  with that prefix are exposed.  The prefix is stripped from the collection
  name that pulse-scan stores internally, so a Chroma collection named
  "prod_docs" with prefix "prod_" appears as "docs" throughout the pipeline.

Embeddings:
  Chroma returns embeddings alongside documents in a single get() call, so
  supports_embeddings_in_fetch = True.  A separate get_embeddings() method
  is also implemented for targeted lookups.

Incremental scanning (since parameter):
  Chroma's metadata filtering supports comparison operators, but timestamp
  fields vary in format (ISO strings vs. epoch ints) and field name.
  For v1, since-filtering is not implemented; the adapter always returns all
  chunks and relies on the ingest stage's content-hash dedup to skip unchanged
  chunks efficiently.

Python 3.14 / protobuf compatibility:
  chromadb 1.x bundles opentelemetry protobuf stubs that conflict with the
  protobuf C extension on Python 3.14.  Set the env var
  PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION=python before importing chromadb to
  use the pure-Python implementation.  This is done automatically in the test
  suite via tests/conftest.py.
"""

from __future__ import annotations

from typing import Any, Iterator, Optional

import numpy as np

from pulse_scan.models import ChunkBatch, ChunkRecord, CollectionInfo


class ChromaAdapter:
    """Implements VectorStoreAdapter for Chroma (HTTP or in-process)."""

    def __init__(
        self,
        host: str = "localhost",
        port: int = 8000,
        collection_prefix: str = "",
        _client: Any = None,  # inject chromadb.EphemeralClient() for tests
    ):
        self.host = host
        self.port = port
        self.collection_prefix = collection_prefix
        self._injected_client = _client
        self._client_cache: Any = None

    @property
    def store_id(self) -> str:
        if self._injected_client is not None:
            return "chroma:ephemeral"
        return f"chroma:{self.host}:{self.port}"

    @property
    def supports_embeddings_in_fetch(self) -> bool:
        return True

    @property
    def supports_metadata_filtering(self) -> bool:
        return True

    # ------------------------------------------------------------------
    # Protocol implementation
    # ------------------------------------------------------------------

    def list_collections(self) -> list[CollectionInfo]:
        client = self._get_client()
        chroma_collections = client.list_collections()
        result = []
        for col in chroma_collections:
            if self.collection_prefix and not col.name.startswith(self.collection_prefix):
                continue
            name = self._strip_prefix(col.name)
            result.append(
                CollectionInfo(
                    name=name,
                    store_id=self.store_id,
                    count=col.count(),
                )
            )
        return sorted(result, key=lambda c: c.name)

    def fetch_chunks(
        self,
        collection: str,
        batch_size: int,
        since=None,  # not implemented for v1; always returns all chunks
    ) -> Iterator[ChunkBatch]:
        client = self._get_client()
        chroma_name = self._add_prefix(collection)
        col = client.get_collection(chroma_name)
        total = col.count()
        if total == 0:
            return

        offset = 0
        while offset < total:
            result = col.get(
                include=["documents", "embeddings", "metadatas"],
                limit=batch_size,
                offset=offset,
            )
            ids = result["ids"]
            if not ids:
                break

            chunks: list[ChunkRecord] = []
            docs = result.get("documents")
            if docs is None:
                docs = [None] * len(ids)
            embs = result.get("embeddings")
            if embs is None:
                embs = [None] * len(ids)
            metas = result.get("metadatas")
            if metas is None:
                metas = [{}] * len(ids)

            for i, chunk_id in enumerate(ids):
                raw_emb = embs[i]
                chunks.append(
                    ChunkRecord(
                        chunk_id=chunk_id,
                        text=docs[i] or "",
                        embedding=(
                            np.array(raw_emb, dtype=np.float32)
                            if raw_emb is not None
                            else None
                        ),
                        metadata=metas[i] or {},
                    )
                )

            yield ChunkBatch(collection=collection, chunks=chunks)
            offset += batch_size

    def get_embeddings(
        self,
        collection: str,
        chunk_ids: list[str],
    ) -> np.ndarray:
        client = self._get_client()
        chroma_name = self._add_prefix(collection)
        col = client.get_collection(chroma_name)
        result = col.get(ids=chunk_ids, include=["embeddings"])
        id_to_emb = {
            cid: emb
            for cid, emb in zip(result["ids"], result["embeddings"])
        }
        return np.array(
            [id_to_emb[cid] for cid in chunk_ids],
            dtype=np.float32,
        )

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _get_client(self) -> Any:
        if self._injected_client is not None:
            return self._injected_client
        if self._client_cache is None:
            import chromadb
            self._client_cache = chromadb.HttpClient(host=self.host, port=self.port)
        return self._client_cache

    def _strip_prefix(self, chroma_name: str) -> str:
        if self.collection_prefix and chroma_name.startswith(self.collection_prefix):
            return chroma_name[len(self.collection_prefix):]
        return chroma_name

    def _add_prefix(self, collection: str) -> str:
        return f"{self.collection_prefix}{collection}"
