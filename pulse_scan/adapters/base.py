from datetime import datetime
from typing import Iterator, Optional, Protocol, runtime_checkable
import numpy as np

from pulse_scan.models import ChunkBatch, CollectionInfo


@runtime_checkable
class VectorStoreAdapter(Protocol):
    def list_collections(self) -> list[CollectionInfo]: ...

    def fetch_chunks(
        self,
        collection: str,
        batch_size: int,
        since: Optional[datetime] = None,
    ) -> Iterator[ChunkBatch]: ...

    def get_embeddings(
        self,
        collection: str,
        chunk_ids: list[str],
    ) -> np.ndarray: ...

    @property
    def supports_embeddings_in_fetch(self) -> bool: ...

    @property
    def supports_metadata_filtering(self) -> bool: ...
