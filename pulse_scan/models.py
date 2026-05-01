from dataclasses import dataclass, field
from typing import Optional
import numpy as np


@dataclass
class CollectionInfo:
    name: str
    store_id: str
    count: Optional[int] = None
    metadata: dict = field(default_factory=dict)


@dataclass
class ChunkRecord:
    chunk_id: str
    text: str
    embedding: Optional[np.ndarray]
    metadata: dict = field(default_factory=dict)


@dataclass
class ChunkBatch:
    collection: str
    chunks: list[ChunkRecord]
