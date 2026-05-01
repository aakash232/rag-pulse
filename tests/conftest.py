import json
import os
import pytest
import numpy as np
from pathlib import Path


# Disable GPU check for all tests
os.environ.setdefault("PULSE_SKIP_GPU_CHECK", "1")


DIM = 4  # small embedding dimension for tests


def make_chunk(i: int, text: str | None = None, embedding: list | None = None) -> dict:
    return {
        "id": f"chunk-{i:03d}",
        "text": text if text is not None else f"This is chunk number {i}.",
        "embedding": embedding if embedding is not None else [float(i)] * DIM,
        "metadata": {"created_at": f"2024-01-{(i % 28) + 1:02d}T00:00:00Z"},
    }


@pytest.fixture()
def corpus_dir(tmp_path) -> Path:
    """5-chunk fixture corpus with a single 'docs' collection."""
    d = tmp_path / "corpus"
    d.mkdir()
    chunks = [make_chunk(i) for i in range(5)]
    (d / "docs.json").write_text(json.dumps(chunks))
    return d


@pytest.fixture()
def data_dir(tmp_path) -> Path:
    d = tmp_path / "data"
    d.mkdir()
    return d
