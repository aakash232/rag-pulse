"""Tests for ChromaAdapter using an injected EphemeralClient (no server needed)."""

import numpy as np
import pytest

chromadb = pytest.importorskip("chromadb", reason="chromadb not installed; pip install pulse-scan[chroma]")

from pulse_scan.adapters.chroma import ChromaAdapter  # noqa: E402

DIM = 4


@pytest.fixture()
def chroma_client(tmp_path):
    """Truly isolated chroma client backed by a per-test temp directory."""
    client = chromadb.PersistentClient(path=str(tmp_path / "chroma"))
    yield client


def _add_chunks(col, n: int, dim: int = DIM, id_prefix: str = "chunk") -> list[str]:
    ids = [f"{id_prefix}-{i:03d}" for i in range(n)]
    col.add(
        ids=ids,
        documents=[f"text for chunk {i}" for i in range(n)],
        embeddings=[[float(i)] * dim for i in range(n)],
        metadatas=[{"source": f"doc-{i}"} for i in range(n)],
    )
    return ids


# ---------------------------------------------------------------------------
# store_id / properties
# ---------------------------------------------------------------------------


def test_store_id_ephemeral(chroma_client):
    adapter = ChromaAdapter(_client=chroma_client)
    assert adapter.store_id == "chroma:ephemeral"


def test_store_id_http():
    adapter = ChromaAdapter(host="myhost", port=9999)
    assert adapter.store_id == "chroma:myhost:9999"


def test_store_id_http_default():
    adapter = ChromaAdapter()
    assert adapter.store_id == "chroma:localhost:8000"


def test_supports_embeddings_in_fetch(chroma_client):
    adapter = ChromaAdapter(_client=chroma_client)
    assert adapter.supports_embeddings_in_fetch is True


def test_supports_metadata_filtering(chroma_client):
    adapter = ChromaAdapter(_client=chroma_client)
    assert adapter.supports_metadata_filtering is True


# ---------------------------------------------------------------------------
# list_collections
# ---------------------------------------------------------------------------


def test_list_collections_empty(chroma_client):
    adapter = ChromaAdapter(_client=chroma_client)
    assert adapter.list_collections() == []


def test_list_collections_single_collection(chroma_client):
    col = chroma_client.create_collection("widgets")
    _add_chunks(col, 3)

    adapter = ChromaAdapter(_client=chroma_client)
    result = adapter.list_collections()
    assert len(result) == 1
    assert result[0].name == "widgets"
    assert result[0].count == 3
    assert result[0].store_id == "chroma:ephemeral"


def test_list_collections_sorted_alphabetically(chroma_client):
    chroma_client.create_collection("zebra")
    chroma_client.create_collection("apple")
    chroma_client.create_collection("mango")

    adapter = ChromaAdapter(_client=chroma_client)
    names = [c.name for c in adapter.list_collections()]
    assert names == ["apple", "mango", "zebra"]


def test_list_collections_prefix_filter_excludes_non_matching(chroma_client):
    chroma_client.create_collection("prod_docs")
    chroma_client.create_collection("prod_faq")
    chroma_client.create_collection("staging_docs")

    adapter = ChromaAdapter(_client=chroma_client, collection_prefix="prod_")
    result = adapter.list_collections()
    assert len(result) == 2
    names = {c.name for c in result}
    assert names == {"docs", "faq"}


def test_list_collections_prefix_stripped_from_exposed_name(chroma_client):
    col = chroma_client.create_collection("myprefix_myname")
    _add_chunks(col, 2)

    adapter = ChromaAdapter(_client=chroma_client, collection_prefix="myprefix_")
    result = adapter.list_collections()
    assert len(result) == 1
    assert result[0].name == "myname"


def test_list_collections_no_prefix_all_visible(chroma_client):
    chroma_client.create_collection("alpha")
    chroma_client.create_collection("beta")

    adapter = ChromaAdapter(_client=chroma_client)
    assert len(adapter.list_collections()) == 2


# ---------------------------------------------------------------------------
# fetch_chunks
# ---------------------------------------------------------------------------


def test_fetch_chunks_basic(chroma_client):
    col = chroma_client.create_collection("docs")
    _add_chunks(col, 5)

    adapter = ChromaAdapter(_client=chroma_client)
    batches = list(adapter.fetch_chunks("docs", batch_size=10))
    assert len(batches) == 1
    assert len(batches[0].chunks) == 5
    assert batches[0].collection == "docs"


def test_fetch_chunks_pagination_multiple_batches(chroma_client):
    col = chroma_client.create_collection("docs")
    _add_chunks(col, 7)

    adapter = ChromaAdapter(_client=chroma_client)
    batches = list(adapter.fetch_chunks("docs", batch_size=3))
    assert len(batches) == 3
    total = sum(len(b.chunks) for b in batches)
    assert total == 7


def test_fetch_chunks_pagination_all_ids_returned(chroma_client):
    col = chroma_client.create_collection("docs")
    inserted_ids = set(_add_chunks(col, 10))

    adapter = ChromaAdapter(_client=chroma_client)
    batches = list(adapter.fetch_chunks("docs", batch_size=4))
    returned_ids = {c.chunk_id for b in batches for c in b.chunks}
    assert returned_ids == inserted_ids


def test_fetch_chunks_exact_batch_boundary(chroma_client):
    col = chroma_client.create_collection("docs")
    _add_chunks(col, 6)

    adapter = ChromaAdapter(_client=chroma_client)
    batches = list(adapter.fetch_chunks("docs", batch_size=3))
    assert len(batches) == 2
    assert all(len(b.chunks) == 3 for b in batches)


def test_fetch_chunks_embeddings_are_float32(chroma_client):
    col = chroma_client.create_collection("docs")
    _add_chunks(col, 2)

    adapter = ChromaAdapter(_client=chroma_client)
    batches = list(adapter.fetch_chunks("docs", batch_size=10))
    for chunk in batches[0].chunks:
        assert chunk.embedding is not None
        assert chunk.embedding.dtype == np.float32


def test_fetch_chunks_empty_collection_yields_nothing(chroma_client):
    chroma_client.create_collection("empty")

    adapter = ChromaAdapter(_client=chroma_client)
    assert list(adapter.fetch_chunks("empty", batch_size=10)) == []


def test_fetch_chunks_with_prefix(chroma_client):
    col = chroma_client.create_collection("prod_docs")
    _add_chunks(col, 3)

    adapter = ChromaAdapter(_client=chroma_client, collection_prefix="prod_")
    batches = list(adapter.fetch_chunks("docs", batch_size=10))
    assert len(batches) == 1
    assert len(batches[0].chunks) == 3


def test_fetch_chunks_chunk_fields(chroma_client):
    col = chroma_client.create_collection("docs")
    col.add(
        ids=["id-001"],
        documents=["Hello world"],
        embeddings=[[1.0, 2.0, 3.0, 4.0]],
        metadatas=[{"source": "test-doc"}],
    )

    adapter = ChromaAdapter(_client=chroma_client)
    batches = list(adapter.fetch_chunks("docs", batch_size=10))
    chunk = batches[0].chunks[0]
    assert chunk.chunk_id == "id-001"
    assert chunk.text == "Hello world"
    assert chunk.metadata == {"source": "test-doc"}
    assert list(chunk.embedding) == pytest.approx([1.0, 2.0, 3.0, 4.0])


def test_fetch_chunks_since_ignored_returns_all(chroma_client):
    """since= parameter is not implemented in v1; all chunks returned regardless."""
    col = chroma_client.create_collection("docs")
    _add_chunks(col, 5)

    adapter = ChromaAdapter(_client=chroma_client)
    batches_all = list(adapter.fetch_chunks("docs", batch_size=10, since=None))
    batches_since = list(adapter.fetch_chunks("docs", batch_size=10, since="2024-01-01"))
    total_all = sum(len(b.chunks) for b in batches_all)
    total_since = sum(len(b.chunks) for b in batches_since)
    assert total_all == total_since == 5


# ---------------------------------------------------------------------------
# get_embeddings
# ---------------------------------------------------------------------------


def test_get_embeddings_order_preserved(chroma_client):
    col = chroma_client.create_collection("docs")
    col.add(
        ids=["a", "b", "c"],
        documents=["x", "y", "z"],
        embeddings=[[1.0, 0.0], [2.0, 0.0], [3.0, 0.0]],
        metadatas=[{"k": "a"}, {"k": "b"}, {"k": "c"}],
    )

    adapter = ChromaAdapter(_client=chroma_client)
    result = adapter.get_embeddings("docs", ["c", "a", "b"])
    assert result.shape == (3, 2)
    assert result[0, 0] == pytest.approx(3.0)  # c
    assert result[1, 0] == pytest.approx(1.0)  # a
    assert result[2, 0] == pytest.approx(2.0)  # b


def test_get_embeddings_dtype_float32(chroma_client):
    col = chroma_client.create_collection("docs")
    col.add(ids=["x"], documents=["t"], embeddings=[[0.5, 1.5]], metadatas=[{"k": "v"}])

    adapter = ChromaAdapter(_client=chroma_client)
    result = adapter.get_embeddings("docs", ["x"])
    assert result.dtype == np.float32


def test_get_embeddings_with_prefix(chroma_client):
    col = chroma_client.create_collection("prod_items")
    col.add(ids=["i1"], documents=["text"], embeddings=[[9.0, 0.0]], metadatas=[{"k": "v"}])

    adapter = ChromaAdapter(_client=chroma_client, collection_prefix="prod_")
    result = adapter.get_embeddings("items", ["i1"])
    assert result[0, 0] == pytest.approx(9.0)
