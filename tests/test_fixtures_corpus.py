"""Smoke tests for the 50-chunk fixtures corpus."""

import json
from pathlib import Path

import numpy as np
import pytest

CORPUS_DIR = Path(__file__).parent.parent / "fixtures" / "corpus"
DIM = 384


def _load_all() -> dict[str, dict]:
    chunks = {}
    for f in CORPUS_DIR.glob("*.json"):
        for c in json.loads(f.read_text()):
            chunks[c["id"]] = c
    return chunks


def _cos(a: list, b: list) -> float:
    va, vb = np.array(a, dtype=np.float64), np.array(b, dtype=np.float64)
    return float(np.dot(va, vb) / (np.linalg.norm(va) * np.linalg.norm(vb)))


@pytest.fixture(scope="module")
def corpus() -> dict[str, dict]:
    return _load_all()


def test_total_chunk_count(corpus):
    assert len(corpus) == 50


def test_all_collections_present():
    collections = [f.stem for f in CORPUS_DIR.glob("*.json")]
    assert set(collections) == {"docs", "pricing", "api"}


def test_collection_sizes():
    sizes = {f.stem: len(json.loads(f.read_text())) for f in CORPUS_DIR.glob("*.json")}
    assert sizes["docs"] == 20
    assert sizes["pricing"] == 15
    assert sizes["api"] == 15


def test_all_embeddings_correct_dim(corpus):
    for chunk_id, chunk in corpus.items():
        assert len(chunk["embedding"]) == DIM, f"{chunk_id} has wrong dim"


def test_all_embeddings_are_unit_vectors(corpus):
    for chunk_id, chunk in corpus.items():
        norm = np.linalg.norm(chunk["embedding"])
        assert abs(norm - 1.0) < 1e-4, f"{chunk_id} is not a unit vector (norm={norm:.4f})"


def test_near_duplicate_pairs_have_high_cosine(corpus):
    """Lexical and semantic dups must be within dedup detection range."""
    dup_pairs = [
        ("docs-refund-001", "docs-refund-002"),  # lexical dup
        ("docs-refund-001", "docs-refund-003"),  # semantic dup
        ("docs-support-001", "docs-support-002"),  # lexical dup
    ]
    for a, b in dup_pairs:
        sim = _cos(corpus[a]["embedding"], corpus[b]["embedding"])
        assert sim > 0.85, f"Near-dup pair {a}↔{b} cosine={sim:.4f} (expected >0.85)"


def test_contradiction_pairs_have_detectable_cosine(corpus):
    """Contradicting chunks must be similar enough to pass the cosine pre-filter."""
    contradiction_pairs = [
        ("pricing-starter-001", "pricing-starter-002"),
        ("pricing-pro-001", "pricing-pro-002"),
        ("pricing-enterprise-001", "pricing-enterprise-002"),
        ("api-auth-003", "api-auth-004"),
        ("api-install-001", "api-install-002"),
        ("api-install-003", "api-install-004"),
        ("api-methods-001", "api-methods-002"),
    ]
    for a, b in contradiction_pairs:
        sim = _cos(corpus[a]["embedding"], corpus[b]["embedding"])
        assert sim > 0.80, f"Contradiction pair {a}↔{b} cosine={sim:.4f} (expected >0.80)"


def test_orphan_chunks_differ_from_main_clusters(corpus):
    """Orphan chunks should have low cosine with refund/support/pricing topics."""
    orphans = [corpus[f"docs-orphan-{i:03d}"]["embedding"] for i in range(1, 7)]
    ref = corpus["docs-refund-001"]["embedding"]
    for i, o in enumerate(orphans, 1):
        sim = _cos(ref, o)
        assert sim < 0.5, f"orphan-{i:03d} is too similar to refund cluster (cos={sim:.4f})"


def test_time_bounded_content_present(corpus):
    promo_ids = [k for k in corpus if k.startswith("docs-promo-")]
    assert len(promo_ids) == 3
    for pid in promo_ids:
        text = corpus[pid]["text"].lower()
        # Should contain a year reference (time-bounded indicator)
        assert any(y in text for y in ["2022", "2023", "2024"]), f"{pid} has no year"


def test_all_chunks_have_metadata_and_source(corpus):
    for chunk_id, chunk in corpus.items():
        assert "metadata" in chunk, f"{chunk_id} missing metadata"
        assert "source" in chunk["metadata"], f"{chunk_id} missing source"
        assert "created_at" in chunk["metadata"], f"{chunk_id} missing created_at"
