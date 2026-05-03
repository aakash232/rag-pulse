#!/usr/bin/env python3
"""
Ingest PDFs from ./pdfs/ into a local ChromaDB instance using OpenAI embeddings.

Steps:
    1. Add OPENAI_API_KEY to local_test/.env
    2. Drop one or more PDFs into local_test/pdfs/
    3. Run:  uv run python local_test/ingest.py
    4. Start ChromaDB:  uv run chroma run --path local_test/chroma_data --port 8010
    5. Run scanner:     uv run pulse scan --config local_test/pulse.config.yaml
"""

import hashlib
import os
import re
import sys
import time
import unicodedata
from datetime import datetime, timedelta, timezone
from pathlib import Path

import chromadb
import pypdf
from dotenv import load_dotenv
from openai import OpenAI

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

SCRIPT_DIR = Path(__file__).parent
PDF_DIR = SCRIPT_DIR / "pdfs"
CHROMA_DIR = SCRIPT_DIR / "chroma_data"
CONFIG_OUT = SCRIPT_DIR / "pulse.config.yaml"

EMBED_MODEL = "text-embedding-3-small"
EMBED_BATCH = 100
CHUNK_MAX_CHARS = 800
CHUNK_OVERLAP = 100
HALF_LIFE_DAYS = 90

# ---------------------------------------------------------------------------
# Text extraction + cleanup + chunking
# ---------------------------------------------------------------------------


def clean_text(text: str) -> str:
    """Normalize PDF text without destroying useful Unicode."""
    text = unicodedata.normalize("NFKC", text)
    text = text.replace("\x00", "")
    text = re.sub(r"[\u200b\u200c\u200d\ufeff]", "", text)
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def extract_pages(pdf_path: Path) -> list[tuple[int, str]]:
    reader = pypdf.PdfReader(str(pdf_path))
    pages: list[tuple[int, str]] = []

    for i, page in enumerate(reader.pages, start=1):
        text = clean_text(page.extract_text() or "")
        if text:
            pages.append((i, text))

    return pages


def chunk_text(text: str) -> list[str]:
    """Chunk text with overlap, avoiding infinite loops at the end."""
    if len(text) <= CHUNK_MAX_CHARS:
        return [text]

    chunks: list[str] = []
    start = 0

    while start < len(text):
        end = min(start + CHUNK_MAX_CHARS, len(text))

        if end < len(text):
            boundary = text.rfind(". ", max(start, end - 120), end)
            if boundary == -1:
                boundary = text.rfind(" ", max(start, end - 60), end)
            if boundary != -1 and boundary > start:
                end = boundary + 1

        chunk = text[start:end].strip()
        if chunk:
            chunks.append(chunk)

        if end >= len(text):
            break

        next_start = end - CHUNK_OVERLAP
        if next_start <= start:
            next_start = end

        start = next_start

    return chunks


# ---------------------------------------------------------------------------
# Timestamps
# ---------------------------------------------------------------------------


def _timestamps_for_chunks(n: int) -> list[str]:
    now = datetime.now(timezone.utc)
    result: list[str] = []

    for i in range(n):
        if i % 5 == 0:
            days_ago = (i % 30) + 1
        else:
            days_ago = int(730 * (1 - i / max(n, 1))) + (i % 14)

        result.append((now - timedelta(days=days_ago)).strftime("%Y-%m-%dT%H:%M:%SZ"))

    return result


# ---------------------------------------------------------------------------
# OpenAI embeddings
# ---------------------------------------------------------------------------


def embed(client: OpenAI, texts: list[str]) -> list[list[float]]:
    all_vecs: list[list[float]] = []

    for i in range(0, len(texts), EMBED_BATCH):
        batch = texts[i : i + EMBED_BATCH]
        batch = [t.replace("\n", " ") for t in batch]

        resp = client.embeddings.create(
            model=EMBED_MODEL,
            input=batch,
        )

        all_vecs.extend(item.embedding for item in resp.data)

        if i + EMBED_BATCH < len(texts):
            time.sleep(0.2)

    return all_vecs


# ---------------------------------------------------------------------------
# Chroma helpers
# ---------------------------------------------------------------------------


def collection_name(pdf_path: Path) -> str:
    """Create a safe Chroma collection name from PDF filename."""
    stem = pdf_path.stem.lower()
    stem = unicodedata.normalize("NFKC", stem)
    stem = re.sub(r"[^a-z0-9_-]", "-", stem)
    stem = re.sub(r"-{2,}", "-", stem)
    stem = stem.strip("-_")
    stem = stem[:60]

    if len(stem) < 3:
        stem = f"doc-{stem or 'pdf'}"

    stem = re.sub(r"^[^a-z0-9]+", "", stem)
    stem = re.sub(r"[^a-z0-9]+$", "", stem)

    return stem if len(stem) >= 3 else "documents"


def chunk_id(col: str, page: int, idx: int, text: str) -> str:
    digest = hashlib.sha1(text.encode("utf-8")).hexdigest()[:8]
    return f"{col}-p{page:03d}-c{idx:03d}-{digest}"


def load_into_chroma(
    chroma: chromadb.ClientAPI,
    openai_client: OpenAI,
    col_name: str,
    records: list[dict],
) -> int:
    try:
        chroma.delete_collection(col_name)
    except Exception:
        pass

    collection = chroma.get_or_create_collection(col_name)

    texts = [r["text"] for r in records]
    ids = [r["id"] for r in records]
    metadatas = [r["metadata"] for r in records]

    print(f"  embedding {len(texts)} chunks via OpenAI ...", end="", flush=True)
    embeddings = embed(openai_client, texts)
    print(" done")

    for i in range(0, len(records), 100):
        collection.upsert(
            ids=ids[i : i + 100],
            documents=texts[i : i + 100],
            embeddings=embeddings[i : i + 100],
            metadatas=metadatas[i : i + 100],
        )

    return len(records)


# ---------------------------------------------------------------------------
# Config generation
# ---------------------------------------------------------------------------


def write_config(collection_names: list[str]) -> None:
    col_lines = "\n".join(
        f"  - name: {name}\n    timestamp_field: created_at\n    half_life_days: {HALF_LIFE_DAYS}"
        for name in collection_names
    )

    CONFIG_OUT.write_text(
        f"""\
# Auto-generated by local_test/ingest.py
# WARNING: chunk text is stored in plain form. Do not use on PII data.

store:
  type: chroma
  connection:
    host: localhost
    port: 8010

collections:
{col_lines}

scan:
  cost_budget: 100000
  nli_batch_size: 16
  contradiction_candidates_per_chunk: 5
  enable_numeric_detector: true
  enable_version_detector: true

clustering:
  min_cluster_size: auto
  use_gpu: false

inference:
  nli_model: cross-encoder/nli-deberta-v3-base
  device: mps

dashboard:
  host: 127.0.0.1
  port: 8501
""",
        encoding="utf-8",
    )

    print(f"  config -> {CONFIG_OUT}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> None:
    load_dotenv(SCRIPT_DIR / ".env")

    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        print("ERROR: OPENAI_API_KEY not found in local_test/.env")
        sys.exit(1)

    PDF_DIR.mkdir(parents=True, exist_ok=True)

    pdfs = sorted(PDF_DIR.glob("*.pdf"))
    if not pdfs:
        print(f"No PDFs found in {PDF_DIR}/ — drop PDFs there and re-run.")
        sys.exit(0)

    openai_client = OpenAI(api_key=api_key)

    CHROMA_DIR.mkdir(parents=True, exist_ok=True)
    chroma_client = chromadb.PersistentClient(path=str(CHROMA_DIR))

    col_names: list[str] = []

    for pdf_path in pdfs:
        col = collection_name(pdf_path)
        print(f"\n{pdf_path.name} -> '{col}'")

        pages = extract_pages(pdf_path)
        if not pages:
            print("  no extractable text, skipping")
            continue

        records: list[dict] = []

        for page_num, page_text in pages:
            for chunk_idx, chunk in enumerate(chunk_text(page_text)):
                records.append(
                    {
                        "id": chunk_id(col, page_num, chunk_idx, chunk),
                        "text": chunk,
                        "metadata": {
                            "source": pdf_path.name,
                            "page": page_num,
                            "chunk_idx": chunk_idx,
                            "created_at": "",
                        },
                    }
                )

        if not records:
            print("  no chunks produced, skipping")
            continue

        for record, ts in zip(records, _timestamps_for_chunks(len(records))):
            record["metadata"]["created_at"] = ts

        n = load_into_chroma(chroma_client, openai_client, col, records)
        col_names.append(col)

        print(f"  {len(pages)} pages -> {n} chunks")

    if not col_names:
        print("\nNo collections ingested.")
        sys.exit(0)

    print(f"\nIngested {len(col_names)} collection(s): {', '.join(col_names)}")
    write_config(col_names)

    print("\nNext:")
    print(f"  uv run chroma run --path {CHROMA_DIR} --port 8010")
    print(f"  uv run pulse scan --config {CONFIG_OUT}")


if __name__ == "__main__":
    main()
