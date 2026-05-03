# pulse-scan

**A read-only health scanner for RAG knowledge bases.** Connects to your vector store, analyzes the corpus, and produces a report identifying stale chunks, near-duplicates, contradictions, and superseded content — without writing a single byte back to your store.

---

## The problem

Vector stores own retrieval. Eval tools own output quality. **No tool owns corpus integrity.**

Chunks go stale. Documentation gets updated in one collection but not another. SDK versions diverge. The same fact appears in three chunks with three different answers. None of this is visible to your retrieval pipeline — it silently degrades the answers your RAG system gives, and existing tooling has no way to see it.

pulse-scan is the integrity layer.

---

## What it finds

| Signal | What it means |
|--------|--------------|
| **Stale chunks** | Content that has aged past its useful half-life — scored `fresh` / `aging` / `stale` / `abandoned` on a continuous 0–1 scale |
| **Near-duplicates** | Lexically identical or semantically equivalent chunks that waste retrieval budget and confuse ranking |
| **Contradictions** | Chunk pairs that make conflicting claims — caught by three parallel detectors (NLI, numeric, version) |
| **Supersession candidates** | Older chunks of the same content that a newer chunk has replaced |

---

## Why this is new

The math is borrowed — Grofsky 2025 time-decay for staleness, standard NLI for contradiction, HDBSCAN for clustering, conformal prediction for calibration. The **form factor is novel**: a vector-store-agnostic scanner that produces a corpus health report and a navigable dashboard.

The visual language for "what does corpus health look like" is unclaimed territory. pulse-scan treats it as a first-class deliverable, not an afterthought.

It runs entirely on your infrastructure. No data leaves your environment. Cost is bounded and predictable (~$6–10/month for weekly scans of a 50k-chunk corpus on AWS spot).

---

## How it works

The scan pipeline runs in seven stages:

```
[Your Vector Store]
        │
        │  (read-only adapter)
        ▼
┌─────────────────────────────────────┐
│  Stage 0    Ingestion + cache       │  fetch chunks → DuckDB + numpy memmap
│  Stage 1    Calibration            │  fit per-corpus cosine thresholds (HNSW)
│  Stage 2    Deduplication          │  MinHash+LSH (text) + HNSW (embedding)
│  Stage 3    Clustering             │  UMAP → HDBSCAN → cluster assignments
│  Stage 4    Triage                 │  cost-budgeted priority queue
│  Stage 5    NLI contradiction      │  DeBERTa-v3 cross-encoder (GPU)
│  Stage 6    Regex detectors        │  numeric + version contradictions
│  Stage 7    Staleness scoring      │  Grofsky half-life + cluster drift
│  Stage 8    Report + Dashboard     │  JSON/Parquet + Streamlit
└─────────────────────────────────────┘
        │
        ▼
┌─────────────────────────────────────┐
│  Control plane (local, persistent)  │
│  DuckDB  — chunks, clusters, edges  │
│  NumPy memmap — embeddings          │
│  Parquet — exported reports         │
└─────────────────────────────────────┘
```

**Three contradiction detectors run in parallel** because each catches what the others miss:
- **NLI** (DeBERTa-v3 on MNLI) — semantic contradictions between claims
- **Numeric** — `2.4%` vs `2.9%`, `$5` vs `$8`, conflicting rate limits or thresholds
- **Version** — `package@1.2.3` vs `package@2.0.0` in similar context

**Calibration is per-corpus, not hardcoded.** A cosine of 0.85 means "near-duplicate" for one embedding model and "vaguely related" for another. Stage 1 fits thresholds to your actual similarity distribution before any thresholded operation runs. Contradiction confidence calibration emerges from your own approve/reject feedback in the dashboard — no seed labels shipped, no onboarding tax.

**Scans are incremental.** After the first run, only changed or new chunks are processed. The control plane persists between scans; chunks deferred by the triage budget are prioritized next run.

---

## Quick start

### Requirements

- Python 3.11+
- GPU: NVIDIA ≥8GB VRAM (`device: cuda`) or Apple Silicon (`device: mps`) — **no CPU fallback**

### Install

```bash
# Using uv (recommended)
curl -LsSf https://astral.sh/uv/install.sh | sh
uv sync --extra chroma
```

```bash
# Using pip
python -m venv .venv && source .venv/bin/activate
pip install -e ".[chroma]"
pip install -e ".[dev]"    # + test tooling
```

### Configure

```bash
cp pulse.config.yaml.example pulse.config.yaml
```

Edit `pulse.config.yaml` — set your vector store connection, collections, and device:
- See `pulse.config.yaml.example` for the full reference.

### Run a scan

```bash
uv run pulse scan --config pulse.config.yaml
# or with venv activated: pulse scan --config pulse.config.yaml
```

### Open the dashboard

```bash
uv run pulse dashboard --config pulse.config.yaml
# open http://localhost:8501
```

The dashboard provides five views:

- **Corpus map** — 2D UMAP scatter, points colored by staleness, sized by retrieval count. Click to drill in; drag to filter a region.
- **Contradiction graph** — force-directed graph of contradiction edges. Edge thickness = confidence. Inline approve / reject / skip drives calibration.
- **Findings table** — sortable, filterable chunk list. Default sort: staleness descending.
- **Chunk drill-down** — full text, cosine neighbors, contradicting chunks, supersession candidates, first/last-seen timeline.
- **Scan summary** — counts by category, corpus health time-series across scans, calibration state.

---

## Try it without a vector store

`local_test/` is a self-contained sandbox. It ships with 10 synthetic PDFs designed to surface every signal pulse-scan detects: contradictory auth policies, stale deployment guides, API version drift, and duplicate specs. See [`local_test/README.md`](local_test/README.md) for setup (5 minutes, requires an OpenAI API key for embeddings).

---

## Development (no GPU)

```bash
PULSE_SKIP_GPU_CHECK=1 uv run pulse scan --config pulse.config.yaml.example
```

## Tests

```bash
uv run pytest
uv run pytest --cov=pulse_scan --cov-report=term-missing
```

---

## Hardware requirements

| | Minimum | Recommended |
|---|---|---|
| GPU | NVIDIA T4 / RTX 3060 (8GB VRAM) or Apple M-series | A10, A100, M2 Pro+ |
| RAM | 8GB | 16GB+ for >500k chunks |
| Disk | ~11KB per chunk (5KB DuckDB + 6KB embeddings at 1536d float32) | SSD — memmap is latency-sensitive |

At 1M chunks with 5 candidate pairs per chunk, NLI runs millions of inferences. GPU brings this from hours to minutes. No CPU fallback — the scanner exits at startup if no GPU is detected.

---

## Scale

| Corpus size | Scan time (T4) | Monthly cost (weekly scans, AWS spot) |
|---|---|---|
| 10k–50k chunks | 15–30 min | ~$6–7 |
| 100k–500k chunks | 30–60 min | ~$8–10 |
| 1M chunks | ~90 min | ~$15–20 |

---

## ⚠ PII warning

The dashboard and JSON reports display chunk text in plain form. **Do not run pulse-scan on corpora containing PII or other sensitive content.** PII redaction is planned for v1.1.

---

## Build status

All core stages are complete. The final step (AWS ECS deployment scripts) is in progress.

1. [x] Fixture adapter + Stage 0 ingestion + DuckDB schema
2. [x] Synthetic 50-chunk fixtures corpus
3. [x] Calibration (Stage 1)
4. [x] Embedding-channel dedup (Stage 2, half)
5. [x] Clustering (Stage 3)
6. [x] Triage with cost budget (Stage 4)
7. [x] NLI contradiction detection (Stage 5)
8. [x] Numeric/version detectors (Stage 6)
9. [x] Staleness scoring (Stage 7)
10. [x] JSON report (Stage 8)
11. [x] Streamlit dashboard with calibration loop
12. [x] Chroma adapter
13. [ ] AWS ECS deployment scripts

---

## Design document

Full low-level design — pipeline math, data model, calibration algorithm, adapter protocol, AWS deployment architecture — in [`docs/lld-v1.pdf`](docs/lld-v1.pdf).
