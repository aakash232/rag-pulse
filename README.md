# pulse-scan

**A read-only health scanner for RAG knowledge bases.** Connects to your vector store, analyzes the corpus, and produces a report identifying stale chunks, near-duplicates, contradictions, and superseded content — without writing a single byte back to your store.

<img width="1471" height="794" alt="image" src="https://github.com/user-attachments/assets/65f0440d-2a66-4793-a96e-bbe9f28cb988" />

<img width="1508" height="803" alt="image" src="https://github.com/user-attachments/assets/097fee6a-bb8b-4078-9678-19605174558f" />

<img width="1506" height="831" alt="image" src="https://github.com/user-attachments/assets/e5fb771f-a410-49a7-a3ab-ed0f4fbfa9c2" />

<img width="1112" height="788" alt="image" src="https://github.com/user-attachments/assets/c4c97191-9275-40e8-bfdb-0b1fcdc397a2" />



---

## The problem

Your RAG app answers from whatever chunks land in the top-k. If those chunks contradict each other, or one comes from a doc updated two years ago, the model synthesizes an answer anyway. No error, no flag.

Chunks go stale. The same fact appears in three chunks with three different answers. SDK versions drift between collections. None of this is visible to your retrieval pipeline.

`pulse-scan` finds that drift. It reads your vector store, runs analysis, and produces a report. It writes nothing back.

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

The math is inspired — Grofsky 2025 time-decay for staleness, standard NLI for contradiction, HDBSCAN for clustering, conformal prediction for calibration. The **form factor is novel**: a vector-store-agnostic scanner that produces a corpus health report and a navigable dashboard.

The visual language for "what does corpus health look like" is unclaimed territory. pulse-scan treats it as a first-class deliverable, not an afterthought.

It runs entirely on your infrastructure. No data leaves your environment. 

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

**Three contradiction detectors** because each catches what the others miss:
- **NLI** (DeBERTa-v3 on MNLI) — semantic contradictions between claims
- **Numeric** — `2.4%` vs `2.9%`, `$5` vs `$8`, conflicting rate limits or thresholds
- **Version** — `package@1.2.3` vs `package@2.0.0` in similar context

**Calibration is per-corpus, not hardcoded.** A cosine of 0.85 means "near-duplicate" for one embedding model and "vaguely related" for another. Stage 1 fits thresholds to your actual similarity distribution before any thresholded operation runs. Contradiction confidence calibration emerges from your own approve/reject feedback in the dashboard — no seed labels shipped, no onboarding tax.

**Scans are incremental.** After the first run, only changed or new chunks are processed. The control plane persists between scans; chunks deferred by the triage budget are prioritized next run.

---

## Quick start

### Requirements

- Python 3.11+
- GPU: NVIDIA CUDA (8GB+ VRAM) or Apple Silicon MPS. The CLI exits at startup if neither is detected. Most pipeline stages run on CPU; NLI inference (Stage 5) is the bottleneck that needs the GPU. Bypass for dev: `PULSE_SKIP_GPU_CHECK=1`.
- An OpenAI API key (ONLY for sample corpus embeddings)

### Install

```bash
uv sync --extra chroma   # or: pip install -e ".[chroma]"
```

### Try it on the sample corpus

The repo ships 10 synthetic documents built to surface every signal pulse-scan detects: contradictory auth policies, stale deployment guides, API version drift, and duplicate specs. No vector store to set up.

```bash
echo "OPENAI_API_KEY=sk-..." > local_test/.env
uv run python local_test/ingest.py
```

Start ChromaDB in a separate terminal and keep it running:

```bash
uv run chroma run --path local_test/chroma_data --port 8010
```

Then scan and open the dashboard:

```bash
uv run pulse scan --config local_test/pulse.config.yaml
uv run pulse dashboard --config local_test/pulse.config.yaml
# open http://localhost:8501
```

See [`local_test/README.md`](local_test/README.md) for details on the sample documents and how to bring your own PDFs.

### Connect to your own vector store

```bash
cp pulse.config.yaml.example pulse.config.yaml
# edit: set your vector store connection, collections, and device
uv run pulse scan --config pulse.config.yaml
uv run pulse dashboard --config pulse.config.yaml
```

See `pulse.config.yaml.example` for the full reference.

### Dashboard views

- **Overview** - Key metrics (active chunks, % fresh, dedup groups, open contradictions), staleness distribution bar chart, collections breakdown, and contradiction review progress.
- **Duplicates** - Paginated list of near-duplicate groups. Each group shows member chunks, how they were detected (text or embedding channel), and which is canonical.
- **Contradictions** - Contradiction pairs shown side-by-side. Filter by detector (NLI, numeric, version) or review status. Inline verdict (confirm / false positive / skip) feeds threshold calibration on the next scan.
- **Staleness** - Per-chunk staleness scores (0.0 to 1.0) with a four-component breakdown: age decay, semantic drift, contradiction evidence, and supersession evidence. Filter by label (fresh / aging / stale / abandoned).

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

## What it does not do yet

- **Chroma only.** Pinecone, Weaviate, Qdrant, and pgvector adapters are planned but not built for v1.
- **NLI inference needs a GPU.** Most stages run on CPU. Stage 5 (DeBERTa NLI) is the bottleneck; the CLI exits at startup if neither CUDA nor Apple Silicon MPS is available. For dev/testing without a GPU, set `PULSE_SKIP_GPU_CHECK=1` — NLI will fall back to CPU, which is workable for small corpora.
- **No PII handling.** The dashboard and reports display chunk text as plain text. Do not run this on corpora containing personal data. Redaction is planned for v1.1.
- **Not production-ready.** No auth layer, no multi-user support, no access control.
- **AWS deployment pending.** ECS deployment scripts are the one remaining incomplete item.

---

## FAQ

**Why not LangChain, Langfuse, or DeepEval?**
Those tools measure output quality: whether the answer your app produced was correct or useful. pulse-scan measures corpus quality: whether the raw material your retriever pulls from is internally consistent and current. They operate at different layers and can run alongside each other.

**Does this work with Pinecone, Weaviate, or Qdrant?**
Not in v1. Chroma is the only supported vector store right now. The adapter protocol is documented in the LLD if you want to add one.

**Does my data leave my infrastructure?**
No. The scanner runs on your hardware. The only external call is to whichever embedding model you configure, the same call your existing RAG pipeline already makes. No telemetry, no callbacks.

**How is this different from an eval suite?**
An eval suite runs queries against your app and scores the answers. pulse-scan never runs a query. It analyzes the chunks themselves: are they stale, do any two chunks contradict each other, does the same fact appear three ways? Different question, different tool.

**Is this production-ready?**
No. The core pipeline is complete and tested. There is no auth layer, multi-tenancy, or hardened deployment. Treat it as a developer inspection tool for now.

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
