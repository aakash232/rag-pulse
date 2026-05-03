# local_test — rag-pulse sandbox

A self-contained playground for running rag-pulse end-to-end on your machine.
The `pdfs/` folder ships with 10 synthetic documents that are designed to surface
contradictions, stale content, and version drift — exactly the kind of signal
rag-pulse is built to detect.

## What's here

| Path | Purpose |
|------|---------|
| `pdfs/` | 10 pre-generated sample PDFs (see below) |
| `ingest.py` | Chunks PDFs, embeds via OpenAI, loads into local ChromaDB, writes `pulse.config.yaml` |
| `pdf_generator.py` | Re-generates the sample PDFs with OpenAI (only needed if you want fresh content) |

Generated at runtime (gitignored):

| Path | Purpose |
|------|---------|
| `chroma_data/` | Local ChromaDB persistent store |
| `.env` | Your `OPENAI_API_KEY` |
| `pulse.config.yaml` | Auto-written by `ingest.py` |

## Quick start

### 1. Prerequisites

```bash
# from the repo root
uv sync
```

You need an OpenAI API key for embeddings.

### 2. Create `.env`

```bash
echo "OPENAI_API_KEY=sk-..." > local_test/.env
```

### 3. Ingest PDFs into ChromaDB

```bash
uv run python local_test/ingest.py
```

This will:
- Extract and chunk every PDF in `local_test/pdfs/`
- Call OpenAI `text-embedding-3-small` to embed chunks
- Write them into `local_test/chroma_data/`
- Generate `local_test/pulse.config.yaml`

### 4. Start ChromaDB

```bash
uv run chroma run --path local_test/chroma_data --port 8010
```

Keep this running in a separate terminal.

### 5. Run the scan

```bash
uv run pulse scan --config local_test/pulse.config.yaml
```

### 6. Open the dashboard

```bash
uv run pulse dashboard --config local_test/pulse.config.yaml
```

---

## Sample documents

The included PDFs are synthetic technical docs generated to trigger rag-pulse detectors:

| File | Contents |
|------|---------|
| `api_v1.pdf` / `api_v2.pdf` | API design guidelines — version drift between v1 and v2 |
| `security_v1.pdf` / `security_v2.pdf` | Security policy — contradictory auth rules (OAuth vs API keys) |
| `deploy_2019.pdf` / `deploy_2025.pdf` | Deployment guides — Swarm/Jenkins vs Kubernetes/ArgoCD |
| `arch_v1.pdf` / `arch_v2.pdf` | System architecture — component and data-flow changes |
| `postmortem.pdf` | Incident postmortem |
| `product_spec.pdf` | Product requirements — intentionally includes duplicates and a conflict |

---

## Bring your own PDFs

Drop any PDFs into `local_test/pdfs/` and re-run `ingest.py`. Each PDF becomes
its own ChromaDB collection. The config is regenerated automatically.

---

## Re-generate sample PDFs

If you want a fresh batch of synthetic docs (uses OpenAI, costs a few cents):

```bash
uv run python local_test/pdf_generator.py
```

Then re-run `ingest.py`.
