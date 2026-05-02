# pulse-scan

A read-only RAG corpus health scanner. Connects to your vector store, analyzes the corpus,
and produces a health report identifying stale chunks, near-duplicates, contradictions, and
supersession candidates.

---

## ⚠ PII Warning

This tool displays chunk text in plain form in the dashboard and JSON reports.
**Do not run on corpora containing PII or other sensitive content.**
PII redaction is planned for v1.1.

---

## Requirements

- Python 3.11+
- **GPU required** — one of:
  - NVIDIA GPU with ≥8GB VRAM (T4, 3060, 3090, A10, A100) — use `device: cuda`
  - Apple Silicon (M1/M2/M3/M4) — use `device: mps`
- No CPU-only fallback — the scanner exits at startup if no GPU is detected

## Environment setup

### Using uv (recommended)

```bash
# Install uv if you don't have it
curl -LsSf https://astral.sh/uv/install.sh | sh

# Create a virtual environment and install all dependencies
uv sync

# For Chroma vector-store support, install the optional extra
uv sync --extra chroma

# Activate the environment
source .venv/bin/activate
```

### Using pip

```bash
python -m venv .venv
source .venv/bin/activate      # Windows: .venv\Scripts\activate

pip install -e .               # core dependencies
pip install -e ".[chroma]"     # + Chroma adapter
pip install -e ".[dev]"        # + test tooling
```

## Quick start

```bash
# Copy and edit the example config
cp pulse.config.yaml.example pulse.config.yaml
```

Edit `pulse.config.yaml` and set the correct device for your machine:

```yaml
inference:
  device: cuda   # NVIDIA GPU
  # device: mps  # Apple Silicon (M1/M2/M3/M4)
  # device: cpu  # no GPU — slow, not recommended for production
```

```bash
# Run a scan
uv run pulse scan --config pulse.config.yaml

# Or, if you have the venv activated (source .venv/bin/activate):
pulse scan --config pulse.config.yaml
```

## Dashboard server

The dashboard is a Streamlit app served by the `pulse dashboard` command.

```bash
# Start on the default port (8501)
pulse dashboard

# Custom port
pulse dashboard --port 8080

# Point at a non-default data directory
pulse dashboard --data-dir /path/to/.pulse --port 8080
```

Open `http://localhost:8501` in your browser. The server streams scan results
from the DuckDB control plane in `.pulse/` — run `pulse scan` at least once
before opening the dashboard.

To run Streamlit directly (useful during UI development):

```bash
streamlit run pulse_scan/dashboard/app.py -- --data-dir .pulse
```

## Development (no GPU)

```bash
PULSE_SKIP_GPU_CHECK=1 uv run pulse scan --config pulse.config.yaml.example
```

## Running tests

```bash
# All tests
uv run pytest

# With coverage
uv run pytest --cov=pulse_scan --cov-report=term-missing
```

## Configuration

See `pulse.config.yaml.example` for the full configuration reference.

## Build order

v1 is being built in the order described in §10 of the LLD. Each step is gated on approval:

1. [x] Fixture adapter + Stage 0 ingestion + DuckDB schema
2. [x] Synthetic 50-chunk benchmark corpus
3. [x] Calibration (Stage 0.5)
4. [x] Embedding-channel dedup (Stage 1, half)
5. [x] Clustering (Stage 2)
6. [x] NLI contradiction detection (Stage 4, one detector)
7. [x] Staleness scoring (Stage 5)
8. [x] JSON report (Stage 6, half)
9. [x] Text-channel dedup, numeric/version detectors
10. [x] Triage with cost budget (Stage 3)
11. [x] Streamlit dashboard with calibration loop
12. [x] Chroma adapter
13. [ ] AWS ECS deployment scripts
