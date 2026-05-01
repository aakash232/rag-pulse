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
- **NVIDIA GPU with ≥8GB VRAM** (T4, 3060, 3090, A10, A100 all work)
- No CPU inference fallback — the scanner exits at startup if no GPU is detected

## Quick start

```bash
# Install
pip install pulse-scan

# Copy and edit the example config
cp pulse.config.yaml.example pulse.config.yaml

# Run a scan
pulse scan --config pulse.config.yaml

# Launch the dashboard (Step 11, not yet implemented)
pulse dashboard
```

## Development (no GPU)

```bash
PULSE_SKIP_GPU_CHECK=1 pulse scan --config pulse.config.yaml.example
```

## Configuration

See `pulse.config.yaml.example` for the full configuration reference.

## Build order

v1 is being built in the order described in §10 of the LLD. Each step is gated on approval:

1. [x] Fixture adapter + Stage 0 ingestion + DuckDB schema
2. [ ] Synthetic 50-chunk benchmark corpus
3. [ ] Calibration (Stage 0.5)
4. [ ] Embedding-channel dedup (Stage 1, half)
5. [ ] Clustering (Stage 2)
6. [ ] NLI contradiction detection (Stage 4, one detector)
7. [ ] Staleness scoring (Stage 5)
8. [ ] JSON report (Stage 6, half)
9. [ ] Text-channel dedup, numeric/version detectors
10. [ ] Triage with cost budget (Stage 3)
11. [ ] Streamlit dashboard with calibration loop
12. [ ] Chroma adapter
13. [ ] AWS ECS deployment scripts
