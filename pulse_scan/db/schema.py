from pathlib import Path

import duckdb

_DDL = """
CREATE TABLE IF NOT EXISTS chunks (
    store_id            TEXT,
    collection          TEXT,
    chunk_id            TEXT,
    text                TEXT,
    content_hash        TEXT,
    resolved_timestamp  TIMESTAMP,
    timestamp_source    TEXT,
    embedding_offset    BIGINT,
    cluster_id          INTEGER,
    staleness_score     DOUBLE,
    staleness_label     TEXT,
    staleness_components JSON,
    first_seen_by_pulse TIMESTAMP,
    last_seen_by_pulse  TIMESTAMP,
    deleted_at          TIMESTAMP,
    version             INTEGER DEFAULT 1,
    PRIMARY KEY (store_id, collection, chunk_id)
);

CREATE TABLE IF NOT EXISTS dedup_groups (
    group_id            INTEGER,
    canonical_chunk_id  TEXT,
    member_chunk_ids    JSON,
    detection_channels  JSON
);

CREATE TABLE IF NOT EXISTS clusters (
    cluster_id  INTEGER,
    chunk_id    TEXT
);

CREATE TABLE IF NOT EXISTS cluster_centroids (
    cluster_id  INTEGER,
    centroid    BLOB,
    n_chunks    INTEGER
);

CREATE TABLE IF NOT EXISTS contradictions (
    chunk_a               TEXT,
    chunk_b               TEXT,
    detector              TEXT,
    raw_score             DOUBLE,
    calibrated_confidence DOUBLE,
    calibration_state     TEXT,
    direction             TEXT,
    scan_run_id           TEXT,
    user_resolution       TEXT,
    resolved_at           TIMESTAMP
);

CREATE TABLE IF NOT EXISTS scan_runs (
    run_id      TEXT PRIMARY KEY,
    started_at  TIMESTAMP,
    finished_at TIMESTAMP,
    config      JSON,
    stats       JSON
);

CREATE TABLE IF NOT EXISTS calibration (
    scan_run_id                     TEXT,
    sample_hash                     TEXT,
    dedup_threshold                 DOUBLE,
    contradiction_candidate_threshold DOUBLE,
    cluster_min_density             DOUBLE,
    distributions                   JSON
);

CREATE TABLE IF NOT EXISTS triage_log (
    scan_run_id TEXT,
    chunk_id    TEXT,
    priority    DOUBLE,
    components  JSON,
    was_scanned BOOLEAN
);
"""


def open_db(data_dir: Path) -> duckdb.DuckDBPyConnection:
    data_dir.mkdir(parents=True, exist_ok=True)
    conn = duckdb.connect(str(data_dir / "pulse.db"))
    conn.executemany("PRAGMA", []) if False else None  # no-op
    conn.execute(_DDL)
    return conn
