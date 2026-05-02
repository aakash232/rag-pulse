"""Database query functions for the Streamlit dashboard.

All functions accept an open DuckDB connection (read-write, same file the
scanner writes to) and return plain Python structures or pandas DataFrames.
Kept separate from app.py so they can be unit-tested without Streamlit.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Optional

import duckdb
import pandas as pd


def open_connection(data_dir: Path) -> Optional[duckdb.DuckDBPyConnection]:
    db_path = data_dir / "pulse.db"
    if not db_path.exists():
        return None
    return duckdb.connect(str(db_path), read_only=True)


def _open_rw(data_dir: Path) -> duckdb.DuckDBPyConnection:
    """Short-lived read-write connection for mutations (closes immediately after use)."""
    return duckdb.connect(str(data_dir / "pulse.db"))


# ---------------------------------------------------------------------------
# Read queries
# ---------------------------------------------------------------------------

def get_available_runs(conn: duckdb.DuckDBPyConnection) -> list[str]:
    rows = conn.execute(
        "SELECT run_id FROM scan_runs ORDER BY started_at DESC"
    ).fetchall()
    return [r[0] for r in rows]


def get_corpus_overview(conn: duckdb.DuckDBPyConnection) -> dict:
    total = conn.execute("SELECT COUNT(*) FROM chunks").fetchone()[0]
    active = conn.execute(
        "SELECT COUNT(*) FROM chunks WHERE deleted_at IS NULL"
    ).fetchone()[0]
    deleted = total - active
    n_groups = conn.execute("SELECT COUNT(*) FROM dedup_groups").fetchone()[0]
    n_contra = conn.execute(
        "SELECT COUNT(*) FROM contradictions WHERE user_resolution IS NULL"
    ).fetchone()[0]
    collections = conn.execute(
        "SELECT collection, COUNT(*) AS n "
        "FROM chunks WHERE deleted_at IS NULL "
        "GROUP BY collection ORDER BY collection"
    ).fetchall()
    return {
        "total_chunks": total,
        "active_chunks": active,
        "deleted_chunks": deleted,
        "dedup_groups": n_groups,
        "open_contradictions": n_contra,
        "collections": [{"name": c, "count": n} for c, n in collections],
    }


def get_staleness_label_counts(conn: duckdb.DuckDBPyConnection) -> dict:
    rows = conn.execute(
        "SELECT staleness_label, COUNT(*) "
        "FROM chunks WHERE deleted_at IS NULL AND staleness_label IS NOT NULL "
        "GROUP BY staleness_label"
    ).fetchall()
    counts = {"fresh": 0, "aging": 0, "stale": 0, "abandoned": 0}
    for label, n in rows:
        counts[label] = n
    return counts


def get_staleness_count(
    conn: duckdb.DuckDBPyConnection,
    label_filter: Optional[str] = None,
) -> int:
    conditions = ["deleted_at IS NULL", "staleness_score IS NOT NULL"]
    params: list = []
    if label_filter and label_filter != "all":
        conditions.append("staleness_label = ?")
        params.append(label_filter)
    where = "WHERE " + " AND ".join(conditions)
    return conn.execute(f"SELECT COUNT(*) FROM chunks {where}", params).fetchone()[0]


def get_staleness_df(
    conn: duckdb.DuckDBPyConnection,
    label_filter: Optional[str] = None,
    limit: Optional[int] = None,
    offset: int = 0,
) -> pd.DataFrame:
    params: list = []
    conditions = ["deleted_at IS NULL", "staleness_score IS NOT NULL"]
    if label_filter and label_filter != "all":
        conditions.append("staleness_label = ?")
        params.append(label_filter)
    where = "WHERE " + " AND ".join(conditions)
    pagination = f" LIMIT {int(limit)} OFFSET {int(offset)}" if limit is not None else ""
    rows = conn.execute(
        f"SELECT chunk_id, collection, staleness_score, staleness_label, "
        f"       staleness_components, text, resolved_timestamp "
        f"FROM chunks {where} ORDER BY staleness_score DESC{pagination}",
        params,
    ).fetchall()
    records = []
    for chunk_id, col, score, label, comp_json, text, ts in rows:
        comp = json.loads(comp_json) if comp_json else {}
        records.append({
            "chunk_id": chunk_id,
            "collection": col,
            "score": round(score, 4) if score is not None else None,
            "label": label,
            "age_decay": comp.get("age_decay"),
            "cluster_drift": comp.get("cluster_drift"),
            "contradiction_evidence": comp.get("contradiction_evidence"),
            "supersession_evidence": comp.get("supersession_evidence"),
            "text": (text or "")[:200],
            "resolved_timestamp": str(ts) if ts else None,
        })
    return pd.DataFrame(records)


def get_dedup_groups_count(conn: duckdb.DuckDBPyConnection) -> int:
    return conn.execute("SELECT COUNT(*) FROM dedup_groups").fetchone()[0]


def get_dedup_groups(
    conn: duckdb.DuckDBPyConnection,
    limit: Optional[int] = None,
    offset: int = 0,
) -> list[dict]:
    sql = (
        "SELECT group_id, canonical_chunk_id, member_chunk_ids, detection_channels "
        "FROM dedup_groups ORDER BY group_id"
    )
    if limit is not None:
        sql += f" LIMIT {int(limit)} OFFSET {int(offset)}"
    rows = conn.execute(sql).fetchall()
    if not rows:
        return []

    # Parse all groups first, collect every chunk ID in one pass.
    parsed: list[tuple] = []
    all_ids: list[str] = []
    for gid, canonical, members_json, channels_json in rows:
        member_ids = json.loads(members_json)
        channels = json.loads(channels_json)
        parsed.append((gid, canonical, member_ids, channels))
        all_ids.extend(member_ids)

    # Single bulk lookup instead of one query per group.
    chunk_map: dict[str, tuple] = {}
    if all_ids:
        placeholders = ", ".join("?" * len(all_ids))
        chunk_rows = conn.execute(
            f"SELECT chunk_id, text, collection, staleness_score "
            f"FROM chunks WHERE chunk_id IN ({placeholders})",
            all_ids,
        ).fetchall()
        chunk_map = {r[0]: r for r in chunk_rows}

    result = []
    for gid, canonical, member_ids, channels in parsed:
        members = [
            {
                "chunk_id": cid,
                "collection": chunk_map.get(cid, (None,) * 4)[2],
                "text": (chunk_map.get(cid, (None,) * 4)[1] or "")[:300],
                "staleness_score": chunk_map.get(cid, (None,) * 4)[3],
                "is_canonical": cid == canonical,
            }
            for cid in member_ids
        ]
        result.append({
            "group_id": gid,
            "canonical_chunk_id": canonical,
            "detection_channels": channels,
            "members": members,
            "n_members": len(members),
        })
    return result


def get_contradictions_count(
    conn: duckdb.DuckDBPyConnection,
    run_id: Optional[str] = None,
    unresolved_only: bool = True,
    detector: Optional[str] = None,
) -> int:
    conditions: list[str] = []
    params: list = []
    if run_id:
        conditions.append("scan_run_id = ?")
        params.append(run_id)
    if unresolved_only:
        conditions.append("user_resolution IS NULL")
    if detector and detector != "all":
        conditions.append("detector = ?")
        params.append(detector)
    where = ("WHERE " + " AND ".join(conditions)) if conditions else ""
    return conn.execute(
        f"SELECT COUNT(*) FROM contradictions {where}", params
    ).fetchone()[0]


def get_contradictions(
    conn: duckdb.DuckDBPyConnection,
    run_id: Optional[str] = None,
    unresolved_only: bool = True,
    detector: Optional[str] = None,
    limit: Optional[int] = None,
    offset: int = 0,
) -> list[dict]:
    conditions = []
    params: list = []
    if run_id:
        conditions.append("c.scan_run_id = ?")
        params.append(run_id)
    if unresolved_only:
        conditions.append("c.user_resolution IS NULL")
    if detector and detector != "all":
        conditions.append("c.detector = ?")
        params.append(detector)
    where = ("WHERE " + " AND ".join(conditions)) if conditions else ""
    pagination = f" LIMIT {int(limit)} OFFSET {int(offset)}" if limit is not None else ""
    rows = conn.execute(
        f"""SELECT c.chunk_a, c.chunk_b, c.detector, c.raw_score,
                   c.calibration_state, c.direction, c.user_resolution,
                   c.scan_run_id,
                   ca.text AS text_a, cb.text AS text_b,
                   ca.collection AS col_a, cb.collection AS col_b
            FROM contradictions c
            LEFT JOIN chunks ca ON ca.chunk_id = c.chunk_a
            LEFT JOIN chunks cb ON cb.chunk_id = c.chunk_b
            {where}
            ORDER BY c.raw_score DESC{pagination}""",
        params,
    ).fetchall()
    result = []
    for row in rows:
        (chunk_a, chunk_b, detector_, score, cal_state, direction,
         user_res, run, text_a, text_b, col_a, col_b) = row
        result.append({
            "chunk_a": chunk_a,
            "chunk_b": chunk_b,
            "detector": detector_,
            "raw_score": round(score, 4) if score else None,
            "calibration_state": cal_state,
            "direction": direction,
            "user_resolution": user_res,
            "scan_run_id": run,
            "text_a": text_a or "",
            "text_b": text_b or "",
            "collection_a": col_a or "",
            "collection_b": col_b or "",
        })
    return result


def get_triage_summary(
    conn: duckdb.DuckDBPyConnection,
    run_id: str,
) -> dict:
    row = conn.execute(
        "SELECT COUNT(*), SUM(CASE WHEN was_scanned THEN 1 ELSE 0 END) "
        "FROM triage_log WHERE scan_run_id = ?",
        [run_id],
    ).fetchone()
    if not row or row[0] == 0:
        return {"chunks_scored": 0, "chunks_scanned": 0}
    return {"chunks_scored": row[0], "chunks_scanned": row[1]}


# ---------------------------------------------------------------------------
# Write queries
# ---------------------------------------------------------------------------

def resolve_contradiction(
    data_dir: Path,
    chunk_a: str,
    chunk_b: str,
    resolution: str,
) -> None:
    """Set user_resolution on a contradiction pair (both ordered directions)."""
    rw = _open_rw(data_dir)
    try:
        rw.execute(
            "UPDATE contradictions "
            "SET user_resolution = ?, resolved_at = CURRENT_TIMESTAMP "
            "WHERE (chunk_a = ? AND chunk_b = ?) OR (chunk_a = ? AND chunk_b = ?)",
            [resolution, chunk_a, chunk_b, chunk_b, chunk_a],
        )
    finally:
        rw.close()


def get_resolution_summary(conn: duckdb.DuckDBPyConnection) -> dict:
    rows = conn.execute(
        "SELECT user_resolution, COUNT(*) FROM contradictions GROUP BY user_resolution"
    ).fetchall()
    summary = {"confirmed": 0, "false_positive": 0, "unresolved": 0}
    for res, n in rows:
        if res is None:
            summary["unresolved"] += n
        elif res in summary:
            summary[res] += n
    return summary
