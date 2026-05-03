"""Stage 8: JSON Report.

Reads finalized DuckDB state and writes a structured JSON report to
<data_dir>/report-<run_id>.json.

The report covers the four active detection channels: embedding-channel dedup,
clustering, NLI contradictions, and staleness scoring.  Text-channel dedup and
numeric/version detectors (Step 9) will add a second pass to the same file once
they exist — hence "Stage 6, half".
"""

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

import duckdb
import structlog

log = structlog.get_logger()


class ReportStage:
    def __init__(
        self,
        conn: duckdb.DuckDBPyConnection,
        data_dir: Path,
        run_id: str,
        store_type: str = "fixture",
    ):
        self.conn = conn
        self.data_dir = data_dir
        self.run_id = run_id
        self.store_type = store_type

    def run(self) -> Path:
        report = {
            "run_id": self.run_id,
            "generated_at": datetime.now(timezone.utc).isoformat(),
            "corpus_info": self._corpus_info(),
            "calibration": self._calibration(),
            "summary": self._summary(),
            "dedup_groups": self._dedup_groups(),
            "contradictions": self._contradictions(),
            "staleness": self._staleness(),
        }

        out_path = self.data_dir / f"report-{self.run_id}.json"
        out_path.write_text(json.dumps(report, indent=2, default=str))
        log.info("report_written", path=str(out_path), run_id=self.run_id)
        return out_path

    # ------------------------------------------------------------------
    # Sections
    # ------------------------------------------------------------------

    def _corpus_info(self) -> dict:
        total = self.conn.execute("SELECT COUNT(*) FROM chunks").fetchone()[0]
        active = self.conn.execute("SELECT COUNT(*) FROM chunks WHERE deleted_at IS NULL").fetchone()[0]
        collections = self.conn.execute(
            "SELECT collection, COUNT(*) AS n "
            "FROM chunks WHERE deleted_at IS NULL "
            "GROUP BY collection ORDER BY collection"
        ).fetchall()
        return {
            "store_type": self.store_type,
            "total_chunks": total,
            "active_chunks": active,
            "collections": [{"name": col, "chunk_count": n} for col, n in collections],
        }

    def _calibration(self) -> Optional[dict]:
        row = self.conn.execute(
            "SELECT dedup_threshold, contradiction_candidate_threshold, cluster_min_density "
            "FROM calibration ORDER BY rowid DESC LIMIT 1"
        ).fetchone()
        if row is None:
            return None
        return {
            "dedup_cosine_threshold": row[0],
            "contradiction_candidate_threshold": row[1],
            "cluster_min_density": row[2],
        }

    def _summary(self) -> dict:
        n_groups = self.conn.execute("SELECT COUNT(*) FROM dedup_groups").fetchone()[0]
        n_in_groups = self.conn.execute(
            "SELECT COUNT(*) FROM ("
            "  SELECT json_each.value FROM dedup_groups, "
            "  json_each(dedup_groups.member_chunk_ids)"
            ")"
        ).fetchone()[0]
        n_contradictions = self.conn.execute(
            "SELECT COUNT(*) FROM contradictions WHERE scan_run_id = ? AND user_resolution IS NULL",
            [self.run_id],
        ).fetchone()[0]
        label_rows = self.conn.execute(
            "SELECT staleness_label, COUNT(*) FROM chunks "
            "WHERE deleted_at IS NULL AND staleness_label IS NOT NULL "
            "GROUP BY staleness_label"
        ).fetchall()
        label_counts = {label: n for label, n in label_rows}
        return {
            "dedup_groups": n_groups,
            "chunks_in_dedup_groups": n_in_groups,
            "contradictions_unresolved": n_contradictions,
            "staleness_labels": {
                "fresh": label_counts.get("fresh", 0),
                "aging": label_counts.get("aging", 0),
                "stale": label_counts.get("stale", 0),
                "abandoned": label_counts.get("abandoned", 0),
            },
        }

    def _dedup_groups(self) -> list:
        groups = self.conn.execute(
            "SELECT group_id, canonical_chunk_id, member_chunk_ids, detection_channels "
            "FROM dedup_groups ORDER BY group_id"
        ).fetchall()

        result = []
        for group_id, canonical, members_json, channels_json in groups:
            member_ids = json.loads(members_json)
            # Enrich with chunk text + metadata
            placeholders = ", ".join("?" * len(member_ids))
            chunk_rows = self.conn.execute(
                f"SELECT chunk_id, text, collection, resolved_timestamp FROM chunks WHERE chunk_id IN ({placeholders})",
                member_ids,
            ).fetchall()
            chunk_map = {r[0]: r for r in chunk_rows}
            members = [
                {
                    "chunk_id": cid,
                    "text": chunk_map[cid][1] if cid in chunk_map else None,
                    "collection": chunk_map[cid][2] if cid in chunk_map else None,
                    "resolved_timestamp": (
                        chunk_map[cid][3].isoformat() if cid in chunk_map and chunk_map[cid][3] else None
                    ),
                    "is_canonical": cid == canonical,
                }
                for cid in member_ids
            ]
            result.append(
                {
                    "group_id": group_id,
                    "canonical_chunk_id": canonical,
                    "detection_channels": json.loads(channels_json),
                    "members": members,
                }
            )
        return result

    def _contradictions(self) -> list:
        rows = self.conn.execute(
            "SELECT chunk_a, chunk_b, detector, raw_score, calibrated_confidence, "
            "       calibration_state, direction, user_resolution "
            "FROM contradictions "
            "WHERE scan_run_id = ? AND user_resolution IS NULL "
            "ORDER BY chunk_a, chunk_b",
            [self.run_id],
        ).fetchall()

        if not rows:
            return []

        # Collect all chunk IDs for batch text lookup
        all_ids = list({cid for row in rows for cid in (row[0], row[1])})
        placeholders = ", ".join("?" * len(all_ids))
        chunk_rows = self.conn.execute(
            f"SELECT chunk_id, text, collection FROM chunks WHERE chunk_id IN ({placeholders})",
            all_ids,
        ).fetchall()
        chunk_map = {r[0]: (r[1], r[2]) for r in chunk_rows}

        result = []
        for chunk_a, chunk_b, detector, raw_score, cal_conf, cal_state, direction, _ in rows:
            result.append(
                {
                    "chunk_a_id": chunk_a,
                    "chunk_b_id": chunk_b,
                    "chunk_a_text": chunk_map.get(chunk_a, (None,))[0],
                    "chunk_b_text": chunk_map.get(chunk_b, (None,))[0],
                    "chunk_a_collection": chunk_map.get(chunk_a, (None, None))[1],
                    "chunk_b_collection": chunk_map.get(chunk_b, (None, None))[1],
                    "detector": detector,
                    "raw_score": raw_score,
                    "calibrated_confidence": cal_conf,
                    "calibration_state": cal_state,
                    "direction": direction,
                }
            )
        return result

    def _staleness(self) -> list:
        rows = self.conn.execute(
            "SELECT chunk_id, collection, text, staleness_score, staleness_label, "
            "       staleness_components, resolved_timestamp, cluster_id "
            "FROM chunks "
            "WHERE deleted_at IS NULL AND staleness_score IS NOT NULL "
            "ORDER BY staleness_score DESC, chunk_id"
        ).fetchall()

        superseded = self._load_superseded_ids()

        result = []
        for (
            chunk_id,
            collection,
            text,
            score,
            label,
            components_json,
            resolved_ts,
            cluster_id,
        ) in rows:
            result.append(
                {
                    "chunk_id": chunk_id,
                    "collection": collection,
                    "text": text,
                    "staleness_score": score,
                    "staleness_label": label,
                    "staleness_components": json.loads(components_json) if components_json else None,
                    "resolved_timestamp": resolved_ts.isoformat() if resolved_ts else None,
                    "cluster_id": cluster_id,
                    "is_superseded": chunk_id in superseded,
                }
            )
        return result

    def _load_superseded_ids(self) -> set:
        rows = self.conn.execute("SELECT canonical_chunk_id, member_chunk_ids FROM dedup_groups").fetchall()
        superseded: set[str] = set()
        for canonical, members_json in rows:
            for member in json.loads(members_json):
                if member != canonical:
                    superseded.add(member)
        return superseded
