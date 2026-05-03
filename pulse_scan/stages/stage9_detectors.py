"""Stage 9 detectors: Numeric and Version contradiction detection.

Both detectors scan pairs of chunks within the same cluster for two specific
contradiction patterns:

  NumericContradictionDetector — same textual context but different numbers
  VersionContradictionDetector — same product/package but different version strings

Unlike NLI, these detectors are regex-based and require no GPU.  They write
to the contradictions table with detector='numeric' or 'version'.

Candidate pairs: all pairs within the same cluster, skipping clusters with
more than MAX_CLUSTER_SIZE chunks to avoid O(N²) blowup.
"""

import re

import duckdb
import structlog

log = structlog.get_logger()

# Safety valve: skip clusters larger than this
MAX_CLUSTER_SIZE = 50

# ---- shared regex helpers --------------------------------------------------

_NUMBER_RE = re.compile(r"\b\d+(?:[.,]\d+)?\b")
_VERSION_RE = re.compile(r"v?(\d+\.\d+(?:\.\d+)*)\b", re.I)

# Minimum word-level Jaccard between context-normalized texts to count as same topic
_CONTEXT_SIMILARITY_THRESHOLD = 0.5


def _word_jaccard(text_a: str, text_b: str) -> float:
    words_a = set(text_a.lower().split())
    words_b = set(text_b.lower().split())
    if not words_a and not words_b:
        return 1.0
    union = len(words_a | words_b)
    return len(words_a & words_b) / union if union else 0.0


def _set_jaccard(s_a: set, s_b: set) -> float:
    if not s_a and not s_b:
        return 1.0
    union = len(s_a | s_b)
    return len(s_a & s_b) / union if union else 0.0


# ---------------------------------------------------------------------------
# Numeric detector
# ---------------------------------------------------------------------------


def _extract_numbers(text: str) -> set[float]:
    nums: set[float] = set()
    for m in _NUMBER_RE.finditer(text):
        try:
            nums.add(float(m.group().replace(",", "")))
        except ValueError:
            pass
    return nums


def _normalize_numbers(text: str) -> str:
    return _NUMBER_RE.sub("__NUM__", text)


def is_numeric_contradiction(text_a: str, text_b: str) -> tuple[bool, float]:
    """Return (is_contradiction, score).

    Flags when both texts contain numbers, their number sets differ, and
    their context (numbers stripped) is similar enough to be the same topic.
    Score = 1 − Jaccard(nums_a, nums_b).
    """
    nums_a = _extract_numbers(text_a)
    nums_b = _extract_numbers(text_b)
    if not nums_a or not nums_b:
        return False, 0.0
    if nums_a == nums_b:
        return False, 0.0
    ctx_similarity = _word_jaccard(_normalize_numbers(text_a), _normalize_numbers(text_b))
    if ctx_similarity < _CONTEXT_SIMILARITY_THRESHOLD:
        return False, 0.0
    score = 1.0 - _set_jaccard(nums_a, nums_b)
    return True, round(score, 4)


class NumericContradictionDetector:
    def __init__(
        self,
        conn: duckdb.DuckDBPyConnection,
        scan_run_id: str,
        scan_config=None,
    ):
        self.conn = conn
        self.scan_run_id = scan_run_id

    def run(self) -> dict:
        self.conn.execute(
            "DELETE FROM contradictions WHERE scan_run_id = ? AND detector = 'numeric'",
            [self.scan_run_id],
        )

        pairs = _cluster_pairs(self.conn)
        found = 0

        for a_id, a_text, b_id, b_text in pairs:
            contradiction, score = is_numeric_contradiction(a_text, b_text)
            if not contradiction:
                continue
            self.conn.execute(
                "INSERT INTO contradictions "
                "(chunk_a, chunk_b, detector, raw_score, calibrated_confidence, "
                " calibration_state, direction, scan_run_id, user_resolution, resolved_at) "
                "VALUES (?, ?, 'numeric', ?, NULL, 'uncalibrated', 'both', ?, NULL, NULL)",
                [a_id, b_id, score, self.scan_run_id],
            )
            found += 1

        log.info("numeric_detector_complete", pairs_checked=len(pairs), contradictions_found=found)
        return {"pairs_checked": len(pairs), "contradictions_found": found}


# ---------------------------------------------------------------------------
# Version detector
# ---------------------------------------------------------------------------


def _extract_versions(text: str) -> set[str]:
    return set(_VERSION_RE.findall(text))


def _normalize_versions(text: str) -> str:
    return _VERSION_RE.sub("__VER__", text)


def is_version_contradiction(text_a: str, text_b: str) -> tuple[bool, float]:
    """Return (is_contradiction, score).

    Flags when both texts mention version strings, the version sets differ, and
    the surrounding context is similar enough to be about the same thing.
    Score = 1.0 (binary: either there are conflicting versions or there aren't).
    """
    vers_a = _extract_versions(text_a)
    vers_b = _extract_versions(text_b)
    if not vers_a or not vers_b:
        return False, 0.0
    if vers_a == vers_b:
        return False, 0.0
    ctx_similarity = _word_jaccard(_normalize_versions(text_a), _normalize_versions(text_b))
    if ctx_similarity < _CONTEXT_SIMILARITY_THRESHOLD:
        return False, 0.0
    return True, 1.0


class VersionContradictionDetector:
    def __init__(
        self,
        conn: duckdb.DuckDBPyConnection,
        scan_run_id: str,
        scan_config=None,
    ):
        self.conn = conn
        self.scan_run_id = scan_run_id

    def run(self) -> dict:
        self.conn.execute(
            "DELETE FROM contradictions WHERE scan_run_id = ? AND detector = 'version'",
            [self.scan_run_id],
        )

        pairs = _cluster_pairs(self.conn)
        found = 0

        for a_id, a_text, b_id, b_text in pairs:
            contradiction, score = is_version_contradiction(a_text, b_text)
            if not contradiction:
                continue
            self.conn.execute(
                "INSERT INTO contradictions "
                "(chunk_a, chunk_b, detector, raw_score, calibrated_confidence, "
                " calibration_state, direction, scan_run_id, user_resolution, resolved_at) "
                "VALUES (?, ?, 'version', ?, NULL, 'uncalibrated', 'both', ?, NULL, NULL)",
                [a_id, b_id, score, self.scan_run_id],
            )
            found += 1

        log.info("version_detector_complete", pairs_checked=len(pairs), contradictions_found=found)
        return {"pairs_checked": len(pairs), "contradictions_found": found}


# ---------------------------------------------------------------------------
# Shared candidate generation
# ---------------------------------------------------------------------------


def _cluster_pairs(conn: duckdb.DuckDBPyConnection) -> list[tuple[str, str, str, str]]:
    """All (chunk_a_id, text_a, chunk_b_id, text_b) pairs within same cluster.

    Skips clusters larger than MAX_CLUSTER_SIZE to cap quadratic growth.
    """
    rows = conn.execute(
        "SELECT chunk_id, cluster_id, text "
        "FROM chunks "
        "WHERE deleted_at IS NULL AND cluster_id IS NOT NULL "
        "ORDER BY cluster_id, chunk_id"
    ).fetchall()

    clusters: dict[int, list[tuple[str, str]]] = {}
    for chunk_id, cluster_id, text in rows:
        clusters.setdefault(cluster_id, []).append((chunk_id, text or ""))

    pairs: list[tuple[str, str, str, str]] = []
    for cluster_chunks in clusters.values():
        if len(cluster_chunks) > MAX_CLUSTER_SIZE:
            continue
        for i in range(len(cluster_chunks)):
            for j in range(i + 1, len(cluster_chunks)):
                pairs.append(
                    (
                        cluster_chunks[i][0],
                        cluster_chunks[i][1],
                        cluster_chunks[j][0],
                        cluster_chunks[j][1],
                    )
                )
    return pairs
