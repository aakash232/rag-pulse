"""Stage 4: NLI Contradiction Detection (Detector A — NLI cross-encoder).

For each cluster, finds candidate pairs with cosine > contradiction_candidate_threshold
(top-k per chunk, dedup-group pairs excluded), then runs a DeBERTa-v3-base NLI
cross-encoder in both ordered directions. Any pair where either direction has
contradiction probability > 0.5 is recorded as a contradiction candidate in cold-start
mode (calibration_state='uncalibrated', calibrated_confidence=NULL).

The predict_fn can be injected for testing to avoid loading the full model.
"""

import json
import random
from collections import defaultdict
from pathlib import Path
from typing import Callable, Optional

import duckdb
import numpy as np
import structlog

log = structlog.get_logger()

_NLI_SCORE_THRESHOLD_DEFAULT = 0.5  # used when no calibrated threshold exists


class NLIContradictionStage:
    def __init__(
        self,
        conn: duckdb.DuckDBPyConnection,
        data_dir: Path,
        scan_run_id: str,
        inference_config=None,  # InferenceConfig or None
        scan_config=None,  # ScanConfig or None
        _predict_fn: Optional[Callable] = None,
    ):
        self.conn = conn
        self.data_dir = data_dir
        self.scan_run_id = scan_run_id
        self._inf_cfg = inference_config
        self._scan_cfg = scan_config
        self.__predict_fn = _predict_fn  # injected for testing; bypasses model load

    # ------------------------------------------------------------------
    # Public interface
    # ------------------------------------------------------------------

    def run(self, allowed_chunk_ids: Optional[set] = None) -> dict:
        """Run NLI on candidate pairs.

        allowed_chunk_ids: if provided (from TriageStage), only chunks in this
        set are used as query points when generating candidate pairs.  Neighbors
        may still be any clustered chunk.  Pass None to scan all chunks (default,
        used in tests and when budget is unlimited).
        """
        threshold = self._get_candidate_threshold()
        nli_threshold = self._get_nli_score_threshold()
        k = self._scan_cfg.contradiction_candidates_per_chunk if self._scan_cfg else 5
        batch_size = self._scan_cfg.nli_batch_size if self._scan_cfg else 64
        max_cluster_size = self._scan_cfg.max_nli_cluster_size if self._scan_cfg else 2000

        rows = self.conn.execute(
            "SELECT chunk_id, cluster_id, text, embedding_offset "
            "FROM chunks "
            "WHERE deleted_at IS NULL AND cluster_id IS NOT NULL AND embedding_offset >= 0 "
            "ORDER BY chunk_id"
        ).fetchall()

        if len(rows) < 2:
            log.info("nli_skipped_too_few_clustered_chunks", n_clustered=len(rows))
            return {"pairs_checked": 0, "contradictions_found": 0}

        meta = json.loads((self.data_dir / "embeddings.meta.json").read_text())
        dim: int = meta["dim"]
        n_allocated: int = meta["n_allocated"]

        mmap = np.memmap(
            self.data_dir / "embeddings.f32.npy",
            dtype=np.float32,
            mode="r",
            shape=(n_allocated, dim),
        )
        chunk_ids = [r[0] for r in rows]
        cluster_ids = [r[1] for r in rows]
        texts = [r[2] for r in rows]
        offsets = [r[3] for r in rows]
        embeddings = np.array(mmap[offsets], dtype=np.float32)
        del mmap

        norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
        norms[norms == 0] = 1.0
        embeddings /= norms

        dedup_pairs = self._load_dedup_pairs()
        candidate_pairs = self._find_candidates(
            chunk_ids,
            cluster_ids,
            texts,
            embeddings,
            dedup_pairs,
            threshold,
            k,
            allowed_query_ids=allowed_chunk_ids,
            max_cluster_size=max_cluster_size,
        )

        if not candidate_pairs:
            log.info("nli_no_candidates_above_threshold", threshold=round(threshold, 4))
            return {"pairs_checked": 0, "contradictions_found": 0}

        # Idempotent: clear previous NLI results for this run before inserting
        self.conn.execute(
            "DELETE FROM contradictions WHERE scan_run_id = ? AND detector = 'nli'",
            [self.scan_run_id],
        )

        predict = self._get_predict_fn()
        n_contradictions = self._run_nli(candidate_pairs, predict, batch_size, nli_threshold)

        log.info(
            "nli_complete",
            pairs_checked=len(candidate_pairs),
            contradictions_found=n_contradictions,
            candidate_threshold=round(threshold, 4),
            nli_score_threshold=round(nli_threshold, 4),
        )
        return {"pairs_checked": len(candidate_pairs), "contradictions_found": n_contradictions}

    # ------------------------------------------------------------------
    # Candidate pair generation
    # ------------------------------------------------------------------

    def _find_candidates(
        self,
        chunk_ids: list[str],
        cluster_ids: list[int],
        texts: list[str],
        embeddings: np.ndarray,
        dedup_pairs: set,
        threshold: float,
        k: int,
        allowed_query_ids: Optional[set] = None,
        max_cluster_size: int = 2000,
    ) -> list[tuple]:
        """Return list of (id_a, text_a, id_b, text_b) unique unordered pairs.

        When allowed_query_ids is provided, only chunks in that set are used as
        query points (a → budget-gated).  Neighbor chunks (b) may be any cluster
        member.
        """
        cluster_to_indices: dict[int, list[int]] = defaultdict(list)
        for idx, cid in enumerate(cluster_ids):
            cluster_to_indices[cid].append(idx)

        seen: set[frozenset] = set()
        pairs = []

        for cid, indices in cluster_to_indices.items():
            if len(indices) < 2:
                continue
            if len(indices) > max_cluster_size:
                log.warning(
                    "nli_cluster_size_capped",
                    cluster_id=cid,
                    original_size=len(indices),
                    capped_to=max_cluster_size,
                )
                indices = random.sample(indices, max_cluster_size)
            embs = embeddings[indices]
            cosines = embs @ embs.T  # pairwise cosines (all normalized)

            for local_i, global_i in enumerate(indices):
                if allowed_query_ids is not None and chunk_ids[global_i] not in allowed_query_ids:
                    continue  # triage: skip non-allowed query chunks
                row_cosines = cosines[local_i]
                ranked = sorted(
                    [
                        (float(row_cosines[local_j]), global_j)
                        for local_j, global_j in enumerate(indices)
                        if local_j != local_i
                    ],
                    reverse=True,
                )[:k]

                for cos_val, global_j in ranked:
                    if cos_val <= threshold:
                        break
                    pair_key = frozenset([chunk_ids[global_i], chunk_ids[global_j]])
                    if pair_key in seen or pair_key in dedup_pairs:
                        continue
                    seen.add(pair_key)
                    pairs.append(
                        (
                            chunk_ids[global_i],
                            texts[global_i],
                            chunk_ids[global_j],
                            texts[global_j],
                        )
                    )

        return pairs

    def _load_dedup_pairs(self) -> set:
        rows = self.conn.execute("SELECT member_chunk_ids FROM dedup_groups").fetchall()
        pairs: set[frozenset] = set()
        for (members_json,) in rows:
            members = json.loads(members_json)
            for i in range(len(members)):
                for j in range(i + 1, len(members)):
                    pairs.add(frozenset([members[i], members[j]]))
        return pairs

    # ------------------------------------------------------------------
    # NLI inference
    # ------------------------------------------------------------------

    def _run_nli(
        self,
        candidate_pairs: list[tuple],
        predict_fn: Callable,
        batch_size: int,
        nli_threshold: float = _NLI_SCORE_THRESHOLD_DEFAULT,
    ) -> int:
        forward_inputs = [(ta, tb) for (_, ta, _, tb) in candidate_pairs]
        backward_inputs = [(tb, ta) for (_, ta, _, tb) in candidate_pairs]

        scores_fwd = self._batch_predict(predict_fn, forward_inputs, batch_size)
        scores_bwd = self._batch_predict(predict_fn, backward_inputs, batch_size)

        n_contradictions = 0
        for i, (id_a, _, id_b, _) in enumerate(candidate_pairs):
            cf = scores_fwd[i].get("contradiction", 0.0)
            cb = scores_bwd[i].get("contradiction", 0.0)

            fwd_hit = cf > nli_threshold
            bwd_hit = cb > nli_threshold

            if not fwd_hit and not bwd_hit:
                continue

            if fwd_hit and bwd_hit:
                direction = "both"
                raw_score = max(cf, cb)
            elif fwd_hit:
                direction = "a->b"
                raw_score = cf
            else:
                direction = "b->a"
                raw_score = cb

            self.conn.execute(
                """INSERT INTO contradictions
                   (chunk_a, chunk_b, detector, raw_score, calibrated_confidence,
                    calibration_state, direction, scan_run_id)
                   VALUES (?, ?, 'nli', ?, NULL, 'uncalibrated', ?, ?)""",
                [id_a, id_b, float(raw_score), direction, self.scan_run_id],
            )
            n_contradictions += 1

        return n_contradictions

    @staticmethod
    def _batch_predict(
        predict_fn: Callable,
        pairs: list[tuple[str, str]],
        batch_size: int,
    ) -> list[dict]:
        results = []
        for i in range(0, len(pairs), batch_size):
            results.extend(predict_fn(pairs[i : i + batch_size]))
        return results

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _get_candidate_threshold(self) -> float:
        row = self.conn.execute(
            "SELECT contradiction_candidate_threshold FROM calibration ORDER BY rowid DESC LIMIT 1"
        ).fetchone()
        if row is None:
            raise RuntimeError("No calibration found. Run 'pulse calibrate' first.")
        return float(row[0])

    def _get_nli_score_threshold(self) -> float:
        from pulse_scan.stages.stage05_calibrate import load_nli_score_threshold

        return load_nli_score_threshold(self.conn)

    def _get_predict_fn(self) -> Callable:
        if self.__predict_fn is not None:
            return self.__predict_fn
        return self._load_pipeline()

    def _load_pipeline(self) -> Callable:
        model_name = self._inf_cfg.nli_model
        device_str = self._inf_cfg.device
        if device_str == "cuda":
            device = 0
        elif device_str == "mps":
            device = "mps"
        else:
            device = -1

        from transformers import pipeline as hf_pipeline

        pipe = hf_pipeline(
            "text-classification",
            model=model_name,
            device=device,
            top_k=None,
        )

        # Fail fast if this model doesn't expose a 'contradiction' label.
        # Standard NLI cross-encoders do; models with LABEL_0/1/2 style labels
        # would silently return 0.0 for every pair without this check.
        id2label: dict = pipe.model.config.id2label
        if "contradiction" not in {v.lower() for v in id2label.values()}:
            raise ValueError(
                f"NLI model '{model_name}' does not expose a 'contradiction' label. "
                f"Found: {sorted(id2label.values())}. "
                "Set inference.nli_model to a standard NLI cross-encoder "
                "(e.g. cross-encoder/nli-deberta-v3-base)."
            )

        def predict(pairs: list[tuple[str, str]]) -> list[dict]:
            inputs = [{"text": a, "text_pair": b} for a, b in pairs]
            results = pipe(inputs)
            out = []
            for result in results:
                scores = {r["label"].lower(): r["score"] for r in result}
                out.append(scores)
            return out

        return predict
