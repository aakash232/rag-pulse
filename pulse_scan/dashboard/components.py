"""Reusable rendering components for the Pulse Scan Streamlit dashboard.

Each public `render_*` function owns exactly one tab's visual output.
Private `_render_*` helpers handle individual cards/rows within a tab.
Constants at the top are the single source of truth for labels, colors, and copy.
"""

from __future__ import annotations

from pathlib import Path

import altair as alt
import pandas as pd
import streamlit as st

from pulse_scan.dashboard.queries import (
    get_contradictions,
    get_contradictions_count,
    get_corpus_overview,
    get_dedup_groups,
    get_dedup_groups_count,
    get_resolution_summary,
    get_staleness_count,
    get_staleness_df,
    get_staleness_label_counts,
    get_triage_summary,
    resolve_contradiction,
)

# ---------------------------------------------------------------------------
# Constants — single source of truth for labels, colors, copy
# ---------------------------------------------------------------------------

LABEL_COLOR = {
    "fresh":     "#2ecc71",
    "aging":     "#f39c12",
    "stale":     "#e67e22",
    "abandoned": "#e74c3c",
}

LABEL_ICON = {
    "fresh":     "🟢",
    "aging":     "🟡",
    "stale":     "🟠",
    "abandoned": "🔴",
}

LABEL_DESC = {
    "fresh":     "Recently updated. No significant decay signals.",
    "aging":     "Getting old. Worth monitoring but not yet urgent.",
    "stale":     "Likely outdated. Review and update recommended.",
    "abandoned": "Very old, contradicted, or semantically drifted. Probable dead content.",
}

# (display name, tooltip explanation) for each staleness component
COMPONENT_META: dict[str, tuple[str, str]] = {
    "age_decay": (
        "Age decay",
        "Content age relative to this collection's configured half-life. "
        "A chunk past its half-life scores close to 1.0 here.",
    ),
    "cluster_drift": (
        "Semantic drift",
        "How far this chunk has drifted from its topic-cluster centroid. "
        "High values mean the chunk is now a semantic outlier in its cluster.",
    ),
    "contradiction_evidence": (
        "Contradiction evidence",
        "Degree of involvement in NLI, numeric, or version contradictions "
        "detected during this scan.",
    ),
    "supersession_evidence": (
        "Supersession evidence",
        "Extent to which newer chunks appear to replace or supersede this content.",
    ),
}

DETECTOR_ICON = {"nli": "🧠", "numeric": "🔢", "version": "🏷️"}
DIRECTION_LABEL = {"a->b": "→", "b->a": "←", "both": "↔"}

LABEL_FILTER_OPTIONS: dict[str, str | None] = {
    "All":          None,
    "🔴 Abandoned": "abandoned",
    "🟠 Stale":     "stale",
    "🟡 Aging":     "aging",
    "🟢 Fresh":     "fresh",
}

VERDICT_OPTIONS: dict[str, str | None] = {
    "(skip for now)":                        None,
    "✅ Confirmed — real contradiction":     "confirmed",
    "❌ False positive — not a real conflict": "false_positive",
}

# Items rendered per page — keep low enough that Streamlit stays responsive.
PAGE_SIZE_CONTRA = 25
PAGE_SIZE_STALE = 50
PAGE_SIZE_DUPS = 25

# ---------------------------------------------------------------------------
# Pagination helper
# ---------------------------------------------------------------------------

def _pagination_nav(state_key: str, total: int, page_size: int, key_suffix: str = "") -> int:
    """Renders Prev / page-info / Next row. Returns the current 0-based page.

    Pass key_suffix="_bot" for a second nav rendered below the same list to
    avoid Streamlit duplicate-key errors.
    """
    total_pages = max(1, (total + page_size - 1) // page_size)
    page = int(st.session_state.get(state_key, 0))
    page = max(0, min(page, total_pages - 1))
    st.session_state[state_key] = page

    c_prev, c_info, c_next = st.columns([1, 4, 1])
    with c_prev:
        if st.button("← Prev", key=f"{state_key}_prev{key_suffix}", disabled=(page == 0)):
            st.session_state[state_key] -= 1
            st.rerun()
    with c_info:
        st.caption(f"Page {page + 1} / {total_pages}  ·  {total:,} items total")
    with c_next:
        if st.button("Next →", key=f"{state_key}_next{key_suffix}", disabled=(page >= total_pages - 1)):
            st.session_state[state_key] += 1
            st.rerun()

    return st.session_state[state_key]


# ---------------------------------------------------------------------------
# Low-level visual helpers
# ---------------------------------------------------------------------------

def _score_bar_html(score: float, color: str) -> str:
    pct = min(100, int(score * 100))
    return (
        f'<div style="background:#e0e0e0;border-radius:6px;height:14px;margin:6px 0 2px">'
        f'<div style="background:{color};width:{pct}%;height:14px;border-radius:6px"></div>'
        f"</div>"
        f'<small style="color:#888">{score:.3f} / 1.0</small>'
    )


def _resolution_badge(resolution: str) -> None:
    if resolution == "confirmed":
        st.success("✅ Confirmed as a real contradiction")
    else:
        st.warning("❌ Marked as a false positive")


# ---------------------------------------------------------------------------
# Staleness tab
# ---------------------------------------------------------------------------

def _render_staleness_legend() -> None:
    with st.expander("ℹ️ How to read staleness scores", expanded=False):
        st.markdown(
            "Each chunk receives a **staleness score** from **0.0** (healthy) to **1.0** (dead), "
            "computed as a weighted combination of four components:\n"
        )
        for key, (name, desc) in COMPONENT_META.items():
            st.markdown(f"- **{name}** — {desc}")

        st.markdown("\n**Labels** are assigned by score range:\n")
        for label in ["fresh", "aging", "stale", "abandoned"]:
            icon = LABEL_ICON[label]
            st.markdown(f"- {icon} **{label.capitalize()}** — {LABEL_DESC[label]}")


def _render_component_breakdown(chunk: dict) -> None:
    st.markdown("**Score breakdown** — what drove this score:")
    cols = st.columns(4)
    for col, (key, (name, help_text)) in zip(cols, COMPONENT_META.items()):
        val = chunk.get(key) or 0.0
        col.metric(name, f"{val:.3f}", help=help_text)


def _render_staleness_chunk(chunk: dict) -> None:
    label = chunk.get("label") or "unknown"
    score = chunk.get("score") or 0.0
    icon = LABEL_ICON.get(label, "⚪")
    color = LABEL_COLOR.get(label, "#aaa")

    header = (
        f"{icon} **{chunk['chunk_id']}** &nbsp;·&nbsp; "
        f"`{chunk['collection']}` &nbsp;·&nbsp; "
        f"score **{score:.3f}**"
    )
    with st.expander(header):
        st.markdown(
            _score_bar_html(score, color),
            unsafe_allow_html=True,
        )
        st.divider()
        _render_component_breakdown(chunk)
        if chunk.get("text"):
            st.divider()
            st.markdown("**Content preview**")
            st.caption(chunk["text"][:400])


def render_staleness(conn) -> None:
    st.header("Staleness scores")
    _render_staleness_legend()

    label_counts = get_staleness_label_counts(conn)

    c1, c2, c3, c4 = st.columns(4)
    for col, label in zip([c1, c2, c3, c4], ["fresh", "aging", "stale", "abandoned"]):
        col.metric(
            f"{LABEL_ICON[label]} {label.capitalize()}",
            label_counts.get(label, 0),
            help=LABEL_DESC[label],
        )

    st.divider()

    label_choice = st.radio(
        "Show",
        list(LABEL_FILTER_OPTIONS.keys()),
        horizontal=True,
        key="stale_label_filter",
    )
    filter_val = LABEL_FILTER_OPTIONS[label_choice]

    # Reset page when filter changes.
    filter_sig = filter_val
    if st.session_state.get("stale_filter_sig") != filter_sig:
        st.session_state["stale_filter_sig"] = filter_sig
        st.session_state["stale_page"] = 0

    total = get_staleness_count(conn, label_filter=filter_val)
    if total == 0:
        st.info("No chunks match this filter. Run `pulse scan` to populate staleness scores.")
        return

    page = _pagination_nav("stale_page", total, PAGE_SIZE_STALE)
    offset = page * PAGE_SIZE_STALE

    df_stale = get_staleness_df(conn, label_filter=filter_val, limit=PAGE_SIZE_STALE, offset=offset)
    st.divider()
    for _, row in df_stale.iterrows():
        _render_staleness_chunk(row.to_dict())

    # Bottom nav for long pages
    if total > PAGE_SIZE_STALE:
        st.divider()
        _pagination_nav("stale_page", total, PAGE_SIZE_STALE, key_suffix="_bot")


# ---------------------------------------------------------------------------
# Overview tab
# ---------------------------------------------------------------------------

def render_overview(conn, run_id: str) -> None:
    st.header("Corpus overview")
    overview = get_corpus_overview(conn)
    label_counts = get_staleness_label_counts(conn)
    triage = get_triage_summary(conn, run_id)
    res = get_resolution_summary(conn)

    active = overview["active_chunks"] or 1
    pct_fresh = round(100 * label_counts.get("fresh", 0) / active)

    c1, c2, c3, c4, c5 = st.columns(5)
    c1.metric("Active chunks", overview["active_chunks"])
    c2.metric("% Fresh", f"{pct_fresh}%", help="Fraction of chunks with no decay signals.")
    c3.metric(
        "Dedup groups", overview["dedup_groups"],
        help="Groups of near-duplicate chunks. Each group should ideally have one canonical version.",
    )
    c4.metric(
        "Open contradictions", overview["open_contradictions"],
        help="Contradiction pairs not yet reviewed in the dashboard.",
    )
    c5.metric(
        "NLI scanned", triage["chunks_scanned"],
        help="Chunks evaluated for contradictions in this run (budget-gated).",
    )

    st.divider()
    col_l, col_r = st.columns(2)

    with col_l:
        st.subheader("Staleness distribution")
        if any(label_counts.values()):
            df = pd.DataFrame([
                {"label": k, "count": v}
                for k, v in label_counts.items()
                if v > 0
            ])
            chart = (
                alt.Chart(df)
                .mark_bar(cornerRadiusTopLeft=4, cornerRadiusTopRight=4)
                .encode(
                    x=alt.X(
                        "label:N",
                        sort=["fresh", "aging", "stale", "abandoned"],
                        title=None,
                        axis=alt.Axis(labelAngle=0),
                    ),
                    y=alt.Y("count:Q", title="Chunks"),
                    color=alt.Color(
                        "label:N",
                        scale=alt.Scale(
                            domain=list(LABEL_COLOR.keys()),
                            range=list(LABEL_COLOR.values()),
                        ),
                        legend=None,
                    ),
                    tooltip=[
                        alt.Tooltip("label:N", title="Label"),
                        alt.Tooltip("count:Q", title="Chunks"),
                    ],
                )
                .properties(height=220)
            )
            st.altair_chart(chart, use_container_width=True)
        else:
            st.info("No staleness scores yet — run `pulse scan` to completion.")

    with col_r:
        st.subheader("Collections")
        if overview["collections"]:
            df_col = pd.DataFrame(overview["collections"])
            df_col.columns = ["Collection", "Active chunks"]
            st.dataframe(df_col, use_container_width=True, hide_index=True)
        else:
            st.info("No collections found.")

    st.divider()
    st.subheader("Contradiction review progress")
    st.caption(
        "Reviewing contradictions is optional but improves detection accuracy — "
        "confirmed/rejected labels re-tune the NLI threshold on the next scan."
    )
    r1, r2, r3 = st.columns(3)
    r1.metric("✅ Confirmed", res["confirmed"], help="Marked as real contradictions.")
    r2.metric(
        "❌ False positives", res["false_positive"],
        help="Marked as false alarms. Used to raise the NLI confidence threshold.",
    )
    r3.metric("⏳ Unreviewed", res["unresolved"])


# ---------------------------------------------------------------------------
# Duplicates tab
# ---------------------------------------------------------------------------

def _render_dup_group(g: dict) -> None:
    channels_str = " · ".join(g["detection_channels"])
    header = (
        f"🔁 Group {g['group_id']} &nbsp;·&nbsp; "
        f"{g['n_members']} members &nbsp;·&nbsp; "
        f"detected via: {channels_str}"
    )
    with st.expander(header):
        for m in g["members"]:
            role = "✅ canonical" if m["is_canonical"] else "duplicate"
            score_str = (
                f"staleness {m['staleness_score']:.3f}"
                if m["staleness_score"] is not None
                else "no staleness score"
            )
            st.markdown(
                f"**{m['chunk_id']}** &nbsp; `{m['collection']}` "
                f"&nbsp;—&nbsp; {role} &nbsp;·&nbsp; {score_str}"
            )
            st.caption(m["text"] or "(no text)")
            st.divider()


def render_duplicates(conn) -> None:
    st.header("Duplicate groups")
    total = get_dedup_groups_count(conn)

    if total == 0:
        st.info("No duplicate groups found in this scan.")
        return

    st.caption(
        f"{total:,} group(s) detected. "
        "Each group contains near-identical chunks — keep the canonical one, remove the rest."
    )

    page = _pagination_nav("dup_page", total, PAGE_SIZE_DUPS)
    offset = page * PAGE_SIZE_DUPS

    groups = get_dedup_groups(conn, limit=PAGE_SIZE_DUPS, offset=offset)
    st.divider()
    for g in groups:
        _render_dup_group(g)

    if total > PAGE_SIZE_DUPS:
        st.divider()
        _pagination_nav("dup_page", total, PAGE_SIZE_DUPS, key_suffix="_bot")


# ---------------------------------------------------------------------------
# Contradictions tab
# ---------------------------------------------------------------------------

def _make_verdict_callback(chunk_a: str, chunk_b: str, radio_key: str):
    """Returns an on_change callback that stages the verdict in session_state."""
    def _cb() -> None:
        val = st.session_state.get(radio_key)
        resolution = VERDICT_OPTIONS.get(val)
        pv: dict = st.session_state.setdefault("pending_verdicts", {})
        pair = (chunk_a, chunk_b)
        if resolution:
            pv[pair] = resolution
        elif pair in pv:
            del pv[pair]
    return _cb


def _render_contradiction_card(c: dict, key_prefix: str) -> None:
    """Renders one contradiction card; verdict is staged in session_state via on_change."""
    icon = DETECTOR_ICON.get(c["detector"], "⚡")
    direction = DIRECTION_LABEL.get(c["direction"], c["direction"])
    score_pct = int((c["raw_score"] or 0) * 100)
    header = (
        f"{icon} {c['detector'].upper()} &nbsp;·&nbsp; "
        f"{c['chunk_a'][:22]} {direction} {c['chunk_b'][:22]} "
        f"&nbsp;·&nbsp; confidence {score_pct}%"
    )
    with st.expander(header):
        col_a, col_b = st.columns(2)
        with col_a:
            st.markdown(f"**{c['chunk_a']}** &nbsp; `{c['collection_a']}`")
            st.text_area("", value=c["text_a"], height=140, key=f"{key_prefix}_ta", disabled=True)
        with col_b:
            st.markdown(f"**{c['chunk_b']}** &nbsp; `{c['collection_b']}`")
            st.text_area("", value=c["text_b"], height=140, key=f"{key_prefix}_tb", disabled=True)

        if c["user_resolution"]:
            _resolution_badge(c["user_resolution"])
            return

        radio_key = f"{key_prefix}_res"
        st.radio(
            "Your verdict",
            list(VERDICT_OPTIONS.keys()),
            horizontal=True,
            key=radio_key,
            on_change=_make_verdict_callback(c["chunk_a"], c["chunk_b"], radio_key),
        )


def render_contradictions(conn, run_id: str, data_dir: Path) -> None:
    st.header("Contradictions")
    st.caption(
        "Pairs the scanner flagged as potentially contradictory. "
        "Your verdicts feed back into NLI threshold calibration on the next scan."
    )

    # --- Filters ---
    f1, f2 = st.columns(2)
    with f1:
        detector_filter = st.selectbox(
            "Detector", ["all", "nli", "numeric", "version"], key="det_filter"
        )
    with f2:
        show_all = st.checkbox("Show resolved contradictions", value=False)

    # Reset to page 0 when any filter changes.
    filter_sig = (run_id, detector_filter, show_all)
    if st.session_state.get("contra_filter_sig") != filter_sig:
        st.session_state["contra_filter_sig"] = filter_sig
        st.session_state["contra_page"] = 0

    unresolved_only = not show_all
    det = detector_filter if detector_filter != "all" else None

    total = get_contradictions_count(conn, run_id=run_id, unresolved_only=unresolved_only, detector=det)

    if total == 0:
        st.success("No contradictions match the current filters.")
        return

    # --- Show saved-verdicts success toast (set by Save button on previous run) ---
    if "contra_save_count" in st.session_state:
        n_saved = st.session_state.pop("contra_save_count")
        st.toast(f"Saved {n_saved} verdict(s).", icon="✅")

    # --- Pending-verdict save bar ---
    pv: dict = st.session_state.setdefault("pending_verdicts", {})
    n_pending = len(pv)
    if n_pending:
        save_col, info_col = st.columns([1, 4])
        with save_col:
            if st.button(f"💾 Save {n_pending} verdict(s)", type="primary"):
                # Close the read-only conn before opening a RW one; st.rerun()
                # immediately follows so a fresh conn is opened on the next pass.
                conn.close()
                for (ca, cb), res in list(pv.items()):
                    resolve_contradiction(data_dir, ca, cb, res)
                st.session_state["pending_verdicts"] = {}
                st.session_state["contra_save_count"] = n_pending
                st.rerun()
        with info_col:
            st.caption(
                f"{n_pending} verdict(s) staged across all pages — "
                "navigating pages does **not** lose them."
            )

    # --- Pagination + items ---
    page = _pagination_nav("contra_page", total, PAGE_SIZE_CONTRA)
    offset = page * PAGE_SIZE_CONTRA

    contradictions = get_contradictions(
        conn,
        run_id=run_id,
        unresolved_only=unresolved_only,
        detector=det,
        limit=PAGE_SIZE_CONTRA,
        offset=offset,
    )

    st.divider()
    for i, c in enumerate(contradictions):
        global_idx = offset + i
        key_prefix = f"contra_{global_idx}_{c['chunk_a']}_{c['chunk_b']}"
        _render_contradiction_card(c, key_prefix)

    # Bottom nav
    if total > PAGE_SIZE_CONTRA:
        st.divider()
        _pagination_nav("contra_page", total, PAGE_SIZE_CONTRA, key_suffix="_bot")
