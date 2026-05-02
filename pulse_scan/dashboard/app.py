"""Pulse Scan — Streamlit dashboard.

Launch via:
    streamlit run pulse_scan/dashboard/app.py -- --data-dir .pulse
or via the CLI:
    pulse dashboard --data-dir .pulse
"""

from __future__ import annotations

import sys
from pathlib import Path

import altair as alt
import pandas as pd
import streamlit as st

from pulse_scan.dashboard.queries import (
    get_available_runs,
    get_corpus_overview,
    get_contradictions,
    get_dedup_groups,
    get_resolution_summary,
    get_staleness_df,
    get_staleness_label_counts,
    get_triage_summary,
    open_connection,
    resolve_contradiction,
)

# ---------------------------------------------------------------------------
# Page config
# ---------------------------------------------------------------------------

st.set_page_config(
    page_title="Pulse Scan",
    page_icon="📡",
    layout="wide",
)

# ---------------------------------------------------------------------------
# Sidebar — data dir + run selection
# ---------------------------------------------------------------------------

def _default_data_dir() -> str:
    for i, arg in enumerate(sys.argv):
        if arg == "--data-dir" and i + 1 < len(sys.argv):
            return sys.argv[i + 1]
    return ".pulse"


with st.sidebar:
    st.title("📡 Pulse Scan")
    data_dir_str = st.text_input("Data directory", value=_default_data_dir())
    data_dir = Path(data_dir_str)

    conn = open_connection(data_dir)
    if conn is None:
        st.error(f"No database at `{data_dir}/pulse.db`.\nRun `pulse scan` first.")
        st.stop()

    runs = get_available_runs(conn)
    if not runs:
        st.warning("No scan runs found. Run `pulse scan` first.")
        st.stop()

    run_id = st.selectbox("Scan run", runs, index=0)
    st.caption(f"Run: `{run_id[:8]}…`")

# ---------------------------------------------------------------------------
# Tabs
# ---------------------------------------------------------------------------

tab_overview, tab_dups, tab_contra, tab_stale = st.tabs(
    ["📊 Overview", "🔁 Duplicates", "⚡ Contradictions", "🕰 Staleness"]
)

# ── Overview ─────────────────────────────────────────────────────────────────

with tab_overview:
    st.header("Corpus overview")
    overview = get_corpus_overview(conn)
    label_counts = get_staleness_label_counts(conn)
    triage = get_triage_summary(conn, run_id)

    c1, c2, c3, c4, c5 = st.columns(5)
    c1.metric("Active chunks", overview["active_chunks"])
    c2.metric("Deleted chunks", overview["deleted_chunks"])
    c3.metric("Dedup groups", overview["dedup_groups"])
    c4.metric("Open contradictions", overview["open_contradictions"])
    c5.metric("Triage scanned", triage["chunks_scanned"])

    st.divider()

    col_l, col_r = st.columns(2)

    with col_l:
        st.subheader("Staleness distribution")
        if any(label_counts.values()):
            df_labels = pd.DataFrame([
                {"label": k, "count": v, "order": i}
                for i, (k, v) in enumerate(label_counts.items())
            ])
            color_map = {
                "fresh": "#2ecc71",
                "aging": "#f39c12",
                "stale": "#e67e22",
                "abandoned": "#e74c3c",
            }
            chart = (
                alt.Chart(df_labels)
                .mark_bar()
                .encode(
                    x=alt.X("label:N", sort=list(label_counts.keys()), title="Label"),
                    y=alt.Y("count:Q", title="Chunks"),
                    color=alt.Color(
                        "label:N",
                        scale=alt.Scale(
                            domain=list(color_map.keys()),
                            range=list(color_map.values()),
                        ),
                        legend=None,
                    ),
                    tooltip=["label", "count"],
                )
                .properties(height=250)
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
    res_summary = get_resolution_summary(conn)
    st.subheader("Contradiction resolution summary")
    r1, r2, r3 = st.columns(3)
    r1.metric("Confirmed", res_summary["confirmed"])
    r2.metric("False positives", res_summary["false_positive"])
    r3.metric("Unresolved", res_summary["unresolved"])

# ── Duplicates ────────────────────────────────────────────────────────────────

with tab_dups:
    st.header("Dedup groups")
    groups = get_dedup_groups(conn)

    if not groups:
        st.info("No dedup groups found in this scan.")
    else:
        st.caption(f"{len(groups)} group(s) found")
        for g in groups:
            channels_str = " + ".join(g["detection_channels"])
            label = f"Group {g['group_id']} — {g['n_members']} members  [{channels_str}]"
            with st.expander(label):
                for m in g["members"]:
                    badge = " ✅ canonical" if m["is_canonical"] else ""
                    st.markdown(
                        f"**{m['chunk_id']}** | `{m['collection']}`{badge}  "
                        f"staleness: `{m['staleness_score']:.3f}`"
                        if m["staleness_score"] is not None
                        else f"**{m['chunk_id']}** | `{m['collection']}`{badge}"
                    )
                    st.text(m["text"] or "(no text)")
                    st.divider()

# ── Contradictions ────────────────────────────────────────────────────────────

with tab_contra:
    st.header("Contradictions")

    filter_col1, filter_col2 = st.columns(2)
    with filter_col1:
        detector_filter = st.selectbox(
            "Detector", ["all", "nli", "numeric", "version"], key="det_filter"
        )
    with filter_col2:
        show_all = st.checkbox("Show resolved contradictions", value=False)

    contradictions = get_contradictions(
        conn,
        run_id=run_id,
        unresolved_only=not show_all,
        detector=detector_filter if detector_filter != "all" else None,
    )

    if not contradictions:
        st.success("No contradictions match the current filters.")
    else:
        st.caption(f"{len(contradictions)} contradiction(s)")
        resolutions_to_save: dict[tuple, str] = {}

        for i, c in enumerate(contradictions):
            key_prefix = f"contra_{i}_{c['chunk_a']}_{c['chunk_b']}"
            header = (
                f"[{c['detector'].upper()}] {c['chunk_a'][:12]}… vs "
                f"{c['chunk_b'][:12]}…  score={c['raw_score']}  "
                f"dir={c['direction']}"
            )
            with st.expander(header, expanded=False):
                col_a, col_b = st.columns(2)
                with col_a:
                    st.markdown(f"**{c['chunk_a']}** (`{c['collection_a']}`)")
                    st.text_area("", value=c["text_a"], height=150, key=f"{key_prefix}_ta",
                                 disabled=True)
                with col_b:
                    st.markdown(f"**{c['chunk_b']}** (`{c['collection_b']}`)")
                    st.text_area("", value=c["text_b"], height=150, key=f"{key_prefix}_tb",
                                 disabled=True)

                if c["user_resolution"]:
                    st.info(f"Resolved: **{c['user_resolution']}**")
                else:
                    res_choice = st.radio(
                        "Resolution",
                        ["(skip)", "confirmed", "false_positive"],
                        horizontal=True,
                        key=f"{key_prefix}_res",
                    )
                    if res_choice != "(skip)":
                        resolutions_to_save[(c["chunk_a"], c["chunk_b"])] = res_choice

        if resolutions_to_save:
            if st.button(f"💾 Save {len(resolutions_to_save)} resolution(s)"):
                for (ca, cb), res in resolutions_to_save.items():
                    resolve_contradiction(data_dir, ca, cb, res)
                st.success(f"Saved {len(resolutions_to_save)} resolution(s).")
                st.rerun()

# ── Staleness ─────────────────────────────────────────────────────────────────

with tab_stale:
    st.header("Staleness scores")

    label_choice = st.selectbox(
        "Filter by label",
        ["all", "abandoned", "stale", "aging", "fresh"],
        key="stale_label",
    )
    df_stale = get_staleness_df(conn, label_filter=label_choice)

    if df_stale.empty:
        st.info("No staleness scores. Run `pulse scan` to completion.")
    else:
        st.caption(f"{len(df_stale)} chunk(s)")
        st.dataframe(
            df_stale[["chunk_id", "collection", "score", "label",
                       "age_decay", "cluster_drift",
                       "contradiction_evidence", "supersession_evidence", "text"]],
            use_container_width=True,
            hide_index=True,
            column_config={
                "score": st.column_config.NumberColumn("Score", format="%.4f"),
                "age_decay": st.column_config.NumberColumn("Age", format="%.3f"),
                "cluster_drift": st.column_config.NumberColumn("Drift", format="%.3f"),
                "contradiction_evidence": st.column_config.NumberColumn("Contra", format="%.3f"),
                "supersession_evidence": st.column_config.NumberColumn("Super", format="%.3f"),
                "text": st.column_config.TextColumn("Text (preview)", max_chars=120),
            },
        )

# Release the read-only lock so pulse scan can acquire a write lock between renders.
conn.close()
