"""Pulse Scan — Streamlit dashboard.

Launch via:
    streamlit run pulse_scan/dashboard/app.py -- --data-dir .pulse
or via the CLI:
    pulse dashboard --data-dir .pulse
"""

from __future__ import annotations

import sys
from pathlib import Path

import streamlit as st

from pulse_scan.dashboard.components import (
    render_contradictions,
    render_duplicates,
    render_overview,
    render_staleness,
)
from pulse_scan.dashboard.queries import get_available_runs, open_connection

# ---------------------------------------------------------------------------
# Page config
# ---------------------------------------------------------------------------

st.set_page_config(page_title="Pulse Scan", page_icon="📡", layout="wide")

# ---------------------------------------------------------------------------
# Sidebar — data directory + run selection
# ---------------------------------------------------------------------------

def _default_data_dir() -> str:
    for i, arg in enumerate(sys.argv):
        if arg == "--data-dir" and i + 1 < len(sys.argv):
            return sys.argv[i + 1]
    return ".pulse"


with st.sidebar:
    st.title("📡 Pulse Scan")
    data_dir = Path(st.text_input("Data directory", value=_default_data_dir()))

    conn = open_connection(data_dir)
    if conn is None:
        st.error(f"No database at `{data_dir}/pulse.db`. Run `pulse scan` first.")
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

with tab_overview:
    render_overview(conn, run_id)

with tab_dups:
    render_duplicates(conn)

with tab_contra:
    render_contradictions(conn, run_id, data_dir)

with tab_stale:
    render_staleness(conn)

# Release the read-only lock so pulse scan can acquire a write lock between renders.
conn.close()
