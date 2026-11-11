"""Dashboard — home page with processing history and quick links."""

from __future__ import annotations

import pandas as pd
import streamlit as st

from mathematical_engine.history import (
    filtered_by_operation,
    recent,
    summarize,
    to_rows,
)
from mathematical_engine.trace_manager import TraceManager


st.set_page_config(page_title="Dashboard · Chest X-ray Lab",
                   page_icon="📊", layout="wide")

if "trace_manager" not in st.session_state:
    st.session_state.trace_manager = TraceManager()

tm: TraceManager = st.session_state.trace_manager
gray = st.session_state.get("gray_image")

st.title("📊 Dashboard")

# ---------------------------------------------------------------------------
# Status bar
# ---------------------------------------------------------------------------
if gray is None:
    st.warning(
        "⬅️ No image loaded yet. Upload image first **main page** "
        "to begin."
    )
else:
    info = st.session_state.get("metadata")
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Image", f"{info.height}×{info.width}"
              if info else "?")
    c2.metric("dtype", info.array_dtype if info else "?")
    c3.metric("Mean", f"{info.intensity_mean:.1f}"
              if info else "?")
    c4.metric("Range", f"{info.intensity_min}–{info.intensity_max}"
              if info else "?")

# ---------------------------------------------------------------------------
# Traces summary
# ---------------------------------------------------------------------------
st.markdown("## 🧾 Processing history")
summary = summarize(tm.traces)
m1, m2, m3, m4 = st.columns(4)
m1.metric("Operations", summary["count"])
m2.metric("Total time", f"{summary['total_time_ms']:.1f} ms")
m3.metric("Mean time", f"{summary['mean_time_ms']:.2f} ms")
m4.metric("Last op", summary["last_operation"] or "—")

if summary["count"] == 0:
    st.info(
        "No operations yet. Try one of the pages below to get started:"
    )
else:
    with st.expander("All traces (most recent first)", expanded=False):
        rows = to_rows(recent(tm.traces, n=100))
        st.dataframe(pd.DataFrame(rows), use_container_width=True)


# ---------------------------------------------------------------------------
# Quick links
# ---------------------------------------------------------------------------
st.markdown("## 🚀 Quick links")
st.caption("Navigate using the sidebar, or click any of the buttons below.")

c1, c2, c3 = st.columns(3)
c1.markdown(
    "**🔢 Matrix Explorer**  \n"
    "Pixels as numbers, transpose, row/col inspection"
)
c2.markdown(
    "**✨ Enhancement**  \n"
    "Brightness, contrast, histogram equalization, CLAHE"
)
c3.markdown(
    "**🧹 Noise Reduction**  \n"
    "Mean, Gaussian, median, custom kernels with pixel trace"
)

c4, c5, c6 = st.columns(3)
c4.markdown(
    "**📐 Edge Detection**  \n"
    "Sobel X/Y, gradient magnitude, Laplacian, Canny"
)
c5.markdown(
    "**🧬 PCA Analysis**  \n"
    "Eigen-analysis over a reference stack"
)
c6.markdown(
    "**🗜️ SVD Compression**  \n"
    "Rank-k approximation with error metrics"
)


# ---------------------------------------------------------------------------
# Recent traces with click-through detail
# ---------------------------------------------------------------------------
if summary["count"] > 0:
    st.markdown("## 🧮 Inspect a recent computation")
    recent_traces = recent(tm.traces, n=10)
    labels = [
        f"{t.operation_id} · {t.operation} · {t.execution_time_ms:.2f} ms"
        for t in recent_traces
    ]
    idx = st.selectbox("Select a trace", options=range(len(labels)),
                       format_func=lambda i: labels[i], key="dash_trace")
    selected = recent_traces[idx]
    st.json(selected.to_dict())