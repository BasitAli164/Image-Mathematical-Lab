"""Matrix Explorer page — uses the shared 3-level workspace (Phase 5)."""

from __future__ import annotations

import traceback

import matplotlib.pyplot as plt
import numpy as np
import streamlit as st

from core.matrix_operations import (
    add_matrices,
    elementwise_multiply,
    extract_block,
    matrix_statistics,
    transpose_matrix,
)
from mathematical_engine.explanation_templates import (
    explain_add,
    explain_elementwise,
    explain_mean,
    explain_transpose,
)
from mathematical_engine.matrix_inspector import (
    neighborhood,
    row_preview,
    column_preview,
)
from mathematical_engine.numerical_calculator import safe_stats
from mathematical_engine.trace_manager import TraceManager
from visualization.math_workspace import render as render_math
from visualization.comparison import side_by_side


st.set_page_config(page_title="Matrix Explorer · Image Mathematical Lab",
                   page_icon="🔢", layout="wide")

if "trace_manager" not in st.session_state:
    st.session_state.trace_manager = TraceManager()

tm: TraceManager = st.session_state.trace_manager
gray = st.session_state.get("gray_image")

st.title("🔢 Matrix Explorer")
st.caption("Every pixel is a number. Explore the matrix.")

if gray is None:
    st.warning("⬅️ Please upload an image on the main page first.")
    st.stop()


left, right = st.columns([1.15, 1.0], gap="large")

# ---------------------------------------------------------------------------
# LEFT: previews + operation controls
# ---------------------------------------------------------------------------
with left:
    st.subheader("🖼️ Image & matrix preview")
    tab_img, tab_block, tab_hist = st.tabs(
        ["Image", "Matrix block", "Histogram"]
    )

    with tab_img:
        st.image(gray, caption=f"Grayscale {gray.shape}",
                 use_column_width=True, clamp=True)

    with tab_block:
        r0 = st.number_input("Origin row", 0, gray.shape[0] - 1, 0,
                             step=1, key="me_r0")
        c0 = st.number_input("Origin col", 0, gray.shape[1] - 1, 0,
                             step=1, key="me_c0")
        bh = st.number_input("Block height", 1,
                             min(32, gray.shape[0]), min(8, gray.shape[0]),
                             step=1, key="me_bh")
        bw = st.number_input("Block width", 1,
                             min(32, gray.shape[1]), min(8, gray.shape[1]),
                             step=1, key="me_bw")
        try:
            block = extract_block(gray, int(r0), int(c0),
                                  int(bh), int(bw))
            st.caption(
                f"Block at ({int(r0)}, {int(c0)}) "
                f"size {block.shape[0]}×{block.shape[1]}"
            )
            st.dataframe(block, use_container_width=False)
        except Exception as exc:
            st.error(f"Block extraction failed: {exc}")

    with tab_hist:
        fig, ax = plt.subplots(figsize=(5, 3))
        ax.hist(gray.ravel(), bins=64, range=(0, 255), color="#0B6E99")
        ax.set_xlabel("Intensity")
        ax.set_ylabel("Count")
        ax.set_title("Intensity histogram")
        st.pyplot(fig, use_container_width=True)
        plt.close(fig)

    st.markdown("### ⚡ Experiment")
    op = st.selectbox(
        "Operation",
        ["Transpose", "Inspect row", "Inspect column",
         "Pixel neighborhood", "Matrix statistics",
         "Add (A + Aᵀ cropped)", "Element-wise multiply (A ⊙ Aᵀ cropped)"],
        key="me_op",
    )

    params: dict = {}
    if op == "Inspect row":
        params["row"] = st.number_input(
            "Row index", 0, gray.shape[0] - 1, gray.shape[0] // 2,
            step=1, key="me_row_idx",
        )
    elif op == "Inspect column":
        params["col"] = st.number_input(
            "Column index", 0, gray.shape[1] - 1, gray.shape[1] // 2,
            step=1, key="me_col_idx",
        )
    elif op == "Pixel neighborhood":
        params["row"] = st.number_input(
            "Pixel row", 0, gray.shape[0] - 1, gray.shape[0] // 2,
            step=1, key="me_px_row",
        )
        params["col"] = st.number_input(
            "Pixel col", 0, gray.shape[1] - 1, gray.shape[1] // 2,
            step=1, key="me_px_col",
        )
        params["size"] = st.selectbox(
            "Neighborhood size", [3, 5, 7], index=0,
            key="me_px_size",
        )

    run = st.button("Run", type="primary", key="me_run")


# ---------------------------------------------------------------------------
# Execute + record trace
# ---------------------------------------------------------------------------
if run:
    try:
        if op == "Transpose":
            out = transpose_matrix(gray)
            derivation = explain_transpose(gray, out)
            key = "transpose"
        elif op == "Inspect row":
            row = gray[int(params["row"]), :]
            out = row.reshape(1, -1)
            derivation = explain_mean(row)
            key = "mean"
        elif op == "Inspect column":
            col = gray[:, int(params["col"])]
            out = col.reshape(-1, 1)
            derivation = explain_mean(col)
            key = "mean"
        elif op == "Pixel neighborhood":
            nb = neighborhood(
                gray, int(params["row"]), int(params["col"]),
                size=int(params["size"]),
            )
            out = nb
            derivation = explain_mean(nb)
            key = "mean"
        elif op == "Matrix statistics":
            out = np.array(matrix_statistics(gray)["mean"])
            derivation = explain_mean(gray)
            key = "mean"
        elif op == "Add (A + Aᵀ cropped)":
            n = min(gray.shape)
            A = gray[:n, :n]
            At = transpose_matrix(A)
            out = add_matrices(A, At)
            derivation = explain_add(A, At, out)
            key = "add"
        else:  # Element-wise
            n = min(gray.shape)
            A = gray[:n, :n]
            At = transpose_matrix(A)
            out = elementwise_multiply(A, At)
            derivation = explain_elementwise(A, At, out)
            key = "elementwise_multiply"

        tr = tm.record(
            operation=key,
            input_array=gray,
            parameters={**params, "mode": op},
            derivation=derivation,
            output=out,
            validation={"ok": True, "warnings": [], "errors": []},
        )
        st.session_state["me_last_trace"] = tr.to_dict()
        if op == "Transpose":
            st.session_state["me_transposed"] = out
    except Exception as exc:
        st.error(f"❌ Failed: {exc}")
        st.code(traceback.format_exc())


last = st.session_state.get("me_last_trace")
transposed = st.session_state.get("me_transposed")


# ---------------------------------------------------------------------------
# LEFT: rendered output
# ---------------------------------------------------------------------------
with left:
    if transposed is not None and last and last.get("operation") == "transpose":
        st.markdown("### 🔄 Transpose comparison")
        side_by_side(gray, transposed,
                     labels=("Original A", "Transpose Aᵀ"))


# ---------------------------------------------------------------------------
# RIGHT: unified mathematical workspace
# ---------------------------------------------------------------------------
with right:
    st.subheader("🧮 Mathematical workspace")
    if last is None:
        st.info(
            "Choose an operation, click **Run**, then the three levels of "
            "mathematical transparency appear here."
        )
    else:
        render_math(last, key_prefix="me")