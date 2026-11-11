"""Edge Detection page — Phase 5 version with 3-level workspace."""

from __future__ import annotations

import time
import traceback

import numpy as np
import streamlit as st

from core.edge_detection import (
    LAPLACIAN_4, LAPLACIAN_8, SOBEL_X, SOBEL_Y,
    canny, edge_density, gradient_magnitude,
    laplacian, laplacian_uint8, sobel_uint8, sobel_x, sobel_y,
)
from mathematical_engine.pixel_trace import trace_sobel_pixel
from mathematical_engine.trace_manager import TraceManager
from visualization.comparison import side_by_side
from visualization.math_workspace import render as render_math


st.set_page_config(page_title="Edge Detection · Image Mathematical Lab",
                   page_icon="📐", layout="wide")

if "trace_manager" not in st.session_state:
    st.session_state.trace_manager = TraceManager()

tm: TraceManager = st.session_state.trace_manager
gray = st.session_state.get("gray_image")

st.title("📐 Edge Detection")
st.caption("Gradient-based edge detection with explicit kernels.")

if gray is None:
    st.warning("⬅️ Upload an image on the main page first.")
    st.stop()


left, right = st.columns([1.15, 1.0], gap="large")


# ---------------------------------------------------------------------------
# LEFT: controls
# ---------------------------------------------------------------------------
with left:
    st.subheader("🖼️ Preview")
    tab_o, tab_e, tab_c = st.tabs(["Original", "Edges", "Compare"])
    with tab_o:
        st.image(gray, caption=f"Original {gray.shape}",
                 use_column_width=True, clamp=True)

    st.markdown("### ⚡ Experiment")
    op = st.selectbox(
        "Operation",
        ["Sobel X", "Sobel Y", "Gradient Magnitude",
         "Laplacian", "Canny"],
        key="ed_op",
    )

    params: dict = {}
    if op == "Canny":
        params["low"] = float(st.slider(
            "low threshold", 0.0, 255.0, 50.0, 5.0, key="ed_cl"
        ))
        params["high"] = float(st.slider(
            "high threshold", 0.0, 255.0, 150.0, 5.0, key="ed_ch"
        ))
        if params["low"] >= params["high"]:
            st.warning("low threshold must be < high threshold.")
    elif op == "Laplacian":
        params["variant"] = st.selectbox(
            "variant", ["8", "4"], index=0, key="ed_lap"
        )

    trace_row = st.number_input(
        "trace pixel row", 0, gray.shape[0] - 1, gray.shape[0] // 2,
        step=1, key="ed_tr_r",
    )
    trace_col = st.number_input(
        "trace pixel col", 0, gray.shape[1] - 1, gray.shape[1] // 2,
        step=1, key="ed_tr_c",
    )

    run = st.button("Detect", type="primary", key="ed_run")


# ---------------------------------------------------------------------------
# Execute
# ---------------------------------------------------------------------------
if run:
    try:
        t0 = time.perf_counter()
        intermediate: list[dict] = []

        if op == "Sobel X":
            signed = sobel_x(gray)
            out = sobel_uint8(gray, axis="x")
            key = "sobel_x"
            kernel = SOBEL_X
            try:
                pt = trace_sobel_pixel(
                    gray, int(trace_row), int(trace_col), axis="x"
                )
                intermediate.append(pt)
            except Exception:
                pass
        elif op == "Sobel Y":
            signed = sobel_y(gray)
            out = sobel_uint8(gray, axis="y")
            key = "sobel_y"
            kernel = SOBEL_Y
            try:
                pt = trace_sobel_pixel(
                    gray, int(trace_row), int(trace_col), axis="y"
                )
                intermediate.append(pt)
            except Exception:
                pass
        elif op == "Gradient Magnitude":
            gx = sobel_x(gray)
            gy = sobel_y(gray)
            mag = gradient_magnitude(gx, gy)
            signed = mag
            out = np.clip(np.round(mag), 0, 255).astype(np.uint8)
            key = "gradient_magnitude"
            kernel = None
        elif op == "Laplacian":
            signed = laplacian(gray, variant=params["variant"])
            out = laplacian_uint8(gray, variant=params["variant"])
            key = "laplacian"
            kernel = (LAPLACIAN_8 if params["variant"] == "8"
                      else LAPLACIAN_4)
        else:
            out = canny(gray,
                        low_threshold=params["low"],
                        high_threshold=params["high"])
            signed = out.astype(np.float64)
            key = "canny"
            kernel = None

        elapsed_ms = (time.perf_counter() - t0) * 1000.0
        density = edge_density(out, threshold=0)

        deriv = [
            {
                "step": 1,
                "title": "Kernels",
                "expression": (
                    "Gx = I * Kx, Gy = I * Ky where Kx and Ky are the "
                    "Sobel kernels (or Laplacian kernel if selected)."
                ),
                "latex": r"G_x = I * K_x,\quad G_y = I * K_y",
                "values": {
                    "kernel": (kernel.tolist()
                               if kernel is not None else None),
                },
            },
            {
                "step": 2,
                "title": "Gradient magnitude",
                "expression": "|∇I| = sqrt(Gx² + Gy²)",
                "latex": r"|\nabla I| = \sqrt{G_x^2 + G_y^2}",
                "values": None,
            },
            {
                "step": 3,
                "title": "Edge density",
                "expression": (
                    f"Fraction of pixels above 0 = {density * 100:.2f}%"
                ),
                "latex": None,
                "values": {
                    "edge_density": density,
                    "max_signed": float(np.max(np.abs(signed))),
                },
            },
        ]

        tr = tm.record(
            operation=key,
            input_array=gray,
            parameters={
                "operation": op,
                **params,
                "trace_pixel": [int(trace_row), int(trace_col)],
            },
            derivation=deriv,
            intermediate_results=intermediate,
            output=out,
            execution_time_ms=elapsed_ms,
            validation={"ok": True, "warnings": [], "errors": []},
        )
        st.session_state["ed_last_trace"] = tr.to_dict()
        st.session_state["ed_last_image"] = out
        st.session_state["ed_last_density"] = density
    except Exception as exc:
        st.error(f"❌ Edge detection failed: {exc}")
        st.code(traceback.format_exc())


last = st.session_state.get("ed_last_trace")
out_img = st.session_state.get("ed_last_image")
density = st.session_state.get("ed_last_density", 0.0)


# ---------------------------------------------------------------------------
# LEFT: output + metrics
# ---------------------------------------------------------------------------
with left:
    if out_img is not None:
        with tab_e:
            st.image(out_img, caption="Edge map",
                     use_column_width=True, clamp=True)
        with tab_c:
            side_by_side(gray, out_img, labels=("Original", "Edges"))
        st.metric("Edge density", f"{density * 100:.2f}%")


# ---------------------------------------------------------------------------
# RIGHT: unified workspace
# ---------------------------------------------------------------------------

with right:
    st.subheader("🧮 Mathematical workspace")
    if last is None:
        st.info("Pick a detector and click **Detect**.")
    else:
        render_math(last, key_prefix="ed")