"""Noise Reduction page — Phase 5 version with pixel trace integrated
into Level 3 of the math workspace."""

from __future__ import annotations

import time
import traceback

import numpy as np
import streamlit as st

from core.convolution import convolve_manual, compare_implementations
from core.filtering import (
    gaussian_filter,
    gaussian_kernel,
    mean_filter,
    mean_kernel,
    median_filter,
)
from mathematical_engine.pixel_trace import trace_convolution_pixel
from mathematical_engine.trace_manager import TraceManager
from visualization.comparison import side_by_side
from visualization.math_workspace import render as render_math


st.set_page_config(page_title="Noise Reduction · Image Mathematical Lab",
                   page_icon="🧹", layout="wide")

if "trace_manager" not in st.session_state:
    st.session_state.trace_manager = TraceManager()

tm: TraceManager = st.session_state.trace_manager
gray = st.session_state.get("gray_image")

st.title("🧹 Noise Reduction & Filtering")
st.caption("Convolution-based smoothing with pixel-level trace.")

if gray is None:
    st.warning("⬅️ Upload an image on the main page first.")
    st.stop()


left, right = st.columns([1.15, 1.0], gap="large")


# ---------------------------------------------------------------------------
# LEFT: controls
# ---------------------------------------------------------------------------
with left:
    st.subheader("🖼️ Preview")
    tab_o, tab_p, tab_c = st.tabs(["Original", "Filtered", "Compare"])
    with tab_o:
        st.image(gray, caption=f"Original {gray.shape}",
                 use_column_width=True, clamp=True)

    st.markdown("### ⚡ Experiment")
    op = st.selectbox(
        "Operation",
        ["Mean Filter", "Gaussian Filter", "Median Filter",
         "Custom Kernel (Manual Convolution)"],
        key="flt_op",
    )

    kernel_matrix = None
    params: dict = {}

    if op == "Mean Filter":
        params["kernel_size"] = int(st.slider(
            "kernel size", 3, 15, 3, step=2, key="flt_mean_k"
        ))
    elif op == "Gaussian Filter":
        params["kernel_size"] = int(st.slider(
            "kernel size", 3, 15, 5, step=2, key="flt_gauss_k"
        ))
        params["sigma"] = float(st.slider(
            "sigma", 0.1, 5.0, 1.0, step=0.1, key="flt_gauss_s"
        ))
    elif op == "Median Filter":
        params["kernel_size"] = int(st.slider(
            "kernel size", 3, 15, 3, step=2, key="flt_med_k"
        ))
    else:
        st.markdown("**3×3 custom kernel**")
        c1, c2, c3 = st.columns(3)
        vals = []
        for i in range(3):
            row_vals = []
            for j in range(3):
                v = st.number_input(
                    f"K[{i},{j}]",
                    value=1.0 if (i == 1 and j == 1) else 0.0,
                    step=0.1,
                    key=f"flt_c_{i}_{j}",
                    label_visibility="collapsed",
                )
                row_vals.append(float(v))
            vals.append(row_vals)
        kernel_matrix = np.array(vals, dtype=np.float64)

    padding = st.selectbox(
        "padding", ["reflect", "zero", "replicate", "wrap"],
        index=0, key="flt_pad",
    )

    trace_row = st.number_input(
        "trace pixel row", 0, gray.shape[0] - 1, gray.shape[0] // 2,
        step=1, key="flt_tr_r",
    )
    trace_col = st.number_input(
        "trace pixel col", 0, gray.shape[1] - 1, gray.shape[1] // 2,
        step=1, key="flt_tr_c",
    )

    run = st.button("Apply Filter", type="primary", key="flt_run")


# ---------------------------------------------------------------------------
# Execute
# ---------------------------------------------------------------------------
if run:
    try:
        t0 = time.perf_counter()
        if op == "Mean Filter":
            out = mean_filter(gray, kernel_size=params["kernel_size"])
            K = mean_kernel(params["kernel_size"])
            key = "mean_filter"
        elif op == "Gaussian Filter":
            out = gaussian_filter(
                gray,
                kernel_size=params["kernel_size"],
                sigma=params["sigma"],
            )
            K = gaussian_kernel(params["kernel_size"], params["sigma"])
            key = "gaussian_filter"
        elif op == "Median Filter":
            out = median_filter(gray, kernel_size=params["kernel_size"])
            K = None
            key = "median_filter"
        else:
            K = kernel_matrix
            manual = convolve_manual(gray, K, padding=padding)
            out = np.clip(np.round(manual), 0, 255).astype(np.uint8)
            key = "convolution"

        elapsed_ms = (time.perf_counter() - t0) * 1000.0

        # Build derivation
        deriv: list[dict] = [
            {
                "step": 1,
                "title": "Convolution formula",
                "expression": (
                    "O(i,j) = sum_u sum_v I(i-u, j-v) * K(u,v)"
                ),
                "latex": (
                    r"O(i,j) = \sum_u \sum_v I(i-u,\,j-v)\,K(u,v)"
                ),
                "values": {
                    "padding": padding,
                    "operation": op,
                },
            },
        ]

        # Add pixel-level trace if a linear kernel exists
        intermediate: list[dict] = []
        if K is not None:
            try:
                pt = trace_convolution_pixel(
                    gray, K, int(trace_row), int(trace_col),
                    padding=padding,
                )
                intermediate.append(pt)
                deriv.append({
                    "step": 2,
                    "title": f"Pixel trace at ({trace_row}, {trace_col})",
                    "expression": (
                        f"window {pt['kernel_shape']} · "
                        f"sum = {pt['sum']:.4f}"
                    ),
                    "latex": None,
                    "values": {
                        "pixel": [int(trace_row), int(trace_col)],
                        "result": float(pt["result"]),
                    },
                })
            except Exception as e:
                deriv.append({
                    "step": 2,
                    "title": "Pixel trace",
                    "expression": f"Not available: {e}",
                    "latex": None,
                    "values": None,
                })
        else:
            deriv.append({
                "step": 2,
                "title": "Non-linear filter",
                "expression": (
                    "Median filter has no linear kernel; no per-pixel "
                    "multiplication trace."
                ),
                "latex": None,
                "values": None,
            })

        tr = tm.record(
            operation=key,
            input_array=gray,
            parameters={
                "operation": op,
                "padding": padding,
                **params,
                "trace_pixel": [int(trace_row), int(trace_col)],
            },
            derivation=deriv,
            intermediate_results=intermediate,
            output=out,
            execution_time_ms=elapsed_ms,
            validation={"ok": True, "warnings": [], "errors": []},
        )
        st.session_state["flt_last_trace"] = tr.to_dict()
        st.session_state["flt_last_image"] = out
        st.session_state["flt_last_kernel"] = (
            K.tolist() if K is not None else None
        )
        st.session_state["flt_last_padding"] = padding
    except Exception as exc:
        st.error(f"❌ Filtering failed: {exc}")
        st.code(traceback.format_exc())


last = st.session_state.get("flt_last_trace")
out_img = st.session_state.get("flt_last_image")
last_kernel = st.session_state.get("flt_last_kernel")
last_padding = st.session_state.get("flt_last_padding", "reflect")


# ---------------------------------------------------------------------------
# LEFT: output + kernel + comparison
# ---------------------------------------------------------------------------
with left:
    if out_img is not None:
        with tab_p:
            st.image(out_img, caption="Filtered",
                     use_column_width=True, clamp=True)
        with tab_c:
            side_by_side(gray, out_img)

        if last_kernel is not None:
            K = np.array(last_kernel)
            st.markdown(
                f"**Kernel** (sum of weights = {float(K.sum()):.4f})"
            )
            st.dataframe(K, use_container_width=False)

            with st.expander("🔎 Manual vs optimized", expanded=False):
                try:
                    report = compare_implementations(
                        gray, K, padding=last_padding
                    )
                    st.json(report)
                except Exception as exc:
                    st.warning(f"Comparison skipped: {exc}")


# ---------------------------------------------------------------------------
# RIGHT: unified workspace
# ---------------------------------------------------------------------------
with right:
    st.subheader("🧮 Mathematical workspace")
    if last is None:
        st.info("Configure and click **Apply Filter** — all three "
                "transparency levels appear here.")
    else:
        render_math(last, key_prefix="flt")