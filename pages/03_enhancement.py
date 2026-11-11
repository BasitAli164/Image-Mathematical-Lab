"""Enhancement page — Phase 5 version using unified math workspace."""

from __future__ import annotations

import time
import traceback

import matplotlib.pyplot as plt
import numpy as np
import streamlit as st

from core.enhancement import (
    adjust_brightness,
    adjust_contrast,
    apply_clahe,
    histogram_equalization_manual,
    histogram_equalization_opencv,
    min_max_normalize,
    sharpen,
)
from mathematical_engine.explanation_templates import (
    explain_brightness,
    explain_contrast,
    explain_hist_equalization,
    explain_min_max,
    explain_sharpen,
)
from mathematical_engine.numerical_calculator import maximum_abs_difference
from mathematical_engine.trace_manager import TraceManager
from visualization.comparison import difference_view, side_by_side
from visualization.math_workspace import render as render_math


st.set_page_config(page_title="Enhancement · Image Mathematical Lab",
                   page_icon="✨", layout="wide")

if "trace_manager" not in st.session_state:
    st.session_state.trace_manager = TraceManager()

tm: TraceManager = st.session_state.trace_manager
gray = st.session_state.get("gray_image")

st.title("✨ Image Enhancement")
st.caption("Point-wise intensity transformations with full math transparency.")

if gray is None:
    st.warning("⬅️ Upload an image on the main page first.")
    st.stop()


left, right = st.columns([1.15, 1.0], gap="large")


# ---------------------------------------------------------------------------
# LEFT: controls
# ---------------------------------------------------------------------------
with left:
    st.subheader("🖼️ Preview")
    tab_orig, tab_out, tab_cmp = st.tabs(
        ["Original", "Processed", "Compare"]
    )
    with tab_orig:
        st.image(gray, caption=f"Original {gray.shape}",
                 use_column_width=True, clamp=True)

    st.markdown("### ⚡ Experiment")
    op = st.selectbox(
        "Operation",
        ["Brightness", "Contrast", "Min-Max Normalize",
         "Histogram Equalization", "CLAHE", "Sharpen"],
        key="enh_op",
    )

    params: dict = {}
    if op == "Brightness":
        params["offset"] = st.slider(
            "offset b", -128, 128, 30, step=1, key="enh_b"
        )
    elif op == "Contrast":
        params["gain"] = st.slider(
            "gain g", 0.1, 3.0, 1.5, step=0.05, key="enh_g"
        )
    elif op == "CLAHE":
        params["clip_limit"] = st.slider(
            "clip limit", 0.5, 8.0, 2.0, step=0.1, key="enh_cl"
        )
        params["tile_grid"] = st.selectbox(
            "tile grid", [4, 8, 16], index=1, key="enh_tg"
        )
    elif op == "Sharpen":
        params["strength"] = st.slider(
            "strength alpha", 0.0, 3.0, 1.0, step=0.1, key="enh_sh"
        )

    run = st.button("Apply", type="primary", key="enh_run")


# ---------------------------------------------------------------------------
# Execute
# ---------------------------------------------------------------------------
if run:
    try:
        t0 = time.perf_counter()
        if op == "Brightness":
            out = adjust_brightness(gray, offset=int(params["offset"]))
            derivation = explain_brightness(gray, int(params["offset"]), out)
            key = "brightness"
        elif op == "Contrast":
            out = adjust_contrast(gray, gain=float(params["gain"]))
            derivation = explain_contrast(gray, float(params["gain"]), out)
            key = "contrast"
        elif op == "Min-Max Normalize":
            out = min_max_normalize(gray)
            derivation = explain_min_max(gray, out)
            key = "min_max_normalize"
        elif op == "Histogram Equalization":
            out = histogram_equalization_manual(gray)
            flat = gray.ravel()
            hist = np.bincount(flat, minlength=256)
            cdf = np.cumsum(hist)
            cdf_min = int(cdf[cdf > 0][0]) if np.any(cdf > 0) else 0
            derivation = explain_hist_equalization(
                gray, out, cdf_min, flat.size
            )
            key = "histogram_equalization"
        elif op == "CLAHE":
            out = apply_clahe(
                gray,
                clip_limit=float(params["clip_limit"]),
                tile_grid_size=(int(params["tile_grid"]),
                                int(params["tile_grid"])),
            )
            derivation = [{
                "step": 1,
                "title": "Algorithm",
                "expression": (
                    f"CLAHE with clip_limit={params['clip_limit']} "
                    f"tile_grid={params['tile_grid']}x{params['tile_grid']}"
                ),
                "latex": r"s = \mathrm{CLAHE}(r;\ c,\ g)",
                "values": params,
            }, {
                "step": 2,
                "title": "Spot check",
                "expression": (
                    f"pixel(0,0): {int(gray[0, 0])} -> {int(out[0, 0])}"
                ),
                "latex": None,
                "values": {"A_00": int(gray[0, 0]),
                           "C_00": int(out[0, 0])},
            }]
            key = "clahe"
        else:  # Sharpen
            out = sharpen(gray, strength=float(params["strength"]))
            derivation = explain_sharpen(gray, float(params["strength"]), out)
            key = "sharpen"

        elapsed_ms = (time.perf_counter() - t0) * 1000.0

        tr = tm.record(
            operation=key,
            input_array=gray,
            parameters={**params, "mode": op},
            derivation=derivation,
            output=out,
            execution_time_ms=elapsed_ms,
            validation={"ok": True, "warnings": [], "errors": []},
        )
        st.session_state["enh_last_trace"] = tr.to_dict()
        st.session_state["enh_last_image"] = out
    except Exception as exc:
        st.error(f"❌ Enhancement failed: {exc}")
        st.code(traceback.format_exc())


last = st.session_state.get("enh_last_trace")
out_img = st.session_state.get("enh_last_image")


# ---------------------------------------------------------------------------
# LEFT: show processed + comparison + histograms
# ---------------------------------------------------------------------------
with left:
    if out_img is not None:
        with tab_out:
            st.image(out_img, caption="Processed",
                     use_column_width=True, clamp=True)
        with tab_cmp:
            side_by_side(gray, out_img)
            difference_view(gray, out_img)

        st.markdown("#### Histogram comparison")
        fig, axes = plt.subplots(1, 2, figsize=(10, 3))
        axes[0].hist(gray.ravel(), bins=64, range=(0, 256), color="#555")
        axes[0].set_title("Original")
        axes[1].hist(out_img.ravel(), bins=64, range=(0, 256),
                     color="#0B6E99")
        axes[1].set_title("Processed")
        plt.tight_layout()
        st.pyplot(fig, use_container_width=True)
        plt.close(fig)

        if last and last.get("operation") == "histogram_equalization":
            cv = histogram_equalization_opencv(gray)
            d = maximum_abs_difference(
                out_img.astype(np.float64),
                cv.astype(np.float64),
            )
            with st.expander("🔎 Manual vs OpenCV comparison", expanded=False):
                st.markdown(
                    f"Max abs difference: **{d:.0f}** intensity levels"
                )
                side_by_side(out_img, cv, labels=("Manual", "OpenCV"))


# ---------------------------------------------------------------------------
# RIGHT: unified workspace
# ---------------------------------------------------------------------------
# ---------------------------------------------------------------------------
# RIGHT: unified workspace
# ---------------------------------------------------------------------------
with right:
    st.subheader("🧮 Mathematical workspace")
    if last is None:
        st.info("Choose an operation, click **Apply**, then all three "
                "levels of transparency appear here.")
    else:
        render_math(last, key_prefix="enh")