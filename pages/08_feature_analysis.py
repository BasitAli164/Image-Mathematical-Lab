"""Feature Analysis page — mathematical image descriptors."""

from __future__ import annotations

import time
import traceback

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st

from core.feature_extraction import extract_all_features
from mathematical_engine.trace_manager import TraceManager


st.set_page_config(page_title="Feature Analysis ·  Image Mathematical Lab",
                   page_icon="📊", layout="wide")

if "trace_manager" not in st.session_state:
    st.session_state.trace_manager = TraceManager()

tm: TraceManager = st.session_state.trace_manager
gray = st.session_state.get("gray_image")

st.title("📊 Image Feature Analysis")
st.caption(
    "Mathematical characteristics of your image. **These are not "
    "clinical indicators.** No diagnostic interpretation is implied."
)

if gray is None:
    st.warning("⬅️ Upload an image on the main page first.")
    st.stop()


# ---------------------------------------------------------------------------
# Controls
# ---------------------------------------------------------------------------
st.markdown("### ⚡ Experiment")
include_glcm = st.checkbox(
    "Include GLCM texture features (slower — quantizes to 8 levels)",
    value=False,
    key="feat_glcm",
)

run = st.button("Extract features", type="primary", key="feat_run")


# ---------------------------------------------------------------------------
# Execute
# ---------------------------------------------------------------------------
if run:
    try:
        with st.spinner("Extracting features..."):
            t0 = time.perf_counter()
            features = extract_all_features(
                gray, include_glcm=include_glcm
            )
            elapsed_ms = (time.perf_counter() - t0) * 1000.0

        st.session_state["feat_result"] = features
        st.session_state["feat_ms"] = elapsed_ms

        # Trace
        deriv = [
            {
                "step": 1,
                "title": "Intensity statistics",
                "expression": (
                    f"mean = {features['intensity']['mean']:.3f}, "
                    f"std = {features['intensity']['std']:.3f}, "
                    f"min = {features['intensity']['min']}, "
                    f"max = {features['intensity']['max']}"
                ),
                "latex": r"\mu = \frac{1}{N}\sum x_i",
                "values": features["intensity"],
            },
            {
                "step": 2,
                "title": "Gradient statistics",
                "expression": (
                    f"|∇| mean = {features['gradient']['magnitude_mean']:.3f}, "
                    f"|∇| max = {features['gradient']['magnitude_max']:.3f}"
                ),
                "latex": r"|\nabla I| = \sqrt{G_x^2 + G_y^2}",
                "values": features["gradient"],
            },
            {
                "step": 3,
                "title": "Edge density",
                "expression": (
                    f"canny = "
                    f"{features['edges']['edge_density_canny'] * 100:.2f}%, "
                    f"sobel = "
                    f"{features['edges']['edge_density_sobel_threshold'] * 100:.2f}%"
                ),
                "latex": None,
                "values": features["edges"],
            },
            {
                "step": 4,
                "title": "Histogram statistics",
                "expression": (
                    f"entropy = {features['histogram']['entropy_bits']:.3f} bits, "
                    f"uniformity = {features['histogram']['uniformity']:.5f}"
                ),
                "latex": r"H = -\sum p_i \log_2 p_i",
                "values": features["histogram"],
            },
        ]
        if include_glcm and "glcm" in features:
            deriv.append({
                "step": 5,
                "title": "GLCM texture",
                "expression": "Co-occurrence texture features",
                "latex": None,
                "values": features["glcm"],
            })

        tr = tm.record(
            operation="mean",     # generic operation key — covered by registry
            input_array=gray,
            parameters={"include_glcm": include_glcm},
            derivation=deriv,
            output=np.array(features["intensity"]["mean"]),
            execution_time_ms=elapsed_ms,
            validation={"ok": True, "warnings": [], "errors": []},
        )
        st.session_state["feat_last_trace"] = tr.to_dict()
    except Exception as exc:
        st.error(f"❌ Feature extraction failed: {exc}")
        st.code(traceback.format_exc())


features = st.session_state.get("feat_result")
elapsed_ms = st.session_state.get("feat_ms", 0.0)


# ---------------------------------------------------------------------------
# Display
# ---------------------------------------------------------------------------
if features is None:
    st.info("Click **Extract features** to begin.")
    st.stop()

st.success(f"Extracted in **{elapsed_ms:.2f} ms**")


# ---- Top-line metrics ----
st.markdown("### 🔢 Intensity")
inten = features["intensity"]
c1, c2, c3, c4, c5 = st.columns(5)
c1.metric("Mean", f"{inten['mean']:.2f}")
c2.metric("Std", f"{inten['std']:.2f}")
c3.metric("Min", inten["min"])
c4.metric("Max", inten["max"])
c5.metric("Variance", f"{inten['variance']:.2f}")


# ---- Intensity detail table ----
st.markdown("#### Intensity statistics")
st.dataframe(
    pd.DataFrame([inten]),
    use_container_width=True,
)


# ---- Histogram ----
st.markdown("### 📊 Histogram")
fig, ax = plt.subplots(figsize=(9, 3.2))
ax.hist(gray.ravel(), bins=128, range=(0, 256), color="#0B6E99")
ax.axvline(inten["mean"], color="red", linestyle="--",
           label=f"mean={inten['mean']:.1f}")
ax.axvline(inten["median"], color="orange", linestyle="--",
           label=f"median={inten['median']:.1f}")
ax.set_xlabel("Intensity")
ax.set_ylabel("Count")
ax.legend()
st.pyplot(fig, use_container_width=True)
plt.close(fig)


# ---- Gradient ----
st.markdown("### 🌀 Gradient statistics")
st.dataframe(
    pd.DataFrame([features["gradient"]]),
    use_container_width=True,
)


# ---- Edges ----
st.markdown("### 📐 Edge density")
edges = features["edges"]
e1, e2, e3 = st.columns(3)
e1.metric("Canny density", f"{edges['edge_density_canny'] * 100:.2f}%")
e2.metric("Sobel-thr density",
          f"{edges['edge_density_sobel_threshold'] * 100:.2f}%")
e3.metric("Edge pixels", f"{edges['canny_edge_pixels']:,}")


# ---- Histogram stats ----
st.markdown("### 📈 Histogram statistics")
st.dataframe(
    pd.DataFrame([features["histogram"]]),
    use_container_width=True,
)


# ---- GLCM ----
if "glcm" in features:
    st.markdown("### 🎨 GLCM texture features")
    st.dataframe(
        pd.DataFrame([features["glcm"]]),
        use_container_width=True,
    )


# ---- Full structured summary ----
st.markdown("### 🧾 Full feature summary")
st.json(features)


# ---- Download ----
def _flatten(d: dict, prefix: str = "") -> dict:
    out: dict = {}
    for k, v in d.items():
        key = f"{prefix}{k}" if not prefix else f"{prefix}.{k}"
        if isinstance(v, dict):
            out.update(_flatten(v, key))
        else:
            out[key] = v
    return out


flat = _flatten(features)
csv_bytes = pd.DataFrame([flat]).to_csv(index=False).encode("utf-8")
st.download_button(
    "⬇️ Download features as CSV",
    data=csv_bytes,
    file_name="feature_analysis.csv",
    mime="text/csv",
    key="feat_dl",
)


# ---- Recent traces ----
st.divider()
st.markdown("### 🧾 Recent feature extractions")
recent = [t for t in tm.traces if "include_glcm" in t.parameters][-5:]
if not recent:
    st.caption("No feature extractions yet.")
else:
    for t in reversed(recent):
        with st.expander(
            f"{t.operation_id} · {t.execution_time_ms:.2f} ms"
        ):
            st.json(t.to_dict())
# ---------------------------------------------------------------------------
# AI EXPLANATION SECTION (Added for Feature Analysis)
# ---------------------------------------------------------------------------
st.divider()
last_trace = st.session_state.get("feat_last_trace")

if last_trace:
    st.subheader("🧮 Mathematical workspace")
    # Yahan hum math_workspace ka render function call kar rahe hain
    from visualization.math_workspace import render as render_math
    render_math(last_trace, key_prefix="feat")