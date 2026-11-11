"""Image Mathematical Lab — Linear Algebra Image Processing.

Phase 6 (final) entry point. All modules are now available via the
custom sidebar navigation.

Each operation records an actual mathematical trace with three levels of
transparency (formula / derivation / pixel-level inspection).
"""

from __future__ import annotations

import time
import traceback

import numpy as np
import streamlit as st

from core.image_loader import (
    build_metadata,
    load_image,
    to_grayscale,
)
from core.image_preprocessing import ensure_grayscale, ensure_uint8
from mathematical_engine.trace_manager import TraceManager


st.set_page_config(
    page_title="Image Processing Mathematical Lab",
    page_icon="🖼️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# =========================================================
# CUSTOM SIDEBAR CSS (Hides default streamlit navigation)
# =========================================================
st.markdown("""
    <style>
        [data-testid="stSidebarNav"] {
            display: none;
        }
    </style>
""", unsafe_allow_html=True)


def _init_session_state() -> None:
    defaults = {
        "gray_image": None,
        "original_image": None,
        "metadata": None,
        "filename": None,
        "trace_manager": TraceManager(),
    }
    for key, val in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = val


_init_session_state()


def _render_matrix_preview(arr: np.ndarray, n_rows: int = 6,
                            n_cols: int = 6) -> None:
    h, w = arr.shape
    r = min(n_rows, h)
    c = min(n_cols, w)
    st.caption(f"Top-left {r}x{c} of {h}x{w}:")
    st.dataframe(arr[:r, :c], use_container_width=False,
                 height=min(40 + 35 * r, 300))


def _render_sidebar_uploader() -> None:
    with st.sidebar:
        # =========================================================
        # NAYA CUSTOM NAVIGATION MENU
        # =========================================================
        st.title("🧭 Navigation")
        st.markdown("---")
        
        st.page_link("pages/01_dashboard.py", label="📊 Dashboard", help="Session summary and history")
        st.page_link("pages/02_matrix_explorer.py", label="🔢 Matrix Explorer", help="Pixels as numbers, matrix operations")
        st.page_link("pages/03_enhancement.py", label="✨ Enhancement", help="Brightness, contrast, hist-eq, CLAHE")
        st.page_link("pages/04_noise_reduction.py", label="🧹 Noise Reduction", help="Mean, Gaussian, median, custom kernels")
        st.page_link("pages/05_edge_detection.py", label="📐 Edge Detection", help="Sobel, Laplacian, Canny")
        st.page_link("pages/06_pca_analysis.py", label="🧬 PCA Analysis", help="Eigen-decomposition over reference stack")
        st.page_link("pages/07_svd_compression.py", label="🗜️ SVD Compression", help="Rank-k reconstruction")
        st.page_link("pages/08_feature_analysis.py", label="📊 Feature Analysis", help="Mathematical image descriptors")
        st.page_link("pages/09_reports.py", label="📄 Reports", help="Comparison, metrics, downloads")

        st.markdown("---")
        
        # =========================================================
        # UPLOAD SECTION
        # =========================================================
        st.markdown("## 📤 Upload Image")
        uploaded = st.file_uploader(
            "PNG, JPG or JPEG · max 15 MB",
            type=["png", "jpg", "jpeg"],
            accept_multiple_files=False,
            key="main_uploader",
        )
       
        st.divider()
        st.markdown("### ℹ️ About")
        st.markdown(
            "An interactive **Linear Algebra sandbox** for image "
            "processing. Every operation exposes its actual "
            "mathematical trace — formulas, derivations, pixel-level "
            "inspection."
        )
        st.divider()

        if uploaded is not None:
            try:
                with st.spinner("Validating & decoding…"):
                    t0 = time.perf_counter()
                    original = load_image(uploaded.getvalue(),
                                          uploaded.name)
                    gray = to_grayscale(original)
                    gray = ensure_grayscale(ensure_uint8(gray))
                    elapsed_ms = (time.perf_counter() - t0) * 1000.0
                    meta = build_metadata(original, gray)

                st.session_state.original_image = original
                st.session_state.gray_image = gray
                st.session_state.metadata = meta
                st.session_state.filename = uploaded.name
                st.session_state.trace_manager.clear()

                st.success(
                    f"Loaded **{uploaded.name}** in {elapsed_ms:.1f} ms "
                    f"({gray.shape[1]}x{gray.shape[0]} px, uint8)."
                )
            except ValueError as exc:
                for key in ("gray_image", "original_image", "metadata",
                            "filename"):
                    st.session_state[key] = None
                st.error(f"❌ {exc}")
            except Exception as exc:
                for key in ("gray_image", "original_image", "metadata",
                            "filename"):
                    st.session_state[key] = None
                st.error(f"❌ Unexpected error: {exc}")
                with st.expander("Show traceback"):
                    st.code(traceback.format_exc())


_render_sidebar_uploader()

st.title("🖼️ Image Mathematical Lab")
st.markdown(
    "**Linear Algebra-based Image Processing & Enhancement** · "
)

st.info(
    "This application demonstrates **mathematical** linear-algebra "
    "techniques on images."
)

gray = st.session_state.gray_image
meta = st.session_state.metadata

if gray is None:
    st.warning("⬅️ **Upload an image in the sidebar** to begin.")
    st.markdown(
        """
        ### Explore all modules (sidebar navigation)
        - **📊 Dashboard** — Session summary and history
        - **🔢 Matrix Explorer** — Pixels as numbers, matrix operations
        - **✨ Enhancement** — Brightness, contrast, hist-eq, CLAHE
        - **🧹 Noise Reduction** — Mean, Gaussian, median, custom kernels
        - **📐 Edge Detection** — Sobel, Laplacian, Canny
        - **🧬 PCA Analysis** — Eigen-decomposition over reference stack
        - **🗜️ SVD Compression** — Rank-k reconstruction
        - **📊 Feature Analysis** — Mathematical image descriptors
        - **📄 Reports** — Comparison, metrics, downloads

        Every operation reveals its **actual mathematical trace** —
        formulas, step-by-step derivations, pixel-level inspection.
        """
    )
    st.stop()


left, right = st.columns([1.15, 1.0], gap="large")

with left:
    st.subheader("🖼️ Image Workspace")
    tab_orig, tab_gray, tab_cmp = st.tabs(
        ["Original", "Grayscale", "Compare"]
    )
    original = st.session_state.original_image

    with tab_orig:
        if original.ndim == 3 and original.shape[2] >= 3:
            st.image(original, caption="Original color upload",
                     use_column_width=True)
        else:
            st.image(original, caption="Original grayscale upload",
                     use_column_width=True, clamp=True)

    with tab_gray:
        st.image(gray, caption="Grayscale (BT.601 luma)",
                 use_column_width=True, clamp=True)

    with tab_cmp:
        c1, c2 = st.columns(2)
        c1.image(original, caption="Original", use_column_width=True,
                 clamp=True)
        c2.image(gray, caption="Grayscale", use_column_width=True,
                 clamp=True)

with right:
    st.subheader("🧮 Mathematical Workspace")
    st.caption("All values computed from your uploaded image.")

    m1, m2, m3 = st.columns(3)
    m1.metric("Height", f"{meta.height}")
    m2.metric("Width", f"{meta.width}")
    m3.metric("Pixels", f"{meta.height * meta.width:,}")

    m4, m5, m6 = st.columns(3)
    m4.metric("dtype", meta.array_dtype)
    m5.metric("Range", f"{meta.intensity_min}–{meta.intensity_max}")
    m6.metric("Mode", meta.original_mode)

    st.markdown("#### Numerical summary")
    st.table({
        "Statistic": ["Mean", "Median", "Std", "Variance",
                      "Minimum", "Maximum"],
        "Value": [
            f"{meta.intensity_mean:.3f}",
            f"{meta.intensity_median:.3f}",
            f"{meta.intensity_std:.3f}",
            f"{meta.intensity_std ** 2:.3f}",
            f"{meta.intensity_min}",
            f"{meta.intensity_max}",
        ],
    })

    with st.expander("🔍 Matrix preview", expanded=False):
        _render_matrix_preview(gray)

    with st.expander("ℹ️ Grayscale formula", expanded=False):
        st.latex(r"Y = 0.299\,R + 0.587\,G + 0.114\,B")
        st.caption("ITU-R BT.601 luma coefficients.")

st.divider()
st.markdown("### 🚀 Next steps")
st.markdown(
    "Use the **sidebar** to open any module. Start with **🔢 Matrix "
    "Explorer** or **📊 Dashboard**."
)

st.caption(
    f"File: `{st.session_state.filename}` · "
    f"Shape `{gray.shape}` · dtype `{gray.dtype}`"
)