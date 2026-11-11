"""SVD Compression page — Phase 5 version with unified workspace."""

from __future__ import annotations

import time
import traceback

import matplotlib.pyplot as plt
import numpy as np
import streamlit as st

from core.svd import (
    low_rank_approximate,
    reconstruction_error,
    singular_value_summary,
)
from mathematical_engine.explanation_templates import explain_svd
from mathematical_engine.trace_manager import TraceManager
from visualization.comparison import difference_view, side_by_side
from visualization.math_workspace import render as render_math


st.set_page_config(page_title="SVD Compression · Image Mathematical Lab",
                   page_icon="🗜️", layout="wide")

if "trace_manager" not in st.session_state:
    st.session_state.trace_manager = TraceManager()

tm: TraceManager = st.session_state.trace_manager
gray = st.session_state.get("gray_image")

st.title("🗜️ SVD Image Compression")
st.caption(
    "Rank-k approximation via SVD. Every number below comes from the "
    "actual matrix A of your image."
)

if gray is None:
    st.warning("⬅️ Upload an image on the main page first.")
    st.stop()

m, n = gray.shape
kmax = min(m, n)
st.markdown(
    f"**Image:** {m}×{n} · dtype `{gray.dtype}` · max rank = **{kmax}**"
)


left, right = st.columns([1.15, 1.0], gap="large")


# ---------------------------------------------------------------------------
# LEFT: preview + slider
# ---------------------------------------------------------------------------
with left:
    st.subheader("🖼️ Compression")
    tab_orig, tab_rec, tab_cmp = st.tabs(
        ["Original", "Reconstructed", "Compare"]
    )
    with tab_orig:
        st.image(gray, caption=f"Original {m}×{n}",
                 use_column_width=True, clamp=True)

    st.markdown("### ⚡ Experiment")
    k = st.slider("rank k", 1, kmax, min(20, kmax), key="svd_k")
    run = st.button("Run SVD", type="primary", key="svd_run")


if run:
    try:
        with st.spinner("Computing SVD..."):
            t0 = time.perf_counter()
            res = low_rank_approximate(gray, k=int(k))
            elapsed_ms = (time.perf_counter() - t0) * 1000.0
        recon_u8 = np.clip(np.round(res.reconstruction), 0, 255).astype(
            np.uint8
        )
        errs = reconstruction_error(gray, res.reconstruction)
        sv_summary = singular_value_summary(res.S, top=10)

        st.session_state["svd_result"] = res
        st.session_state["svd_recon"] = recon_u8
        st.session_state["svd_errors"] = errs
        st.session_state["svd_summary"] = sv_summary
        st.session_state["svd_k_used"] = int(k)
        st.session_state["svd_time_ms"] = elapsed_ms

        deriv = explain_svd(
            m, n, res.S, int(k), res.relative_error, res.compression_ratio,
        )
        tr = tm.record(
            operation="svd_rank_k",
            input_array=gray,
            parameters={"k": int(k), "max_rank": int(kmax)},
            derivation=deriv,
            output=recon_u8,
            execution_time_ms=elapsed_ms,
            validation={"ok": True, "warnings": [], "errors": []},
        )
        st.session_state["svd_last_trace"] = tr.to_dict()
    except Exception as exc:
        st.error(f"❌ SVD failed: {exc}")
        st.code(traceback.format_exc())


last = st.session_state.get("svd_last_trace")
res = st.session_state.get("svd_result")
recon = st.session_state.get("svd_recon")
errs = st.session_state.get("svd_errors", {})
sv_summary = st.session_state.get("svd_summary", {})
k_used = st.session_state.get("svd_k_used", 0)
elapsed_ms = st.session_state.get("svd_time_ms", 0.0)


# ---------------------------------------------------------------------------
# LEFT: display
# ---------------------------------------------------------------------------
with left:
    if recon is not None:
        with tab_rec:
            st.image(recon, caption=f"Rank-{k_used}",
                     use_column_width=True, clamp=True)
        with tab_cmp:
            side_by_side(gray, recon,
                         labels=("Original", f"Rank-{k_used}"))
            difference_view(gray, recon, label="|Recon − Original|")

        st.markdown("### 📊 Singular values")
        fig, axes = plt.subplots(1, 2, figsize=(11, 3.5))
        top_n = min(60, res.S.size)
        axes[0].plot(np.arange(1, top_n + 1), res.S[:top_n], marker=".")
        axes[0].set_title("Singular values (top 60)")
        axes[0].set_xlabel("Index")
        axes[0].set_ylabel("σ")

        energy = res.S ** 2
        cum = np.cumsum(energy) / np.sum(energy)
        axes[1].plot(np.arange(1, cum.size + 1), cum * 100)
        axes[1].axhline(95, color="red", linestyle="--", label="95%")
        axes[1].axhline(99, color="orange", linestyle="--", label="99%")
        axes[1].set_title("Cumulative energy")
        axes[1].set_xlabel("k")
        axes[1].set_ylabel("%")
        axes[1].legend()
        plt.tight_layout()
        st.pyplot(fig, use_container_width=True)
        plt.close(fig)


# ---------------------------------------------------------------------------
# RIGHT: unified workspace
# ---------------------------------------------------------------------------
with right:
    st.subheader("🧮 Mathematical workspace")
    if last is None:
        st.info("Adjust rank, click **Run SVD** — the three levels of "
                "mathematical transparency appear here.")
    else:
        m1, m2, m3 = st.columns(3)
        m1.metric("Relative error", f"{res.relative_error:.4f}")
        m2.metric("PSNR (dB)", f"{errs.get('psnr_db', 0.0):.2f}" if np.isfinite(errs.get("psnr_db", 0.0)) else "∞")
        m3.metric("Compression", f"{res.compression_ratio:.2f}×")

        with st.expander("🎯 Top singular values", expanded=False):
            st.json(sv_summary)

        with st.expander("🔢 Matrix summaries", expanded=False):
            U, S, Vt = res.U, res.S, res.Vt
            st.markdown(f"**U** {U.shape}  \n**Σ** {S.shape}  \n**Vt** {Vt.shape}")
            c1, c2 = st.columns(2)
            c1.markdown("**U (5×5)**")
            c1.dataframe(U[:5, :5], use_container_width=False)
            c2.markdown("**Vt (5×5)**")
            c2.dataframe(Vt[:5, :5], use_container_width=False)
            st.write("**Σ top-5**", [float(x) for x in S[:5]])

        # Yahan render_math call ho raha hai
        render_math(last, key_prefix="svd")