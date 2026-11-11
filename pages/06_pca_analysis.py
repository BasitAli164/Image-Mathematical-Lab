"""PCA page — Phase 5 version using unified workspace."""

from __future__ import annotations

import time
import traceback

import matplotlib.pyplot as plt
import numpy as np
import streamlit as st

from core.pca import (
    choose_n_for_variance,
    matrix_row_to_image,
    pca_dual,
    stack_to_matrix,
)
from core.reference_dataset import (
    describe_stack,
    load_reference_folder,
    synthesize_reference_stack,
)
from mathematical_engine.trace_manager import TraceManager
from visualization.math_workspace import render as render_math


st.set_page_config(page_title="PCA · Image Mathematical Lab", page_icon="🧬",
                   layout="wide")

if "trace_manager" not in st.session_state:
    st.session_state.trace_manager = TraceManager()

tm: TraceManager = st.session_state.trace_manager

st.title("🧬 PCA & Eigenvalue Analysis")
st.caption("Multi-image PCA over a reference stack — never single-image.")
st.info(
    "**Synthetic stack** (reproducible) or **your own folder**. "
    "No external dataset bundled — licensing/size."
)


# ---------------------------------------------------------------------------
# Sidebar: dataset source
# ---------------------------------------------------------------------------
st.sidebar.header("📚 Reference dataset")
source = st.sidebar.radio(
    "Choose source",
    ["Synthetic (recommended)", "My folder"],
    index=0, key="pca_source",
)

stack = None
source_label = ""

if source == "Synthetic (recommended)":
    n_images = st.sidebar.slider("n_images", 5, 50, 20, key="pca_n")
    size = st.sidebar.select_slider(
        "size (px)", [32, 64, 96, 128], value=64, key="pca_size"
    )
    seed = st.sidebar.number_input(
        "seed", 0, 9999, 42, step=1, key="pca_seed"
    )
    if st.sidebar.button("Generate stack", key="pca_gen"):
        try:
            s = synthesize_reference_stack(
                n_images=int(n_images), size=int(size), seed=int(seed)
            )
            st.session_state["pca_stack"] = s
            st.session_state["pca_source_label"] = (
                f"Synthetic · N={n_images} · {size}x{size} · seed={seed}"
            )
        except Exception as exc:
            st.sidebar.error(f"Generate failed: {exc}")
    stack = st.session_state.get("pca_stack")
else:
    folder = st.sidebar.text_input("Folder path", "", key="pca_folder")
    size = st.sidebar.select_slider(
        "target size", [32, 64, 96, 128], value=64, key="pca_folder_size"
    )
    if st.sidebar.button("Load folder", key="pca_load"):
        try:
            s = load_reference_folder(
                folder, target_size=int(size), max_images=50
            )
            st.session_state["pca_stack"] = s
            st.session_state["pca_source_label"] = (
                f"Folder · N={s.shape[0]} · {size}x{size} · {folder}"
            )
        except Exception as exc:
            st.sidebar.error(f"Load failed: {exc}")
    stack = st.session_state.get("pca_stack")

source_label = st.session_state.get("pca_source_label", "")

if stack is None:
    st.warning(
        "⬅️ Use the sidebar to generate a synthetic reference stack or "
        "load a folder of images."
    )
    st.stop()

N, H, W = stack.shape
D = H * W
if N < 2:
    st.error(f"Need at least 2 images, got N={N}.")
    st.stop()

st.success(f"Stack ready: {source_label or f'N={N} · {H}x{W}'}")


left, right = st.columns([1.15, 1.0], gap="large")


# ---------------------------------------------------------------------------
# LEFT
# ---------------------------------------------------------------------------
with left:
    st.subheader("🖼️ Reference stack")
    info = describe_stack(stack)
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("N", info["n_images"])
    c2.metric("Size", f"{info['height']}×{info['width']}")
    c3.metric("D", f"{info['pixel_count_per_image']:,}")
    c4.metric("Memory", f"{info['total_bytes'] / 1024:.1f} KB")

    st.markdown("**Sample (first 6)**")
    n_show = min(6, N)
    fig, axes = plt.subplots(1, n_show, figsize=(2 * n_show, 2.5))
    if n_show == 1:
        axes = [axes]
    for i in range(n_show):
        axes[i].imshow(stack[i], cmap="gray", vmin=0, vmax=255)
        axes[i].axis("off")
        axes[i].set_title(f"#{i}", fontsize=9)
    plt.tight_layout()
    st.pyplot(fig, use_container_width=True)
    plt.close(fig)

    st.markdown("### ⚡ Experiment")
    k_max = min(N - 1, D)
    k = int(st.slider("n_components k", 1, k_max,
                      min(10, k_max), key="pca_k"))
    run = st.button("Run PCA", type="primary", key="pca_run")


if run:
    try:
        with st.spinner("Dual PCA..."):
            t0 = time.perf_counter()
            X = stack_to_matrix(stack)
            res = pca_dual(X, n_components=k)
            elapsed_ms = (time.perf_counter() - t0) * 1000.0
        st.session_state["pca_result"] = res
        st.session_state["pca_X_shape"] = X.shape
        st.session_state["pca_size_hw"] = (H, W)
        st.session_state["pca_stack_ref"] = stack
        st.session_state["pca_time_ms"] = elapsed_ms

        deriv = [
            {
                "step": 1,
                "title": "Data shape",
                "expression": f"X is {X.shape[0]} x {X.shape[1]}",
                "latex": r"X \in \mathbb{R}^{N \times D}",
                "values": {"N": X.shape[0], "D": X.shape[1]},
            },
            {
                "step": 2,
                "title": "Mean-center",
                "expression": "Z = X - mean(X, axis=0)",
                "latex": r"Z = X - \bar{X}",
                "values": None,
            },
            {
                "step": 3,
                "title": "Dual Gram matrix",
                "expression": (
                    f"G = Z Z^T / (N-1) is "
                    f"{X.shape[0]} x {X.shape[0]}"
                ),
                "latex": r"G = \frac{1}{N-1} Z Z^{\top}",
                "values": {"gram_shape": [X.shape[0], X.shape[0]]},
            },
            {
                "step": 4,
                "title": "Eigen-decomposition",
                "expression": "G u_i = lambda_i u_i",
                "latex": r"G u_i = \lambda_i u_i",
                "values": {
                    "top_eigenvalues": [
                        float(v) for v in res.eigenvalues[:5]
                    ],
                },
            },
            {
                "step": 5,
                "title": "Explained variance",
                "expression": (
                    f"EVR_1 = {float(res.explained_variance_ratio[0]):.4f}, "
                    f"cumulative@{k} = "
                    f"{float(res.cumulative_variance_ratio[-1]):.4f}"
                ),
                "latex": r"\mathrm{EVR}_i = \frac{\lambda_i}{\sum_j \lambda_j}",
                "values": {
                    "evr_1": float(res.explained_variance_ratio[0]),
                    "cumulative": float(res.cumulative_variance_ratio[-1]),
                },
            },
            {
                "step": 6,
                "title": "Reconstruction",
                "expression": f"X_hat = mean + A_k V_k^T (k={k})",
                "latex": r"\hat{X} = \bar{X} + A_k V_k^{\top}",
                "values": {"k": k},
            },
        ]

        tr = tm.record(
            operation="pca",
            input_array=X,
            parameters={"n_components": k, "method": "dual"},
            derivation=deriv,
            output=res.reconstruction,
            execution_time_ms=elapsed_ms,
            validation={"ok": True, "warnings": [], "errors": [],
                        "method": res.method},
        )
        st.session_state["pca_last_trace"] = tr.to_dict()
    except Exception as exc:
        st.error(f"❌ PCA failed: {exc}")
        st.code(traceback.format_exc())


last = st.session_state.get("pca_last_trace")
res = st.session_state.get("pca_result")
size_hw = st.session_state.get("pca_size_hw", (H, W))
stack_ref = st.session_state.get("pca_stack_ref", stack)


# ---------------------------------------------------------------------------
# LEFT: variance plots + reconstruction
# ---------------------------------------------------------------------------
with left:
    if res is not None:
        st.markdown("### 📈 Variance")
        fig, axes = plt.subplots(1, 2, figsize=(11, 3.5))
        axes[0].plot(np.arange(1, res.eigenvalues.size + 1),
                     res.eigenvalues, marker="o")
        axes[0].set_title("Scree plot")
        axes[0].set_xlabel("Component")
        axes[0].set_ylabel("λ")

        axes[1].plot(np.arange(1, res.cumulative_variance_ratio.size + 1),
                     res.cumulative_variance_ratio * 100, marker="o",
                     color="#0B6E99")
        axes[1].axhline(95, color="red", linestyle="--", label="95%")
        axes[1].axhline(99, color="orange", linestyle="--", label="99%")
        axes[1].set_title("Cumulative EVR")
        axes[1].set_xlabel("k")
        axes[1].set_ylabel("%")
        axes[1].legend()
        plt.tight_layout()
        st.pyplot(fig, use_container_width=True)
        plt.close(fig)

        target = st.slider("Variance target (%)", 50, 99, 95,
                           step=1, key="pca_target")
        k_needed = choose_n_for_variance(
            res.cumulative_variance_ratio, target / 100.0
        )
        if k_needed > 0:
            st.info(f"**{target}%** variance needs k = **{k_needed}**")
        else:
            st.warning("Target not reached with available components.")

        st.markdown("### 🔄 Reconstruction")
        idx = st.slider("Image index", 0, stack_ref.shape[0] - 1, 0,
                        key="pca_idx")
        orig = stack_ref[idx]
        recon = matrix_row_to_image(res.reconstruction[idx], size_hw)
        c1, c2 = st.columns(2)
        c1.image(orig, caption="Original", use_column_width=True,
                 clamp=True)
        c2.image(recon, caption="Reconstructed", use_column_width=True,
                 clamp=True)

        with st.expander("🎨 Mean image & first PCs", expanded=False):
            mean_img = matrix_row_to_image(res.mean_image, size_hw)
            cols = st.columns(3)
            cols[0].image(mean_img, caption="Mean", use_column_width=True,
                          clamp=True)
            for i, col in enumerate(cols[1:]):
                if i < res.components.shape[0]:
                    comp = res.components[i]
                    cmin, cmax = float(comp.min()), float(comp.max())
                    norm = (
                        (comp - cmin) / (cmax - cmin) * 255
                        if cmax > cmin else np.zeros_like(comp)
                    )
                    img = norm.astype(np.uint8).reshape(size_hw)
                    col.image(img, caption=f"PC{i + 1}",
                              use_column_width=True, clamp=True)


# ---------------------------------------------------------------------------
# RIGHT: unified workspace
# ---------------------------------------------------------------------------
with right:
    st.subheader("🧮 Mathematical workspace")
    if last is None:
        st.info("Configure the stack and click **Run PCA** — the three "
                "levels of mathematical transparency will appear here.")
    else:
        render_math(last, key_prefix="pca")