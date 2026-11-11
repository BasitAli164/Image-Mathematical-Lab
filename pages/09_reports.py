"""Consolidated reports page — image comparison + metrics + downloads."""

from __future__ import annotations

import io
from datetime import datetime

import numpy as np
import pandas as pd
import streamlit as st

from core.evaluation import DISCLAIMER, evaluate_all
from mathematical_engine.history import summarize, to_rows
from mathematical_engine.trace_manager import TraceManager
from visualization.comparison import difference_view, side_by_side


st.set_page_config(page_title="Reports · Image Mathematical Lab",
                   page_icon="📄", layout="wide")

if "trace_manager" not in st.session_state:
    st.session_state.trace_manager = TraceManager()

tm: TraceManager = st.session_state.trace_manager
gray = st.session_state.get("gray_image")

st.title("📄 Results & Reports")
st.caption(
    "Consolidated metrics, image comparison, downloads. All metrics are "
    "mathematical; they are **not clinical** evidence."
)

if gray is None:
    st.warning("⬅️ Upload an image on the main page first.")
    st.stop()


# ---------------------------------------------------------------------------
# Select processed image
# ---------------------------------------------------------------------------
st.markdown("## 1. Choose a processed image")
processed_candidates = {
    "Enhancement output": st.session_state.get("enh_last_image"),
    "Filtering output": st.session_state.get("flt_last_image"),
    "Edge map": st.session_state.get("ed_last_image"),
    "SVD reconstruction": st.session_state.get("svd_recon"),
}

available = {k: v for k, v in processed_candidates.items() if v is not None}

if not available:
    st.info(
        "No processed images yet. Run an operation on the Enhancement, "
        "Noise Reduction, Edge Detection or SVD Compression page first."
    )
    st.stop()

choice = st.selectbox(
    "Which processed image?", list(available.keys()), key="rep_choice"
)
processed = available[choice]


# ---------------------------------------------------------------------------
# Comparison + metrics
# ---------------------------------------------------------------------------
st.markdown("## 2. Comparison")
side_by_side(gray, processed, labels=("Original", choice))
difference_view(gray, processed)


st.markdown("## 3. Image quality metrics")
st.caption(DISCLAIMER)

try:
    metrics = evaluate_all(gray, processed)
    c1, c2, c3 = st.columns(3)
    c1.metric(
        "MSE",
        f"{metrics['mse']:.3f}" if metrics["mse"] is not None else "N/A",
    )
    mse_val = metrics.get("mse")
    psnr_val = metrics.get("psnr_db")
    if psnr_val is not None and mse_val is not None:
        if np.isfinite(psnr_val):
            c2.metric("PSNR (dB)", f"{psnr_val:.2f}")
        else:
            c2.metric("PSNR (dB)", "∞")
    else:
        c2.metric("PSNR (dB)", "N/A")
    ssim_val = metrics.get("ssim")
    c3.metric(
        "SSIM",
        f"{ssim_val:.4f}" if ssim_val is not None else "N/A",
    )

    if metrics.get("notes"):
        for note in metrics["notes"]:
            st.warning(note)

    with st.expander("Metric details", expanded=False):
        st.json({
            "mse": metrics["mse"],
            "psnr_db": (metrics["psnr_db"]
                        if metrics["psnr_db"] is None
                        or np.isfinite(metrics["psnr_db"])
                        else "inf"),
            "ssim": metrics["ssim"],
            "shape": metrics["shape"],
        })
except Exception as exc:
    st.error(f"Evaluation failed: {exc}")


# ---------------------------------------------------------------------------
# Session summary
# ---------------------------------------------------------------------------
st.markdown("## 4. Session summary")
summary = summarize(tm.traces)
s1, s2, s3, s4 = st.columns(4)
s1.metric("Operations", summary["count"])
s2.metric("Total time", f"{summary['total_time_ms']:.1f} ms")
s3.metric("Mean time", f"{summary['mean_time_ms']:.2f} ms")
s4.metric("Last op", summary["last_operation"] or "—")

if summary["operations"]:
    with st.expander("Operation counts", expanded=False):
        st.json(summary["operations"])


# ---------------------------------------------------------------------------
# Downloads
# ---------------------------------------------------------------------------
st.markdown("## 5. Downloads")

# ---- Processed image ----
st.markdown("**Processed image (PNG)**")
buf = io.BytesIO()
from PIL import Image
Image.fromarray(processed.astype(np.uint8)).save(buf, format="PNG")
st.download_button(
    "⬇️ Download processed image",
    data=buf.getvalue(),
    file_name=f"processed_{choice.lower().replace(' ', '_')}.png",
    mime="image/png",
    key="rep_dl_img",
)

# ---- Numerical report (CSV) ----
st.markdown("**Processing history (CSV)**")
if tm.traces:
    rows = to_rows(tm.traces)
    csv_bytes = pd.DataFrame(rows).to_csv(index=False).encode("utf-8")
    st.download_button(
        "⬇️ Download processing history CSV",
        data=csv_bytes,
        file_name="processing_history.csv",
        mime="text/csv",
        key="rep_dl_hist",
    )
else:
    st.caption("No traces to export yet.")

# ---- Metric report (CSV) ----
st.markdown("**Metric report (CSV)**")
try:
    metric_row = {
        "image": choice,
        "shape": str(metrics["shape"]),
        "mse": metrics["mse"],
        "psnr_db": (metrics["psnr_db"]
                    if metrics["psnr_db"] is None
                    or np.isfinite(metrics["psnr_db"])
                    else "inf"),
        "ssim": metrics["ssim"],
    }
    metric_csv = pd.DataFrame([metric_row]).to_csv(index=False).encode("utf-8")
    st.download_button(
        "⬇️ Download metric report CSV",
        data=metric_csv,
        file_name="metric_report.csv",
        mime="text/csv",
        key="rep_dl_metric",
    )
except Exception:
    st.caption("Metrics unavailable for download.")

# ---- PDF summary (reportlab) ----
st.markdown("**PDF summary**")
try:
    from reportlab.lib.pagesizes import A4
    from reportlab.pdfgen import canvas

    pdf_buf = io.BytesIO()
    c = canvas.Canvas(pdf_buf, pagesize=A4)
    width, height = A4

    c.setFont("Helvetica-Bold", 16)
    c.drawString(50, height - 50, " Image Mathematical Lab")
    c.setFont("Helvetica", 10)
    c.drawString(50, height - 70,
                 f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    c.drawString(50, height - 85, "Educational mathematical tool — NOT medical.")

    y = height - 120
    c.setFont("Helvetica-Bold", 12)
    c.drawString(50, y, "Comparison:")
    c.setFont("Helvetica", 10)
    y -= 20
    c.drawString(60, y, f"Source: uploaded image")
    y -= 15
    c.drawString(60, y, f"Processed: {choice}")
    y -= 15
    c.drawString(60, y, f"Shape: {metrics['shape']}")
    y -= 30

    c.setFont("Helvetica-Bold", 12)
    c.drawString(50, y, "Metrics:")
    y -= 20
    c.setFont("Helvetica", 10)
    for k in ("mse", "psnr_db", "ssim"):
        v = metrics.get(k)
        if isinstance(v, float) and not np.isfinite(v):
            v = "inf"
        c.drawString(60, y, f"{k}: {v}")
        y -= 15
    y -= 10

    c.setFont("Helvetica-Bold", 12)
    c.drawString(50, y, "Operations performed:")
    y -= 20
    c.setFont("Helvetica", 10)
    for t in tm.traces[-10:]:
        line = (
            f"{t.operation_id}: {t.operation} "
            f"({t.execution_time_ms:.2f} ms)"
        )
        c.drawString(60, y, line[:90])
        y -= 12
        if y < 80:
            c.showPage()
            y = height - 50

    c.setFont("Helvetica-Italic", 8)
    c.drawString(50, 40,
                 "All values computed from the actual uploaded image. "
                 "Metrics are not clinical.")
    c.save()

    st.download_button(
        "⬇️ Download PDF summary",
        data=pdf_buf.getvalue(),
        file_name="chest_xray_report.pdf",
        mime="application/pdf",
        key="rep_dl_pdf",
    )
except Exception as exc:
    st.caption(f"PDF export unavailable: {exc}")


# ---------------------------------------------------------------------------
# Recent traces table
# ---------------------------------------------------------------------------
st.markdown("## 6. Recent traces")
if tm.traces:
    rows = to_rows(tm.traces[-15:])
    st.dataframe(pd.DataFrame(rows), use_container_width=True)
else:
    st.caption("No traces recorded yet.")