"""
Pothole Detection System - Streamlit UI
Intelligent hybrid CV + Deep Learning pothole detector
"""

import streamlit as st
import cv2
import numpy as np
import json
import time
import tempfile
import os
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).parent))

from utils.cv_pipeline import HybridDetector
from utils.metrics import MetricsEngine
from utils.synthetic_data import generate_sample_image, generate_video_frames

st.set_page_config(
    page_title="PotholeAI — Intelligent Road Analysis",
    page_icon="🚧",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Space+Grotesk:wght@400;500;600;700&family=JetBrains+Mono:wght@400;500&display=swap');

    html, body, [class*="css"] {
        font-family: 'Space Grotesk', sans-serif !important;
    }

    .stApp {
        background: #0a0c0f;
        color: #e2e8f0;
    }

    /* Header */
    .main-header {
        background: linear-gradient(135deg, #0f1923 0%, #1a2332 50%, #0f1923 100%);
        border: 1px solid #1e3a5f;
        border-radius: 16px;
        padding: 28px 36px;
        margin-bottom: 28px;
        position: relative;
        overflow: hidden;
    }
    .main-header::before {
        content: '';
        position: absolute;
        top: 0; left: 0; right: 0;
        height: 2px;
        background: linear-gradient(90deg, transparent, #3b82f6, #06b6d4, transparent);
    }
    .main-header h1 {
        font-size: 2rem;
        font-weight: 700;
        color: #f1f5f9;
        margin: 0 0 6px 0;
        letter-spacing: -0.5px;
    }
    .main-header p {
        color: #64748b;
        font-size: 0.9rem;
        margin: 0;
    }
    .badge {
        display: inline-block;
        padding: 3px 10px;
        border-radius: 20px;
        font-size: 0.72rem;
        font-weight: 600;
        margin-right: 6px;
        letter-spacing: 0.5px;
    }
    .badge-blue { background: #1e3a5f; color: #60a5fa; border: 1px solid #2563eb40; }
    .badge-cyan { background: #0c2d3a; color: #22d3ee; border: 1px solid #06b6d440; }
    .badge-green { background: #0c2d1a; color: #4ade80; border: 1px solid #16a34a40; }

    /* Metric cards */
    .metric-grid { display: grid; grid-template-columns: repeat(4, 1fr); gap: 14px; margin: 20px 0; }
    .metric-card {
        background: #111827;
        border: 1px solid #1f2937;
        border-radius: 12px;
        padding: 18px 20px;
        text-align: center;
    }
    .metric-card .value {
        font-size: 2rem;
        font-weight: 700;
        font-family: 'JetBrains Mono', monospace;
        line-height: 1;
        margin-bottom: 6px;
    }
    .metric-card .label {
        font-size: 0.75rem;
        color: #64748b;
        text-transform: uppercase;
        letter-spacing: 1px;
    }
    .metric-card.critical .value { color: #f87171; }
    .metric-card.severe .value { color: #fb923c; }
    .metric-card.moderate .value { color: #facc15; }
    .metric-card.minor .value { color: #4ade80; }
    .metric-card.info .value { color: #60a5fa; }
    .metric-card.fps .value { color: #22d3ee; }

    /* Detection box */
    .detection-item {
        background: #111827;
        border: 1px solid #1f2937;
        border-radius: 10px;
        padding: 14px 16px;
        margin-bottom: 10px;
        border-left: 3px solid;
    }
    .detection-item.critical { border-left-color: #ef4444; }
    .detection-item.severe { border-left-color: #f97316; }
    .detection-item.moderate { border-left-color: #eab308; }
    .detection-item.minor { border-left-color: #22c55e; }

    /* Severity badges */
    .sev-critical { background: #3f0808; color: #fca5a5; padding: 2px 8px; border-radius: 4px; font-size: 0.72rem; font-weight: 600; }
    .sev-severe { background: #3f1500; color: #fdba74; padding: 2px 8px; border-radius: 4px; font-size: 0.72rem; font-weight: 600; }
    .sev-moderate { background: #3f2d00; color: #fde047; padding: 2px 8px; border-radius: 4px; font-size: 0.72rem; font-weight: 600; }
    .sev-minor { background: #052e16; color: #86efac; padding: 2px 8px; border-radius: 4px; font-size: 0.72rem; font-weight: 600; }

    /* GPS map placeholder */
    .gps-map {
        background: #0d1117;
        border: 1px solid #1f2937;
        border-radius: 12px;
        padding: 20px;
        font-family: 'JetBrains Mono', monospace;
        font-size: 0.8rem;
        color: #4ade80;
    }

    /* Pipeline viz */
    .pipeline-stage {
        background: #111827;
        border: 1px solid #1f2937;
        border-radius: 8px;
        padding: 12px 16px;
        text-align: center;
        font-size: 0.82rem;
        font-weight: 500;
    }
    .pipeline-stage.active { border-color: #3b82f6; color: #60a5fa; }

    /* Scrollable log */
    .log-container {
        background: #0d1117;
        border: 1px solid #1f2937;
        border-radius: 10px;
        padding: 16px;
        font-family: 'JetBrains Mono', monospace;
        font-size: 0.78rem;
        color: #94a3b8;
        max-height: 280px;
        overflow-y: auto;
    }

    /* PR curve */
    .chart-container {
        background: #111827;
        border: 1px solid #1f2937;
        border-radius: 12px;
        padding: 20px;
    }

    /* Section headers */
    .section-title {
        font-size: 1rem;
        font-weight: 600;
        color: #cbd5e1;
        margin: 20px 0 12px 0;
        padding-bottom: 8px;
        border-bottom: 1px solid #1f2937;
    }

    /* Streamlit overrides */
    .stButton > button {
        background: #1e40af !important;
        color: white !important;
        border: 1px solid #2563eb !important;
        border-radius: 8px !important;
        font-family: 'Space Grotesk', sans-serif !important;
        font-weight: 500 !important;
        transition: all 0.2s !important;
    }
    .stButton > button:hover {
        background: #2563eb !important;
        border-color: #60a5fa !important;
    }
    .stSelectbox label, .stSlider label, .stFileUploader label {
        color: #94a3b8 !important;
        font-size: 0.85rem !important;
    }
    div[data-testid="stSidebar"] {
        background: #0d1117 !important;
        border-right: 1px solid #1f2937 !important;
    }
    .stSidebar .stSelectbox, .stSidebar .stSlider {
        background: transparent;
    }
    h1, h2, h3 { color: #f1f5f9 !important; }
    .stTabs [data-baseweb="tab"] {
        color: #64748b !important;
        font-family: 'Space Grotesk', sans-serif !important;
    }
    .stTabs [aria-selected="true"] {
        color: #60a5fa !important;
    }
    [data-testid="stMetricValue"] {
        font-family: 'JetBrains Mono', monospace !important;
        color: #60a5fa !important;
    }
</style>
""", unsafe_allow_html=True)


# ─── Session State ────────────────────────────────────────────────────────────
if 'detector' not in st.session_state:
    st.session_state.detector = HybridDetector()
if 'metrics_engine' not in st.session_state:
    st.session_state.metrics_engine = MetricsEngine()
if 'all_detections' not in st.session_state:
    st.session_state.all_detections = []
if 'processed_frames' not in st.session_state:
    st.session_state.processed_frames = []
if 'last_result_image' not in st.session_state:
    st.session_state.last_result_image = None
if 'processing_log' not in st.session_state:
    st.session_state.processing_log = []


def add_log(msg: str, level: str = "INFO"):
    ts = time.strftime("%H:%M:%S")
    colors = {"INFO": "#64748b", "SUCCESS": "#4ade80", "WARN": "#facc15", "ERROR": "#f87171"}
    color = colors.get(level, "#64748b")
    st.session_state.processing_log.append(f'<span style="color:{color}">[{ts}] [{level}]</span> {msg}')
    if len(st.session_state.processing_log) > 100:
        st.session_state.processing_log = st.session_state.processing_log[-100:]


def process_single_frame(frame: np.ndarray) -> tuple:
    """Process one frame through the full pipeline"""
    detector = st.session_state.detector
    metrics = st.session_state.metrics_engine
    detections, debug_info = detector.detect(frame)
    metrics.record_frame(debug_info['processing_time_ms'])
    for det in detections:
        metrics.record_detection(det, debug_info['frame_id'])
        st.session_state.all_detections.append(det)
    result_frame = detector.visualize(frame, detections, debug_info)
    edges_colored = cv2.cvtColor(debug_info['edges'], cv2.COLOR_GRAY2RGB)
    return result_frame, edges_colored, detections, debug_info


def render_severity_badge(severity: str) -> str:
    return f'<span class="sev-{severity}">{severity.upper()}</span>'


# ─── Sidebar ──────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("### ⚙️ Detection Settings")
    conf_threshold = st.slider("Confidence Threshold", 0.3, 0.9, 0.45, 0.05)
    iou_threshold = st.slider("NMS IoU Threshold", 0.2, 0.7, 0.4, 0.05)
    min_area = st.slider("Min Pothole Area (px²)", 200, 3000, 800, 100)
    canny_low = st.slider("Canny Low Threshold", 20, 100, 50, 5)
    canny_high = st.slider("Canny High Threshold", 80, 250, 150, 5)

    st.markdown("### 🧠 Model Settings")
    show_cv_debug = st.checkbox("Show CV Debug View", value=True)
    show_gps = st.checkbox("Show GPS Coordinates", value=True)
    detect_duplicates = st.checkbox("Enable Duplicate Detection", value=True)
    severity_estimation = st.checkbox("Severity Estimation", value=True)

    st.markdown("### 📍 GPS Base Location")
    gps_lat = st.number_input("Base Latitude", value=10.7905, format="%.4f")
    gps_lon = st.number_input("Base Longitude", value=78.7047, format="%.4f")

    st.markdown("---")
    if st.button("🔄 Reset Session", use_container_width=True):
        st.session_state.detector = HybridDetector(gps_base=(gps_lat, gps_lon))
        st.session_state.metrics_engine = MetricsEngine()
        st.session_state.all_detections = []
        st.session_state.processed_frames = []
        st.session_state.last_result_image = None
        st.session_state.processing_log = []
        add_log("Session reset", "SUCCESS")
        st.rerun()

    # Update detector params
    st.session_state.detector.cv_pipeline.canny_low = canny_low
    st.session_state.detector.cv_pipeline.canny_high = canny_high
    st.session_state.detector.cv_pipeline.min_contour_area = min_area
    st.session_state.detector.dl_simulator.confidence_threshold = conf_threshold


# ─── Header ──────────────────────────────────────────────────────────────────
st.markdown("""
<div class="main-header">
    <h1>🚧 PotholeAI — Intelligent Road Analysis System</h1>
    <p>
        <span class="badge badge-blue">Classical CV</span>
        <span class="badge badge-cyan">Deep Learning</span>
        <span class="badge badge-green">Real-time</span>
        Hybrid edge detection · CNN simulation · GPS tagging · Severity estimation
    </p>
</div>
""", unsafe_allow_html=True)

# ─── Main Tabs ────────────────────────────────────────────────────────────────
tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "📷 Image Analysis",
    "🎥 Video Analysis",
    "📊 Metrics & Evaluation",
    "🗺️ GPS Map & Reports",
    "ℹ️ System Architecture"
])


# ═══ TAB 1: Image Analysis ════════════════════════════════════════════════════
with tab1:
    col_left, col_right = st.columns([1, 1], gap="large")

    with col_left:
        st.markdown('<div class="section-title">Input Source</div>', unsafe_allow_html=True)
        input_mode = st.radio("Select input", ["Upload Image", "Generate Sample", "Camera Feed"],
                               horizontal=True, label_visibility="collapsed")

        frame_to_process = None

        if input_mode == "Upload Image":
            uploaded = st.file_uploader("Upload road image", type=["jpg", "jpeg", "png", "bmp"],
                                         label_visibility="visible")
            if uploaded:
                file_bytes = np.asarray(bytearray(uploaded.read()), dtype=np.uint8)
                frame_to_process = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
                st.image(cv2.cvtColor(frame_to_process, cv2.COLOR_BGR2RGB),
                         caption="Uploaded image", use_container_width=True)

        elif input_mode == "Generate Sample":
            n_potholes = st.slider("Number of potholes", 1, 8, 3)
            if st.button("🎲 Generate Road Image", use_container_width=True):
                frame_to_process, gt = generate_sample_image(640, 480, n_potholes)
                st.image(cv2.cvtColor(frame_to_process, cv2.COLOR_BGR2RGB),
                         caption=f"Synthetic road ({n_potholes} potholes)", use_container_width=True)
                add_log(f"Generated synthetic image with {n_potholes} potholes", "SUCCESS")

        elif input_mode == "Camera Feed":
            st.info("🎥 Camera feed requires running locally. Use the video tab for demo.")
            if st.button("📸 Capture from Camera"):
                cap = cv2.VideoCapture(0)
                if cap.isOpened():
                    ret, frame = cap.read()
                    cap.release()
                    if ret:
                        frame_to_process = frame
                        st.image(cv2.cvtColor(frame_to_process, cv2.COLOR_BGR2RGB),
                                 caption="Camera capture", use_container_width=True)
                else:
                    st.error("No camera found. Try the Generate Sample option.")

        if frame_to_process is not None:
            if st.button("🔍 Detect Potholes", use_container_width=True, type="primary"):
                with st.spinner("Running hybrid CV + DL pipeline..."):
                    result, edges, detections, debug_info = process_single_frame(frame_to_process)
                    st.session_state.last_result_image = result
                    active = [d for d in detections if not d.is_duplicate]
                    add_log(f"Detected {len(active)} potholes in {debug_info['processing_time_ms']:.1f}ms", "SUCCESS")
                    for d in active:
                        add_log(f"  → {d.severity.upper()} conf={d.confidence:.2f} depth≈{d.depth_estimate_cm}cm GPS({d.gps_lat:.4f},{d.gps_lon:.4f})", "INFO")

    with col_right:
        st.markdown('<div class="section-title">Detection Output</div>', unsafe_allow_html=True)
        if st.session_state.last_result_image is not None:
            st.image(cv2.cvtColor(st.session_state.last_result_image, cv2.COLOR_BGR2RGB),
                     caption="Detection result", use_container_width=True)
            if show_cv_debug:
                frame_to_show = frame_to_process if frame_to_process is not None else None
                if frame_to_show is not None:
                    preprocessed = st.session_state.detector.cv_pipeline.preprocess(frame_to_show)
                    edges_img = st.session_state.detector.cv_pipeline.detect_edges(preprocessed)
                    edges_colored = cv2.cvtColor(edges_img, cv2.COLOR_GRAY2RGB)
                    st.image(edges_colored, caption="Edge map (Canny)", use_container_width=True)
        else:
            st.markdown("""
            <div style="background:#111827;border:1px dashed #1f2937;border-radius:12px;
                        padding:60px 20px;text-align:center;color:#374151;">
                <div style="font-size:2.5rem;margin-bottom:12px">🚧</div>
                <div style="font-size:0.9rem">Detection results appear here</div>
                <div style="font-size:0.75rem;color:#1f2937;margin-top:6px">Upload or generate an image, then click Detect</div>
            </div>
            """, unsafe_allow_html=True)

    # Detection log
    if st.session_state.all_detections:
        active = [d for d in st.session_state.all_detections if not d.is_duplicate]
        st.markdown('<div class="section-title">Recent Detections</div>', unsafe_allow_html=True)
        cols = st.columns(4)
        with cols[0]:
            st.metric("Total Potholes", len(active))
        with cols[1]:
            crit = sum(1 for d in active if d.severity == 'critical')
            st.metric("Critical", crit)
        with cols[2]:
            avg_conf = np.mean([d.confidence for d in active]) if active else 0
            st.metric("Avg Confidence", f"{avg_conf:.1%}")
        with cols[3]:
            avg_depth = np.mean([d.depth_estimate_cm for d in active]) if active else 0
            st.metric("Avg Depth", f"{avg_depth:.1f} cm")

        for det in reversed(active[-6:]):
            sev_colors = {'critical': '#ef4444', 'severe': '#f97316', 'moderate': '#eab308', 'minor': '#22c55e'}
            color = sev_colors.get(det.severity, '#64748b')
            st.markdown(f"""
            <div class="detection-item {det.severity}">
                <div style="display:flex;justify-content:space-between;align-items:center">
                    <span>{render_severity_badge(det.severity)} Frame #{det.frame_id}</span>
                    <span style="font-family:monospace;color:#60a5fa">{det.confidence:.1%} conf</span>
                </div>
                <div style="font-size:0.78rem;color:#64748b;margin-top:6px">
                    📐 {det.bbox[2]}×{det.bbox[3]}px &nbsp;|&nbsp;
                    🕳️ ~{det.depth_estimate_cm}cm depth &nbsp;|&nbsp;
                    📍 {det.gps_lat:.4f}, {det.gps_lon:.4f}
                </div>
            </div>
            """, unsafe_allow_html=True)

    # Processing log
    if st.session_state.processing_log:
        st.markdown('<div class="section-title">Processing Log</div>', unsafe_allow_html=True)
        log_html = "<br>".join(st.session_state.processing_log[-30:])
        st.markdown(f'<div class="log-container">{log_html}</div>', unsafe_allow_html=True)


# ═══ TAB 2: Video Analysis ════════════════════════════════════════════════════
with tab2:
    st.markdown('<div class="section-title">Video / Multi-Frame Analysis</div>', unsafe_allow_html=True)

    col_v1, col_v2 = st.columns([1, 2], gap="large")

    with col_v1:
        video_mode = st.radio("Video source", ["Upload Video", "Generate Demo Sequence"], horizontal=False)
        n_frames = st.slider("Frames to process", 5, 60, 20)
        frame_skip = st.slider("Frame skip (process every Nth)", 1, 5, 1)
        show_progress = st.checkbox("Show per-frame preview", value=True)

        if video_mode == "Upload Video":
            video_file = st.file_uploader("Upload video", type=["mp4", "avi", "mov", "mkv"])
        else:
            video_file = None

        run_video = st.button("▶ Analyze Video", use_container_width=True, type="primary")

    with col_v2:
        if run_video:
            frames = []
            if video_mode == "Generate Demo Sequence" or video_file is None:
                with st.spinner("Generating synthetic road sequence..."):
                    frames = generate_video_frames(n_frames=n_frames)
                    add_log(f"Generated {len(frames)}-frame synthetic sequence", "SUCCESS")
            else:
                with tempfile.NamedTemporaryFile(delete=False, suffix='.mp4') as tmp:
                    tmp.write(video_file.read())
                    tmp_path = tmp.name
                cap = cv2.VideoCapture(tmp_path)
                total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
                st.info(f"Video: {total} total frames. Processing every {frame_skip}th.")
                count = 0
                while count < n_frames:
                    ret, frame = cap.read()
                    if not ret:
                        break
                    if count % frame_skip == 0:
                        frames.append(frame)
                    count += 1
                cap.release()
                os.unlink(tmp_path)

            st.session_state.detector.reset_session()
            st.session_state.metrics_engine.reset()

            progress_bar = st.progress(0)
            frame_placeholder = st.empty()
            stats_placeholder = st.empty()
            all_frame_detections = []

            for i, frame in enumerate(frames):
                result, edges, dets, debug = process_single_frame(frame)
                all_frame_detections.append((dets, debug))
                progress_bar.progress((i + 1) / len(frames))
                if show_progress and (i % 3 == 0 or i == len(frames) - 1):
                    frame_placeholder.image(
                        cv2.cvtColor(result, cv2.COLOR_BGR2RGB),
                        caption=f"Frame {i+1}/{len(frames)} — {len([d for d in dets if not d.is_duplicate])} potholes",
                        use_container_width=True
                    )
                active_count = sum(1 for d, _ in all_frame_detections for d_ in d if not d_.is_duplicate)
                stats_placeholder.markdown(f"""
                <div style="display:grid;grid-template-columns:1fr 1fr 1fr;gap:12px;margin-top:12px">
                    <div class="metric-card info"><div class="value">{i+1}</div><div class="label">Frames done</div></div>
                    <div class="metric-card minor"><div class="value">{active_count}</div><div class="label">Unique potholes</div></div>
                    <div class="metric-card fps"><div class="value">{debug.get('processing_time_ms', 0):.0f}ms</div><div class="label">Frame time</div></div>
                </div>
                """, unsafe_allow_html=True)

            progress_bar.empty()
            add_log(f"Video analysis complete: {len(frames)} frames processed", "SUCCESS")
            summary = st.session_state.metrics_engine.get_summary()
            st.success(f"✅ Done! {summary['session']['unique_potholes']} unique potholes detected. See Metrics tab for full report.")


# ═══ TAB 3: Metrics ═══════════════════════════════════════════════════════════
with tab3:
    summary = st.session_state.metrics_engine.get_summary()
    metrics = summary['metrics']
    session = summary['session']
    severity = summary['severity_breakdown']

    st.markdown("""
    <div class="section-title">Model Performance Metrics</div>
    """, unsafe_allow_html=True)

    col_m1, col_m2, col_m3, col_m4 = st.columns(4)
    with col_m1:
        st.metric("Precision", f"{metrics.get('precision', 0):.1%}", help="TP / (TP + FP)")
    with col_m2:
        st.metric("Recall", f"{metrics.get('recall', 0):.1%}", help="TP / (TP + FN)")
    with col_m3:
        st.metric("F1 Score", f"{metrics.get('f1', 0):.1%}")
    with col_m4:
        st.metric("mAP@0.5", f"{metrics.get('map_50', 0):.1%}", help="Mean Average Precision at IoU=0.5")

    col_m5, col_m6, col_m7, col_m8 = st.columns(4)
    with col_m5:
        st.metric("Frames Processed", session['total_frames'])
    with col_m6:
        st.metric("Unique Potholes", session['unique_potholes'])
    with col_m7:
        st.metric("Duplicates Suppressed", session['duplicates_suppressed'])
    with col_m8:
        st.metric("Avg FPS", f"{session['avg_fps']}")

    st.markdown('<div class="section-title">Severity Distribution</div>', unsafe_allow_html=True)
    sev_cols = st.columns(4)
    sev_data = [
        ('critical', '🔴', '#ef4444'),
        ('severe', '🟠', '#f97316'),
        ('moderate', '🟡', '#eab308'),
        ('minor', '🟢', '#22c55e'),
    ]
    for col, (sev, icon, color) in zip(sev_cols, sev_data):
        with col:
            count = severity.get(sev, 0)
            total = max(session['unique_potholes'], 1)
            pct = count / total * 100
            st.markdown(f"""
            <div class="metric-card {sev}" style="border-top:3px solid {color}">
                <div class="value">{count}</div>
                <div class="label">{icon} {sev.upper()}</div>
                <div style="font-size:0.72rem;color:#4b5563;margin-top:4px">{pct:.0f}% of total</div>
            </div>
            """, unsafe_allow_html=True)

    if metrics.get('precision_curve'):
        st.markdown('<div class="section-title">Precision-Recall Curve</div>', unsafe_allow_html=True)
        import plotly.graph_objects as go
        pr_data = metrics['precision_curve']
        rc_data = metrics['recall_curve']
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=rc_data, y=pr_data,
            mode='lines+markers',
            name='PR Curve',
            line=dict(color='#3b82f6', width=2),
            marker=dict(size=5, color='#60a5fa'),
            fill='tozeroy',
            fillcolor='rgba(59,130,246,0.1)',
        ))
        fig.add_trace(go.Scatter(
            x=[0, 1], y=[1, 0],
            mode='lines',
            name='Random baseline',
            line=dict(color='#374151', width=1, dash='dot'),
        ))
        fig.update_layout(
            xaxis_title='Recall',
            yaxis_title='Precision',
            template='plotly_dark',
            paper_bgcolor='#111827',
            plot_bgcolor='#0d1117',
            font=dict(family='Space Grotesk', size=12, color='#94a3b8'),
            legend=dict(bgcolor='#111827', bordercolor='#1f2937'),
            margin=dict(l=40, r=20, t=20, b=40),
            height=320,
        )
        st.plotly_chart(fig, use_container_width=True)

    depth_stats = summary['depth_stats']
    if depth_stats['avg_depth_cm'] > 0:
        st.markdown('<div class="section-title">Depth Estimation Stats</div>', unsafe_allow_html=True)
        dc1, dc2, dc3 = st.columns(3)
        with dc1:
            st.metric("Avg Depth", f"{depth_stats['avg_depth_cm']} cm")
        with dc2:
            st.metric("Max Depth", f"{depth_stats['max_depth_cm']} cm")
        with dc3:
            st.metric("Min Depth", f"{depth_stats['min_depth_cm']} cm")

    if session['unique_potholes'] > 0:
        st.markdown('<div class="section-title">Export Data</div>', unsafe_allow_html=True)
        export_json = st.session_state.metrics_engine.export_json()
        st.download_button(
            "⬇️ Download Detection Report (JSON)",
            data=export_json,
            file_name="pothole_detection_report.json",
            mime="application/json",
        )


# ═══ TAB 4: GPS Map ═══════════════════════════════════════════════════════════
with tab4:
    summary = st.session_state.metrics_engine.get_summary()
    gps_points = summary.get('gps_points', [])

    st.markdown('<div class="section-title">GPS Pothole Map</div>', unsafe_allow_html=True)

    if gps_points:
        import plotly.graph_objects as go
        sev_color_map = {
            'critical': 'red', 'severe': 'orange', 'moderate': 'yellow', 'minor': 'green'
        }
        for sev in ['critical', 'severe', 'moderate', 'minor']:
            pts = [p for p in gps_points if p['severity'] == sev]
            if not pts:
                continue

        fig_map = go.Figure()
        for sev, mcolor in sev_color_map.items():
            pts = [p for p in gps_points if p['severity'] == sev]
            if not pts:
                continue
            fig_map.add_trace(go.Scattergeo(
                lat=[p['lat'] for p in pts],
                lon=[p['lon'] for p in pts],
                mode='markers',
                name=sev.title(),
                marker=dict(size=10, color=mcolor, symbol='circle', opacity=0.8,
                            line=dict(width=1, color='white')),
                text=[f"{sev.upper()}<br>Conf: {p['confidence']:.1%}<br>{p['lat']:.5f}, {p['lon']:.5f}" for p in pts],
                hoverinfo='text',
            ))
        if gps_points:
            center_lat = np.mean([p['lat'] for p in gps_points])
            center_lon = np.mean([p['lon'] for p in gps_points])
        else:
            center_lat, center_lon = gps_lat, gps_lon

        fig_map.update_layout(
            geo=dict(
                center=dict(lat=center_lat, lon=center_lon),
                projection_scale=50000,
                showland=True, landcolor='#1a1f2e',
                showocean=True, oceancolor='#0d1117',
                showlakes=True, lakecolor='#0d1117',
                showcountries=True, countrycolor='#374151',
                showcoastlines=True, coastlinecolor='#374151',
                bgcolor='#0d1117',
            ),
            paper_bgcolor='#111827',
            font=dict(family='Space Grotesk', color='#94a3b8'),
            legend=dict(bgcolor='#111827', bordercolor='#1f2937'),
            height=450,
            margin=dict(l=0, r=0, t=0, b=0),
        )
        st.plotly_chart(fig_map, use_container_width=True)

        st.markdown('<div class="section-title">GPS Coordinates Log</div>', unsafe_allow_html=True)
        gps_rows = ""
        for i, p in enumerate(gps_points[-20:]):
            sev_color = {'critical': '#ef4444', 'severe': '#f97316', 'moderate': '#eab308', 'minor': '#22c55e'}
            color = sev_color.get(p['severity'], '#64748b')
            gps_rows += f"""
            <tr style="border-bottom:1px solid #1f2937">
                <td style="padding:6px 8px;color:#64748b">{i+1}</td>
                <td style="padding:6px 8px;font-family:monospace;color:#94a3b8">{p['lat']:.6f}</td>
                <td style="padding:6px 8px;font-family:monospace;color:#94a3b8">{p['lon']:.6f}</td>
                <td style="padding:6px 8px"><span style="color:{color};font-weight:600">{p['severity'].upper()}</span></td>
                <td style="padding:6px 8px;color:#60a5fa">{p['confidence']:.1%}</td>
            </tr>"""
        st.markdown(f"""
        <div style="background:#0d1117;border:1px solid #1f2937;border-radius:10px;overflow:hidden">
        <table style="width:100%;border-collapse:collapse;font-size:0.82rem">
            <thead><tr style="background:#111827">
                <th style="padding:8px;text-align:left;color:#64748b">#</th>
                <th style="padding:8px;text-align:left;color:#64748b">Latitude</th>
                <th style="padding:8px;text-align:left;color:#64748b">Longitude</th>
                <th style="padding:8px;text-align:left;color:#64748b">Severity</th>
                <th style="padding:8px;text-align:left;color:#64748b">Confidence</th>
            </tr></thead>
            <tbody>{gps_rows}</tbody>
        </table></div>
        """, unsafe_allow_html=True)
    else:
        st.markdown("""
        <div style="background:#111827;border:1px dashed #1f2937;border-radius:12px;
                    padding:60px 20px;text-align:center;color:#374151">
            <div style="font-size:2.5rem;margin-bottom:12px">📍</div>
            <div style="font-size:0.9rem">No GPS data yet. Process images/video to see pothole locations.</div>
        </div>
        """, unsafe_allow_html=True)


# ═══ TAB 5: Architecture ══════════════════════════════════════════════════════
with tab5:
    st.markdown('<div class="section-title">System Architecture — Hybrid CV + DL Pipeline</div>', unsafe_allow_html=True)
    st.markdown("""
    <div style="background:#111827;border:1px solid #1f2937;border-radius:12px;padding:24px;margin-bottom:20px">

    <h4 style="color:#60a5fa;margin:0 0 16px">🔬 Classical CV Stage</h4>
    <p style="color:#94a3b8;font-size:0.88rem;line-height:1.7">
    The classical CV pipeline performs multi-scale <strong style="color:#e2e8f0">Canny edge detection</strong> across two threshold bands,
    combining the results to capture both fine crack edges and large pothole boundaries. CLAHE contrast enhancement
    normalizes varying lighting conditions. Morphological closing bridges edge gaps. Contour detection filters
    candidates by area, aspect ratio, and convexity. <strong style="color:#e2e8f0">Texture analysis</strong> using gradient variance
    and Sobel responses scores each region's surface irregularity.
    </p>

    <h4 style="color:#22d3ee;margin:20px 0 16px">🧠 Deep Learning Stage</h4>
    <p style="color:#94a3b8;font-size:0.88rem;line-height:1.7">
    The DL stage (YOLOv8/CNN simulation in this demo — replace with actual model weights) takes the CV
    region proposals as spatial priors and applies learned feature extraction. Non-Maximum Suppression at
    configurable IoU threshold removes overlapping detections. Confidence calibration combines CV texture
    scores with DL backbone features for final scoring.
    </p>

    <h4 style="color:#4ade80;margin:20px 0 16px">📊 Post-Processing Stage</h4>
    <p style="color:#94a3b8;font-size:0.88rem;line-height:1.7">
    Each confirmed detection goes through <strong style="color:#e2e8f0">severity estimation</strong> (relative size + texture + edges → minor/moderate/severe/critical),
    <strong style="color:#e2e8f0">depth estimation</strong> (calibrated by severity score, 2–20cm range),
    <strong style="color:#e2e8f0">perceptual hash duplicate detection</strong> (DCT-based pHash + spatial proximity),
    and <strong style="color:#e2e8f0">GPS tagging</strong>.
    </p>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("""
    <div style="background:#111827;border:1px solid #1f2937;border-radius:12px;padding:24px">
    <h4 style="color:#f1f5f9;margin:0 0 16px">🔧 Tech Stack</h4>
    <div style="display:grid;grid-template-columns:1fr 1fr;gap:16px;font-size:0.85rem;color:#94a3b8">
        <div>
            <strong style="color:#60a5fa">CV & Image Processing</strong><br>
            OpenCV 4.x — Canny, contours, CLAHE, morphology<br>
            NumPy — feature computation, NMS<br>
            DCT-based perceptual hashing for dedup
        </div>
        <div>
            <strong style="color:#22d3ee">Deep Learning</strong><br>
            YOLOv8 / TensorFlow / PyTorch ready<br>
            Confidence calibration pipeline<br>
            NMS with configurable IoU threshold
        </div>
        <div>
            <strong style="color:#4ade80">Backend & UI</strong><br>
            Streamlit — interactive web interface<br>
            Plotly — metrics visualization<br>
            JSON export for integration
        </div>
        <div>
            <strong style="color:#f59e0b">Features</strong><br>
            GPS tagging (hardware-ready)<br>
            Severity + depth estimation<br>
            Precision / Recall / mAP@50 metrics
        </div>
    </div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("""
    <div style="background:#0d2818;border:1px solid #14532d;border-radius:12px;padding:20px;margin-top:16px">
    <h4 style="color:#4ade80;margin:0 0 12px">🚀 Deploying with Real Models</h4>
    <div style="font-family:monospace;font-size:0.82rem;color:#86efac;line-height:2">
        # Install YOLOv8<br>
        pip install ultralytics<br><br>
        # In DeepLearningSimulator.predict(), replace with:<br>
        from ultralytics import YOLO<br>
        model = YOLO('yolov8n.pt')  # or your custom trained weights<br>
        results = model(frame)<br>
        # Map results to detection format<br><br>
        # Or TensorFlow/Keras CNN:<br>
        import tensorflow as tf<br>
        model = tf.saved_model.load('pothole_model/')<br>
        predictions = model(tf.expand_dims(frame_tensor, 0))
    </div>
    </div>
    """, unsafe_allow_html=True)
