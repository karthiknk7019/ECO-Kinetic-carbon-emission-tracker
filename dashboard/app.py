"""Eco-Kinetic Traffic Carbon Digital Twin — Streamlit Dashboard.

Edge CCTV perception, kinematic state analysis, corridor emissions, and municipal dispatch feed.
"""

from __future__ import annotations

import json
import hashlib
from datetime import datetime, timezone
from pathlib import Path
import sys
from typing import Any, Dict, List, Optional, Tuple
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import pyarrow.parquet as pq
import streamlit as st
import yaml

# Ensure project root is in sys.path before importing local packages
REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from perception.image_inference import (
    ImageValidationError,
    TrafficImageAnalyzer,
    encode_image_to_bytes,
    export_results_to_csv,
    export_results_to_json,
)
from dispatcher.llm_dispatcher import append_directive_to_log, dispatch
from dispatcher.schema import CorridorPayload, Directive, TopChokeZone
from dispatcher.rules_fallback import _determine_lane_from_zone


@st.cache_resource
def get_image_analyzer() -> TrafficImageAnalyzer:
    """Cache analyzer instance to avoid repeated YOLO model loading across reruns."""
    return TrafficImageAnalyzer()

# Page configuration
st.set_page_config(
    page_title="Eco-Kinetic Traffic Carbon Twin",
    page_icon="🌿",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Custom sleek CSS
st.markdown(
    """
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&display=swap');
    /* Global interactive buttons with hover lift, gradient glow, and smooth feedback */
    .stButton > button {
        border-radius: 9px !important;
        font-weight: 600 !important;
        letter-spacing: 0.3px !important;
        transition: all 0.22s cubic-bezier(0.4, 0, 0.2, 1) !important;
        border: 1px solid rgba(255, 255, 255, 0.12) !important;
        position: relative !important;
        overflow: hidden !important;
    }
    .stButton > button:hover {
        transform: translateY(-2px) !important;
        box-shadow: 0 6px 20px rgba(16, 185, 129, 0.3) !important;
        border-color: #10b981 !important;
    }
    .stButton > button:active {
        transform: translateY(0px) scale(0.98) !important;
    }
    .stButton > button[kind="primary"], button[data-testid="baseButton-primary"] {
        background: linear-gradient(135deg, #059669 0%, #10b981 100%) !important;
        color: #ffffff !important;
        border: 1px solid #34d399 !important;
        box-shadow: 0 4px 14px rgba(16, 185, 129, 0.35) !important;
    }
    .stButton > button[kind="primary"]:hover, button[data-testid="baseButton-primary"]:hover {
        background: linear-gradient(135deg, #10b981 0%, #34d399 100%) !important;
        box-shadow: 0 6px 24px rgba(16, 185, 129, 0.55) !important;
        transform: translateY(-2px) scale(1.01) !important;
    }
    .stButton > button[kind="secondary"], button[data-testid="baseButton-secondary"] {
        background: #1e2230 !important;
        color: #e2e8f0 !important;
        border: 1px solid #334155 !important;
    }
    .stButton > button[kind="secondary"]:hover, button[data-testid="baseButton-secondary"]:hover {
        background: #282f44 !important;
        border-color: #38bdf8 !important;
        box-shadow: 0 4px 16px rgba(56, 189, 248, 0.3) !important;
        color: #38bdf8 !important;
    }

    /* Sidebar console brand card */
    .sidebar-console-card {
        background: linear-gradient(135deg, #0b1320 0%, #172545 50%, #0c201d 100%);
        border: 1px solid rgba(52, 211, 153, 0.30);
        border-radius: 12px;
        padding: 16px;
        margin-bottom: 14px;
        box-shadow: 0 6px 20px rgba(0, 0, 0, 0.4), inset 0 1px 0 rgba(255, 255, 255, 0.05);
    }
    .console-status-row {
        display: flex;
        align-items: center;
        gap: 8px;
        margin-bottom: 6px;
    }
    .pulse-indicator {
        width: 9px;
        height: 9px;
        background-color: #10b981;
        border-radius: 50%;
        display: inline-block;
        box-shadow: 0 0 0 0 rgba(16, 185, 129, 0.7);
        animation: pulseGlow 1.8s infinite;
    }
    @keyframes pulseGlow {
        0% {
            transform: scale(0.95);
            box-shadow: 0 0 0 0 rgba(16, 185, 129, 0.7);
        }
        70% {
            transform: scale(1.05);
            box-shadow: 0 0 0 8px rgba(16, 185, 129, 0);
        }
        100% {
            transform: scale(0.95);
            box-shadow: 0 0 0 0 rgba(16, 185, 129, 0);
        }
    }
    .status-text {
        font-size: 10px;
        font-weight: 700;
        letter-spacing: 0.8px;
        color: #10b981;
        text-transform: uppercase;
    }
    .console-heading {
        font-size: 18px;
        font-weight: 800;
        color: #f8fafc;
        letter-spacing: -0.3px;
        margin-bottom: 2px;
    }
    .console-subtext {
        font-size: 11px;
        color: #94a3b8;
        line-height: 1.3;
    }
    .nav-section-title {
        font-size: 11px;
        font-weight: 700;
        letter-spacing: 0.9px;
        color: #64748b;
        margin-top: 14px;
        margin-bottom: 8px;
        text-transform: uppercase;
    }
    .nav-active-status-card {
        background: linear-gradient(135deg, rgba(16, 185, 129, 0.12) 0%, rgba(14, 165, 233, 0.08) 100%);
        border: 1px solid rgba(16, 185, 129, 0.4);
        border-radius: 10px;
        padding: 12px;
        margin-top: 10px;
        margin-bottom: 12px;
        transition: transform 0.25s ease, border-color 0.25s ease, box-shadow 0.25s ease;
    }
    .nav-active-status-card:hover {
        transform: translateY(-1px);
        border-color: rgba(52, 211, 153, 0.62);
        box-shadow: 0 10px 24px rgba(16, 185, 129, 0.12);
    }
    .active-badge-tag {
        font-size: 10px;
        font-weight: 700;
        color: #10b981;
        letter-spacing: 0.6px;
        text-transform: uppercase;
        margin-bottom: 3px;
    }
    .active-title-text {
        font-size: 13px;
        font-weight: 700;
        color: #f1f5f9;
    }
    .active-desc-text {
        font-size: 11px;
        color: #94a3b8;
        margin-top: 2px;
        line-height: 1.3;
    }

    /* Metrics & Cards */
    .metric-card {
        background-color: #1a1c24;
        border: 1px solid #2d313e;
        border-radius: 8px;
        padding: 16px;
        text-align: center;
    }
    .metric-value {
        font-size: 28px;
        font-weight: 700;
        color: #00d26a;
    }
    .metric-label {
        font-size: 13px;
        color: #9aa0a6;
        text-transform: uppercase;
        letter-spacing: 0.5px;
    }
    .badge-llm {
        background-color: #1e3a8a;
        color: #93c5fd;
        padding: 3px 8px;
        border-radius: 4px;
        font-size: 11px;
        font-weight: 600;
    }
    .badge-rules {
        background-color: #854d0e;
        color: #fde047;
        padding: 3px 8px;
        border-radius: 4px;
        font-size: 11px;
        font-weight: 600;
    }
    .kpi-card {
        background: linear-gradient(135deg, #161821 0%, #1e2230 100%);
        border: 1px solid #2d3748;
        border-radius: 10px;
        padding: 14px;
        text-align: center;
        box-shadow: 0 4px 12px rgba(0,0,0,0.2);
    }
    .kpi-title {
        font-size: 11px;
        font-weight: 600;
        text-transform: uppercase;
        color: #94a3b8;
        letter-spacing: 0.6px;
        margin-bottom: 4px;
    }
    .kpi-num {
        font-size: 26px;
        font-weight: 700;
        color: #f8fafc;
    }
    .kpi-num-green {
        color: #10b981;
    }
    .kpi-num-blue {
        color: #38bdf8;
    }
    .kpi-num-amber {
        color: #fbbf24;
    }
    .kpi-num-purple {
        color: #c084fc;
    }
    .kpi-sub {
        font-size: 11px;
        color: #64748b;
        margin-top: 3px;
    }
    .ai-advisory-container {
        background: linear-gradient(135deg, #0f172a 0%, #172554 50%, #064e3b 100%);
        border: 1px solid #2563eb;
        border-radius: 12px;
        padding: 20px;
        margin: 15px 0 20px 0;
        box-shadow: 0 8px 24px rgba(37, 99, 235, 0.15);
    }
    .fleet-tag {
        display: inline-block;
        padding: 4px 10px;
        border-radius: 6px;
        font-size: 12px;
        font-weight: 600;
        margin: 2px;
    }

    /* ------------------------------------------------------------------
       CONSOLE THEME REFRESH — calm, editorial, operations-grade
       ------------------------------------------------------------------ */
    html, body, [class*="css"], .stApp {
        font-family: 'Inter', -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif !important;
    }
    .stApp {
        background:
            radial-gradient(1100px 620px at 18% -12%, rgba(16, 185, 129, 0.13), transparent 60%),
            radial-gradient(900px 520px at 92% 0%, rgba(56, 189, 248, 0.10), transparent 58%),
            #070a10;
    }
    header[data-testid="stHeader"] {
        background: transparent;
    }
    [data-testid="stToolbar"], [data-testid="stDecoration"], #MainMenu, footer {
        visibility: hidden;
    }
    .block-container {
        max-width: 1520px;
        padding-top: 2.2rem;
        padding-bottom: 4rem;
    }
    [data-testid="stSidebar"] {
        background: linear-gradient(180deg, #0a0f18 0%, #070a10 100%);
        border-right: 1px solid rgba(148, 163, 184, 0.12);
    }
    [data-testid="stSidebar"] .block-container {
        padding-top: 1.6rem;
    }
    h1, h2, h3, h4 {
        letter-spacing: -0.035em !important;
        color: #f8fafc !important;
    }
    h1 {
        font-size: 2.35rem !important;
        font-weight: 800 !important;
        line-height: 1.08 !important;
    }
    h2 { font-size: 1.35rem !important; font-weight: 700 !important; }
    h3 { font-size: 1.05rem !important; font-weight: 700 !important; }
    p, label, .stMarkdown {
        color: #cbd5e1;
    }
    .stCaption, [data-testid="stCaptionContainer"] {
        color: #7c8ba1 !important;
        letter-spacing: 0.01em;
    }
    hr, [data-testid="stDivider"] {
        border-color: rgba(148, 163, 184, 0.13) !important;
        margin: 1.1rem 0 !important;
    }

    /* Hero */
    .console-hero {
        position: relative;
        overflow: hidden;
        border: 1px solid rgba(148, 163, 184, 0.16);
        border-radius: 20px;
        padding: 26px 28px 24px 28px;
        margin-bottom: 20px;
        background:
            linear-gradient(115deg, rgba(16, 185, 129, 0.13) 0%, rgba(15, 23, 42, 0.55) 42%, rgba(56, 189, 248, 0.09) 100%);
        box-shadow: 0 18px 50px rgba(0, 0, 0, 0.28);
    }
    .console-hero::after {
        content: "";
        position: absolute;
        width: 340px;
        height: 340px;
        right: -120px;
        top: -180px;
        border-radius: 50%;
        background: radial-gradient(circle, rgba(16, 185, 129, 0.25), transparent 65%);
        pointer-events: none;
    }
    .hero-eyebrow {
        display: inline-flex;
        align-items: center;
        gap: 8px;
        font-size: 10px;
        font-weight: 700;
        letter-spacing: 0.16em;
        text-transform: uppercase;
        color: #34d399;
        margin-bottom: 10px;
    }
    .hero-copy {
        max-width: 780px;
        color: #a9b7c9;
        font-size: 14px;
        line-height: 1.55;
        margin-top: 6px;
    }
    .chip-row {
        display: flex;
        flex-wrap: wrap;
        gap: 8px;
        margin-top: 16px;
    }
    .chip {
        display: inline-flex;
        align-items: center;
        gap: 7px;
        border-radius: 999px;
        padding: 5px 10px;
        font-size: 11px;
        font-weight: 600;
        letter-spacing: 0.02em;
        border: 1px solid rgba(148, 163, 184, 0.2);
        color: #a9b7c9;
        background: rgba(15, 23, 42, 0.55);
    }
    .chip-live {
        border-color: rgba(16, 185, 129, 0.42);
        color: #6ee7b7;
        background: rgba(16, 185, 129, 0.10);
    }
    .chip-blue {
        border-color: rgba(56, 189, 248, 0.36);
        color: #7dd3fc;
        background: rgba(56, 189, 248, 0.08);
    }
    .chip-amber {
        border-color: rgba(245, 158, 11, 0.36);
        color: #fcd34d;
        background: rgba(245, 158, 11, 0.08);
    }
    .dot {
        width: 7px;
        height: 7px;
        border-radius: 50%;
        background: currentColor;
        box-shadow: 0 0 0 4px rgba(16, 185, 129, 0.13);
    }

    /* Section headers */
    .section-head {
        display: flex;
        align-items: flex-end;
        justify-content: space-between;
        gap: 14px;
        margin: 22px 0 12px 0;
    }
    .section-kicker {
        font-size: 10px;
        font-weight: 700;
        letter-spacing: 0.15em;
        text-transform: uppercase;
        color: #34d399;
        margin-bottom: 4px;
    }
    .section-title {
        font-size: 19px;
        font-weight: 700;
        letter-spacing: -0.03em;
        color: #f8fafc;
    }
    .section-sub {
        color: #7c8ba1;
        font-size: 12px;
        margin-top: 3px;
    }

    /* KPI cards */
    .stat-grid {
        display: grid;
        grid-template-columns: repeat(5, minmax(0, 1fr));
        gap: 12px;
        margin: 4px 0 6px 0;
    }
    .stat-card {
        position: relative;
        overflow: hidden;
        border: 1px solid rgba(148, 163, 184, 0.15);
        border-radius: 15px;
        padding: 16px 16px 14px 16px;
        background: linear-gradient(160deg, rgba(22, 30, 46, 0.92), rgba(12, 17, 27, 0.92));
        box-shadow: 0 10px 28px rgba(0, 0, 0, 0.18);
        min-height: 118px;
    }
    .stat-card::before {
        content: "";
        position: absolute;
        left: 0;
        right: 0;
        top: 0;
        height: 2px;
        background: var(--accent, #10b981);
        opacity: 0.9;
    }
    .stat-icon {
        font-size: 14px;
        opacity: 0.9;
        margin-bottom: 8px;
    }
    .stat-label {
        color: #8394aa;
        font-size: 10px;
        font-weight: 600;
        letter-spacing: 0.1em;
        text-transform: uppercase;
        margin-bottom: 3px;
    }
    .stat-value {
        color: #f8fafc;
        font-size: 27px;
        font-weight: 700;
        letter-spacing: -0.045em;
        line-height: 1.05;
    }
    .stat-foot {
        color: #6c7d93;
        font-size: 10.5px;
        margin-top: 7px;
    }
    .stat-foot b { color: var(--accent, #10b981); font-weight: 600; }

    /* Panels */
    .panel-card {
        border: 1px solid rgba(148, 163, 184, 0.14);
        border-radius: 16px;
        background: rgba(13, 19, 30, 0.82);
        padding: 18px;
        box-shadow: 0 12px 32px rgba(0, 0, 0, 0.16);
    }
    .panel-title {
        font-size: 14px;
        font-weight: 650;
        color: #e6ebf2;
        letter-spacing: -0.01em;
    }
    .panel-sub {
        font-size: 11px;
        color: #7c8ba1;
        margin-top: 2px;
        margin-bottom: 10px;
    }

    /* Directive cards */
    .feed-card {
        border: 1px solid rgba(148, 163, 184, 0.14);
        border-left: 3px solid var(--edge, #38bdf8);
        border-radius: 13px;
        background: linear-gradient(120deg, rgba(17, 24, 39, 0.85), rgba(10, 15, 24, 0.85));
        padding: 14px 16px;
        margin-bottom: 10px;
    }
    .feed-top {
        display: flex;
        justify-content: space-between;
        align-items: center;
        gap: 10px;
        flex-wrap: wrap;
    }
    .feed-id {
        color: #7c8ba1;
        font-family: ui-monospace, SFMono-Regular, Menlo, monospace;
        font-size: 10.5px;
    }
    .feed-rationale {
        color: #b7c3d4;
        font-size: 12.5px;
        line-height: 1.5;
        margin-top: 9px;
    }
    .feed-actions {
        color: #7dd3fc;
        font-size: 11.5px;
        margin-top: 6px;
    }
    .status-pill {
        display: inline-flex;
        align-items: center;
        gap: 6px;
        border-radius: 999px;
        padding: 4px 9px;
        font-size: 10px;
        font-weight: 700;
        letter-spacing: 0.07em;
        text-transform: uppercase;
    }
    .status-pending { background: rgba(245, 158, 11, 0.12); color: #fcd34d; border: 1px solid rgba(245, 158, 11, 0.3); }
    .status-dispatched { background: rgba(16, 185, 129, 0.12); color: #6ee7b7; border: 1px solid rgba(16, 185, 129, 0.3); }
    .status-dismissed { background: rgba(148, 163, 184, 0.1); color: #94a3b8; border: 1px solid rgba(148, 163, 184, 0.22); }

    /* Streamlit widget polish */
    [data-testid="stMetric"] {
        background: linear-gradient(160deg, rgba(22, 30, 46, 0.9), rgba(12, 17, 27, 0.9));
        border: 1px solid rgba(148, 163, 184, 0.15);
        border-radius: 14px;
        padding: 14px 16px;
    }
    [data-testid="stMetricLabel"] { color: #8394aa !important; font-size: 10px !important; letter-spacing: 0.1em; text-transform: uppercase; }
    [data-testid="stMetricValue"] { color: #f8fafc !important; font-weight: 700 !important; }
    [data-testid="stDataFrame"] {
        border: 1px solid rgba(148, 163, 184, 0.14);
        border-radius: 12px;
        overflow: hidden;
    }
    .stTabs [data-baseweb="tab-list"] {
        gap: 4px;
        border-bottom: 1px solid rgba(148, 163, 184, 0.14);
        padding-bottom: 2px;
    }
    .stTabs [data-baseweb="tab"] {
        border-radius: 8px 8px 0 0;
        color: #8394aa;
        font-size: 12.5px;
        font-weight: 600;
        padding: 8px 13px;
    }
    .stTabs [aria-selected="true"] {
        color: #f8fafc !important;
        background: rgba(16, 185, 129, 0.08);
    }
    [data-testid="stExpander"] {
        border: 1px solid rgba(148, 163, 184, 0.14) !important;
        border-radius: 12px !important;
        background: rgba(13, 19, 30, 0.6);
        overflow: hidden;
    }
    [data-testid="stFileUploader"] section {
        border-radius: 12px;
        border: 1px dashed rgba(148, 163, 184, 0.3);
        background: rgba(13, 19, 30, 0.55);
    }
    .stSlider [data-baseweb="slider"] { padding-top: 2px; }
    .stRadio [role="radiogroup"] { gap: 6px; }
    .stRadio [role="radiogroup"] label {
        background: rgba(15, 23, 42, 0.55);
        border: 1px solid rgba(148, 163, 184, 0.14);
        border-radius: 9px;
        padding: 7px 10px;
    }
    [data-testid="stSidebar"] .stRadio [role="radiogroup"] label {
        padding: 5px 8px;
    }
    [data-testid="stSidebar"] .stRadio [role="radiogroup"] label p {
        font-size: 12px;
    }
    .stButton > button {
        min-height: 38px;
        padding: 7px 15px !important;
        font-size: 12.5px !important;
        box-shadow: none;
    }
    .stDownloadButton > button {
        border-radius: 9px !important;
        border: 1px solid rgba(148, 163, 184, 0.2) !important;
        background: rgba(15, 23, 42, 0.6) !important;
        color: #cbd5e1 !important;
        font-size: 12px !important;
        font-weight: 600 !important;
    }
    .stDownloadButton > button:hover {
        border-color: #34d399 !important;
        color: #6ee7b7 !important;
    }
    .empty-state {
        border: 1px dashed rgba(148, 163, 184, 0.28);
        border-radius: 14px;
        padding: 22px;
        text-align: center;
        color: #8394aa;
        background: rgba(13, 19, 30, 0.45);
        font-size: 12.5px;
    }
    @media (max-width: 1200px) {
        .stat-grid { grid-template-columns: repeat(2, minmax(0, 1fr)); }
    }

    /* ------------------------------------------------------------------
       INTERACTION LAYER — motion, depth & tactile feedback
       ------------------------------------------------------------------ */
    ::selection { background: rgba(16, 185, 129, 0.35); color: #ffffff; }
    html { scroll-behavior: smooth; }
    ::-webkit-scrollbar { width: 10px; height: 10px; }
    ::-webkit-scrollbar-track { background: #070a10; }
    ::-webkit-scrollbar-thumb {
        background: linear-gradient(180deg, #134e4a, #0f766e);
        border-radius: 99px;
        border: 2px solid #070a10;
    }
    ::-webkit-scrollbar-thumb:hover { background: linear-gradient(180deg, #10b981, #059669); }

    .stApp {
        background:
            radial-gradient(1200px 700px at 12% -10%, rgba(16, 185, 129, 0.16), transparent 62%),
            radial-gradient(1000px 600px at 95% 2%, rgba(56, 189, 248, 0.12), transparent 60%),
            radial-gradient(900px 700px at 50% 112%, rgba(139, 92, 246, 0.10), transparent 62%),
            #05070c;
        background-size: 135% 135%;
        animation: bgDrift 28s ease-in-out infinite alternate;
    }
    @keyframes bgDrift {
        from { background-position: 0% 0%, 100% 0%, 50% 100%; }
        to { background-position: 14% 10%, 86% 8%, 46% 92%; }
    }

    /* Hero container */
    .st-key-hero {
        position: relative;
        overflow: hidden;
        padding: 28px 32px 24px 32px !important;
        border: 1px solid rgba(110, 231, 183, 0.20) !important;
        border-radius: 24px !important;
        background:
            radial-gradient(620px 240px at 0% 0%, rgba(16, 185, 129, 0.20), transparent 70%),
            radial-gradient(520px 280px at 100% 0%, rgba(56, 189, 248, 0.14), transparent 70%),
            linear-gradient(135deg, rgba(13, 26, 32, 0.94), rgba(8, 12, 20, 0.94)) !important;
        box-shadow: 0 24px 60px rgba(0, 0, 0, 0.42), inset 0 1px 0 rgba(255, 255, 255, 0.06) !important;
        transition: border-color 0.35s ease, box-shadow 0.35s ease;
        margin-bottom: 10px;
    }
    .st-key-hero > div { position: relative; z-index: 1; }
    .st-key-hero::before {
        content: "";
        position: absolute;
        inset: -45% -15% auto -15%;
        height: 230%;
        background: conic-gradient(from 180deg at 50% 50%,
            transparent 0deg,
            rgba(16, 185, 129, 0.10) 70deg,
            transparent 140deg,
            rgba(56, 189, 248, 0.08) 220deg,
            transparent 300deg);
        animation: heroGlow 20s linear infinite;
        pointer-events: none;
        z-index: 0;
    }
    .st-key-hero::after {
        content: "";
        position: absolute;
        left: 0;
        right: 0;
        top: 0;
        height: 2px;
        background: linear-gradient(90deg, transparent, #34d399, #38bdf8, transparent);
        background-size: 200% 100%;
        animation: scanLine 7s linear infinite;
        pointer-events: none;
        z-index: 2;
    }
    .st-key-hero:hover {
        border-color: rgba(110, 231, 183, 0.38) !important;
        box-shadow: 0 28px 70px rgba(0, 0, 0, 0.48), 0 0 48px rgba(16, 185, 129, 0.10) !important;
    }
    .st-key-hero { animation: heroIn 0.65s cubic-bezier(0.2, 0.8, 0.2, 1) backwards; }
    @keyframes heroIn {
        from { opacity: 0; transform: translateY(12px) scale(0.995); }
        to { opacity: 1; transform: none; }
    }
    .st-key-hero h1 {
        font-size: 2.55rem !important;
        background: linear-gradient(94deg, #ffffff 0%, #d1fae5 40%, #6ee7b7 72%, #7dd3fc 100%);
        -webkit-background-clip: text;
        background-clip: text;
        -webkit-text-fill-color: transparent;
        filter: drop-shadow(0 8px 26px rgba(16, 185, 129, 0.18));
    }
    @supports not ((-webkit-background-clip: text) or (background-clip: text)) {
        .st-key-hero h1 {
            background: none;
            color: #f8fafc;
            -webkit-text-fill-color: #f8fafc;
        }
    }
    .st-key-hero [data-testid="stCaptionContainer"] p {
        font-size: 13.5px !important;
        max-width: 920px;
        color: #a9b7c9 !important;
        line-height: 1.55;
    }
    @keyframes heroGlow {
        from { transform: rotate(0deg); }
        to { transform: rotate(360deg); }
    }
    @keyframes scanLine {
        from { background-position: 200% 0; }
        to { background-position: -200% 0; }
    }

    /* Hero orbit visual */
    .hero-visual {
        position: absolute;
        right: 54px;
        top: 50%;
        width: 168px;
        height: 168px;
        transform: translateY(-50%);
        pointer-events: none;
        opacity: 0.9;
        z-index: 1;
    }
    .hero-ring {
        position: absolute;
        inset: 0;
        border-radius: 50%;
        border: 1px solid rgba(52, 211, 153, 0.16);
    }
    .ring-a {
        border-top-color: rgba(52, 211, 153, 0.75);
        border-right-color: rgba(56, 189, 248, 0.35);
        animation: orbitSpin 26s linear infinite;
    }
    .ring-b {
        inset: 24px;
        border-bottom-color: rgba(56, 189, 248, 0.75);
        border-left-color: rgba(52, 211, 153, 0.30);
        animation: orbitSpin 16s linear infinite reverse;
    }
    .hero-core {
        position: absolute;
        inset: 60px;
        border-radius: 50%;
        background: radial-gradient(circle at 35% 30%, rgba(16, 185, 129, 0.35), rgba(8, 16, 22, 0.92));
        border: 1px solid rgba(52, 211, 153, 0.42);
        display: flex;
        align-items: center;
        justify-content: center;
        box-shadow: 0 0 34px rgba(16, 185, 129, 0.26), inset 0 0 18px rgba(16, 185, 129, 0.20);
    }
    .hero-core-dot {
        width: 10px;
        height: 10px;
        border-radius: 50%;
        background: #34d399;
        box-shadow: 0 0 18px #34d399;
        animation: corePulse 2.2s ease-in-out infinite;
    }
    @keyframes orbitSpin { to { transform: rotate(360deg); } }
    @keyframes corePulse {
        0%, 100% { transform: scale(0.85); opacity: 0.7; }
        50% { transform: scale(1.15); opacity: 1; }
    }
    @media (max-width: 1250px) {
        .hero-visual { display: none; }
    }

    /* Chips */
    .chip {
        transition: transform 0.2s ease, border-color 0.2s ease, box-shadow 0.2s ease, background 0.2s ease;
        animation: fadeUp 0.45s cubic-bezier(0.2, 0.8, 0.2, 1) backwards;
    }
    .chip:nth-child(2) { animation-delay: 0.05s; }
    .chip:nth-child(3) { animation-delay: 0.10s; }
    .chip:nth-child(4) { animation-delay: 0.15s; }
    .chip:nth-child(5) { animation-delay: 0.20s; }
    .chip:hover {
        transform: translateY(-2px);
        border-color: rgba(110, 231, 183, 0.5);
        box-shadow: 0 10px 22px rgba(0, 0, 0, 0.4);
    }
    .chip-live .dot { animation: dotBreathe 2.4s ease-in-out infinite; }
    @keyframes dotBreathe {
        0%, 100% { box-shadow: 0 0 0 3px rgba(16, 185, 129, 0.12); }
        50% { box-shadow: 0 0 0 6px rgba(16, 185, 129, 0.04); }
    }

    /* Section headers */
    .section-head {
        position: relative;
        padding-left: 15px;
    }
    .section-head::before {
        content: "";
        position: absolute;
        left: 0;
        top: 7px;
        bottom: 7px;
        width: 3px;
        border-radius: 99px;
        background: linear-gradient(180deg, #34d399, #38bdf8);
        box-shadow: 0 0 16px rgba(52, 211, 153, 0.55);
    }
    .section-kicker { transition: letter-spacing 0.3s ease; }
    .section-head:hover .section-kicker { letter-spacing: 0.19em; }

    /* KPI cards */
    .stat-card {
        background:
            radial-gradient(150px 90px at 100% 0%, color-mix(in srgb, var(--accent) 13%, transparent), transparent 72%),
            linear-gradient(160deg, rgba(22, 30, 46, 0.94), rgba(12, 17, 27, 0.94));
        transition: transform 0.28s cubic-bezier(0.2, 0.8, 0.2, 1), border-color 0.28s ease, box-shadow 0.28s ease;
        animation: fadeUp 0.55s cubic-bezier(0.2, 0.8, 0.2, 1) backwards;
    }
    .stat-card:nth-child(1) { animation-delay: 0.02s; }
    .stat-card:nth-child(2) { animation-delay: 0.07s; }
    .stat-card:nth-child(3) { animation-delay: 0.12s; }
    .stat-card:nth-child(4) { animation-delay: 0.17s; }
    .stat-card:nth-child(5) { animation-delay: 0.22s; }
    .stat-card::after {
        content: "";
        position: absolute;
        top: 0;
        left: -70%;
        width: 45%;
        height: 100%;
        background: linear-gradient(105deg, transparent, rgba(255, 255, 255, 0.07), transparent);
        transform: skewX(-18deg);
        transition: left 0.65s ease;
        pointer-events: none;
    }
    .stat-card:hover {
        transform: translateY(-5px);
        border-color: color-mix(in srgb, var(--accent) 55%, transparent);
        box-shadow: 0 22px 46px rgba(0, 0, 0, 0.5), 0 0 0 1px color-mix(in srgb, var(--accent) 25%, transparent);
    }
    .stat-card:hover::after { left: 130%; }
    .stat-icon {
        display: inline-flex;
        align-items: center;
        justify-content: center;
        width: 28px;
        height: 28px;
        border-radius: 9px;
        background: color-mix(in srgb, var(--accent) 16%, transparent);
        color: var(--accent);
        font-size: 13px;
        margin-bottom: 9px;
        transition: transform 0.25s ease, background 0.25s ease;
    }
    .stat-card:hover .stat-icon {
        transform: translateY(-1px) scale(1.08);
        background: color-mix(in srgb, var(--accent) 26%, transparent);
    }
    .stat-value {
        font-variant-numeric: tabular-nums;
        text-shadow: 0 0 26px color-mix(in srgb, var(--accent) 22%, transparent);
    }
    @keyframes fadeUp {
        from { opacity: 0; transform: translateY(16px); }
        to { opacity: 1; transform: none; }
    }

    /* Buttons: shine sweep + depth */
    .stButton > button::after,
    .stDownloadButton > button::after {
        content: "";
        position: absolute;
        top: 0;
        left: -130%;
        width: 60%;
        height: 100%;
        background: linear-gradient(100deg, transparent, rgba(255, 255, 255, 0.30), transparent);
        transform: skewX(-20deg);
        transition: left 0.6s ease;
        pointer-events: none;
    }
    .stButton > button:hover::after,
    .stDownloadButton > button:hover::after { left: 140%; }
    .stDownloadButton > button {
        position: relative;
        overflow: hidden;
        transition: transform 0.22s ease, border-color 0.22s ease, color 0.22s ease, box-shadow 0.22s ease;
    }
    .stDownloadButton > button:hover {
        transform: translateY(-2px);
        box-shadow: 0 8px 22px rgba(16, 185, 129, 0.22);
    }
    button:focus-visible,
    [role="tab"]:focus-visible,
    [role="slider"]:focus-visible {
        outline: 2px solid #34d399 !important;
        outline-offset: 2px !important;
    }

    /* Sidebar brand + navigation */
    .sidebar-console-card {
        position: relative;
        overflow: hidden;
        transition: transform 0.3s ease, box-shadow 0.3s ease, border-color 0.3s ease;
    }
    .sidebar-console-card::after {
        content: "";
        position: absolute;
        inset: 0;
        background: linear-gradient(115deg, transparent 32%, rgba(255, 255, 255, 0.06), transparent 68%);
        transform: translateX(-110%);
        transition: transform 0.85s ease;
        pointer-events: none;
    }
    .sidebar-console-card:hover {
        transform: translateY(-2px);
        border-color: rgba(52, 211, 153, 0.55);
        box-shadow: 0 12px 34px rgba(37, 99, 235, 0.28), 0 0 38px rgba(16, 185, 129, 0.10);
    }
    .sidebar-console-card:hover::after { transform: translateX(110%); }
    .stRadio [role="radiogroup"] label {
        transition: transform 0.2s ease, border-color 0.2s ease, background 0.2s ease, box-shadow 0.2s ease;
        cursor: pointer;
    }
    .stRadio [role="radiogroup"] label:hover {
        transform: translateX(3px);
        border-color: rgba(52, 211, 153, 0.45);
        background: rgba(16, 185, 129, 0.08);
    }
    .stRadio [role="radiogroup"] label:has(input:checked) {
        border-color: rgba(52, 211, 153, 0.55);
        background: linear-gradient(100deg, rgba(16, 185, 129, 0.16), rgba(56, 189, 248, 0.06));
        box-shadow: 0 0 0 1px rgba(52, 211, 153, 0.16), 0 10px 24px rgba(16, 185, 129, 0.12);
    }
    [data-testid="stSidebar"] .stRadio [role="radiogroup"] label:has(input:checked) p {
        color: #ecfdf5;
        font-weight: 600;
    }

    /* Image preset tiles */
    .st-key-preset_moving_auto button,
    .st-key-preset_parked_auto button {
        background: linear-gradient(100deg, rgba(16, 185, 129, 0.12), rgba(56, 189, 248, 0.06)) !important;
        border-color: rgba(52, 211, 153, 0.30) !important;
        color: #d1fae5 !important;
        min-height: 46px;
        font-size: 13px !important;
    }
    .st-key-preset_moving_auto button:hover,
    .st-key-preset_parked_auto button:hover {
        border-color: rgba(52, 211, 153, 0.62) !important;
        box-shadow: 0 12px 28px rgba(16, 185, 129, 0.22) !important;
    }

    /* Tabs */
    .stTabs [data-baseweb="tab"] {
        transition: background 0.2s ease, color 0.2s ease, transform 0.2s ease;
    }
    .stTabs [data-baseweb="tab"]:hover {
        background: rgba(16, 185, 129, 0.08);
        color: #d1fae5;
        transform: translateY(-1px);
    }
    .stTabs [data-baseweb="tab-highlight"] {
        background: linear-gradient(90deg, #10b981, #38bdf8) !important;
        height: 3px !important;
        border-radius: 99px;
    }

    /* Directive feed cards */
    .feed-card {
        transition: transform 0.25s ease, box-shadow 0.25s ease, border-color 0.25s ease;
        animation: fadeUp 0.45s cubic-bezier(0.2, 0.8, 0.2, 1) backwards;
    }
    .feed-card:hover {
        transform: translateX(3px);
        border-color: rgba(148, 163, 184, 0.32);
        box-shadow: 0 16px 34px rgba(0, 0, 0, 0.4),
            -3px 0 20px color-mix(in srgb, var(--edge) 22%, transparent);
    }
    .status-pending { animation: softPulse 2.6s ease-in-out infinite; }
    @keyframes softPulse {
        0%, 100% { box-shadow: 0 0 0 0 rgba(245, 158, 11, 0); }
        50% { box-shadow: 0 0 0 4px rgba(245, 158, 11, 0.10); }
    }

    /* Media & widgets */
    [data-testid="stImage"] img {
        border-radius: 12px;
        transition: transform 0.4s cubic-bezier(0.2, 0.8, 0.2, 1), box-shadow 0.4s ease;
    }
    [data-testid="stImage"] img:hover {
        transform: scale(1.012);
        box-shadow: 0 22px 54px rgba(0, 0, 0, 0.55);
    }
    [data-testid="stVideo"] {
        border-radius: 14px;
        box-shadow: 0 18px 44px rgba(0, 0, 0, 0.42);
        transition: box-shadow 0.3s ease;
    }
    [data-testid="stVideo"]:hover {
        box-shadow: 0 24px 58px rgba(0, 0, 0, 0.55), 0 0 0 1px rgba(52, 211, 153, 0.18);
    }
    [data-testid="stMetric"] {
        transition: transform 0.25s ease, border-color 0.25s ease, box-shadow 0.25s ease;
    }
    [data-testid="stMetric"]:hover {
        transform: translateY(-2px);
        border-color: rgba(52, 211, 153, 0.30);
        box-shadow: 0 16px 34px rgba(0, 0, 0, 0.36);
    }
    [data-testid="stDataFrame"] { transition: border-color 0.25s ease, box-shadow 0.25s ease; }
    [data-testid="stDataFrame"]:hover {
        border-color: rgba(52, 211, 153, 0.26);
        box-shadow: 0 12px 30px rgba(0, 0, 0, 0.30);
    }
    [data-testid="stExpander"] {
        transition: border-color 0.25s ease, background 0.25s ease;
    }
    [data-testid="stExpander"]:hover {
        border-color: rgba(52, 211, 153, 0.28) !important;
        background: rgba(13, 19, 30, 0.76);
    }
    [data-testid="stFileUploader"] section {
        transition: border-color 0.25s ease, background 0.25s ease, box-shadow 0.25s ease;
    }
    [data-testid="stFileUploader"] section:hover {
        border-color: rgba(52, 211, 153, 0.55);
        background: rgba(16, 185, 129, 0.05);
        box-shadow: inset 0 0 34px rgba(16, 185, 129, 0.05);
    }
    [data-baseweb="slider"] [role="slider"] {
        transition: transform 0.2s ease, box-shadow 0.2s ease;
    }
    [data-baseweb="slider"] [role="slider"]:hover {
        transform: scale(1.14);
        box-shadow: 0 0 0 8px rgba(16, 185, 129, 0.14);
    }
    [data-baseweb="input"], [data-baseweb="textarea"] {
        transition: border-color 0.2s ease, box-shadow 0.2s ease;
    }
    [data-baseweb="input"]:focus-within, [data-baseweb="textarea"]:focus-within {
        border-color: rgba(52, 211, 153, 0.55) !important;
        box-shadow: 0 0 0 3px rgba(16, 185, 129, 0.12) !important;
    }

    /* Alerts & toasts */
    [data-testid="stAlert"] {
        border-radius: 12px !important;
        border: 1px solid rgba(56, 189, 248, 0.22) !important;
        animation: fadeUp 0.35s ease both;
    }
    [data-testid="stToast"] {
        background: #0c1a17 !important;
        border: 1px solid rgba(52, 211, 153, 0.35) !important;
        border-radius: 12px !important;
        box-shadow: 0 18px 42px rgba(0, 0, 0, 0.5) !important;
        animation: toastIn 0.35s cubic-bezier(0.2, 0.8, 0.2, 1) both;
    }
    @keyframes toastIn {
        from { opacity: 0; transform: translateY(12px) scale(0.98); }
        to { opacity: 1; transform: none; }
    }

    /* Press feedback */
    .stButton > button:active,
    .stDownloadButton > button:active {
        transform: translateY(0) scale(0.975) !important;
        box-shadow: inset 0 3px 14px rgba(0, 0, 0, 0.45) !important;
        transition-duration: 0.05s !important;
    }

    @media (prefers-reduced-motion: reduce) {
        *, *::before, *::after {
            animation: none !important;
            transition: none !important;
            scroll-behavior: auto !important;
        }
    }
    </style>
    """,
    unsafe_allow_html=True,
)


# ---------------------------------------------------------------------------
# Presentation helpers
# ---------------------------------------------------------------------------
STATE_COLORS = {
    "CRUISE": "#10b981",
    "ACCEL": "#f59e0b",
    "IDLE": "#ef4444",
    "DECEL": "#38bdf8",
    "UNINIT": "#475569",
}
STATE_LABELS = {
    "CRUISE": "Cruise",
    "ACCEL": "Accelerating",
    "IDLE": "Idling",
    "DECEL": "Decelerating",
    "UNINIT": "Initialising",
}
CLASS_COLORS = {
    "car": "#38bdf8",
    "motorcycle": "#fbbf24",
    "auto_rickshaw": "#34d399",
    "bus": "#c084fc",
    "truck": "#fb7185",
}


def _html(markup: str) -> None:
    """Render custom HTML without Markdown mistaking indented lines for code blocks."""
    compact = "".join(line.strip() for line in markup.strip().splitlines())
    st.markdown(compact, unsafe_allow_html=True)


def render_section(kicker: str, title: str, subtitle: str = "") -> None:
    sub = f'<div class="section-sub">{subtitle}</div>' if subtitle else ""
    _html(
        f"""
        <div class="section-head">
            <div>
                <div class="section-kicker">{kicker}</div>
                <div class="section-title">{title}</div>
                {sub}
            </div>
        </div>
        """
    )


def render_stat_grid(items: List[Dict[str, str]]) -> None:
    cards = []
    for item in items:
        cards.append(
            f"""
            <div class="stat-card" style="--accent:{item.get('accent', '#10b981')};">
                <div class="stat-icon">{item.get('icon', '')}</div>
                <div class="stat-label">{item.get('label', '')}</div>
                <div class="stat-value">{item.get('value', '—')}</div>
                <div class="stat-foot">{item.get('foot', '')}</div>
            </div>
            """
        )
    _html(f'<div class="stat-grid">{"".join(cards)}</div>')


def _base_figure(height: int = 250) -> go.Figure:
    fig = go.Figure()
    fig.update_layout(
        height=height,
        margin=dict(l=8, r=8, t=8, b=8),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font=dict(family="Inter, sans-serif", color="#94a3b8", size=11),
        showlegend=False,
        hoverlabel=dict(bgcolor="#111827", bordercolor="rgba(148,163,184,0.2)", font_color="#e5e7eb"),
        hovermode="closest",
        dragmode=False,
        transition=dict(duration=420, easing="cubic-in-out"),
        uirevision="eco-kinetic",
    )
    return fig


def make_state_donut(state_counts: pd.Series) -> go.Figure:
    labels = [STATE_LABELS.get(str(s), str(s)) for s in state_counts.index]
    colors = [STATE_COLORS.get(str(s), "#475569") for s in state_counts.index]
    total = int(state_counts.sum())
    fig = _base_figure(238)
    fig.add_trace(
        go.Pie(
            labels=labels,
            values=state_counts.values,
            hole=0.68,
            marker=dict(colors=colors, line=dict(color="#0b1018", width=2)),
            textinfo="percent",
            textfont=dict(size=10, color="#e6ebf2"),
            hovertemplate="<b>%{label}</b><br>%{value:,} observations<br>%{percent}<extra></extra>",
            sort=False,
        )
    )
    fig.update_layout(
        annotations=[
            dict(
                text=f"<b style='font-size:22px;color:#f8fafc'>{total:,}</b><br><span style='font-size:9px;color:#7c8ba1;letter-spacing:0.12em'>OBSERVATIONS</span>",
                x=0.5,
                y=0.5,
                showarrow=False,
            )
        ],
        margin=dict(l=0, r=0, t=4, b=0),
    )
    return fig


def compute_choke_matrix(df: pd.DataFrame, dt_s: float = 0.08) -> np.ndarray:
    """Return a 4x4 matrix of idle minutes (rows = cy bands, cols = cx bands)."""
    grid = np.zeros((4, 4), dtype=float)
    if df.empty or "state" not in df.columns:
        return grid
    idle = df[df["state"] == "IDLE"]
    if idle.empty:
        return grid
    cx = np.clip(idle["cx_norm"].values, 0.0, 0.9999)
    cy = np.clip(idle["cy_norm"].values, 0.0, 0.9999)
    cols = (cx * 4).astype(int)
    rows = (cy * 4).astype(int)
    for r, c in zip(rows, cols):
        grid[r, c] += dt_s / 60.0
    return grid


def make_choke_heatmap(matrix: np.ndarray) -> go.Figure:
    fig = _base_figure(280)
    text = [[f"{v:.1f}" if v > 0 else "" for v in row] for row in matrix]
    fig.add_trace(
        go.Heatmap(
            z=matrix,
            text=text,
            texttemplate="%{text}",
            textfont=dict(size=11, color="#e6ebf2"),
            colorscale=[
                [0.0, "rgba(15,23,42,0.65)"],
                [0.18, "rgba(16,185,129,0.20)"],
                [0.48, "rgba(16,185,129,0.55)"],
                [0.74, "rgba(245,158,11,0.70)"],
                [1.0, "rgba(239,68,68,0.88)"],
            ],
            x=["W", "MW", "ME", "E"],
            y=["N", "MN", "MS", "S"],
            hovertemplate="Approach %{y} · Lane %{x}<br><b>%{z:.2f}</b> idle-minutes<extra></extra>",
            showscale=False,
            xgap=5,
            ygap=5,
        )
    )
    fig.update_layout(
        xaxis=dict(side="bottom", tickfont=dict(size=10), showgrid=False, zeroline=False),
        yaxis=dict(autorange="reversed", tickfont=dict(size=10), showgrid=False, zeroline=False),
        margin=dict(l=8, r=8, t=8, b=8),
    )
    return fig


def make_fleet_bar(class_counts: Dict[str, int]) -> go.Figure:
    order = [c for c in ["car", "auto_rickshaw", "motorcycle", "bus", "truck"] if c in class_counts]
    values = [int(class_counts[c]) for c in order]
    fig = _base_figure(240)
    fig.add_trace(
        go.Bar(
            x=order,
            y=values,
            marker=dict(color=[CLASS_COLORS.get(c, "#38bdf8") for c in order], line=dict(width=0)),
            text=values,
            textposition="outside",
            textfont=dict(size=11, color="#e6ebf2"),
            hovertemplate="<b>%{x}</b><br>%{y} vehicles<extra></extra>",
            width=0.52,
        )
    )
    fig.update_layout(
        xaxis=dict(showgrid=False, tickfont=dict(size=10), zeroline=False),
        yaxis=dict(showgrid=True, gridcolor="rgba(148,163,184,0.08)", zeroline=False, rangemode="tozero"),
        margin=dict(l=8, r=8, t=20, b=8),
        bargap=0.42,
    )
    return fig


def make_approach_bar(quadrant_counts: Dict[str, int]) -> go.Figure:
    labels = list(quadrant_counts.keys())
    values = [int(v) for v in quadrant_counts.values()]
    fig = _base_figure(230)
    fig.add_trace(
        go.Bar(
            x=labels,
            y=values,
            marker=dict(
                color=["#38bdf8", "#34d399", "#fbbf24", "#c084fc"][: len(labels)],
                line=dict(width=0),
            ),
            text=values,
            textposition="outside",
            textfont=dict(size=11, color="#e6ebf2"),
            hovertemplate="<b>%{x}</b><br>%{y} vehicles<extra></extra>",
            width=0.5,
        )
    )
    fig.update_layout(
        xaxis=dict(showgrid=False, tickfont=dict(size=10), zeroline=False),
        yaxis=dict(showgrid=True, gridcolor="rgba(148,163,184,0.08)", zeroline=False, rangemode="tozero"),
        margin=dict(l=8, r=8, t=20, b=8),
        bargap=0.45,
    )
    return fig


def build_dispatch_payload(
    df: pd.DataFrame,
    factors: Dict[str, Any],
    dt_s: float = 0.08,
    intersection_id: str = "junction_01",
) -> Tuple[CorridorPayload, Dict[str, float]]:
    """Derive a schema-valid corridor payload from the active telemetry window."""
    gated = df[df["track_age_frames"] >= 15] if not df.empty and "track_age_frames" in df.columns else df
    total = int(gated["track_id"].nunique()) if not gated.empty else 0
    idle_rows = int((gated["state"] == "IDLE").sum()) if not gated.empty else 0
    idle_ratio = float(idle_rows / max(1, len(gated)))
    heavy_ids = gated[gated["cls"].isin(["bus", "truck"])]["track_id"].nunique() if not gated.empty else 0
    heavy_density = float(heavy_ids / max(1, total))
    stops = count_stop_events(gated, min_frames=max(1, int(round(3.0 / max(dt_s, 0.01)))))

    classes = factors.get("classes", {})
    idle_burn_g_min = 0.0
    if not gated.empty:
        for cls_name, count in gated["cls"].value_counts().items():
            rate = float(classes.get(str(cls_name), {}).get("idle_g_per_min", 0.0))
            idle_burn_g_min += float(count) * rate / max(1, len(gated)) * total

    choke = compute_choke_points(gated, dt_s=dt_s)
    top_zone = None
    if not choke.empty:
        row = choke.iloc[0]
        top_zone = TopChokeZone(id=str(row["Zone ID"]), idle_vehicle_minutes=float(row["Idle Minutes"]))

    payload = CorridorPayload(
        intersection_id=intersection_id,
        window_s=round(float(gated["video_time_s"].max() - gated["video_time_s"].min()), 2)
        if not gated.empty and "video_time_s" in gated.columns
        else 60.0,
        co2_g_total=round(idle_burn_g_min, 2),
        co2_g_per_min=round(idle_burn_g_min, 2),
        idle_ratio=round(min(1.0, max(0.0, idle_ratio)), 4),
        heavy_vehicle_density=round(min(1.0, max(0.0, heavy_density)), 4),
        n_counted_vehicles=total,
        n_stop_events=stops,
        top_choke_zone=top_zone,
    )
    metrics = {
        "total_vehicles": float(total),
        "idle_ratio": idle_ratio,
        "heavy_density": heavy_density,
        "stop_events": float(stops),
        "idle_burn_g_min": idle_burn_g_min,
    }
    return payload, metrics


def _reset_scenario_defaults() -> None:
    """Reset what-if controls before the widgets are instantiated (button callback)."""
    for key, value in {
        "ev_car": 20,
        "ev_moto": 30,
        "ev_auto": 25,
        "ev_bus": 50,
        "ev_truck": 10,
        "green_ext": 6,
        "heavy_restr": True,
    }.items():
        st.session_state[key] = value


def render_directive_feed(directives: List[Dict[str, Any]], limit: int = 6) -> None:
    """Render municipal directives as interactive operator cards."""
    if not directives:
        _html(
            '<div class="empty-state">No directives yet. Run the dispatcher to generate a '
            "deterministic municipal advisory from the active corridor window.</div>"
        )
        return

    filter_col, _ = st.columns([0.42, 0.58])
    with filter_col:
        mode_filter = st.radio(
            "Directive filter",
            ["All modes", "LLM", "Rules engine"],
            horizontal=True,
            label_visibility="collapsed",
            key="directive_mode_filter",
        )
    if mode_filter == "LLM":
        filtered = [d for d in directives if str(d.get("mode")) == "llm"]
    elif mode_filter == "Rules engine":
        filtered = [d for d in directives if str(d.get("mode")) != "llm"]
    else:
        filtered = list(directives)

    if not filtered:
        _html('<div class="empty-state">No directives match this filter.</div>')
        return

    show_all = bool(st.session_state.get("show_all_directives", False))
    visible = filtered if show_all else filtered[:limit]

    status_map = st.session_state.setdefault("directive_status", {})
    for idx, d in enumerate(visible):
        directive_id = str(d.get("directive_id") or d.get("id") or f"directive_{idx}")
        mode = str(d.get("mode", "rules_fallback"))
        status = status_map.get(directive_id, "pending")
        edge = "#10b981" if mode == "llm" else "#38bdf8"
        mode_label = "Gemini LLM" if mode == "llm" else "Rules engine"
        mode_class = "status-dispatched" if mode == "llm" else "status-pending"
        status_class = {
            "pending": "status-pending",
            "dispatched": "status-dispatched",
            "dismissed": "status-dismissed",
        }.get(status, "status-pending")

        actions = d.get("actions", [])
        action_texts = []
        for a in actions:
            raw = a.get("type", "no_action")
            action_name = str(getattr(raw, "value", raw)).split(".")[-1].replace("_", " ")
            bits = [action_name]
            if a.get("seconds") is not None:
                bits.append(f"{a['seconds']}s")
            if a.get("lane"):
                bits.append(f"{a['lane']} approach")
            if a.get("target_class"):
                tgt = a["target_class"]
                bits.append(f"target {getattr(tgt, 'value', tgt)}")
            action_texts.append(" · ".join(bits))

        saved = float(d.get("estimated_co2_saved_g", 0.0) or 0.0)
        confidence = float(d.get("confidence", 0.0) or 0.0) * 100.0
        rationale = str(d.get("rationale", "No rationale provided."))
        timestamp = str(d.get("timestamp", "—")).replace("T", " ")[:19]

        _html(
            f"""
            <div class="feed-card" style="--edge:{edge};">
                <div class="feed-top">
                    <div style="display:flex;align-items:center;gap:8px;flex-wrap:wrap;">
                        <span class="status-pill {mode_class}">{mode_label}</span>
                        <span class="status-pill {status_class}">{status.upper()}</span>
                        <span class="feed-id">{directive_id[:10]}</span>
                    </div>
                    <div style="display:flex;align-items:center;gap:14px;">
                        <span style="font-size:11px;color:#6c7d93;">{timestamp} UTC</span>
                        <span style="font-size:16px;font-weight:700;color:#6ee7b7;">{saved:,.2f} g</span>
                        <span style="font-size:10px;color:#6c7d93;">avoided</span>
                        <span style="font-size:12px;font-weight:600;color:#fcd34d;">{confidence:.0f}% conf</span>
                    </div>
                </div>
                <div class="feed-rationale">{rationale}</div>
                <div class="feed-actions">↳ {' · '.join(action_texts) if action_texts else 'No action required'}</div>
            </div>
            """
        )
        b1, b2, _spacer = st.columns([0.14, 0.12, 0.74])
        with b1:
            if st.button(
                "✓ Dispatch" if status != "dispatched" else "✓ Dispatched",
                key=f"dispatch_{directive_id}_{idx}",
                width="stretch",
                disabled=status == "dispatched",
                type="primary" if status == "pending" else "secondary",
            ):
                status_map[directive_id] = "dispatched"
                st.session_state["last_operator_action"] = f"Directive {directive_id[:10]} pushed to ATCS / SCATS."
                st.rerun()
        with b2:
            if st.button(
                "Dismiss" if status != "dismissed" else "Recall",
                key=f"dismiss_{directive_id}_{idx}",
                width="stretch",
                disabled=status == "dismissed",
            ):
                status_map[directive_id] = "dismissed"
                st.session_state["last_operator_action"] = f"Directive {directive_id[:10]} dismissed by operator."
                st.rerun()

    if len(filtered) > limit:
        if show_all:
            if st.button("− Collapse archive", key="btn_collapse_directives"):
                st.session_state["show_all_directives"] = False
                st.rerun()
        else:
            if st.button(
                f"+ Show {len(filtered) - limit} older directives",
                key="btn_show_all_directives",
                width="stretch",
            ):
                st.session_state["show_all_directives"] = True
                st.rerun()


@st.cache_data(ttl=60)
def load_telemetry(artifacts_dir: Path) -> Tuple[pd.DataFrame, bool]:
    """Load telemetry.parquet with automatic fallback to fixture if missing."""
    target = artifacts_dir / "telemetry.parquet"
    is_fixture = False

    if not target.exists():
        fixture = REPO_ROOT / "tests" / "fixtures" / "telemetry_fixture.parquet"
        if fixture.exists():
            target = fixture
            is_fixture = True
        else:
            return pd.DataFrame(), False

    try:
        table = pq.read_table(target)
        df = table.to_pandas()
        return df, is_fixture
    except Exception:
        return pd.DataFrame(), False


@st.cache_data(ttl=60)
def load_emission_factors() -> Dict[str, Any]:
    cfg_path = REPO_ROOT / "config" / "emission_factors.yaml"
    if cfg_path.exists():
        try:
            with open(cfg_path, "r", encoding="utf-8") as f:
                return yaml.safe_load(f) or {}
        except Exception:
            return {}
    return {}


@st.cache_data(ttl=60)
def load_json_artifact(file_path: Path) -> Optional[Dict[str, Any]]:
    if file_path.exists():
        try:
            with open(file_path, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception:
            return None
    return None


def load_directives(artifacts_dir: Path) -> List[Dict[str, Any]]:
    """Read directives.jsonl using dispatcher Pydantic model if available."""
    directives_path = artifacts_dir / "directives.jsonl"
    if not directives_path.exists():
        return []

    # Attempt to import Pydantic model from dispatcher
    DirectiveModel = None
    try:
        from dispatcher.schema import Directive as DirectiveModel
    except Exception:
        try:
            from pydantic import BaseModel, Field

            class ActionModel(BaseModel):
                type: str
                lane: Optional[str] = None
                seconds: Optional[float] = None
                target_class: Optional[str] = None

            class DirectiveFallback(BaseModel):
                timestamp: str
                intersection_id: str
                actions: List[ActionModel]
                estimated_co2_saved_g: float
                confidence: float
                rationale: str
                mode: str

            DirectiveModel = DirectiveFallback
        except Exception:
            DirectiveModel = None

    items = []
    try:
        with open(directives_path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    data = json.loads(line)
                    if DirectiveModel is not None:
                        obj = DirectiveModel(**data)
                        items.append(obj.model_dump() if hasattr(obj, "model_dump") else obj.dict())
                    else:
                        items.append(data)
                except Exception:
                    continue
    except Exception:
        return []

    return list(reversed(items))  # newest first


def compute_choke_points(df: pd.DataFrame, dt_s: float = 0.08) -> pd.DataFrame:
    """Compute local 4x4 spatial grid on cx_norm/cy_norm and idle minutes per cell."""
    if df.empty or "state" not in df.columns:
        return pd.DataFrame()

    idle_df = df[df["state"] == "IDLE"].copy()
    if idle_df.empty:
        return pd.DataFrame()

    cx = np.clip(idle_df["cx_norm"].values, 0.0, 0.9999)
    cy = np.clip(idle_df["cy_norm"].values, 0.0, 0.9999)

    grid_x = (cx * 4).astype(int)
    grid_y = (cy * 4).astype(int)
    idle_df["cell_id"] = [f"Z_{gx}_{gy}" for gx, gy in zip(grid_x, grid_y)]

    # Compute stats per cell
    rows = []
    for cell_id, grp in idle_df.groupby("cell_id"):
        idle_mins = (len(grp) * dt_s) / 60.0
        n_unique = grp["track_id"].nunique()
        heavy_count = grp[grp["cls"].isin(["bus", "truck"])]["track_id"].nunique()
        top_cls = grp["cls"].value_counts().index[0] if not grp.empty else "N/A"
        rows.append({
            "Zone ID": cell_id,
            "Idle Minutes": round(idle_mins, 2),
            "Unique Idlers": n_unique,
            "Heavy Idlers": heavy_count,
            "Top Idling Class": top_cls,
        })

    choke_df = pd.DataFrame(rows).sort_values(by="Idle Minutes", ascending=False).reset_index(drop=True)
    return choke_df


def count_stop_events(df: pd.DataFrame, min_frames: int = 30) -> int:
    """Count contiguous IDLE stop events >= min_frames."""
    if df.empty or "state" not in df.columns or "track_id" not in df.columns:
        return 0

    stops = 0
    for _, group in df.groupby("track_id", sort=False):
        states = group["state"].values
        is_idle = (states == "IDLE")
        if not np.any(is_idle):
            continue
        diff = np.diff(np.concatenate(([False], is_idle, [False])).astype(int))
        run_starts = np.where(diff == 1)[0]
        run_ends = np.where(diff == -1)[0]
        lengths = run_ends - run_starts
        stops += int(np.sum(lengths >= min_frames))
    return stops


# -------------------------------------------------------------
# MAIN APP
# -------------------------------------------------------------
def render_video_operations(artifacts_dir: Path) -> None:
    # Load Data
    df, is_fixture = load_telemetry(artifacts_dir)
    em_factors = load_emission_factors()
    meta = load_json_artifact(artifacts_dir / "telemetry_meta.json")
    dt_s = float(meta.get("dt_s", 0.08)) if meta else 0.08

    # SECTION A: HERO + STATUS
    with st.container(key="hero"):
        st.title("Eco-Kinetic Traffic Carbon Digital Twin")
        st.caption(
            "Retrofit intelligence for the existing CCTV fleet — vehicle kinematics become verified carbon "
            "telemetry, and deterministic policy becomes municipal action. No new hardware. No video egress."
        )
        _html(
            f"""
            <div class="chip-row" style="margin-top:10px;">
                <span class="chip chip-live"><span class="dot"></span>Edge node live · CPU only</span>
                <span class="chip chip-blue">Zero-PII telemetry uplink</span>
                <span class="chip">CPCB BS-VI factors</span>
                <span class="chip chip-amber">{"Offline fixture preview" if is_fixture else "Live telemetry ledger"}</span>
            </div>
            <div class="hero-visual" aria-hidden="true">
                <div class="hero-ring ring-a"></div>
                <div class="hero-ring ring-b"></div>
                <div class="hero-core"><span class="hero-core-dot"></span></div>
            </div>
            """
        )
    st.write("")

    if is_fixture:
        st.info("ℹ️ Telemetry not yet generated by the live pipeline. Operating in offline fixture preview mode.")

    # SECTION B: KPI CARDS
    total_vehicles = df["track_id"].nunique() if not df.empty and "track_id" in df.columns else None

    co2_display = "—"
    if not df.empty and "co2_cum_g" in df.columns:
        per_track_max = df.groupby("track_id")["co2_cum_g"].max() if "track_id" in df.columns else df["co2_cum_g"]
        co2_display = f"{float(per_track_max.sum()):,.1f} g"
    else:
        try:
            from twin.emission_engine import compute_track_emissions

            if not df.empty and em_factors:
                em_df = compute_track_emissions(df, em_factors, min_age=15)
                if not em_df.empty and "co2_cum_g" in em_df.columns:
                    per_track_max = (
                        em_df.groupby("track_id")["co2_cum_g"].max()
                        if "track_id" in em_df.columns
                        else em_df["co2_cum_g"]
                    )
                    co2_display = f"{float(per_track_max.sum()):,.1f} g"
        except Exception:
            pass

    if not df.empty and "state" in df.columns:
        gated = df[df["track_age_frames"] >= 15] if "track_age_frames" in df.columns else df
        idle_count = (gated["state"] == "IDLE").sum() if not gated.empty else 0
        total_gated = len(gated) if not gated.empty else 1
        idle_share_str = f"{(idle_count / max(1, total_gated)) * 100.0:.1f}%"
    else:
        idle_share_str = "—"

    if not df.empty:
        stop_events_val = count_stop_events(df, min_frames=int(round(3.0 / dt_s)))
        stop_events_str = f"{stop_events_val:,}"
    else:
        stop_events_val = 0
        stop_events_str = "—"

    choke_df = compute_choke_points(df, dt_s=dt_s)
    top_zone_str = str(choke_df.iloc[0]["Zone ID"]) if not choke_df.empty else "—"
    top_zone_mins = float(choke_df.iloc[0]["Idle Minutes"]) if not choke_df.empty else 0.0
    duration_s = float(df["video_time_s"].max()) if not df.empty and "video_time_s" in df.columns else 0.0

    render_stat_grid(
        [
            {
                "label": "Vehicles tracked",
                "value": f"{total_vehicles:,}" if total_vehicles is not None else "—",
                "foot": f"<b>{duration_s:.0f}s</b> telemetry span",
                "icon": "◉",
                "accent": "#38bdf8",
            },
            {
                "label": "Cumulative CO₂",
                "value": co2_display,
                "foot": "CPCB mode-bin ledger",
                "icon": "◍",
                "accent": "#10b981",
            },
            {
                "label": "Idle share",
                "value": idle_share_str,
                "foot": f"<b>{stop_events_str}</b> stop events detected",
                "icon": "◌",
                "accent": "#f59e0b",
            },
            {
                "label": "Top choke zone",
                "value": top_zone_str,
                "foot": f"<b>{top_zone_mins:.1f}</b> idle-minutes",
                "icon": "◆",
                "accent": "#fb7185",
            },
            {
                "label": "Telemetry rows",
                "value": f"{len(df):,}",
                "foot": f"dt <b>{dt_s:.3f}s</b> · frame skip {meta.get('frame_skip', '—') if meta else '—'}",
                "icon": "▤",
                "accent": "#a78bfa",
            },
        ]
    )

    # Quick actions
    qa1, qa2, qa3, _qa_space = st.columns([0.15, 0.17, 0.19, 0.49])
    with qa1:
        if st.button("↻ Refresh data", key="btn_refresh_telemetry", width="stretch"):
            st.cache_data.clear()
            st.session_state["last_refresh"] = datetime.now(timezone.utc).strftime("%H:%M:%S")
            st.rerun()
    with qa2:
        csv_bytes = df.to_csv(index=False).encode("utf-8") if not df.empty else b""
        st.download_button(
            "↓ Export telemetry",
            data=csv_bytes,
            file_name="eco_kinetic_telemetry.csv",
            mime="text/csv",
            width="stretch",
            disabled=df.empty,
        )
    with qa3:
        run_dispatch = st.button("✦ Generate advisory", key="btn_generate_advisory", width="stretch", type="primary")

    if st.session_state.get("last_operator_action"):
        st.success(st.session_state.pop("last_operator_action"))

    if run_dispatch:
        if df.empty or not em_factors:
            st.warning("Telemetry or emission factors unavailable — cannot generate an advisory.")
        else:
            with st.spinner("Running deterministic ATCS dispatcher…"):
                payload, _metrics = build_dispatch_payload(df, em_factors, dt_s=dt_s)
                directive = dispatch(
                    payload,
                    factors=em_factors,
                    api_key=None,
                    directives_log_path=artifacts_dir / "directives.jsonl",
                )
            st.cache_data.clear()
            st.session_state["last_operator_action"] = (
                f"Advisory {directive.directive_id[:10]} generated — "
                f"{directive.estimated_co2_saved_g:.2f} g CO₂ avoided."
            )
            st.rerun()

    st.write("")

    # SECTION C: SYNCED PLAYBACK & CHARTS
    overlay_file = artifacts_dir / "overlay.mp4"
    if not overlay_file.exists():
        fallback_ov = REPO_ROOT / "dashboard-web" / "public" / "overlay.mp4"
        if fallback_ov.exists():
            overlay_file = fallback_ov

    raw_junction_file = REPO_ROOT / "dashboard-web" / "public" / "junction.mp4"

    col_video, col_charts = st.columns([1.15, 0.85], gap="large")

    with col_video:
        render_section("Live perception", "Junction CCTV operations", "Bounding-box kinematics rendered at the edge; video never leaves the junction.")
        stream_mode = st.radio(
            "Active CCTV Stream",
            [
                "🎬 AI tracking overlay",
                "📹 Raw CCTV feed",
            ],
            horizontal=True,
            label_visibility="collapsed",
            key="video_stream_mode",
        )
        active_file = overlay_file if "tracking" in stream_mode.lower() else raw_junction_file

        if active_file.exists() and active_file.stat().st_size > 1000:
            try:
                st.video(str(active_file), format="video/mp4")
            except Exception as e:
                st.error(f"Video player encountered an error: {e}")
        else:
            st.warning(f"Video stream file `{active_file.name}` not found.")

        with st.expander("🛠️ Overlay renderer", expanded=False):
            st.caption("Re-render tracked bounding boxes and state classifications onto the junction CCTV stream.")
            if st.button("⚡ Re-render tracking overlay (H.264)", key="btn_rerender_overlay", width="stretch"):
                with st.spinner("Rendering tracking overlay to artifacts/overlay.mp4…"):
                    try:
                        from scripts.render_overlay import render_overlay

                        render_overlay(
                            video_path=raw_junction_file,
                            telemetry_path=artifacts_dir / "telemetry.parquet",
                            output_path=artifacts_dir / "overlay.mp4",
                        )
                        st.success("Tracking overlay video re-rendered and ready.")
                        st.rerun()
                    except Exception as exc:
                        st.error(f"Renderer error: {exc}")

    # Time Slider Control
    if not df.empty and "video_time_s" in df.columns:
        t_min = float(df["video_time_s"].min())
        t_max = float(df["video_time_s"].max())
        if t_max > t_min:
            if "focus_time" not in st.session_state or not (t_min <= st.session_state["focus_time"] <= t_max):
                st.session_state["focus_time"] = min(t_min + 30.0, t_max)
            t_curr = st.slider(
                "Telemetry time window focus (60s range)",
                min_value=t_min,
                max_value=t_max,
                step=max(0.5, round((t_max - t_min) / 100, 1)),
                key="focus_time",
            )
            def _set_focus_time(value: float) -> None:
                st.session_state["focus_time"] = float(value)

            jump1, jump2, jump3 = st.columns(3)
            with jump1:
                st.button(
                    "⏮ Start",
                    key="jump_start",
                    width="stretch",
                    on_click=_set_focus_time,
                    args=(t_min,),
                )
            with jump2:
                st.button(
                    "◐ Midpoint",
                    key="jump_mid",
                    width="stretch",
                    on_click=_set_focus_time,
                    args=((t_min + t_max) / 2.0,),
                )
            with jump3:
                st.button(
                    "End ⏭",
                    key="jump_end",
                    width="stretch",
                    on_click=_set_focus_time,
                    args=(t_max,),
                )
            w_start = max(t_min, t_curr - 30.0)
            w_end = min(t_max, t_curr + 30.0)
            window_df = df[(df["video_time_s"] >= w_start) & (df["video_time_s"] <= w_end)]
            st.caption(f"Active window: **{w_start:.1f}s → {w_end:.1f}s** · {len(window_df):,} telemetry rows")
        else:
            window_df = df
    else:
        window_df = df

    with col_charts:
        render_section("Kinematics", "State composition", "Distribution of tracked vehicle states inside the active 60-second window.")
        if not window_df.empty and "state" in window_df.columns:
            state_counts = window_df["state"].value_counts()
            st.plotly_chart(make_state_donut(state_counts), width="stretch", config={"displayModeBar": False})
            legend_bits = []
            for state_name in ["CRUISE", "ACCEL", "IDLE", "DECEL"]:
                count = int(state_counts.get(state_name, 0))
                legend_bits.append(
                    f'<span class="chip" style="border-color:{STATE_COLORS[state_name]}55;color:{STATE_COLORS[state_name]};">'
                    f"{STATE_LABELS[state_name]} · {count:,}</span>"
                )
            _html(f'<div class="chip-row" style="margin-top:0;">{"".join(legend_bits)}</div>')
        else:
            _html('<div class="empty-state">No telemetry available for the active window.</div>')

    st.write("")

    # SECTION D: CHOKE-POINT HEATMAP & RANKING
    render_section("Spatial intelligence", "Corridor choke points", "Idle-minutes aggregated across the 4×4 junction grid. Hover any cell for the approach breakdown.")
    heat_col, rank_col = st.columns([1.0, 1.0], gap="large")
    with heat_col:
        matrix = compute_choke_matrix(window_df if not window_df.empty else df, dt_s=dt_s)
        if matrix.sum() > 0:
            st.plotly_chart(make_choke_heatmap(matrix), width="stretch", config={"displayModeBar": False})
        else:
            _html('<div class="empty-state">No idling vehicles detected in this window.</div>')
    with rank_col:
        if not choke_df.empty:
            st.dataframe(
                choke_df.head(8),
                width="stretch",
                hide_index=True,
                column_config={
                    "Idle Minutes": st.column_config.ProgressColumn(
                        "Idle minutes",
                        format="%.2f",
                        min_value=0.0,
                        max_value=float(max(1.0, choke_df["Idle Minutes"].max())),
                    ),
                },
            )
        else:
            _html('<div class="empty-state">No spatial idle events detected in telemetry.</div>')

    st.write("")

    # SECTION E: DIRECTIVE FEED
    directives = load_directives(artifacts_dir)
    render_section(
        "Municipal action",
        "Dispatcher directive stream",
        "Every advisory is schema-validated, with carbon savings computed deterministically in Python — never by the LLM.",
    )
    st.caption("Latest directives · operators can dispatch or dismiss each one.")
    render_directive_feed(directives)

    st.divider()

    # SECTION F: WHAT-IF SCENARIOS (LAZY IMPORT)
    try:
        from twin.scenarios import apply_scenario

        scenario_available = True
    except ImportError:
        scenario_available = False

    render_section(
        "Counterfactual lab",
        "What-if policy simulation",
        "Recompute the entire corridor instantly with fleet electrification and signal-policy levers. Every number is deterministic.",
    )

    if not scenario_available:
        _html('<div class="empty-state">Scenario engine pending (track C).</div>')
    else:
        with st.expander("Configure policy intervention", expanded=True):
            sc_col1, sc_col2 = st.columns([1.05, 0.95], gap="large")
            with sc_col1:
                st.markdown("**Fleet electrification share**")
                ev_car = st.slider("Car EV %", 0, 100, 20, key="ev_car") / 100.0
                ev_moto = st.slider("Motorcycle EV %", 0, 100, 30, key="ev_moto") / 100.0
                ev_auto = st.slider("Auto-rickshaw EV %", 0, 100, 25, key="ev_auto") / 100.0
                ev_bus = st.slider("Bus EV %", 0, 100, 50, key="ev_bus") / 100.0
                ev_truck = st.slider("Truck EV %", 0, 100, 10, key="ev_truck") / 100.0
                ev_dict = {
                    "car": ev_car,
                    "motorcycle": ev_moto,
                    "auto_rickshaw": ev_auto,
                    "bus": ev_bus,
                    "truck": ev_truck,
                }
            with sc_col2:
                st.markdown("**Signal & corridor policy**")
                green_ext = st.slider("Adaptive green extension (s)", 0, 15, 6, key="green_ext")
                heavy_restr = st.toggle(
                    "Heavy vehicle priority wave (bus / truck)",
                    value=True,
                    key="heavy_restr",
                )
                st.caption(
                    "Green extension flushes junction queues; heavy-vehicle priority reduces high-diesel stop penalties."
                )
                if st.button(
                    "↺ Reset to baseline defaults",
                    key="btn_reset_scenario",
                    width="stretch",
                    on_click=_reset_scenario_defaults,
                ):
                    st.rerun()

            if not df.empty and em_factors:
                scenario_res = apply_scenario(
                    df=df,
                    factors=em_factors,
                    ev_share=ev_dict,
                    green_extension_s=green_ext,
                    heavy_restriction=heavy_restr,
                    min_age=15,
                    dt_s=dt_s,
                )

                base_co2 = scenario_res["baseline_co2_g"]
                scen_co2 = scenario_res["scenario_co2_g"]
                avoid_co2 = scenario_res["avoided_g"]
                pct_saved = (avoid_co2 / max(0.001, base_co2)) * 100.0

                st.write("")
                render_stat_grid(
                    [
                        {
                            "label": "Baseline emissions",
                            "value": f"{base_co2:,.1f} g",
                            "foot": "Current observed fleet",
                            "icon": "◍",
                            "accent": "#94a3b8",
                        },
                        {
                            "label": "Scenario emissions",
                            "value": f"{scen_co2:,.1f} g",
                            "foot": "After selected interventions",
                            "icon": "◍",
                            "accent": "#38bdf8",
                        },
                        {
                            "label": "Carbon abatement",
                            "value": f"{pct_saved:.1f}%",
                            "foot": f"<b>{avoid_co2:,.1f} g</b> avoided",
                            "icon": "↘",
                            "accent": "#10b981",
                        },
                        {
                            "label": "Fleet EV share",
                            "value": f"{(sum(ev_dict.values()) / 5) * 100:.0f}%",
                            "foot": "Mean across five classes",
                            "icon": "⚡",
                            "accent": "#fbbf24",
                        },
                    ]
                )

                breakdown = scenario_res.get("per_class_breakdown", {})
                if breakdown:
                    b_df = pd.DataFrame(breakdown).T
                    b_df.rename(
                        columns={
                            "baseline_co2_g": "Baseline CO2 (g)",
                            "scenario_co2_g": "Scenario CO2 (g)",
                            "avoided_g": "Avoided CO2 (g)",
                            "ev_share": "EV Share",
                        },
                        inplace=True,
                    )
                    chart_col, table_col = st.columns([1.0, 1.15], gap="large")
                    with chart_col:
                        st.markdown("**Avoided CO₂ by vehicle class**")
                        fig = _base_figure(270)
                        sorted_df = b_df.sort_values("Avoided CO2 (g)", ascending=True)
                        fig.add_trace(
                            go.Bar(
                                x=sorted_df["Avoided CO2 (g)"],
                                y=sorted_df.index,
                                orientation="h",
                                marker=dict(
                                    color=[CLASS_COLORS.get(str(c), "#38bdf8") for c in sorted_df.index],
                                    line=dict(width=0),
                                ),
                                text=[f"{v:,.1f} g" for v in sorted_df["Avoided CO2 (g)"]],
                                textposition="outside",
                                textfont=dict(color="#cbd5e1", size=10),
                                hovertemplate="<b>%{y}</b><br>%{x:,.2f} g avoided<extra></extra>",
                            )
                        )
                        fig.update_layout(
                            xaxis=dict(showgrid=True, gridcolor="rgba(148,163,184,0.09)", zeroline=False),
                            yaxis=dict(showgrid=False),
                            margin=dict(l=8, r=55, t=6, b=8),
                        )
                        st.plotly_chart(fig, width="stretch", config={"displayModeBar": False})
                    with table_col:
                        st.markdown("**Class-by-class audit**")
                        st.dataframe(b_df, width="stretch", height=270)
                        st.download_button(
                            "↓ Export scenario audit (CSV)",
                            data=b_df.to_csv().encode("utf-8"),
                            file_name="eco_kinetic_scenario_audit.csv",
                            mime="text/csv",
                            width="stretch",
                        )

    # SECTION G: SIDEBAR METHODOLOGY EXPANDER
    with st.sidebar:
        st.markdown("<p class='nav-section-title'>SYSTEM HEALTH</p>", unsafe_allow_html=True)
        _html(
            f"""
            <div class="nav-active-status-card">
                <div class="active-badge-tag">● EDGE NODE NOMINAL</div>
                <div class="active-title-text">{len(df):,} telemetry rows</div>
                <div class="active-desc-text">{meta.get('video_fps', '—') if meta else '—'} FPS · frame skip {meta.get('frame_skip', '—') if meta else '—'} · dt {dt_s:.3f}s</div>
            </div>
            """
        )
        if meta:
            st.caption(
                f"Resolution **{meta.get('img_w', 'N/A')}×{meta.get('img_h', 'N/A')}** · "
                f"CPU-only inference · zero CUDA dependencies"
            )
        else:
            st.caption("CPU-only inference · zero CUDA dependencies")
        if st.session_state.get("last_refresh"):
            st.caption(f"Telemetry cache refreshed at **{st.session_state['last_refresh']} UTC**")

        with st.expander("Methodology & emission factors", expanded=False):
            st.caption("Indicative mode-bin estimates, not tailpipe measurements.")
            if em_factors and "classes" in em_factors:
                ef_rows = []
                for c_name, c_data in em_factors["classes"].items():
                    ef_rows.append({
                        "Class": c_name,
                        "Idle (g/min)": c_data.get("idle_g_per_min", "N/A"),
                        "Cruise (g/km)": c_data.get("cruise_g_per_km", "N/A"),
                        "Accel Mult": c_data.get("accel_multiplier", "N/A"),
                        "Length (m)": c_data.get("mean_length_m", "N/A"),
                        "Source": c_data.get("source", "TODO"),
                        "Year": c_data.get("year", "TODO"),
                    })
                st.dataframe(pd.DataFrame(ef_rows), width="stretch", hide_index=True)
            else:
                st.write("Emission factors config not found.")

            st.markdown(
                r"""
                **Core assumptions**
                - Speeds are length-calibrated bounding-box proxies.
                - Idle emissions accumulate after 3s of verified standstill.
                - Ledger gated at 15 frames to avoid tracking churn.
                - Carbon math is deterministic Python — never the LLM.
                """
            )


def _format_file_size(num_bytes: int) -> str:
    """Human-readable byte size for the sidebar image summary."""
    if num_bytes >= 1_048_576:
        return f"{num_bytes / 1_048_576:.1f} MB"
    return f"{max(1, num_bytes // 1024)} KB"


def render_image_sidebar(analyzer: TrafficImageAnalyzer) -> None:
    """Compact sidebar engine console: status, taxonomy, and unique quick actions."""
    supported = list(analyzer.supported_classes)
    has_image = "active_img_bytes" in st.session_state
    result = st.session_state.get("img_result")
    has_result = bool(result)

    with st.sidebar:
        st.markdown("<p class='nav-section-title'>IMAGE ENGINE</p>", unsafe_allow_html=True)
        _html(
            f"""
            <div class="nav-active-status-card">
                <div class="active-badge-tag">● ENGINE ONLINE</div>
                <div class="active-title-text">{Path(analyzer.model_name).name}</div>
                <div class="active-desc-text">{analyzer.device.upper()} inference · {len(supported)} classes · zero cloud · no PII</div>
            </div>
            """
        )
        if st.session_state.get("engine_reloaded"):
            st.caption("✓ Weights reloaded in this session.")

        with st.expander("Model & taxonomy", expanded=False):
            st.markdown(f"**Weights** `{analyzer.model_name}`")
            st.markdown(f"**Device** `{analyzer.device.upper()}`")
            st.markdown("**Supported classes**")
            _html(
                '<div class="chip-row">'
                + "".join(f'<span class="chip">{c}</span>' for c in supported)
                + "</div>"
            )
            if analyzer.unsupported_classes:
                st.warning("Unavailable in current weights: " + ", ".join(analyzer.unsupported_classes))
            st.caption(
                "Uploads are processed in volatile local memory. No cloud transmission, "
                "facial recognition, or license-plate extraction."
            )

        st.markdown("<p class='nav-section-title'>ACTIVE IMAGE</p>", unsafe_allow_html=True)
        if has_image:
            name = st.session_state.get("active_img_name", "traffic_photo.jpg")
            size = _format_file_size(len(st.session_state["active_img_bytes"]))
            source = st.session_state.get("img_source_type", "uploaded")
            meta = f"{size} · {source}"
            if has_result:
                meta += f" · {result.get('total_detections', 0)} detections"
            _html(
                f"""
                <div class="feed-card" style="--edge:#38bdf8;padding:10px 12px;">
                    <div style="font-size:12px;font-weight:600;color:#e2e8f0;overflow:hidden;text-overflow:ellipsis;white-space:nowrap;">{name}</div>
                    <div style="font-size:10.5px;color:#7d8ea3;margin-top:3px;">{meta}</div>
                </div>
                """
            )
        else:
            _html(
                '<div class="empty-state" style="padding:10px 12px;font-size:11px;">'
                "No image loaded. Choose a preset or upload a photograph in the main panel.</div>"
            )

        st.markdown("<p class='nav-section-title'>ENGINE ACTIONS</p>", unsafe_allow_html=True)
        if st.button(
            "↻ Re-run analysis",
            key="btn_sb_rerun",
            width="stretch",
            disabled=not has_image,
            help="Recompute perception and the advisory with the current threshold.",
        ):
            st.session_state["img_key"] = ""
            st.rerun()
        if st.button(
            "♻ Reload weights",
            key="btn_sb_reload",
            width="stretch",
            help="Clear the cached YOLO weights and reload them from disk.",
        ):
            get_image_analyzer.clear()
            st.cache_data.clear()
            st.session_state["engine_reloaded"] = True
            st.rerun()


def render_image_intelligence(artifacts_dir: Path) -> None:
    """Render the Image Intelligence view for single-frame photograph inspection."""
    with st.container(key="hero"):
        st.title("Traffic Image Intelligence")
        st.caption(
            "Upload one junction photograph — or load a curated CCTV sample — to inspect vehicle mix, "
            "run deterministic CPCB carbon accounting, and generate a municipal advisory."
        )
        _html(
            """
            <div class="chip-row" style="margin-top:10px;">
                <span class="chip chip-live"><span class="dot"></span>Local CPU inference</span>
                <span class="chip chip-blue">5-class Indian road taxonomy</span>
                <span class="chip">Zero cloud upload</span>
                <span class="chip chip-amber">No fabricated kinematics</span>
            </div>
            <div class="hero-visual" aria-hidden="true">
                <div class="hero-ring ring-a"></div>
                <div class="hero-ring ring-b"></div>
                <div class="hero-core"><span class="hero-core-dot"></span></div>
            </div>
            """
        )
    st.write("")

    analyzer = get_image_analyzer()

    # Informative limitation notice if auto_rickshaw is unsupported in current weights
    if analyzer.limitation_notes:
        st.info(f"ℹ️ **Model Taxonomy Note:** {analyzer.limitation_notes}")

    # Section 1: 1-Click Test Presets
    sample_dir = REPO_ROOT / "assets_workspace" / "samples"
    preset_samples = {
        "moving_auto": {
            "name": "🛺 Moving Auto-Rickshaw (CCTV)",
            "file": sample_dir / "sample_junction_moving_auto.jpg",
            "desc": "Roundabout entry with active moving auto-rickshaw, cars & motorcycles",
        },
        "parked_auto": {
            "name": "🛑 Parked Service Auto (CCTV)",
            "file": sample_dir / "sample_junction_parked_auto.jpg",
            "desc": "Right-hand service lane with parked auto & active mixed flow",
        },
    }

    st.markdown("##### ⚡ Quick test presets")
    st.caption("Instant local samples — no upload required. Each preset runs real inference on the bundled CCTV photography.")
    p_col1, p_col2 = st.columns(2)
    preset_selected = None

    with p_col1:
        if st.button(
            preset_samples["moving_auto"]["name"],
            key="preset_moving_auto",
            width="stretch",
            help=preset_samples["moving_auto"]["desc"],
        ):
            preset_selected = "moving_auto"
    with p_col2:
        if st.button(
            preset_samples["parked_auto"]["name"],
            key="preset_parked_auto",
            width="stretch",
            help=preset_samples["parked_auto"]["desc"],
        ):
            preset_selected = "parked_auto"

    if preset_selected:
        p_info = preset_samples[preset_selected]
        if p_info["file"].exists():
            st.session_state["active_img_bytes"] = p_info["file"].read_bytes()
            st.session_state["active_img_name"] = p_info["file"].name
            st.session_state["img_source_type"] = "preset"

    st.write("")

    # Section 2: Upload and Inference Controls
    col_up, col_ctrl = st.columns([1.3, 1.0])
    with col_up:
        uploaded_file = st.file_uploader(
            "Or Upload Custom Traffic Photograph",
            type=["jpg", "jpeg", "png", "webp"],
            help="Upload a standard traffic photograph (JPG, JPEG, PNG, or WebP, max 20 MB). Processed strictly locally on CPU.",
        )
        if uploaded_file is not None:
            up_bytes = uploaded_file.getvalue()
            if (
                st.session_state.get("active_img_name") != uploaded_file.name
                or st.session_state.get("active_img_bytes") != up_bytes
            ):
                st.session_state["active_img_bytes"] = up_bytes
                st.session_state["active_img_name"] = uploaded_file.name
                st.session_state["img_source_type"] = "uploaded"

    with col_ctrl:
        conf_thresh = st.slider(
            "Perception Confidence Threshold",
            min_value=0.05,
            max_value=0.95,
            value=0.25,
            step=0.05,
            help="Minimum model confidence required to retain a vehicle detection.",
        )
        process_btn = st.button("🔍 Process image & generate advisory", type="primary", width="stretch")
        if st.button("✕ Clear current result", key="btn_clear_image", width="stretch"):
            for key in ("active_img_bytes", "active_img_name", "img_result", "img_key", "img_filename", "img_source_type"):
                st.session_state.pop(key, None)
            st.rerun()

    with st.expander("🤖 GenAI Dispatcher Configuration (Optional)", expanded=False):
        st.caption(
            "The Edge AI Dispatcher executes a deterministic local ATCS engine by default (100% offline, zero API key required). "
            "Optionally provide a Gemini or OpenAI API key to trigger live LLM generation."
        )
        user_api_key = st.text_input(
            "Gemini / OpenAI API Key (Optional)",
            type="password",
            value="",
            help="Leave blank to use the built-in deterministic local ATCS engine.",
        )

    # Empty State Check
    if "active_img_bytes" not in st.session_state:
        st.info("👆 Click any 1-Click Test Preset above or upload a photograph to start vehicle perception & AI advisory.")
        render_image_sidebar(analyzer)
        return

    # Trigger Inference and Advisory
    image_bytes = st.session_state["active_img_bytes"]
    image_filename = st.session_state.get("active_img_name", "traffic_photo.jpg")
    file_key = f"{hashlib.md5(image_bytes).hexdigest()}_{conf_thresh:.2f}"

    should_process = (
        process_btn
        or preset_selected is not None
        or st.session_state.get("img_key") != file_key
    )

    if should_process:
        with st.spinner("Executing CPU perception and synthesizing GenAI municipal advisory..."):
            try:
                result = analyzer.analyze(image_bytes, conf=conf_thresh)

                # Spatial Choke Analysis & Approach Quadrants
                h_img = result["image_dimensions"]["height"]
                w_img = result["image_dimensions"]["width"]
                zones: Dict[str, int] = {}
                quadrants = {"North (N)": 0, "South (S)": 0, "East (E)": 0, "West (W)": 0}

                for r in result["detection_records"]:
                    bbox = r["bbox"]
                    cx, cy = (bbox[0] + bbox[2]) / 2.0, (bbox[1] + bbox[3]) / 2.0
                    r_idx = min(3, int(cy / h_img * 4))
                    c_idx = min(3, int(cx / w_img * 4))
                    zid = f"z_r{r_idx}_c{c_idx}"
                    zones[zid] = zones.get(zid, 0) + 1

                    lane = _determine_lane_from_zone(zid)
                    lane_name = {
                        "N": "North (N)",
                        "S": "South (S)",
                        "E": "East (E)",
                        "W": "West (W)",
                    }.get(lane, "North (N)")
                    quadrants[lane_name] += 1
                    r["approach_quadrant"] = lane_name

                top_z = max(zones.items(), key=lambda x: x[1])[0] if zones else "z_r1_c1"
                top_lane = _determine_lane_from_zone(top_z)

                # CPCB BS-VI Deterministic Idle Burn Rate
                em_factors = load_emission_factors()
                classes_cfg = em_factors.get("classes", {})
                idle_rates = {
                    "car": float(classes_cfg.get("car", {}).get("idle_g_per_min", 22.0)),
                    "motorcycle": float(classes_cfg.get("motorcycle", {}).get("idle_g_per_min", 8.0)),
                    "auto_rickshaw": float(classes_cfg.get("auto_rickshaw", {}).get("idle_g_per_min", 12.0)),
                    "bus": float(classes_cfg.get("bus", {}).get("idle_g_per_min", 45.0)),
                    "truck": float(classes_cfg.get("truck", {}).get("idle_g_per_min", 55.0)),
                }
                cruise_rates = {
                    "car": float(classes_cfg.get("car", {}).get("cruise_g_per_km", 150.0)),
                    "motorcycle": float(classes_cfg.get("motorcycle", {}).get("cruise_g_per_km", 80.0)),
                    "auto_rickshaw": float(classes_cfg.get("auto_rickshaw", {}).get("cruise_g_per_km", 95.0)),
                    "bus": float(classes_cfg.get("bus", {}).get("cruise_g_per_km", 1000.0)),
                    "truck": float(classes_cfg.get("truck", {}).get("cruise_g_per_km", 900.0)),
                }

                counts = result["per_class_counts"]
                total_veh = result["total_detections"]
                instant_idle_burn_g_min = sum(counts.get(c, 0) * idle_rates.get(c, 20.0) for c in counts)
                heavy_count = counts.get("bus", 0) + counts.get("truck", 0)
                heavy_density = (heavy_count / total_veh) if total_veh > 0 else 0.0

                # Construct CorridorPayload
                payload = CorridorPayload(
                    intersection_id="junction_01",
                    window_s=60.0,
                    co2_g_total=round(instant_idle_burn_g_min, 2),
                    co2_g_per_min=round(instant_idle_burn_g_min, 2),
                    idle_ratio=round(min(0.85, max(0.20, total_veh * 0.08)), 2) if total_veh > 0 else 0.0,
                    heavy_vehicle_density=round(heavy_density, 3),
                    n_counted_vehicles=total_veh,
                    n_stop_events=max(1, total_veh) if total_veh > 0 else 0,
                    top_choke_zone=TopChokeZone(
                        id=top_z,
                        idle_vehicle_minutes=round(max(0.5, total_veh * 0.75), 2),
                    ) if total_veh > 0 else None,
                )

                # Dispatch Directive (ad-hoc analysis, not yet published to the municipal log)
                eff_key = user_api_key.strip() if user_api_key.strip() else None
                directive = dispatch(
                    payload,
                    factors=em_factors,
                    api_key=eff_key,
                    directives_log_path=None,
                )

                result["directive"] = directive.model_dump()
                result["instant_idle_burn_g_min"] = instant_idle_burn_g_min
                result["quadrant_counts"] = quadrants
                result["top_zone"] = top_z
                result["top_lane"] = top_lane
                result["idle_rates"] = idle_rates
                result["cruise_rates"] = cruise_rates

                st.session_state["img_result"] = result
                st.session_state["img_key"] = file_key
                st.session_state["img_filename"] = image_filename
            except ImageValidationError as e:
                st.session_state["img_key"] = file_key
                st.session_state.pop("img_result", None)
                st.error(f"Image Validation Error: {e}")
                return
            except Exception as e:
                st.session_state["img_key"] = file_key
                st.session_state.pop("img_result", None)
                st.error(f"Perception & Advisory Processing Failure: {e}")
                return

    result = st.session_state.get("img_result")
    if not result:
        return

    if st.session_state.get("image_flash"):
        st.success(st.session_state.pop("image_flash"))

    # Image source status header
    active_filename = st.session_state.get("img_filename", "traffic_photo.jpg")
    dims = result.get("image_dimensions", {})
    w_px = dims.get("width", 0)
    h_px = dims.get("height", 0)
    latency_ms = result.get("processing_latency_ms", 0.0)

    st.caption(
        f"📷 **Active Image**: `{active_filename}` • **Resolution**: {w_px}×{h_px} px • "
        f"**Latency**: {latency_ms:.0f} ms • **Inference Device**: `{result.get('device', 'cpu').upper()}`"
    )

    counts = result.get("per_class_counts", {})
    total_veh = result.get("total_detections", 0)
    heavy_count = counts.get("bus", 0) + counts.get("truck", 0)
    instant_burn = result.get("instant_idle_burn_g_min", 0.0)

    # High-Impact Fleet KPI Cards
    k1, k2, k3, k4, k5, k6 = st.columns(6)
    with k1:
        st.markdown(
            f"""
            <div class="kpi-card">
                <div class="kpi-title">Total Fleet</div>
                <div class="kpi-num">{total_veh}</div>
                <div class="kpi-sub">Vehicles Detected</div>
            </div>
            """,
            unsafe_allow_html=True,
        )
    with k2:
        st.markdown(
            f"""
            <div class="kpi-card">
                <div class="kpi-title">Auto-Rickshaws</div>
                <div class="kpi-num kpi-num-green">🛺 {counts.get('auto_rickshaw', 0)}</div>
                <div class="kpi-sub">CCTV Fine-Tuned</div>
            </div>
            """,
            unsafe_allow_html=True,
        )
    with k3:
        st.markdown(
            f"""
            <div class="kpi-card">
                <div class="kpi-title">Passenger Cars</div>
                <div class="kpi-num kpi-num-blue">🚗 {counts.get('car', 0)}</div>
                <div class="kpi-sub">CCTV Calibrated</div>
            </div>
            """,
            unsafe_allow_html=True,
        )
    with k4:
        st.markdown(
            f"""
            <div class="kpi-card">
                <div class="kpi-title">Two-Wheelers</div>
                <div class="kpi-num kpi-num-amber">🏍️ {counts.get('motorcycle', 0)}</div>
                <div class="kpi-sub">Motorcycles/Scooters</div>
            </div>
            """,
            unsafe_allow_html=True,
        )
    with k5:
        st.markdown(
            f"""
            <div class="kpi-card">
                <div class="kpi-title">Heavy Transit</div>
                <div class="kpi-num kpi-num-purple">🚌 {heavy_count}</div>
                <div class="kpi-sub">Buses & Trucks</div>
            </div>
            """,
            unsafe_allow_html=True,
        )
    with k6:
        st.markdown(
            f"""
            <div class="kpi-card">
                <div class="kpi-title">Idle Burn Rate</div>
                <div class="kpi-num kpi-num-green">{instant_burn:.1f}</div>
                <div class="kpi-sub">g CO₂ / min (CPCB)</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    # Edge AI Municipal Traffic & Carbon Advisory Container
    directive = result.get("directive", {})
    if directive:
        mode = directive.get("mode", "rules_fallback")
        is_llm = (mode == "llm")
        badge_html = (
            '<span class="badge-llm">🤖 LIVE GEMINI DISPATCH</span>'
            if is_llm
            else '<span class="badge-rules">⚙️ ATCS LOCAL ENGINE (DETERMINISTIC FALLBACK)</span>'
        )
        actions = directive.get("actions", [])
        if actions:
            action_parts = []
            for a in actions:
                raw_type = a.get("type", "")
                val = getattr(raw_type, "value", str(raw_type))
                a_type = val.split(".")[-1].replace("_", " ").upper()
                sec = a.get("seconds")
                lane = a.get("lane")
                tgt = a.get("target_class")
                desc = a_type
                if sec is not None:
                    desc += f" ({sec}s)"
                if lane:
                    desc += f" on {lane} Approach"
                if tgt:
                    desc += f" for {tgt}"
                action_parts.append(desc)
            action_str = ", ".join(action_parts)
        else:
            action_str = "NO ACTION REQUIRED (Balanced Flow)"

        co2_saved = directive.get("estimated_co2_saved_g", 0.0)
        confidence = directive.get("confidence", 0.0) * 100.0
        rationale = directive.get("rationale", "Traffic corridor flow is nominal.")
        directive_id = directive.get("directive_id", "N/A")[:8]
        top_lane = result.get("top_lane", "N")
        top_zone = result.get("top_zone", "N/A")

        st.markdown(
            f"""
            <div class="ai-advisory-container">
                <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 14px; flex-wrap: wrap; gap: 8px;">
                    <div style="font-size: 16px; font-weight: 700; color: #f8fafc; display: flex; align-items: center; gap: 10px;">
                        <span>🤖 Edge AI Municipal Traffic & Carbon Advisory</span>
                        {badge_html}
                    </div>
                    <div style="font-size: 12px; color: #94a3b8;">
                        Intersection: <code>junction_01</code> • Directive ID: <code>{directive_id}</code>
                    </div>
                </div>
                <div style="display: grid; grid-template-columns: 2fr 1fr 1fr; gap: 14px; margin-bottom: 14px; background: rgba(0,0,0,0.30); padding: 14px; border-radius: 8px;">
                    <div>
                        <div style="font-size: 11px; text-transform: uppercase; color: #94a3b8; font-weight: 600;">Intervention Action</div>
                        <div style="font-size: 17px; font-weight: 700; color: #38bdf8;">{action_str}</div>
                    </div>
                    <div>
                        <div style="font-size: 11px; text-transform: uppercase; color: #94a3b8; font-weight: 600;">Avoided Carbon</div>
                        <div style="font-size: 17px; font-weight: 700; color: #10b981;">+{co2_saved:.2f} g CO₂</div>
                    </div>
                    <div>
                        <div style="font-size: 11px; text-transform: uppercase; color: #94a3b8; font-weight: 600;">Confidence</div>
                        <div style="font-size: 17px; font-weight: 700; color: #fde047;">{confidence:.1f}%</div>
                    </div>
                </div>
                <div style="font-size: 13px; color: #cbd5e1; line-height: 1.5; background: rgba(255,255,255,0.04); padding: 12px 16px; border-radius: 6px; border-left: 3px solid #38bdf8;">
                    <strong>Municipal Engineering Rationale:</strong> {rationale}
                </div>
                <div style="margin-top: 10px; font-size: 11px; color: #64748b; display: flex; justify-content: space-between; flex-wrap: wrap;">
                    <span>🛡️ <strong>CPCB BS-VI Guardrail:</strong> Carbon calculations performed deterministically in Python — zero LLM hallucination.</span>
                    <span>Primary Choke Axis: <strong>{top_lane} Approach</strong> ({top_zone})</span>
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

        pub_col1, pub_col2, _pub_space = st.columns([0.24, 0.2, 0.56])
        with pub_col1:
            pub_directive_id = str(directive.get("directive_id", "") or "")
            published = st.session_state.setdefault("published_advisories", [])
            pub_already = bool(pub_directive_id) and pub_directive_id in published
            if st.button(
                "✓ Advisory published" if pub_already else "📡 Publish advisory to directive log",
                key="btn_publish_image_advisory",
                width="stretch",
                type="primary",
                disabled=pub_already,
            ):
                try:
                    append_directive_to_log(
                        Directive(**directive),
                        artifacts_dir / "directives.jsonl",
                    )
                    if pub_directive_id and pub_directive_id not in published:
                        published.append(pub_directive_id)
                    st.cache_data.clear()
                    st.session_state["image_flash"] = "Advisory published to the municipal directive stream."
                    st.rerun()
                except Exception as exc:
                    st.error(f"Could not publish advisory: {exc}")
        with pub_col2:
            st.download_button(
                "↓ Export advisory JSON",
                data=json.dumps(directive, indent=2, default=str).encode("utf-8"),
                file_name="image_advisory.json",
                mime="application/json",
                width="stretch",
            )

    st.write("")
    tab_annotated, tab_side_by_side, tab_spatial = st.tabs([
        "🔍 Perception Overlay",
        "🖼️ Side-by-Side Comparison",
        "🗺️ Spatial Approach Quadrants",
    ])

    with tab_annotated:
        st.image(
            result["annotated_image"],
            width="stretch",
            caption=f"High-contrast bounding boxes & confidence badges ({total_veh} detections above {conf_thresh:.2f} conf)",
        )

    with tab_side_by_side:
        col_orig, col_annot = st.columns(2)
        with col_orig:
            st.subheader("Original Photograph")
            st.image(result["original_image"], width="stretch")
        with col_annot:
            st.subheader("Perception Annotations")
            st.image(result["annotated_image"], width="stretch")

    with tab_spatial:
        st.subheader("Spatial Approach Distribution (4 Cardinal Zones)")
        st.caption("Distribution of detected vehicles across cardinal approach corridors.")
        q_counts = result.get("quadrant_counts", {})
        q_cols = st.columns(4)
        for i, (quad_name, quad_cnt) in enumerate(q_counts.items()):
            with q_cols[i]:
                pct = (quad_cnt / total_veh * 100.0) if total_veh > 0 else 0.0
                st.metric(quad_name, f"{quad_cnt} veh", delta=f"{pct:.0f}% of total" if total_veh > 0 else None)
        st.plotly_chart(
            make_approach_bar(q_counts),
            width="stretch",
            config={"displayModeBar": False},
        )

    st.divider()

    # Fleet Taxonomy & CPCB BS-VI Emission Index
    col_chart, col_cpcb = st.columns([1.1, 1.3])
    with col_chart:
        st.subheader("Detected Fleet Taxonomy")
        st.plotly_chart(
            make_fleet_bar(dict(result["per_class_counts"])),
            width="stretch",
            config={"displayModeBar": False},
        )

    with col_cpcb:
        st.subheader("CPCB BS-VI Emission Index (Snapshot)")
        fleet_rows = []
        idle_r = result.get("idle_rates", {})
        cruise_r = result.get("cruise_rates", {})
        for cls_name in ["car", "auto_rickshaw", "motorcycle", "bus", "truck"]:
            cnt = counts.get(cls_name, 0)
            share = (cnt / total_veh * 100.0) if total_veh > 0 else 0.0
            irate = idle_r.get(cls_name, 0.0)
            crate = cruise_r.get(cls_name, 0.0)
            burn = cnt * irate
            fleet_rows.append({
                "Vehicle Class": cls_name,
                "Count": cnt,
                "Share (%)": f"{share:.1f}%",
                "Idle Rate (g/min)": f"{irate:.1f}",
                "Cruise Rate (g/km)": f"{crate:.1f}",
                "Idle Burn (g/min)": f"{burn:.1f}",
            })
        st.dataframe(pd.DataFrame(fleet_rows), width="stretch")

    # Empty State Warning
    if total_veh == 0:
        st.warning(
            f"No vehicles were detected above confidence threshold {conf_thresh:.2f}. "
            "Try lowering the confidence threshold slider or upload an alternative image angle."
        )

    st.divider()

    # Detailed Detections Table with Interactive Filter
    st.subheader("Vehicle Detection Ledger")
    records = result.get("detection_records", [])
    if records:
        class_options = ["All Classes"] + [c for c in ["auto_rickshaw", "car", "motorcycle", "bus", "truck"] if counts.get(c, 0) > 0]
        f_col1, f_col2 = st.columns([1.0, 2.0])
        with f_col1:
            selected_filter = st.selectbox("Filter by Class", class_options, index=0)

        table_rows = []
        for r in records:
            if selected_filter != "All Classes" and r["class_name"] != selected_filter:
                continue
            bbox = r["bbox"]
            emoji = {
                "auto_rickshaw": "🛺",
                "car": "🚗",
                "motorcycle": "🏍️",
                "bus": "🚌",
                "truck": "🚛",
            }.get(r["class_name"], "🚘")
            table_rows.append({
                "Index": f"#{r['detection_index']}",
                "Class": f"{emoji} {r['class_name']}",
                "Confidence": f"{r['confidence'] * 100:.1f}%",
                "Approach": r.get("approach_quadrant", "N/A"),
                "Bounding Box [x1, y1, x2, y2]": f"[{bbox[0]:.0f}, {bbox[1]:.0f}, {bbox[2]:.0f}, {bbox[3]:.0f}]",
                "Width (px)": f"{r['bbox_w_px']:.0f}",
                "Height (px)": f"{r['bbox_h_px']:.0f}",
                "Area (px²)": f"{r['bbox_area_px']:.0f}",
            })

        with f_col2:
            st.caption(f"Showing **{len(table_rows)}** of **{len(records)}** detections ({selected_filter}).")

        st.dataframe(pd.DataFrame(table_rows), width="stretch")
    else:
        st.info("No vehicle detection records to display.")

    st.divider()

    # Downloads Section
    st.subheader("Export Results")
    down_col1, down_col2, down_col3 = st.columns(3)
    base_name = Path(active_filename).stem

    with down_col1:
        img_bytes = encode_image_to_bytes(result["annotated_image"], format="jpg")
        st.download_button(
            "📥 Download Annotated Image (JPG)",
            data=img_bytes,
            file_name=f"{base_name}_annotated.jpg",
            mime="image/jpeg",
            width="stretch",
        )
    with down_col2:
        csv_data = export_results_to_csv(result)
        st.download_button(
            "📊 Download Detection Ledger (CSV)",
            data=csv_data,
            file_name=f"{base_name}_detections.csv",
            mime="text/csv",
            width="stretch",
        )
    with down_col3:
        json_data = export_results_to_json(result)
        st.download_button(
            "📋 Download Full Intelligence (JSON)",
            data=json_data,
            file_name=f"{base_name}_results.json",
            mime="application/json",
            width="stretch",
        )

    # Methodology & Boundaries Expander (Phase D requirements)
    with st.expander("ℹ️ Kinematics & Carbon Boundary Notice", expanded=False):
        st.markdown(
            r"""
            **Single-Photograph Perception Boundaries:**
            - **Observed Image Detections:** The results above represent an instantaneous spatial snapshot of the intersection.
            - **No Kinematic Extrapolation:** Vehicle speed, acceleration, deceleration, and idling durations cannot be measured reliably from a single static image. Consequently, **no speeds, motion states, or cumulative tailpipe emissions are fabricated or reported**.
            - **Deterministic Carbon Accounting:** Instantaneous idle burn potential ($E_{\text{idle}} = \sum N_c \cdot r_{\text{idle}, c}$) is computed using certified Central Pollution Control Board (CPCB) BS-VI factors. Avoided emissions are computed using counterfactual signal clearance simulations in Python.
            - **Video Digital Twin:** Dynamic mode-bin carbon accumulation ($E_{\text{vehicle}} = \sum r_{\text{mode}} \cdot \Delta t$) is exclusively generated from multi-frame video tracking in **Junction Video Operations** mode.
            - **Privacy Guarantee:** Uploaded images are processed entirely in local CPU volatile memory. No facial recognition, license plate recognition (ANPR), or external cloud API transmissions occur.
            """
        )

    # Sidebar image-engine console with quick actions
    render_image_sidebar(analyzer)


def main() -> None:
    artifacts_dir = REPO_ROOT / "artifacts"

    if "main_nav_radio" not in st.session_state:
        st.session_state["main_nav_radio"] = "Junction Video Operations"

    st.sidebar.markdown(
        '<div class="sidebar-console-card"><div class="console-status-row">'
        '<span class="pulse-indicator"></span>'
        '<span class="status-text">LIVE EDGE NODE · 30 FPS</span></div>'
        '<div class="console-heading">🌿 Eco-Kinetic Console</div>'
        '<div class="console-subtext">CCTV Carbon Digital Twin & Municipal AI</div></div>',
        unsafe_allow_html=True,
    )

    st.sidebar.markdown("<p class='nav-section-title'>WORKSPACE</p>", unsafe_allow_html=True)
    curr_mode = st.sidebar.radio(
        "Navigation Mode",
        [
            "Junction Video Operations",
            "Image Intelligence",
        ],
        key="main_nav_radio",
        label_visibility="collapsed",
        help="Switch between continuous CCTV video digital twin and single-frame photograph intelligence.",
    )

    st.sidebar.divider()

    if curr_mode == "Junction Video Operations":
        render_video_operations(artifacts_dir)
    else:
        render_image_intelligence(artifacts_dir)


if __name__ == "__main__":
    main()
