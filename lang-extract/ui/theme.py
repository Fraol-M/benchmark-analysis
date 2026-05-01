"""Shared CSS + visual helpers for the Streamlit UI."""

from __future__ import annotations

import streamlit as st


CUSTOM_CSS = """
<style>
  /* ── base ────────────────────────────────────────────────────────────── */
  html, body, [class*="css"] {
    font-family: "Inter", "Segoe UI", system-ui, -apple-system, sans-serif;
  }

  .main .block-container {
    padding-top: 2rem;
    padding-bottom: 3rem;
    max-width: 1280px;
  }

  /* ── hero header ─────────────────────────────────────────────────────── */
  .lx-hero {
    background: linear-gradient(135deg,
      rgba(34, 211, 238, 0.18) 0%,
      rgba(139, 92, 246, 0.18) 50%,
      rgba(236, 72, 153, 0.12) 100%);
    border: 1px solid rgba(148, 163, 184, 0.18);
    border-radius: 18px;
    padding: 28px 32px;
    margin-bottom: 28px;
    position: relative;
    overflow: hidden;
  }
  .lx-hero::after {
    content: "";
    position: absolute;
    inset: 0;
    background: radial-gradient(
      circle at 90% 10%,
      rgba(34, 211, 238, 0.25),
      transparent 40%);
    pointer-events: none;
  }
  .lx-hero h1 {
    margin: 0 0 10px 0;
    font-size: 2rem;
    font-weight: 700;
    letter-spacing: -0.02em;
    background: linear-gradient(90deg, #22d3ee, #a78bfa);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    background-clip: text;
  }
  .lx-hero p {
    margin: 0;
    color: #cbd5e1;
    font-size: 0.95rem;
    max-width: 780px;
    line-height: 1.55;
  }
  .lx-hero .lx-kbd {
    display: inline-block;
    padding: 1px 8px;
    background: rgba(15, 23, 42, 0.6);
    border: 1px solid rgba(148, 163, 184, 0.25);
    border-radius: 6px;
    font-family: "JetBrains Mono", "Fira Code", monospace;
    font-size: 0.8rem;
    color: #22d3ee;
  }

  /* ── card panel ──────────────────────────────────────────────────────── */
  .lx-card {
    background: rgba(17, 26, 51, 0.55);
    border: 1px solid rgba(148, 163, 184, 0.15);
    border-radius: 14px;
    padding: 20px 22px;
    margin-bottom: 18px;
  }

  /* ── section headers ─────────────────────────────────────────────────── */
  .lx-section {
    display: flex;
    align-items: center;
    gap: 10px;
    margin: 6px 0 14px 0;
    color: #e2e8f0;
    font-weight: 600;
    font-size: 0.95rem;
    letter-spacing: 0.01em;
  }
  .lx-section .lx-dot {
    width: 8px;
    height: 8px;
    border-radius: 50%;
    background: linear-gradient(135deg, #22d3ee, #a78bfa);
    box-shadow: 0 0 12px rgba(34, 211, 238, 0.6);
  }

  /* ── buttons ─────────────────────────────────────────────────────────── */
  .stButton > button {
    border-radius: 10px !important;
    font-weight: 500 !important;
    transition: transform 0.08s ease, box-shadow 0.18s ease !important;
    border: 1px solid rgba(148, 163, 184, 0.2) !important;
  }
  .stButton > button:hover {
    transform: translateY(-1px);
    box-shadow: 0 6px 18px rgba(34, 211, 238, 0.15);
  }
  .stButton > button[kind="primary"] {
    background: linear-gradient(135deg, #22d3ee, #7c3aed) !important;
    border: none !important;
    color: white !important;
  }
  .stButton > button[kind="primary"]:hover {
    box-shadow: 0 8px 22px rgba(124, 58, 237, 0.35) !important;
  }
  .stDownloadButton > button {
    border-radius: 10px !important;
  }

  /* ── inputs ──────────────────────────────────────────────────────────── */
  .stTextInput input, .stTextArea textarea, .stSelectbox div[data-baseweb="select"] > div {
    border-radius: 10px !important;
    border: 1px solid rgba(148, 163, 184, 0.2) !important;
    background: rgba(11, 16, 32, 0.6) !important;
  }
  .stTextArea textarea {
    font-family: "JetBrains Mono", "Fira Code", monospace !important;
    font-size: 0.88rem !important;
  }

  /* ── metrics ─────────────────────────────────────────────────────────── */
  [data-testid="stMetric"] {
    background: rgba(17, 26, 51, 0.6);
    border: 1px solid rgba(148, 163, 184, 0.15);
    border-radius: 12px;
    padding: 14px 16px;
    transition: border-color 0.2s ease;
  }
  [data-testid="stMetric"]:hover {
    border-color: rgba(34, 211, 238, 0.4);
  }
  [data-testid="stMetricLabel"] {
    color: #94a3b8 !important;
    font-size: 0.78rem !important;
    font-weight: 500 !important;
    text-transform: uppercase;
    letter-spacing: 0.05em;
  }
  [data-testid="stMetricValue"] {
    color: #22d3ee !important;
    font-size: 1.6rem !important;
    font-weight: 700 !important;
  }

  /* ── tabs ────────────────────────────────────────────────────────────── */
  .stTabs [data-baseweb="tab-list"] {
    gap: 4px;
    border-bottom: 1px solid rgba(148, 163, 184, 0.15);
  }
  .stTabs [data-baseweb="tab"] {
    border-radius: 10px 10px 0 0;
    padding: 10px 18px;
    background: transparent;
    color: #94a3b8;
    font-weight: 500;
  }
  .stTabs [aria-selected="true"] {
    background: rgba(34, 211, 238, 0.08) !important;
    color: #22d3ee !important;
    border-bottom: 2px solid #22d3ee !important;
  }

  /* ── code blocks ─────────────────────────────────────────────────────── */
  [data-testid="stCodeBlock"] {
    border-radius: 10px !important;
    border: 1px solid rgba(148, 163, 184, 0.15) !important;
  }
  [data-testid="stCodeBlock"] pre {
    background: rgba(11, 16, 32, 0.8) !important;
    padding: 14px 16px !important;
  }

  /* ── alerts ──────────────────────────────────────────────────────────── */
  [data-testid="stAlert"] {
    border-radius: 10px;
    border: 1px solid rgba(148, 163, 184, 0.15);
  }

  /* ── sidebar hidden ──────────────────────────────────────────────────── */
  [data-testid="stSidebar"],
  [data-testid="stSidebarCollapsedControl"],
  [data-testid="collapsedControl"] {
    display: none !important;
  }
  [data-testid="stAppViewContainer"] > section:first-child {
    display: none !important;
  }

  /* ── divider ─────────────────────────────────────────────────────────── */
  hr {
    border: none;
    height: 1px;
    background: linear-gradient(
      90deg,
      transparent,
      rgba(148, 163, 184, 0.25) 20%,
      rgba(148, 163, 184, 0.25) 80%,
      transparent);
    margin: 24px 0 !important;
  }

  /* ── scrollbar ───────────────────────────────────────────────────────── */
  ::-webkit-scrollbar { width: 10px; height: 10px; }
  ::-webkit-scrollbar-track { background: #0b1020; }
  ::-webkit-scrollbar-thumb {
    background: rgba(148, 163, 184, 0.25);
    border-radius: 10px;
  }
  ::-webkit-scrollbar-thumb:hover { background: rgba(34, 211, 238, 0.4); }
</style>
"""


def inject_css() -> None:
    """Inject the shared stylesheet. Call once near the top of app.py."""
    st.markdown(CUSTOM_CSS, unsafe_allow_html=True)


def section_header(label: str) -> None:
    """Render a small gradient-dot section heading inside the main area."""
    st.markdown(
        f'<div class="lx-section"><span class="lx-dot"></span>{label}</div>',
        unsafe_allow_html=True,
    )


def hero(title: str, subtitle_html: str) -> None:
    """Render the page hero banner."""
    st.markdown(
        f"""
        <div class="lx-hero">
          <h1>{title}</h1>
          <p>{subtitle_html}</p>
        </div>
        """,
        unsafe_allow_html=True,
    )
