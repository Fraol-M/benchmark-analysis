"""
Streamlit UI for the LangExtract -> AtomSpace pipeline.

Run with:
    streamlit run demo/ui/app.py

or:
    python demo/run_ui.py
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

# Make `demo/src` importable regardless of where streamlit is launched from.
_THIS = Path(__file__).resolve()
_SRC = _THIS.parents[1] / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

import streamlit as st

from langextract_atomspace.pipeline import run_extraction_stream

from state import init_state, set_extraction, set_error, has_result, get_result, DEFAULT_TEXT
from results import (
    render_metta_tab,
    render_atomspace_tab,
    render_raw_tab,
    render_query_tab,
)
from theme import inject_css, hero, section_header


# ── page config ───────────────────────────────────────────────────────────────

st.set_page_config(
    page_title="NL → AtomSpace",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="collapsed",
)

inject_css()
init_state()


# ── fixed config (sidebar removed) ────────────────────────────────────────────

MODEL_ID = "gemini-2.5-flash"
API_KEY = os.getenv("LANGEXTRACT_API_KEY") or None
EXTRACTION_PASSES = 1
SKIP_FUZZY = True


def auto_chunk_size(n_chars: int) -> int:
    """Pick a reasonable max_char_buffer based on the input size.

    Gemini 2.5 Flash has a 1M-token context window, so we prefer fewer/larger
    chunks to cut LLM round-trips (each chunk = one full prompt + call).
    """
    if n_chars <= 0:
        return 2000
    if n_chars <= 2000:
        return max(500, n_chars)     # single chunk
    if n_chars <= 10_000:
        return 2500
    if n_chars <= 30_000:
        return 5000
    if n_chars <= 100_000:
        return 8000
    return 10_000


URL_DEFAULT_CHUNK = 6000  # used when input is a URL (content length not known yet)


# ── hero ──────────────────────────────────────────────────────────────────────

hero(
    "Natural Language → MeTTa / AtomSpace",
    "Paste text or point at a public webpage, then explore the resulting "
    "<span class='lx-kbd'>MeTTa atoms</span>, the populated "
    "<span class='lx-kbd'>AtomSpace</span>, and the raw extraction output. "
    f"Powered by <span class='lx-kbd'>{MODEL_ID}</span>.",
)


# ── input area ────────────────────────────────────────────────────────────────

section_header("Input")

col_input, col_actions = st.columns([4, 1], gap="large")

with col_input:
    input_mode = st.radio(
        "Input mode",
        ["Text", "URL"],
        horizontal=True,
        key="input_mode",
        label_visibility="collapsed",
    )

    text = st.session_state["input_text"]
    url = st.session_state["input_url"]
    if input_mode == "Text":
        text = st.text_area(
            "Input text",
            value=st.session_state["input_text"],
            height=220,
            key="input_text_widget",
            placeholder="Paste your text here…",
            label_visibility="collapsed",
        )
        st.session_state["input_text"] = text
        chunk_size = auto_chunk_size(len(text))
        st.caption(
            f"📝 {len(text):,} characters &nbsp;·&nbsp; "
            f"🧩 chunk size **auto → {chunk_size:,}**",
            unsafe_allow_html=True,
        )
    else:
        url = st.text_input(
            "Public webpage URL",
            value=st.session_state["input_url"],
            key="input_url_widget",
            placeholder="https://example.com/article",
            label_visibility="collapsed",
        )
        st.session_state["input_url"] = url
        chunk_size = URL_DEFAULT_CHUNK
        st.caption(
            f"🔗 Public http/https HTML pages &nbsp;·&nbsp; "
            f"🧷 URL safe mode uses **1 worker** &nbsp;·&nbsp; "
            f"📏 trims to about **8,000 chars / 12 paragraphs** &nbsp;·&nbsp; "
            f"🧩 chunk size **auto → {chunk_size:,}**",
            unsafe_allow_html=True,
        )

with col_actions:
    st.markdown("<div style='height: 6px'></div>", unsafe_allow_html=True)
    extract_clicked = st.button(
        "⚡ Extract",
        type="primary",
        use_container_width=True,
        disabled=not (text.strip() if input_mode == "Text" else url.strip()),
    )
    if st.button("📋 Load example", use_container_width=True):
        st.session_state["input_mode"] = "Text"
        st.session_state["input_text"] = DEFAULT_TEXT
        st.rerun()
    if st.button("🗑️ Clear", use_container_width=True):
        st.session_state["input_text"] = ""
        st.session_state["input_url"] = ""
        st.session_state["last_extraction"] = None
        st.rerun()


# ── compact controls row (workers only) ───────────────────────────────────────

with st.container(border=True):
    wcol, icol = st.columns([2, 5], gap="large")
    with wcol:
        max_workers = st.slider(
            "⚙️ Parallel workers",
            min_value=1,
            max_value=20,
            value=st.session_state.get("max_workers", 5),
            key="max_workers",
            help=(
                "Number of chunks processed in parallel. "
                "Free-tier Gemini = 15 req/min, so 3–5 is usually best. "
                "Paid tier: crank it up."
            ),
        )
    with icol:
        st.markdown("<div style='height: 8px'></div>", unsafe_allow_html=True)
        if input_mode == "URL":
            st.caption(
                f"Model **{MODEL_ID}** &nbsp;·&nbsp; "
                f"{EXTRACTION_PASSES} extraction pass &nbsp;·&nbsp; "
                "URL mode runs sequentially with **1 worker** for safer cancellation and shutdown.",
                unsafe_allow_html=True,
            )
        else:
            st.caption(
                f"Model **{MODEL_ID}** &nbsp;·&nbsp; "
                f"{EXTRACTION_PASSES} extraction pass &nbsp;·&nbsp; "
                f"chunk size chosen automatically from input length.",
                unsafe_allow_html=True,
            )


# ── run extraction on click (streaming) ──────────────────────────────────────

PHASE_LABELS = {
    "fetching":   "🌐 Fetching webpage…",
    "chunking":   "✂️  Chunking text…",
    "extracting": "🧠 Extracting with LLM…",
    "populating": "🌌 Populating AtomSpace…",
    "validating": "🔎 Validating atoms…",
}

if extract_clicked:
    effective_workers = 1 if input_mode == "URL" else max_workers
    status = st.status(
        f"Starting extraction — {MODEL_ID}, chunk ≈ {chunk_size}, {effective_workers} worker(s)",
        expanded=True,
    )
    progress_bar = status.progress(0.0, text="Preparing…")
    log_box = status.empty()
    log_lines: list[str] = []

    def push(line: str) -> None:
        log_lines.append(line)
        log_box.markdown("\n".join(f"- {l}" for l in log_lines[-8:]))

    stream_kwargs = dict(
        model_id=MODEL_ID,
        model_url=None,
        api_key=API_KEY,
        extraction_passes=EXTRACTION_PASSES,
        max_char_buffer=chunk_size,
        max_workers=effective_workers,
        skip_fuzzy=SKIP_FUZZY,
    )
    if input_mode == "URL":
        stream_kwargs["url"] = url
    else:
        stream_kwargs["text"] = text

    final_result = None
    had_error = False
    try:
        for event in run_extraction_stream(**stream_kwargs):
            kind = event["kind"]
            if kind == "phase":
                label = PHASE_LABELS.get(event["name"], event["name"])
                status.update(label=label, state="running")
                push(label)
            elif kind == "fetched":
                message = (
                    f"📄 Fetched **{event['title'] or event['url']}** — "
                    f"{event['char_count']:,} chars, {event['paragraph_count']} paragraphs"
                )
                if event.get("truncated"):
                    message += (
                        f" (trimmed from {event.get('original_char_count', event['char_count']):,} chars / "
                        f"{event.get('original_paragraph_count', event['paragraph_count'])} paragraphs)"
                    )
                push(message)
            elif kind == "chunked":
                push(f"✂️  Split into **{event['count']} chunk(s)**")
            elif kind == "chunk_done":
                frac = event["done"] / max(event["total"], 1)
                elapsed = event.get("elapsed_s", 0.0)
                in_flight = event.get("in_flight", 0)
                progress_bar.progress(
                    frac,
                    text=(
                        f"Chunk {event['done']}/{event['total']} &nbsp;·&nbsp; "
                        f"{event['extractions_so_far']} extractions &nbsp;·&nbsp; "
                        f"{elapsed:.1f}s elapsed &nbsp;·&nbsp; "
                        f"{in_flight} still in-flight"
                    ),
                )
                if event["done"] == event["total"]:
                    push(
                        f"✅ All {event['total']} chunk(s) processed in "
                        f"{elapsed:.1f}s — {event['extractions_so_far']} extractions"
                    )
            elif kind == "error":
                had_error = True
                status.update(label=f"❌ {event['message']}", state="error", expanded=True)
                set_error(event["message"])
                break
            elif kind == "done":
                final_result = event["result"]
    except Exception as exc:
        had_error = True
        status.update(label=f"❌ Extraction failed: {exc}", state="error", expanded=True)
        set_error(str(exc))

    if final_result is not None and not had_error:
        set_extraction(final_result)
        atom_count = len(final_result["atoms"])
        progress_bar.progress(1.0, text=f"{atom_count} atoms loaded")
        status.update(
            label=f"🎉 Done — {atom_count} atoms loaded into &self",
            state="complete",
            expanded=False,
        )
        st.toast(f"✅ {atom_count} atoms loaded", icon="🎉")


# ── show error or results ─────────────────────────────────────────────────────

if st.session_state.get("last_error"):
    st.error(st.session_state["last_error"])

if not has_result():
    st.info(
        "Click **⚡ Extract** to run the pipeline. Results will appear in tabs below.",
        icon="ℹ️",
    )
    st.stop()

result = get_result()

source = result.get("source") or {}
if source.get("kind") == "url":
    title = source.get("title") or "Untitled page"
    paragraph_count = source.get("paragraph_count", 0)
    st.caption(
        f"🌐 Fetched from **{source.get('url', '')}** &middot; _{title}_ &middot; "
        f"{paragraph_count} paragraph(s) extracted"
    )
    with st.expander("📄 Extracted webpage text", expanded=False):
        st.text_area(
            "Preview",
            value=source.get("text", ""),
            height=220,
            disabled=True,
            key="source_preview",
            label_visibility="collapsed",
        )

st.divider()
section_header("Results")

tab_metta, tab_atoms, tab_raw, tab_query = st.tabs(
    ["🧬 MeTTa code", "🌌 AtomSpace", "📦 Raw JSON", "🔍 Query REPL"]
)

with tab_metta:
    render_metta_tab(result)

with tab_atoms:
    render_atomspace_tab(result)

with tab_raw:
    render_raw_tab(result)

with tab_query:
    render_query_tab(result)
