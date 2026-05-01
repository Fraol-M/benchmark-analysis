"""Results tab renderers."""

from __future__ import annotations

import json
import re
from collections import Counter

import streamlit as st


def _pill(label: str, value: str, color: str = "#22d3ee") -> str:
    return (
        f'<span style="'
        f'display:inline-flex;align-items:center;gap:6px;'
        f'padding:4px 10px;border-radius:999px;'
        f'background:rgba(148,163,184,0.10);'
        f'border:1px solid {color}44;'
        f'font-size:0.78rem;color:#e2e8f0;margin-right:6px;'
        f'">'
        f'<span style="color:{color};font-weight:600;">{label}</span>'
        f'<span style="color:#cbd5e1;">{value}</span>'
        f'</span>'
    )


def _render_rejected(rejected: list) -> None:
    if not rejected:
        return
    with st.expander(f"🛑 Filtered extractions ({len(rejected)})", expanded=False):
        st.caption(
            "Dropped before reaching the AtomSpace — fuzzy alignment, free "
            "variables outside rules, or unparseable rule bodies."
        )
        for item in rejected:
            cls = item.get("extraction_class", "?")
            reason = item.get("reason", "?")
            text = item.get("extraction_text", "")
            st.markdown(f"**[{cls}]** _{reason}_")
            if text:
                st.code(text, language="text")


def render_metta_tab(result: dict) -> None:
    metta_str = result["metta_str"]
    atoms = result["atoms"]
    validation = result.get("validation") or {}
    rejected = result.get("rejected") or []
    atom_to_source = result.get("atom_to_source") or {}

    col_a, col_b = st.columns([3, 1], gap="medium")
    with col_a:
        st.markdown(
            _pill("Atoms", str(len(atoms)))
            + _pill("Namespace", "&self", color="#a78bfa")
            + _pill("Filtered", str(len(rejected)), color="#f87171"),
            unsafe_allow_html=True,
        )
    col_b.download_button(
        "⬇ Download .metta",
        data=metta_str,
        file_name="extracted.metta",
        mime="text/plain",
        use_container_width=True,
    )

    if not metta_str:
        st.info("No atoms extracted. Check your input text or extraction passes.")
        _render_rejected(rejected)
        return

    st.markdown("<div style='height: 8px'></div>", unsafe_allow_html=True)
    status_cols = st.columns(3, gap="medium")
    status_cols[0].metric("Atoms", validation.get("atom_count", len(atoms)))
    status_cols[1].metric("Parsed", validation.get("parsed_count", len(atoms)))
    status_cols[2].metric("Loaded", validation.get("loaded_count", len(atoms)))

    if validation.get("valid", True):
        st.success("✅ Validation passed — generated MeTTa parsed and loaded successfully.")
    else:
        st.error("❌ Validation failed — some generated MeTTa atoms could not be parsed or loaded.")
        for err in validation.get("errors", []):
            stage = err.get("stage", "unknown")
            atom_index = err.get("atom_index", "?")
            with st.expander(f"{stage.title()} error at atom {atom_index}"):
                st.code(err.get("atom", ""), language="lisp")
                st.caption(err.get("message", "Unknown validation error"))

    _render_rejected(rejected)

    st.markdown("##### Generated MeTTa")
    st.code(metta_str, language="lisp")

    if atom_to_source:
        with st.expander("🔗 Atom → source sentence", expanded=False):
            st.caption("Click any atom to jump to the sentence that produced it.")
            for atom_str in [str(a) for a in atoms]:
                src = atom_to_source.get(atom_str)
                if not src:
                    continue
                interval = src.get("char_interval") or {}
                pos = ""
                if interval:
                    pos = f" · chars {interval.get('start_pos','?')}–{interval.get('end_pos','?')}"
                st.code(atom_str, language="lisp")
                st.caption(f"📎 “{src.get('text','')}”{pos}")


def render_atomspace_tab(result: dict) -> None:
    atoms = result["atoms"]
    if not atoms:
        st.info("Empty AtomSpace.")
        return

    atom_strs = [str(a) for a in atoms]
    head_counts = Counter()
    for atom_str in atom_strs:
        match = re.match(r"^\((\S+)", atom_str)
        head_counts[match.group(1) if match else "(other)"] += 1

    st.markdown("##### Atom distribution by head")
    sorted_heads = sorted(head_counts.items(), key=lambda item: -item[1])
    cols = st.columns(min(len(sorted_heads), 6) or 1, gap="medium")
    for i, (head, count) in enumerate(sorted_heads):
        cols[i % len(cols)].metric(head, count)

    st.markdown("<div style='height: 10px'></div>", unsafe_allow_html=True)
    search = st.text_input(
        "Filter atoms",
        placeholder="🔎 type a substring to filter…",
        label_visibility="collapsed",
    )
    filtered = [a for a in atom_strs if not search or search.lower() in a.lower()]

    st.caption(f"Showing **{len(filtered)}** of {len(atom_strs)} atoms.")

    atom_to_source = result.get("atom_to_source") or {}
    with st.container(height=500, border=True):
        for atom_str in filtered:
            st.code(atom_str, language="lisp")
            src = atom_to_source.get(atom_str)
            if src and src.get("text"):
                st.caption(f"📎 “{src['text']}”")


def render_raw_tab(result: dict) -> None:
    extractions = result["extractions"]
    source = result.get("source") or {}

    if source:
        with st.expander("🌐 Source metadata", expanded=False):
            st.json(source)

    if not extractions:
        st.info("No extractions.")
        return

    payload = []
    for extraction in extractions:
        payload.append(
            {
                "extraction_class": extraction.extraction_class,
                "extraction_text": extraction.extraction_text,
                "attributes": extraction.attributes,
                "alignment_status": str(extraction.alignment_status) if extraction.alignment_status else None,
                "char_interval": (
                    {
                        "start_pos": extraction.char_interval.start_pos,
                        "end_pos": extraction.char_interval.end_pos,
                    }
                    if extraction.char_interval
                    else None
                ),
            }
        )

    col_a, col_b = st.columns([3, 1], gap="medium")
    col_a.markdown(
        _pill("Extractions", str(len(payload)), color="#a78bfa"),
        unsafe_allow_html=True,
    )
    col_b.download_button(
        "⬇ Download .jsonl",
        data="\n".join(json.dumps(item) for item in payload),
        file_name="extractions.jsonl",
        mime="application/jsonlines",
        use_container_width=True,
    )
    st.json(payload)


PREBUILT_QUERIES = {
    "— pick a preset —": "",
    "Who is a frog?": "! (match &self (isa $who frog) $who)",
    "What inheritance links exist?": "! (match &self (Inheritance $c $p) ($c $p))",
    "What types are declared?": "! (match &self (: $x $t) ($x : $t))",
    "Get all stored properties": "! (match &self (has $e $p $v) ($e $p $v))",
    "Negated facts": "! (match &self (not $x) $x)",
    "All rule heads": "! (match &self (= $head $body) $head)",
}


def render_query_tab(result: dict) -> None:
    metta = result.get("metta")
    if metta is None:
        st.info("MeTTa runner not ready.")
        return

    col_preset, _ = st.columns([2, 3], gap="medium")
    with col_preset:
        preset = st.selectbox("Preset query", list(PREBUILT_QUERIES.keys()))
    default = PREBUILT_QUERIES[preset] or st.session_state.get("query_input", "")

    query = st.text_area(
        "MeTTa query (prefix with `!` to evaluate)",
        value=default,
        height=100,
        key="query_text",
    )

    run_col, _ = st.columns([1, 5])
    with run_col:
        run = st.button("▶ Run query", type="primary", use_container_width=True)

    if run:
        if not query.strip():
            st.warning("Query is empty.")
            return
        try:
            results = metta.run(query)
            flat = [str(item) for block in results for item in block]
            if flat:
                st.success(f"✅ {len(flat)} result(s)")
                with st.container(border=True):
                    for item in flat:
                        st.code(item, language="lisp")
            else:
                st.info("No results.")
        except Exception as exc:
            st.error(f"Query failed: {exc}")
