"""Session-state helpers so Streamlit doesn't re-run LangExtract on every rerun."""

from __future__ import annotations

import streamlit as st


DEFAULT_TEXT = """\
Sam is a frog. Tom is a cat. Bob is a dog.
Sam croaks. Sam eats flies.
Tom is white.
A frog is a kind of animal. A cat is a kind of animal.
Sam is of type Amphibian. Tom is of type Feline.
Every frog that croaks and eats flies is green.
Tom is not a frog.
"""


def init_state() -> None:
    st.session_state.setdefault("input_text", DEFAULT_TEXT)
    st.session_state.setdefault("input_url", "")
    st.session_state.setdefault("input_mode", "Text")
    st.session_state.setdefault("last_extraction", None)
    st.session_state.setdefault("last_error", None)
    st.session_state.setdefault("query_input", "! (match &self (isa $who frog) $who)")
    st.session_state.setdefault("pln_query_input", "")
    st.session_state.setdefault("pln_load_result", None)


def set_extraction(result_dict: dict) -> None:
    st.session_state["last_extraction"] = result_dict
    st.session_state["last_error"] = None


def set_error(err: str) -> None:
    st.session_state["last_error"] = err
    st.session_state["last_extraction"] = None


def has_result() -> bool:
    return st.session_state.get("last_extraction") is not None


def get_result() -> dict | None:
    return st.session_state.get("last_extraction")
