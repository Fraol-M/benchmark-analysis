import json
import httpx
import streamlit as st

st.set_page_config(page_title="PLN-RAG Debug UI", layout="wide")

st.title("PLN-RAG Debug UI")

api_base = st.text_input("API base URL", value="http://localhost:8001")

def _split_texts(raw_text: str) -> list[str]:
    parts = [part.strip() for part in raw_text.split("\n\n") if part.strip()]
    return parts

st.header("Ingest (Debug)")
raw_texts = st.text_area(
    "Texts (separate with a blank line)",
    height=180,
    placeholder="Paste one or more paragraphs here...",
)

if st.button("Run debug ingest"):
    texts = _split_texts(raw_texts)
    if not texts:
        st.warning("Please provide at least one text block.")
    else:
        try:
            resp = httpx.post(
                f"{api_base}/debug/ingest",
                json={"texts": texts},
                timeout=60,
            )
            resp.raise_for_status()
            payload = resp.json()
            st.success(f"Processed {payload.get('processed_count', 0)} texts")

            for item in payload.get("results", []):
                label = f"Text: {item.get('text', '')[:80]}"
                with st.expander(label, expanded=False):
                    if item.get("status") != "success":
                        st.error(item.get("error", "Unknown error"))
                        continue
                    for idx, chunk in enumerate(item.get("chunks", []), start=1):
                        st.subheader(f"Chunk {idx}")
                        st.code(chunk.get("chunk", ""))
                        tabs = st.tabs([
                            "LangExtract",
                            "PLN",
                            "PeTTaChainer",
                            "Context",
                        ])
                        with tabs[0]:
                            st.markdown("**Post-processed**")
                            st.json(chunk.get("langextract_postprocessed", {}))
                        with tabs[1]:
                            st.markdown("**Canonicalized statements**")
                            st.json(chunk.get("pln_canonicalized", []))
                        with tabs[2]:
                            st.markdown("**Atoms added**")
                            st.json(chunk.get("atomspace_added", []))
                        with tabs[3]:
                            st.markdown("**Context**")
                            st.json(chunk.get("context", []))
        except Exception as exc:
            st.error(f"Request failed: {exc}")

st.header("Query (Debug)")
question = st.text_input("Question")

if st.button("Run debug query"):
    if not question.strip():
        st.warning("Please enter a question.")
    else:
        try:
            resp = httpx.post(
                f"{api_base}/debug/query",
                json={"question": question},
                timeout=60,
            )
            resp.raise_for_status()
            payload = resp.json()

            st.subheader("Answer")
            st.write(payload.get("answer", ""))

            st.subheader("Proof")
            st.code(payload.get("proof", ""))

            st.subheader("Parser output")
            tabs = st.tabs([
                "LangExtract",
                "PLN",
                "PeTTaChainer",
                "Context",
            ])
            with tabs[0]:
                st.markdown("**Post-processed**")
                st.json(payload.get("langextract_postprocessed", {}))
            with tabs[1]:
                st.markdown("**Canonicalized queries**")
                st.json(payload.get("pln_canonicalized_queries", []))
                st.markdown("**Supporting statements**")
                st.json(payload.get("supporting_statements", []))
            with tabs[2]:
                st.markdown("**Execution**")
                st.json(
                    {
                        "executed_query": payload.get("executed_query", ""),
                        "fallback_used": payload.get("fallback_used", False),
                        "query_status": payload.get("query_status", ""),
                        "sources": payload.get("sources", []),
                    }
                )
            with tabs[3]:
                st.markdown("**Context**")
                st.json(payload.get("context", []))
        except Exception as exc:
            st.error(f"Request failed: {exc}")
