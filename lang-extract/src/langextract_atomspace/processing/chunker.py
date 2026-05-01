import re


_MERGE_CUES = {
    "it", "its", "they", "their", "them", "this", "these", "those",
    "he", "she", "his", "her",
    "therefore", "thus", "hence", "however", "so",
}
_MERGE_PREFIXES = (
    "this result", "this finding", "these findings",
    "this method", "this approach", "this model",
    "this suggests", "this indicates", "these results",
    "as a result", "for this reason",
)


def _should_merge_with_previous(sentence: str) -> bool:
    """Heuristic: should this sentence stay glued to the previous one?

    Triggered by leading pronouns/discourse markers that lose their referent
    when split across chunks. Mirrors PLN-RAG core/chunker.py.
    """
    norm = sentence.strip().lower()
    if not norm:
        return False
    tokens = norm.split()
    if not tokens:
        return False
    if tokens[0] in _MERGE_CUES:
        return True
    prefix2 = " ".join(tokens[:2])
    if prefix2 in _MERGE_PREFIXES:
        return True
    prefix3 = " ".join(tokens[:3])
    return prefix3 in _MERGE_PREFIXES


def _merge_sentence_groups(sentences: list[str]) -> list[str]:
    """Glue coreferent sentences into compound groups before chunking."""
    merged: list[str] = []
    for s in sentences:
        if merged and _should_merge_with_previous(s):
            merged[-1] = f"{merged[-1]} {s}".strip()
        else:
            merged.append(s)
    return merged


def _split_text(text: str, chunk_size: int) -> list[str]:
    """Split text into chunks near ``chunk_size`` respecting paragraph,
    sentence, and coreference boundaries."""
    text = text.strip()
    if not text:
        return []
    if len(text) <= chunk_size:
        return [text]

    paragraphs = [p.strip() for p in re.split(r"\n\s*\n", text) if p.strip()]
    chunks: list[str] = []
    cur: list[str] = []
    cur_len = 0

    def flush() -> None:
        nonlocal cur, cur_len
        if cur:
            chunks.append("\n\n".join(cur))
            cur = []
            cur_len = 0

    for p in paragraphs:
        if len(p) > chunk_size:
            flush()
            sentences = [s.strip() for s in re.split(r"(?<=[.!?])\s+", p) if s.strip()]
            merged = _merge_sentence_groups(sentences)
            sub: list[str] = []
            sub_len = 0
            for s in merged:
                if sub_len + len(s) > chunk_size and sub:
                    chunks.append(" ".join(sub))
                    sub = [s]
                    sub_len = len(s)
                else:
                    sub.append(s)
                    sub_len += len(s) + 1
            if sub:
                chunks.append(" ".join(sub))
            continue

        if cur_len + len(p) > chunk_size and cur:
            flush()
        cur.append(p)
        cur_len += len(p) + 2

    flush()
    return chunks
