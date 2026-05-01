from __future__ import annotations

import logging
import re
from typing import Any, Iterable, Optional

from hyperon import E, GroundingSpaceRef, MeTTa, S, V
from langextract.data import AlignmentStatus, Extraction

logger = logging.getLogger(__name__)

_parser = MeTTa()
_COMMA = _parser.parse_single(",")
_TRUE = _parser.parse_single("True")

_EXACT_STATUSES = {
    AlignmentStatus.MATCH_EXACT,
    AlignmentStatus.MATCH_GREATER,
    AlignmentStatus.MATCH_LESSER,
}




def build_canonicalization_context(source_text: str) -> dict[str, Any]:
    """Scan source text once and build a proper-name map.

    A token is treated as a proper name when every occurrence outside a
    sentence-initial position is capitalized. Sentence-initial casing alone
    is ambiguous, so we ignore it when deciding.
    """
    ctx: dict[str, Any] = {"proper": {}}
    if not source_text:
        return ctx

    sentence_starts: set[int] = set()
    for match in re.finditer(r"(?:^|[.!?]\s+)([A-Z][A-Za-z0-9_'-]*)", source_text):
        sentence_starts.add(match.start(1))

    occurrences: dict[str, list[tuple[str, bool]]] = {}
    for match in re.finditer(r"\b([A-Za-z][A-Za-z0-9_'-]*)\b", source_text):
        token = match.group(1)
        is_initial = match.start(1) in sentence_starts
        occurrences.setdefault(token.lower(), []).append((token, is_initial))

    proper: dict[str, str] = {}
    for lower, occs in occurrences.items():
        non_initial = [tok for tok, initial in occs if not initial]
        if non_initial and all(t[0].isupper() for t in non_initial):
            proper[lower] = non_initial[0]
    ctx["proper"] = proper
    return ctx


_SINGULAR_INVARIANT_SUFFIXES = ("ics", "ous", "ness", "ship", "ment")
_SINGULAR_INVARIANT_WORDS = {
    "series", "species", "rabies", "news", "physics", "mathematics",
    "economics", "electronics", "ethics", "politics",
}


def _singularize(word: str) -> str:
    if len(word) <= 3:
        return word
    if word in _SINGULAR_INVARIANT_WORDS:
        return word
    if word.endswith("ies") and len(word) > 4:
        return word[:-3] + "y"
    if word.endswith("ses") and len(word) > 4:
        return word[:-2]
    if word.endswith(_SINGULAR_INVARIANT_SUFFIXES):
        return word
    if word.endswith("s") and not word.endswith(("ss", "us", "is")):
        return word[:-1]
    return word


def _canonicalize_token(token: str, ctx: Optional[dict]) -> str:
    """Canonicalize one whitespace-separated token.

    - Variables (``$x``) preserved.
    - Hyphenated or numeric identifiers (``Object-77``, ``rack-01``) preserved
      verbatim — never split or lemmatized.
    - Proper names (per ``ctx``) preserve their source-text casing.
    - Common nouns: lowercase + singularize so plural/singular collapse.
    """
    if not token:
        return token
    if token.startswith("$"):
        return token
    if "-" in token or any(ch.isdigit() for ch in token):
        return token
    lower = token.lower()
    if ctx and lower in ctx.get("proper", {}):
        return ctx["proper"][lower]
    return _singularize(lower)


# ── text → atom helpers ──────────────────────────────────────────────────────


def _normalize_text(
    value: str,
    *,
    lowercase: bool = False,
    ctx: Optional[dict] = None,
) -> str:
    """Normalize unquoted symbol-like text into valid MeTTa token text.

    ``lowercase=True`` is the predicate path: lowercase + hyphen-join, no
    canonicalization (predicate names are already authored in canonical form
    via the few-shot examples).

    Otherwise (entity / class / value path), apply canonicalization tokenwise.
    """
    text = value.strip()
    if not text:
        raise ValueError("Atom text cannot be empty")
    if text.startswith(("(", '"', "$")):
        return text
    if lowercase:
        return text.lower().replace(" ", "-")
    parts = text.split()
    canon = [_canonicalize_token(p, ctx) for p in parts]
    return "-".join(part for part in canon if part)


def _atom_from_text(
    value: str,
    *,
    lowercase: bool = False,
    ctx: Optional[dict] = None,
):
    """Parse a single symbol / variable / grounded literal / expression atom."""
    text = _normalize_text(value, lowercase=lowercase, ctx=ctx)
    if text.startswith("$"):
        return V(text[1:])
    return _parser.parse_single(text)


def _coerce_argument_list(value, *, split_strings: bool = False) -> list[str]:
    """Return a clean list of argument strings from LangExtract attributes."""
    if value is None:
        return []
    if isinstance(value, str):
        if split_strings:
            return [item for item in value.strip().split() if item]
        return [value] if value.strip() else []
    if isinstance(value, Iterable):
        result = []
        for item in value:
            item_text = str(item).strip()
            if item_text:
                result.append(item_text)
        return result
    item_text = str(value).strip()
    return [item_text] if item_text else []


def _coerce_head_argument_atoms(value) -> list[object]:
    """Return parsed head-argument atoms, preserving quoted strings and expressions."""
    if value is None:
        return []

    if isinstance(value, str):
        text = value.strip()
        if not text:
            return []
        parsed = _parser.parse_single(f"({text})")
        if hasattr(parsed, "get_children"):
            return list(parsed.get_children())
        return [_atom_from_text(text)]

    if isinstance(value, Iterable):
        result = []
        for item in value:
            item_text = str(item).strip()
            if item_text:
                result.append(_atom_from_text(item_text))
        return result

    item_text = str(value).strip()
    return [_atom_from_text(item_text)] if item_text else []


def _fact_arguments(attrs: dict) -> list[str]:
    """Support both generic argument lists and subject/object convenience keys."""
    if "arguments" in attrs:
        return _coerce_argument_list(attrs["arguments"])

    args = []
    if "subject" in attrs and str(attrs["subject"]).strip():
        args.append(str(attrs["subject"]))
    if "object" in attrs and str(attrs["object"]).strip():
        args.append(str(attrs["object"]))
    return args


def _predicate_expression(predicate: str, args: Iterable[str], ctx: Optional[dict] = None):
    """Create a predicate application with arbitrary arity."""
    return E(
        S(_normalize_text(predicate, lowercase=True)),
        *[_atom_from_text(arg, ctx=ctx) for arg in args],
    )


def _parse_body(body_str: str):
    """Parse a rule body expressed as a valid MeTTa S-expression."""
    return _parser.parse_single(body_str.strip())


def _condition_to_match(condition_atom):
    """Turn a declarative condition into a query against the current space."""
    return E(S("match"), S("&self"), condition_atom, _TRUE)


def _flatten_conjuncts(body_atom) -> list[object]:
    """Flatten nested `(and ...)` bodies into a single list of conjunct atoms."""
    if hasattr(body_atom, "get_children"):
        children = body_atom.get_children()
        if children and str(children[0]) == "and":
            conjuncts: list[object] = []
            for child in children[1:]:
                conjuncts.extend(_flatten_conjuncts(child))
            return conjuncts
    return [body_atom]


def _rule_body_to_query(body_atom):
    """
    Convert a declarative body like `(and (isa $x frog) (croaks $x))`
    into executable queries against `&self`.
    """
    conjuncts = _flatten_conjuncts(body_atom)
    if not conjuncts:
        raise ValueError("Rule body conjunction cannot be empty")
    if len(conjuncts) == 1:
        return _condition_to_match(conjuncts[0])
    return _condition_to_match(E(_COMMA, *conjuncts))


# ── safety filter ────────────────────────────────────────────────────────────


def _has_free_variable(value) -> bool:
    if value is None:
        return False
    if isinstance(value, str):
        return value.strip().startswith("$")
    if isinstance(value, Iterable):
        return any(str(item).strip().startswith("$") for item in value)
    return False


def is_safe_extraction(ext: Extraction) -> tuple[bool, str]:
    """Check if an extraction can be safely loaded into the AtomSpace.

    Mirrors PLN-RAG's _filter_statements + _has_valid_implication_shape:
    drops free-variable facts (would create meaningless `(eats $x flies)`
    style atoms) and rules whose body fails to parse.

    Returns (ok, reason).
    """
    cls = ext.extraction_class
    attrs = ext.attributes or {}

    if cls in {"fact", "negation", "property", "type_decl", "inheritance"}:
        keys = ("subject", "object", "entity", "type", "value", "child", "parent")
        for key in keys:
            if key in attrs and _has_free_variable(attrs[key]):
                return False, f"free variable in '{key}' for {cls}"
        if "arguments" in attrs and _has_free_variable(attrs["arguments"]):
            return False, f"free variable in 'arguments' for {cls}"

    if cls == "rule":
        body = attrs.get("body", "")
        if not body or not str(body).strip():
            return False, "rule has empty body"
        try:
            _parse_body(str(body))
        except Exception as exc:
            return False, f"rule body unparseable: {exc}"
        if not str(attrs.get("head_predicate", "")).strip():
            return False, "rule has empty head_predicate"

    return True, ""


# ── extraction → atom ────────────────────────────────────────────────────────


def extraction_to_atom(ext: Extraction, ctx: Optional[dict] = None) -> Optional[object]:
    """
    Map one Extraction to one MeTTa Atom.
    Returns None if the extraction_class is unknown or attributes are missing.
    """
    cls = ext.extraction_class
    attrs = ext.attributes or {}

    try:
        if cls == "fact":
            return _predicate_expression(
                attrs["predicate"], _fact_arguments(attrs), ctx=ctx
            )

        if cls == "type_decl":
            return E(
                S(":"),
                _atom_from_text(attrs["entity"], ctx=ctx),
                _atom_from_text(attrs["type"], ctx=ctx),
            )

        if cls == "property":
            return E(
                S("has"),
                _atom_from_text(attrs["entity"], ctx=ctx),
                _atom_from_text(attrs["property"], lowercase=True),
                _atom_from_text(attrs["value"], ctx=ctx),
            )

        if cls == "inheritance":
            return E(
                S("Inheritance"),
                _atom_from_text(attrs["child"], ctx=ctx),
                _atom_from_text(attrs["parent"], ctx=ctx),
            )

        if cls == "rule":
            head_args = _coerce_head_argument_atoms(attrs.get("head_args", "$x"))
            head = E(
                S(_normalize_text(attrs["head_predicate"], lowercase=True)),
                *head_args,
            )
            body = _rule_body_to_query(_parse_body(attrs["body"]))
            return E(S("="), head, body)

        if cls == "negation":
            inner = _predicate_expression(
                attrs["predicate"], _fact_arguments(attrs), ctx=ctx
            )
            return E(S("not"), inner)

        logger.debug("Unknown extraction_class %r - skipping", cls)
        return None

    except KeyError as exc:
        logger.warning("Missing attribute %s in %r extraction - skipping", exc, cls)
        return None
    except Exception as exc:
        logger.warning("Failed to translate extraction %r: %s - skipping", cls, exc)
        return None


# ── populate space (rich return) ─────────────────────────────────────────────


def populate_space(
    extractions: list[Extraction],
    space: Optional[GroundingSpaceRef] = None,
    skip_fuzzy: bool = True,
    source_text: str = "",
) -> tuple[GroundingSpaceRef, dict[str, Any]]:
    """Translate Extractions into Atoms and add them to a space.

    Args:
        extractions: output from LangExtract (result.extractions)
        space: existing space to populate; creates a new one if None
        skip_fuzzy: when True, only accept MATCH_EXACT / MATCH_GREATER /
                    MATCH_LESSER alignments (drops LLM paraphrases that
                    don't appear verbatim in the source)
        source_text: original text — used to build canonicalization context
                     (proper-name protection)

    Returns:
        (space, meta) where meta = {
            "atom_to_source": {atom_str: {"text", "char_interval", "class"}},
            "rejected": [{"reason", "extraction_class", "extraction_text"}],
            "added_count": int,
            "ctx": {...},
        }
    """
    if space is None:
        space = GroundingSpaceRef()

    ctx = build_canonicalization_context(source_text)
    atom_to_source: dict[str, dict[str, Any]] = {}
    rejected: list[dict[str, Any]] = []
    seen_atoms: set[str] = set()
    added = 0

    for ext in extractions:
        if skip_fuzzy and ext.alignment_status not in _EXACT_STATUSES:
            logger.debug("Skipping fuzzy extraction: %r", ext.extraction_text)
            rejected.append(
                {
                    "reason": "fuzzy alignment",
                    "extraction_class": ext.extraction_class,
                    "extraction_text": ext.extraction_text,
                }
            )
            continue

        ok, reason = is_safe_extraction(ext)
        if not ok:
            logger.warning(
                "Filtered unsafe extraction (%s): %r", reason, ext.extraction_text
            )
            rejected.append(
                {
                    "reason": reason,
                    "extraction_class": ext.extraction_class,
                    "extraction_text": ext.extraction_text,
                }
            )
            continue

        atom = extraction_to_atom(ext, ctx=ctx)
        if atom is None:
            rejected.append(
                {
                    "reason": "translate failed",
                    "extraction_class": ext.extraction_class,
                    "extraction_text": ext.extraction_text,
                }
            )
            continue

        atom_str = str(atom)
        if atom_str in seen_atoms:
            # Canonicalization collapsed two paraphrases to the same atom;
            # don't add twice. Source map keeps the first occurrence.
            continue
        seen_atoms.add(atom_str)
        space.add_atom(atom)
        char_interval = None
        if ext.char_interval is not None:
            char_interval = {
                "start_pos": ext.char_interval.start_pos,
                "end_pos": ext.char_interval.end_pos,
            }
        atom_to_source[atom_str] = {
            "text": ext.extraction_text,
            "char_interval": char_interval,
            "class": ext.extraction_class,
        }
        added += 1

    logger.info(
        "Added %d atoms to space (%d extractions, %d rejected)",
        added,
        len(extractions),
        len(rejected),
    )

    meta = {
        "atom_to_source": atom_to_source,
        "rejected": rejected,
        "added_count": added,
        "ctx": ctx,
    }
    return space, meta


def collect_predicate_heads(extractions: Iterable[Extraction]) -> list[str]:
    """Pull predicate heads (and rule head predicates) out of extractions.

    Used by the streaming pipeline to give later chunks a vocabulary hint.
    """
    heads: list[str] = []
    seen: set[str] = set()
    for ext in extractions:
        attrs = ext.attributes or {}
        for key in ("predicate", "head_predicate"):
            value = attrs.get(key)
            if isinstance(value, str) and value.strip():
                head = value.strip().lower().replace(" ", "-")
                if head not in seen:
                    seen.add(head)
                    heads.append(head)
    return heads
