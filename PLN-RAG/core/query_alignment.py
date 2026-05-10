from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any, Iterable


STOPWORDS = {
    "a",
    "an",
    "and",
    "are",
    "as",
    "at",
    "be",
    "by",
    "do",
    "does",
    "did",
    "for",
    "from",
    "has",
    "have",
    "had",
    "how",
    "in",
    "is",
    "it",
    "of",
    "on",
    "or",
    "that",
    "the",
    "this",
    "to",
    "was",
    "were",
    "what",
    "when",
    "where",
    "which",
    "who",
    "why",
    "with",
}
YES_NO_STARTERS = {
    "is",
    "are",
    "was",
    "were",
    "does",
    "do",
    "did",
    "can",
    "could",
    "has",
    "have",
    "had",
}
NON_QUERY_TARGET_HEADS = {
    "Implication",
    "Premises",
    "Conclusions",
    "STV",
    "And",
    "Or",
}

SExpr = str | list["SExpr"]


@dataclass
class QueryAlignmentResult:
    queries: list[str] = field(default_factory=list)
    matches: list[dict[str, Any]] = field(default_factory=list)


def extract_query_targets(statements: Iterable[str]) -> list[str]:
    """Extract fact payloads and rule conclusions that can become queries."""
    targets: list[str] = []
    seen: set[str] = set()

    for statement in statements:
        for expr in _parse_sexprs(statement):
            for target in _targets_from_statement(expr):
                serialized = _serialize(target)
                if serialized and serialized not in seen:
                    seen.add(serialized)
                    targets.append(serialized)

    return targets


def build_aligned_queries(
    question: str,
    matches: Iterable[dict[str, Any]],
) -> QueryAlignmentResult:
    question_entities = extract_question_entities(question)
    question_terms = extract_question_terms(question)
    is_yes_no = _is_yes_no_question(question)

    queries: list[str] = []
    seen_queries: set[str] = set()
    debug_matches: list[dict[str, Any]] = []

    for match in matches:
        targets = _targets_from_match(match)
        match_queries: list[str] = []

        for target in targets:
            for expr in _parse_sexprs(target):
                grounded = _ground_target(
                    expr,
                    question_entities=question_entities,
                    question_terms=question_terms,
                    is_yes_no=is_yes_no,
                )
                if grounded is None:
                    continue
                query = f"(: $prf {_serialize(grounded)} $tv)"
                if query in seen_queries:
                    continue
                seen_queries.add(query)
                queries.append(query)
                match_queries.append(query)

        debug_matches.append(
            {
                "score": match.get("score"),
                "nl": match.get("nl", ""),
                "pln": match.get("pln", []),
                "query_targets": targets,
                "aligned_queries": match_queries,
            }
        )

    return QueryAlignmentResult(queries=queries, matches=debug_matches)


def extract_question_entities(question: str) -> list[str]:
    entities: list[str] = []
    for token in re.findall(r"\b[A-Z][A-Za-z0-9_-]*\b", question):
        if token.lower() in STOPWORDS or token.lower() in YES_NO_STARTERS:
            continue
        canonical = _canonical_symbol(token, lemmatize=False)
        if canonical and canonical not in entities:
            entities.append(canonical)
    return entities


def extract_question_terms(question: str) -> set[str]:
    terms: set[str] = set()
    for token in re.findall(r"\b[A-Za-z0-9][A-Za-z0-9_-]*\b", question):
        lowered = token.lower()
        if len(lowered) < 2 or lowered in STOPWORDS:
            continue
        canonical = _canonical_symbol(lowered)
        if canonical:
            terms.add(canonical)
    return terms


def _targets_from_match(match: dict[str, Any]) -> list[str]:
    targets = match.get("query_targets") or []
    if isinstance(targets, list) and targets:
        return [str(target) for target in targets if str(target).strip()]

    pln = match.get("pln") or []
    if isinstance(pln, list):
        return extract_query_targets([str(item) for item in pln])
    return []


def _targets_from_statement(expr: SExpr) -> list[SExpr]:
    if not isinstance(expr, list) or len(expr) < 3 or expr[0] != ":":
        return []

    payload = expr[2]
    if not isinstance(payload, list) or not payload:
        return []

    if payload[0] == "Implication":
        return [
            conclusion
            for conclusion in _extract_conclusions(payload)
            if _is_query_target(conclusion)
        ]

    return [payload] if _is_query_target(payload) else []


def _extract_conclusions(expr: list[SExpr]) -> list[SExpr]:
    conclusions: list[SExpr] = []
    for item in expr:
        if (
            isinstance(item, list)
            and item
            and item[0] == "Conclusions"
        ):
            conclusions.extend(child for child in item[1:] if isinstance(child, list))
    return conclusions


def _is_query_target(expr: SExpr) -> bool:
    return (
        isinstance(expr, list)
        and bool(expr)
        and isinstance(expr[0], str)
        and expr[0] not in NON_QUERY_TARGET_HEADS
    )


def _ground_target(
    expr: SExpr,
    *,
    question_entities: list[str],
    question_terms: set[str],
    is_yes_no: bool,
) -> SExpr | None:
    if not _is_query_target(expr):
        return None

    if is_yes_no:
        return None

    variables = sorted(_variables(expr))
    if not variables:
        constants = _constants(expr)
        if question_terms and constants and not question_terms.intersection(constants):
            return None
        return expr

    if len(variables) == 1 and len(question_entities) == 1:
        return _replace_symbol(expr, variables[0], question_entities[0])


def _variables(expr: SExpr) -> set[str]:
    if isinstance(expr, str):
        return {expr} if expr.startswith(("$", "?")) and expr not in {"$prf", "$tv", "?prf", "?tv"} else set()
    variables: set[str] = set()
    for item in expr:
        variables.update(_variables(item))
    return variables


def _constants(expr: SExpr, *, is_head: bool = True) -> set[str]:
    if isinstance(expr, str):
        if is_head or expr.startswith(("$", "?")):
            return set()
        canonical = _canonical_symbol(expr)
        return {canonical} if canonical and not canonical.replace("_", "").isdigit() else set()

    constants: set[str] = set()
    for index, item in enumerate(expr):
        constants.update(_constants(item, is_head=index == 0))
    return constants


def _replace_symbol(expr: SExpr, old: str, new: str) -> SExpr:
    if isinstance(expr, str):
        return new if expr == old else expr
    return [_replace_symbol(item, old, new) for item in expr]


def _is_yes_no_question(question: str) -> bool:
    tokens = re.findall(r"\b[A-Za-z]+\b", question.lower())
    return bool(tokens) and tokens[0] in YES_NO_STARTERS


def _parse_sexprs(text: str) -> list[SExpr]:
    tokens = re.findall(r"\(|\)|[^\s()]+", text)
    roots: list[SExpr] = []
    stack: list[list[SExpr]] = []

    for token in tokens:
        if token == "(":
            node: list[SExpr] = []
            if stack:
                stack[-1].append(node)
            else:
                roots.append(node)
            stack.append(node)
            continue
        if token == ")":
            if not stack:
                return []
            stack.pop()
            continue
        if stack:
            stack[-1].append(token)
        else:
            roots.append(token)

    return roots if not stack else []


def _serialize(expr: SExpr) -> str:
    if isinstance(expr, str):
        return expr
    return "(" + " ".join(_serialize(item) for item in expr) + ")"


def _canonical_symbol(token: str, lemmatize: bool = True) -> str:
    token = token.strip()
    if not token:
        return token
    token = re.sub(r"(?<=[a-z0-9])(?=[A-Z])", "_", token)
    token = token.replace("-", "_")
    token = re.sub(r"[^A-Za-z0-9_]", "_", token)
    token = re.sub(r"_+", "_", token).strip("_")
    token = token.lower()
    if lemmatize:
        token = "_".join(_singularize(part) for part in token.split("_") if part)
    return token


def _singularize(word: str) -> str:
    if len(word) <= 3:
        return word
    if word.endswith("ies") and len(word) > 4:
        return word[:-3] + "y"
    if word.endswith("ses") and len(word) > 4:
        return word[:-2]
    if word.endswith("s") and not word.endswith(("ss", "us", "is")):
        return word[:-1]
    return word
