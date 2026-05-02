from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Iterable, Sequence


@dataclass
class TranslationReject:
    atom: str
    reason: str


@dataclass
class PLNTranslationResult:
    statements: list[str] = field(default_factory=list)
    rejected: list[TranslationReject] = field(default_factory=list)
    atom_map: dict[str, str] = field(default_factory=dict)

    @property
    def statement_count(self) -> int:
        return len(self.statements)

    @property
    def rejected_count(self) -> int:
        return len(self.rejected)


class PLNTranslationError(ValueError):
    pass


_STRUCTURAL_HEADS = {
    "and": "And",
    ",": "And",
    "or": "Or",
    "not": "Not",
    "Not": "Not",
    "And": "And",
    "Or": "Or",
}


def translate_atoms_to_pln(
    atoms: Iterable[object],
    *,
    truth_value: str = "(STV 1.0 1.0)",
) -> PLNTranslationResult:
    """Translate this project's Hyperon-style MeTTa atoms to PeTTaChainer PLN.

    The current extractor emits executable Hyperon rules such as:

        (= (smart $x) (match &self (eats $x fish) True))

    PeTTaChainer expects statements wrapped with proof ids and truth values:

        (: smart_rule (Implication ...) (STV 1.0 1.0))
    """
    result = PLNTranslationResult()
    seen: set[str] = set()

    for index, atom in enumerate(atoms, start=1):
        atom_text = " ".join(str(atom).split())
        try:
            expr = _parse_sexp(atom_text)
            statement = _translate_top_level(expr, index, truth_value)
        except Exception as exc:
            result.rejected.append(
                TranslationReject(atom=atom_text, reason=str(exc))
            )
            continue

        if statement in seen:
            continue
        seen.add(statement)
        result.statements.append(statement)
        result.atom_map[atom_text] = statement

    return result


def build_pln_query(predicate_expr: str) -> str:
    """Build a PeTTaChainer query from a bare predicate expression.

    Example:
        ``(smart kebede)`` -> ``(: $prf (Smart kebede) $tv)``
    """
    expr = _parse_sexp(predicate_expr)
    predicate = _translate_predicate_expr(expr)
    return f"(: $prf {predicate} $tv)"


def suggest_pln_queries(atoms: Iterable[object], limit: int = 8) -> list[str]:
    """Generate simple query candidates from translated facts and rule heads."""
    suggestions: list[str] = []
    seen: set[str] = set()

    for atom in atoms:
        atom_text = " ".join(str(atom).split())
        try:
            expr = _parse_sexp(atom_text)
            predicate_expr = _queryable_expr(expr)
            if predicate_expr is None:
                continue
            query = build_pln_query(_render_source(predicate_expr))
        except Exception:
            continue
        if query in seen:
            continue
        seen.add(query)
        suggestions.append(query)
        if len(suggestions) >= limit:
            break

    return suggestions


def _translate_top_level(expr, index: int, truth_value: str) -> str:
    if not isinstance(expr, list) or not expr:
        raise PLNTranslationError("top-level atom must be an S-expression")

    head = str(expr[0])
    if head == "=":
        name, payload = _translate_rule(expr, index)
    elif head == ":":
        name, payload = _translate_type_decl(expr, index)
    elif head in {"not", "Not"}:
        if len(expr) != 2:
            raise PLNTranslationError("negation must contain exactly one expression")
        name = f"neg_{index}"
        payload = f"(Not {_translate_predicate_expr(expr[1])})"
    else:
        name = _statement_name(expr, index, "fact")
        payload = _translate_predicate_expr(expr)

    return f"(: {name} {payload} {truth_value})"


def _translate_type_decl(expr, index: int) -> tuple[str, str]:
    if len(expr) != 3:
        raise PLNTranslationError("type declaration must look like (: entity type)")
    payload = f"(IsA {_translate_arg(expr[1])} {_translate_arg(expr[2])})"
    return f"type_{index}", payload


def _translate_rule(expr, index: int) -> tuple[str, str]:
    if len(expr) != 3:
        raise PLNTranslationError("rule must look like (= head body)")
    head_expr = expr[1]
    body_expr = expr[2]
    conclusion = _translate_predicate_expr(head_expr)
    premises = _extract_rule_premises(body_expr)
    if not premises:
        raise PLNTranslationError("rule body produced no premises")

    premise_block = " ".join(premises)
    payload = (
        f"(Implication (Premises {premise_block}) "
        f"(Conclusions {conclusion}))"
    )
    return _statement_name(head_expr, index, "rule"), payload


def _extract_rule_premises(body_expr) -> list[str]:
    if (
        isinstance(body_expr, list)
        and len(body_expr) >= 4
        and str(body_expr[0]) == "match"
        and str(body_expr[1]) == "&self"
    ):
        body_expr = body_expr[2]

    return _premise_items(body_expr)


def _premise_items(expr) -> list[str]:
    if not isinstance(expr, list) or not expr:
        raise PLNTranslationError(f"rule premise must be an expression: {expr!r}")

    head = str(expr[0])
    if head in {",", "and", "And"}:
        premises: list[str] = []
        for child in expr[1:]:
            premises.extend(_premise_items(child))
        return premises

    if head in {"or", "Or"}:
        return [_translate_structural_expr(expr)]

    if head in {"not", "Not"}:
        return [_translate_structural_expr(expr)]

    return [_translate_predicate_expr(expr)]


def _translate_structural_expr(expr) -> str:
    if not isinstance(expr, list) or not expr:
        raise PLNTranslationError("structural expression cannot be empty")
    head = _STRUCTURAL_HEADS.get(str(expr[0]))
    if head is None:
        raise PLNTranslationError(f"unsupported structural head: {expr[0]}")
    children = " ".join(_translate_predicate_expr(child) for child in expr[1:])
    return f"({head} {children})"


def _translate_predicate_expr(expr) -> str:
    if not isinstance(expr, list) or not expr:
        raise PLNTranslationError(f"predicate must be an S-expression: {expr!r}")

    raw_head = str(expr[0])
    if raw_head in {"not", "Not", "or", "Or", "and", "And", ","}:
        return _translate_structural_expr(expr)

    translated_head = _translate_head(raw_head, expr)
    args = " ".join(_translate_arg(arg) for arg in expr[1:])
    return f"({translated_head}{(' ' + args) if args else ''})"


def _translate_head(raw_head: str, expr: Sequence[object]) -> str:
    if raw_head in {"isa", "Inheritance", "IsA"} and len(expr) == 3:
        return "IsA"
    if raw_head == ":":
        return "IsA"
    return _pascal_symbol(raw_head)


def _translate_arg(arg) -> str:
    if isinstance(arg, list):
        return _translate_predicate_expr(arg)
    token = str(arg)
    if token.startswith("$") or token.startswith("?"):
        return "$" + _snake_symbol(token[1:], preserve_case=True)
    if token.startswith('"') and token.endswith('"'):
        return token
    return _snake_symbol(token)


def _statement_name(expr, index: int, suffix: str) -> str:
    if isinstance(expr, list) and expr:
        head = _snake_symbol(str(expr[0]))
        args = [
            _snake_symbol(str(arg).lstrip("$?"))
            for arg in expr[1:3]
            if not isinstance(arg, list)
        ]
        parts = [part for part in args + [head, suffix] if part]
        return "_".join(parts)[:80] or f"{suffix}_{index}"
    return f"{suffix}_{index}"


def _queryable_expr(expr):
    if not isinstance(expr, list) or not expr:
        return None
    if str(expr[0]) == "=" and len(expr) == 3:
        return expr[1]
    if str(expr[0]) == ":" and len(expr) == 3:
        return ["isa", expr[1], expr[2]]
    if str(expr[0]) in {"not", "Not"}:
        return None
    return expr


def _pascal_symbol(value: str) -> str:
    if value in {"IsA", "And", "Or", "Not"}:
        return value
    parts = [p for p in re.split(r"[^A-Za-z0-9]+", value) if p]
    if not parts:
        return value
    return "".join(part[:1].upper() + part[1:].lower() for part in parts)


def _snake_symbol(value: str, *, preserve_case: bool = False) -> str:
    value = value.strip()
    if not value:
        return value
    value = re.sub(r"(?<=[a-z0-9])(?=[A-Z])", "_", value)
    value = re.sub(r"[^A-Za-z0-9_]+", "_", value)
    value = re.sub(r"_+", "_", value).strip("_")
    return value if preserve_case else value.lower()


def _parse_sexp(text: str):
    tokens = _tokenize(text)
    if not tokens:
        raise PLNTranslationError("empty atom")
    expr, pos = _parse_tokens(tokens, 0)
    if pos != len(tokens):
        raise PLNTranslationError("unexpected trailing tokens")
    return expr


def _tokenize(text: str) -> list[str]:
    tokens: list[str] = []
    i = 0
    while i < len(text):
        ch = text[i]
        if ch.isspace():
            i += 1
            continue
        if ch in "()":
            tokens.append(ch)
            i += 1
            continue
        if ch == '"':
            j = i + 1
            escaped = False
            while j < len(text):
                if text[j] == '"' and not escaped:
                    break
                escaped = text[j] == "\\" and not escaped
                if text[j] != "\\":
                    escaped = False
                j += 1
            if j >= len(text):
                raise PLNTranslationError("unterminated string literal")
            tokens.append(text[i : j + 1])
            i = j + 1
            continue
        j = i
        while j < len(text) and not text[j].isspace() and text[j] not in "()":
            j += 1
        tokens.append(text[i:j])
        i = j
    return tokens


def _parse_tokens(tokens: list[str], pos: int):
    token = tokens[pos]
    if token != "(":
        return token, pos + 1

    pos += 1
    expr = []
    while pos < len(tokens) and tokens[pos] != ")":
        child, pos = _parse_tokens(tokens, pos)
        expr.append(child)
    if pos >= len(tokens):
        raise PLNTranslationError("missing closing parenthesis")
    return expr, pos + 1


def _render_source(expr) -> str:
    if isinstance(expr, list):
        return "(" + " ".join(_render_source(child) for child in expr) + ")"
    return str(expr)
