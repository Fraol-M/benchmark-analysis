"""
Scoring functions for benchmark comparison.

Three types of metrics:
  1. Answer accuracy  — does the answer match the gold?
  2. Extraction stats — atom count, predicate diversity, parse success rate
  3. System metrics   — latency, errors
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import List


# ── Answer scoring ───────────────────────────────────────────────────────────


@dataclass
class AnswerScore:
    """Score for a single QA case."""
    case_id: str
    correct: bool
    keyword_hits: int
    keyword_total: int
    keyword_precision: float
    answer: str
    gold_answer: str
    gold_keywords: List[str]


def score_answer(
    case_id: str,
    answer: str,
    gold_answer: str,
    gold_keywords: List[str],
) -> AnswerScore:
    """
    Score an answer against gold using keyword overlap.

    A case is correct if at least one gold keyword appears in the answer.
    Keyword precision = (matched keywords) / (total gold keywords).
    """
    if not answer:
        return AnswerScore(
            case_id=case_id,
            correct=False,
            keyword_hits=0,
            keyword_total=len(gold_keywords),
            keyword_precision=0.0,
            answer="",
            gold_answer=gold_answer,
            gold_keywords=gold_keywords,
        )

    answer_lower = answer.lower()
    # Also check individual tokens for atom-style answers like "Sam"
    answer_tokens = set(re.findall(r"[a-z0-9_-]+", answer_lower))

    hits = 0
    for keyword in gold_keywords:
        kw = keyword.lower()
        if kw in answer_lower or kw in answer_tokens:
            hits += 1

    return AnswerScore(
        case_id=case_id,
        correct=hits > 0,
        keyword_hits=hits,
        keyword_total=len(gold_keywords),
        keyword_precision=hits / len(gold_keywords) if gold_keywords else 0.0,
        answer=answer,
        gold_answer=gold_answer,
        gold_keywords=gold_keywords,
    )


# ── Extraction scoring ──────────────────────────────────────────────────────


@dataclass
class ExtractionScore:
    """Extraction-level metrics for a single case."""
    case_id: str
    atom_count: int
    unique_heads: int
    head_list: List[str]
    has_rules: bool
    has_facts: bool
    error: str | None = None


def score_extraction(case_id: str, atoms: List[str]) -> ExtractionScore:
    """Compute extraction metrics from a list of atom strings."""
    heads = set()
    has_rules = False
    has_facts = False

    for atom_str in atoms:
        atom_str = atom_str.strip()
        if not atom_str:
            continue

        # Detect rules
        if atom_str.startswith("(=") or "Implication" in atom_str:
            has_rules = True
        else:
            has_facts = True

        # Extract head symbol
        match = re.match(r"^\((\S+)", atom_str)
        if match:
            heads.add(match.group(1))

    return ExtractionScore(
        case_id=case_id,
        atom_count=len(atoms),
        unique_heads=len(heads),
        head_list=sorted(heads),
        has_rules=has_rules,
        has_facts=has_facts,
    )


# ── Aggregate metrics ────────────────────────────────────────────────────────


@dataclass
class AggregateMetrics:
    """Aggregate metrics across all cases for one backend."""
    backend: str
    total_cases: int = 0
    correct_count: int = 0
    accuracy: float = 0.0
    avg_keyword_precision: float = 0.0
    avg_atom_count: float = 0.0
    avg_ingest_latency_s: float = 0.0
    avg_query_latency_s: float = 0.0
    total_errors: int = 0
    cases_with_rules: int = 0
    cases_with_facts: int = 0

    # Per-category accuracy
    category_accuracy: dict = field(default_factory=dict)
    # Per-hop-depth accuracy
    hop_accuracy: dict = field(default_factory=dict)


def compute_aggregate(
    backend: str,
    answer_scores: List[AnswerScore],
    extraction_scores: List[ExtractionScore],
    ingest_latencies: List[float],
    query_latencies: List[float],
    errors: List[str | None],
    categories: List[str],
    hop_depths: List[int],
) -> AggregateMetrics:
    """Compute aggregate metrics from per-case scores."""
    n = len(answer_scores)
    if n == 0:
        return AggregateMetrics(backend=backend)

    correct = sum(1 for s in answer_scores if s.correct)
    avg_kp = sum(s.keyword_precision for s in answer_scores) / n
    avg_atoms = sum(s.atom_count for s in extraction_scores) / n if extraction_scores else 0
    avg_ingest = sum(ingest_latencies) / n if ingest_latencies else 0
    avg_query = sum(query_latencies) / n if query_latencies else 0
    total_errors = sum(1 for e in errors if e)
    rules_count = sum(1 for s in extraction_scores if s.has_rules)
    facts_count = sum(1 for s in extraction_scores if s.has_facts)

    # Per-category accuracy
    cat_correct: dict[str, list[bool]] = {}
    for score, cat in zip(answer_scores, categories):
        cat_correct.setdefault(cat, []).append(score.correct)
    category_accuracy = {
        cat: sum(vals) / len(vals) for cat, vals in cat_correct.items()
    }

    # Per-hop accuracy
    hop_correct: dict[int, list[bool]] = {}
    for score, hop in zip(answer_scores, hop_depths):
        hop_correct.setdefault(hop, []).append(score.correct)
    hop_accuracy = {
        hop: sum(vals) / len(vals) for hop, vals in hop_correct.items()
    }

    return AggregateMetrics(
        backend=backend,
        total_cases=n,
        correct_count=correct,
        accuracy=correct / n,
        avg_keyword_precision=avg_kp,
        avg_atom_count=avg_atoms,
        avg_ingest_latency_s=avg_ingest,
        avg_query_latency_s=avg_query,
        total_errors=total_errors,
        cases_with_rules=rules_count,
        cases_with_facts=facts_count,
        category_accuracy=category_accuracy,
        hop_accuracy=hop_accuracy,
    )
