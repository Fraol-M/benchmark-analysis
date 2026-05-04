"""
Scoring functions for NL → AtomSpace extraction benchmarks.

Metrics:
  1. Extraction accuracy  — do expected concept keywords appear in the atoms?
  2. Extraction richness  — atom count, predicate diversity, rules vs facts
  3. System metrics       — latency, errors
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import List


# ── Extraction accuracy ──────────────────────────────────────────────────────


@dataclass
class ExtractionAccuracyScore:
    """Accuracy score for a single case based purely on extracted atoms."""
    case_id: str
    correct: bool           # at least one expected keyword found in atoms
    keyword_hits: int
    keyword_total: int
    keyword_precision: float  # hits / total
    matched_keywords: List[str]
    missing_keywords: List[str]
    atom_sample: List[str]  # atoms that contained a match


def score_extraction_accuracy(
    case_id: str,
    atoms: List[str],
    expected_keywords: List[str],
) -> ExtractionAccuracyScore:
    """
    Check whether expected concept keywords appear anywhere in the atom strings.

    A case is correct if at least one expected keyword is found.
    Precision = matched / total expected keywords.
    """
    if not atoms:
        return ExtractionAccuracyScore(
            case_id=case_id,
            correct=False,
            keyword_hits=0,
            keyword_total=len(expected_keywords),
            keyword_precision=0.0,
            matched_keywords=[],
            missing_keywords=list(expected_keywords),
            atom_sample=[],
        )

    atoms_blob = " ".join(atoms).lower()
    atom_tokens = set(re.findall(r"[a-z0-9_-]+", atoms_blob))

    matched: List[str] = []
    missing: List[str] = []
    hit_atoms: List[str] = []

    for kw in expected_keywords:
        kw_lower = kw.lower()
        found = kw_lower in atoms_blob or kw_lower in atom_tokens
        if found:
            matched.append(kw)
            # collect up to 2 atoms that contain this keyword
            for a in atoms:
                if kw_lower in a.lower() and a not in hit_atoms:
                    hit_atoms.append(a)
                    break
        else:
            missing.append(kw)

    total = len(expected_keywords)
    hits = len(matched)

    return ExtractionAccuracyScore(
        case_id=case_id,
        correct=hits > 0,
        keyword_hits=hits,
        keyword_total=total,
        keyword_precision=hits / total if total else 0.0,
        matched_keywords=matched,
        missing_keywords=missing,
        atom_sample=hit_atoms[:5],
    )


# ── Extraction richness ──────────────────────────────────────────────────────


@dataclass
class ExtractionScore:
    """Structural richness metrics for a single case."""
    case_id: str
    atom_count: int
    unique_heads: int
    head_list: List[str]
    has_rules: bool
    has_facts: bool
    rule_count: int = 0
    fact_count: int = 0
    rule_ratio: float = 0.0   # rules / total atoms
    atoms_per_sentence: float = 0.0
    error: str | None = None


@dataclass
class ReasoningScore:
    """Proof/no-proof score for a single reasoning case."""
    case_id: str
    correct: bool
    expected_proof: bool
    has_proof: bool
    term_hits: int
    term_total: int
    matched_terms: List[str]
    missing_terms: List[str]
    query_error: str | None = None


def score_reasoning(
    case_id: str,
    proof_traces: List[str],
    *,
    expected_proof: bool,
    expected_terms: List[str] | None = None,
    query_error: str | None = None,
) -> ReasoningScore:
    """Score whether a reasoning query produced the expected proof status."""
    expected_terms = expected_terms or []
    has_proof = bool(proof_traces)
    proof_blob = " ".join(proof_traces).lower()

    matched: List[str] = []
    missing: List[str] = []
    for term in expected_terms:
        if term.lower() in proof_blob:
            matched.append(term)
        else:
            missing.append(term)

    terms_ok = not expected_terms or len(matched) == len(expected_terms)
    correct = query_error is None and has_proof == expected_proof and (
        terms_ok if expected_proof else True
    )

    return ReasoningScore(
        case_id=case_id,
        correct=correct,
        expected_proof=expected_proof,
        has_proof=has_proof,
        term_hits=len(matched),
        term_total=len(expected_terms),
        matched_terms=matched,
        missing_terms=missing,
        query_error=query_error,
    )


def score_extraction(
    case_id: str,
    atoms: List[str],
    sentence_count: int = 1,
) -> ExtractionScore:
    """Compute structural richness metrics from a list of atom strings."""
    heads: set[str] = set()
    rule_count = 0
    fact_count = 0

    for atom_str in atoms:
        atom_str = atom_str.strip()
        if not atom_str:
            continue

        if atom_str.startswith("(=") or "Implication" in atom_str:
            rule_count += 1
        else:
            fact_count += 1

        match = re.match(r"^\((\S+)", atom_str)
        if match:
            heads.add(match.group(1))

    total = len(atoms)
    return ExtractionScore(
        case_id=case_id,
        atom_count=total,
        unique_heads=len(heads),
        head_list=sorted(heads),
        has_rules=rule_count > 0,
        has_facts=fact_count > 0,
        rule_count=rule_count,
        fact_count=fact_count,
        rule_ratio=rule_count / total if total else 0.0,
        atoms_per_sentence=total / sentence_count if sentence_count else 0.0,
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
    avg_unique_heads: float = 0.0
    avg_rule_ratio: float = 0.0
    avg_atoms_per_sentence: float = 0.0
    avg_ingest_latency_s: float = 0.0
    total_errors: int = 0
    cases_with_rules: int = 0
    cases_with_facts: int = 0

    # Per-category accuracy
    category_accuracy: dict = field(default_factory=dict)
    # Per-category avg atom count
    category_avg_atoms: dict = field(default_factory=dict)
    # Per-hop-depth accuracy
    hop_accuracy: dict = field(default_factory=dict)


@dataclass
class ReasoningAggregateMetrics:
    """Aggregate proof/no-proof metrics for one backend."""
    backend: str
    total_cases: int = 0
    correct_count: int = 0
    accuracy: float = 0.0
    avg_query_latency_s: float = 0.0
    total_errors: int = 0
    expected_proof_cases: int = 0
    expected_no_proof_cases: int = 0
    false_negatives: int = 0
    false_positives: int = 0
    category_accuracy: dict = field(default_factory=dict)
    hop_accuracy: dict = field(default_factory=dict)


def compute_aggregate(
    backend: str,
    accuracy_scores: List[ExtractionAccuracyScore],
    extraction_scores: List[ExtractionScore],
    ingest_latencies: List[float],
    errors: List[str | None],
    categories: List[str],
    hop_depths: List[int],
) -> AggregateMetrics:
    """Compute aggregate metrics from per-case scores."""
    n = len(accuracy_scores)
    if n == 0:
        return AggregateMetrics(backend=backend)

    correct = sum(1 for s in accuracy_scores if s.correct)
    avg_kp = sum(s.keyword_precision for s in accuracy_scores) / n
    avg_atoms = sum(s.atom_count for s in extraction_scores) / n if extraction_scores else 0.0
    avg_heads = sum(s.unique_heads for s in extraction_scores) / n if extraction_scores else 0.0
    avg_rule_ratio = sum(s.rule_ratio for s in extraction_scores) / n if extraction_scores else 0.0
    avg_aps = sum(s.atoms_per_sentence for s in extraction_scores) / n if extraction_scores else 0.0
    avg_ingest = sum(ingest_latencies) / n if ingest_latencies else 0.0
    total_errors = sum(1 for e in errors if e)
    rules_count = sum(1 for s in extraction_scores if s.has_rules)
    facts_count = sum(1 for s in extraction_scores if s.has_facts)

    # Per-category accuracy + avg atoms
    cat_correct: dict[str, list[bool]] = {}
    cat_atoms: dict[str, list[int]] = {}
    for acc, ext, cat in zip(accuracy_scores, extraction_scores, categories):
        cat_correct.setdefault(cat, []).append(acc.correct)
        cat_atoms.setdefault(cat, []).append(ext.atom_count)

    category_accuracy = {cat: sum(v) / len(v) for cat, v in cat_correct.items()}
    category_avg_atoms = {cat: sum(v) / len(v) for cat, v in cat_atoms.items()}

    # Per-hop accuracy
    hop_correct: dict[int, list[bool]] = {}
    for acc, hop in zip(accuracy_scores, hop_depths):
        hop_correct.setdefault(hop, []).append(acc.correct)
    hop_accuracy = {hop: sum(v) / len(v) for hop, v in hop_correct.items()}

    return AggregateMetrics(
        backend=backend,
        total_cases=n,
        correct_count=correct,
        accuracy=correct / n,
        avg_keyword_precision=avg_kp,
        avg_atom_count=avg_atoms,
        avg_unique_heads=avg_heads,
        avg_rule_ratio=avg_rule_ratio,
        avg_atoms_per_sentence=avg_aps,
        avg_ingest_latency_s=avg_ingest,
        total_errors=total_errors,
        cases_with_rules=rules_count,
        cases_with_facts=facts_count,
        category_accuracy=category_accuracy,
        category_avg_atoms=category_avg_atoms,
        hop_accuracy=hop_accuracy,
    )


def compute_reasoning_aggregate(
    backend: str,
    reasoning_scores: List[ReasoningScore],
    query_latencies: List[float],
    categories: List[str],
    hop_depths: List[int],
) -> ReasoningAggregateMetrics:
    """Compute aggregate metrics from per-case reasoning scores."""
    n = len(reasoning_scores)
    if n == 0:
        return ReasoningAggregateMetrics(backend=backend)

    correct = sum(1 for score in reasoning_scores if score.correct)
    errors = sum(1 for score in reasoning_scores if score.query_error)
    expected_proof = sum(1 for score in reasoning_scores if score.expected_proof)
    expected_no_proof = n - expected_proof
    false_negatives = sum(
        1
        for score in reasoning_scores
        if score.expected_proof and not score.has_proof and not score.query_error
    )
    false_positives = sum(
        1
        for score in reasoning_scores
        if not score.expected_proof and score.has_proof and not score.query_error
    )

    cat_correct: dict[str, list[bool]] = {}
    for score, cat in zip(reasoning_scores, categories):
        cat_correct.setdefault(cat, []).append(score.correct)
    category_accuracy = {
        cat: sum(values) / len(values) for cat, values in cat_correct.items()
    }

    hop_correct: dict[int, list[bool]] = {}
    for score, hop in zip(reasoning_scores, hop_depths):
        hop_correct.setdefault(hop, []).append(score.correct)
    hop_accuracy = {hop: sum(values) / len(values) for hop, values in hop_correct.items()}

    return ReasoningAggregateMetrics(
        backend=backend,
        total_cases=n,
        correct_count=correct,
        accuracy=correct / n,
        avg_query_latency_s=sum(query_latencies) / n if query_latencies else 0.0,
        total_errors=errors,
        expected_proof_cases=expected_proof,
        expected_no_proof_cases=expected_no_proof,
        false_negatives=false_negatives,
        false_positives=false_positives,
        category_accuracy=category_accuracy,
        hop_accuracy=hop_accuracy,
    )
