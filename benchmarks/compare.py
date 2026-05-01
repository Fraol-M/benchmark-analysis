#!/usr/bin/env python3
"""
Benchmark orchestrator: Demo vs PLN-RAG.

Loads test cases, runs each through one or both backends,
scores results, and generates a markdown report.

Usage:
    python benchmarks/compare.py --backend demo
    python benchmarks/compare.py --backend plnrag
    python benchmarks/compare.py --backend both
    python benchmarks/compare.py --backend demo --cases fish-smart-yesno,frog-green-rule
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from dataclasses import asdict
from pathlib import Path
from typing import List

# Ensure benchmarks/ is on path
_BENCH_ROOT = Path(__file__).resolve().parent
_WORKSPACE = _BENCH_ROOT.parent
sys.path.insert(0, str(_BENCH_ROOT))
sys.path.insert(0, str(_WORKSPACE))

from metrics.scoring import (
    AnswerScore,
    ExtractionScore,
    AggregateMetrics,
    score_answer,
    score_extraction,
    compute_aggregate,
)


# ── Dataset loading ──────────────────────────────────────────────────────────


def load_cases(path: str | None = None, filter_ids: List[str] | None = None) -> List[dict]:
    """Load benchmark cases from JSON."""
    if path is None:
        path = str(_BENCH_ROOT / "datasets" / "cases.json")
    with open(path, "r", encoding="utf-8") as f:
        cases = json.load(f)
    if filter_ids:
        cases = [c for c in cases if c["id"] in filter_ids]
    return cases


# ── Backend loading ──────────────────────────────────────────────────────────


def load_backend(name: str):
    """Lazy-load a backend by name."""
    if name == "demo":
        from runners.demo_backend import DemoBackend
        return DemoBackend()
    elif name == "plnrag":
        from runners.plnrag_backend import PLNRAGBackend
        return PLNRAGBackend()
    else:
        raise ValueError(f"Unknown backend: {name}. Use 'demo' or 'plnrag'.")


# ── Single case runner ───────────────────────────────────────────────────────


def run_case(backend, case: dict) -> dict:
    """Run a single benchmark case through a backend."""
    case_id = case["id"]
    print(f"  [{backend.name}] Running case: {case_id} ...", end=" ", flush=True)

    # Reset backend state
    backend.reset()

    # For PLN-RAG, set case-specific isolation
    if hasattr(backend, "set_case_id"):
        backend.set_case_id(case_id)

    # Ingest
    ingest_result = backend.ingest(case["texts"])

    if ingest_result.error:
        print(f"INGEST ERROR: {ingest_result.error}")
        return {
            "case_id": case_id,
            "category": case.get("category", ""),
            "hop_depth": case.get("hop_depth", 0),
            "ingest_error": ingest_result.error,
            "ingest_latency_s": ingest_result.latency_s,
            "atom_count": 0,
            "atoms": [],
            "query_error": None,
            "query_latency_s": 0.0,
            "answer": "",
            "answer_score": None,
            "extraction_score": None,
        }

    # Query
    query_result = backend.query(case["question"])

    # Get all atoms for extraction scoring
    atoms = backend.get_atoms()

    # Score
    ans_score = score_answer(
        case_id=case_id,
        answer=query_result.answer,
        gold_answer=case["gold_answer"],
        gold_keywords=case["gold_keywords"],
    )
    ext_score = score_extraction(case_id=case_id, atoms=atoms)

    status = "✅" if ans_score.correct else "❌"
    print(f"{status} (atoms={ext_score.atom_count}, answer='{query_result.answer[:60]}')")

    # Cleanup
    backend.reset()

    return {
        "case_id": case_id,
        "category": case.get("category", ""),
        "hop_depth": case.get("hop_depth", 0),
        "texts": case["texts"],
        "question": case["question"],
        "gold_answer": case["gold_answer"],
        "gold_keywords": case["gold_keywords"],
        "ingest_error": ingest_result.error,
        "ingest_latency_s": ingest_result.latency_s,
        "atom_count": ingest_result.atom_count,
        "atoms": atoms[:50],  # Cap for report size
        "query_error": query_result.error,
        "query_latency_s": query_result.latency_s,
        "answer": query_result.answer,
        "raw_atoms": query_result.raw_atoms[:20],
        "answer_score": asdict(ans_score),
        "extraction_score": asdict(ext_score),
    }


# ── Report generation ────────────────────────────────────────────────────────


def generate_report(
    results: dict[str, list[dict]],
    aggregates: dict[str, AggregateMetrics],
    output_path: str,
) -> str:
    """Generate a markdown benchmark report."""
    lines = [
        "# Benchmark Report — Demo vs PLN-RAG",
        "",
        f"Generated: {time.strftime('%Y-%m-%d %H:%M:%S')}",
        "",
        "---",
        "",
    ]

    # Summary table
    lines.append("## Summary")
    lines.append("")
    lines.append("| Metric | " + " | ".join(aggregates.keys()) + " |")
    lines.append("| --- | " + " | ".join("---" for _ in aggregates) + " |")

    metrics_rows = [
        ("Total Cases", lambda a: str(a.total_cases)),
        ("Correct", lambda a: f"{a.correct_count}/{a.total_cases}"),
        ("**Accuracy**", lambda a: f"**{a.accuracy:.1%}**"),
        ("Avg Keyword Precision", lambda a: f"{a.avg_keyword_precision:.2f}"),
        ("Avg Atom Count", lambda a: f"{a.avg_atom_count:.1f}"),
        ("Cases with Rules", lambda a: str(a.cases_with_rules)),
        ("Cases with Facts", lambda a: str(a.cases_with_facts)),
        ("Avg Ingest Latency", lambda a: f"{a.avg_ingest_latency_s:.2f}s"),
        ("Avg Query Latency", lambda a: f"{a.avg_query_latency_s:.2f}s"),
        ("Total Errors", lambda a: str(a.total_errors)),
    ]

    for label, fn in metrics_rows:
        row = f"| {label} | " + " | ".join(fn(a) for a in aggregates.values()) + " |"
        lines.append(row)

    lines.append("")

    # Per-category breakdown
    lines.append("## Accuracy by Category")
    lines.append("")

    all_categories = set()
    for a in aggregates.values():
        all_categories.update(a.category_accuracy.keys())

    if all_categories:
        lines.append("| Category | " + " | ".join(aggregates.keys()) + " |")
        lines.append("| --- | " + " | ".join("---" for _ in aggregates) + " |")
        for cat in sorted(all_categories):
            row = f"| {cat} | "
            for a in aggregates.values():
                val = a.category_accuracy.get(cat)
                row += f"{val:.0%} | " if val is not None else "N/A | "
            lines.append(row)
        lines.append("")

    # Per-hop breakdown
    lines.append("## Accuracy by Reasoning Depth")
    lines.append("")

    all_hops = set()
    for a in aggregates.values():
        all_hops.update(a.hop_accuracy.keys())

    if all_hops:
        lines.append("| Hop Depth | " + " | ".join(aggregates.keys()) + " |")
        lines.append("| --- | " + " | ".join("---" for _ in aggregates) + " |")
        for hop in sorted(all_hops):
            row = f"| {hop} | "
            for a in aggregates.values():
                val = a.hop_accuracy.get(hop)
                row += f"{val:.0%} | " if val is not None else "N/A | "
            lines.append(row)
        lines.append("")

    # Detailed results per backend
    for backend_name, case_results in results.items():
        lines.append(f"## Detailed Results — {backend_name}")
        lines.append("")
        lines.append("| Case | Category | Hops | Correct | Answer (truncated) | Atoms |")
        lines.append("| --- | --- | --- | --- | --- | --- |")

        for cr in case_results:
            ans_score = cr.get("answer_score") or {}
            correct = "✅" if ans_score.get("correct") else "❌"
            answer = (cr.get("answer") or "")[:40].replace("|", "\\|")
            atoms = cr.get("atom_count", 0)
            lines.append(
                f"| {cr['case_id']} | {cr['category']} | {cr['hop_depth']} "
                f"| {correct} | {answer} | {atoms} |"
            )
        lines.append("")

        # Show errors
        errors = [cr for cr in case_results if cr.get("ingest_error") or cr.get("query_error")]
        if errors:
            lines.append(f"### Errors — {backend_name}")
            lines.append("")
            for cr in errors:
                lines.append(f"- **{cr['case_id']}**: ingest={cr.get('ingest_error', 'ok')}, query={cr.get('query_error', 'ok')}")
            lines.append("")

    # Write report
    report = "\n".join(lines)
    with open(output_path, "w", encoding="utf-8") as f:
        f.write(report)

    return report


# ── Main ─────────────────────────────────────────────────────────────────────


def main() -> int:
    cli = argparse.ArgumentParser(description="Run Demo vs PLN-RAG benchmarks.")
    cli.add_argument(
        "--backend",
        choices=["demo", "plnrag", "both"],
        default="demo",
        help="Which backend(s) to benchmark (default: demo)",
    )
    cli.add_argument(
        "--cases",
        default=None,
        help="Comma-separated case IDs to run (default: all)",
    )
    cli.add_argument(
        "--dataset",
        default=None,
        help="Path to cases JSON file (default: datasets/cases.json)",
    )
    cli.add_argument(
        "--output",
        default=None,
        help="Output report path (default: benchmarks/report/results.md)",
    )
    args = cli.parse_args()

    # Load cases
    filter_ids = args.cases.split(",") if args.cases else None
    cases = load_cases(path=args.dataset, filter_ids=filter_ids)
    print(f"Loaded {len(cases)} benchmark cases.")

    # Determine backends
    backend_names = ["demo", "plnrag"] if args.backend == "both" else [args.backend]

    # Run benchmarks
    all_results: dict[str, list[dict]] = {}
    all_aggregates: dict[str, AggregateMetrics] = {}

    for backend_name in backend_names:
        print(f"\n{'='*60}")
        print(f"Running backend: {backend_name}")
        print(f"{'='*60}")

        try:
            backend = load_backend(backend_name)
        except Exception as exc:
            print(f"  ⚠️  Could not load {backend_name}: {exc}")
            print(f"  Skipping this backend.")
            continue

        case_results = []
        for case in cases:
            try:
                result = run_case(backend, case)
                case_results.append(result)
            except Exception as exc:
                print(f"  ❌ Case {case['id']} crashed: {exc}")
                case_results.append({
                    "case_id": case["id"],
                    "category": case.get("category", ""),
                    "hop_depth": case.get("hop_depth", 0),
                    "ingest_error": str(exc),
                    "ingest_latency_s": 0.0,
                    "atom_count": 0,
                    "atoms": [],
                    "query_error": str(exc),
                    "query_latency_s": 0.0,
                    "answer": "",
                    "answer_score": {"correct": False, "keyword_hits": 0,
                                     "keyword_total": 0, "keyword_precision": 0.0,
                                     "answer": "", "gold_answer": case["gold_answer"],
                                     "gold_keywords": case["gold_keywords"]},
                    "extraction_score": {"case_id": case["id"], "atom_count": 0,
                                         "unique_heads": 0, "head_list": [],
                                         "has_rules": False, "has_facts": False},
                })

        all_results[backend_name] = case_results

        # Compute aggregate
        answer_scores = [
            AnswerScore(**cr["answer_score"]) if cr.get("answer_score") else
            AnswerScore(case_id=cr["case_id"], correct=False, keyword_hits=0,
                        keyword_total=0, keyword_precision=0.0, answer="",
                        gold_answer="", gold_keywords=[])
            for cr in case_results
        ]
        extraction_scores = [
            ExtractionScore(**cr["extraction_score"]) if cr.get("extraction_score") else
            ExtractionScore(case_id=cr["case_id"], atom_count=0, unique_heads=0,
                            head_list=[], has_rules=False, has_facts=False)
            for cr in case_results
        ]

        agg = compute_aggregate(
            backend=backend_name,
            answer_scores=answer_scores,
            extraction_scores=extraction_scores,
            ingest_latencies=[cr.get("ingest_latency_s", 0) for cr in case_results],
            query_latencies=[cr.get("query_latency_s", 0) for cr in case_results],
            errors=[cr.get("ingest_error") or cr.get("query_error") for cr in case_results],
            categories=[cr.get("category", "") for cr in case_results],
            hop_depths=[cr.get("hop_depth", 0) for cr in case_results],
        )
        all_aggregates[backend_name] = agg

        print(f"\n  Summary: {agg.correct_count}/{agg.total_cases} correct ({agg.accuracy:.0%})")
        print(f"  Avg ingest: {agg.avg_ingest_latency_s:.2f}s, Avg query: {agg.avg_query_latency_s:.2f}s")

    # Generate report
    output_path = args.output or str(_BENCH_ROOT / "report" / "results.md")
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    report = generate_report(all_results, all_aggregates, output_path)
    print(f"\n📄 Report written to: {output_path}")

    # Also save raw JSON
    json_path = output_path.replace(".md", ".json")
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump({
            "results": all_results,
            "aggregates": {k: asdict(v) for k, v in all_aggregates.items()},
        }, f, indent=2, default=str)
    print(f"📊 Raw data saved to: {json_path}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
