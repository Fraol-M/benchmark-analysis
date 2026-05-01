#!/usr/bin/env python3
"""
Benchmark orchestrator: Demo vs PLN-RAG.

Measures NL → AtomSpace extraction quality only.
No query or reasoning step is performed.

Usage:
    python benchmarks/compare.py --backend demo
    python benchmarks/compare.py --backend plnrag
    python benchmarks/compare.py --backend both
    python benchmarks/compare.py --backend demo --cases fish-smart-yesno,frog-basic-facts
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

_BENCH_ROOT = Path(__file__).resolve().parent
_WORKSPACE = _BENCH_ROOT.parent
sys.path.insert(0, str(_BENCH_ROOT))
sys.path.insert(0, str(_WORKSPACE))

from metrics.scoring import (
    ExtractionAccuracyScore,
    ExtractionScore,
    AggregateMetrics,
    score_extraction_accuracy,
    score_extraction,
    compute_aggregate,
)


# ── Dataset ──────────────────────────────────────────────────────────────────


def load_cases(path: str | None = None, filter_ids: List[str] | None = None) -> List[dict]:
    if path is None:
        path = str(_BENCH_ROOT / "datasets" / "cases.json")
    with open(path, "r", encoding="utf-8") as f:
        cases = json.load(f)
    if filter_ids:
        cases = [c for c in cases if c["id"] in filter_ids]
    return cases


# ── Backend loading ───────────────────────────────────────────────────────────


def load_backend(name: str):
    if name == "demo":
        from runners.demo_backend import DemoBackend
        return DemoBackend()
    elif name == "plnrag":
        from runners.plnrag_backend import PLNRAGBackend
        return PLNRAGBackend()
    raise ValueError(f"Unknown backend: {name}. Use 'demo' or 'plnrag'.")


# ── Single case runner ────────────────────────────────────────────────────────


def run_case(backend, case: dict) -> dict:
    case_id = case["id"]
    texts = case["texts"]
    expected_keywords = case.get("expected_keywords", [])

    print(f"  [{backend.name}] {case_id} ...", end=" ", flush=True)

    backend.reset()
    if hasattr(backend, "set_case_id"):
        backend.set_case_id(case_id)

    ingest_result = backend.ingest(texts)

    if ingest_result.error:
        print(f"ERROR: {ingest_result.error}")
        return _error_record(case, ingest_result.latency_s, ingest_result.error)

    atoms = backend.get_atoms()
    acc_score = score_extraction_accuracy(case_id, atoms, expected_keywords)
    ext_score = score_extraction(case_id, atoms, sentence_count=len(texts))

    status = "✅" if acc_score.correct else "❌"
    print(
        f"{status}  atoms={ext_score.atom_count}  "
        f"heads={ext_score.unique_heads}  "
        f"kw={acc_score.keyword_hits}/{acc_score.keyword_total}  "
        f"{ingest_result.latency_s:.1f}s"
    )

    backend.reset()

    return {
        "case_id": case_id,
        "category": case.get("category", ""),
        "hop_depth": case.get("hop_depth", 0),
        "texts": texts,
        "expected_keywords": expected_keywords,
        "ingest_error": None,
        "ingest_latency_s": ingest_result.latency_s,
        "atom_count": ext_score.atom_count,
        "atoms": atoms[:60],
        "accuracy_score": asdict(acc_score),
        "extraction_score": asdict(ext_score),
    }


def _error_record(case: dict, latency: float, error: str) -> dict:
    case_id = case["id"]
    expected_keywords = case.get("expected_keywords", [])
    return {
        "case_id": case_id,
        "category": case.get("category", ""),
        "hop_depth": case.get("hop_depth", 0),
        "texts": case["texts"],
        "expected_keywords": expected_keywords,
        "ingest_error": error,
        "ingest_latency_s": latency,
        "atom_count": 0,
        "atoms": [],
        "accuracy_score": asdict(ExtractionAccuracyScore(
            case_id=case_id, correct=False, keyword_hits=0,
            keyword_total=len(expected_keywords), keyword_precision=0.0,
            matched_keywords=[], missing_keywords=list(expected_keywords), atom_sample=[],
        )),
        "extraction_score": asdict(ExtractionScore(
            case_id=case_id, atom_count=0, unique_heads=0, head_list=[],
            has_rules=False, has_facts=False,
        )),
    }


# ── Report ────────────────────────────────────────────────────────────────────


def generate_report(
    results: dict[str, list[dict]],
    aggregates: dict[str, AggregateMetrics],
    output_path: str,
) -> str:
    lines = [
        "# Benchmark Report — NL → AtomSpace Extraction",
        "",
        f"Generated: {time.strftime('%Y-%m-%d %H:%M:%S')}",
        "",
        "---",
        "",
        "## Summary",
        "",
        "| Metric | " + " | ".join(aggregates.keys()) + " |",
        "| --- | " + " | ".join("---" for _ in aggregates) + " |",
    ]

    rows = [
        ("Total Cases",            lambda a: str(a.total_cases)),
        ("Keyword Hits",           lambda a: f"{a.correct_count}/{a.total_cases}"),
        ("**Extraction Accuracy**",lambda a: f"**{a.accuracy:.1%}**"),
        ("Avg Keyword Precision",  lambda a: f"{a.avg_keyword_precision:.2f}"),
        ("Avg Atom Count",         lambda a: f"{a.avg_atom_count:.1f}"),
        ("Avg Unique Heads",       lambda a: f"{a.avg_unique_heads:.1f}"),
        ("Avg Rule Ratio",         lambda a: f"{a.avg_rule_ratio:.2f}"),
        ("Avg Atoms / Sentence",   lambda a: f"{a.avg_atoms_per_sentence:.2f}"),
        ("Cases with Rules",       lambda a: str(a.cases_with_rules)),
        ("Cases with Facts",       lambda a: str(a.cases_with_facts)),
        ("Avg Ingest Latency",     lambda a: f"{a.avg_ingest_latency_s:.2f}s"),
        ("Total Errors",           lambda a: str(a.total_errors)),
    ]
    for label, fn in rows:
        lines.append("| " + label + " | " + " | ".join(fn(a) for a in aggregates.values()) + " |")

    lines += ["", "## Extraction Accuracy by Category", ""]
    all_cats = sorted({cat for a in aggregates.values() for cat in a.category_accuracy})
    if all_cats:
        lines.append("| Category | " + " | ".join(aggregates.keys()) + " |")
        lines.append("| --- | " + " | ".join("---" for _ in aggregates) + " |")
        for cat in all_cats:
            row = f"| {cat} | "
            for a in aggregates.values():
                v = a.category_accuracy.get(cat)
                row += (f"{v:.0%} | " if v is not None else "N/A | ")
            lines.append(row)

    lines += ["", "## Avg Atom Count by Category", ""]
    if all_cats:
        lines.append("| Category | " + " | ".join(aggregates.keys()) + " |")
        lines.append("| --- | " + " | ".join("---" for _ in aggregates) + " |")
        for cat in all_cats:
            row = f"| {cat} | "
            for a in aggregates.values():
                v = a.category_avg_atoms.get(cat)
                row += (f"{v:.1f} | " if v is not None else "N/A | ")
            lines.append(row)

    lines += ["", "## Accuracy by Input Complexity (Hop Depth)", ""]
    all_hops = sorted({h for a in aggregates.values() for h in a.hop_accuracy})
    if all_hops:
        lines.append("| Hop Depth | " + " | ".join(aggregates.keys()) + " |")
        lines.append("| --- | " + " | ".join("---" for _ in aggregates) + " |")
        for hop in all_hops:
            row = f"| {hop} | "
            for a in aggregates.values():
                v = a.hop_accuracy.get(hop)
                row += (f"{v:.0%} | " if v is not None else "N/A | ")
            lines.append(row)

    for backend_name, case_results in results.items():
        lines += ["", f"## Detailed Results — {backend_name}", ""]
        lines.append("| Case | Category | Hops | ✓ | KW | Atoms | Heads | Rules | Latency |")
        lines.append("| --- | --- | --- | --- | --- | --- | --- | --- | --- |")
        for cr in case_results:
            acc = cr.get("accuracy_score") or {}
            ext = cr.get("extraction_score") or {}
            ok = "✅" if acc.get("correct") else "❌"
            kw = f"{acc.get('keyword_hits',0)}/{acc.get('keyword_total',0)}"
            lines.append(
                f"| {cr['case_id']} | {cr['category']} | {cr['hop_depth']} "
                f"| {ok} | {kw} | {ext.get('atom_count',0)} "
                f"| {ext.get('unique_heads',0)} | {ext.get('rule_count',0)} "
                f"| {cr.get('ingest_latency_s',0):.1f}s |"
            )

        errors = [cr for cr in case_results if cr.get("ingest_error")]
        if errors:
            lines += ["", f"### Errors — {backend_name}", ""]
            for cr in errors:
                lines.append(f"- **{cr['case_id']}**: {cr['ingest_error']}")

    report = "\n".join(lines)
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        f.write(report)
    return report


# ── Main ──────────────────────────────────────────────────────────────────────


def main() -> int:
    cli = argparse.ArgumentParser(description="Benchmark NL → AtomSpace extraction.")
    cli.add_argument("--backend", choices=["demo", "plnrag", "both"], default="demo")
    cli.add_argument("--cases", default=None, help="Comma-separated case IDs")
    cli.add_argument("--dataset", default=None, help="Path to cases JSON")
    cli.add_argument("--output", default=None, help="Output .md path")
    args = cli.parse_args()

    filter_ids = args.cases.split(",") if args.cases else None
    cases = load_cases(path=args.dataset, filter_ids=filter_ids)
    print(f"Loaded {len(cases)} benchmark cases.")

    backend_names = ["demo", "plnrag"] if args.backend == "both" else [args.backend]
    all_results: dict[str, list[dict]] = {}
    all_aggregates: dict[str, AggregateMetrics] = {}

    for backend_name in backend_names:
        print(f"\n{'='*60}\nRunning backend: {backend_name}\n{'='*60}")
        try:
            backend = load_backend(backend_name)
        except Exception as exc:
            print(f"  ⚠️  Could not load {backend_name}: {exc}\n  Skipping.")
            continue

        case_results = []
        for case in cases:
            try:
                case_results.append(run_case(backend, case))
            except Exception as exc:
                print(f"  ❌ {case['id']} crashed: {exc}")
                case_results.append(_error_record(case, 0.0, str(exc)))

        all_results[backend_name] = case_results

        acc_scores = [ExtractionAccuracyScore(**cr["accuracy_score"]) for cr in case_results]
        ext_scores = [ExtractionScore(**cr["extraction_score"]) for cr in case_results]

        agg = compute_aggregate(
            backend=backend_name,
            accuracy_scores=acc_scores,
            extraction_scores=ext_scores,
            ingest_latencies=[cr.get("ingest_latency_s", 0) for cr in case_results],
            errors=[cr.get("ingest_error") for cr in case_results],
            categories=[cr.get("category", "") for cr in case_results],
            hop_depths=[cr.get("hop_depth", 0) for cr in case_results],
        )
        all_aggregates[backend_name] = agg

        print(f"\n  Accuracy: {agg.correct_count}/{agg.total_cases} ({agg.accuracy:.0%})")
        print(f"  Avg atoms: {agg.avg_atom_count:.1f}  Avg heads: {agg.avg_unique_heads:.1f}  Avg latency: {agg.avg_ingest_latency_s:.2f}s")

    output_path = args.output or str(_BENCH_ROOT / "report" / "results.md")
    report = generate_report(all_results, all_aggregates, output_path)
    print(f"\n📄 Report: {output_path}")

    json_path = output_path.replace(".md", ".json")
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump({"results": all_results, "aggregates": {k: asdict(v) for k, v in all_aggregates.items()}}, f, indent=2, default=str)
    print(f"📊 Raw data: {json_path}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
