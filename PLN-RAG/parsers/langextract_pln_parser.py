from __future__ import annotations

import os
from typing import Any, List
from dataclasses import dataclass, field

from config import get_settings
from core.extraction.langextract_chunker import LangExtractChunker
from core.extraction.langextract_examples import load_langextract_prompt_spec
from core.extraction.langextract_pln import (
    build_pln_query,
    collect_predicate_heads,
    format_context_hint,
    log_rejections,
    translate_extractions_to_pln,
    translate_query_extractions_to_pln,
)
from core.pln.postprocessor import PLNPostprocessor
from core.pln.predicate_mapping import (
    LLMPredicateRelationClassifier,
    PredicateMappingEngine,
)
from core.pln.schema_alignment import PLNSchemaAligner
from core.pln.predicate_registry import PredicateRegistry


@dataclass
class ParseResult:
    """Result of parsing natural language into PLN."""
    statements: List[str] = field(default_factory=list)
    queries: List[str] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)


class LangExtractPLNParser:
    """
    LangExtract-based parser for PLN-RAG.

    Pipeline:
        natural language -> LangExtract extraction objects -> PLN strings
    """

    def __init__(self):
        cfg = get_settings()
        self._model_id = _first_nonempty(
            cfg.langextract_model_id,
            os.environ.get("LANGEXTRACT_MODEL_ID"),
            "gemini-2.5-flash",
        )
        self._model_url = _first_nonempty(
            cfg.langextract_model_url,
            os.environ.get("LANGEXTRACT_MODEL_URL"),
        )
        self._api_key = _first_nonempty(
            cfg.langextract_api_key,
            os.environ.get("LANGEXTRACT_API_KEY"),
            cfg.gemini_api_key,
            os.environ.get("GEMINI_API_KEY"),
            cfg.openai_api_key,
            os.environ.get("OPENAI_API_KEY"),
        )
        self._extraction_passes = cfg.langextract_extraction_passes
        self._max_workers = cfg.langextract_max_workers
        self._skip_fuzzy = cfg.langextract_skip_fuzzy
        self._predicate_heads: list[str] = []
        predicate_registry = None
        if cfg.predicate_registry_enabled:
            schema_aligner = PLNSchemaAligner(PLNPostprocessor.STRUCTURAL_HEADS)
            model_id_lower = str(cfg.langextract_model_id or "").lower()
            mapping_gemini_key = cfg.gemini_api_key
            mapping_openai_key = cfg.openai_api_key
            if not mapping_gemini_key and "gemini" in model_id_lower:
                mapping_gemini_key = cfg.langextract_api_key
            if not mapping_openai_key and any(
                marker in model_id_lower for marker in ("openai", "gpt")
            ):
                mapping_openai_key = cfg.langextract_api_key
            classifier = LLMPredicateRelationClassifier(
                enabled=(
                    cfg.predicate_mapping_enabled
                    and cfg.predicate_mapping_llm_enabled
                ),
                gemini_api_key=mapping_gemini_key,
                gemini_model=cfg.gemini_model,
                openai_api_key=mapping_openai_key,
                openai_model=cfg.openai_model,
                timeout_seconds=cfg.predicate_mapping_timeout,
            )
            mapping_engine = (
                PredicateMappingEngine(
                    classifier=classifier,
                    proof_threshold=cfg.predicate_mapping_proof_threshold,
                    retrieval_min_score=cfg.predicate_mapping_min_score,
                    retrieval_top_k=cfg.predicate_mapping_top_k,
                    max_candidates=cfg.predicate_mapping_max_candidates,
                    total_timeout_seconds=cfg.predicate_mapping_total_timeout,
                )
                if cfg.predicate_mapping_enabled
                else None
            )
            predicate_registry = PredicateRegistry(
                path=cfg.predicate_registry_path,
                schema_aligner=schema_aligner,
                mapping_engine=mapping_engine,
            )
        self._postprocessor = PLNPostprocessor(
            predicate_registry=predicate_registry,
        )

        if not self._api_key and not self._model_url:
            raise ValueError(
                "No LangExtract model credentials found. Set LANGEXTRACT_API_KEY "
                "or GEMINI_API_KEY, or set LANGEXTRACT_MODEL_URL for a local model."
            )

        spec = load_langextract_prompt_spec(cfg.langextract_examples_path)
        self._statement_prompt = spec.statement_prompt
        self._query_prompt = spec.query_prompt
        self._statement_examples = spec.statement_examples
        self._query_examples = spec.query_examples

    def create_chunker(self) -> LangExtractChunker:
        return LangExtractChunker()

    def set_predicate_card_store(self, card_store) -> None:
        self._postprocessor.set_predicate_card_store(card_store)

    def reset(self, clear_registry: bool = False) -> None:
        self._predicate_heads = []
        if clear_registry:
            self._postprocessor.reset_registry()

    def parse(self, text: str, context: list[str]) -> ParseResult:
        try:
            prompt = self._statement_prompt + format_context_hint(
                context,
                self._predicate_heads,
            )
            extractions = self._extract(text, prompt, self._statement_examples)
            self._remember_predicates(collect_predicate_heads(extractions))

            translated = translate_extractions_to_pln(
                extractions,
                source_text=text,
                skip_fuzzy=self._skip_fuzzy,
            )
            processed = self._postprocessor.process(
                text=text,
                statements=translated.statements,
                queries=[],
                context=context,
                plan_queries=False,
            )
            log_rejections("LangExtractPLNParser", translated.rejected)
            return ParseResult(
                statements=processed.statements,
                queries=[],
                metadata={
                    "statement_to_source": _remap_metadata(
                        translated.statements,
                        processed.statements,
                        translated.statement_to_source,
                    ),
                    "rejected": [
                        {
                            "extraction_class": item.extraction_class,
                            "extraction_text": item.extraction_text,
                            "reason": item.reason,
                        }
                        for item in translated.rejected
                    ],
                    "canonicalization_context": translated.ctx,
                    "schema_alignment": processed.alignment_decisions,
                    "predicate_registry": processed.registry_decisions,
                },
            )
        except Exception as exc:
            print(f"[LangExtractPLNParser] Failed for '{text}': {exc}")
            return ParseResult()

    def debug_parse(self, text: str, context: list[str]) -> dict[str, Any]:
        prompt = self._statement_prompt + format_context_hint(
            context,
            self._predicate_heads,
        )
        extractions = self._extract(text, prompt, self._statement_examples)
        self._remember_predicates(collect_predicate_heads(extractions))

        translated = translate_extractions_to_pln(
            extractions,
            source_text=text,
            skip_fuzzy=self._skip_fuzzy,
        )
        processed = self._postprocessor.process(
            text=text,
            statements=translated.statements,
            queries=[],
            context=context,
            plan_queries=False,
        )

        return {
            "langextract_postprocessed": {
                "statements": translated.statements,
                "rejected": [
                    {
                        "extraction_class": item.extraction_class,
                        "extraction_text": item.extraction_text,
                        "reason": item.reason,
                    }
                    for item in translated.rejected
                ],
                "canonicalization_context": translated.ctx,
                "statement_sources": translated.statement_to_source,
            },
            "pln_canonicalized": processed.statements,
            "schema_alignment": processed.alignment_decisions,
            "predicate_registry": processed.registry_decisions,
        }

    def parse_query(self, text: str, context: list[str]) -> ParseResult:
        try:
            prompt = self._query_prompt + format_context_hint(
                context,
                self._predicate_heads,
            )
            extractions = self._extract(text, prompt, self._query_examples)
            translated = translate_query_extractions_to_pln(
                extractions,
                source_text=text,
            )
            processed = self._postprocessor.process(
                text=text,
                statements=translated.statements,
                queries=translated.queries,
                context=context,
                plan_queries=True,
            )
            log_rejections("LangExtractPLNParser.query", translated.rejected)
            return ParseResult(
                statements=processed.statements,
                queries=processed.queries,
                metadata={
                    "query_to_source": _remap_metadata(
                        translated.queries,
                        processed.queries,
                        translated.query_to_source,
                    ),
                    "rejected": [
                        {
                            "extraction_class": item.extraction_class,
                            "extraction_text": item.extraction_text,
                            "reason": item.reason,
                        }
                        for item in translated.rejected
                    ],
                    "canonicalization_context": translated.ctx,
                    "schema_alignment": processed.alignment_decisions,
                    "predicate_registry": processed.registry_decisions,
                },
            )
        except Exception as exc:
            print(f"[LangExtractPLNParser] Query failed for '{text}': {exc}")
            return ParseResult()

    def debug_parse_query(self, text: str, context: list[str]) -> dict[str, Any]:
        prompt = self._query_prompt + format_context_hint(
            context,
            self._predicate_heads,
        )
        extractions = self._extract(text, prompt, self._query_examples)
        translated = translate_query_extractions_to_pln(
            extractions,
            source_text=text,
        )
        processed = self._postprocessor.process(
            text=text,
            statements=translated.statements,
            queries=translated.queries,
            context=context,
            plan_queries=True,
        )

        return {
            "langextract_postprocessed": {
                "queries": translated.queries,
                "rejected": [
                    {
                        "extraction_class": item.extraction_class,
                        "extraction_text": item.extraction_text,
                        "reason": item.reason,
                    }
                    for item in translated.rejected
                ],
                "canonicalization_context": translated.ctx,
                "query_sources": translated.query_to_source,
            },
            "pln_canonicalized": processed.queries,
            "supporting_statements": processed.statements,
            "schema_alignment": processed.alignment_decisions,
            "predicate_registry": processed.registry_decisions,
        }

    def _extract(self, text: str, prompt: str, examples: list[Any]) -> list[Any]:
        import langextract as lx

        result = lx.extract(
            text_or_documents=text,
            prompt_description=prompt,
            examples=examples,
            model_id=self._model_id,
            model_url=self._model_url,
            api_key=self._api_key,
            extraction_passes=self._extraction_passes,
            max_char_buffer=max(len(text) + 1, 1000),
            max_workers=self._max_workers,
            show_progress=False,
        )
        return list(result.extractions) if hasattr(result, "extractions") else []

    def _remember_predicates(self, heads: list[str]) -> None:
        for head in heads:
            if head not in self._predicate_heads:
                self._predicate_heads.append(head)


def _first_nonempty(*values: Any) -> str | None:
    for value in values:
        if value is not None and str(value).strip():
            return str(value).strip()
    return None


def _remap_metadata(
    before: list[str],
    after: list[str],
    metadata: dict[str, dict[str, Any]],
) -> dict[str, dict[str, Any]]:
    remapped: dict[str, dict[str, Any]] = {}
    for index, item in enumerate(after):
        if item in metadata:
            remapped[item] = metadata[item]
            continue
        if index < len(before) and before[index] in metadata:
            remapped[item] = metadata[before[index]]
    return remapped


__all__ = [
    "LangExtractPLNParser",
    "build_pln_query",
    "translate_extractions_to_pln",
    "translate_query_extractions_to_pln",
]
