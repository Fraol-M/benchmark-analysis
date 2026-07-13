import asyncio
import re
import time
from typing import List

from config import get_settings
from parsers.langextract_pln_parser import LangExtractPLNParser
from core.query.alignment import (
    build_aligned_queries,
    extract_forward_seed_terms,
    extract_query_targets,
    filter_queries_by_question_intent,
)
from core.query.intent import (
    QuestionIntent,
    QuestionMode,
    parse_question_intent,
    query_intent_score,
    query_matches_intent,
)
from core.reasoning.reasoner import ProofOutcome, Reasoner
from core.answering.answer_generator import AnswerGenerator
from storage.vector_store import VectorStore
from api.models import (
    IngestItemResult,
    QueryResponse,
    DebugIngestItemResult,
    DebugIngestChunkResult,
    LangExtractPostprocessed,
    LangExtractQueryPostprocessed,
    DebugQueryResponse,
)


class PLNRAGService:
    """
    Orchestrates the full LangExtract pipeline:
      Text → Chunker → LangExtractPLNParser → Reasoner → AnswerGenerator
    """

    def __init__(self, parser: LangExtractPLNParser):
        cfg = get_settings()
        self._parser = parser
        self._chunker = parser.create_chunker()
        self._reasoner = Reasoner()
        self._vector_store = VectorStore() if cfg.use_vector_store else None
        if self._vector_store and hasattr(parser, "set_predicate_card_store"):
            parser.set_predicate_card_store(self._vector_store)
        self._answer_gen = AnswerGenerator()
        self._context_top_k = cfg.context_top_k
        self._query_fallback_enabled = cfg.query_fallback_enabled

    #  Ingest

    async def ingest_batch(self, texts: List[str]) -> List[IngestItemResult]:
        """
        Process texts sequentially so each sentence can see
        all previously ingested atoms as context.
        """
        results = []
        for text in texts:
            result = await asyncio.get_event_loop().run_in_executor(
                None, self._ingest_single, text
            )
            results.append(result)
        return results

    async def debug_ingest_batch(self, texts: List[str]) -> List[DebugIngestItemResult]:
        results = []
        for text in texts:
            result = await asyncio.get_event_loop().run_in_executor(
                None, self._debug_ingest_single, text
            )
            results.append(result)
        return results

    def _ingest_single(self, text: str) -> IngestItemResult:
        try:
            # 1. Chunk large texts
            chunks = self._chunker.chunk(text)
            all_atoms: List[str] = []

            for chunk in chunks:
                # 2. Retrieve context from atomspace + vector store
                if self._vector_store:
                    context, vector = self._vector_store.retrieve_context(
                        chunk, top_k=self._context_top_k
                    )
                else:
                    context, vector = [], []
                # Also supplement with recent atoms from disk
                context = self._enrich_context(context)

                # 3. Parse chunk → PLN atoms
                parse_result = self._parser.parse(chunk, context)

                if not parse_result.statements:
                    print(f"[Service] No statements for chunk: '{chunk[:60]}...'")
                    continue

                # 4. Add to atomspace via reasoner
                added = self._reasoner.add_statements(
                    parse_result.statements,
                    provenance=parse_result.metadata.get("statement_to_source", {}),
                )
                
                all_atoms.extend(added)

                # 5. Store in vector DB for future context retrieval
                if added and self._vector_store:
                    self._vector_store.store(
                        chunk,
                        added,
                        vector,
                        metadata=parse_result.metadata,
                        query_targets=extract_query_targets(added),
                    )

            return IngestItemResult(text=text, atoms=all_atoms, status="success")

        except Exception as e:
            import traceback

            traceback.print_exc()
            return IngestItemResult(text=text, status="failed", error=str(e))

    def _debug_ingest_single(self, text: str) -> DebugIngestItemResult:
        try:
            chunks = self._chunker.chunk(text)
            chunk_results: List[DebugIngestChunkResult] = []

            for chunk in chunks:
                if self._vector_store:
                    context, vector = self._vector_store.retrieve_context(
                        chunk, top_k=self._context_top_k
                    )
                else:
                    context, vector = [], []
                context = self._enrich_context(context)

                debug_info = self._parser.debug_parse(chunk, context)
                langextract_info = debug_info.get("langextract_postprocessed", {})
                pln_canonicalized = debug_info.get("pln_canonicalized", [])
                schema_alignment = debug_info.get("schema_alignment", [])
                predicate_registry = debug_info.get("predicate_registry", [])
                debug_metadata = {
                    "statement_to_source": langextract_info.get(
                        "statement_sources",
                        {},
                    ),
                    "rejected": langextract_info.get("rejected", []),
                    "canonicalization_context": langextract_info.get(
                        "canonicalization_context",
                        {},
                    ),
                    "schema_alignment": schema_alignment,
                    "predicate_registry": predicate_registry,
                }

                added = self._reasoner.add_statements(
                    pln_canonicalized,
                    provenance=debug_metadata.get("statement_to_source", {}),
                )

                if added and self._vector_store:
                    self._vector_store.store(
                        chunk,
                        added,
                        vector,
                        metadata=debug_metadata,
                        query_targets=extract_query_targets(added),
                    )

                chunk_results.append(
                    DebugIngestChunkResult(
                        chunk=chunk,
                        context=context,
                        langextract_postprocessed=LangExtractPostprocessed(
                            **langextract_info
                        ),
                        pln_canonicalized=pln_canonicalized,
                        atomspace_added=added,
                        schema_alignment=schema_alignment,
                        predicate_registry=predicate_registry,
                    )
                )

            return DebugIngestItemResult(
                text=text,
                chunks=chunk_results,
                status="success",
            )

        except Exception as e:
            import traceback

            traceback.print_exc()
            return DebugIngestItemResult(text=text, status="failed", error=str(e))

    def _enrich_context(self, rag_context: List[str], max_atoms: int = 50) -> List[str]:
        """
        Supplement RAG-retrieved context with the most recent atoms
        from the atomspace file, deduplicating and capping at max_atoms.
        This ensures the parser always has the full predicate vocabulary
        even when RAG similarity scores are low.
        """
        from config import get_settings
        import os

        cfg = get_settings()
        file_atoms: List[str] = []
        if os.path.exists(cfg.atomspace_path):
            with open(cfg.atomspace_path, "r") as f:
                file_atoms = [l.strip() for l in f if l.strip()]

        seen = set()
        merged = []
        for atom in file_atoms + rag_context:
            if atom not in seen:
                seen.add(atom)
                merged.append(atom)

        return merged[-max_atoms:]

    #  Query

    def _qdrant_context_with_matches(
        self,
        text: str,
        top_k: int,
    ) -> tuple[List[str], List[dict]]:
        if not self._vector_store:
            return [], []
        matches, _ = self._vector_store.search(text, top_k=top_k)
        context: List[str] = []
        for item in matches:
            pln = item.get("pln", [])
            if isinstance(pln, list):
                context.extend(pln)
        return context, matches

    def _alignment_matches(self, matches: List[dict]) -> List[dict]:
        cfg = get_settings()
        min_score = float(getattr(cfg, "query_alignment_min_score", 0.0) or 0.0)
        return [
            match
            for match in matches
            if float(match.get("score") or 0.0) >= min_score
        ]

    def _query_candidates(
        self,
        question: str,
        _qdrant_queries: List[str],
        parser_queries: List[str],
    ) -> List[tuple[str, str]]:
        entries: List[tuple[str, str]] = []
        seen: set[str] = set()
        intent = parse_question_intent(question)
        filtered_parser = filter_queries_by_question_intent(
            question,
            parser_queries,
        )
        # Retrieval supplies context and vocabulary, never proof targets.
        # A candidate rejected by the intent gate must not be restored.
        for source, queries in (("parser", filtered_parser),):
            for query in queries:
                clean = " ".join(str(query).split())
                if not clean or clean in seen or not query_matches_intent(intent, clean):
                    continue
                seen.add(clean)
                entries.append((clean, source))
        entries.sort(
            key=lambda item: query_intent_score(intent, item[0]),
            reverse=True,
        )
        return entries

    async def query(self, question: str) -> QueryResponse:
        cfg = get_settings()
        intent = parse_question_intent(question)
        # 1. Retrieve context for translation
        t0 = time.perf_counter()
        if self._vector_store:
            context, qdrant_matches = self._qdrant_context_with_matches(
                question,
                top_k=max(
                    self._context_top_k,
                    int(getattr(cfg, "query_alignment_top_k", 0) or 0),
                ),
            )
        else:
            context, qdrant_matches = [], []
        alignment_matches = self._alignment_matches(qdrant_matches)
        qdrant_alignment = (
            build_aligned_queries(question, alignment_matches)
            if getattr(cfg, "query_alignment_enabled", True)
            else None
        )
        qdrant_queries = qdrant_alignment.queries if qdrant_alignment else []
        seed_terms = extract_forward_seed_terms(alignment_matches)
        context = self._enrich_context(context)
        context_retrieval_seconds = time.perf_counter() - t0

        # 2. Parse question → PLN query
        t1 = time.perf_counter()
        parse_result = self._parser.parse_query(question, context)
        parse_query_seconds = time.perf_counter() - t1

        original_query = parse_result.queries[0] if parse_result.queries else (
            qdrant_queries[0] if qdrant_queries else ""
        )
        candidate_entries = self._query_candidates(
            question,
            qdrant_queries,
            parse_result.queries,
        )
        if intent.mode in {QuestionMode.FACTORS, QuestionMode.SUFFICIENCY}:
            requirements = self._reasoner.explain_requirements(
                set(intent.terms),
                entity=intent.entities[0] if intent.entities else "",
                direction=intent.direction,
            )
            answer = self._answer_structured_intent(intent, requirements)
            return QueryResponse(
                question=question,
                pln_query="",
                original_query=original_query,
                executed_query="",
                query_source="none",
                fallback_used=False,
                query_status="well_aligned" if requirements else "weakly_aligned",
                raw_proof="",
                sources=[],
                answer=answer,
                qdrant_aligned_queries=qdrant_queries,
                execution_candidates=[],
                intent_mode=intent.mode.value,
                proof_status="unanswered",
                requirements=requirements,
                context_retrieval_seconds=round(context_retrieval_seconds, 4),
                parse_query_seconds=round(parse_query_seconds, 4),
                reasoning_seconds=0.0,
                source_lookup_seconds=0.0,
                answer_generation_seconds=0.0,
            )
        if not candidate_entries:
            return QueryResponse(
                question=question,
                pln_query="",
                original_query="",
                executed_query="",
                query_source="none",
                fallback_used=False,
                query_status="no_query",
                raw_proof="",
                sources=[],
                answer="I couldn't translate this question into a logical query.",
                qdrant_aligned_queries=qdrant_queries,
                execution_candidates=[],
                intent_mode=intent.mode.value,
                proof_status="unanswered",
                context_retrieval_seconds=round(context_retrieval_seconds, 4),
                parse_query_seconds=round(parse_query_seconds, 4),
                reasoning_seconds=0.0,
                source_lookup_seconds=0.0,
                answer_generation_seconds=0.0,
            )

        # 3. Run reasoning read-only via PeTTaChainer against ordered candidates
        t2 = time.perf_counter()
        proof_traces: List[str] = []
        proof_outcome = ProofOutcome()
        first_outcome: ProofOutcome | None = None
        executed_query = ""
        executed_source = "none"
        candidates = (
            candidate_entries
            if self._query_fallback_enabled
            else candidate_entries[:1]
        )

        candidate_count_total = len(candidates)
        max_tries = int(getattr(cfg, "query_candidate_max_tries", 0) or 0)
        if self._query_fallback_enabled and max_tries > 0:
            candidates = candidates[:max_tries]
        candidate_count_tried = len(candidates)

        executed_candidate_index: int | None = None
        retry_used = False
        for idx, (candidate, source) in enumerate(candidates):
            executed_query = candidate
            executed_source = source
            executed_candidate_index = idx
            proof_outcome = self._reasoner.query_polarity(
                candidate,
                seed_terms=seed_terms,
            )
            if first_outcome is None:
                first_outcome = proof_outcome
            proof_traces = proof_outcome.proof
            if proof_outcome.status != "unknown":
                break

        if not proof_traces and hasattr(self._parser, "retry_parse_query"):
            try:
                retry = getattr(self._parser, "retry_parse_query")
                retry_result = retry(question, context, executed_query)
                if retry_result and retry_result.queries:
                    retry_used = True
                    validated_retry = [
                        candidate
                        for candidate, _source in self._query_candidates(
                            question,
                            [],
                            retry_result.queries,
                        )
                    ]
                    more = (
                        validated_retry
                        if self._query_fallback_enabled
                        else validated_retry[:1]
                    )
                    candidate_count_total += len(more)
                    if self._query_fallback_enabled and max_tries > 0:
                        remaining = max_tries - candidate_count_tried
                        more = more[: max(remaining, 0)]
                    candidate_count_tried += len(more)
                    for idx, candidate in enumerate(
                        more, start=(executed_candidate_index or 0) + 1
                    ):
                        executed_query = candidate
                        executed_source = "parser"
                        executed_candidate_index = idx
                        proof_outcome = self._reasoner.query_polarity(
                            candidate,
                            seed_terms=seed_terms,
                        )
                        proof_traces = proof_outcome.proof
                        if proof_outcome.status != "unknown":
                            break
            except Exception as exc:
                print(f"[Service] retry_parse_query failed: {exc}")

        if not proof_traces and candidates:
            executed_query, executed_source = candidates[0]
            executed_candidate_index = 0
            proof_outcome = first_outcome or ProofOutcome(
                positive_query=executed_query,
                negative_query=self._reasoner.negated_query(executed_query),
            )

        reasoning_seconds = time.perf_counter() - t2

        raw_proof = str(
            {
                "positive": proof_outcome.positive_proof,
                "negative": proof_outcome.negative_proof,
            }
        )
        fallback_used = bool(
            executed_source == "parser"
            and executed_query
            and original_query
            and executed_query != original_query
        )
        query_status = self._classify_query_status(
            question,
            executed_query or original_query,
            fallback_used,
        )

        # 5. Reverse-lookup NL sources from proof atoms
        t3 = time.perf_counter()
        sources = self._extract_sources(
            proof_traces, max_atoms=cfg.source_lookup_max_atoms
        )
        source_lookup_seconds = time.perf_counter() - t3

        # 6. Generate natural language answer
        t4 = time.perf_counter()
        requirements: List[dict] = []
        if intent.mode == QuestionMode.EXPLANATION:
            requirements = self._reasoner.explain_requirements(
                set(intent.terms),
                entity=intent.entities[0] if intent.entities else "",
                direction=intent.direction,
            )
        if requirements:
            answer = self._answer_structured_intent(intent, requirements)
            answer_generation_seconds = time.perf_counter() - t4
        elif cfg.answer_generation_enabled:
            answer = self._answer_gen.generate_from_polarity(
                question,
                executed_query,
                proof_outcome.status,
                proof_outcome.positive_proof,
                proof_outcome.negative_proof,
            )
            answer_generation_seconds = time.perf_counter() - t4
        else:
            answer = ""
            answer_generation_seconds = 0.0
        if not proof_traces and query_status == "weakly_aligned":
            answer = (
                "No proof was found. The generated query is only weakly aligned with the current "
                "knowledge base, so the failure may come from query shape mismatch or missing witness facts."
            )

        return QueryResponse(
            question=question,
            pln_query=executed_query,
            original_query=original_query,
            executed_query=executed_query,
            query_source=executed_source,
            fallback_used=fallback_used,
            query_status=query_status,
            raw_proof=raw_proof,
            sources=sources,
            answer=answer,
            qdrant_aligned_queries=qdrant_queries,
            execution_candidates=[candidate for candidate, _ in candidates],
            candidate_count=candidate_count_total,
            candidate_count_tried=candidate_count_tried,
            executed_candidate_index=executed_candidate_index,
            retry_used=retry_used,
            context_retrieval_seconds=round(context_retrieval_seconds, 4),
            parse_query_seconds=round(parse_query_seconds, 4),
            reasoning_seconds=round(reasoning_seconds, 4),
            source_lookup_seconds=round(source_lookup_seconds, 4),
            answer_generation_seconds=round(answer_generation_seconds, 4),
            intent_mode=intent.mode.value,
            proof_status=proof_outcome.status,
            negative_query=proof_outcome.negative_query,
            positive_proof=proof_outcome.positive_proof,
            negative_proof=proof_outcome.negative_proof,
            requirements=requirements,
        )

    async def debug_query(self, question: str) -> DebugQueryResponse:
        cfg = get_settings()
        intent = parse_question_intent(question)
        if self._vector_store:
            context, qdrant_matches = self._qdrant_context_with_matches(
                question,
                top_k=max(
                    self._context_top_k,
                    int(getattr(cfg, "query_alignment_top_k", 0) or 0),
                ),
            )
        else:
            context, qdrant_matches = [], []
        alignment_matches = self._alignment_matches(qdrant_matches)
        qdrant_alignment = (
            build_aligned_queries(question, alignment_matches)
            if getattr(cfg, "query_alignment_enabled", True)
            else None
        )
        qdrant_queries = qdrant_alignment.queries if qdrant_alignment else []
        seed_terms = extract_forward_seed_terms(alignment_matches)
        context = self._enrich_context(context)

        debug_info = self._parser.debug_parse_query(question, context)
        langextract_info = debug_info.get("langextract_postprocessed", {})
        pln_candidates = debug_info.get("pln_canonicalized", [])
        supporting_statements = debug_info.get("supporting_statements", [])

        original_query = pln_candidates[0] if pln_candidates else (
            qdrant_queries[0] if qdrant_queries else ""
        )
        candidate_entries = self._query_candidates(
            question,
            qdrant_queries,
            pln_candidates,
        )
        if intent.mode in {QuestionMode.FACTORS, QuestionMode.SUFFICIENCY}:
            requirements = self._reasoner.explain_requirements(
                set(intent.terms),
                entity=intent.entities[0] if intent.entities else "",
                direction=intent.direction,
            )
            return DebugQueryResponse(
                question=question,
                context=context,
                qdrant_matches=qdrant_matches,
                qdrant_aligned_queries=qdrant_queries,
                execution_candidates=[],
                langextract_postprocessed=LangExtractQueryPostprocessed(
                    **langextract_info
                ),
                pln_canonicalized_queries=pln_candidates,
                supporting_statements=supporting_statements,
                executed_query="",
                query_source="none",
                fallback_used=False,
                query_status="well_aligned" if requirements else "weakly_aligned",
                proof="",
                sources=[],
                answer=self._answer_structured_intent(intent, requirements),
                intent_mode=intent.mode.value,
                proof_status="unanswered",
                requirements=requirements,
            )
        if not candidate_entries:
            return DebugQueryResponse(
                question=question,
                context=context,
                qdrant_matches=qdrant_matches,
                qdrant_aligned_queries=qdrant_queries,
                execution_candidates=[],
                langextract_postprocessed=LangExtractQueryPostprocessed(
                    **langextract_info
                ),
                pln_canonicalized_queries=[],
                supporting_statements=supporting_statements,
                executed_query="",
                query_source="none",
                fallback_used=False,
                query_status="no_query",
                proof="",
                sources=[],
                answer="I couldn't translate this question into a logical query.",
                intent_mode=intent.mode.value,
                proof_status="unanswered",
            )

        candidates = (
            candidate_entries
            if self._query_fallback_enabled
            else candidate_entries[:1]
        )
        max_tries = int(getattr(cfg, "query_candidate_max_tries", 0) or 0)
        if self._query_fallback_enabled and max_tries > 0:
            candidates = candidates[:max_tries]

        executed_query = ""
        executed_source = "none"
        proof_traces: List[str] = []
        proof_outcome = ProofOutcome()
        first_outcome: ProofOutcome | None = None
        for candidate, source in candidates:
            executed_query = candidate
            executed_source = source
            proof_outcome = self._reasoner.query_polarity(
                candidate,
                seed_terms=seed_terms,
            )
            if first_outcome is None:
                first_outcome = proof_outcome
            proof_traces = proof_outcome.proof
            if proof_outcome.status != "unknown":
                break

        if not proof_traces and candidates:
            executed_query, executed_source = candidates[0]
            proof_outcome = first_outcome or ProofOutcome(
                positive_query=executed_query,
                negative_query=self._reasoner.negated_query(executed_query),
            )

        fallback_used = bool(
            executed_source == "parser"
            and executed_query
            and original_query
            and executed_query != original_query
        )
        query_status = self._classify_query_status(
            question,
            executed_query or original_query,
            fallback_used,
        )
        sources = self._extract_sources(
            proof_traces, max_atoms=cfg.source_lookup_max_atoms
        )
        requirements: List[dict] = []
        if intent.mode == QuestionMode.EXPLANATION:
            requirements = self._reasoner.explain_requirements(
                set(intent.terms),
                entity=intent.entities[0] if intent.entities else "",
                direction=intent.direction,
            )
        if requirements:
            answer = self._answer_structured_intent(intent, requirements)
        elif cfg.answer_generation_enabled:
            answer = self._answer_gen.generate_from_polarity(
                question,
                executed_query,
                proof_outcome.status,
                proof_outcome.positive_proof,
                proof_outcome.negative_proof,
            )
        else:
            answer = ""
        if not proof_traces and query_status == "weakly_aligned":
            answer = (
                "No proof was found. The generated query is only weakly aligned with the current "
                "knowledge base, so the failure may come from query shape mismatch or missing witness facts."
            )

        return DebugQueryResponse(
            question=question,
            context=context,
            qdrant_matches=qdrant_matches,
            qdrant_aligned_queries=qdrant_queries,
            execution_candidates=[candidate for candidate, _ in candidates],
            langextract_postprocessed=LangExtractQueryPostprocessed(
                **langextract_info
            ),
            pln_canonicalized_queries=pln_candidates,
            supporting_statements=supporting_statements,
            executed_query=executed_query,
            query_source=executed_source,
            fallback_used=fallback_used,
            query_status=query_status,
            proof=str(
                {
                    "positive": proof_outcome.positive_proof,
                    "negative": proof_outcome.negative_proof,
                }
            ),
            sources=sources,
            answer=answer,
            intent_mode=intent.mode.value,
            proof_status=proof_outcome.status,
            negative_query=proof_outcome.negative_query,
            positive_proof=proof_outcome.positive_proof,
            negative_proof=proof_outcome.negative_proof,
            requirements=requirements,
        )

    def _answer_structured_intent(
        self,
        intent: QuestionIntent,
        explanations: List[dict],
    ) -> str:
        if intent.mode == QuestionMode.SUFFICIENCY:
            if not explanations:
                return (
                    "The knowledge base does not establish that the stated cause alone is "
                    "sufficient. A related fact is not a proof of causal sufficiency."
                )
            atoms = self._requirement_atoms(explanations)
            additional = [atom for atom, status in atoms if status != "positive"]
            if additional:
                return (
                    "No proof establishes sufficiency from that condition alone. "
                    "The relevant rules also require: " + ", ".join(additional[:8]) + "."
                )
            return (
                "The relevant rule requirements are proved, but this planner does not treat "
                "that as proof that the named condition alone is sufficient."
            )

        if not explanations:
            return "No proof-backed factors matching this question were found in the rules."
        atoms = self._requirement_atoms(explanations)
        if not atoms:
            return "Relevant rules were found, but they contain no inspectable factor premises."
        rendered = ", ".join(f"{atom} [{status}]" for atom, status in atoms[:12])
        if intent.mode == QuestionMode.EXPLANATION:
            return "The proof rules depend on these requirements: " + rendered + "."
        return "The relevant rules identify these requirements: " + rendered + "."

    def _requirement_atoms(self, explanations: List[dict]) -> List[tuple[str, str]]:
        atoms: List[tuple[str, str]] = []
        seen: set[str] = set()

        def visit(item: dict) -> None:
            atom = str(item.get("atom", ""))
            if atom and atom not in seen:
                seen.add(atom)
                atoms.append((atom, str(item.get("status", "unknown"))))
            for key in ("children", "options"):
                for child in item.get(key, []) or []:
                    if isinstance(child, dict):
                        visit(child)

        for explanation in explanations:
            for requirement in explanation.get("requirements", []) or []:
                if isinstance(requirement, dict):
                    visit(requirement)
        return atoms

    def _classify_query_status(
        self, question: str, original_query: str, fallback_used: bool
    ) -> str:
        if not original_query:
            return "no_query"
        if fallback_used:
            return "weakly_aligned"

        normalized = question.strip().lower()
        is_yes_no = normalized.startswith(
            (
                "is ",
                "are ",
                "was ",
                "were ",
                "does ",
                "do ",
                "did ",
                "can ",
                "could ",
                "has ",
                "have ",
                "had ",
            )
        )
        has_variables = self._query_has_goal_variables(original_query)
        if is_yes_no and has_variables:
            return "weakly_aligned"
        return "well_aligned"

    def _query_has_goal_variables(self, query: str) -> bool:
        variables = set(re.findall(r"[$?][A-Za-z_][A-Za-z0-9_]*", query))
        return bool(variables - {"$prf", "$tv", "?prf", "?tv"})

    def _extract_sources(self, proof_traces: List[str], max_atoms: int = 30) -> List[str]:
        """
        Extract atom names from proof traces and reverse-lookup
        their NL source sentences from the vector store.
        """
        exact_sources = self._reasoner.sources_for_proof(proof_traces)
        if exact_sources:
            return exact_sources
        if max_atoms <= 0:
            return []

        atoms_to_search = set()
        if not self._vector_store:
            return []
        for trace in proof_traces:
            for match in re.findall(r"\([^()]+?\)", str(trace)):
                if "STV" not in match and len(match) >= 5:
                    atoms_to_search.add(match)

        sources = set()
        if len(atoms_to_search) > max_atoms:
            atoms_to_search = set(list(atoms_to_search)[:max_atoms])

        for atom_str in atoms_to_search:
            try:
                vector = self._vector_store.embed(atom_str)
                ctx, _ = self._vector_store.retrieve_context(atom_str, top_k=1)
                # retrieve_context returns atoms, not NL — direct search needed
                resp = self._vector_store._client.post(
                    f"{self._vector_store._qdrant}/collections"
                    f"/{self._vector_store._collection}/points/search",
                    json={"vector": vector, "limit": 1, "with_payload": True},
                )
                results = resp.json().get("result", [])
                if results and results[0].get("score", 0) > 0.6:
                    nl = results[0].get("payload", {}).get("nl")
                    if nl:
                        sources.add(nl)
            except Exception:
                pass

        return list(sources)

    #  Reset

    def reset(self, scope: str):
        if scope in ("all", "atomspace"):
            self._reasoner.reset()
            self._parser.reset(clear_registry=scope == "all")
        if scope in ("all", "vectordb") and self._vector_store:
            self._vector_store.reset()

    #  Health

    def health(self) -> dict:
        return {
            "atomspace_size": self._reasoner.size,
            "vectordb_count": self._vector_store.count if self._vector_store else 0,
            "parser": self._parser.__class__.__name__,
        }

    def debug_qdrant(self, limit: int = 50) -> dict:
        if not self._vector_store:
            return {
                "enabled": False,
                "count": 0,
                "points": [],
                "predicate_count": 0,
                "predicate_points": [],
            }
        capped_limit = max(1, min(limit, 200))
        return {
            "enabled": True,
            "count": self._vector_store.count,
            "points": self._vector_store.list_points(limit=capped_limit),
            "predicate_count": self._vector_store.predicate_count,
            "predicate_points": self._vector_store.list_predicate_points(
                limit=capped_limit
            ),
        }
