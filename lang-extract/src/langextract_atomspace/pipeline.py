from __future__ import annotations

import os
import time
from concurrent.futures import ThreadPoolExecutor, TimeoutError as FutureTimeout, as_completed, wait
from typing import Any, Iterator

import langextract as lx
from hyperon import GroundingSpaceRef, MeTTa

from .mapping.examples import METTA_EXAMPLES, EXTRACTION_PROMPT
from .mapping.translator import collect_predicate_heads, populate_space
from .core.env import _load_repo_env
from .processing.scraper import extract_text_from_url, _URL_SAFE_MAX_WORKERS
from .processing.chunker import _split_text
from .core.validation import validate_metta_atoms


def run_extraction(
    text: str,
    model_id: str | None = "gemini-2.5-flash",
    model_url: str | None = None,
    api_key: str | None = None,
    extra_examples: list | None = None,
    extraction_passes: int = 1,
    max_char_buffer: int = 1000,
    max_workers: int = 10,
    skip_fuzzy: bool = True,
    debug: bool = False,
) -> dict:
    """
    Run the full extraction pipeline and return everything the UI needs:

        {
          "result":      AnnotatedDocument
          "extractions": list[Extraction],
          "space":       GroundingSpaceRef,
          "atoms":       list[Atom],
          "metta_str":   str,
          "metta":       MeTTa runner with all atoms already loaded,
        }

    All extraction parameters (passes, chunk size, workers) are exposed so
    the UI can tune them for large documents.
    """
    _load_repo_env()
    model_id = model_id or os.environ.get("LANGEXTRACT_MODEL_ID", "gemini-2.5-flash")
    examples = METTA_EXAMPLES + (extra_examples or [])

    result = lx.extract(
        text_or_documents=text,
        prompt_description=EXTRACTION_PROMPT,
        examples=examples,
        model_id=model_id,
        model_url=model_url,
        api_key=api_key,
        extraction_passes=extraction_passes,
        max_char_buffer=max_char_buffer,
        max_workers=max_workers,
        debug=debug,
        show_progress=False,
    )

    extractions = result.extractions if hasattr(result, "extractions") else []
    space, meta = populate_space(extractions, skip_fuzzy=skip_fuzzy, source_text=text)
    atoms = space.get_atoms()

    metta, validation = validate_metta_atoms(atoms)

    return {
        "result": result,
        "extractions": extractions,
        "space": space,
        "atoms": atoms,
        "metta_str": "\n".join(str(a) for a in atoms),
        "metta": metta,
        "validation": validation,
        "atom_to_source": meta["atom_to_source"],
        "rejected": meta["rejected"],
        "source": {
            "kind": "text",
            "text": text,
        },
    }


def run_extraction_from_url(
    url: str,
    model_id: str | None = "gemini-2.5-flash",
    model_url: str | None = None,
    api_key: str | None = None,
    extra_examples: list | None = None,
    extraction_passes: int = 1,
    max_char_buffer: int = 1000,
    max_workers: int = 10,
    skip_fuzzy: bool = True,
    debug: bool = False,
) -> dict:
    """
    Fetch a public webpage, extract readable text, and run the standard
    LangExtract -> MeTTa pipeline on the resulting text.
    """
    source = extract_text_from_url(url)
    effective_workers = _URL_SAFE_MAX_WORKERS
    result = run_extraction(
        text=source["text"],
        model_id=model_id,
        model_url=model_url,
        api_key=api_key,
        extra_examples=extra_examples,
        extraction_passes=extraction_passes,
        max_char_buffer=max_char_buffer,
        max_workers=effective_workers,
        skip_fuzzy=skip_fuzzy,
        debug=debug,
    )
    result["source"] = {
        "kind": "url",
        "url": source["url"],
        "title": source["title"],
        "text": source["text"],
        "paragraph_count": source["paragraph_count"],
        "content_type": source["content_type"],
        "char_count": source["char_count"],
        "original_paragraph_count": source["original_paragraph_count"],
        "original_char_count": source["original_char_count"],
        "truncated": source["truncated"],
        "effective_workers": effective_workers,
    }
    return result


def text_to_atomspace(
    text: str,
    model_id: str | None = "gemini-2.5-flash",
    model_url: str | None = None,
    api_key: str | None = None,
    extra_examples: list | None = None,
    extraction_passes: int = 1,
    skip_fuzzy: bool = True,
    debug: bool = False,
) -> GroundingSpaceRef:
    """
    Parse natural language text and return a populated AtomSpace.

    Args:
        text: the natural language input
        model_id: LLM to use ('gemini-2.5-flash', 'gpt-4o', or an Ollama model name)
        model_url: base URL for Ollama or other self-hosted backends
        api_key: API key override. LangExtract also picks up environment keys
                 such as LANGEXTRACT_API_KEY and OPENAI_API_KEY.
        extra_examples: additional ExampleData objects appended to the built-in set
        extraction_passes: number of sequential LangExtract passes (use >1 for
                           complex texts where one pass misses relations)
        skip_fuzzy: drop extractions whose text doesn't appear verbatim in source
        debug: pass debug=True to LangExtract for verbose LLM output

    Returns:
        GroundingSpaceRef with extracted MeTTa atoms.
    """
    _load_repo_env()
    model_id = model_id or os.environ.get("LANGEXTRACT_MODEL_ID", "gemini-2.5-flash")
    examples = METTA_EXAMPLES + (extra_examples or [])

    result = lx.extract(
        text_or_documents=text,
        prompt_description=EXTRACTION_PROMPT,
        examples=examples,
        model_id=model_id,
        model_url=model_url,
        api_key=api_key,
        extraction_passes=extraction_passes,
        debug=debug,
        show_progress=False,
    )

    # lx.extract returns AnnotatedDocument when given a single string.
    extractions = result.extractions if hasattr(result, "extractions") else []
    space, _meta = populate_space(extractions, skip_fuzzy=skip_fuzzy, source_text=text)
    return space


def text_to_metta_str(text: str, **kwargs) -> str:
    """
    Parse natural language text and return a MeTTa program string.

    The returned string can be written to a .metta file or run directly with
    MeTTa().run(text_to_metta_str(...)).
    """
    space = text_to_atomspace(text, **kwargs)
    atoms = space.get_atoms()
    return "\n".join(str(a) for a in atoms)


def _format_predicate_hint(heads: list[str]) -> str:
    if not heads:
        return ""
    snippet = ", ".join(heads[:24])
    return (
        "\n\nReuse predicate vocabulary established earlier in this document. "
        f"Preferred predicate heads (do not invent new spellings for the same concepts): {snippet}."
    )


def _extract_one_chunk(
    chunk: str,
    model_id: str,
    model_url: str | None,
    api_key: str | None,
    examples: list,
    extraction_passes: int,
    debug: bool,
    prompt_suffix: str = "",
) -> list:
    """Run lx.extract on a single chunk and return the extractions list."""
    prompt = EXTRACTION_PROMPT + prompt_suffix if prompt_suffix else EXTRACTION_PROMPT
    result = lx.extract(
        text_or_documents=chunk,
        prompt_description=prompt,
        examples=examples,
        model_id=model_id,
        model_url=model_url,
        api_key=api_key,
        extraction_passes=extraction_passes,
        max_char_buffer=max(len(chunk) + 1, 1000),  # don't re-chunk inside
        max_workers=1,                               # outer pool parallelizes
        debug=debug,
        show_progress=False,
    )
    return list(result.extractions) if hasattr(result, "extractions") else []


def run_extraction_stream(
    *,
    text: str | None = None,
    url: str | None = None,
    model_id: str | None = "gemini-2.5-flash",
    model_url: str | None = None,
    api_key: str | None = None,
    extra_examples: list | None = None,
    extraction_passes: int = 1,
    max_char_buffer: int = 1500,
    max_workers: int = 10,
    skip_fuzzy: bool = True,
    debug: bool = False,
) -> Iterator[dict[str, Any]]:
    """
    Streaming version of run_extraction.

    Yields progress events so UIs can show live feedback:

      {"kind": "phase",      "name": "fetching" | "chunking" | "extracting" |
                                     "populating" | "validating"}
      {"kind": "fetched",    "char_count": int, "paragraph_count": int,
                             "url": str, "title": str}
      {"kind": "chunked",    "count": int}
      {"kind": "chunk_done", "done": int, "total": int, "extractions_so_far": int}
      {"kind": "done",       "result": <same dict run_extraction returns>}
      {"kind": "error",      "message": str}
    """
    _load_repo_env()
    model_id = model_id or os.environ.get("LANGEXTRACT_MODEL_ID", "gemini-2.5-flash")
    examples = METTA_EXAMPLES + (extra_examples or [])

    source_meta: dict[str, Any] | None = None
    effective_workers = max(1, max_workers)
    if url:
        yield {"kind": "phase", "name": "fetching"}
        source = extract_text_from_url(url)
        text = source["text"]
        effective_workers = _URL_SAFE_MAX_WORKERS
        source_meta = {
            "kind": "url",
            "url": source["url"],
            "title": source["title"],
            "text": source["text"],
            "paragraph_count": source["paragraph_count"],
            "content_type": source["content_type"],
            "char_count": source["char_count"],
            "original_paragraph_count": source["original_paragraph_count"],
            "original_char_count": source["original_char_count"],
            "truncated": source["truncated"],
            "effective_workers": effective_workers,
        }
        yield {
            "kind": "fetched",
            "char_count": source["char_count"],
            "paragraph_count": source["paragraph_count"],
            "url": source["url"],
            "title": source["title"],
            "truncated": source["truncated"],
            "original_char_count": source["original_char_count"],
            "original_paragraph_count": source["original_paragraph_count"],
        }

    if not text or not text.strip():
        yield {"kind": "error", "message": "No text to extract."}
        return

    yield {"kind": "phase", "name": "chunking"}
    chunks = _split_text(text, max_char_buffer)
    yield {"kind": "chunked", "count": len(chunks)}

    yield {"kind": "phase", "name": "extracting"}
    all_extractions: list = []
    t0 = time.monotonic()
    # Safety cap: if no chunk completes for this long, bail out. Typical
    # Gemini Flash responses are 3–15 s; 120 s means something is really stuck
    # (rate-limit backoff storm, dropped connection, etc.).
    PER_CHUNK_TIMEOUT_S = 120

    if effective_workers <= 1:
        # Sequential path: feed each chunk the predicate vocabulary collected
        # from previous chunks so the LLM reuses heads instead of inventing
        # new spellings for the same concepts.
        accumulated_heads: list[str] = []
        for idx, chunk in enumerate(chunks, start=1):
            suffix = _format_predicate_hint(accumulated_heads)
            chunk_extractions = _extract_one_chunk(
                chunk,
                model_id,
                model_url,
                api_key,
                examples,
                extraction_passes,
                debug,
                prompt_suffix=suffix,
            )
            all_extractions.extend(chunk_extractions)
            for head in collect_predicate_heads(chunk_extractions):
                if head not in accumulated_heads:
                    accumulated_heads.append(head)
            yield {
                "kind": "chunk_done",
                "done": idx,
                "total": len(chunks),
                "in_flight": 0,
                "extractions_so_far": len(all_extractions),
                "elapsed_s": time.monotonic() - t0,
            }
    else:
        with ThreadPoolExecutor(max_workers=effective_workers) as pool:
            futures = {
                pool.submit(
                    _extract_one_chunk,
                    chunk, model_id, model_url, api_key, examples,
                    extraction_passes, debug,
                ): i
                for i, chunk in enumerate(chunks)
            }
            pending = set(futures.keys())
            done = 0
            while pending:
                finished, pending = wait(
                    pending,
                    timeout=PER_CHUNK_TIMEOUT_S,
                    return_when="FIRST_COMPLETED",
                )
                if not finished:
                    for f in pending:
                        f.cancel()
                    yield {
                        "kind": "error",
                        "message": (
                            f"No chunk completed for {PER_CHUNK_TIMEOUT_S}s. "
                            "The LLM is likely rate-limited or unreachable. "
                            "Try lowering the Workers slider (free-tier Gemini "
                            "= 15 req/min) or check your API key."
                        ),
                    }
                    return
                for fut in finished:
                    try:
                        all_extractions.extend(fut.result())
                    except Exception as exc:
                        for f in pending:
                            f.cancel()
                        yield {"kind": "error", "message": f"Chunk failed: {exc}"}
                        return
                    done += 1
                    yield {
                        "kind": "chunk_done",
                        "done": done,
                        "total": len(chunks),
                        "in_flight": len(pending),
                        "extractions_so_far": len(all_extractions),
                        "elapsed_s": time.monotonic() - t0,
                    }

    yield {"kind": "phase", "name": "populating"}
    space, meta = populate_space(
        all_extractions, skip_fuzzy=skip_fuzzy, source_text=text or ""
    )
    atoms = space.get_atoms()

    yield {"kind": "phase", "name": "validating"}
    metta, validation = validate_metta_atoms(atoms)

    result_dict = {
        "result": None,  # full AnnotatedDocument not preserved in streaming mode
        "extractions": all_extractions,
        "space": space,
        "atoms": atoms,
        "metta_str": "\n".join(str(a) for a in atoms),
        "metta": metta,
        "validation": validation,
        "atom_to_source": meta["atom_to_source"],
        "rejected": meta["rejected"],
        "source": source_meta or {"kind": "text", "text": text},
    }
    yield {"kind": "done", "result": result_dict}


def text_to_metta_runner(text: str, **kwargs) -> MeTTa:
    """
    Parse natural language text and return a ready-to-query MeTTa runner
    with all extracted atoms already loaded into &self.

    Atoms are re-parsed inside the runner so grounded operations such as `and`
    keep the runner-local behavior required for executable rules.
    """
    space = text_to_atomspace(text, **kwargs)
    metta, validation = validate_metta_atoms(space.get_atoms())
    if not validation["load_ok"]:
        first_error = validation["errors"][0]
        raise ValueError(
            f"Generated MeTTa failed validation at atom {first_error['atom_index']}: "
            f"{first_error['message']}"
        )
    return metta


# Re-export for convenience so callers can import everything from .pipeline.
__all__ = [
    "extract_text_from_url",
    "run_extraction",
    "run_extraction_from_url",
    "run_extraction_stream",
    "text_to_atomspace",
    "text_to_metta_str",
    "text_to_metta_runner",
    "validate_metta_atoms",
]
