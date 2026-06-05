from pydantic import BaseModel, Field
from typing import Any, Dict, List, Optional, Literal


#  Ingest 

class IngestRequest(BaseModel):
    texts: List[str]


class IngestItemResult(BaseModel):
    text: str
    atoms: List[str] = []
    status: Literal["success", "failed"]
    error: Optional[str] = None
    chunk_count: int = 0
    batch_count: int = 0
    batch_sizes: List[int] = []
    parser_calls: int = 0
    rejected_count: int = 0
    rejected_samples: List[str] = []


class IngestResponse(BaseModel):
    processed_count: int
    results: List[IngestItemResult]


#  Query 

class QueryRequest(BaseModel):
    question: str


class QueryResponse(BaseModel):
    question: str
    pln_query: str
    original_query: str
    executed_query: str
    query_source: Literal["qdrant_alignment", "parser", "none"] = "none"
    fallback_used: bool
    query_status: Literal["well_aligned", "weakly_aligned", "malformed", "no_query"]
    raw_proof: str
    sources: List[str]       # NL sentences that contributed to the proof
    answer: str
    qdrant_aligned_queries: List[str] = Field(default_factory=list)
    execution_candidates: List[str] = Field(default_factory=list)
    candidate_count: Optional[int] = None
    candidate_count_tried: Optional[int] = None
    executed_candidate_index: Optional[int] = None
    retry_used: Optional[bool] = None
    context_retrieval_seconds: Optional[float] = None
    parse_query_seconds: Optional[float] = None
    reasoning_seconds: Optional[float] = None
    source_lookup_seconds: Optional[float] = None
    answer_generation_seconds: Optional[float] = None


#  Reset 

class ResetRequest(BaseModel):
    scope: Literal["all", "vectordb", "atomspace"] = "all"


class ResetResponse(BaseModel):
    status: Literal["ok"]
    scope: str


#  Health 

class HealthResponse(BaseModel):
    status: Literal["ok", "degraded"]
    parser: str
    atomspace_size: int
    vectordb_count: int
    uptime_seconds: float


#  Debug

class LangExtractPostprocessed(BaseModel):
    statements: List[str] = Field(default_factory=list)
    rejected: List[Dict[str, Any]] = Field(default_factory=list)
    canonicalization_context: Dict[str, Any] = Field(default_factory=dict)
    statement_sources: Dict[str, Dict[str, Any]] = Field(default_factory=dict)


class LangExtractQueryPostprocessed(BaseModel):
    queries: List[str] = Field(default_factory=list)
    rejected: List[Dict[str, Any]] = Field(default_factory=list)
    canonicalization_context: Dict[str, Any] = Field(default_factory=dict)
    query_sources: Dict[str, Dict[str, Any]] = Field(default_factory=dict)


class DebugIngestRequest(BaseModel):
    texts: List[str]


class DebugIngestChunkResult(BaseModel):
    chunk: str
    context: List[str]
    langextract_postprocessed: LangExtractPostprocessed
    pln_canonicalized: List[str] = Field(default_factory=list)
    atomspace_added: List[str] = Field(default_factory=list)
    schema_alignment: List[Dict[str, Any]] = Field(default_factory=list)


class DebugIngestItemResult(BaseModel):
    text: str
    chunks: List[DebugIngestChunkResult] = Field(default_factory=list)
    status: Literal["success", "failed"]
    error: Optional[str] = None


class DebugIngestResponse(BaseModel):
    processed_count: int
    results: List[DebugIngestItemResult]


class DebugQueryRequest(BaseModel):
    question: str


class DebugQueryResponse(BaseModel):
    question: str
    context: List[str] = Field(default_factory=list)
    qdrant_matches: List[Dict[str, Any]] = Field(default_factory=list)
    qdrant_aligned_queries: List[str] = Field(default_factory=list)
    execution_candidates: List[str] = Field(default_factory=list)
    langextract_postprocessed: LangExtractQueryPostprocessed
    pln_canonicalized_queries: List[str] = Field(default_factory=list)
    supporting_statements: List[str] = Field(default_factory=list)
    executed_query: str
    query_source: Literal["qdrant_alignment", "parser", "none"] = "none"
    fallback_used: bool
    query_status: Literal["well_aligned", "weakly_aligned", "malformed", "no_query"]
    proof: str
    sources: List[str]
    answer: str


class DebugQdrantPoint(BaseModel):
    id: str | int
    payload: Dict[str, Any] = Field(default_factory=dict)


class DebugQdrantResponse(BaseModel):
    enabled: bool
    count: int
    points: List[DebugQdrantPoint] = Field(default_factory=list)
