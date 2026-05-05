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
    fallback_used: bool
    query_status: Literal["well_aligned", "weakly_aligned", "malformed", "no_query"]
    raw_proof: str
    sources: List[str]       # NL sentences that contributed to the proof
    answer: str


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
    langextract_postprocessed: LangExtractQueryPostprocessed
    pln_canonicalized_queries: List[str] = Field(default_factory=list)
    supporting_statements: List[str] = Field(default_factory=list)
    executed_query: str
    fallback_used: bool
    query_status: Literal["well_aligned", "weakly_aligned", "malformed", "no_query"]
    proof: str
    sources: List[str]
    answer: str
