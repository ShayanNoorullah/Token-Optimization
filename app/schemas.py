import uuid
from datetime import datetime

from pydantic import BaseModel, Field


class UserCreate(BaseModel):
    email: str | None = None


class UserResponse(BaseModel):
    id: uuid.UUID
    email: str | None
    created_at: datetime

    model_config = {"from_attributes": True}


class SessionCreate(BaseModel):
    user_id: uuid.UUID
    title: str | None = None


class SessionResponse(BaseModel):
    id: uuid.UUID
    user_id: uuid.UUID
    title: str | None
    status: str
    created_at: datetime

    model_config = {"from_attributes": True}


class MessageResponse(BaseModel):
    id: uuid.UUID
    session_id: uuid.UUID
    role: str
    content: str
    token_count: int | None
    turn_index: int
    is_summarized: bool
    created_at: datetime

    model_config = {"from_attributes": True}


class UserFactItem(BaseModel):
    fact_key: str
    fact_value: str
    confidence: float

    model_config = {"from_attributes": True}


class SessionSummaryResponse(BaseModel):
    id: uuid.UUID
    session_id: uuid.UUID
    summary_text: str
    covers_from: int | None
    covers_to: int | None
    token_count: int | None
    created_at: datetime

    model_config = {"from_attributes": True}


class ChatRequest(BaseModel):
    user_id: uuid.UUID
    session_id: uuid.UUID
    message: str = Field(min_length=1)


class RetrievedMemory(BaseModel):
    chunk_id: uuid.UUID
    content: str
    score: float
    memory_type: str


class ContextBreakdown(BaseModel):
    user_facts_tokens: int
    session_summary_tokens: int
    retrieved_memories_tokens: int
    short_term_tokens: int
    query_tokens: int
    overhead_tokens: int
    total_context_tokens: int
    token_budget: int


class ChatResponse(BaseModel):
    message_id: uuid.UUID
    response: str
    context_tokens_used: int
    naive_baseline_tokens: int
    savings_percent: float
    retrieved_memories: list[RetrievedMemory]
    response_tokens: int
    context_breakdown: ContextBreakdown
    short_term_message_count: int
    has_session_summary: bool
    session_summary_preview: str | None
    user_facts: list[UserFactItem]
    retrieval_threshold: float
    latency_ms: float
    assembled_context_preview: str | None = None


class MetricsSummary(BaseModel):
    total_requests: int
    avg_context_tokens: float
    avg_naive_baseline_tokens: float
    avg_savings_percent: float
    avg_retrieval_count: float


class EvaluationResult(BaseModel):
    query: str
    retrieved_count: int
    context_tokens: int
    passed_threshold: bool
