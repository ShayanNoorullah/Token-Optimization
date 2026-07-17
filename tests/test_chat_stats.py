import uuid

import pytest

from app.schemas import ContextBreakdown
from app.services.chat import ChatOrchestrator
from app.services.context_builder import ContextBuilder
from app.utils.tokens import count_tokens


def test_context_breakdown_fields():
    breakdown = ContextBreakdown(
        user_facts_tokens=50,
        session_summary_tokens=100,
        retrieved_memories_tokens=150,
        short_term_tokens=120,
        query_tokens=30,
        overhead_tokens=50,
        total_context_tokens=500,
        token_budget=4096,
    )
    slot_sum = (
        breakdown.user_facts_tokens
        + breakdown.session_summary_tokens
        + breakdown.retrieved_memories_tokens
        + breakdown.short_term_tokens
        + breakdown.query_tokens
        + breakdown.overhead_tokens
    )
    assert slot_sum == breakdown.total_context_tokens


def test_build_context_breakdown():
    orchestrator = ChatOrchestrator()
    builder = ContextBuilder()
    context = builder.build(
        user_facts=[],
        session_summary=None,
        retrieved_memories=[],
        short_term_messages=[],
        current_user_message="What is Qdrant?",
    )
    breakdown = orchestrator._build_context_breakdown(context, "What is Qdrant?")

    assert breakdown.query_tokens == count_tokens("What is Qdrant?")
    assert breakdown.total_context_tokens == context.total_tokens
    assert breakdown.token_budget > 0
    assert breakdown.user_facts_tokens >= 0
    assert breakdown.overhead_tokens >= 0


def test_chat_response_schema_has_all_frontend_fields():
    """Verify ChatResponse includes every field the frontend expects."""
    from app.schemas import ChatResponse, RetrievedMemory, UserFactItem

    required_fields = {
        "message_id", "response", "context_tokens_used", "naive_baseline_tokens",
        "savings_percent", "retrieved_memories", "response_tokens",
        "context_breakdown", "short_term_message_count", "has_session_summary",
        "session_summary_preview", "user_facts", "retrieval_threshold", "latency_ms",
        "assembled_context_preview",
    }
    schema_fields = set(ChatResponse.model_fields.keys())
    assert required_fields.issubset(schema_fields)

    breakdown_fields = {
        "user_facts_tokens", "session_summary_tokens", "retrieved_memories_tokens",
        "short_term_tokens", "query_tokens", "overhead_tokens",
        "total_context_tokens", "token_budget",
    }
    from app.schemas import ContextBreakdown
    assert breakdown_fields.issubset(set(ContextBreakdown.model_fields.keys()))


@pytest.mark.asyncio
async def test_sessions_messages_endpoint(client):
    from app.database import get_db
    from app.main import app

    session_id = uuid.uuid4()

    async def override_get_db():
        db = AsyncMock_mock_session_messages()
        yield db

    app.dependency_overrides[get_db] = override_get_db
    try:
        response = await client.get(f"/sessions/{session_id}/messages")
        assert response.status_code == 200
        assert isinstance(response.json(), list)
    finally:
        app.dependency_overrides.clear()


def AsyncMock_mock_session_messages():
    from unittest.mock import AsyncMock, MagicMock

    db = AsyncMock()
    mock_session = MagicMock()
    db.get = AsyncMock(return_value=mock_session)

    mock_result = MagicMock()
    mock_result.scalars.return_value.all.return_value = []
    db.execute = AsyncMock(return_value=mock_result)
    db.commit = AsyncMock()
    db.rollback = AsyncMock()
    return db
