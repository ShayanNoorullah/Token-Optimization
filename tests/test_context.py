import uuid
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from app.config import get_settings
from app.schemas import ChatResponse, ContextBreakdown, UserFactItem
from app.services.context_builder import ContextBuilder
from app.services.embedding import SearchResult
from app.utils.tokens import count_tokens


def _sample_chat_response() -> ChatResponse:
    return ChatResponse(
        message_id=uuid.uuid4(),
        response="Mock response",
        context_tokens_used=500,
        naive_baseline_tokens=4000,
        savings_percent=87.5,
        retrieved_memories=[],
        response_tokens=80,
        context_breakdown=ContextBreakdown(
            user_facts_tokens=50,
            session_summary_tokens=100,
            retrieved_memories_tokens=150,
            short_term_tokens=120,
            query_tokens=30,
            overhead_tokens=50,
            total_context_tokens=500,
            token_budget=4096,
        ),
        short_term_message_count=4,
        has_session_summary=False,
        session_summary_preview=None,
        user_facts=[
            UserFactItem(fact_key="preferred_language", fact_value="Python", confidence=0.9),
        ],
        retrieval_threshold=0.72,
        latency_ms=123.45,
        assembled_context_preview="## User Profile\nNo known user facts.",
    )


def test_count_tokens():
    assert count_tokens("hello world") > 0


def test_context_builder_token_budget():
    builder = ContextBuilder()
    context = builder.build(
        user_facts=[],
        session_summary=None,
        retrieved_memories=[],
        short_term_messages=[],
        current_user_message="What is Qdrant?",
    )
    assert context.total_tokens < get_settings().context_token_budget
    assert "User Profile" in context.full_system_message


def test_context_builder_truncates_memories():
    builder = ContextBuilder()
    memories = [
        SearchResult(
            chunk_id=uuid.uuid4(),
            content="x" * 5000,
            score=0.9,
            memory_type="turn_pair",
            session_id=None,
            importance=0.5,
            timestamp=0,
        )
    ]
    context = builder.build(
        user_facts=[],
        session_summary=None,
        retrieved_memories=memories,
        short_term_messages=[],
        current_user_message="test",
    )
    assert count_tokens(context.retrieved_memories_text) <= get_settings().retrieved_memories_tokens + 50


@pytest.mark.asyncio
async def test_health_endpoint(client):
    response = await client.get("/health")
    assert response.status_code == 200
    assert response.json()["status"] == "ok"


@pytest.mark.asyncio
@patch("app.api.routes.chat.orchestrator.handle_message", new_callable=AsyncMock)
@patch("app.api.routes.chat.orchestrator.post_turn_processing", new_callable=AsyncMock)
async def test_chat_endpoint_mock(mock_post_turn, mock_handle, client):
    mock_handle.return_value = _sample_chat_response()

    user_id = uuid.uuid4()
    session_id = uuid.uuid4()

    mock_session = MagicMock()
    mock_session.user_id = user_id

    async def override_get_db():
        db = AsyncMock()
        db.get = AsyncMock(return_value=mock_session)
        db.commit = AsyncMock()
        db.rollback = AsyncMock()
        yield db

    from app.database import get_db
    from app.main import app

    app.dependency_overrides[get_db] = override_get_db
    try:
        response = await client.post(
            "/chat",
            json={
                "user_id": str(user_id),
                "session_id": str(session_id),
                "message": "Hello",
            },
        )
        assert response.status_code == 200
        data = response.json()
        assert data["response"] == "Mock response"
        assert "context_breakdown" in data
        assert data["response_tokens"] == 80
        assert data["latency_ms"] == 123.45
        assert len(data["user_facts"]) == 1
    finally:
        app.dependency_overrides.clear()
