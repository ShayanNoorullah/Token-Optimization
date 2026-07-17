"""Tests for ARIA log replay prompts and summarization triggers."""

import uuid
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from app.config import get_settings
from app.services.summarizer import SummarizerService

ARIA_PROMPT_SAMPLES = [
    "Give me a dashboard briefing on the current pipeline.",
    "What is the biggest lever to move next month's number?",
    "Rank all branches and explain the bottom two performers.",
]


def test_summarize_turn_threshold_configured():
    settings = get_settings()
    assert settings.summarize_turn_threshold >= 1


@pytest.mark.asyncio
async def test_summarize_triggers_on_turn_count():
    summarizer = SummarizerService()

    messages = []
    for i in range(12):
        msg = MagicMock()
        msg.turn_index = i
        msg.content = f"message {i}"
        msg.token_count = 50
        msg.is_summarized = False
        messages.append(msg)

    mock_result = MagicMock()
    mock_result.scalars.return_value.all.return_value = messages

    db = AsyncMock()
    db.execute = AsyncMock(return_value=mock_result)
    db.add = MagicMock()
    db.flush = AsyncMock()

    summarizer.llm.summarize = AsyncMock(return_value="Summary of conversation")
    summarizer.llm.merge_summaries = AsyncMock(return_value="Summary of conversation")
    summarizer.embedding.upsert_memory = AsyncMock(return_value="qdrant-id")

    result = await summarizer.maybe_summarize(db, uuid.uuid4(), uuid.uuid4())
    assert result is not None


def test_aria_prompts_are_non_empty():
    assert all(len(p) > 10 for p in ARIA_PROMPT_SAMPLES)


@pytest.mark.asyncio
@patch("app.api.routes.chat.orchestrator.handle_message", new_callable=AsyncMock)
async def test_chat_with_long_analytical_prompt(mock_handle, client):
    from app.database import get_db
    from app.main import app
    from tests.test_context import _sample_chat_response

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

    app.dependency_overrides[get_db] = override_get_db
    try:
        for prompt in ARIA_PROMPT_SAMPLES:
            response = await client.post(
                "/chat",
                json={"user_id": str(user_id), "session_id": str(session_id), "message": prompt},
            )
            assert response.status_code == 200
            assert "assembled_context_preview" in response.json()
    finally:
        app.dependency_overrides.clear()
