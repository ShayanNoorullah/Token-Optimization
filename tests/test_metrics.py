import uuid
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from app.services.context_builder import ContextBuilder
from app.services.llm import LLMService
from app.services.summarizer import SummarizerService
from app.utils.tokens import count_tokens


@pytest.mark.asyncio
async def test_llm_mock_summarize():
    from app.config import get_settings

    settings = get_settings()
    settings.llm_mock = True
    llm = LLMService(settings)
    summary = await llm.summarize("user: hello\nassistant: hi there")
    assert "Discussed" in summary or len(summary) > 0


@pytest.mark.asyncio
async def test_llm_mock_extract_facts():
    from app.config import get_settings

    settings = get_settings()
    settings.llm_mock = True
    llm = LLMService(settings)
    facts = await llm.extract_facts("I prefer Python", "Great choice!")
    assert "preferred_language" in facts


def test_token_savings_calculation():
    context_tokens = 900
    naive_baseline = 5400
    savings = round((1 - context_tokens / naive_baseline) * 100, 2)
    assert savings == 83.33


@pytest.mark.asyncio
@patch("app.api.routes.metrics.HybridRetriever")
async def test_evaluate_endpoint(mock_retriever_cls, client):
    from app.database import get_db
    from app.main import app
    from app.services.embedding import SearchResult

    mock_retriever = MagicMock()
    mock_retriever.retrieve = AsyncMock(
        return_value=[
            SearchResult(
                chunk_id=uuid.uuid4(),
                content="Qdrant setup instructions",
                score=0.85,
                memory_type="turn_pair",
                session_id=None,
                importance=0.7,
                timestamp=0,
            )
        ]
    )
    mock_retriever_cls.return_value = mock_retriever

    async def override_get_db():
        db = AsyncMock()
        db.commit = AsyncMock()
        db.rollback = AsyncMock()
        yield db

    app.dependency_overrides[get_db] = override_get_db
    try:
        response = await client.post(
            "/metrics/evaluate",
            params={"user_id": str(uuid.uuid4())},
            json=["How do I set up Qdrant?"],
        )
        assert response.status_code == 200
        data = response.json()
        assert len(data) == 1
        assert data[0]["passed_threshold"] is True
    finally:
        app.dependency_overrides.clear()
