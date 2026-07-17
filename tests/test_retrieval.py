import math
import uuid

import pytest

from app.services.embedding import SearchResult
from app.services.retrieval import HybridRetriever


class FakeEmbeddingService:
    settings = None

    def _embed_sync(self, texts, is_query=False):
        return [[1.0, 0.0, 0.0] for _ in texts]

    async def embed(self, text, is_query=False):
        return [1.0, 0.0, 0.0]

    async def search(self, query, user_id, limit=None, session_id=None):
        return [
            SearchResult(
                chunk_id=uuid.uuid4(),
                content="Discussed Qdrant vector database setup",
                score=0.85,
                memory_type="turn_pair",
                session_id=str(session_id) if session_id else None,
                importance=0.7,
                timestamp=int(__import__("time").time()),
            ),
            SearchResult(
                chunk_id=uuid.uuid4(),
                content="Unrelated cooking recipe discussion",
                score=0.3,
                memory_type="turn_pair",
                session_id=None,
                importance=0.3,
                timestamp=0,
            ),
        ]


def test_temporal_decay():
    retriever = HybridRetriever()
    retriever.embedding = FakeEmbeddingService()

    candidates = [
        SearchResult(
            chunk_id=uuid.uuid4(),
            content="old",
            score=0.9,
            memory_type="turn_pair",
            session_id=None,
            importance=0.5,
            timestamp=0,
        )
    ]
    decayed = retriever._apply_temporal_decay(candidates)
    assert decayed[0].score < 0.9


def test_cosine_similarity():
    retriever = HybridRetriever()
    sim = retriever._cosine_similarity([1.0, 0.0], [1.0, 0.0])
    assert math.isclose(sim, 1.0)


def test_deduplicate():
    retriever = HybridRetriever()
    retriever.embedding = FakeEmbeddingService()

    candidates = [
        SearchResult(
            chunk_id=uuid.uuid4(),
            content="same content",
            score=0.9,
            memory_type="turn_pair",
            session_id=None,
            importance=0.5,
            timestamp=0,
        ),
        SearchResult(
            chunk_id=uuid.uuid4(),
            content="same content",
            score=0.8,
            memory_type="turn_pair",
            session_id=None,
            importance=0.5,
            timestamp=0,
        ),
    ]
    deduped = retriever._deduplicate(candidates)
    assert len(deduped) == 1


@pytest.mark.asyncio
async def test_mmr_select():
    retriever = HybridRetriever()
    retriever.embedding = FakeEmbeddingService()

    candidates = [
        SearchResult(
            chunk_id=uuid.uuid4(),
            content=f"topic {i}",
            score=0.9 - i * 0.1,
            memory_type="turn_pair",
            session_id=None,
            importance=0.5,
            timestamp=0,
        )
        for i in range(5)
    ]
    selected = retriever._mmr_select(candidates, [1.0, 0.0, 0.0], top_k=3)
    assert len(selected) == 3
