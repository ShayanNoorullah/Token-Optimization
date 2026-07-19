import logging
import math
import uuid
from datetime import datetime, timezone

import numpy as np
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.config import Settings, get_settings
from app.models.tables import MemoryChunk, Message
from app.services.embedding import EmbeddingService, SearchResult
from app.utils.tokens import run_sync

logger = logging.getLogger(__name__)


class HybridRetriever:
    def __init__(
        self,
        settings: Settings | None = None,
        embedding_service: EmbeddingService | None = None,
    ):
        self.settings = settings or get_settings()
        self.embedding = embedding_service or EmbeddingService(self.settings)
        self._reranker = None

    def _get_reranker(self):
        if self._reranker is None:
            from sentence_transformers import CrossEncoder

            logger.info("Loading reranker model: %s", self.settings.reranker_model)
            self._reranker = CrossEncoder(self.settings.reranker_model)
        return self._reranker

    def _rerank_sync(self, query: str, candidates: list[SearchResult]) -> list[SearchResult]:
        if not candidates:
            return []

        reranker = self._get_reranker()
        pairs = [[query, c.content] for c in candidates]
        # sentence-transformers CrossEncoder already maps BGE scores to ~[0, 1].
        scores = reranker.predict(pairs)

        reranked = []
        for candidate, score in zip(candidates, scores):
            updated = SearchResult(
                chunk_id=candidate.chunk_id,
                content=candidate.content,
                score=float(score),
                memory_type=candidate.memory_type,
                session_id=candidate.session_id,
                importance=candidate.importance,
                timestamp=candidate.timestamp,
            )
            reranked.append(updated)

        reranked.sort(key=lambda x: x.score, reverse=True)
        return reranked

    def _apply_temporal_decay(self, candidates: list[SearchResult]) -> list[SearchResult]:
        now = datetime.now(timezone.utc).timestamp()
        decayed = []
        for c in candidates:
            age_days = max(0, (now - c.timestamp) / 86400) if c.timestamp is not None else 0
            decay_factor = math.exp(-self.settings.temporal_decay_lambda * age_days)
            # Importance is centered at 0.5 (neutral). Older formula (0.5+0.5*imp)
            # halved default scores and made the 0.72 threshold almost unreachable.
            importance_factor = 1.0 + 0.25 * (c.importance - 0.5)
            adjusted_score = c.score * decay_factor * importance_factor
            decayed.append(
                SearchResult(
                    chunk_id=c.chunk_id,
                    content=c.content,
                    score=adjusted_score,
                    memory_type=c.memory_type,
                    session_id=c.session_id,
                    importance=c.importance,
                    timestamp=c.timestamp,
                )
            )
        decayed.sort(key=lambda x: x.score, reverse=True)
        return decayed

    def _deduplicate(self, candidates: list[SearchResult]) -> list[SearchResult]:
        if len(candidates) <= 1:
            return candidates

        texts = [c.content for c in candidates]
        vectors = self.embedding._embed_sync(texts, is_query=False)

        kept: list[SearchResult] = []
        kept_vectors: list[list[float]] = []

        for candidate, vector in zip(candidates, vectors):
            if not kept_vectors:
                kept.append(candidate)
                kept_vectors.append(vector)
                continue

            max_sim = max(
                self._cosine_similarity(vector, kv) for kv in kept_vectors
            )
            if max_sim < self.settings.dedup_similarity_threshold:
                kept.append(candidate)
                kept_vectors.append(vector)

        return kept

    @staticmethod
    def _cosine_similarity(a: list[float], b: list[float]) -> float:
        va = np.array(a)
        vb = np.array(b)
        denom = np.linalg.norm(va) * np.linalg.norm(vb)
        if denom == 0:
            return 0.0
        return float(np.dot(va, vb) / denom)

    def _mmr_select(
        self,
        candidates: list[SearchResult],
        query_vector: list[float],
        top_k: int,
    ) -> list[SearchResult]:
        if not candidates:
            return []

        candidate_vectors = self.embedding._embed_sync([c.content for c in candidates], is_query=False)
        selected: list[SearchResult] = []
        selected_vectors: list[list[float]] = []
        remaining = list(zip(candidates, candidate_vectors))

        lambda_param = self.settings.mmr_lambda

        while remaining and len(selected) < top_k:
            best_idx = 0
            best_score = float("-inf")

            for idx, (candidate, vector) in enumerate(remaining):
                relevance = self._cosine_similarity(query_vector, vector) * candidate.score
                redundancy = 0.0
                if selected_vectors:
                    redundancy = max(
                        self._cosine_similarity(vector, sv) for sv in selected_vectors
                    )
                mmr_score = lambda_param * relevance - (1 - lambda_param) * redundancy
                if mmr_score > best_score:
                    best_score = mmr_score
                    best_idx = idx

            chosen, chosen_vector = remaining.pop(best_idx)
            selected.append(chosen)
            selected_vectors.append(chosen_vector)

        return selected

    async def retrieve(
        self,
        query: str,
        user_id: uuid.UUID,
        session_id: uuid.UUID | None = None,
        short_term_context: list[Message] | None = None,
    ) -> list[SearchResult]:
        context_snippet = ""
        if short_term_context:
            recent = short_term_context[-2:]
            context_snippet = " ".join(f"{m.role}: {m.content}" for m in recent)

        query_text = f"{query} {context_snippet}".strip()

        candidates = await self.embedding.search(
            query=query_text,
            user_id=user_id,
            session_id=session_id,
        )

        if not candidates:
            return []

        # Rerank first (logits → probabilities), then apply temporal decay.
        candidates = await run_sync(self._rerank_sync, query, candidates)
        candidates = self._apply_temporal_decay(candidates)

        filtered = [
            c for c in candidates if c.score >= self.settings.similarity_threshold
        ]

        if not filtered and candidates and session_id:
            same_session = [
                c for c in candidates
                if c.session_id == str(session_id)
            ]
            if same_session:
                filtered = same_session[: self.settings.retrieval_top_k]

        if not filtered and candidates:
            filtered = candidates[:1]

        filtered = self._deduplicate(filtered)

        query_vector = await self.embedding.embed(query_text, is_query=True)
        top_k = self._mmr_select(
            filtered,
            query_vector,
            self.settings.retrieval_top_k,
        )

        return top_k

    async def keyword_boost(
        self,
        db: AsyncSession,
        query: str,
        vector_results: list[SearchResult],
        user_id: uuid.UUID,
        limit: int = 5,
    ) -> list[SearchResult]:
        """Simple BM25-style keyword fallback using PostgreSQL ILIKE."""
        keywords = [w for w in query.lower().split() if len(w) > 3][:5]
        if not keywords:
            return vector_results

        stmt = select(MemoryChunk).where(MemoryChunk.user_id == user_id).limit(limit * 2)
        result = await db.execute(stmt)
        chunks = result.scalars().all()

        keyword_hits: list[SearchResult] = []
        existing_ids = {r.chunk_id for r in vector_results}

        for chunk in chunks:
            content_lower = chunk.content.lower()
            matches = sum(1 for kw in keywords if kw in content_lower)
            if matches > 0 and chunk.id not in existing_ids:
                keyword_hits.append(
                    SearchResult(
                        chunk_id=chunk.id,
                        content=chunk.content,
                        score=0.5 + 0.1 * matches,
                        memory_type=chunk.memory_type,
                        session_id=str(chunk.session_id) if chunk.session_id else None,
                        importance=chunk.importance,
                        timestamp=int(chunk.created_at.timestamp()),
                    )
                )

        merged = vector_results + keyword_hits
        merged.sort(key=lambda x: x.score, reverse=True)
        return merged[: self.settings.retrieval_top_k]
