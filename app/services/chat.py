import logging
import time
import uuid

from sqlalchemy.ext.asyncio import AsyncSession

from app.config import get_settings
from app.models.tables import RetrievalLog, TokenUsageLog
from app.schemas import (
    ChatResponse,
    ContextBreakdown,
    RetrievedMemory,
    UserFactItem,
)
from app.services.context_builder import ContextBuilder
from app.services.embedding import EmbeddingService
from app.services.llm import LLMService
from app.services.memory import MemoryService
from app.services.retrieval import HybridRetriever
from app.services.summarizer import SummarizerService
from app.utils.tokens import count_tokens

logger = logging.getLogger(__name__)
settings = get_settings()


class ChatOrchestrator:
    def __init__(self):
        self.embedding = EmbeddingService()
        self.llm = LLMService()
        self.memory = MemoryService(embedding_service=self.embedding, llm_service=self.llm)
        self.retriever = HybridRetriever(embedding_service=self.embedding)
        self.summarizer = SummarizerService(embedding_service=self.embedding, llm_service=self.llm)
        self.context_builder = ContextBuilder()

    def _build_context_breakdown(
        self,
        context,
        user_message: str,
    ) -> ContextBreakdown:
        facts_tokens = count_tokens(context.user_facts_text)
        summary_tokens = count_tokens(context.session_summary_text)
        memories_tokens = count_tokens(context.retrieved_memories_text)
        short_term_tokens = count_tokens(context.short_term_text)
        query_tokens = count_tokens(user_message)
        slot_total = facts_tokens + summary_tokens + memories_tokens + short_term_tokens + query_tokens
        overhead_tokens = max(0, context.total_tokens - slot_total)

        return ContextBreakdown(
            user_facts_tokens=facts_tokens,
            session_summary_tokens=summary_tokens,
            retrieved_memories_tokens=memories_tokens,
            short_term_tokens=short_term_tokens,
            query_tokens=query_tokens,
            overhead_tokens=overhead_tokens,
            total_context_tokens=context.total_tokens,
            token_budget=settings.context_token_budget,
        )

    async def handle_message(
        self,
        db: AsyncSession,
        user_id: uuid.UUID,
        session_id: uuid.UUID,
        user_message: str,
    ) -> ChatResponse:
        start = time.perf_counter()

        user_msg = await self.memory.save_message(db, session_id, "user", user_message)
        short_term = await self.memory.get_short_term_messages(db, session_id)

        retrieved = await self.retriever.retrieve(
            query=user_message,
            user_id=user_id,
            session_id=session_id,
            short_term_context=short_term,
        )
        retrieved = await self.retriever.keyword_boost(db, user_message, retrieved, user_id)

        session_summary = await self.memory.get_latest_summary(db, session_id)
        user_facts = await self.memory.get_user_facts(db, user_id)

        context = self.context_builder.build(
            user_facts=user_facts,
            session_summary=session_summary,
            retrieved_memories=retrieved,
            short_term_messages=short_term,
            current_user_message=user_message,
        )

        llm_response = await self.llm.chat(
            system_prompt=context.full_system_message,
            user_message=user_message,
        )

        assistant_msg = await self.memory.save_message(
            db,
            session_id,
            "assistant",
            llm_response.content,
        )

        naive_baseline = await self.memory.get_naive_baseline_tokens(db, session_id, window=18)
        savings = 0.0
        if naive_baseline > 0:
            savings = round((1 - context.total_tokens / naive_baseline) * 100, 2)

        breakdown = self._build_context_breakdown(context, user_message)
        latency_ms = round((time.perf_counter() - start) * 1000, 2)

        retrieval_log = RetrievalLog(
            message_id=user_msg.id,
            query_text=user_message,
            retrieved_ids=[r.chunk_id for r in retrieved],
            scores=[r.score for r in retrieved],
            token_budget=settings.context_token_budget,
            context_tokens_used=context.total_tokens,
        )
        db.add(retrieval_log)

        token_log = TokenUsageLog(
            session_id=session_id,
            message_id=assistant_msg.id,
            context_tokens=context.total_tokens,
            response_tokens=llm_response.completion_tokens,
            naive_baseline_tokens=naive_baseline,
            savings_percent=savings,
        )
        db.add(token_log)
        await db.flush()

        summary_preview = None
        if session_summary:
            text = session_summary.summary_text
            summary_preview = text[:300] + ("..." if len(text) > 300 else "")

        context_preview = context.full_system_message
        if len(context_preview) > 1500:
            context_preview = context_preview[:1500] + "\n... [truncated]"

        return ChatResponse(
            message_id=assistant_msg.id,
            response=llm_response.content,
            context_tokens_used=context.total_tokens,
            naive_baseline_tokens=naive_baseline,
            savings_percent=savings,
            retrieved_memories=[
                RetrievedMemory(
                    chunk_id=r.chunk_id,
                    content=r.content[:200] + ("..." if len(r.content) > 200 else ""),
                    score=round(r.score, 4),
                    memory_type=r.memory_type,
                )
                for r in retrieved
            ],
            response_tokens=llm_response.completion_tokens,
            context_breakdown=breakdown,
            short_term_message_count=len(short_term),
            has_session_summary=session_summary is not None,
            session_summary_preview=summary_preview,
            user_facts=[
                UserFactItem(
                    fact_key=f.fact_key,
                    fact_value=f.fact_value,
                    confidence=f.confidence,
                )
                for f in user_facts
            ],
            retrieval_threshold=settings.similarity_threshold,
            latency_ms=latency_ms,
            assembled_context_preview=context_preview,
        )

    async def post_turn_processing(
        self,
        user_id: uuid.UUID,
        session_id: uuid.UUID,
        user_message: str,
        assistant_message: str,
    ) -> None:
        from app.database import async_session_factory

        try:
            async with async_session_factory() as db:
                await self.memory.store_turn_pair(
                    db, user_id, session_id, user_message, assistant_message
                )
                await self.summarizer.extract_and_store_facts(
                    db, user_id, user_message, assistant_message
                )
                await self.summarizer.maybe_summarize(db, session_id, user_id)
                await db.commit()
        except Exception:
            logger.exception("Post-turn processing failed for session %s", session_id)
