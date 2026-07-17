import logging
import uuid

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.config import Settings, get_settings
from app.models.tables import ConversationSummary, MemoryChunk, Message
from app.services.embedding import EmbeddingService
from app.services.llm import LLMService
from app.utils.tokens import count_tokens

logger = logging.getLogger(__name__)


class SummarizerService:
    def __init__(
        self,
        settings: Settings | None = None,
        embedding_service: EmbeddingService | None = None,
        llm_service: LLMService | None = None,
    ):
        self.settings = settings or get_settings()
        self.embedding = embedding_service or EmbeddingService(self.settings)
        self.llm = llm_service or LLMService(self.settings)

    def _chunk_by_tokens(self, messages: list[Message], max_tokens: int = 2000) -> list[list[Message]]:
        batches: list[list[Message]] = []
        current: list[Message] = []
        current_tokens = 0

        for msg in messages:
            msg_tokens = msg.token_count or count_tokens(msg.content)
            if current_tokens + msg_tokens > max_tokens and current:
                batches.append(current)
                current = []
                current_tokens = 0
            current.append(msg)
            current_tokens += msg_tokens

        if current:
            batches.append(current)
        return batches

    def _format_messages(self, messages: list[Message]) -> str:
        return "\n".join(f"{m.role}: {m.content}" for m in messages)

    async def maybe_summarize(
        self,
        db: AsyncSession,
        session_id: uuid.UUID,
        user_id: uuid.UUID,
    ) -> ConversationSummary | None:
        result = await db.execute(
            select(Message)
            .where(
                Message.session_id == session_id,
                Message.is_summarized.is_(False),
            )
            .order_by(Message.turn_index)
        )
        messages = list(result.scalars().all())
        total_tokens = sum(m.token_count or count_tokens(m.content) for m in messages)
        turn_count = len(messages)

        token_trigger = total_tokens >= self.settings.summarize_threshold
        turn_trigger = turn_count >= self.settings.summarize_turn_threshold

        if not token_trigger and not turn_trigger:
            return None

        summarize_count = max(1, len(messages) // 2)
        to_summarize = messages[:summarize_count]
        batches = self._chunk_by_tokens(to_summarize)

        batch_summaries: list[str] = []
        for batch in batches:
            text = self._format_messages(batch)
            summary = await self.llm.summarize(text)
            batch_summaries.append(summary)

        if len(batch_summaries) > 1:
            final_summary = await self.llm.merge_summaries(batch_summaries)
        else:
            final_summary = batch_summaries[0]

        covers_from = to_summarize[0].turn_index
        covers_to = to_summarize[-1].turn_index

        summary_id = uuid.uuid4()
        qdrant_id = await self.embedding.upsert_memory(
            chunk_id=summary_id,
            content=final_summary,
            user_id=user_id,
            session_id=session_id,
            memory_type="summary",
            importance=0.8,
        )

        summary = ConversationSummary(
            id=summary_id,
            session_id=session_id,
            user_id=user_id,
            summary_text=final_summary,
            covers_from=covers_from,
            covers_to=covers_to,
            token_count=count_tokens(final_summary),
            embedding_id=qdrant_id,
        )
        db.add(summary)

        memory_chunk = MemoryChunk(
            id=summary_id,
            user_id=user_id,
            session_id=session_id,
            memory_type="summary",
            content=final_summary,
            qdrant_id=qdrant_id,
            importance=0.8,
        )
        db.add(memory_chunk)

        for msg in to_summarize:
            msg.is_summarized = True

        await db.flush()
        logger.info(
            "Summarized session %s turns %d-%d (%d tokens -> %d)",
            session_id,
            covers_from,
            covers_to,
            total_tokens,
            summary.token_count,
        )
        return summary

    async def extract_and_store_facts(
        self,
        db: AsyncSession,
        user_id: uuid.UUID,
        user_message: str,
        assistant_message: str,
    ) -> dict[str, str]:
        from app.services.memory import MemoryService

        facts = await self.llm.extract_facts(user_message, assistant_message)
        if facts:
            memory = MemoryService(self.settings, self.embedding, self.llm)
            await memory.upsert_facts(db, user_id, facts, source="extracted")
        return facts
