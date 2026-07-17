import logging
import uuid

from sqlalchemy import func, select
from sqlalchemy.ext.asyncio import AsyncSession

from app.config import Settings, get_settings
from app.models.tables import ConversationSummary, MemoryChunk, Message, UserFact
from app.services.embedding import EmbeddingService
from app.services.llm import LLMService
from app.utils.tokens import count_tokens

logger = logging.getLogger(__name__)


class MemoryService:
    def __init__(
        self,
        settings: Settings | None = None,
        embedding_service: EmbeddingService | None = None,
        llm_service: LLMService | None = None,
    ):
        self.settings = settings or get_settings()
        self.embedding = embedding_service or EmbeddingService(self.settings)
        self.llm = llm_service or LLMService(self.settings)

    async def save_message(
        self,
        db: AsyncSession,
        session_id: uuid.UUID,
        role: str,
        content: str,
    ) -> Message:
        result = await db.execute(
            select(func.coalesce(func.max(Message.turn_index), -1)).where(
                Message.session_id == session_id
            )
        )
        max_turn = result.scalar_one()
        turn_index = max_turn + 1

        message = Message(
            session_id=session_id,
            role=role,
            content=content,
            token_count=count_tokens(content),
            turn_index=turn_index,
        )
        db.add(message)
        await db.flush()
        return message

    async def get_short_term_messages(
        self,
        db: AsyncSession,
        session_id: uuid.UUID,
    ) -> list[Message]:
        result = await db.execute(
            select(Message)
            .where(
                Message.session_id == session_id,
                Message.is_summarized.is_(False),
            )
            .order_by(Message.turn_index.desc())
            .limit(self.settings.short_term_max_messages)
        )
        messages = list(result.scalars().all())
        messages.reverse()

        total_tokens = 0
        trimmed: list[Message] = []
        for msg in reversed(messages):
            msg_tokens = msg.token_count or count_tokens(msg.content)
            if total_tokens + msg_tokens > self.settings.short_term_max_tokens:
                break
            trimmed.insert(0, msg)
            total_tokens += msg_tokens

        return trimmed

    async def get_latest_summary(
        self,
        db: AsyncSession,
        session_id: uuid.UUID,
    ) -> ConversationSummary | None:
        result = await db.execute(
            select(ConversationSummary)
            .where(ConversationSummary.session_id == session_id)
            .order_by(ConversationSummary.created_at.desc())
            .limit(1)
        )
        return result.scalar_one_or_none()

    async def get_user_facts(
        self,
        db: AsyncSession,
        user_id: uuid.UUID,
        limit: int = 10,
    ) -> list[UserFact]:
        result = await db.execute(
            select(UserFact)
            .where(UserFact.user_id == user_id)
            .order_by(UserFact.confidence.desc(), UserFact.updated_at.desc())
            .limit(limit)
        )
        return list(result.scalars().all())

    async def upsert_facts(
        self,
        db: AsyncSession,
        user_id: uuid.UUID,
        facts: dict[str, str],
        source: str = "extracted",
    ) -> None:
        for key, value in facts.items():
            result = await db.execute(
                select(UserFact).where(
                    UserFact.user_id == user_id,
                    UserFact.fact_key == key,
                )
            )
            existing = result.scalar_one_or_none()
            if existing:
                existing.fact_value = value
                existing.confidence = min(1.0, existing.confidence + 0.1)
                existing.source = source
            else:
                db.add(
                    UserFact(
                        user_id=user_id,
                        fact_key=key,
                        fact_value=value,
                        confidence=0.8,
                        source=source,
                    )
                )
        await db.flush()

    async def store_turn_pair(
        self,
        db: AsyncSession,
        user_id: uuid.UUID,
        session_id: uuid.UUID,
        user_message: str,
        assistant_message: str,
    ) -> MemoryChunk:
        chunk_text = f"User: {user_message}\nAssistant: {assistant_message}"
        chunk_id = uuid.uuid4()

        qdrant_id = await self.embedding.upsert_memory(
            chunk_id=chunk_id,
            content=chunk_text,
            user_id=user_id,
            session_id=session_id,
            memory_type="turn_pair",
            importance=0.5,
        )

        chunk = MemoryChunk(
            id=chunk_id,
            user_id=user_id,
            session_id=session_id,
            memory_type="turn_pair",
            content=chunk_text,
            qdrant_id=qdrant_id,
            importance=0.5,
        )
        db.add(chunk)
        await db.flush()
        return chunk

    async def get_unsummarized_token_count(
        self,
        db: AsyncSession,
        session_id: uuid.UUID,
    ) -> int:
        result = await db.execute(
            select(Message).where(
                Message.session_id == session_id,
                Message.is_summarized.is_(False),
            )
        )
        messages = result.scalars().all()
        return sum(m.token_count or count_tokens(m.content) for m in messages)

    async def get_naive_baseline_tokens(
        self,
        db: AsyncSession,
        session_id: uuid.UUID,
        window: int = 18,
    ) -> int:
        result = await db.execute(
            select(Message)
            .where(Message.session_id == session_id)
            .order_by(Message.turn_index.desc())
            .limit(window)
        )
        messages = result.scalars().all()
        return sum(m.token_count or count_tokens(m.content) for m in messages)
