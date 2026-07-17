import uuid

from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.database import get_db
from app.models.tables import ConversationSummary, Message, Session, User, UserFact
from app.schemas import MessageResponse, SessionSummaryResponse, UserFactItem

router = APIRouter(tags=["sessions"])


@router.get("/sessions/{session_id}/messages", response_model=list[MessageResponse])
async def get_session_messages(
    session_id: uuid.UUID,
    db: AsyncSession = Depends(get_db),
) -> list[Message]:
    session = await db.get(Session, session_id)
    if not session:
        raise HTTPException(status_code=404, detail="Session not found")

    result = await db.execute(
        select(Message)
        .where(Message.session_id == session_id)
        .order_by(Message.turn_index)
    )
    return list(result.scalars().all())


@router.get("/users/{user_id}/facts", response_model=list[UserFactItem])
async def get_user_facts(
    user_id: uuid.UUID,
    db: AsyncSession = Depends(get_db),
) -> list[UserFact]:
    user = await db.get(User, user_id)
    if not user:
        raise HTTPException(status_code=404, detail="User not found")

    result = await db.execute(
        select(UserFact)
        .where(UserFact.user_id == user_id)
        .order_by(UserFact.confidence.desc(), UserFact.updated_at.desc())
    )
    return list(result.scalars().all())


@router.get("/sessions/{session_id}/summary", response_model=SessionSummaryResponse | None)
async def get_session_summary(
    session_id: uuid.UUID,
    db: AsyncSession = Depends(get_db),
):
    session = await db.get(Session, session_id)
    if not session:
        raise HTTPException(status_code=404, detail="Session not found")

    result = await db.execute(
        select(ConversationSummary)
        .where(ConversationSummary.session_id == session_id)
        .order_by(ConversationSummary.created_at.desc())
        .limit(1)
    )
    return result.scalar_one_or_none()
