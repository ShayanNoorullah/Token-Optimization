import uuid

from fastapi import APIRouter, BackgroundTasks, Depends, HTTPException
from sqlalchemy.ext.asyncio import AsyncSession

from app.database import get_db
from app.models.tables import Session
from app.schemas import ChatRequest, ChatResponse
from app.services.chat import ChatOrchestrator

router = APIRouter(prefix="/chat", tags=["chat"])
orchestrator = ChatOrchestrator()


@router.post("", response_model=ChatResponse)
async def chat(
    request: ChatRequest,
    background_tasks: BackgroundTasks,
    db: AsyncSession = Depends(get_db),
) -> ChatResponse:
    session = await db.get(Session, request.session_id)
    if not session:
        raise HTTPException(status_code=404, detail="Session not found")
    if session.user_id != request.user_id:
        raise HTTPException(status_code=403, detail="Session does not belong to user")

    response = await orchestrator.handle_message(
        db=db,
        user_id=request.user_id,
        session_id=request.session_id,
        user_message=request.message,
    )

    background_tasks.add_task(
        orchestrator.post_turn_processing,
        request.user_id,
        request.session_id,
        request.message,
        response.response,
    )
    return response
