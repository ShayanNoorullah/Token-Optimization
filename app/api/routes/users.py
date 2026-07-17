import uuid

from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.database import get_db
from app.models.tables import Session, User
from app.schemas import SessionCreate, SessionResponse, UserCreate, UserResponse

router = APIRouter(tags=["users"])


@router.post("/users", response_model=UserResponse)
async def create_user(payload: UserCreate, db: AsyncSession = Depends(get_db)) -> User:
    user = User(email=payload.email)
    db.add(user)
    await db.flush()
    return user


@router.get("/users/{user_id}", response_model=UserResponse)
async def get_user(user_id: uuid.UUID, db: AsyncSession = Depends(get_db)) -> User:
    user = await db.get(User, user_id)
    if not user:
        raise HTTPException(status_code=404, detail="User not found")
    return user


@router.post("/sessions", response_model=SessionResponse)
async def create_session(payload: SessionCreate, db: AsyncSession = Depends(get_db)) -> Session:
    user = await db.get(User, payload.user_id)
    if not user:
        raise HTTPException(status_code=404, detail="User not found")

    session = Session(user_id=payload.user_id, title=payload.title)
    db.add(session)
    await db.flush()
    return session


@router.get("/sessions/{session_id}", response_model=SessionResponse)
async def get_session(session_id: uuid.UUID, db: AsyncSession = Depends(get_db)) -> Session:
    session = await db.get(Session, session_id)
    if not session:
        raise HTTPException(status_code=404, detail="Session not found")
    return session
