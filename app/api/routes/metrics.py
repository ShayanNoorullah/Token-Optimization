import uuid

from fastapi import APIRouter, Depends
from sqlalchemy import func, select
from sqlalchemy.ext.asyncio import AsyncSession

from app.database import get_db
from app.models.tables import RetrievalLog, TokenUsageLog
from app.schemas import EvaluationResult, MetricsSummary
from app.services.embedding import EmbeddingService
from app.services.retrieval import HybridRetriever

router = APIRouter(prefix="/metrics", tags=["metrics"])


@router.get("/summary", response_model=MetricsSummary)
async def get_metrics_summary(db: AsyncSession = Depends(get_db)) -> MetricsSummary:
    result = await db.execute(
        select(
            func.count(TokenUsageLog.id),
            func.avg(TokenUsageLog.context_tokens),
            func.avg(TokenUsageLog.naive_baseline_tokens),
            func.avg(TokenUsageLog.savings_percent),
        )
    )
    row = result.one()

    retrieval_result = await db.execute(
        select(func.avg(func.cardinality(RetrievalLog.retrieved_ids)))
    )
    avg_retrieval = retrieval_result.scalar() or 0.0

    return MetricsSummary(
        total_requests=row[0] or 0,
        avg_context_tokens=round(float(row[1] or 0), 2),
        avg_naive_baseline_tokens=round(float(row[2] or 0), 2),
        avg_savings_percent=round(float(row[3] or 0), 2),
        avg_retrieval_count=round(float(avg_retrieval), 2),
    )


@router.get("/token-usage")
async def get_token_usage(
    session_id: uuid.UUID | None = None,
    limit: int = 50,
    db: AsyncSession = Depends(get_db),
):
    stmt = select(TokenUsageLog).order_by(TokenUsageLog.created_at.desc()).limit(limit)
    if session_id:
        stmt = stmt.where(TokenUsageLog.session_id == session_id)

    result = await db.execute(stmt)
    logs = result.scalars().all()
    return [
        {
            "id": str(log.id),
            "session_id": str(log.session_id),
            "context_tokens": log.context_tokens,
            "response_tokens": log.response_tokens,
            "naive_baseline_tokens": log.naive_baseline_tokens,
            "savings_percent": log.savings_percent,
            "created_at": log.created_at.isoformat(),
        }
        for log in logs
    ]


@router.get("/retrieval-logs")
async def get_retrieval_logs(limit: int = 50, db: AsyncSession = Depends(get_db)):
    result = await db.execute(
        select(RetrievalLog).order_by(RetrievalLog.created_at.desc()).limit(limit)
    )
    logs = result.scalars().all()
    return [
        {
            "id": str(log.id),
            "message_id": str(log.message_id) if log.message_id else None,
            "query_text": log.query_text,
            "retrieved_count": len(log.retrieved_ids or []),
            "scores": log.scores,
            "context_tokens_used": log.context_tokens_used,
            "created_at": log.created_at.isoformat(),
        }
        for log in logs
    ]


@router.post("/evaluate", response_model=list[EvaluationResult])
async def evaluate_retrieval(
    user_id: uuid.UUID,
    queries: list[str],
    db: AsyncSession = Depends(get_db),
):
    """Basic evaluation harness for retrieval quality and token usage."""
    from app.config import get_settings
    from app.services.context_builder import ContextBuilder

    settings = get_settings()
    retriever = HybridRetriever()
    builder = ContextBuilder()
    results: list[EvaluationResult] = []

    for query in queries:
        retrieved = await retriever.retrieve(query=query, user_id=user_id)
        context = builder.build(
            user_facts=[],
            session_summary=None,
            retrieved_memories=retrieved,
            short_term_messages=[],
            current_user_message=query,
        )
        passed = any(r.score >= settings.similarity_threshold for r in retrieved)
        results.append(
            EvaluationResult(
                query=query,
                retrieved_count=len(retrieved),
                context_tokens=context.total_tokens,
                passed_threshold=passed,
            )
        )

    return results
