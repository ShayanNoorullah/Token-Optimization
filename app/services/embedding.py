import logging
import uuid
from dataclasses import dataclass
from typing import Any

from qdrant_client import QdrantClient
from qdrant_client.http import models as qmodels

from app.config import Settings, get_settings
from app.utils.tokens import normalize_text, run_sync

logger = logging.getLogger(__name__)

QUERY_PREFIX = "Represent this sentence for searching relevant passages: "
PASSAGE_PREFIX = "Represent this sentence for storing in a vector database: "


@dataclass
class SearchResult:
    chunk_id: uuid.UUID
    content: str
    score: float
    memory_type: str
    session_id: str | None
    importance: float
    timestamp: int


class EmbeddingService:
    def __init__(self, settings: Settings | None = None):
        self.settings = settings or get_settings()
        self._model = None
        self._qdrant: QdrantClient | None = None

    def _get_model(self):
        if self._model is None:
            from sentence_transformers import SentenceTransformer

            logger.info("Loading embedding model: %s", self.settings.embedding_model)
            self._model = SentenceTransformer(self.settings.embedding_model)
        return self._model

    def _get_qdrant(self) -> QdrantClient:
        if self._qdrant is None:
            self._qdrant = QdrantClient(
                host=self.settings.qdrant_host,
                port=self.settings.qdrant_port,
                check_compatibility=False,
            )
            self._ensure_collection()
        return self._qdrant

    def _ensure_collection(self) -> None:
        client = self._qdrant
        assert client is not None
        collections = [c.name for c in client.get_collections().collections]
        if self.settings.qdrant_collection not in collections:
            client.create_collection(
                collection_name=self.settings.qdrant_collection,
                vectors_config=qmodels.VectorParams(
                    size=self.settings.embedding_dimension,
                    distance=qmodels.Distance.COSINE,
                ),
            )
            logger.info("Created Qdrant collection: %s", self.settings.qdrant_collection)

    def _embed_sync(self, texts: list[str], is_query: bool = False) -> list[list[float]]:
        model = self._get_model()
        prefix = QUERY_PREFIX if is_query else PASSAGE_PREFIX
        prefixed = [prefix + normalize_text(t) for t in texts]
        vectors = model.encode(prefixed, normalize_embeddings=True)
        return vectors.tolist()

    async def embed(self, text: str, is_query: bool = False) -> list[float]:
        vectors = await run_sync(self._embed_sync, [text], is_query)
        return vectors[0]

    async def embed_batch(self, texts: list[str], is_query: bool = False) -> list[list[float]]:
        return await run_sync(self._embed_sync, texts, is_query)

    def build_metadata_prefix(self, session_id: uuid.UUID | None, memory_type: str, topics: list[str] | None = None) -> str:
        parts = []
        if session_id:
            parts.append(f"[session:{session_id}]")
        parts.append(f"[type:{memory_type}]")
        if topics:
            parts.append(f"[topics:{','.join(topics)}]")
        return " ".join(parts)

    async def upsert_memory(
        self,
        chunk_id: uuid.UUID,
        content: str,
        user_id: uuid.UUID,
        session_id: uuid.UUID | None,
        memory_type: str,
        importance: float = 0.5,
        topics: list[str] | None = None,
    ) -> str:
        prefix = self.build_metadata_prefix(session_id, memory_type, topics)
        full_text = f"{prefix} {normalize_text(content)}"
        vector = (await self.embed_batch([full_text]))[0]

        point_id = str(chunk_id)
        payload: dict[str, Any] = {
            "user_id": str(user_id),
            "session_id": str(session_id) if session_id else "",
            "memory_type": memory_type,
            "chunk_id": point_id,
            "content": content,
            "importance": importance,
            "timestamp": int(__import__("time").time()),
        }
        if topics:
            payload["topics"] = topics

        client = self._get_qdrant()
        client.upsert(
            collection_name=self.settings.qdrant_collection,
            points=[
                qmodels.PointStruct(id=point_id, vector=vector, payload=payload),
            ],
        )
        return point_id

    async def search(
        self,
        query: str,
        user_id: uuid.UUID,
        limit: int | None = None,
        session_id: uuid.UUID | None = None,
    ) -> list[SearchResult]:
        limit = limit or self.settings.retrieval_candidates
        query_vector = await self.embed(query, is_query=True)

        must_conditions: list[qmodels.FieldCondition] = [
            qmodels.FieldCondition(
                key="user_id",
                match=qmodels.MatchValue(value=str(user_id)),
            )
        ]

        client = self._get_qdrant()
        results = client.query_points(
            collection_name=self.settings.qdrant_collection,
            query=query_vector,
            query_filter=qmodels.Filter(must=must_conditions),
            limit=limit,
            with_payload=True,
        ).points

        search_results: list[SearchResult] = []
        for hit in results:
            payload = hit.payload or {}
            chunk_id_str = payload.get("chunk_id", str(hit.id))
            try:
                chunk_id = uuid.UUID(chunk_id_str)
            except ValueError:
                continue

            search_results.append(
                SearchResult(
                    chunk_id=chunk_id,
                    content=payload.get("content", ""),
                    score=float(hit.score),
                    memory_type=payload.get("memory_type", "unknown"),
                    session_id=payload.get("session_id") or None,
                    importance=float(payload.get("importance", 0.5)),
                    timestamp=int(payload.get("timestamp", 0)),
                )
            )

        if session_id:
            sid = str(session_id)
            for r in search_results:
                if r.session_id == sid:
                    r.score = min(1.0, r.score * 1.1)

            search_results.sort(key=lambda x: x.score, reverse=True)

        return search_results

    def delete_by_chunk_ids(self, chunk_ids: list[str]) -> None:
        if not chunk_ids:
            return
        client = self._get_qdrant()
        client.delete(
            collection_name=self.settings.qdrant_collection,
            points_selector=qmodels.PointIdsList(points=chunk_ids),
        )
