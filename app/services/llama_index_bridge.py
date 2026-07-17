"""LlamaIndex integration helpers for indexing and retrieval pipelines."""

from __future__ import annotations

import uuid

from llama_index.core import Document, Settings as LlamaSettings, VectorStoreIndex
from llama_index.core.node_parser import SentenceSplitter
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.vector_stores.qdrant import QdrantVectorStore
from qdrant_client import QdrantClient

from app.config import Settings, get_settings


def configure_llamaindex(settings: Settings | None = None) -> HuggingFaceEmbedding:
    settings = settings or get_settings()
    embed_model = HuggingFaceEmbedding(model_name=settings.embedding_model)
    LlamaSettings.embed_model = embed_model
    return embed_model


def get_qdrant_vector_store(settings: Settings | None = None) -> QdrantVectorStore:
    settings = settings or get_settings()
    client = QdrantClient(host=settings.qdrant_host, port=settings.qdrant_port)
    return QdrantVectorStore(
        client=client,
        collection_name=settings.qdrant_collection,
    )


def index_conversation_chunks(
    chunks: list[str],
    user_id: uuid.UUID,
    session_id: uuid.UUID,
    settings: Settings | None = None,
) -> VectorStoreIndex:
    """Index conversation turn-pairs using LlamaIndex VectorStoreIndex."""
    settings = settings or get_settings()
    configure_llamaindex(settings)
    vector_store = get_qdrant_vector_store(settings)

    documents = [
        Document(
            text=chunk,
            metadata={
                "user_id": str(user_id),
                "session_id": str(session_id),
                "memory_type": "turn_pair",
            },
        )
        for chunk in chunks
    ]

    splitter = SentenceSplitter(chunk_size=512, chunk_overlap=64)
    return VectorStoreIndex.from_documents(
        documents,
        vector_store=vector_store,
        transformations=[splitter],
        show_progress=False,
    )


def build_context_chat_engine(index: VectorStoreIndex, token_limit: int = 1500):
    """Create a LlamaIndex context chat engine for retrieval-augmented chat."""
    from llama_index.core.chat_engine import ContextChatEngine
    from llama_index.core.memory import ChatMemoryBuffer

    memory = ChatMemoryBuffer.from_defaults(token_limit=token_limit)
    return index.as_chat_engine(chat_mode="context", memory=memory, similarity_top_k=3)
