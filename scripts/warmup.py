"""Pre-download and warm up embedding + reranker models."""

from sentence_transformers import CrossEncoder, SentenceTransformer

from app.config import get_settings


def main() -> None:
    settings = get_settings()
    print(f"Loading embedding model: {settings.embedding_model}")
    emb = SentenceTransformer(settings.embedding_model)
    emb.encode(["warmup sentence"], normalize_embeddings=True)
    print("Embedding model ready.")

    print(f"Loading reranker model: {settings.reranker_model}")
    rr = CrossEncoder(settings.reranker_model)
    rr.predict([["query", "passage"]])
    print("Reranker model ready.")


if __name__ == "__main__":
    main()
