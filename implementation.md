# Implementation Guide

Technical documentation for the Token-Efficient Context Management System.

## System Architecture

```
┌─────────────┐     ┌──────────────────────────────────────────────────┐
│  Frontend   │────▶│              FastAPI Gateway                     │
│  (React)    │     │  /chat  /users  /sessions  /metrics              │
└─────────────┘     └──────┬───────────────────────────┬─────────────┘
                             │                           │
                    ┌────────▼────────┐         ┌────────▼────────┐
                    │   PostgreSQL    │         │     Qdrant      │
                    │  messages       │         │  vector search  │
                    │  summaries      │         │  768-dim cosine │
                    │  facts          │         └─────────────────┘
                    │  memory_chunks  │
                    │  token_logs     │
                    └─────────────────┘
```

## Request Lifecycle

1. **User sends message** → saved to PostgreSQL, appended to short-term buffer
2. **Query embedding** generated (BGE-base with query prefix)
3. **Vector search** in Qdrant (top-20 candidates, filtered by user_id)
4. **Rerank** with BGE-reranker cross-encoder
5. **Filter** by similarity threshold (≥ 0.72)
6. **Deduplicate** near-identical chunks (> 0.95 similarity)
7. **MMR select** top-3 diverse results
8. **Fetch** session summary + user facts from PostgreSQL
9. **Build context** within token budget (4096 total)
10. **Call LLM** with assembled system prompt + user message
11. **Log** token usage and retrieval audit trail
12. **Async post-processing**: embed turn-pair, extract facts, check summarization

## Memory Tiers

### Short-Term Memory
- **Contents:** Last 6–8 raw messages of current session
- **Storage:** PostgreSQL `messages` table (where `is_summarized = false`)
- **Usage:** Every request; provides pronoun resolution and follow-up continuity
- **Budget:** Up to 1200 tokens

### Long-Term Memory (Episodic)
- **Contents:** Embedded turn-pair chunks from all sessions
- **Storage:** PostgreSQL `memory_chunks` + Qdrant vectors
- **Usage:** Retrieved via semantic search when query references past topics
- **Chunk format:** `User: ...\nAssistant: ...`

### Persistent User Facts
- **Contents:** Structured profile (name, preferences, goals)
- **Storage:** PostgreSQL `user_facts` table
- **Usage:** Every request (top 10 by confidence)
- **Extraction:** LLM-based after each turn; upserted with confidence scoring

### Conversation Summaries
- **Contents:** Hierarchical summaries per session
- **Storage:** PostgreSQL `conversation_summaries` + Qdrant
- **Trigger:** When unsummarized messages exceed 3000 tokens
- **Process:** Batch summarize → merge → mark messages as summarized

## Retrieval Pipeline

```
Query → Embed → Qdrant ANN (top-20)
  → Temporal Decay (exp(-λ * age_days))
  → Cross-Encoder Rerank
  → Threshold Filter (≥ 0.72)
  → Deduplication (> 0.95 similarity)
  → MMR Selection (top-3)
  → Keyword Boost (PostgreSQL ILIKE fallback)
```

### Similarity Methods
| Method | Implementation |
|--------|---------------|
| Cosine similarity | Qdrant default (L2-normalized BGE vectors) |
| Cross-encoder reranking | `BAAI/bge-reranker-base` |
| MMR | λ=0.5 balancing relevance vs diversity |
| Temporal decay | λ=0.01 per day |
| Keyword boost | PostgreSQL ILIKE on memory_chunks |

## Context Construction

Token budget allocation per request:

| Slot | Budget | Source |
|------|--------|--------|
| System prompt | ~300 | Fixed |
| User facts | 150 | PostgreSQL user_facts |
| Session summary | 400 | conversation_summaries |
| Retrieved memories | 600 | Qdrant top-3 |
| Short-term messages | 1200 | Recent unsummarized messages |
| Query + overhead | ~250 | Current message + separators |
| **Total target** | **~2900** | vs ~5400+ for naive-18 |

Assembly order: system → facts → summary → memories → short-term → query.

## Database Schema

### PostgreSQL Tables

| Table | Purpose |
|-------|---------|
| `users` | User accounts |
| `sessions` | Chat sessions per user |
| `messages` | Raw messages (source of truth) |
| `conversation_summaries` | Hierarchical session summaries |
| `user_facts` | Structured persistent facts |
| `memory_chunks` | Episodic memory metadata |
| `retrieval_logs` | Per-request retrieval audit |
| `token_usage_logs` | Per-request token metrics |

### Qdrant Collection

```json
{
  "collection": "conversation_memory",
  "vectors": { "size": 768, "distance": "Cosine" },
  "payload": ["user_id", "session_id", "memory_type", "chunk_id", "content", "importance", "timestamp"]
}
```

## LlamaIndex Integration

Located in `app/services/llama_index_bridge.py`:

- `configure_llamaindex()` — sets HuggingFace embedding model
- `get_qdrant_vector_store()` — wraps Qdrant client
- `index_conversation_chunks()` — batch index via VectorStoreIndex
- `build_context_chat_engine()` — ContextChatEngine with ChatMemoryBuffer

## Configuration Reference

| Variable | Default | Description |
|----------|---------|-------------|
| `DATABASE_URL` | `postgresql+asyncpg://...` | PostgreSQL connection |
| `QDRANT_HOST` | `localhost` | Qdrant host |
| `QDRANT_PORT` | `6333` | Qdrant port |
| `EMBEDDING_MODEL` | `BAAI/bge-base-en-v1.5` | Sentence-transformer model |
| `RERANKER_MODEL` | `BAAI/bge-reranker-base` | Cross-encoder model |
| `LLM_MOCK` | `false` | Use mock LLM responses |
| `LLM_BASE_URL` | `http://localhost:11434/v1` | Ollama/OpenAI endpoint |
| `SIMILARITY_THRESHOLD` | `0.72` | Min retrieval score |
| `RETRIEVAL_TOP_K` | `3` | Final memories returned |
| `CONTEXT_TOKEN_BUDGET` | `4096` | Max context tokens |
| `SHORT_TERM_MAX_TOKENS` | `1200` | Short-term window budget |
| `SUMMARIZE_THRESHOLD` | `3000` | Tokens before summarization |
| `API_PORT` | `9200` | API server port |
| `CORS_ORIGINS` | `http://localhost:5173,...` | Allowed frontend origins |

## Pseudocode: Full Request Handler

```python
async def handle_message(user_id, session_id, message):
    start = time.perf_counter()

    # 1. Persist and load short-term
    save_message(session_id, "user", message)
    short_term = get_short_term_messages(session_id, max_tokens=1200)

    # 2. Semantic retrieval
    query_embed = embed(message + recent_context)
    candidates = qdrant.query_points(vector=query_embed, filter=user_id, limit=20)
    ranked = rerank(message, candidates)
    filtered = [c for c in ranked if c.score >= 0.72]
    top_k = mmr_select(filtered, query_embed, k=3)

    # 3. Load memory tiers
    summary = get_latest_summary(session_id)
    facts = get_user_facts(user_id, limit=10)

    # 4. Build token-budgeted context
    context = build_context(facts, summary, top_k, short_term, message)

    # 5. Generate response
    response = llm.chat(context.system_prompt, message)

    # 6. Log metrics
    save_message(session_id, "assistant", response)
    log_tokens(context.total_tokens, naive_baseline_18, savings)
    log_retrieval(top_k, scores)

    # 7. Async: embed, extract facts, summarize
    background: store_turn_pair, extract_facts, maybe_summarize

    return ChatResponse(..., latency_ms=elapsed)
```

## File Map

| File | Responsibility |
|------|---------------|
| `app/main.py` | FastAPI app, CORS, lifespan |
| `app/config.py` | Settings from .env |
| `app/schemas.py` | Pydantic request/response models |
| `app/models/tables.py` | SQLAlchemy ORM models |
| `app/services/chat.py` | Request orchestration |
| `app/services/embedding.py` | BGE embeddings + Qdrant ops |
| `app/services/retrieval.py` | Hybrid retriever pipeline |
| `app/services/memory.py` | Tiered memory read/write |
| `app/services/summarizer.py` | Conversation summarization |
| `app/services/context_builder.py` | Token-budgeted prompt assembly |
| `app/services/llm.py` | OpenAI-compatible LLM client |
| `app/api/routes/chat.py` | POST /chat endpoint |
| `app/api/routes/sessions.py` | Message/facts/summary reads |
| `app/api/routes/metrics.py` | Metrics and evaluation |
| `frontend/src/` | React testing dashboard |
