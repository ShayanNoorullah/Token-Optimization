# Improvements and Future Features

Potential enhancements organized by priority and category.

## Near-Term (High Value)

### Streaming LLM Responses
- Add SSE or WebSocket endpoint for token-by-token response streaming
- Show partial response in frontend as it generates
- Reduces perceived latency for long answers

### Real LLM Integration with Model Selector
- UI dropdown to switch between Ollama models and OpenAI API
- Display actual model name and provider in stats panel
- Toggle `LLM_MOCK` from frontend settings

### Debug Mode: Full Context Preview
- Show the complete assembled system prompt sent to the LLM
- Expandable panel in stats view for debugging retrieval quality
- Highlight which facts/memories were injected vs skipped

### Alembic Database Migrations
- Replace auto-create-all with versioned migrations
- Support schema evolution without data loss
- `alembic revision --autogenerate` workflow

### Full Docker Compose Stack
- Add API and frontend services to `docker-compose.yml`
- Single `docker compose up` for entire system
- Multi-stage Dockerfile for Python API + Node frontend build

## Memory and Retrieval

### Hybrid BM25 + Vector Search
- Add PostgreSQL full-text search or Elasticsearch for keyword matching
- Combine BM25 and cosine scores with configurable weights
- Better recall for exact codes, names, and error messages

### Cross-Session Episodic Memory Visualization
- Timeline view of all stored memory chunks across sessions
- Filter by topic, date, importance score
- Manual memory pinning and deletion

### Configurable Thresholds via UI
- Sliders for similarity threshold, top-k, token budgets
- Live preview of how threshold changes affect retrieval
- Per-session or global configuration

### Memory Expiration and TTL
- Auto-expire low-importance episodic chunks after 90 days
- UI to set TTL per memory type
- Never expire user facts without explicit consent

### "Remember This" Explicit Fact Pinning
- User command to force-extract and pin a fact
- Higher confidence and priority in context assembly
- Visual indicator in facts panel for pinned vs auto-extracted

## Production Readiness

### JWT Authentication
- User login/registration with token-based auth
- Per-user data isolation enforced at API and Qdrant filter level
- Session ownership validation

### Rate Limiting and Quotas
- Per-user request limits (requests/minute, tokens/day)
- Redis-backed sliding window rate limiter
- Quota display in frontend dashboard

### Celery + Redis Async Pipeline
- Move embedding, summarization, and fact extraction to background workers
- Faster chat response time (no blocking on post-turn processing)
- Retry logic for failed embedding/summary jobs

### Observability (Langfuse / Phoenix)
- Trace full retrieval pipeline per request
- Dashboard for retrieval precision, token usage trends
- Alert on retrieval quality degradation

### GDPR User Data Deletion
- `DELETE /users/{id}` cascades messages, vectors, facts, logs
- Qdrant point deletion by user_id filter
- Audit trail for deletion requests

## Frontend Enhancements

### Session Manager
- List all sessions for a user
- Switch between sessions without losing context
- Archive/delete old sessions

### Export Conversation and Metrics
- Download chat history + per-turn stats as CSV or JSON
- Include retrieval logs and token breakdowns
- Useful for thesis evaluation and reporting

### A/B Comparison View
- Side-by-side: semantic retrieval context vs naive-18 context
- Show exact token difference and response quality
- Toggle between modes per request

### Dark/Light Theme and Responsive Layout
- Theme toggle in session bar
- Mobile-friendly stacked layout
- Accessible color contrast ratios

## Performance

### Embedding Batch Pipeline
- Batch embed multiple chunks in single model forward pass
- Queue turn-pairs and embed in groups of 8–16
- Reduce per-turn embedding latency

### Qdrant Collection Sharding
- Separate collections per user or per time period
- Faster search on large datasets
- Easier data lifecycle management

### GPU Acceleration
- Optional CUDA support for embedding and reranking
- Config flag `EMBEDDING_DEVICE=cuda`
- Significant speedup on batch operations

### Response Caching
- Cache LLM responses for identical query + context hash
- Redis-backed with TTL
- Skip retrieval + LLM for repeated questions

## Research and Evaluation

### Retrieval Quality Benchmarks
- Standardized test set with ground-truth relevant chunks
- Measure precision@k, recall@k, MRR
- Automated regression testing on model/threshold changes

### Token Savings Report Generator
- PDF/HTML report comparing semantic vs naive over N conversations
- Charts, tables, and statistical significance tests
- Thesis-ready output format

### Multi-Model Embedding Comparison
- A/B test BGE-base vs BGE-large vs OpenAI embeddings
- Side-by-side retrieval quality metrics
- Cost vs accuracy tradeoff analysis

## Infrastructure

### Kubernetes Deployment
- Helm chart for API, frontend, PostgreSQL, Qdrant
- Horizontal pod autoscaling for API
- Persistent volume claims for databases

### CI/CD Pipeline
- GitHub Actions: lint, test, build, deploy
- Docker image publishing
- Automated smoke test on PR

### Health Check Dashboard
- Service status page (PostgreSQL, Qdrant, Redis, LLM)
- Model load status and version info
- Connection pool metrics
