# Test Data

Curated test scenarios for validating the Token-Efficient Context Management System.

## Smoke Test (5 turns)

From `scripts/smoke_test.py`. Quick functional validation.

| # | Prompt | Expected Behavior |
|---|--------|-------------------|
| 1 | "Hi, my name is Alex and I prefer Python for all my projects." | 0 retrieved; facts extracted (`preferred_language=Python`) |
| 2 | "I'm building a token-efficient chatbot for my university thesis." | 0 retrieved; fact `project_goal` may be extracted |
| 3 | "Which vector database should I use for local deployment?" | 0–1 retrieved; context grows |
| 4 | "How do I set up Qdrant with Docker?" | 1–3 retrieved (prior Qdrant discussion) |
| 5 | "Remind me, what language did I say I prefer?" | 0–2 retrieved; should reference Python from facts or memory |

**Run:**

```bash
$env:SMOKE_BASE_URL="http://127.0.0.1:9200"
python -m scripts.smoke_test
```

## Benchmark (10 realistic turns)

From `scripts/benchmark.py`. Substantive messages simulating a thesis consultation.

| # | Topic | Expected Retrieval |
|---|-------|--------------------|
| 1 | Self-introduction (Python, CS student) | 0 |
| 2 | Thesis chatbot goal | 0 |
| 3 | Problem with last-18 approach | 0 |
| 4 | Memory tier explanation request | 0–1 |
| 5 | Chunking strategy question | 1 |
| 6 | Embedding model recommendation | 1–2 |
| 7 | Reranking explanation | 1–2 |
| 8 | Summarization triggers | 2 |
| 9 | Database schema question | 2 |
| 10 | Recall test (language + thesis goal) | 2–3 |

**Run:**

```bash
python -m scripts.benchmark
```

## Retrieval Test Cases

Designed to trigger semantic memory recall.

| Prompt | What It Tests | Min Retrieved |
|--------|--------------|---------------|
| "What vector database did we discuss?" | Topic recall | 1 |
| "Remind me what language I prefer" | Fact + episodic recall | 0 (facts should answer) |
| "What did we decide about embeddings?" | Cross-turn recall | 1 |
| "Summarize our conversation so far" | Summary presence | 0 (summary in context) |
| "How do I set up Qdrant?" | Exact topic match | 1 |

## Summarization Trigger Test

Send 20+ messages with ~200 tokens each to exceed the 3000-token `SUMMARIZE_THRESHOLD`.

**Sample long message:**

```
I'm working on a detailed comparison of vector databases for my thesis.
I need to evaluate Qdrant, Pinecone, Weaviate, Milvus, ChromaDB, and FAISS
across speed, cost, scalability, ease of use, local deployment, and cloud
deployment. For each database, I want to understand the embedding dimensions
supported, filtering capabilities, ANN algorithm used, and LlamaIndex integration
status. My laptop has 16GB RAM and no GPU, so local CPU performance matters.
```

After ~8–10 such messages:
- `has_session_summary` should become `true`
- `session_summary_preview` should appear in stats
- Older messages marked `is_summarized = true`
- Short-term buffer shrinks to recent unsummarized messages only

## Expected Stats Per Scenario

| Scenario | Context Tokens | Naive-18 | Savings | Retrieved |
|----------|---------------|----------|---------|-----------|
| First message (short) | 80–150 | 50–100 | Negative* | 0 |
| Turn 5 (medium convo) | 200–500 | 300–600 | 0–40% | 0–1 |
| Turn 10 (long convo) | 400–800 | 800–1500 | 20–50% | 1–3 |
| After summarization | 300–600 | 1500–3000 | 50–80% | 1–3 |
| Recall query | 300–700 | 1000–2000 | 30–60% | 2–3 |

*Negative savings on early short messages is expected — the fixed context scaffolding (system prompt, headers) exceeds the naive baseline when messages are tiny.

## Evaluation Endpoint

Test retrieval quality without sending chat messages:

```bash
curl -X POST "http://127.0.0.1:9200/metrics/evaluate?user_id=<USER_ID>" \
  -H "Content-Type: application/json" \
  -d '["How do I set up Qdrant?", "What language do I prefer?"]'
```

Response:

```json
[
  {
    "query": "How do I set up Qdrant?",
    "retrieved_count": 2,
    "context_tokens": 450,
    "passed_threshold": true
  }
]
```

## Frontend Test Workflow

1. Open http://localhost:5173
2. Session auto-initializes
3. Click sample prompt chips or type custom messages
4. After each response, verify in the right panel:
   - **Stats Panel**: context tokens, savings %, breakdown bars
   - **Token Chart**: growing comparison bars
   - **Retrieval Panel**: memory cards with scores
   - **Facts Panel**: extracted key-value pairs
   - **Global Metrics**: running averages
5. Click "New Session" to reset and test fresh conversation

## Automated Test Suite

```bash
pytest tests/ -v
```

| Test File | Coverage |
|-----------|----------|
| `test_context.py` | Context builder, token budget, API health |
| `test_retrieval.py` | Temporal decay, dedup, MMR, cosine similarity |
| `test_metrics.py` | LLM mock, token savings calculation, evaluate endpoint |
| `test_chat_stats.py` | Extended ChatResponse breakdown fields |
