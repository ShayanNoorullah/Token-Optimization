# Token-Efficient Context Management System

Replaces the naive "last 18 messages" chatbot pattern with **semantic memory retrieval** — cutting context tokens by ~60–85% (validated **98.6%** savings vs ARIA logs).

## What it does

| Instead of… | This system… |
|-------------|--------------|
| Sending last 18 messages every time | Retrieves only relevant memories |
| Growing context forever | Summarizes old turns |
| No long-term recall | Stores user facts + episodic memory |

## Stack

Python · FastAPI · PostgreSQL · Qdrant · BGE embeddings · Vanilla HTML/CSS/JS

## Quick start

See **[run.txt](run.txt)** for the shortest setup steps.
Full guide: [run.md](run.md).

```bash
docker compose up -d
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
copy .env.example .env
python -m scripts.warmup
python -m scripts.init_db
uvicorn app.main:app --host 127.0.0.1 --port 9200
```

Open **http://127.0.0.1:9200/**

## Project layout

```
app/                 FastAPI backend
frontend/            Vanilla testing dashboard
scripts/             Warmup, smoke test, log analysis
tests/               Pytest suite
AIConversationLogs/  Sample ARIA logs for comparison
```

## Docs

| File | Content |
|------|---------|
| [run.txt](run.txt) | Minimal setup & run steps |
| [run.md](run.md) | Detailed guide + troubleshooting |
| [implementation.md](implementation.md) | Architecture |
| [test_data.md](test_data.md) | Test prompts |
| [logs_issue.txt](logs_issue.txt) | Issues found in ARIA logs |
| [improvements.md](improvements.md) | Future ideas |

## Tests

```bash
pytest tests/ -v
python -m scripts.smoke_test
```

## License

MIT
