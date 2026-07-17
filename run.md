# Run Guide

## Prerequisites

| Requirement | Version |
|-------------|---------|
| Docker Desktop | Latest |
| Python | 3.11+ |

Node.js is **not required** — the frontend is vanilla HTML/CSS/JS served by FastAPI.

## Step 1: Start Infrastructure

```bash
cd "D:\Token Optimization"
docker compose up -d
docker compose ps
```

## Step 2: Python Environment

```bash
python -m venv .venv
.\.venv\Scripts\activate
pip install -r requirements.txt
copy .env.example .env
```

## Step 3: Warm Up Models

```bash
python -m scripts.warmup
python -m scripts.init_db
```

## Step 4: Start API + Frontend

One command serves both the API and the vanilla frontend:

```bash
uvicorn app.main:app --host 127.0.0.1 --port 9200
```

Open **http://127.0.0.1:9200/** — the testing dashboard loads automatically.

- API docs: http://127.0.0.1:9200/docs
- Health: http://127.0.0.1:9200/health

> **Windows note:** Use port 9200 if 8000/8080 are blocked by Hyper-V.

## Step 5: Analyze ARIA Logs

```bash
python -m scripts.analyze_logs
# → writes logs_issue.txt

python -m scripts.replay_aria_logs
# → writes logs_replay_results.json (requires API running)
```

## Step 6: Run Tests

```bash
pytest tests/ -v
python -m scripts.smoke_test
```

## Troubleshooting

### Port already in use

```bash
netstat -ano | findstr :9200
uvicorn app.main:app --host 127.0.0.1 --port 9300
```

### Frontend shows blank page

Ensure `frontend/index.html`, `frontend/styles.css`, and `frontend/app.js` exist. The API serves them at `/`.

### Model download timeout

Run `python -m scripts.warmup` before starting the API.

### Docker not running

Start Docker Desktop, then `docker compose up -d`.

### Negative token savings on short messages

Expected with tiny messages and mock LLM. Use longer conversations or set `LLM_MOCK=false` with Ollama for realistic savings.

## Stopping

```bash
# Ctrl+C to stop API
docker compose down
```
