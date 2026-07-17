-- PostgreSQL schema for Token-Efficient Context Management System
-- Tables are auto-created by SQLAlchemy on startup; this file is for reference.

CREATE TABLE IF NOT EXISTS users (
    id            UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    email         TEXT UNIQUE,
    created_at    TIMESTAMPTZ DEFAULT now()
);

CREATE TABLE IF NOT EXISTS sessions (
    id            UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    user_id       UUID REFERENCES users(id),
    title         TEXT,
    status        TEXT DEFAULT 'active',
    created_at    TIMESTAMPTZ DEFAULT now(),
    updated_at    TIMESTAMPTZ DEFAULT now()
);

CREATE TABLE IF NOT EXISTS messages (
    id            UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    session_id    UUID REFERENCES sessions(id),
    role          TEXT NOT NULL,
    content       TEXT NOT NULL,
    token_count   INT,
    turn_index    INT NOT NULL,
    is_summarized BOOLEAN DEFAULT false,
    created_at    TIMESTAMPTZ DEFAULT now()
);
CREATE INDEX IF NOT EXISTS idx_messages_session ON messages(session_id, turn_index);

CREATE TABLE IF NOT EXISTS conversation_summaries (
    id            UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    session_id    UUID REFERENCES sessions(id),
    user_id       UUID REFERENCES users(id),
    summary_text  TEXT NOT NULL,
    covers_from   INT,
    covers_to     INT,
    token_count   INT,
    embedding_id  TEXT,
    created_at    TIMESTAMPTZ DEFAULT now()
);

CREATE TABLE IF NOT EXISTS user_facts (
    id            UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    user_id       UUID REFERENCES users(id),
    fact_key      TEXT NOT NULL,
    fact_value    TEXT NOT NULL,
    confidence    FLOAT DEFAULT 1.0,
    source        TEXT,
    embedding_id  TEXT,
    updated_at    TIMESTAMPTZ DEFAULT now(),
    UNIQUE(user_id, fact_key)
);

CREATE TABLE IF NOT EXISTS memory_chunks (
    id            UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    user_id       UUID REFERENCES users(id),
    session_id    UUID REFERENCES sessions(id),
    memory_type   TEXT NOT NULL,
    content       TEXT NOT NULL,
    qdrant_id     TEXT NOT NULL,
    importance    FLOAT DEFAULT 0.5,
    topics        TEXT[],
    created_at    TIMESTAMPTZ DEFAULT now(),
    expires_at    TIMESTAMPTZ
);
CREATE INDEX IF NOT EXISTS idx_memory_user ON memory_chunks(user_id, memory_type);

CREATE TABLE IF NOT EXISTS retrieval_logs (
    id            UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    message_id    UUID REFERENCES messages(id),
    query_text    TEXT,
    retrieved_ids UUID[],
    scores        FLOAT[],
    token_budget  INT,
    context_tokens_used INT,
    created_at    TIMESTAMPTZ DEFAULT now()
);

CREATE TABLE IF NOT EXISTS token_usage_logs (
    id            UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    session_id    UUID REFERENCES sessions(id),
    message_id    UUID REFERENCES messages(id),
    context_tokens INT NOT NULL,
    response_tokens INT,
    naive_baseline_tokens INT,
    savings_percent FLOAT,
    created_at    TIMESTAMPTZ DEFAULT now()
);
