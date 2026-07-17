"""Realistic benchmark: proves token savings vs the naive last-18-messages approach.

Simulates a longer conversation with substantive messages (like real usage),
which is where semantic retrieval + summarization pay off. Compares the system's
bounded context against the naive baseline of resending the last 18 messages.
"""

import os
import uuid

import httpx

BASE = os.environ.get("SMOKE_BASE_URL", "http://127.0.0.1:9200")

# Substantive, realistic-length user turns spanning multiple topics.
TURNS = [
    "Hi, I'm Priya, a final-year computer science student. I mainly work in Python "
    "and I'm comfortable with FastAPI and PostgreSQL but new to vector databases.",
    "For my thesis I'm building a customer-support chatbot that must remember user "
    "preferences across many sessions without blowing up the token budget every call.",
    "Right now my prototype resends the last 18 messages on every request, which is "
    "expensive and slow. I want to replace that with semantic retrieval instead.",
    "Can you explain the difference between short-term memory, long-term memory, and "
    "persistent user facts, and when each one should be included in the prompt?",
    "How should I chunk conversations before embedding them? Should I embed every "
    "single message, or group them into turn-pairs or topic segments?",
    "Which embedding model would you recommend if I want to run everything locally on "
    "a laptop with no GPU, while still getting good retrieval accuracy?",
    "I've heard about cosine similarity and reranking. How do cross-encoder rerankers "
    "improve results compared to plain vector similarity search?",
    "When should the system summarize old messages, and how do I make sure the summary "
    "still captures the important decisions and facts from earlier in the chat?",
    "What database schema would you use to store messages, summaries, user facts, and "
    "the embedding metadata so I can audit retrieval later?",
    "Finally, remind me: which programming language did I say I mainly work in, and what "
    "is the main goal of the chatbot I'm building for my thesis?",
]


def main() -> None:
    with httpx.Client(base_url=BASE, timeout=300.0) as client:
        email = f"priya+{uuid.uuid4().hex[:8]}@uni.edu"
        user = client.post("/users", json={"email": email}).json()
        session = client.post(
            "/sessions", json={"user_id": user["id"], "title": "Thesis Benchmark"}
        ).json()
        print(f"user={user['id']} session={session['id']}\n")

        total_context = 0
        total_naive = 0
        print(f"{'turn':<5}{'ctx_tokens':<12}{'naive_18':<10}{'savings':<10}{'retrieved'}")
        print("-" * 50)

        for i, msg in enumerate(TURNS, 1):
            resp = client.post(
                "/chat",
                json={
                    "user_id": user["id"],
                    "session_id": session["id"],
                    "message": msg,
                },
            ).json()
            ctx = resp["context_tokens_used"]
            naive = resp["naive_baseline_tokens"]
            total_context += ctx
            total_naive += naive
            print(
                f"{i:<5}{ctx:<12}{naive:<10}{resp['savings_percent']:<10}"
                f"{len(resp['retrieved_memories'])}"
            )

        overall = round((1 - total_context / total_naive) * 100, 2) if total_naive else 0
        print("-" * 50)
        print(f"TOTAL context tokens : {total_context}")
        print(f"TOTAL naive-18 tokens: {total_naive}")
        print(f"OVERALL SAVINGS      : {overall}%")

        print("\n=== metrics summary ===")
        print(client.get("/metrics/summary").json())


if __name__ == "__main__":
    main()
