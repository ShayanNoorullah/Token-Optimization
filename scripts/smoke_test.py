"""End-to-end smoke test against a running server.

Creates a user + session, sends several messages, then checks metrics.
Requires the API running on http://127.0.0.1:8000 and Docker services up.
"""

import httpx

import os
import uuid

BASE = os.environ.get("SMOKE_BASE_URL", "http://127.0.0.1:9123")


def main() -> None:
    with httpx.Client(base_url=BASE, timeout=300.0) as client:
        health = client.get("/health").json()
        print("health:", health)

        email = f"alex+{uuid.uuid4().hex[:8]}@uni.edu"
        user = client.post("/users", json={"email": email}).json()
        print("user:", user["id"])

        session = client.post(
            "/sessions", json={"user_id": user["id"], "title": "Thesis Chat"}
        ).json()
        print("session:", session["id"])

        messages = [
            "Hi, my name is Alex and I prefer Python for all my projects.",
            "I'm building a token-efficient chatbot for my university thesis.",
            "Which vector database should I use for local deployment?",
            "How do I set up Qdrant with Docker?",
            "Remind me, what language did I say I prefer?",
        ]

        for i, msg in enumerate(messages, 1):
            resp = client.post(
                "/chat",
                json={
                    "user_id": user["id"],
                    "session_id": session["id"],
                    "message": msg,
                },
            ).json()
            print(f"\n--- turn {i} ---")
            print("user:", msg)
            print("assistant:", resp["response"][:160])
            print(
                "context_tokens:",
                resp["context_tokens_used"],
                "| naive_baseline:",
                resp["naive_baseline_tokens"],
                "| savings:",
                f"{resp['savings_percent']}%",
            )
            print("retrieved:", len(resp["retrieved_memories"]), "memories")

        summary = client.get("/metrics/summary").json()
        print("\n=== metrics summary ===")
        print(summary)


if __name__ == "__main__":
    main()
