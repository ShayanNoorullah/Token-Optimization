import logging
import re
from dataclasses import dataclass

import httpx

from app.config import Settings, get_settings

logger = logging.getLogger(__name__)

SUMMARIZE_PROMPT = """Summarize the following conversation segment concisely.
Preserve key facts, decisions, preferences, and outcomes. Use bullet points.

Conversation:
{text}
"""

MERGE_SUMMARIES_PROMPT = """Merge these conversation summaries into one coherent summary.
Remove redundancy. Preserve all important facts and decisions.

Summaries:
{text}
"""

FACT_EXTRACTION_PROMPT = """Extract persistent user facts from this conversation turn.
Return one fact per line in format: key=value
Only include high-confidence personal facts (name, preferences, goals, constraints).
If none, return NONE.

User: {user_message}
Assistant: {assistant_message}
"""


@dataclass
class LLMResponse:
    content: str
    prompt_tokens: int
    completion_tokens: int


class LLMService:
    def __init__(self, settings: Settings | None = None):
        self.settings = settings or get_settings()

    async def complete(self, prompt: str, max_tokens: int = 1024) -> LLMResponse:
        if self.settings.llm_mock:
            return LLMResponse(
                content=self._mock_response(prompt),
                prompt_tokens=len(prompt.split()),
                completion_tokens=50,
            )

        async with httpx.AsyncClient(timeout=120.0) as client:
            response = await client.post(
                f"{self.settings.llm_base_url.rstrip('/')}/chat/completions",
                headers={
                    "Authorization": f"Bearer {self.settings.llm_api_key}",
                    "Content-Type": "application/json",
                },
                json={
                    "model": self.settings.llm_model,
                    "messages": [{"role": "user", "content": prompt}],
                    "max_tokens": max_tokens,
                    "temperature": 0.3,
                },
            )
            response.raise_for_status()
            data = response.json()

        choice = data["choices"][0]["message"]["content"]
        usage = data.get("usage", {})
        return LLMResponse(
            content=choice.strip(),
            prompt_tokens=usage.get("prompt_tokens", 0),
            completion_tokens=usage.get("completion_tokens", 0),
        )

    async def chat(self, system_prompt: str, user_message: str, max_tokens: int = 1024) -> LLMResponse:
        if self.settings.llm_mock:
            return LLMResponse(
                content=self._mock_chat_response(system_prompt, user_message),
                prompt_tokens=len(system_prompt.split()) + len(user_message.split()),
                completion_tokens=80,
            )

        async with httpx.AsyncClient(timeout=120.0) as client:
            response = await client.post(
                f"{self.settings.llm_base_url.rstrip('/')}/chat/completions",
                headers={
                    "Authorization": f"Bearer {self.settings.llm_api_key}",
                    "Content-Type": "application/json",
                },
                json={
                    "model": self.settings.llm_model,
                    "messages": [
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": user_message},
                    ],
                    "max_tokens": max_tokens,
                    "temperature": 0.7,
                },
            )
            response.raise_for_status()
            data = response.json()

        choice = data["choices"][0]["message"]["content"]
        usage = data.get("usage", {})
        return LLMResponse(
            content=choice.strip(),
            prompt_tokens=usage.get("prompt_tokens", 0),
            completion_tokens=usage.get("completion_tokens", 0),
        )

    def _mock_response(self, prompt: str) -> str:
        if "Summarize" in prompt or "Merge" in prompt:
            return "- Discussed project setup and memory architecture.\n- Chose Qdrant for vector storage."
        if "Extract persistent" in prompt:
            return "preferred_language=Python\nproject_goal=token-efficient chatbot"
        return "Mock LLM response."

    def _mock_chat_response(self, system_prompt: str, user_message: str) -> str:
        if "remember" in user_message.lower():
            return "I've noted that for future reference based on our conversation context."
        return (
            f"Based on the provided context, I can help with: {user_message}. "
            "This is a mock response (set LLM_MOCK=false and configure Ollama/OpenAI for real generation)."
        )

    async def summarize(self, text: str) -> str:
        result = await self.complete(SUMMARIZE_PROMPT.format(text=text))
        return result.content

    async def merge_summaries(self, summaries: list[str]) -> str:
        combined = "\n---\n".join(summaries)
        result = await self.complete(MERGE_SUMMARIES_PROMPT.format(text=combined))
        return result.content

    async def extract_facts(self, user_message: str, assistant_message: str) -> dict[str, str]:
        result = await self.complete(
            FACT_EXTRACTION_PROMPT.format(
                user_message=user_message,
                assistant_message=assistant_message,
            ),
            max_tokens=256,
        )
        facts: dict[str, str] = {}
        if result.content.strip().upper() == "NONE":
            return facts

        for line in result.content.splitlines():
            line = line.strip()
            if "=" in line:
                key, _, value = line.partition("=")
                key = re.sub(r"[^a-z0-9_]", "_", key.strip().lower())
                value = value.strip()
                if key and value:
                    facts[key] = value
        return facts
