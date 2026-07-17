from dataclasses import dataclass, field

from app.config import Settings, get_settings
from app.models.tables import ConversationSummary, Message, UserFact
from app.services.embedding import SearchResult
from app.utils.tokens import count_tokens

SYSTEM_PROMPT = """You are a helpful assistant. Use ONLY the provided context to answer.
If context is insufficient, say so. Do not invent facts about the user."""


@dataclass
class BuiltContext:
    system_prompt: str
    user_facts_text: str
    session_summary_text: str
    retrieved_memories_text: str
    short_term_text: str
    full_system_message: str
    total_tokens: int
    retrieved_chunks: list[SearchResult] = field(default_factory=list)


class ContextBuilder:
    def __init__(self, settings: Settings | None = None):
        self.settings = settings or get_settings()

    def _truncate_to_tokens(self, text: str, max_tokens: int) -> str:
        if count_tokens(text) <= max_tokens:
            return text

        words = text.split()
        low, high = 0, len(words)
        while low < high:
            mid = (low + high + 1) // 2
            candidate = " ".join(words[-mid:])
            if count_tokens(candidate) <= max_tokens:
                low = mid
            else:
                high = mid - 1

        return " ".join(words[-low:]) if low > 0 else ""

    def _format_facts(self, facts: list[UserFact], max_tokens: int) -> str:
        if not facts:
            return "No known user facts."

        lines = [f"{f.fact_key}={f.fact_value}" for f in facts]
        text = "; ".join(lines)
        return self._truncate_to_tokens(text, max_tokens)

    def _format_summary(self, summary: ConversationSummary | None, max_tokens: int) -> str:
        if not summary:
            return "No session summary yet."
        return self._truncate_to_tokens(summary.summary_text, max_tokens)

    def _format_memories(self, memories: list[SearchResult], max_tokens: int) -> str:
        if not memories:
            return "No relevant past context retrieved."

        parts: list[str] = []
        remaining = max_tokens
        for i, mem in enumerate(memories, 1):
            entry = f"[{mem.memory_type}] (score={mem.score:.2f}) {mem.content}"
            entry_tokens = count_tokens(entry)
            if entry_tokens > remaining:
                entry = self._truncate_to_tokens(entry, remaining)
                if entry:
                    parts.append(entry)
                break
            parts.append(entry)
            remaining -= entry_tokens
            if i < len(memories):
                separator_tokens = count_tokens("\n---\n")
                remaining -= separator_tokens

        return "\n---\n".join(parts)

    def _format_short_term(self, messages: list[Message], max_tokens: int) -> str:
        if not messages:
            return "No recent messages."

        lines: list[str] = []
        total = 0
        for msg in messages:
            line = f"{msg.role}: {msg.content}"
            line_tokens = count_tokens(line)
            if total + line_tokens > max_tokens:
                break
            lines.append(line)
            total += line_tokens

        return "\n".join(lines)

    def build(
        self,
        user_facts: list[UserFact],
        session_summary: ConversationSummary | None,
        retrieved_memories: list[SearchResult],
        short_term_messages: list[Message],
        current_user_message: str,
    ) -> BuiltContext:
        facts_text = self._format_facts(user_facts, self.settings.user_facts_tokens)
        summary_text = self._format_summary(session_summary, self.settings.session_summary_tokens)
        memories_text = self._format_memories(
            retrieved_memories,
            self.settings.retrieved_memories_tokens,
        )

        used_tokens = (
            self.settings.system_prompt_tokens
            + count_tokens(facts_text)
            + count_tokens(summary_text)
            + count_tokens(memories_text)
            + count_tokens(current_user_message)
            + 150
        )
        short_term_budget = max(
            200,
            self.settings.context_token_budget - used_tokens,
        )
        short_term_budget = min(short_term_budget, self.settings.short_term_max_tokens)

        short_term_text = self._format_short_term(short_term_messages, short_term_budget)

        full_system = f"""{SYSTEM_PROMPT}

## User Profile
{facts_text}

## Current Session Summary
{summary_text}

## Relevant Past Context
{memories_text}

## Recent Conversation
{short_term_text}"""

        total_tokens = count_tokens(full_system) + count_tokens(current_user_message)

        return BuiltContext(
            system_prompt=SYSTEM_PROMPT,
            user_facts_text=facts_text,
            session_summary_text=summary_text,
            retrieved_memories_text=memories_text,
            short_term_text=short_term_text,
            full_system_message=full_system,
            total_tokens=total_tokens,
            retrieved_chunks=retrieved_memories,
        )
