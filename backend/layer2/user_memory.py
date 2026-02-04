"""
User Memory — fixed-size ring buffer of interactions per user.

Stores the last MAX_HISTORY queries so the widget selector can see
what the user has been asking about and tailor the dashboard story.
"""

import logging

logger = logging.getLogger(__name__)

MAX_HISTORY = 20


class UserMemoryManager:
    """Read/write user interaction history stored in UserMemory model."""

    def record(self, user_id: str, transcript: str, parsed_intent, widget_scenarios: list):
        """Save one interaction and trim to MAX_HISTORY rows for this user."""
        from layer2.models import UserMemory

        UserMemory.objects.create(
            user_id=user_id,
            query=transcript[:500],
            primary_characteristic=getattr(parsed_intent, "primary_characteristic", "") or "",
            domains=getattr(parsed_intent, "domains", []),
            entities_mentioned=(getattr(parsed_intent, "entities", {}) or {}).get("devices", [])[:5],
            scenarios_used=widget_scenarios[:10],
        )
        # Trim: keep only last MAX_HISTORY rows per user
        ids_to_delete = list(
            UserMemory.objects.filter(user_id=user_id)
            .order_by("-created_at")
            .values_list("id", flat=True)[MAX_HISTORY:]
        )
        if ids_to_delete:
            UserMemory.objects.filter(id__in=ids_to_delete).delete()
            logger.debug(f"Trimmed {len(ids_to_delete)} old memory rows for user={user_id}")

    def get_context(self, user_id: str) -> dict:
        """Build a context dict from recent history."""
        from layer2.models import UserMemory

        recent = list(
            UserMemory.objects.filter(user_id=user_id)
            .order_by("-created_at")[:MAX_HISTORY]
            .values("query", "primary_characteristic", "domains",
                    "entities_mentioned", "scenarios_used", "created_at")
        )
        if not recent:
            return {"history_count": 0, "summary": "New user, no history."}

        # Frequency analysis
        char_freq: dict[str, int] = {}
        entity_freq: dict[str, int] = {}
        for r in recent:
            c = r["primary_characteristic"]
            if c:
                char_freq[c] = char_freq.get(c, 0) + 1
            for e in (r["entities_mentioned"] or []):
                entity_freq[e] = entity_freq.get(e, 0) + 1

        top_chars = sorted(char_freq, key=char_freq.get, reverse=True)[:3]
        top_entities = sorted(entity_freq, key=entity_freq.get, reverse=True)[:5]
        last_3 = [r["query"] for r in recent[:3]]

        return {
            "history_count": len(recent),
            "recent_queries": last_3,
            "focus_areas": top_chars,
            "frequent_entities": top_entities,
            "summary": (
                f"User has asked {len(recent)} questions. "
                f"Focus areas: {', '.join(top_chars) or 'general'}. "
                f"Frequently mentioned: {', '.join(top_entities[:3]) or 'various'}."
            ),
        }

    def format_for_prompt(self, user_id: str) -> str:
        """Render context as compact text suitable for LLM prompt injection."""
        ctx = self.get_context(user_id)
        if ctx["history_count"] == 0:
            return "New user — no prior history."
        lines = [f"User has asked {ctx['history_count']} previous questions."]
        if ctx["focus_areas"]:
            lines.append(f"Focus areas: {', '.join(ctx['focus_areas'])}")
        if ctx["frequent_entities"]:
            lines.append(f"Frequently mentioned equipment: {', '.join(ctx['frequent_entities'])}")
        lines.append("Recent questions:")
        for q in ctx["recent_queries"]:
            lines.append(f'  - "{q}"')
        return "\n".join(lines)
