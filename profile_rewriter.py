"""Main orchestration for Memory-aware Profile Rewriting."""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any, Protocol

from memory_store import SessionMemoryStore
from prompts import (
    GLOBAL_PREFERENCE_REASONING_PROMPT,
    INTENT_INTEGRATION_PROMPT,
    INTENT_REFLECTION_PROMPT,
    MEMORY_PREFERENCE_DECISION_PROMPT,
    MEMORY_PREFERENCE_REASONING_PROMPT,
    PROFILE_REFLECTION_PROMPT,
    PROFILE_REWRITE_PROMPT,
    QUERY_ANALYSIS_FIRST_TURN_PROMPT,
    QUERY_ANALYSIS_FOLLOWUP_TURN_PROMPT,
    render_prompt,
)
from qwen_client import QwenChatClient

INVALID = "INVALID"
IRRELEVANT = "irrelevant"


class ChatClient(Protocol):
    def chat(self, messages: list[dict[str, str]]) -> str:
        """Return an assistant response for chat messages."""


DEFAULTS: dict[str, dict[str, str]] = {
    "query_first": {
        "intent_reasoning": INVALID,
        "current_shopping_intent": INVALID,
        "has_new_preference": "no",
        "new_preferences": INVALID,
        "has_related_global_preference": "no",
        "related_global_preferences": INVALID,
    },
    "query_followup": {
        "intent_reasoning": INVALID,
        "current_shopping_intent": INVALID,
        "has_new_preference": "no",
        "new_preferences": INVALID,
    },
    "intent_integration": {
        "reasoning": IRRELEVANT,
        "is_related_to_memory": "no",
        "related_memory_intents": IRRELEVANT,
        "integrated_intent": IRRELEVANT,
    },
    "intent_reflection": {
        "reasoning": IRRELEVANT,
        "has_invalid_information": "no",
        "is_irrelevant_to_current_query": "no",
        "refined_intent": IRRELEVANT,
    },
    "memory_preference_reasoning": {
        "reasoning": INVALID,
    },
    "memory_preferences": {
        "reasoning": INVALID,
        "has_related_memory_preferences": "no",
        "related_memory_preferences": INVALID,
    },
    "global_preferences": {
        "reasoning": INVALID,
        "has_related_global_preferences": "no",
        "related_global_preferences": INVALID,
    },
    "profile_rewrite": {
        "reasoning": INVALID,
        "rewritten_profile": INVALID,
    },
    "profile_reflection": {
        "reasoning": INVALID,
        "needs_adjustment": "no",
        "adjusted_profile": INVALID,
    },
}


class MemoryAwareProfileRewriter:
    """Reusable per-session profile rewriter.

    Create one instance per user session and call ``process_query`` repeatedly.
    The same instance owns the session memory, so multi-turn context is retained.
    """

    def __init__(
        self,
        user_profile: str | None = None,
        user_profile_path: str | Path | None = None,
        llm_client: ChatClient | None = None,
    ) -> None:
        if user_profile is None and user_profile_path is not None:
            user_profile = Path(user_profile_path).read_text(encoding="utf-8")
        self.user_profile = user_profile or ""
        self.llm_client = llm_client or QwenChatClient()
        self.memory_store = SessionMemoryStore()

    @property
    def memory(self) -> list[dict[str, Any]]:
        return self.memory_store.items

    def process_query(self, query: str) -> dict[str, Any]:
        """Process one query and return structured intermediate/final outputs."""

        query = query.strip()
        if not query:
            raise ValueError("query must not be empty")

        previous_memory = self.memory_store.snapshot()
        is_first_turn = len(previous_memory) == 0

        if is_first_turn:
            query_analysis = self._query_analysis_first_turn(query)
        else:
            query_analysis = self._query_analysis_followup_turn(query)

        current_intent = query_analysis.get("current_shopping_intent", INVALID)
        new_preferences = query_analysis.get("new_preferences", INVALID)
        self.memory_store.add_turn(query, current_intent, new_preferences)

        intent_integration: dict[str, str] | None = None
        intent_reflection: dict[str, str] | None = None
        used_intent = current_intent
        related_memory_preferences = INVALID
        memory_preference_reasoning = DEFAULTS["memory_preferences"].copy()

        if not is_first_turn:
            intent_integration = self._intent_integration(query, current_intent, previous_memory)
            integrated_intent = intent_integration.get("integrated_intent", IRRELEVANT)

            if self._is_effective_text(integrated_intent, invalid_value=IRRELEVANT):
                intent_reflection = self._intent_reflection(query, current_intent, integrated_intent)
                refined_intent = intent_reflection.get("refined_intent", IRRELEVANT)
                if self._is_effective_text(refined_intent, invalid_value=IRRELEVANT):
                    used_intent = refined_intent
                else:
                    used_intent = current_intent
                intent_reflection["used_intent"] = used_intent
            else:
                used_intent = current_intent
                intent_reflection = {
                    "has_invalid_information": "no",
                    "is_irrelevant_to_current_query": "yes",
                    "refined_intent": IRRELEVANT,
                    "used_intent": used_intent,
                }

            memory_preference_reasoning = self._memory_preference_reasoning(
                query=query,
                used_intent=used_intent,
                previous_memory=previous_memory,
            )
            related_memory_preferences = memory_preference_reasoning.get(
                "related_memory_preferences", INVALID
            )

        if is_first_turn:
            global_preference_reasoning = {
                "has_related_global_preferences": query_analysis.get(
                    "has_related_global_preference", "no"
                ),
                "related_global_preferences": query_analysis.get(
                    "related_global_preferences", INVALID
                ),
            }
        else:
            global_preference_reasoning = self._global_preference_reasoning(query, used_intent)

        related_global_preferences = global_preference_reasoning.get(
            "related_global_preferences", INVALID
        )
        profile_rewrite = self._profile_rewrite(
            query=query,
            used_intent=used_intent,
            new_preferences=new_preferences,
            related_memory_preferences=related_memory_preferences,
            related_global_preferences=related_global_preferences,
        )
        initial_rewritten_profile = profile_rewrite.get("rewritten_profile", INVALID)
        profile_reflection = self._profile_reflection(
            query=query,
            used_intent=used_intent,
            new_preferences=new_preferences,
            related_memory_preferences=related_memory_preferences,
            related_global_preferences=related_global_preferences,
            rewritten_profile=initial_rewritten_profile,
        )
        adjusted_profile = profile_reflection.get("adjusted_profile", INVALID)
        if (
            profile_reflection.get("needs_adjustment", "no").strip().lower() == "yes"
            and self._is_effective_text(adjusted_profile, invalid_value=INVALID)
        ):
            final_rewritten_profile = adjusted_profile
        else:
            final_rewritten_profile = initial_rewritten_profile

        return {
            "turn_id": len(self.memory_store),
            "query": query,
            "query_analysis": query_analysis,
            "intent_integration": intent_integration,
            "intent_reflection": intent_reflection,
            "memory_preference_reasoning": memory_preference_reasoning,
            "global_preference_reasoning": global_preference_reasoning,
            "profile_rewrite_reasoning": profile_rewrite.get("reasoning", INVALID),
            "initial_rewritten_profile": initial_rewritten_profile,
            "profile_rewrite_reflection": profile_reflection,
            "rewritten_profile": final_rewritten_profile,
            "memory": self.memory_store.snapshot(),
        }

    def _chat_json(self, prompt: str, defaults: dict[str, str]) -> dict[str, str]:
        raw = self.llm_client.chat([{"role": "user", "content": prompt}])
        return parse_llm_output(raw, defaults)

    def _query_analysis_first_turn(self, query: str) -> dict[str, str]:
        prompt = render_prompt(
            QUERY_ANALYSIS_FIRST_TURN_PROMPT,
            query=query,
            user_profile=self.user_profile or INVALID,
        )
        return self._chat_json(prompt, DEFAULTS["query_first"])

    def _query_analysis_followup_turn(self, query: str) -> dict[str, str]:
        prompt = render_prompt(QUERY_ANALYSIS_FOLLOWUP_TURN_PROMPT, query=query)
        return self._chat_json(prompt, DEFAULTS["query_followup"])

    def _intent_integration(
        self, query: str, current_shopping_intent: str, previous_memory: list[dict[str, Any]]
    ) -> dict[str, str]:
        prompt = render_prompt(
            INTENT_INTEGRATION_PROMPT,
            query=query,
            current_shopping_intent=current_shopping_intent,
            previous_memory=json.dumps(previous_memory, ensure_ascii=False, indent=2),
        )
        return self._chat_json(prompt, DEFAULTS["intent_integration"])

    def _intent_reflection(
        self, query: str, current_shopping_intent: str, integrated_intent: str
    ) -> dict[str, str]:
        prompt = render_prompt(
            INTENT_REFLECTION_PROMPT,
            query=query,
            current_shopping_intent=current_shopping_intent,
            integrated_intent=integrated_intent,
        )
        return self._chat_json(prompt, DEFAULTS["intent_reflection"])

    def _memory_preference_reasoning(
        self, query: str, used_intent: str, previous_memory: list[dict[str, Any]]
    ) -> dict[str, str]:
        previous_preferences = [
            {
                "turn_id": item.get("turn_id"),
                "query": item.get("query", ""),
                "new_preferences": item.get("new_preferences", INVALID),
            }
            for item in previous_memory
        ]
        reasoning_prompt = render_prompt(
            MEMORY_PREFERENCE_REASONING_PROMPT,
            query=query,
            used_intent=used_intent,
            previous_memory_preferences=json.dumps(
                previous_preferences, ensure_ascii=False, indent=2
            ),
        )
        first_stage = self._chat_json(
            reasoning_prompt, DEFAULTS["memory_preference_reasoning"]
        )
        decision_prompt = render_prompt(
            MEMORY_PREFERENCE_DECISION_PROMPT,
            query=query,
            used_intent=used_intent,
            previous_memory_preferences=json.dumps(
                previous_preferences, ensure_ascii=False, indent=2
            ),
            memory_preference_reasoning=first_stage.get("reasoning", INVALID),
        )
        return self._chat_json(decision_prompt, DEFAULTS["memory_preferences"])

    def _global_preference_reasoning(self, query: str, used_intent: str) -> dict[str, str]:
        prompt = render_prompt(
            GLOBAL_PREFERENCE_REASONING_PROMPT,
            query=query,
            used_intent=used_intent,
            user_profile=self.user_profile or INVALID,
        )
        return self._chat_json(prompt, DEFAULTS["global_preferences"])

    def _profile_rewrite(
        self,
        query: str,
        used_intent: str,
        new_preferences: str,
        related_memory_preferences: str,
        related_global_preferences: str,
    ) -> dict[str, str]:
        prompt = render_prompt(
            PROFILE_REWRITE_PROMPT,
            query=query,
            used_intent=used_intent,
            new_preferences=new_preferences,
            related_memory_preferences=related_memory_preferences,
            related_global_preferences=related_global_preferences,
        )
        return self._chat_json(prompt, DEFAULTS["profile_rewrite"])

    def _profile_reflection(
        self,
        query: str,
        used_intent: str,
        new_preferences: str,
        related_memory_preferences: str,
        related_global_preferences: str,
        rewritten_profile: str,
    ) -> dict[str, str]:
        prompt = render_prompt(
            PROFILE_REFLECTION_PROMPT,
            query=query,
            used_intent=used_intent,
            new_preferences=new_preferences,
            related_memory_preferences=related_memory_preferences,
            related_global_preferences=related_global_preferences,
            rewritten_profile=rewritten_profile,
        )
        return self._chat_json(prompt, DEFAULTS["profile_reflection"])

    @staticmethod
    def _is_effective_text(value: str | None, invalid_value: str) -> bool:
        if value is None:
            return False
        return value.strip().lower() != invalid_value.lower() and bool(value.strip())


def parse_llm_output(raw_output: str, defaults: dict[str, str]) -> dict[str, str]:
    """Parse a model response into expected fields without making judgments.

    The parser first attempts JSON extraction. If the model returns surrounding
    text, it extracts the first JSON object. If JSON still fails, it falls back
    to label-based field extraction. Missing fields are filled from ``defaults``.
    """

    result = defaults.copy()
    data: Any | None = None
    raw_output = raw_output.strip()

    try:
        data = json.loads(raw_output)
    except json.JSONDecodeError:
        json_match = re.search(r"\{.*\}", raw_output, flags=re.DOTALL)
        if json_match:
            try:
                data = json.loads(json_match.group(0))
            except json.JSONDecodeError:
                data = None

    if isinstance(data, dict):
        for key in defaults:
            if key in data and data[key] is not None:
                result[key] = _stringify_field(data[key])
        return result

    for key in defaults:
        pattern = rf'(?m)["\']?{re.escape(key)}["\']?\s*[:：]\s*["\']?([^,"\'\n}}]+)'
        match = re.search(pattern, raw_output)
        if match:
            result[key] = match.group(1).strip()
    return result


def _stringify_field(value: Any) -> str:
    if isinstance(value, str):
        return value.strip() or INVALID
    if isinstance(value, (list, dict)):
        return json.dumps(value, ensure_ascii=False)
    return str(value).strip() or INVALID
