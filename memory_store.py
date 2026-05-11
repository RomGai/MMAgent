"""Session-level memory for Memory-aware Profile Rewriting."""

from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass, field
from typing import Any


@dataclass
class SessionMemoryStore:
    """Append-only memory storing only query-analysis outputs per turn."""

    _items: list[dict[str, Any]] = field(default_factory=list)

    def add_turn(self, query: str, shopping_intent: str, new_preferences: str) -> dict[str, Any]:
        entry = {
            "turn_id": len(self._items) + 1,
            "query": query,
            "shopping_intent": shopping_intent,
            "new_preferences": new_preferences,
        }
        self._items.append(entry)
        return deepcopy(entry)

    def snapshot(self) -> list[dict[str, Any]]:
        return deepcopy(self._items)

    def previous_snapshot(self) -> list[dict[str, Any]]:
        return deepcopy(self._items[:-1])

    @property
    def items(self) -> list[dict[str, Any]]:
        return self.snapshot()

    def __len__(self) -> int:
        return len(self._items)
