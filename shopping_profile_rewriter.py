"""Shopping intent analysis, session memory, and profile rewriting.

This module intentionally exposes a deterministic rule-based implementation while
keeping the analyze / infer / rewrite steps small and replaceable.  A later LLM
implementation can call the same public functions or replace the helper hooks
marked with TODO comments.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
import argparse
import json
from pathlib import Path
import re
from typing import Any, Callable, Iterable, Literal, NotRequired, TypedDict

INVALID: Literal["INVALID"] = "INVALID"
IRRELEVANT: Literal["irrelevant"] = "irrelevant"
Invalid = Literal["INVALID"]
Irrelevant = Literal["irrelevant"]
StringListOrInvalid = list[str] | Invalid


class MemoryEntry(TypedDict):
    turn_id: int
    query: str
    current_intents: list[str]
    new_preferences: StringListOrInvalid


class QueryAnalysis(TypedDict):
    current_intents: list[str]
    introduces_new_preferences: bool
    new_preferences: StringListOrInvalid


class IntentIntegration(TypedDict):
    has_strong_related_memory_intent: bool
    strongly_related_memory_intents: StringListOrInvalid
    integrated_intent: str | Irrelevant


class Reflection(TypedDict):
    need_reflection: bool
    has_invalid_info: bool
    is_irrelevant_to_current_query: bool
    reflected_intent: str | Irrelevant | Invalid


class MemoryPreferenceReasoning(TypedDict):
    has_related_memory_preferences: bool
    related_memory_preferences: StringListOrInvalid


class GlobalPreferenceReasoning(TypedDict):
    has_related_global_preferences: bool
    related_global_preferences: StringListOrInvalid


class ProcessInput(TypedDict):
    query: str
    global_profile: NotRequired[str | object]
    session_memory: NotRequired[list[MemoryEntry]]
    log_path: NotRequired[str | Path]


class ProcessLogRecord(TypedDict):
    timestamp: str
    query: str
    previous_session_memory: list[MemoryEntry]
    process_result: "ProcessResult"


class ProcessResult(TypedDict):
    query_analysis: QueryAnalysis
    intent_integration: IntentIntegration
    reflection: Reflection
    final_current_intent: str | list[str]
    memory_preference_reasoning: MemoryPreferenceReasoning
    global_preference_reasoning: GlobalPreferenceReasoning
    rewritten_profile: str
    updated_session_memory: list[MemoryEntry]


@dataclass(frozen=True)
class ProductRule:
    category: str
    aliases: tuple[str, ...]


PRODUCT_RULES: tuple[ProductRule, ...] = (
    ProductRule("双肩包", ("双肩包", "背包", "书包", "电脑包", "通勤包")),
    ProductRule("跑鞋", ("跑鞋", "运动鞋", "慢跑鞋", "跑步鞋")),
    ProductRule("手机", ("手机", "拍照手机", "iphone", "安卓机")),
    ProductRule("咖啡机", ("咖啡机", "意式机", "胶囊咖啡机")),
    ProductRule("办公椅", ("办公椅", "人体工学椅", "电脑椅")),
    ProductRule("旅行箱", ("旅行箱", "行李箱", "拉杆箱")),
    ProductRule("咖啡豆", ("咖啡豆",)),
    ProductRule("耳机", ("耳机", "蓝牙耳机", "降噪耳机")),
    ProductRule("礼物", ("礼物", "送礼", "生日礼物")),
    ProductRule("户外徒步装备", ("户外", "徒步", "登山", "露营")),
)

SHOPPING_VERBS = ("买", "购买", "想要", "想找", "寻找", "推荐", "选", "挑", "入手", "比较", "换")
DEPENDENT_CUES = ("最好", "还能", "还要", "也要", "再", "另外", "它", "这个", "上一款", "刚才")

PREFERENCE_PATTERNS: tuple[tuple[str, tuple[str, ...]], ...] = (
    ("通勤场景", ("通勤", "上班", "日常办公")),
    ("大容量", ("大容量", "容量大", "能装", "装得下")),
    ("简洁外观", ("简洁", "简约", "低调", "素一点")),
    ("可放下16寸电脑", ("16 寸电脑", "16寸电脑", "16 英寸电脑", "16英寸电脑")),
    ("适合拍照", ("拍照", "影像", "拍摄", "摄影")),
    ("轻便", ("轻便", "轻量", "轻")),
    ("缓震", ("缓震", "减震")),
    ("防水", ("防水", "防泼水")),
    ("耐用", ("耐用", "结实", "耐磨")),
    ("人体工学支撑", ("人体工学", "腰托", "支撑")),
    ("预算限制", ("预算", "以内", "以下", "不超过")),
)

# Preferences that can be reused across categories only when they are not bound
# to an unrelated product / scenario clause in the source profile.
GENERIC_PREFERENCES = {"耐用", "简洁外观", "预算限制"}


def _normalize_text(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, str):
        return value
    return json.dumps(value, ensure_ascii=False, sort_keys=True)


def _dedupe(items: Iterable[str]) -> list[str]:
    result: list[str] = []
    seen: set[str] = set()
    for item in items:
        normalized = item.strip()
        if normalized and normalized not in seen:
            result.append(normalized)
            seen.add(normalized)
    return result


def _extract_categories(text: str) -> list[str]:
    lowered = text.lower()
    categories: list[str] = []
    for rule in PRODUCT_RULES:
        if any(alias.lower() in lowered for alias in rule.aliases):
            categories.append(rule.category)
    return _dedupe(categories)


def _extract_preferences(text: str, *, require_shopping_related: bool = True) -> list[str]:
    if require_shopping_related and not _has_shopping_signal(text):
        # A dependent shopping turn may omit a verb, e.g. "最好还能放下 16 寸电脑".
        if not any(cue in text for cue in DEPENDENT_CUES):
            return []
    preferences: list[str] = []
    for preference, aliases in PREFERENCE_PATTERNS:
        if any(alias in text for alias in aliases):
            preferences.append(preference)
    budget_matches = re.findall(r"(?:预算)?\s*(\d{2,6})\s*(?:元|块)?\s*(?:以内|以下|左右)?", text)
    for amount in budget_matches:
        if any(marker in text for marker in ("预算", "以内", "以下", "不超过", "左右")):
            preferences.append(f"预算约{amount}元")
    return _dedupe(preferences)


def _has_shopping_signal(query: str) -> bool:
    return any(verb in query for verb in SHOPPING_VERBS) or bool(_extract_categories(query))


def _is_invalid_intent(intents: list[str]) -> bool:
    return intents == [INVALID] or not intents


def _intent_text(final_current_intent: str | list[str]) -> str:
    if isinstance(final_current_intent, list):
        return "、".join(final_current_intent)
    return final_current_intent


def _intent_categories(intent: str | list[str]) -> list[str]:
    return _extract_categories(_intent_text(intent))


def _shares_category(intent_a: str | list[str], intent_b: str | list[str]) -> bool:
    cats_a = set(_intent_categories(intent_a))
    cats_b = set(_intent_categories(intent_b))
    return bool(cats_a and cats_b and cats_a.intersection(cats_b))


def _is_dependent_intent(intent: str | list[str]) -> bool:
    text = _intent_text(intent)
    return any(cue in text for cue in DEPENDENT_CUES) or bool(re.search(r"\d+\s*(?:寸|英寸)", text))


def _format_preference_list(values: StringListOrInvalid) -> str:
    if values == INVALID or not values:
        return "无"
    return "、".join(values)


def analyzeQuery(query: str) -> QueryAnalysis:
    """Analyze the raw query into structured shopping intent and new preferences.

    TODO: Replace this rule-based implementation with an LLM prompt when a
    production model client is available.  The output schema should stay stable.
    """
    normalized_query = (query or "").strip()
    categories = _extract_categories(normalized_query)
    has_shopping_signal = any(verb in normalized_query for verb in SHOPPING_VERBS) or bool(categories)
    preferences = _extract_preferences(normalized_query, require_shopping_related=True)

    if not has_shopping_signal and not any(cue in normalized_query for cue in DEPENDENT_CUES):
        return {
            "current_intents": [INVALID],
            "introduces_new_preferences": False,
            "new_preferences": INVALID,
        }

    intents: list[str] = []
    if categories:
        for category in categories:
            modifiers = [pref for pref in preferences if pref not in {"预算限制"}]
            if modifiers:
                intents.append(f"购买/寻找{_format_preference_list(modifiers)}的{category}")
            else:
                intents.append(f"购买/寻找{category}")
    elif preferences and any(cue in normalized_query for cue in DEPENDENT_CUES):
        intents.append(f"补充当前购物任务需求：{_format_preference_list(preferences)}")
    else:
        intents.append("当前购物需求待结合上下文确认")

    new_preferences: StringListOrInvalid = preferences if preferences else INVALID
    return {
        "current_intents": _dedupe(intents),
        "introduces_new_preferences": new_preferences != INVALID,
        "new_preferences": new_preferences,
    }


def appendToMemory(
    session_memory: list[MemoryEntry] | None,
    query: str,
    query_analysis: QueryAnalysis,
) -> list[MemoryEntry]:
    previous = list(session_memory or [])
    previous.append(
        {
            "turn_id": len(previous) + 1,
            "query": query,
            "current_intents": list(query_analysis["current_intents"]),
            "new_preferences": query_analysis["new_preferences"],
        }
    )
    return previous


def integrateIntent(current_intents: list[str], previous_memory: list[MemoryEntry]) -> IntentIntegration:
    if _is_invalid_intent(current_intents) or not previous_memory:
        return {
            "has_strong_related_memory_intent": False,
            "strongly_related_memory_intents": INVALID,
            "integrated_intent": IRRELEVANT,
        }

    related_intents: list[str] = []
    related_preferences: list[str] = []
    current_text = _intent_text(current_intents)
    dependent = _is_dependent_intent(current_intents)

    for entry in previous_memory:
        memory_intents = entry.get("current_intents", [])
        if _is_invalid_intent(memory_intents):
            continue
        same_category = _shares_category(current_intents, memory_intents)
        current_lacks_category = not _intent_categories(current_intents)
        if same_category or (dependent and current_lacks_category and _intent_categories(memory_intents)):
            related_intents.extend(memory_intents)
            prefs = entry.get("new_preferences", INVALID)
            if prefs != INVALID:
                related_preferences.extend(prefs)

    related_intents = _dedupe(related_intents)
    if not related_intents:
        return {
            "has_strong_related_memory_intent": False,
            "strongly_related_memory_intents": INVALID,
            "integrated_intent": IRRELEVANT,
        }

    categories = _intent_categories(related_intents) or _intent_categories(current_intents)
    all_preferences = _dedupe(related_preferences + _extract_preferences(current_text, require_shopping_related=False))
    if categories:
        integrated = f"寻找{_format_preference_list(all_preferences)}的{categories[0]}"
    else:
        integrated = f"围绕同一购物任务补充需求：{_format_preference_list(all_preferences)}"

    return {
        "has_strong_related_memory_intent": True,
        "strongly_related_memory_intents": related_intents,
        "integrated_intent": integrated,
    }


def reflectIntegratedIntent(
    query: str,
    current_intents: list[str],
    integrated_intent: str | Irrelevant,
) -> tuple[Reflection, str | list[str]]:
    if integrated_intent == IRRELEVANT:
        return (
            {
                "need_reflection": False,
                "has_invalid_info": False,
                "is_irrelevant_to_current_query": False,
                "reflected_intent": INVALID,
            },
            current_intents,
        )

    if _is_invalid_intent(current_intents):
        return (
            {
                "need_reflection": True,
                "has_invalid_info": True,
                "is_irrelevant_to_current_query": True,
                "reflected_intent": IRRELEVANT,
            },
            current_intents,
        )

    current_categories = set(_intent_categories(current_intents))
    integrated_categories = set(_intent_categories(integrated_intent))
    dependent = _is_dependent_intent(current_intents)
    irrelevant = bool(current_categories and integrated_categories and not current_categories.intersection(integrated_categories))
    if irrelevant and not dependent:
        return (
            {
                "need_reflection": True,
                "has_invalid_info": True,
                "is_irrelevant_to_current_query": True,
                "reflected_intent": IRRELEVANT,
            },
            current_intents,
        )

    return (
        {
            "need_reflection": True,
            "has_invalid_info": False,
            "is_irrelevant_to_current_query": False,
            "reflected_intent": integrated_intent,
        },
        integrated_intent,
    )


def inferMemoryPreferences(
    final_current_intent: str | list[str],
    previous_memory: list[MemoryEntry],
) -> MemoryPreferenceReasoning:
    if _intent_text(final_current_intent) == INVALID or not previous_memory:
        return {"has_related_memory_preferences": False, "related_memory_preferences": INVALID}

    preferences: list[str] = []
    final_categories = set(_intent_categories(final_current_intent))
    for entry in previous_memory:
        entry_preferences = entry.get("new_preferences", INVALID)
        if entry_preferences == INVALID:
            continue
        if final_categories and final_categories.intersection(_intent_categories(entry.get("current_intents", []))):
            preferences.extend(entry_preferences)
        elif not final_categories and _is_dependent_intent(final_current_intent):
            preferences.extend(entry_preferences)

    preferences = _dedupe(preferences)
    return {
        "has_related_memory_preferences": bool(preferences),
        "related_memory_preferences": preferences if preferences else INVALID,
    }


def _split_profile_clauses(profile_text: str) -> list[str]:
    return [part.strip() for part in re.split(r"[。；;\n,，]", profile_text) if part.strip()]


def inferGlobalPreferences(
    final_current_intent: str | list[str],
    global_profile: str | object | None,
) -> GlobalPreferenceReasoning:
    if _intent_text(final_current_intent) == INVALID:
        return {"has_related_global_preferences": False, "related_global_preferences": INVALID}

    profile_text = _normalize_text(global_profile)
    if not profile_text.strip():
        return {"has_related_global_preferences": False, "related_global_preferences": INVALID}

    final_categories = set(_intent_categories(final_current_intent))
    related_preferences: list[str] = []
    for clause in _split_profile_clauses(profile_text):
        clause_categories = set(_extract_categories(clause))
        clause_preferences = _extract_preferences(clause, require_shopping_related=False)
        if not clause_preferences:
            continue
        if final_categories and clause_categories.intersection(final_categories):
            related_preferences.extend(clause_preferences)
        elif not clause_categories:
            related_preferences.extend(pref for pref in clause_preferences if pref in GENERIC_PREFERENCES)

    related_preferences = _dedupe(related_preferences)
    return {
        "has_related_global_preferences": bool(related_preferences),
        "related_global_preferences": related_preferences if related_preferences else INVALID,
    }


def rewriteProfile(
    payload: dict[str, str | list[str] | StringListOrInvalid],
) -> str:
    final_current_intent = payload.get("final_current_intent", INVALID)
    query_new_preferences = payload.get("query_new_preferences", INVALID)
    related_memory_preferences = payload.get("related_memory_preferences", INVALID)
    related_global_preferences = payload.get("related_global_preferences", INVALID)

    intent_text = _intent_text(final_current_intent)  # type: ignore[arg-type]
    if intent_text == INVALID:
        return "当前 query 无明确购物推荐意图，无法生成面向推荐任务的有效用户画像。"

    sections = [f"用户当前购物意图：{intent_text}。"]
    if query_new_preferences != INVALID:
        sections.append(f"当前 query 新引入偏好：{_format_preference_list(query_new_preferences)}。")  # type: ignore[arg-type]
    if related_memory_preferences != INVALID:
        sections.append(f"相关会话历史偏好：{_format_preference_list(related_memory_preferences)}。")  # type: ignore[arg-type]
    if related_global_preferences != INVALID:
        sections.append(f"相关长期画像偏好：{_format_preference_list(related_global_preferences)}。")  # type: ignore[arg-type]

    if len(sections) == 1:
        sections.append("未发现可用于当前任务的 query 新偏好、会话历史偏好或长期画像偏好。")
    else:
        all_preferences = _dedupe(
            ([] if query_new_preferences == INVALID else list(query_new_preferences))
            + ([] if related_memory_preferences == INVALID else list(related_memory_preferences))
            + ([] if related_global_preferences == INVALID else list(related_global_preferences))
        )
        sections.append(f"推荐召回/排序应优先匹配：{_format_preference_list(all_preferences)}。")
    return "".join(sections)


def saveProcessLog(
    *,
    log_path: str | Path,
    query: str,
    previous_session_memory: list[MemoryEntry],
    process_result: ProcessResult,
) -> None:
    """Append one JSONL record containing the full structured intermediate result.

    The saved record is intentionally structured, not free-form reasoning: it
    includes the input query, the pre-turn memory snapshot, and the full
    ProcessResult.  This makes every intermediate step auditable while keeping
    the in-memory session memory limited to raw per-turn query analysis.
    """
    path = Path(log_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    record: ProcessLogRecord = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "query": query,
        "previous_session_memory": previous_session_memory,
        "process_result": process_result,
    }
    with path.open("a", encoding="utf-8") as log_file:
        log_file.write(json.dumps(record, ensure_ascii=False) + "\n")


def processShoppingQuery(input: ProcessInput) -> ProcessResult:
    query = input.get("query", "")
    global_profile = input.get("global_profile", "")
    previous_memory = list(input.get("session_memory", []) or [])
    log_path = input.get("log_path")

    query_analysis = analyzeQuery(query)
    updated_session_memory = appendToMemory(previous_memory, query, query_analysis)

    if _is_invalid_intent(query_analysis["current_intents"]):
        intent_integration: IntentIntegration = {
            "has_strong_related_memory_intent": False,
            "strongly_related_memory_intents": INVALID,
            "integrated_intent": IRRELEVANT,
        }
        reflection: Reflection = {
            "need_reflection": False,
            "has_invalid_info": False,
            "is_irrelevant_to_current_query": False,
            "reflected_intent": INVALID,
        }
        final_current_intent: str | list[str] = query_analysis["current_intents"]
        memory_preference_reasoning: MemoryPreferenceReasoning = {
            "has_related_memory_preferences": False,
            "related_memory_preferences": INVALID,
        }
        global_preference_reasoning: GlobalPreferenceReasoning = {
            "has_related_global_preferences": False,
            "related_global_preferences": INVALID,
        }
    elif not previous_memory:
        intent_integration = {
            "has_strong_related_memory_intent": False,
            "strongly_related_memory_intents": INVALID,
            "integrated_intent": IRRELEVANT,
        }
        reflection = {
            "need_reflection": False,
            "has_invalid_info": False,
            "is_irrelevant_to_current_query": False,
            "reflected_intent": INVALID,
        }
        final_current_intent = query_analysis["current_intents"]
        memory_preference_reasoning = {
            "has_related_memory_preferences": False,
            "related_memory_preferences": INVALID,
        }
        global_preference_reasoning = inferGlobalPreferences(final_current_intent, global_profile)
    else:
        intent_integration = integrateIntent(query_analysis["current_intents"], previous_memory)
        reflection, final_current_intent = reflectIntegratedIntent(
            query, query_analysis["current_intents"], intent_integration["integrated_intent"]
        )
        memory_preference_reasoning = inferMemoryPreferences(final_current_intent, previous_memory)
        global_preference_reasoning = inferGlobalPreferences(final_current_intent, global_profile)

    rewritten_profile = rewriteProfile(
        {
            "final_current_intent": final_current_intent,
            "query_new_preferences": query_analysis["new_preferences"],
            "related_memory_preferences": memory_preference_reasoning["related_memory_preferences"],
            "related_global_preferences": global_preference_reasoning["related_global_preferences"],
        }
    )

    result: ProcessResult = {
        "query_analysis": query_analysis,
        "intent_integration": intent_integration,
        "reflection": reflection,
        "final_current_intent": final_current_intent,
        "memory_preference_reasoning": memory_preference_reasoning,
        "global_preference_reasoning": global_preference_reasoning,
        "rewritten_profile": rewritten_profile,
        "updated_session_memory": updated_session_memory,
    }
    if log_path is not None:
        saveProcessLog(
            log_path=log_path,
            query=query,
            previous_session_memory=previous_memory,
            process_result=result,
        )
    return result


def _demo() -> None:
    parser = argparse.ArgumentParser(description="Run the shopping profile rewriter demo.")
    parser.add_argument(
        "--log-path",
        default=None,
        help="Optional JSONL path for saving structured intermediate results.",
    )
    args = parser.parse_args()

    global_profile = "用户购买双肩包时偏好耐用、低调设计；用户喜欢户外徒步装备，偏好防水和轻量。"
    memory: list[MemoryEntry] = []
    demo_queries = [
        "想买一个适合通勤的大容量双肩包，外观简洁一点",
        "最好还能放下 16 寸电脑",
    ]
    for query in demo_queries:
        result = processShoppingQuery(
            {
                "query": query,
                "global_profile": global_profile,
                "session_memory": memory,
                **({"log_path": args.log_path} if args.log_path else {}),
            }
        )
        memory = result["updated_session_memory"]
        print(json.dumps(result, ensure_ascii=False, indent=2))


# Optional dependency injection point for future prompt-based implementations.
ShoppingStep = Callable[..., dict[str, Any] | str]


if __name__ == "__main__":
    _demo()
