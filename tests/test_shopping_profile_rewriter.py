import json
from pathlib import Path
import tempfile
import unittest

from shopping_profile_rewriter import INVALID, IRRELEVANT, processShoppingQuery


class ShoppingProfileRewriterTest(unittest.TestCase):
    def test_first_turn_with_query_preferences_and_related_global_profile(self):
        result = processShoppingQuery(
            {
                "query": "想买一个适合通勤的大容量双肩包，外观简洁一点",
                "global_profile": "用户购买双肩包时偏好耐用、低调设计；也喜欢咖啡豆。",
                "session_memory": [],
            }
        )

        self.assertTrue(any("双肩包" in intent and "通勤" in intent for intent in result["query_analysis"]["current_intents"]))
        self.assertTrue(result["query_analysis"]["introduces_new_preferences"])
        self.assertIn("通勤场景", result["query_analysis"]["new_preferences"])
        self.assertIn("大容量", result["query_analysis"]["new_preferences"])
        self.assertIn("简洁外观", result["query_analysis"]["new_preferences"])
        self.assertEqual(result["intent_integration"]["integrated_intent"], IRRELEVANT)
        self.assertFalse(result["reflection"]["need_reflection"])
        self.assertIn("耐用", result["global_preference_reasoning"]["related_global_preferences"])
        self.assertIn("当前 query 新引入偏好", result["rewritten_profile"])
        self.assertEqual(len(result["updated_session_memory"]), 1)

    def test_second_turn_strongly_related_to_backpack_task(self):
        first = processShoppingQuery(
            {
                "query": "想买一个适合通勤的大容量双肩包，外观简洁一点",
                "global_profile": "用户购买双肩包时偏好耐用。",
                "session_memory": [],
            }
        )
        second = processShoppingQuery(
            {
                "query": "最好还能放下 16 寸电脑",
                "global_profile": "用户购买双肩包时偏好耐用。",
                "session_memory": first["updated_session_memory"],
            }
        )

        self.assertTrue(second["intent_integration"]["has_strong_related_memory_intent"])
        self.assertIn("双肩包", second["intent_integration"]["integrated_intent"])
        self.assertIn("通勤场景", second["intent_integration"]["integrated_intent"])
        self.assertIn("大容量", second["intent_integration"]["integrated_intent"])
        self.assertIn("简洁外观", second["intent_integration"]["integrated_intent"])
        self.assertIn("可放下16寸电脑", second["intent_integration"]["integrated_intent"])
        self.assertIn("通勤场景", second["memory_preference_reasoning"]["related_memory_preferences"])
        self.assertIn("相关会话历史偏好", second["rewritten_profile"])
        self.assertEqual(len(second["updated_session_memory"]), 2)
        self.assertEqual(second["updated_session_memory"][-1]["new_preferences"], ["可放下16寸电脑"])

    def test_second_turn_unrelated_intent_does_not_use_running_memory(self):
        previous = processShoppingQuery({"query": "想买跑鞋", "global_profile": "", "session_memory": []})
        current = processShoppingQuery(
            {
                "query": "推荐一款适合拍照的手机",
                "global_profile": "",
                "session_memory": previous["updated_session_memory"],
            }
        )

        self.assertEqual(current["intent_integration"]["integrated_intent"], IRRELEVANT)
        self.assertTrue(any("手机" in intent for intent in current["final_current_intent"]))
        self.assertEqual(current["memory_preference_reasoning"]["related_memory_preferences"], INVALID)
        self.assertNotIn("跑鞋", current["rewritten_profile"])

    def test_memory_preferences_unrelated_to_current_intent_are_invalid(self):
        memory = [
            {
                "turn_id": 1,
                "query": "想买跑鞋，轻便缓震",
                "current_intents": ["购买/寻找轻便、缓震的跑鞋"],
                "new_preferences": ["轻便", "缓震"],
            }
        ]
        result = processShoppingQuery({"query": "想买咖啡机", "global_profile": "", "session_memory": memory})

        self.assertEqual(result["memory_preference_reasoning"]["related_memory_preferences"], INVALID)
        self.assertNotIn("轻便", result["rewritten_profile"])
        self.assertNotIn("缓震", result["rewritten_profile"])

    def test_global_profile_unrelated_to_current_intent_is_invalid(self):
        result = processShoppingQuery(
            {
                "query": "想买办公椅",
                "global_profile": "用户喜欢户外徒步装备，偏好防水和轻量",
                "session_memory": [],
            }
        )

        self.assertEqual(result["global_preference_reasoning"]["related_global_preferences"], INVALID)
        self.assertNotIn("户外", result["rewritten_profile"])
        self.assertNotIn("防水", result["rewritten_profile"])
        self.assertNotIn("轻量", result["rewritten_profile"])

    def test_non_shopping_query_returns_invalid_schema(self):
        result = processShoppingQuery({"query": "今天天气怎么样", "global_profile": "偏好耐用", "session_memory": []})

        self.assertEqual(result["query_analysis"]["current_intents"], [INVALID])
        self.assertEqual(result["query_analysis"]["new_preferences"], INVALID)
        self.assertEqual(result["memory_preference_reasoning"]["related_memory_preferences"], INVALID)
        self.assertEqual(result["global_preference_reasoning"]["related_global_preferences"], INVALID)
        self.assertIn("无明确购物推荐意图", result["rewritten_profile"])

    def test_process_query_can_save_structured_jsonl_log(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            log_path = Path(temp_dir) / "shopping_process.jsonl"
            result = processShoppingQuery(
                {
                    "query": "想买一个适合通勤的大容量双肩包，外观简洁一点",
                    "global_profile": "用户购买双肩包时偏好耐用。",
                    "session_memory": [],
                    "log_path": log_path,
                }
            )

            lines = log_path.read_text(encoding="utf-8").splitlines()
            self.assertEqual(len(lines), 1)
            record = json.loads(lines[0])
            self.assertEqual(record["query"], "想买一个适合通勤的大容量双肩包，外观简洁一点")
            self.assertEqual(record["previous_session_memory"], [])
            self.assertEqual(record["process_result"]["rewritten_profile"], result["rewritten_profile"])
            self.assertIn("query_analysis", record["process_result"])
            self.assertIn("intent_integration", record["process_result"])


if __name__ == "__main__":
    unittest.main()
