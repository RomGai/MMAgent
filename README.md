# Shopping Profile Rewriter

本仓库提供一个「购物意图分析 + 会话 Memory + Profile 重写」模块。它接收当前购物 query、长期画像 `global_profile` 和当前会话 `session_memory`，返回结构化中间过程与最终 `rewritten_profile`。

## 快速使用

```python
from shopping_profile_rewriter import processShoppingQuery

memory = []
global_profile = "用户购买双肩包时偏好耐用、低调设计；用户喜欢户外徒步装备，偏好防水和轻量。"

first = processShoppingQuery(
    {
        "query": "想买一个适合通勤的大容量双肩包，外观简洁一点",
        "global_profile": global_profile,
        "session_memory": memory,
    }
)
print(first["rewritten_profile"])

# 下一轮必须把上一轮返回的 updated_session_memory 传回去。
second = processShoppingQuery(
    {
        "query": "最好还能放下 16 寸电脑",
        "global_profile": global_profile,
        "session_memory": first["updated_session_memory"],
    }
)
print(second["intent_integration"])
print(second["rewritten_profile"])
```

`processShoppingQuery` 返回完整结构化结果，包含：

- `query_analysis`：当前 query 的原始购物意图与新偏好。
- `intent_integration`：与历史会话意图的强相关整合结果。
- `reflection`：对整合意图的结构化校验/回退结果。
- `final_current_intent`：最终用于推荐任务的当前意图。
- `memory_preference_reasoning`：从历史 session memory 中筛出的相关偏好。
- `global_preference_reasoning`：从长期画像中筛出的相关偏好。
- `rewritten_profile`：可直接交给推荐召回/排序的当前画像。
- `updated_session_memory`：追加本轮原始 query 分析后的新 memory。

## 直接查看示例结果

运行内置 demo：

```bash
python shopping_profile_rewriter.py
```

它会连续处理两轮 query，并把每轮完整 JSON 结果打印到 stdout，便于查看 `rewritten_profile` 和所有中间字段。


## 示例结果片段

第一轮 query `想买一个适合通勤的大容量双肩包，外观简洁一点` 会得到类似结果：

```json
{
  "query_analysis": {
    "current_intents": ["购买/寻找通勤场景、大容量、简洁外观的双肩包"],
    "introduces_new_preferences": true,
    "new_preferences": ["通勤场景", "大容量", "简洁外观"]
  },
  "intent_integration": {
    "has_strong_related_memory_intent": false,
    "strongly_related_memory_intents": "INVALID",
    "integrated_intent": "irrelevant"
  },
  "rewritten_profile": "用户当前购物意图：购买/寻找通勤场景、大容量、简洁外观的双肩包。当前 query 新引入偏好：通勤场景、大容量、简洁外观。相关长期画像偏好：简洁外观、耐用。推荐召回/排序应优先匹配：通勤场景、大容量、简洁外观、耐用。"
}
```

第二轮 query `最好还能放下 16 寸电脑` 会和上一轮双肩包任务强相关，`intent_integration.integrated_intent` 会类似：

```text
寻找通勤场景、大容量、简洁外观、可放下16寸电脑的双肩包
```

## 保存结构化日志

如果想保存每次处理的输入、上一轮 memory 快照、完整中间过程和输出，可以传入 `log_path`：

```python
result = processShoppingQuery(
    {
        "query": "想买一个适合通勤的大容量双肩包，外观简洁一点",
        "global_profile": "用户购买双肩包时偏好耐用。",
        "session_memory": [],
        "log_path": "logs/shopping_profile_rewriter.jsonl",
    }
)
```

也可以在 demo 中保存日志：

```bash
python shopping_profile_rewriter.py --log-path logs/shopping_profile_rewriter.jsonl
```

日志格式为 JSONL，每一行是一轮处理记录：

```json
{
  "timestamp": "2026-05-11T00:00:00+00:00",
  "query": "想买一个适合通勤的大容量双肩包，外观简洁一点",
  "previous_session_memory": [],
  "process_result": {
    "query_analysis": {},
    "intent_integration": {},
    "reflection": {},
    "final_current_intent": [],
    "memory_preference_reasoning": {},
    "global_preference_reasoning": {},
    "rewritten_profile": "...",
    "updated_session_memory": []
  }
}
```

注意：日志中的 `process_result` 保存完整中间过程；但 `updated_session_memory` 本身仍严格只保存每轮原始 `current_intents` 和 `new_preferences`，不会把反思、整合或重写结果写入 memory。

## LLM 接入方式

当前 `shopping_profile_rewriter.py` 是规则实现，函数边界已经按步骤拆分，后续可把 `analyzeQuery`、`inferMemoryPreferences`、`inferGlobalPreferences` 或 `rewriteProfile` 替换为 prompt/LLM 实现。`qwen_3_chat_demo.py` 提供了可导入的 `QwenChatClient` 和 `chat_once`，导入时不会立即加载模型，只有实例化/调用时才会加载。
