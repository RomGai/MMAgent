"""Prompt templates for Memory-aware Profile Rewriting.

Every template asks Qwen3-8B to perform the semantic judgment and to return a
strict JSON object. Downstream code only parses and normalizes these fields; it
does not replace the model's intent, preference, relevance, reflection, or
profile-rewriting decisions with rule logic.
"""

COMMON_REQUIREMENTS = """
你是电商推荐系统中的用户意图与偏好分析模块。
请严格基于输入信息完成判断，不要编造输入中不存在的信息。
不要用规则模板代替语义判断，必须进行语义理解和相关性判断。
如果没有相关信息，请输出 INVALID。
如果意图整合或反思结果无效，请输出 irrelevant。
请严格输出 JSON，不要输出 Markdown，不要输出额外解释。
""".strip()

QUERY_ANALYSIS_FIRST_TURN_PROMPT = """
{common_requirements}

### Query分析（第一轮）###
输入：
- 当前 query：{query}
- 全局用户画像 user_profile：{user_profile}

任务：
a. 基于 query 分析当前购物意图，可能不止一种意图。
b. 判断 query 是否引入新的画像/偏好。
c. 若有，则输出新引入的偏好；若没有，输出 INVALID。
d. 判断全局用户画像中是否有与当前购物意图相关的画像/偏好信息。
e. 若有，则输出具体相关全局历史偏好；若没有，输出 INVALID。

输出 JSON schema：
{{
  "current_shopping_intent": "...",
  "has_new_preference": "yes/no",
  "new_preferences": "... or INVALID",
  "has_related_global_preference": "yes/no",
  "related_global_preferences": "... or INVALID"
}}
""".strip()

QUERY_ANALYSIS_FOLLOWUP_TURN_PROMPT = """
{common_requirements}

### Query分析（第二轮及后续）###
输入：
- 当前 query：{query}

任务：
a. 基于 query 分析当前购物意图，可能不止一种意图。
b. 判断 query 是否引入新的画像/偏好。
c. 若有，则输出新引入的偏好；若没有，输出 INVALID。

输出 JSON schema：
{{
  "current_shopping_intent": "...",
  "has_new_preference": "yes/no",
  "new_preferences": "... or INVALID"
}}
""".strip()

INTENT_INTEGRATION_PROMPT = """
{common_requirements}

### LLM意图整合 ###
输入：
- 当前 query：{query}
- 当前 Query 分析得到的购物意图：{current_shopping_intent}
- 当前轮之前的 session memory：{previous_memory}

任务：
a. 判断当前购物意图是否与 memory 中之前轮次的购物意图强相关。
b. 若强相关：总结强相关历史购物意图，并基于当前购物意图与强相关历史购物意图，推理当前用户更整体性的购物意图。
c. 若不强相关，related_memory_intents 和 integrated_intent 都输出 irrelevant。

输出 JSON schema：
{{
  "is_related_to_memory": "yes/no",
  "related_memory_intents": "... or irrelevant",
  "integrated_intent": "... or irrelevant"
}}
""".strip()

INTENT_REFLECTION_PROMPT = """
{common_requirements}

### LLM反思 ###
输入：
- 当前 query：{query}
- Query 分析得到的当前购物意图：{current_shopping_intent}
- 意图整合得到的整体购物意图：{integrated_intent}

任务：
a. 判断整合后的意图是否引入了对当前 query 无效的信息，或者是否已经与当前 query 关系很弱。
b. 若引入无效信息，则输出去除无效信息后的意图。
c. 若整合后的意图与当前 query 基本无关，则 refined_intent 输出 irrelevant。
d. 若整合后的意图没有问题，则 refined_intent 直接输出整合后的意图。

输出 JSON schema：
{{
  "has_invalid_information": "yes/no",
  "is_irrelevant_to_current_query": "yes/no",
  "refined_intent": "... or irrelevant"
}}
""".strip()

MEMORY_PREFERENCE_REASONING_PROMPT = """
{common_requirements}

### LLM基于Memory的偏好推理 ###
输入：
- 当前 query：{query}
- 用户当下购物意图：{used_intent}
- 当前轮之前 session memory 中的新引入偏好：{previous_memory_preferences}

任务：
a. 判断 memory 中保存的之前新引入偏好，是否有与当前购物意图相关的信息。
b. 若有，则只输出当前交互中与购物意图相关的历史偏好信息。
c. 若没有，related_memory_preferences 输出 INVALID。

输出 JSON schema：
{{
  "has_related_memory_preferences": "yes/no",
  "related_memory_preferences": "... or INVALID"
}}
""".strip()

GLOBAL_PREFERENCE_REASONING_PROMPT = """
{common_requirements}

### LLM全局历史偏好推理 ###
输入：
- 当前 query：{query}
- 用户当下购物意图：{used_intent}
- 全局用户画像 user_profile：{user_profile}

任务：
a. 判断用户全局画像中是否有与当前购物意图相关的画像/偏好信息。
b. 若有，则只输出具体相关全局历史偏好。
c. 若没有，related_global_preferences 输出 INVALID。

输出 JSON schema：
{{
  "has_related_global_preferences": "yes/no",
  "related_global_preferences": "... or INVALID"
}}
""".strip()

PROFILE_REWRITE_PROMPT = """
{common_requirements}

### Profile重写 ###
请基于以下信息，重写一个专门用于当前 query 推荐的用户 profile：
1. 当前 query：{query}
2. 用户当下购物意图：{used_intent}
3. query 新引入的偏好：{new_preferences}
4. 当前交互 memory 中相关的历史偏好：{related_memory_preferences}
5. 全局用户画像中相关的历史偏好：{related_global_preferences}

重写要求：
- 只保留与当前 query 推荐强相关的信息。
- 优先体现 query 中的新偏好。
- 其次使用当前交互 memory 中相关历史偏好。
- 再使用全局 user_profile 中相关历史偏好。
- 删除无关、过时、冲突或对当前推荐无帮助的信息。
- 如果 query 新偏好与历史偏好冲突，以 query 新偏好为准。
- 对购物意图、query 新偏好、当前交互 memory 历史偏好、全局 user_profile 历史偏好中语义重复或表述相近的信息进行去重与有机整合，避免在 rewritten_profile 中重复堆叠同一偏好。
- 将整合后的偏好组织成一段完整、连贯、自然语言的用户偏好描述，信息要完整准确；不要为了简短而遗漏对当前推荐有用的细节。
- reasoning 字段需要显式说明整合和取舍过程，包括保留了哪些信息、合并了哪些重复信息、删除了哪些无关/冲突信息以及冲突时为何采用 query 新偏好。
- 如果没有任何可用偏好，也要输出一个最小可用 profile，说明当前主要购物意图，并在 reasoning 中说明原因。

输出 JSON schema：
{{
  "reasoning": "...",
  "rewritten_profile": "..."
}}
""".strip()


def render_prompt(template: str, **kwargs: object) -> str:
    """Render a prompt with shared requirements injected."""

    return template.format(common_requirements=COMMON_REQUIREMENTS, **kwargs)
