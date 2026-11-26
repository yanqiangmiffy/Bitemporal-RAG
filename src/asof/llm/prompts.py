"""Prompts for the three LLM roles (Chinese-first; the corpus is mostly zh).

Roles (paper §Methodology / Update & Query paths):
  * EXTRACTOR  : notice text -> candidate field-level facts (+ dates, cited ids)
  * CLASSIFIER : (new fact, candidate old facts) -> revision type
  * GENERATOR  : grounded answer over the surviving (as-of filtered) facts
"""
from __future__ import annotations

EXTRACTOR_SYSTEM = (
    "你是一个法规/公告信息抽取器。给定一则通知的正文，抽取其中“字段级”的事实。"
    "每个事实描述某个对象(object)的某个字段(field)在某个适用条件(condition)下的取值(value)，"
    "并给出支撑该取值的原文片段(source_span)。同时抽取该通知的发布日期(publish_date)、"
    "生效日期(effective_date，若含“自X起施行/执行/新发放”等)，以及其引用或修改的文号/条款号(cited_ids)。"
    "只依据给定正文，不要臆测。严格输出 JSON。"
)

EXTRACTOR_USER = """通知标题：{title}
通知正文：
{text}

请输出 JSON，形如：
{{
  "publish_date": "YYYY-MM-DD 或 null",
  "effective_date": "YYYY-MM-DD 或 null",
  "cited_ids": ["被引用/被修改的文号或条款号", ...],
  "facts": [
    {{
      "object": "对象，如 首套个人住房公积金贷款",
      "field": "字段，如 利率 / 受理机关 / 最低首付比例",
      "value": "取值，如 2.6%",
      "condition": {{"维度名": "取值"}},   // 适用条件，如 {{"期限":"5年以上","放款日":">=2025-05-08"}}；无条件写 {{}}
      "source_span": "支撑该取值的原文片段（尽量精确、可在正文中检索到）",
      "clause_id": "条款号或 null"
    }}
  ]
}}"""

CLASSIFIER_SYSTEM = (
    "你是一个修订关系分类器。给定一条来自新通知的事实，以及若干条检索到的旧事实，"
    "判断新事实对每条旧事实构成的修订关系，取值之一："
    "insert（新增，与旧事实无冲突）、replace（整体替换旧值）、delete（废止旧值且无新值）、"
    "partial（仅对旧事实适用条件的一部分生效，其余保留）、unrelated（无关）。"
    "无法高置信判断时输出 to_verify。严格输出 JSON。"
)

CLASSIFIER_USER = """新事实：
{new_fact}

候选旧事实（按检索顺序）：
{candidates}

请输出 JSON，形如：
{{
  "relations": [
    {{"candidate_index": 0, "relation": "replace|insert|delete|partial|unrelated|to_verify",
      "covered_condition": {{"维度名":"取值"}} }}   // 仅 partial 时给出被新值覆盖的子条件，否则 {{}}
  ]
}}"""

GENERATOR_SYSTEM = (
    "你是一个基于证据作答的助手。只依据给定的“在当时可见且有效”的事实作答，"
    "不得使用这些事实之外的信息。若证据不足以作答，请回答“无法确定”。"
    "必须给出答案，并引用支撑答案的原文片段。严格输出 JSON。"
)

GENERATOR_USER = """问题：{question}
提问日期(t_ask)：{ask_date}
业务日期(t_val)：{val_date}

可用事实（已按提问日可见、业务日有效、适用条件匹配过滤）：
{facts}

请输出 JSON，形如：
{{"answer": "简洁答案", "cited_span": "引用的原文片段", "fact_id": "所依据事实的 id 或 null"}}"""


LONGCTX_SYSTEM = (
    "你是一个基于证据作答的助手。下面给出某政策领域的【全部】公告，每条标注了发布日期。"
    "请严格按两条时间轴作答："
    "(1) 只使用在【提问日期】当天或之前已发布的公告——提问日之后才发布的公告在提问当时无法得知，必须忽略；"
    "(2) 在这些“当时可见”的公告中，给出在【业务日期】当天实际生效、且满足问题所述适用条件"
    "（套数/期限/区域/贷款类型等）的取值；注意后发布的公告可能只修改前一版的一部分条件，其余仍沿用旧值。"
    "若当时可见的公告不足以确定答案，请回答“无法确定”。必须给出答案并引用支撑它的原文片段。严格输出 JSON。"
)

LONGCTX_USER = """问题：{question}
提问日期(t_ask)：{ask_date}
业务日期(t_val)：{val_date}

全部公告（按发布时间从早到晚排序，每条含发布日期）：
{docs}

请输出 JSON，形如：
{{"answer": "简洁答案", "cited_span": "引用的原文片段", "doc": "所依据公告的 id"}}"""


def longctx_messages(question: str, ask_date: str, val_date: str, docs: str) -> list[dict[str, str]]:
    return [
        {"role": "system", "content": LONGCTX_SYSTEM},
        {"role": "user", "content": LONGCTX_USER.format(
            question=question, ask_date=ask_date, val_date=val_date, docs=docs)},
    ]


def extractor_messages(title: str, text: str) -> list[dict[str, str]]:
    return [
        {"role": "system", "content": EXTRACTOR_SYSTEM},
        {"role": "user", "content": EXTRACTOR_USER.format(title=title, text=text[:12000])},
    ]


def classifier_messages(new_fact: str, candidates: str) -> list[dict[str, str]]:
    return [
        {"role": "system", "content": CLASSIFIER_SYSTEM},
        {"role": "user", "content": CLASSIFIER_USER.format(new_fact=new_fact, candidates=candidates)},
    ]


def generator_messages(question: str, ask_date: str, val_date: str, facts: str) -> list[dict[str, str]]:
    return [
        {"role": "system", "content": GENERATOR_SYSTEM},
        {"role": "user", "content": GENERATOR_USER.format(
            question=question, ask_date=ask_date, val_date=val_date, facts=facts)},
    ]
