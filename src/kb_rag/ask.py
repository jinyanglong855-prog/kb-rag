"""问答生成层：检索 top-k → LLM 生成带引用的回答（RAG 闭环最后一环）。

引用格式强制：回答中的每个关键论断标注 [n]，n 对应检索来源路径——
让"AI 说的"可以追到"库里写的"。

阶段 2 路由：状态类问题（进度/卡点/下一步）先走结构层——目录摘要树命中
项目节点后，整份状态文件进入上下文，解决"STATUS 长文切块召回不全"。
路由是显式关键词规则（与权威性加权同一哲学：可解释，不做黑盒意图分类）。
"""

import re

from openai import OpenAI

from .config import LLM_API_KEY, LLM_BASE_URL, LLM_MODEL
from .hybrid import hybrid_query
from .wikitree import wiki_query

PROMPT = """你是本地知识库的问答助手。只依据提供的资料回答，不编造。
资料中每条带编号；若包含"状态页（整文件）"，它反映该项目最新状态，优先采信。
回答要求：
1. 关键论断后标注来源编号，如 [1]
2. 资料不足以回答时，明确说"资料中没有提到"，不要猜
3. 中文回答，简洁

资料：
{context}

问题：{question}
"""

# 显式状态意图规则：宁多勿漏——误命中只多带几条状态页上下文，漏命中才是真召回缺失。
# 除关键词外加问句结构信号（"还X吗/要不要/是否"），实测"闲鱼兜底还继续发吗"曾被关键词漏判
STATUS_INTENT = re.compile(
    r"状态|进度|卡|下一步|接下来|进展|做到哪|停在哪|在等|等待|怎么样|还.{0,6}吗|要不要|是否"
    r"|现在|目前|最近|计划|打算|安排|暂停|停止|停了|停掉|取消|放弃|继续"
)


def _is_status_question(q: str) -> bool:
    return bool(STATUS_INTENT.search(q))


def ask(question: str, k: int = 5) -> str:
    if not LLM_API_KEY:
        raise SystemExit("请在 .env 里填 LLM_API_KEY（智谱/DeepSeek 均可，接口兼容 OpenAI SDK）")

    sources: list[tuple[str, str]] = []  # (来源标签, 内容)
    if _is_status_question(question):
        for w in wiki_query(question, k=3):
            if w["status_file"]:
                sources.append((f"{w['status_file']}（{w['title']}·状态页整文件）",
                                f"节点摘要：{w['summary']}\n{w['file_text']}"))
    hits = hybrid_query(question, k=k)
    if not hits and not sources:
        return "知识库中没有检索到相关内容。"
    for h in hits:
        sources.append((f"{h['path']} #{h['heading']}", h["text"][:800]))

    context = "\n\n".join(f"[{i}] 来源：{label}\n{text}" for i, (label, text) in enumerate(sources, 1))
    client = OpenAI(api_key=LLM_API_KEY, base_url=LLM_BASE_URL)
    resp = client.chat.completions.create(
        model=LLM_MODEL,
        messages=[{"role": "user", "content": PROMPT.format(context=context, question=question)}],
        temperature=0.2,
    )
    answer = resp.choices[0].message.content
    footer = "\n".join(f"[{i}] {label}" for i, (label, _) in enumerate(sources, 1))
    return f"{answer}\n\n来源：\n{footer}"
