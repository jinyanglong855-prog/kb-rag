"""问答生成层：检索 top-k → LLM 生成带引用的回答（RAG 闭环最后一环）。

引用格式强制：回答中的每个关键论断标注 [n]，n 对应检索来源路径——
让"AI 说的"可以追到"库里写的"。
"""

import os

from openai import OpenAI

from .config import HYBRID_DEFAULT, LLM_API_KEY, LLM_BASE_URL, LLM_MODEL
from .hybrid import hybrid_query

PROMPT = """你是本地知识库的问答助手。只依据提供的资料回答，不编造。
资料中每条带编号。回答要求：
1. 关键论断后标注来源编号，如 [1]
2. 资料不足以回答时，明确说"资料中没有提到"，不要猜
3. 中文回答，简洁

资料：
{context}

问题：{question}
"""


def ask(question: str, k: int = 5) -> str:
    if not LLM_API_KEY:
        raise SystemExit("请在 .env 里填 LLM_API_KEY（智谱/DeepSeek 均可，接口兼容 OpenAI SDK）")

    hits = hybrid_query(question, k=k)
    if not hits:
        return "知识库中没有检索到相关内容。"
    context = "\n\n".join(
        f"[{i}] 来源：{h['path']} #{h['heading']}\n{h['text'][:800]}"
        for i, h in enumerate(hits, 1)
    )
    client = OpenAI(api_key=LLM_API_KEY, base_url=LLM_BASE_URL)
    resp = client.chat.completions.create(
        model=LLM_MODEL,
        messages=[{"role": "user", "content": PROMPT.format(context=context, question=question)}],
        temperature=0.2,
    )
    answer = resp.choices[0].message.content
    sources = "\n".join(f"[{i}] {h['path']} #{h['heading']}" for i, h in enumerate(hits, 1))
    return f"{answer}\n\n来源：\n{sources}"
