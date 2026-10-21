"""查询改写层（2026-10-17 侦察后新增环节）：Multi-Query + HyDE。

治两种实测丢分：①"一种问法找不到、换个说法就有"的召回缺口（池 20 时代
single-3/status-2 的根因，池 50 缓解但未根治语义鸿沟）；②问题与文档的词汇
鸿沟（否定语义"还开着吗"vs 文档"已停止"，embedding 业界通病）。

做法：LLM 把问题改写成 n 个变体（Multi-Query），另生成一段"假设答案"文档
（HyDE），每路各跑混合检索，RRF 融合（融合分再乘权威乘数，与混合层同一
哲学）后交精排。

可复现性调和（本仓硬承诺）：改写结果按问题 SHA-1 落缓存 index/rewrite_cache.json
——同一问题永远得到同一组改写，检索逐位可复现；仅首次问法需 LLM（glm-4.5-flash，
单次费用可忽略）。API 失败回退原问题：改写层永远不阻塞检索。

开关 KB_QUERY_REWRITE（默认开；置 0 关闭）。实验阶段经用户确认：准确率优先，
不考虑 LLM 调用成本。
"""

import hashlib
import json
import os
from pathlib import Path

from openai import OpenAI

from .authority import authority_multiplier
from .config import INDEX_DIR, LLM_API_KEY, LLM_BASE_URL, LLM_MODEL
from .hybrid import hybrid_query

CACHE = INDEX_DIR / "rewrite_cache.json"

MULTI_PROMPT = """你是检索查询改写器。把用户的问题改写成 {n} 个不同的检索问法：
覆盖不同措辞、同义词、相关关键词与可能的文档用语（包括与问法相反的表述，
如问"还开着吗"也可改写为"运行状态/已停止"）。不要回答问题本身。
只输出一个 JSON 字符串数组，不要输出其他内容。

问题：{question}"""

HYDE_PROMPT = """假设知识库里有一段能完美回答下述问题的文档，用 2-3 句话写出这段
文档的原文（陈述句、含关键术语与数据词），用于语义检索。只输出这段文字。

问题：{question}"""


def enabled() -> bool:
    return os.environ.get("KB_QUERY_REWRITE", "1") != "0" and bool(LLM_API_KEY)


def _llm(prompt: str) -> str:
    client = OpenAI(api_key=LLM_API_KEY, base_url=LLM_BASE_URL)
    resp = client.chat.completions.create(
        model=LLM_MODEL,
        messages=[{"role": "user", "content": prompt}],
        temperature=0.2,
    )
    return resp.choices[0].message.content or ""


def _cache_load() -> dict:
    if CACHE.exists():
        return json.loads(CACHE.read_text(encoding="utf-8"))
    return {}


def _cache_save(cache: dict) -> None:
    INDEX_DIR.mkdir(exist_ok=True)
    CACHE.write_text(json.dumps(cache, ensure_ascii=False, indent=1), encoding="utf-8")


def rewrite_variants(question: str, n: int = 3) -> list[str]:
    """改写变体列表（不含原问题）。缓存命中或任何失败 → 空列表（回退原问题单路）。"""
    key = hashlib.sha1(question.encode()).hexdigest()
    entry = _cache_load().get(key, {})
    if "variants" in entry:  # 字段级判断：变体与 HyDE 分属同一条目，先到先写不能挡住后到
        return entry["variants"]
    variants: list[str] = []
    try:
        raw = _llm(MULTI_PROMPT.format(n=n, question=question))
        start, end = raw.find("["), raw.rfind("]")
        if 0 <= start < end:
            parsed = json.loads(raw[start:end + 1])
            variants = [v.strip() for v in parsed if isinstance(v, str) and v.strip()][:n]
    except Exception:
        variants = []  # 网络/API 失败：优雅降级，不阻塞检索
    _cache_save({**_cache_load(), key: {**entry, "variants": variants}})
    return variants


def hyde_document(question: str) -> str:
    """假设答案文档。缓存命中或任何失败 → 空串（该路跳过）。"""
    key = hashlib.sha1(question.encode()).hexdigest()
    entry = _cache_load().get(key, {})
    if "hyde" in entry:
        return entry["hyde"]
    doc = ""
    try:
        doc = _llm(HYDE_PROMPT.format(question=question)).strip()
    except Exception:
        doc = ""
    _cache_save({**_cache_load(), key: {**entry, "hyde": doc}})
    return doc


def fused_query(question: str, k: int = 8, pool: int = 50) -> list[dict]:
    """多路改写检索 + RRF 融合（×权威乘数）。返回融合后 top-k，字段同 hybrid_query。

    每路取 k_deep 深候选再融合，避免单路截断丢掉融合时有竞争力的块；
    按块行号（idx）键控而非 path#heading——同键多块时"后写覆盖"会把整节
    融合分塌缩到最后一块上；改写不可用（关开关/无 key/API 失败）时退化为
    原问题单路，行为≈hybrid_query。
    """
    routes = [question] + rewrite_variants(question)
    hyde = hyde_document(question)
    if hyde:
        routes.append(hyde)
    k_deep = max(k, 12)
    rrf: dict[int, float] = {}
    chunks_by_idx: dict[int, dict] = {}
    for route in routes:
        for rank, h in enumerate(hybrid_query(route, k=k_deep, vector_candidates=pool), 1):
            chunks_by_idx[h["idx"]] = h
            rrf[h["idx"]] = rrf.get(h["idx"], 0.0) + \
                (1.0 / (60 + rank)) * authority_multiplier(h["path"])
    top = sorted(rrf.items(), key=lambda x: -x[1])[:k]
    return [{**chunks_by_idx[idx], "score": round(s, 6)} for idx, s in top]
