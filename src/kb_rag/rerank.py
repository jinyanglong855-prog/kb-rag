"""精排层：交叉编码器对候选逐对打分，精度高于双塔但只能小候选集用。

流水线定位：召回（向量+BM25 大候选集）→ 融合（RRF+权威性）→ 精排（交叉编码器，top 20 → top k）。
"""

import math
import os
from functools import lru_cache

from fastembed.rerank.cross_encoder import TextCrossEncoder

from .authority import authority_multiplier
from .config import RERANK_MODEL


@lru_cache(maxsize=1)
def get_model() -> TextCrossEncoder:
    threads = int(os.environ.get("RERANK_THREADS", "1"))  # 与嵌入层同理：单线程保证可复现
    return TextCrossEncoder(model_name=RERANK_MODEL, threads=threads)


def _sigmoid(s: float) -> float:
    if s >= 0:
        return 1 / (1 + math.exp(-s))
    e = math.exp(s)
    return e / (1 + e)


def rerank(question: str, hits: list[dict]) -> list[dict]:
    if len(hits) < 2:
        return hits
    docs = [f"{h['path']} {h['heading']}\n{h['text']}" for h in hits]
    scores = list(get_model().rerank(question, docs))
    for h, s in zip(hits, scores):
        h["rerank_score"] = round(float(s), 4)
        # 精排只算语义相关，会丢掉混合阶段已加权的权威分层（实测 status-3：
        # 状态页在池内第 1，纯精排后跌出前 5）。sigmoid 概率 × 权威乘数做
        # 先验融合，有界可解释；不能直接乘原始 logit（负值会反序）。
        h["final_score"] = round(_sigmoid(float(s)) * authority_multiplier(h["path"]), 4)
    return sorted(hits, key=lambda h: -h["final_score"])
