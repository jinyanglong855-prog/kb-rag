"""精排层：交叉编码器对候选逐对打分，精度高于双塔但只能小候选集用。

流水线定位：召回（向量+BM25 大候选集）→ 融合（RRF+权威性）→ 精排（交叉编码器，top 20 → top k）。
"""

from functools import lru_cache

from fastembed.rerank.cross_encoder import TextCrossEncoder

from .config import RERANK_MODEL


@lru_cache(maxsize=1)
def get_model() -> TextCrossEncoder:
    return TextCrossEncoder(model_name=RERANK_MODEL)


def rerank(question: str, hits: list[dict]) -> list[dict]:
    if len(hits) < 2:
        return hits
    docs = [f"{h['path']} {h['heading']}\n{h['text']}" for h in hits]
    scores = list(get_model().rerank(question, docs))
    for h, s in zip(hits, scores):
        h["rerank_score"] = round(float(s), 4)
    return sorted(hits, key=lambda h: -h["rerank_score"])
