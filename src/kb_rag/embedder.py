"""嵌入层：fastembed 本地 ONNX 推理，零 API 成本。

threads=8（2026-10-17 实测更新）：Qwen3-Embedding-0.6B-Q 在 8 线程下三个
独立进程的查询向量逐位一致（旧 bge-small-zh 时代多线程跨进程有浮点差、
93k 近重复语料放大成排名翻转，才钉死单线程；换模型后以新实测为准）。
单线程在本模型要 3.7s/条、全库重建 95 小时，不可行；8 线程 0.1s/条、
重建 2.7 小时。若未来再现实测不稳，EMBED_THREADS=1 可回退。
"""

import os
from functools import lru_cache

from fastembed import TextEmbedding

from .config import EMBED_MODEL


@lru_cache(maxsize=1)
def get_model() -> TextEmbedding:
    threads = int(os.environ.get("EMBED_THREADS", "8"))
    return TextEmbedding(model_name=EMBED_MODEL, threads=threads)


def embed(texts: list[str]) -> list[list[float]]:
    return list(get_model().embed(texts))
