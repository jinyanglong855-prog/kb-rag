"""嵌入层：fastembed 本地 ONNX 推理，零 API 成本。

threads=1：ONNX 多线程浮点求和顺序不定，同一文本跨进程向量有微小差异，
93k 块近重复语料会把差异放大成排名翻转（评测边界题不可复现）；
单线程推理确定化，批量建索引的耗时换可复现检索（EMBED_THREADS 可覆盖）。
"""

import os
from functools import lru_cache

from fastembed import TextEmbedding

from .config import EMBED_MODEL


@lru_cache(maxsize=1)
def get_model() -> TextEmbedding:
    threads = int(os.environ.get("EMBED_THREADS", "1"))
    return TextEmbedding(model_name=EMBED_MODEL, threads=threads)


def embed(texts: list[str]) -> list[list[float]]:
    return list(get_model().embed(texts))
