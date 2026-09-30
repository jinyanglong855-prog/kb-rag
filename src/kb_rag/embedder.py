"""嵌入层：fastembed 本地 ONNX 推理，零 API 成本。"""

from functools import lru_cache

from fastembed import TextEmbedding

from .config import EMBED_MODEL


@lru_cache(maxsize=1)
def get_model() -> TextEmbedding:
    return TextEmbedding(model_name=EMBED_MODEL)


def embed(texts: list[str]) -> list[list[float]]:
    return list(get_model().embed(texts))
