"""混合检索：BM25（关键词精确性）+ 向量（语义泛化），RRF 融合。

设计取舍：BM25 索引直接建在已缓存的 chunks 上，重建零嵌入成本；
RRF 而非加权分数，因为两路分数量纲不可比，排名融合更稳。
"""

import json
from pathlib import Path

import bm25s
import jieba

from .authority import apply as authority_apply
from .config import INDEX_DIR
from .search import query as vector_query


def _sanitize_params() -> None:
    params_path = Path(INDEX_DIR) / "bm25" / "params.index.json"
    params = json.loads(params_path.read_text(encoding="utf-8"))
    if "corpora" in params:
        params.pop("corpora")
        params_path.write_text(json.dumps(params), encoding="utf-8")


def _tokenize(text: str) -> list[str]:
    return [t for t in jieba.lcut(text) if t.strip() and not t.isspace()]


def build_bm25_index() -> int:
    chunks = json.loads((Path(INDEX_DIR) / "chunks.json").read_text(encoding="utf-8"))
    corpus_tokens = [_tokenize(f"{c['path']} {c['heading']} {c['text']}") for c in chunks]
    bm25 = bm25s.BM25()
    bm25.index(corpus_tokens)
    bm25.save(str(INDEX_DIR / "bm25"))
    _sanitize_params()  # bm25s 0.3.11: save 写入的 corpora 键 load 时不被 __init__ 接受
    (Path(INDEX_DIR) / "bm25_meta.json").write_text(
        json.dumps({"count": len(chunks)}, ensure_ascii=False), encoding="utf-8")
    print(f"BM25 索引完成: {len(chunks)} 块")
    return len(chunks)


def _bm25_query(question: str, k: int) -> list[tuple[int, float]]:
    bm25 = bm25s.BM25.load(str(INDEX_DIR / "bm25"))
    results, scores = bm25.retrieve([_tokenize(question)], k=k)
    return [(int(i), float(s)) for i, s in zip(results[0], scores[0])]


def _load_chunks() -> list[dict]:
    return json.loads((Path(INDEX_DIR) / "chunks.json").read_text(encoding="utf-8"))


def hybrid_query(question: str, k: int = 5, vector_candidates: int = 20) -> list[dict]:
    """RRF: score = Σ 1/(60 + rank)。两路各取候选，按融合分排序。"""
    chunks = _load_chunks()

    vec_hits = vector_query(question, k=vector_candidates)
    path_to_rank = {h["path"] + "#" + h["heading"]: r for r, h in enumerate(vec_hits, 1)}
    # 向量路：直接用已算好的序
    vec_ranked = [(path_to_rank[key], r) for key, r in path_to_rank.items()]

    bm_hits = _bm25_query(question, k=vector_candidates)
    bm_ranked = list(enumerate(bm_hits, 1))  # (rank, (idx, score))

    rrf: dict[int, float] = {}
    for rank, (idx, _s) in bm_ranked:
        rrf[idx] = rrf.get(idx, 0.0) + 1.0 / (60 + rank)
    # 向量路需要 path->idx 映射
    path_to_idx = {c["path"] + "#" + c["heading"]: i for i, c in enumerate(chunks)}
    for rank, key in enumerate(
        sorted(path_to_rank, key=lambda kk: path_to_rank[kk]), 1
    ):
        idx = path_to_idx.get(key)
        if idx is not None:
            rrf[idx] = rrf.get(idx, 0.0) + 1.0 / (60 + rank)

    # 权威性加权：知识库结构语义（现行>归档）作为可解释的分数乘数
    weighted = {idx: authority_apply(chunks[idx]["path"], score) for idx, score in rrf.items()}
    top = sorted(weighted.items(), key=lambda x: -x[1])[:k]
    out = []
    for idx, score in top:
        c = chunks[idx]
        out.append({**c, "score": round(score, 4)})
    return out
