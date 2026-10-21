"""混合检索：BM25（关键词精确性）+ 向量（语义泛化），RRF 融合。

设计取舍：BM25 索引直接建在已缓存的 chunks 上，重建零嵌入成本；
RRF 而非加权分数，因为两路分数量纲不可比，排名融合更稳。
"""

import json
import re
import sys
from pathlib import Path

import bm25s
import jieba

from .authority import apply as authority_apply
from .config import INDEX_DIR
from .search import query as vector_query

# 纯标点/符号词元（如全角"？"）：进 BM25 会按 IDF 加权，把恰含该标点的
# 杂乱文档拉到候选前列（实测单字符可把黄金文件顶出 top-20）。查询与索引
# 共用本函数，改后需 index --bm25 重建。
_PUNCT_ONLY = re.compile(r"^[\W_]+$", re.UNICODE)


def _sanitize_params() -> None:
    params_path = Path(INDEX_DIR) / "bm25" / "params.index.json"
    params = json.loads(params_path.read_text(encoding="utf-8"))
    if "corpora" in params:
        params.pop("corpora")
        params_path.write_text(json.dumps(params), encoding="utf-8")


def _tokenize(text: str) -> list[str]:
    return [
        t for t in jieba.lcut(text)
        if t.strip() and not t.isspace() and not _PUNCT_ONLY.match(t)
    ]


def build_bm25_index() -> int:
    chunks = json.loads((Path(INDEX_DIR) / "chunks.json").read_text(encoding="utf-8"))
    corpus_tokens = [_tokenize(f"{c['path']} {c['heading']} {c['text']}") for c in chunks]
    bm25 = bm25s.BM25()
    bm25.index(corpus_tokens)
    bm25.save(str(INDEX_DIR / "bm25"))
    _sanitize_params()  # bm25s 0.3.11: save 写入的 corpora 键 load 时不被 __init__ 接受
    (Path(INDEX_DIR) / "bm25_meta.json").write_text(
        json.dumps({"count": len(chunks)}, ensure_ascii=False), encoding="utf-8")
    # stderr 而非 stdout：MCP stdio 协议下 stdout 是协议通道，进度输出混入
    # 会产生 "Invalid JSON" 帧污染（审查发现 P1）；CLI 侧 stderr 同样可见
    print(f"BM25 索引完成: {len(chunks)} 块", file=sys.stderr)
    return len(chunks)


def _bm25_query(question: str, k: int) -> list[tuple[int, float]]:
    bm25 = bm25s.BM25.load(str(INDEX_DIR / "bm25"))
    results, scores = bm25.retrieve([_tokenize(question)], k=k)
    return [(int(i), float(s)) for i, s in zip(results[0], scores[0])]


def _load_chunks() -> list[dict]:
    return json.loads((Path(INDEX_DIR) / "chunks.json").read_text(encoding="utf-8"))


def hybrid_query(question: str, k: int = 5, vector_candidates: int = 50) -> list[dict]:
    """RRF: score = Σ 1/(60 + rank)。两路各取候选，按融合分排序。

    候选池 50（2026-10-16 实测）：20 池漏 2 个评测用例的金标（35/50 池才进池）；
    加深池只加宽召回。ask 层的精排在父块去重截断前对超量召回（k×3）整池打分。
    """
    chunks = _load_chunks()
    # BM25 与块的行号必须同源：落盘与 BM25 重建之间的窗口里，旧行号映射会错配
    # （复验 P2-2）——计数不一致（或 BM25 尚未建过）时降级纯向量路，慢但正确
    bm25_meta_path = Path(INDEX_DIR) / "bm25_meta.json"
    bm25_count = json.loads(bm25_meta_path.read_text(encoding="utf-8")).get("count") \
        if bm25_meta_path.exists() else None
    if bm25_count != len(chunks):
        print(f"[hybrid] BM25 计数({bm25_count})与块数({len(chunks)})不一致，"
              "本轮降级纯向量检索（跑 kb-rag index --bm25 恢复混合）", file=sys.stderr)
        return vector_query(question, k=k)
    # 候选数不超过语料规模（bm25s 对 k > corpus 抛错；tiny 语料/单测场景触发）
    vector_candidates = min(vector_candidates, len(chunks))

    vec_hits = vector_query(question, k=vector_candidates)

    bm_hits = _bm25_query(question, k=vector_candidates)
    bm_ranked = list(enumerate(bm_hits, 1))  # (rank, (idx, score))

    rrf: dict[int, float] = {}
    for rank, (idx, _s) in bm_ranked:
        rrf[idx] = rrf.get(idx, 0.0) + 1.0 / (60 + rank)
    # 向量路直接用命中自带行号。历史上按"path#heading→行号"中转，同键多块时
    # 字典推导"后行覆盖前行"把融合分记到该键最后一个块上（源码文件全块同键，
    # 向量分系统性锚在文件尾）——父块扩展以行号定窗，错块会被放大成整段错误上下文
    for rank, h in enumerate(vec_hits, 1):
        idx = h["idx"]
        if 0 <= idx < len(chunks):  # hybrid 与 vector 两读之间并发重建换序的窄窗：越界跳过不崩
            rrf[idx] = rrf.get(idx, 0.0) + 1.0 / (60 + rank)

    # 权威性加权：知识库结构语义（现行>归档）作为可解释的分数乘数
    weighted = {idx: authority_apply(chunks[idx]["path"], score) for idx, score in rrf.items()}
    top = sorted(weighted.items(), key=lambda x: -x[1])[:k]
    out = []
    for idx, score in top:
        c = chunks[idx]
        out.append({**c, "score": round(score, 4), "idx": idx})  # idx 供父块扩展定位
    return out
