"""检索与同步入口：query（向量路 + 空间门禁）、staleness（新鲜度）、sync_incremental（add/ask 共用）。

建索引引擎在 indexer.py（index 与 add 共用一条内容寻址流水线）；本模块是它的
对外薄壳加查询路径。历史脚本引用的旧名以别名保留在文末。
"""

import json
import sys
from pathlib import Path

import numpy as np

from .config import INDEX_DIR, validate_kb_path
from .embedder import embed_queries
from .indexer import (embed_input, embed_texts_normalized, ensure_queryable, rebuild,
                      scan_file_hashes)


def query(question: str, k: int = 5) -> list[dict]:
    ensure_queryable()  # 索引缺失/与当前嵌入配置不同空间：在此明确报错，绝不带病检索
    index_dir = Path(INDEX_DIR)
    vectors = np.load(index_dir / "vectors.npy")
    chunks = json.loads((index_dir / "chunks.json").read_text(encoding="utf-8"))
    if len(chunks) != vectors.shape[0]:  # meta 校验拦不住"同块数错位"，这里三元齐全终检
        from .indexer import IndexNotQueryable

        raise IndexNotQueryable(
            f"索引文件错位（chunks {len(chunks)} 行 ≠ vectors {vectors.shape[0]} 行）：拒绝检索。"
            "重跑 uv run kb-rag index 重建。")
    q = np.array(embed_queries([question])[0], dtype=np.float32)
    q /= max(np.linalg.norm(q), 1e-10)
    scores = vectors @ q
    # idx=块在索引中的行号：父块扩展（parents.expand）靠它定位兄弟块
    return [{**chunks[i], "score": round(float(scores[i]), 4), "idx": int(i)}
            for i in np.argsort(-scores)[:k]]


def staleness() -> dict:
    """索引新鲜度：对比磁盘文件指纹与 meta.json 已索引指纹（ask 的 L2 查询兜底）。

    mode=missing 表示索引尚未建立；stale 表示有文件新增/变更/删除未入库。
    只核对切块主索引；wiki 摘要树的路由新鲜度由 index --wiki 维护。
    """
    root = validate_kb_path()
    meta_path = Path(INDEX_DIR) / "meta.json"
    if not meta_path.exists():
        return {"mode": "missing", "changed": 0, "deleted": 0, "total": 0}
    old = json.loads(meta_path.read_text(encoding="utf-8")).get("hashes", {})
    new = scan_file_hashes(root)
    changed = sum(1 for rel, h in new.items() if old.get(rel) != h)
    deleted = len(set(old) - set(new))
    return {"mode": "stale" if (changed or deleted) else "clean",
            "changed": changed, "deleted": deleted, "total": len(new)}


def sync_incremental() -> dict:
    """增量入库并同步 BM25；Chroma 打过期标记延迟同步。

    ask/hybrid 的默认检索路径只依赖 BM25+numpy（秒级），Chroma 全量重灌
    随语料线性变慢，不阻塞收口——下次 --mode chroma 查询时按标记重建。
    `add` 命令与 ask 的过期自动补齐共用此入口——保证两条路径行为一致。
    mode=mismatch（索引与当前嵌入配置不同空间）原样返回，由调用方给出指引。
    """
    r = rebuild(mode="delta")
    if r["mode"] in ("delta", "full"):
        from .hybrid import build_bm25_index
        from .store import mark_stale

        build_bm25_index()
        mark_stale()
        if r["mode"] == "full":
            from .store import build_chroma
            from .wikitree import build_wiki

            build_chroma()
            build_wiki()  # ask 的结构层硬依赖 wikitree.json——首次全量必须补齐
    return r


def build_index(force: bool = False) -> int:
    r = rebuild(mode="full", force=force)
    print(f"全量索引完成: {r['total']} 块 / 复用 {r['reused']} + 新嵌 {r['embedded']}"
          f" / 移除 {r['removed']} / {r['seconds']:.1f}s",
          file=sys.stderr)  # stderr：MCP stdio 下 stdout 是协议通道
    return r["total"]


def incremental_index() -> dict:
    """增量入库：按文件指纹只重嵌变更文件里真缺的块（引擎细节见 indexer.rebuild）。"""
    return rebuild(mode="delta")


# 旧名别名：历史脚本按 search 取这些内部函数（work/embed_resume.py 已被引擎取代，待清理）
_embed_input = embed_input
_scan_hashes = scan_file_hashes
_embed_normalized = embed_texts_normalized
