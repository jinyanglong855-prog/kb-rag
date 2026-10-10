"""阶段 3 mini-GraphRAG：LightRAG 整合（40.1k stars，MIT，复用优先——不自研抽取管线）。

图谱语料 = 主语料的 md 且排除归档/历史区（graph 范围比主检索窄一档）：
多跳问题的价值在活跃知识，历史归档会把过期实体关系注入图里污染答案。

增量策略与 search.py 同构：路径→SHA-1 指纹存 index/graph_meta.json；
变更文件先 adelete_by_doc_id（路径派生的稳定 id）再 ainsert，删除文件只删不插。
"""

import asyncio
import hashlib
import json
from datetime import datetime
from pathlib import Path

import numpy as np

from .config import INDEX_DIR, KB_PATH, LLM_API_KEY, LLM_BASE_URL, LLM_MODEL, validate_kb_path
from .loader import SKIP_DIRS, load_text_files

GRAPH_HISTORY_SKIP = {
    "work", "outputs", "logs", "delivery", "archive", "归档", "predictions",
    "backups", "state", "releases", "history", "数据区", "原始快照",
}
META_JSON = "graph_meta.json"


def _doc_id(rel: str) -> str:
    return hashlib.md5(rel.encode()).hexdigest()


def _graph_files(root: Path) -> list[tuple[Path, str]]:
    out = []
    for p, rel in load_text_files(root):
        if p.suffix.lower() != ".md":
            continue
        if any(part in GRAPH_HISTORY_SKIP for part in Path(rel).parts):
            continue
        out.append((p, rel))
    return out


def _scan_hashes(root: Path) -> dict[str, str]:
    return {rel: hashlib.sha1(p.read_bytes()).hexdigest() for p, rel in _graph_files(root)}


def _meta_path() -> Path:
    return Path(INDEX_DIR) / META_JSON


def _load_meta() -> dict[str, str]:
    f = _meta_path()
    return json.loads(f.read_text()) if f.exists() else {}


def _save_meta(hashes: dict[str, str]) -> None:
    _meta_path().write_text(json.dumps(hashes, ensure_ascii=False, indent=1))


async def _make_rag():
    from lightrag import LightRAG
    from lightrag.base import EmbeddingFunc
    from lightrag.llm.openai import openai_complete_if_cache

    from .embedder import embed

    async def llm_func(prompt, system_prompt=None, history_messages=None, **kwargs):
        return await openai_complete_if_cache(
            LLM_MODEL,
            prompt,
            system_prompt=system_prompt,
            history_messages=history_messages or [],
            base_url=LLM_BASE_URL,
            api_key=LLM_API_KEY,
        )

    async def embed_func(texts: list[str]):
        return await asyncio.to_thread(lambda: np.asarray(embed(texts)))

    rag = LightRAG(
        working_dir=str(Path(INDEX_DIR) / "lightrag"),
        llm_model_func=llm_func,
        llm_model_max_async=8,
        embedding_func=EmbeddingFunc(embedding_dim=512, func=embed_func),
    )
    await rag.initialize_storages()
    return rag


def _is_graph(rel: str) -> bool:
    parts = Path(rel).parts
    return (
        rel.endswith(".md")
        and not any(part in SKIP_DIRS or part.startswith(".") for part in parts)
        and not any(part in GRAPH_HISTORY_SKIP for part in parts)
    )


def build_graph(full: bool = False, scope: str | None = None) -> dict:
    """增量建图：变更文件删旧插新，删除文件只删。scope 指定相对路径前缀则只处理该子树。"""
    validate_kb_path()
    old = {} if full else _load_meta()
    new = _scan_hashes(KB_PATH)
    if scope:
        new = {r: h for r, h in new.items() if r.startswith(scope)}
        old = {r: h for r, h in old.items() if r.startswith(scope)}

    added = [r for r, h in new.items() if old.get(r) != h]
    removed = [r for r in old if r not in new]

    async def run() -> dict:
        rag = await _make_rag()
        t0 = datetime.now()
        try:
            for rel in removed:
                await rag.adelete_by_doc_id(_doc_id(rel))
            for rel in added:
                p = KB_PATH / rel
                await rag.ainsert(
                    p.read_text(encoding="utf-8", errors="ignore"),
                    ids=[_doc_id(rel)],
                    # LightRAG 按文件名 basename 去重且 doc_id 以其为种子——
                    # 库里几十个同名 README/STATUS 会互撞。用全角斜杠保住
                    # 完整路径当唯一键，引用展示也可读。
                    file_paths=[rel.replace("/", "／")],
                )
        finally:
            await rag.finalize_storages()

        merged = {**{r: h for r, h in old.items() if r not in removed}, **new}
        _save_meta(merged)
        return {
            "added": len(added), "removed": len(removed),
            "total": len(merged), "seconds": (datetime.now() - t0).total_seconds(),
        }

    return asyncio.run(run())


def query_graph(question: str, mode: str = "mix") -> str:
    """图检索问答：mix=local+global+naive 融合（LightRAG 默认，效果通常最佳）。"""
    validate_kb_path()

    async def run() -> str:
        from lightrag import QueryParam

        rag = await _make_rag()
        try:
            return await rag.aquery(question, param=QueryParam(mode=mode))
        finally:
            await rag.finalize_storages()

    return asyncio.run(run())


def export_viz(out_html: str | None = None, top: int = 300) -> Path:
    """知识图谱导出 pyvis 交互 HTML。

    默认按度数取前 top 个实体（全图数千节点直接渲染会卡死），按实体类型着色，
    悬停显示描述；写入 index/（不进语料）。
    """
    from networkx.readwrite import read_graphml
    from pyvis.network import Network

    kg = Path(INDEX_DIR) / "lightrag" / "graph_chunk_entity_relation.graphml"
    if not kg.exists():
        raise SystemExit(f"图数据不存在: {kg}（先跑 graph-build）")
    g = read_graphml(kg)
    top_nodes = sorted(g, key=lambda n: g.degree(n), reverse=True)[:top]
    sub = g.subgraph(top_nodes)

    colors = {
        "person": "#e74c3c", "organization": "#e67e22", "artifact": "#f1c40f",
        "concept": "#3498db", "method": "#9b59b6", "location": "#1abc9c",
        "data": "#2ecc71", "event": "#e91e63", "creature": "#e74c3c",
    }
    net = Network(height="900px", width="100%", bgcolor="#111", font_color="white")
    for n, d in sub.nodes(data=True):
        et = (d.get("entity_type") or "other").lower()
        net.add_node(
            n,
            label=str(d.get("entity_id", n))[:24],
            title=f"{d.get('entity_id', '')} [{et}]\n{str(d.get('description', ''))[:300]}",
            color=colors.get(et, "#888"),
        )
    net.add_edges(sub.edges())
    net.repulsion(node_distance=120, spring_length=80)
    out = Path(out_html) if out_html else Path(INDEX_DIR) / "graph_viz.html"
    net.write_html(str(out))
    return out
