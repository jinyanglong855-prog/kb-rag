"""零 API 建图路径：人工（agent）抽取 → 结构化 JSON → LightRAG 本地 API 灌图。

背景：全量 535 篇的 API 抽取费用高（用户暂停于 210/535）。本模块让 agent
自己读文档产出抽取结果，经 acreate/aedit API 写入——合并、嵌入、落盘
全走 LightRAG 本地代码，全程不调任何 LLM API。

实体去重/合并语义：同名实体存在则把新描述以 <SEP> 追加（与 LightRAG 自身
合并格式一致），不覆盖；关系存在则追加描述、weight 累加。

抽取 JSON schema（每文档一份）：
{
  "rel": "公司智能体/项目状态/README.md",
  "entities": [{"name": "公司智能体", "entity_type": "project",
                "description": "自包含的一句话以上描述（写明与本文档的关系）"}],
  "relations": [{"source": "公司智能体", "target": "DeepSeek Harness",
                 "description": "公司智能体基于 DeepSeek Harness 一切皆插件范式构建",
                 "keywords": "框架 依托"}]
}
"""

import asyncio
import hashlib
import json
from datetime import datetime
from pathlib import Path

from .config import INDEX_DIR, KB_PATH, validate_kb_path
from .graph import GRAPH_HISTORY_SKIP, _doc_id, _graph_files, _load_meta, _make_rag, _save_meta  # noqa: F401

EXTRACT_DIR = "work/graph_extract"


def _meta_path() -> Path:
    return Path(INDEX_DIR) / "graph_meta.json"


def _api_processed() -> set[str]:
    """LightRAG doc_status 里已 processed 的文档（全角斜杠路径还原为常规 rel）。"""
    f = Path(INDEX_DIR) / "lightrag" / "kv_store_doc_status.json"
    if not f.exists():
        return set()
    d = json.loads(f.read_text())
    return {
        str(v.get("file_path", "")).replace("／", "/")
        for v in d.values()
        if v.get("status") == "processed"
    }


def _priority_sort(rel: str) -> tuple[int, str]:
    """状态页 > 项目状态区 > 根/顶层 > 其余；同级按路径。"""
    p = Path(rel)
    if p.name in ("README.md", "STATUS.md") and "项目状态" in p.parts:
        tier = 0 if p.parent.name == "项目状态" else 1
    elif "项目状态" in p.parts:
        tier = 2
    elif len(p.parts) == 1:
        tier = 3
    else:
        tier = 4
    return tier, rel


def remaining_docs() -> list[str]:
    """图谱语料中尚未进图的文档，按基础构建优先级排序。"""
    validate_kb_path()
    done = _api_processed() | set(_load_meta())
    rels = [rel for _, rel in _graph_files(KB_PATH) if rel not in done]
    return sorted(rels, key=_priority_sort)


async def _desc_of(info: dict | None) -> str:
    """get_entity_info/get_relation_info 的返回里 description 可能在顶层或 graph_data。"""
    if not info:
        return ""
    return str(info.get("description") or (info.get("graph_data") or {}).get("description") or "")


async def _upsert(rag, data: dict) -> dict:
    rel = data["rel"]
    file_path = rel.replace("/", "／")
    created = merged = 0

    # LightRAG 在 create/edit 内部会对实体名做规范化（去中英间空格、全角转半角等，
    # 见 lightrag.utils.normalize_entity_name）；这里先规范化再查询/写入，否则同名
    # 实体会被判为不存在而重复创建报错，关系也会找不到目标节点。
    from lightrag.utils import normalize_entity_name

    for ent in data.get("entities", []):
        name, desc = normalize_entity_name(ent["name"]).strip(), ent["description"].strip()
        if not name or not desc:
            continue
        payload = {"entity_type": ent.get("entity_type", "content"),
                   "description": desc, "source_id": _doc_id(rel), "file_path": file_path}
        old_desc = await _desc_of(await rag.get_entity_info(name))
        if old_desc:
            payload["description"] = f"{old_desc}\n<SEP>\n{desc}"
            await rag.aedit_entity(name, payload, allow_rename=False)
            merged += 1
        else:
            await rag.acreate_entity(name, payload)
            created += 1

    rel_created = rel_merged = 0
    for r in data.get("relations", []):
        src, tgt, desc = (normalize_entity_name(r["source"]).strip(),
                          normalize_entity_name(r["target"]).strip(),
                          r["description"].strip())
        if not (src and tgt and desc):
            continue
        try:
            old_desc = await _desc_of(await rag.get_relation_info(src, tgt))
        except Exception:
            old_desc = ""
        payload = {"description": desc, "keywords": r.get("keywords", ""),
                   "source_id": _doc_id(rel), "file_path": file_path}
        if old_desc:
            payload["description"] = f"{old_desc}\n<SEP>\n{desc}"
            payload["weight"] = 2.0
            await rag.aedit_relation(src, tgt, payload)
            rel_merged += 1
        else:
            payload["weight"] = 1.0
            await rag.acreate_relation(src, tgt, payload)
            rel_created += 1

    return {"entities_created": created, "entities_merged": merged,
            "relations_created": rel_created, "relations_merged": rel_merged}


def ingest_manual(path: str) -> dict:
    """把一份抽取 JSON 灌入图并标记该文档已完成。"""
    validate_kb_path()
    f = Path(path)
    data = json.loads(f.read_text(encoding="utf-8"))
    rel = data["rel"]
    src = KB_PATH / rel
    if not src.is_file():
        raise SystemExit(f"文档不存在: {rel}")

    async def run() -> dict:
        rag = await _make_rag(no_llm=True)
        try:
            r = await _upsert(rag, data)
        finally:
            await rag.finalize_storages()
        meta = _load_meta()
        meta[rel] = hashlib.sha1(src.read_bytes()).hexdigest()
        _save_meta(meta)
        r["rel"] = rel
        return r

    return asyncio.run(run())


def manual_status() -> dict:
    done = _api_processed() | set(_load_meta())
    total = len(list(_graph_files(KB_PATH)))
    remaining = remaining_docs()
    return {"done": len(done), "total": total, "remaining": len(remaining),
            "next": remaining[:20]}
