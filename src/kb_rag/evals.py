"""阶段 5 evals：自建评测集，五层检索匹配率对比。

层与指标：
- vector@5 / hybrid@8 / hybrid+rerank@5：命中 = 期望来源路径出现在 top-k 的 path 里（前缀匹配）
- status（结构层路由）：命中 = 期望状态页出现在 wiki_query 返回的节点/状态文件里
- graph（LightRAG）：命中 = 期望路径或评测集给的答案事实子串（graph_expected_any，v2 口径：
  手动灌图下 mix 上下文无 file-path 溯源行，路径判分结构性失效时回退事实判分）出现在
  only_need_context 的上下文里（多跳题专属）

评测集在 `kb-rag/项目状态/参考资料/eval_set.json`（该区 json 不进语料，避免自污染）。
题目与黄金路径均经 grep 验证确实含答案事实——评测的可信度先于规模。
"""

import json
import os
import time
from datetime import date
from pathlib import Path

from .config import KB_PATH, validate_kb_path
from .hybrid import hybrid_query
from .rerank import rerank
from .rewrite import fused_query
from .search import query as vector_query
from .wikitree import wiki_query

EVAL_SET = "kb-rag/项目状态/参考资料/eval_set.json"


def _load_cases() -> list[dict]:
    f = KB_PATH / EVAL_SET
    return json.loads(f.read_text(encoding="utf-8"))["cases"]


def _path_hit(expected: list[str], paths: list[str]) -> bool:
    # LightRAG 上下文里的 file-path 用全角斜杠（防 markdown 转义），匹配前归一到半角
    norm = lambda s: s.replace("／", "/")
    return any(any(norm(e) in norm(p) for p in paths) for e in expected)


def _hit_via(expected: list[str], hits: list[dict]) -> str:
    """黄金路径命中的块来源：orig=原文块命中，card=仅分析卡命中（卡片试点口径
    透明化——卡片 path=原路径，不区分会让试点效果虚高），-=未命中。"""
    norm = lambda s: s.replace("／", "/")
    matched = [h for h in hits if any(norm(e) in norm(h["path"]) for e in expected)]
    if not matched:
        return "-"
    return "card" if all(h.get("card") for h in matched) else "orig"


def _graph_context(question: str, mode: str = "mix") -> str:
    from lightrag import QueryParam

    from .graph import _make_rag

    async def run() -> str:
        rag = await _make_rag()
        try:
            return await rag.aquery(question, param=QueryParam(mode=mode, only_need_context=True))
        finally:
            await rag.finalize_storages()

    import asyncio

    return asyncio.run(run())


def _eval_case(case: dict, skip_graph: bool) -> dict:
    q, expected = case["question"], case["expected_any"]
    row: dict = {"id": case["id"], "type": case["type"]}

    if case["type"] in ("single_hop", "status"):
        row["via"] = {}
        vec_hits = vector_query(q, 5)
        row["vector@5"] = _path_hit(expected, [h["path"] for h in vec_hits])
        row["via"]["vector@5"] = _hit_via(expected, vec_hits)
        hybrid = hybrid_query(q, 8)
        row["hybrid@8"] = _path_hit(expected, [h["path"] for h in hybrid])
        row["via"]["hybrid@8"] = _hit_via(expected, hybrid)
        rr = rerank(q, hybrid)[:5]
        row["hybrid+rerank@5"] = _path_hit(expected, [h["path"] for h in rr])
        row["via"]["hybrid+rerank@5"] = _hit_via(expected, rr)
        # 查询改写层对照（Multi-Query + HyDE 融合；改写缓存已预热则逐位可复现）
        fused = fused_query(q, k=8)
        row["改写融合@8"] = _path_hit(expected, [h["path"] for h in fused])
        row["via"]["改写融合@8"] = _hit_via(expected, fused)
        rr2 = rerank(q, fused)[:5]
        row["改写融合+精排@5"] = _path_hit(expected, [h["path"] for h in rr2])
        row["via"]["改写融合+精排@5"] = _hit_via(expected, rr2)
    if case["type"] == "status":
        wiki_paths = []
        for w in wiki_query(q, k=3):
            wiki_paths.append(w["path"])
            if w.get("status_file"):
                # 节点 path 是目录、status_file 相对于它；根节点的 status_file 就是库根的文件
                wiki_paths.append(
                    w["status_file"] if w["path"] == "（根目录）" else f"{w['path']}/{w['status_file']}"
                )
        row["status路由"] = _path_hit(expected, wiki_paths)
    if case["type"] == "multi_hop" and not skip_graph:
        try:
            ctx = _graph_context(q)
            hit = _path_hit(expected, ctx.split())
            if not hit:
                # 手动灌图（batch_ingest 绕过 ainsert）下 mix 上下文无 file-path 溯源行
                # （text_chunks 空），路径判分结构性失效；回退到答案事实子串（eval_set
                # v2 口径：子串逐条 grep 验证在金标文件中在案）
                facts = case.get("graph_expected_any") or []
                hit = any(f in ctx for f in facts)
            row["graph"] = hit
        except Exception as e:  # 图未建或后端异常：如实记失败而不是吞掉
            row["graph"] = f"ERROR: {e}"
    return row


def run_evals(skip_graph: bool = False) -> list[dict]:
    validate_kb_path()
    cases = _load_cases()
    t0 = time.time()
    rows = [_eval_case(c, skip_graph) for c in cases]

    layers = ["vector@5", "hybrid@8", "hybrid+rerank@5", "改写融合@8", "改写融合+精排@5", "status路由", "graph"]
    print(f"{'case':10} {'type':11} " + " ".join(f"{l:16}" for l in layers))
    hits = {l: 0 for l in layers}
    tried = {l: 0 for l in layers}
    for r in rows:
        cells = []
        for l in layers:
            if l not in r:
                cells.append("—")
                continue
            tried[l] += 1
            v = r[l]
            if v is True:
                hits[l] += 1
                cells.append("✅")
            elif v is False:
                cells.append("❌")
            else:
                cells.append("💥")
        print(f"{r['id']:10} {r['type']:11} " + " ".join(f"{c:16}" for c in cells))

    print(f"\n匹配率（{time.time() - t0:.0f}s）：")
    for l in layers:
        if tried[l]:
            print(f"  {l:18} {hits[l]}/{tried[l]} = {hits[l] / tried[l]:.0%}")

    report = {
        "date": date.today().isoformat(),
        "eval_set": EVAL_SET,
        # KB_EVAL_REPORT：报告基名（试点/隔离索引评测用，避免覆盖生产最新报告）
        "index_dir": str(Path(os.environ.get("INDEX_DIR", "index")).name),
        "summary": {l: {"hit": hits[l], "tried": tried[l]} for l in layers if tried[l]},
        "rows": rows,
    }
    base = os.environ.get("KB_EVAL_REPORT", "evals_report")
    out = KB_PATH / EVAL_SET.replace("eval_set.json", f"{base}.json")
    out.write_text(json.dumps(report, ensure_ascii=False, indent=1), encoding="utf-8")
    # 带日期留档：可复现性声称要能事后复核，覆盖写只剩最后一份无法追溯
    dated = KB_PATH / EVAL_SET.replace("eval_set.json", f"{base}_{date.today().isoformat()}.json")
    dated.write_text(json.dumps(report, ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"报告: {out}（留档: {dated.name}）")
    return rows
