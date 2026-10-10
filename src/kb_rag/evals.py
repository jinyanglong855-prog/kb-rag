"""阶段 5 evals：自建评测集，五层检索匹配率对比。

层与指标：
- vector@5 / hybrid@8 / hybrid+rerank@5：命中 = 期望来源路径出现在 top-k 的 path 里（前缀匹配）
- status（结构层路由）：命中 = 期望状态页出现在 wiki_query 返回的节点/状态文件里
- graph（LightRAG）：命中 = 期望路径出现在 only_need_context 的上下文里（多跳题专属）

评测集在 `kb-rag/项目状态/参考资料/eval_set.json`（该区 json 不进语料，避免自污染）。
题目与黄金路径均经 grep 验证确实含答案事实——评测的可信度先于规模。
"""

import json
import time
from datetime import date
from pathlib import Path

from .config import KB_PATH, validate_kb_path
from .hybrid import hybrid_query
from .rerank import rerank
from .search import query as vector_query
from .wikitree import wiki_query

EVAL_SET = "kb-rag/项目状态/参考资料/eval_set.json"


def _load_cases() -> list[dict]:
    f = KB_PATH / EVAL_SET
    return json.loads(f.read_text(encoding="utf-8"))["cases"]


def _path_hit(expected: list[str], paths: list[str]) -> bool:
    return any(any(e in p for p in paths) for e in expected)


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
        row["vector@5"] = _path_hit(expected, [h["path"] for h in vector_query(q, 5)])
        hybrid = hybrid_query(q, 8)
        row["hybrid@8"] = _path_hit(expected, [h["path"] for h in hybrid])
        row["hybrid+rerank@5"] = _path_hit(expected, [h["path"] for h in rerank(q, hybrid)[:5]])
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
            row["graph"] = _path_hit(expected, _graph_context(q).split())
        except Exception as e:  # 图未建或后端异常：如实记失败而不是吞掉
            row["graph"] = f"ERROR: {e}"
    return row


def run_evals(skip_graph: bool = False) -> list[dict]:
    validate_kb_path()
    cases = _load_cases()
    t0 = time.time()
    rows = [_eval_case(c, skip_graph) for c in cases]

    layers = ["vector@5", "hybrid@8", "hybrid+rerank@5", "status路由", "graph"]
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
        "summary": {l: {"hit": hits[l], "tried": tried[l]} for l in layers if tried[l]},
        "rows": rows,
    }
    out = KB_PATH / EVAL_SET.replace("eval_set.json", "evals_report.json")
    out.write_text(json.dumps(report, ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"报告: {out}")
    return rows
