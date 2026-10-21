"""父块扩展（small-to-big，步骤 2/4）：子块管检索命中，父块管回答供给。

切块器按二级标题/整文件顺序产出块，同 (path, heading) 的连续块天然构成父块
（md=章节，源码=整文件）——分组信息不需要存进索引，运行时按行号邻接即可还原。
因此本层不改切块器、不改嵌入输入（缓存键不变），对旧索引立即可用，0.6B 重建
后自动延续。

扩展只发生在 ask 装配层：检索排序与评测口径完全不动。同一父块多次命中去重为
一条（命中已按相关性排序，先到者定义窗口）；父块超预算时以命中块为中心向两
侧交替取兄弟块。父块内容与命中块同文件——scope 过滤在命中时已按 path 判定，
膨胀不引入越权面。行号 idx 由检索层携带（search.query / hybrid_query）。
"""

import json
from pathlib import Path

from .config import INDEX_DIR, PARENT_MAX_CHARS


def _load_chunks() -> list[dict]:
    return json.loads((Path(INDEX_DIR) / "chunks.json").read_text(encoding="utf-8"))


def parent_runs(chunks: list[dict]) -> list[int]:
    """每块所属父块编号：相邻块 (path, heading) 不同则开启新父块。

    按"连续同键"分组而非按键全库合并：同一文件允许出现两个同名标题
    （如两个 "## 备注"），它们是两个章节，不该拼成一个跨区段父块。
    """
    runs = [0] * len(chunks)
    r = 0
    for i in range(1, len(chunks)):
        a, b = chunks[i - 1], chunks[i]
        if a["path"] != b["path"] or a["heading"] != b["heading"]:
            r += 1
        runs[i] = r
    return runs


def _window(chunks: list[dict], members: list[int], center: int, budget: int) -> str:
    """以 center 为中心向两侧交替扩展到预算内，按原顺序拼接。"""
    pos = members.index(center)
    picked = [center]
    size = len(chunks[center]["text"])
    lo, hi = pos - 1, pos + 1
    while True:
        progressed = False
        # 每取一块就用新尺寸重查另一侧：两侧"各自能放下"合起来可能超预算
        if lo >= 0 and size + len(chunks[members[lo]]["text"]) <= budget:
            size += len(chunks[members[lo]]["text"])
            picked.append(members[lo])
            lo -= 1
            progressed = True
        if hi < len(members) and size + len(chunks[members[hi]]["text"]) <= budget:
            size += len(chunks[members[hi]]["text"])
            picked.append(members[hi])
            hi += 1
            progressed = True
        if not progressed:
            break
    return "\n\n".join(chunks[i]["text"] for i in sorted(picked))


def expand(hits: list[dict], budget: int | None = None) -> list[dict]:
    """命中块 → 父块（原顺序、去重后返回）。无行号的命中原样透传（防御老调用方）。"""
    if budget is None:
        budget = PARENT_MAX_CHARS
    if not hits or all("idx" not in h for h in hits):
        return hits
    chunks = _load_chunks()
    runs = parent_runs(chunks)
    seen: set[int] = set()
    out: list[dict] = []
    for h in hits:
        idx = h.get("idx")
        if not isinstance(idx, int) or not 0 <= idx < len(chunks):
            out.append(h)
            continue
        c = chunks[idx]
        if c["path"] != h.get("path") or c["heading"] != h.get("heading"):
            # 检索与扩展之间索引被并发增量重建换序：行号已失义。原样透传——
            # 既不展开成无关文件的内容（scope 只在命中时按 path 判过），也不误导回答
            out.append(h)
            continue
        r = runs[idx]
        if r in seen:  # 同父块已随更靠前的命中带出：让位给下一条来源
            continue
        seen.add(r)
        start = end = idx
        while start > 0 and runs[start - 1] == r:
            start -= 1
        while end + 1 < len(chunks) and runs[end + 1] == r:
            end += 1
        members = list(range(start, end + 1))
        if sum(len(chunks[i]["text"]) for i in members) <= budget:
            text = "\n\n".join(chunks[i]["text"] for i in members)
        else:
            text = _window(chunks, members, idx, budget)
        out.append({**h, "text": text, "parent": True})
    return out
