"""审计消费侧：读 index/audit.jsonl，回答"谁在何时查/写了什么"。

Phase 3 权限层配套——审计只写不读等于没有审计。本模块是审计唯一的读取口，
与 writer.AUDIT_PATH 共用事实源；写入侧见 writer._append_audit。
"""

import json
from collections import Counter
from datetime import datetime

from .writer import AUDIT_PATH


def log_read(caller: str, tool: str, arg: str = "", ok: bool = True, error: str = "") -> None:
    """查询审计统一入口（agent.run_tool / ask / CLI query 共用）。

    失败查询（ok=False）也必须留痕——失败恰是调查最关心的时刻，
    不给探测者留"无痕试错"通道。审计失败不阻塞主流程。
    """
    try:
        AUDIT_PATH.parent.mkdir(parents=True, exist_ok=True)
        record: dict = {"ts": datetime.now().isoformat(timespec="seconds"),
                        "caller": caller, "kind": "read", "tool": tool,
                        "arg": str(arg)[:200], "ok": ok}
        if error:
            record["error"] = error[:200]
        with AUDIT_PATH.open("a", encoding="utf-8") as f:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")
    except OSError:
        pass


def _records() -> list[dict]:
    if not AUDIT_PATH.exists():
        return []
    out = []
    for line in AUDIT_PATH.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            out.append(json.loads(line))
        except json.JSONDecodeError:
            continue  # 半行/损坏行跳过，不影响其余审计
    return out


def tail(caller: str | None = None, kind: str | None = None,
         limit: int = 20) -> list[dict]:
    """最近 limit 条记录（新在前），可按 caller / kind("read"|"write") 过滤。"""
    out: list[dict] = []
    for r in reversed(_records()):
        if caller and r.get("caller") != caller:
            continue
        if kind and r.get("kind") != kind:
            continue
        out.append(r)
        if len(out) >= limit:
            break
    return out


def stats() -> dict:
    """按 caller×kind 汇总：计数 + 各组最近活跃时间。"""
    counts: Counter = Counter()
    last_ts: dict[tuple, str] = {}
    for r in _records():
        key = (r.get("caller", "?"), r.get("kind", "?"))
        counts[key] += 1
        ts = r.get("ts", "")
        if ts and ts > last_ts.get(key, ""):
            last_ts[key] = ts
    return {
        f"{c}/{k}": {"count": n, "last": last_ts.get((c, k), "")}
        for (c, k), n in sorted(counts.items())
    }


def format_records(records: list[dict]) -> str:
    """人类可读视图：时间 调用者 类别 结果 对象（工具或路径）。

    ok=False（写拒绝 / 读拒绝 / 工具异常）一律显示拒绝与原因；其余显示 ok。
    """
    if not records:
        return "（无审计记录）"
    lines = []
    for r in records:
        what = r.get("rel_path") or f"{r.get('tool', '?')}({r.get('arg', '')})"
        if r.get("ok") is False:
            verdict = f"拒绝: {r.get('error', '?')}"
        else:
            verdict = "ok"
        lines.append(
            f"{r.get('ts', '?')}  {r.get('caller', '?'):<12} "
            f"{r.get('kind', '?'):<5}  {verdict:<20}  {what[:80]}"
        )
    return "\n".join(lines)


def format_stats() -> str:
    s = stats()
    if not s:
        return "（无审计记录）"
    width = max(len(k) for k in s)
    return "\n".join(f"{k:<{width}}  {v['count']:>4} 条  最近 {v['last']}" for k, v in s.items())
