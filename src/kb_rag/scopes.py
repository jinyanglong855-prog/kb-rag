"""调用者权限规则表（Phase 3 多人）：按 caller 身份解析读/写范围——单一事实源。

事实源是仓库根 scopes.json（可版本管理，无密钥）：
{
  "default": {"read": ["*"], "write": ["*"]},   ← 未登记 caller 的兜底（今天=全放开）
  "callers": {
    "finance": {"read": ["公司智能体"], "write": ["公司智能体"]}
  }
}
范围语义：一级目录名列表（整个子树可读/写），"*" 放开全部；根目录文件
（STATUS.md 等）仅 "*" 可写；读侧同理。条目不在表内的 caller 落到 default。

环境变量：
- KB_SCOPES：规则表路径覆盖（测试/多部署用；缺省 = 仓库根 scopes.json）
- KB_WRITE_SCOPE：写全局上限——与规则表范围取交集，任何 caller 都越不过它
  （把某次部署整体锁死的开关；不影响读）

失败语义（对抗性审查后收紧，fail-closed）：
- 规则表文件不存在 → 单机零配置，default 全开（保持单用户现状零成本）
- 规则表存在但 JSON 坏/顶层非法 → **deny-all** + stderr 告警——表坏了不许
  静默变成全放开
- 已登记 caller 的范围键是空列表 → 明确无权限（deny）；键缺失 → 继承 default
- 表存在但 caller 未登记且无 default 键 → deny（不猜）

设计取向：未登记 caller 全放开（单人现状零成本），要收紧时在 callers 里
加条目即可——权限收紧是加条目，不是改代码。
"""

import json
import os
import sys
from functools import lru_cache
from pathlib import Path

from .config import REPO_ROOT

READ = "read"
WRITE = "write"
_INVALID = "__invalid__"  # 规则表损坏哨兵：存在但解析失败 → deny-all
_warned = False


def _scopes_path() -> Path:
    override = os.environ.get("KB_SCOPES", "").strip()
    return Path(override) if override else REPO_ROOT / "scopes.json"


@lru_cache(maxsize=8)
def _load(path: str, mtime: float) -> dict:
    p = Path(path)
    if not p.exists():
        return {}  # 无规则表 = 单机零配置，default 全开
    try:
        table = json.loads(p.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return {_INVALID: True}
    return table if isinstance(table, dict) else {_INVALID: True}


def _table() -> dict:
    p = _scopes_path()
    return _load(str(p), p.stat().st_mtime if p.exists() else 0)


def _parse(raw: str | None) -> set[str]:
    if raw is None:
        return set()
    return {s.strip() for s in raw.split(",") if s.strip()}


def _deny_all() -> set[str]:
    global _warned
    if not _warned:
        print("kb-rag scopes: 规则表损坏或条目非法，已按 deny-all 处理（修复 scopes.json 后自动恢复）",
              file=sys.stderr)
        _warned = True
    return {"__none__"}


def resolve(caller: str, kind: str = READ) -> set[str]:
    """caller → 允许的一级目录集合（或 {"*"}）。kind: scopes.READ / scopes.WRITE。"""
    table = _table()
    if table.get(_INVALID):
        return _deny_all()
    if not table:  # 无规则表：单机零配置，全开
        return {"*"}
    entry = table.get("callers", {}).get(caller)
    if entry is None:
        entry = table.get("default")
    if not isinstance(entry, dict):
        return _deny_all()  # 表存在但 caller 未登记且无 default：不猜，拒绝
    scope_list = entry.get(kind, table.get("default", {}).get(kind))
    if scope_list is None:
        return _deny_all()
    allowed = set(scope_list) or {"__none__"}  # 显式空列表 = 明确无权限
    if kind == WRITE:  # 写全局上限：环境变量存在时取交集
        cap = _parse(os.environ.get("KB_WRITE_SCOPE"))
        if cap and "*" not in cap:
            allowed = (allowed - {"*"}) & cap if "*" not in allowed else cap
    return allowed or {"__none__"}


def in_scope(rel_path: str, allowed: set[str]) -> bool:
    """相对路径是否在允许集合内：一级目录匹配；根目录文件仅 "*" 放行。"""
    if "*" in allowed:
        return True
    top = rel_path.split("/", 1)[0]
    return "/" in rel_path and top in allowed


def effective(default: str = "local") -> str:
    """本进程生效身份：KB_CALLER 环境变量；server 传 default="dsh"，CLI 用 "local"。"""
    return os.environ.get("KB_CALLER", "").strip() or default


def describe() -> dict:
    """管理视图：规则表位置 + 每个 caller 的实际生效范围（含 KB_WRITE_SCOPE 交集效果）。"""
    table = _table()
    cap = _parse(os.environ.get("KB_WRITE_SCOPE"))
    names = ["default", *table.get("callers", {}).keys()]
    return {
        "rules_file": str(_scopes_path()),
        "global_write_cap": sorted(cap) if cap else "*",
        "callers": {
            name: {
                "read": sorted(resolve(name, READ)),
                "write": sorted(resolve(name, WRITE)),
            }
            for name in names
        },
    }
