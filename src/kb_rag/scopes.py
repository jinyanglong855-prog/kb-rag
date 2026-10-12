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

设计取向：未登记 caller 全放开（单人现状零成本），要收紧时在 callers 里
加角色即可——权限收紧是加条目，不是改代码。
"""

import json
import os
from functools import lru_cache
from pathlib import Path

from .config import REPO_ROOT

READ = "read"
WRITE = "write"


def _scopes_path() -> Path:
    override = os.environ.get("KB_SCOPES", "").strip()
    return Path(override) if override else REPO_ROOT / "scopes.json"


@lru_cache(maxsize=8)
def _load(path: str, mtime: float) -> dict:
    p = Path(path)
    if not p.exists():
        return {}
    try:
        return json.loads(p.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return {}  # 规则表坏了不静默放行更安全，但单机场景宁可可用：按空表=default 兜底


def _table() -> dict:
    p = _scopes_path()
    return _load(str(p), p.stat().st_mtime if p.exists() else 0)


def _parse(raw: str | None) -> set[str]:
    if raw is None:
        return set()
    return {s.strip() for s in raw.split(",") if s.strip()}


def resolve(caller: str, kind: str = READ) -> set[str]:
    """caller → 允许的一级目录集合（或 {"*"}）。kind: scopes.READ / scopes.WRITE。"""
    table = _table()
    entry = table.get("callers", {}).get(caller) or table.get("default") or {}
    allowed = set(entry.get(kind) or ["*"])
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
