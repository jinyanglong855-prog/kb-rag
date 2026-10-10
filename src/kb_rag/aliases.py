"""项目别名消解：把用户口语里的简称映射到总库项目——契约在库内，不依赖宿主 agent 的记忆。

事实源是 Shared/索引区/项目别名索引.md（人工维护的权威），本模块只做解析与匹配：
- 解析：表格行的 [[../../<项目名>/...]] wikilink 提取项目名，别名列按顿号/逗号拆分
- 匹配：问题包含别名即命中（大小写不敏感），最长别名优先，最多返回 3 个项目

设计取舍：显式子串匹配，不做意图分类——误命中只多带一份项目状态上下文，
漏命中才是真召回缺失（与状态路由、权威性加权同一哲学）。
"""

import re
from dataclasses import dataclass
from functools import lru_cache

ALIAS_INDEX_REL = "Shared/索引区/项目别名索引.md"
MAX_RESOLVED = 3


@dataclass(frozen=True)
class AliasHit:
    project: str  # 项目目录名（根入口为 ""）
    alias: str  # 命中的别名（原始大小写）


@lru_cache(maxsize=1)
def _load_aliases() -> tuple[tuple[str, str], ...]:
    """(别名原文, 项目名) 列表，长别名在前。文件缺失返回空——消解是增强，不是依赖。"""
    from .config import KB_PATH

    path = KB_PATH / ALIAS_INDEX_REL
    if not path.exists():
        return ()
    hits: list[tuple[str, str]] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip().startswith("|"):
            continue
        cells = [c.strip() for c in line.strip().strip("|").split("|")]
        if len(cells) < 2 or cells[0] in ("用户可能说", "---", ""):
            continue
        m = re.search(r"\[\[\.\./\.\./([^/\]]+)/", cells[1]) or re.search(r"\[\[\.\./\.\./README", cells[1])
        project = m.group(1) if m and m.lastindex else ("" if "README" in cells[1] else None)
        if project is None:
            continue  # 非[[路径]]链接（如 Skill路由索引）不参与项目消解
        for alias in re.split(r"[、,，]", cells[0]):
            alias = alias.strip()
            if len(alias) >= 2:
                hits.append((alias, project))
    return tuple(sorted(set(hits), key=lambda x: -len(x[0])))


def alias_pairs() -> list[tuple[str, str]]:
    """(别名, 项目名) 全量列表——manifest 输出契约用。"""
    return list(_load_aliases())


def resolve_aliases(question: str) -> list[AliasHit]:
    """问题→命中项目列表：子串匹配（大小写与空白不敏感——"TK 选品"和"tk选品"都命中），
    按命中别名的具体程度排序，同一项目只出现一次。"""
    q = re.sub(r"\s+", "", question.lower())
    out: list[AliasHit] = []
    seen: set[str] = set()
    for alias, project in _load_aliases():
        if re.sub(r"\s+", "", alias.lower()) in q and project not in seen:
            seen.add(project)
            out.append(AliasHit(project=project, alias=alias))
            if len(out) >= MAX_RESOLVED:
                break
    return out
