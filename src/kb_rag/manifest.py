"""机器可读契约（kb-rag manifest）：插件自描述，宿主 agent / 多用户接入的入口。

设计原则：生成而非手抄——projects 解析自根 STATUS frontmatter 的
project_snapshot（快照缺失时回退扫描含 项目状态/ 的目录），aliases 解析自
Shared/索引区/项目别名索引.md；conventions 只放少量不可推导的契约字段，
并指向权威文件，不复制会漂移的内容。

验收标准：一个零上下文的 agent 只凭本 manifest + 工具描述，就能正确完成
"查询 + 写入 + 让知识可被检索"，不需要问任何人。
"""

import json
from datetime import datetime, timezone
from importlib.metadata import PackageNotFoundError, version as pkg_version
from pathlib import Path

from .aliases import alias_pairs
from .config import KB_PATH, INDEX_DIR, validate_kb_path
from .search import staleness
from .writer import LIFECYCLE_VALUES, REQUIRED_FRONTMATTER

# 对外契约字段名沿用 KNOWLEDGE_FRONTMATTER，值与写路径校验同源（writer）——
# 承诺（manifest）与执行（writer）必须是同一份常量，否则零上下文 agent 凭
# manifest 写入会被拒（审查发现：双份拷贝会漂移）
KNOWLEDGE_FRONTMATTER = REQUIRED_FRONTMATTER


def _version() -> str:
    try:
        return pkg_version("kb-rag")
    except PackageNotFoundError:
        return "unknown"


def _root_frontmatter(root: Path) -> dict[str, str]:
    text = (root / "STATUS.md").read_text(encoding="utf-8", errors="ignore") if (root / "STATUS.md").exists() else ""
    if not text.startswith("---"):
        return {}
    out = {}
    for line in text.split("---", 2)[1].splitlines():
        if ":" in line:
            k, v = line.split(":", 1)
            out[k.strip()] = v.strip()
    return out


def _projects(root: Path) -> dict[str, dict]:
    """项目清单：优先根 STATUS 的 project_snapshot；缺失则扫描含 项目状态/ 的一级目录。"""
    out: dict[str, dict] = {}
    snap = _root_frontmatter(root).get("project_snapshot", "").strip().strip("'\"")
    if snap:
        try:
            for name, info in json.loads(snap).items():
                out[name] = {"updated": info.get("updated", ""), "sync_id": info.get("sync_id", "")}
        except json.JSONDecodeError:
            pass
    if not out:  # 回退：目录扫描（快照缺失也能自描述）
        for d in sorted(root.iterdir()):
            if d.is_dir() and (d / "项目状态").is_dir():
                out[d.name] = {"updated": "", "sync_id": ""}
    for name, info in out.items():
        ps = root / name / "项目状态"
        entry = next((f"{name}/项目状态/{n}" for n in ("STATUS.md", "README.md") if (ps / n).exists()), None)
        info["status_entry"] = entry
        info["lifecycle"] = ""
        if entry:
            head = (root / entry).read_text(encoding="utf-8", errors="ignore")[:400]
            for line in head.splitlines():
                if line.startswith("lifecycle:"):
                    info["lifecycle"] = line.split(":", 1)[1].strip()
                    break
    return out


def _index_stats() -> dict:
    idx = Path(INDEX_DIR)
    stats = {"chunks": 0, "indexed_files": 0, "wiki_nodes": 0}
    if (idx / "chunks.json").exists():
        stats["chunks"] = len(json.loads((idx / "chunks.json").read_text(encoding="utf-8")))
    if (idx / "meta.json").exists():
        stats["indexed_files"] = len(json.loads((idx / "meta.json").read_text(encoding="utf-8"))["hashes"])
    if (idx / "wikitree.json").exists():
        stats["wiki_nodes"] = len(json.loads((idx / "wikitree.json").read_text(encoding="utf-8"))["nodes"])
    return stats


def _caller_scope() -> dict:
    """当前进程身份的生效读写范围——零上下文 agent 读 manifest 即知自身边界。"""
    from .scopes import READ, WRITE, effective, resolve

    cid = effective("local")
    return {
        "id": cid,
        "id_source": "KB_CALLER 环境变量（缺省 local；DSH 每个角色一个 MCP 实例各设一个）",
        "read": sorted(resolve(cid, READ)),
        "write": sorted(resolve(cid, WRITE)),
        "rules": "scopes.json（仓库根）；范围=一级目录列表，\"*\"=全部；越界读写会被拒绝并记入审计",
    }


def build_manifest() -> dict:
    root = validate_kb_path()
    return {
        "plugin": "kb-rag",
        "version": _version(),
        "generated_at": datetime.now(timezone.utc).astimezone().isoformat(timespec="seconds"),
        "vault": {
            "root": str(root),
            "corpus_rules": "md 全收；非 md 源码/配置只收 项目文件/ 仓库区与 Shared/；运行态缓存不进语料——权威规则见 src/kb_rag/loader.py",
        },
        "usage": {
            "ask": 'kb-rag ask "问题" — 带引用问答；状态类问题自动走结构层，口语简称在库内消解',
            "query": 'kb-rag query "问题" --mode hybrid --rerank — 纯检索不生成',
            "wiki": 'kb-rag wiki ["问题"] — 结构层摘要树/目录路由',
            "add": "kb-rag add — 增量入库（收口必跑；按文件指纹只重嵌变更，无变化秒级返回）",
            "ingest": "kb-rag ingest <文件> [--to 库内目录] — PDF/Word/PPT/Excel/HTML/EPUB 转 md 落库并增量入库",
            "index": "kb-rag index --wiki — 全量/分区重建（结构大改或规则变更后才需要）",
            "write": "kb-rag write --path <库内相对路径> [--from 文件] — 写路径唯一入口（frontmatter 强制校验）",
            "scopes": "kb-rag scopes — 权限规则管理视图（各 caller 实际生效读写范围）",
            "audit": "kb-rag audit [--stats] — 审计日志消费（谁在何时查/写了什么）",
            "manifest": "kb-rag manifest — 本契约",
        },
        "mcp": {
            "serve": "uv run kb-rag serve（= kb-rag-mcp）— MCP stdio server，stdio 传输",
            "tool_namespace": "宿主按 mcp__<serverName>__<name> 命名空间化，如 mcp__kb__kb_ask",
            "tools": ["kb_manifest", "kb_ask", "kb_vector_search", "kb_hybrid_search",
                      "kb_status_lookup", "kb_graph_query", "kb_read_file", "kb_write"],
            "identity": "每个 agent 实例在 mcpServers.env 里配不同 KB_CALLER = 不同角色（读写范围见本契约 caller 段）；kb_write 不接受调用方自报身份",
            "env_notes": "嵌入模型缓存已钉到仓库 .cache/fastembed（不依赖宿主 TMPDIR/HOME）",
            "server_config": "完整 mcpServers 配置示例见仓库 README「插件契约」节或 src/kb_rag/server.py 模块 docstring",
        },
        "write_result": {
            "ok": "true=已落盘；sync.mode=clean/delta 时立即可检索",
            "sync_mode": "clean=内容无变化 | delta=增量入库完成 | full=首次自动全量 | error=入库同步失败（文件已落盘但暂不可检索，等下次 add；error 详情见 error 字段）",
        },
        "projects": _projects(root),
        "aliases": {a: p for a, p in alias_pairs()},
        "aliases_note": "别名→项目目录名；空值=指向总库根（如\"知识库\"\"项目文件总库\"）",
        "caller": _caller_scope(),
        "env": {
            "KB_PATH": "知识库根目录（缺省读仓库 .env）",
            "INDEX_DIR": "索引产物目录（缺省 = 仓库根 index/；接第二个库时必须单独设置，否则索引会混进本仓库 index/）",
            "KB_CALLER": "调用者身份（scope 判定与审计；缺省 local）",
            "KB_SCOPES": "权限规则表路径覆盖（缺省 = 仓库根 scopes.json）",
            "KB_WRITE_SCOPE": "写全局上限（与角色范围取交集；部署级锁死开关）",
        },
        "conventions": {
            "knowledge_frontmatter": KNOWLEDGE_FRONTMATTER,
            "lifecycle_values": LIFECYCLE_VALUES,
            "single_authority": "一个事实一个权威位置；其他位置最多 3-5 条摘要加链接（权威：根 README 维护边界）",
            "write_routing": "见总库根 README「写入路由」表（权威），本字段仅为指针",
            "status_maintenance": "状态页替换式维护，不追加日期流水；历史进 CHANGELOG/工作日志",
            "closeout": [
                "更新受影响项目的状态入口（项目状态/STATUS.md 或 README.md）",
                "更新根 STATUS.md 对应行与 project_snapshot",
                "uv run kb-rag add 增量入库，使总结收口即可检索",
            ],
        },
        "index": _index_stats(),
        "freshness": staleness(),
    }


def manifest_text() -> str:
    return json.dumps(build_manifest(), ensure_ascii=False, indent=2)
