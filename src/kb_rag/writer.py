"""写路径 API（Phase 2 插件契约）：校验→scope→落盘→审计→自动增量索引。

外部 agent 只能通过本入口写库，schema 由插件强制，不依赖调用方自觉。
写范围（scope）按调用者身份解析（Phase 3）：scopes.resolve(caller, WRITE)，
规则表 = 仓库根 scopes.json；KB_WRITE_SCOPE 环境变量降级为全局上限
（与身份范围取交集）。
"""

import json
from datetime import datetime
from pathlib import Path

from .config import INDEX_DIR, KB_PATH, validate_kb_path
from .scopes import WRITE, in_scope, resolve
from .search import sync_incremental

# 生命周期枚举，与总库治理规则一致
LIFECYCLE_VALUES = ["active", "experimental", "deprecated", "archived", "generated"]
# markdown 文件 frontmatter 必须齐全的字段
REQUIRED_FRONTMATTER = ["type", "updated", "status", "lifecycle"]
# 禁止出现在路径任何一段的目录名；"." 开头的段一律另案拒绝
FORBIDDEN_SEGMENTS = {"项目文件", ".obsidian"}

# 审计文件放 index/（已 gitignore，属运行态），不进知识库版本管理
AUDIT_PATH = INDEX_DIR / "audit.jsonl"


def _parse_frontmatter(text: str) -> dict[str, str]:
    """解析文件开头的 "---" ... "---" 块，逐行按第一个冒号拆 key/value。"""
    if not text.startswith("---"):
        return {}
    meta: dict[str, str] = {}
    for line in text.split("---", 2)[1].splitlines():
        if ":" not in line:
            continue
        key, _, value = line.partition(":")
        meta[key.strip()] = value.strip()
    return meta


def safe_rel(rel: str) -> str | None:
    """路径形状净化（读/写/入库转换共用的事实源）：拒绝非字符串、空、绝对路径、
    空段/"." 段/".."、以 "." 开头的段；返回规范 posix 相对路径，不合规返回 None。

    必须在原始串上判断——pathlib 会折叠 "a//b"、"a/./b"、"a/../b"，
    折叠后 ".." 穿越与首段伪装（"pub/../sec"）都看不出来了。
    """
    if not isinstance(rel, str) or not rel or Path(rel).is_absolute():
        return None
    parts = rel.split("/")
    if any(seg in ("", ".", "..") or seg.startswith(".") for seg in parts):
        return None
    return "/".join(parts)


def _validate_rel_path(rel: str) -> str | Path:
    """写路径校验：返回错误消息（拒绝）或规范化后的目标 Path。"""
    if not isinstance(rel, str):
        return "rel_path 必须是字符串"
    norm = safe_rel(rel)
    if norm is None:
        if not rel:
            return "rel_path 不能为空"
        if Path(rel).is_absolute():
            return f"rel_path 必须是相对路径: {rel}"
        return f"rel_path 包含空路径段、'.'、'..' 或以 '.' 开头的段: {rel}"
    for seg in norm.split("/"):
        if seg in FORBIDDEN_SEGMENTS:
            return f"rel_path 不允许包含保留目录名 {seg!r}: {rel}"
    if not norm.endswith(".md"):
        return f"本阶段只接受 .md 文件: {rel}"
    root = KB_PATH.resolve()
    target = (root / norm).resolve()
    # resolve 后仍在总库之下，防符号链接与路径穿越
    if not target.is_relative_to(root):
        return f"rel_path 越出知识库根目录: {rel}"
    return target


def _validate_content(text: str) -> str | None:
    """校验内容 frontmatter：返回错误消息或 None（通过）。"""
    meta = _parse_frontmatter(text)
    if not meta:
        return "缺少 frontmatter（文件须以 --- 块开头）"
    missing = [k for k in REQUIRED_FRONTMATTER if not meta.get(k)]
    if missing:
        return f"frontmatter 缺少必填字段: {', '.join(missing)}"
    if meta["lifecycle"] not in LIFECYCLE_VALUES:
        return (
            f"lifecycle 值非法: {meta['lifecycle']!r}，"
            f"须为 {'/'.join(LIFECYCLE_VALUES)} 之一"
        )
    return None


def _check_scope(rel_posix: str, allowed: set[str], caller: str) -> str | None:
    """校验写入范围：返回错误消息或 None（通过）。rel_posix 为库内相对路径。"""
    if in_scope(rel_posix, allowed):
        return None
    return f"scope 拒绝: {rel_posix} 不在调用者 {caller!r} 的写范围内（{','.join(sorted(allowed))}）"


def _append_audit(record: dict) -> None:
    """向审计文件追加一行 JSON；审计失败不阻塞主流程。"""
    try:
        AUDIT_PATH.parent.mkdir(parents=True, exist_ok=True)
        with AUDIT_PATH.open("a", encoding="utf-8") as f:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")
    except OSError:
        pass


def write_entry(content: str, rel_path: str, caller: str = "local") -> dict:
    """写路径唯一入口：校验→scope→落盘→审计→自动增量索引。"""
    validate_kb_path()
    allowed = resolve(caller, WRITE)
    bytes_len = len(content.encode("utf-8"))

    target = _validate_rel_path(rel_path)
    if isinstance(target, str):
        error = target
    else:
        # scope 按相对路径判断，不用 resolve 后的绝对路径（/tmp 等符号链接会改变段）
        error = _validate_content(content) or _check_scope(rel_path, allowed, caller)
    if error:
        _append_audit({
            "ts": datetime.now().isoformat(timespec="seconds"),
            "caller": caller,
            "kind": "write",
            "rel_path": rel_path,
            "ok": False,
            "error": error,
            "bytes": bytes_len,
        })
        return {"ok": False, "error": error}

    try:
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(content, encoding="utf-8")
    except OSError as exc:
        # 落盘失败同样必须留审计——失败是调查最关心的时刻，不留无痕通道
        _append_audit({
            "ts": datetime.now().isoformat(timespec="seconds"),
            "caller": caller,
            "kind": "write",
            "rel_path": rel_path,
            "ok": False,
            "error": f"落盘失败: {exc}",
            "bytes": bytes_len,
        })
        return {"ok": False, "error": f"落盘失败: {exc}"}

    # 同步失败不吞掉写结果：ok 保持 True，异常文本放进 error
    sync: dict = {}
    try:
        sync = sync_incremental()
    except Exception as exc:  # noqa: BLE001
        sync = {"mode": "error"}
        error = str(exc)

    result: dict = {"ok": True, "rel_path": rel_path, "bytes": bytes_len, "sync": sync}
    record: dict = {
        "ts": datetime.now().isoformat(timespec="seconds"),
        "caller": caller,
        "kind": "write",
        "rel_path": rel_path,
        "ok": True,
        "bytes": bytes_len,
        "sync": sync,
    }
    if error:
        result["error"] = error
        record["error"] = error
    _append_audit(record)
    return result
