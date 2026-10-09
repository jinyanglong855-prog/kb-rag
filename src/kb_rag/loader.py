"""加载与分块：Markdown 感知切分（语料）+ 源码文件整载（项目文件区），保留路径与标题面包屑。

语料范围：总库全部 md + 已迁入项目文件区的源码/配置（py/ts/js/json/toml/yaml/sh/sql）。
仓库运行态（.venv、__pycache__、索引产物、构建产物）不进语料——它们是可重建缓存，不是知识。
"""

from dataclasses import dataclass
from pathlib import Path

from .config import CHUNK_SIZE

SKIP_DIRS = {
    ".obsidian", ".git", "node_modules", ".trash",
    ".venv", "venv", "__pycache__", ".pytest_cache",
    "index", "dist", "build", ".idea", ".vscode",
}
TEXT_EXTS = {
    ".md",
    ".py", ".ts", ".tsx", ".js", ".mjs", ".json",
    ".toml", ".yaml", ".yml", ".sh", ".sql",
}
MAX_FILE_CHARS = 200_000  # 超过视为数据文件而非文档，不进语料


@dataclass(frozen=True)
class Chunk:
    text: str
    path: str  # 相对总库根的路径
    heading: str  # 所在标题面包屑（源码文件为"源码"）


def load_text_files(root: Path) -> list[tuple[Path, str]]:
    """返回 (文件, 相对路径)。

    语料规则：md 全收（项目状态区的知识主体）；非 md（源码/配置）只收
    `项目文件/` 仓库区和 `Shared/` 工具脚本——项目状态区里的 json 等是
    原子数据快照（候选池、原始抓取），不是知识，进语料只会淹没检索。
    """
    out: list[tuple[Path, str]] = []
    for p in sorted(root.rglob("*")):
        if not p.is_file():
            continue
        suffix = p.suffix.lower()
        if suffix not in TEXT_EXTS:
            continue
        rel_parts = p.relative_to(root).parts
        if any(part in SKIP_DIRS or part.startswith(".") for part in rel_parts):
            continue
        if suffix != ".md":
            if "项目文件" not in rel_parts and rel_parts[0] != "Shared":
                continue
            if p.stat().st_size > MAX_FILE_CHARS:
                continue
        out.append((p, p.relative_to(root).as_posix()))
    return out


# 兼容旧名：索引与检索仍以"知识库文件"视角调用
load_markdown_files = load_text_files


def chunk_file(text: str, path: str, is_markdown: bool) -> list[Chunk]:
    if not is_markdown:
        # 源码块前置路径头：向量嵌入携带文件身份，中文问题才能对上英文代码
        body = f"{path}（源码文件）\n{text.strip()[:MAX_FILE_CHARS]}"
        return _split_long(body, path, "源码")
    return chunk_markdown(text, path)


def chunk_markdown(text: str, path: str) -> list[Chunk]:
    """按二级标题切块；单块超限再按段落二次切。"""
    chunks: list[Chunk] = []
    heading = ""
    buf: list[str] = []

    def flush() -> None:
        if not buf:
            return
        body = "\n".join(buf).strip()
        if body:
            chunks.extend(_split_long(body, path, heading))
        buf.clear()

    for line in text.splitlines():
        if line.startswith("## "):
            flush()
            heading = line[3:].strip()
            continue
        if line.startswith("# "):  # 一级标题作为新上下文起点
            flush()
            heading = line[2:].strip()
            continue
        buf.append(line)
    flush()
    return chunks


def _split_long(body: str, path: str, heading: str) -> list[Chunk]:
    if len(body) <= CHUNK_SIZE:
        return [Chunk(text=body, path=path, heading=heading)]
    parts, current = [], ""
    for para in body.split("\n\n"):
        if current and len(current) + len(para) > CHUNK_SIZE:
            parts.append(current)
            current = ""
        current = f"{current}\n\n{para}".strip()
    if current:
        parts.append(current)
    # 无空行的密集文本（源码常见）：段落切不开时按行硬切
    final: list[str] = []
    for part in parts:
        if len(part) <= CHUNK_SIZE * 1.5:
            final.append(part)
            continue
        line_block, size = [], 0
        for line in part.splitlines():
            if size + len(line) > CHUNK_SIZE and line_block:
                final.append("\n".join(line_block))
                line_block, size = [], 0
            line_block.append(line)
            size += len(line) + 1
        if line_block:
            final.append("\n".join(line_block))
    return [Chunk(text=p, path=path, heading=heading) for p in final]
