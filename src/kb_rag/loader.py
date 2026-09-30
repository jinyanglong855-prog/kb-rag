"""加载与分块：Markdown 感知的朴素切分，保留路径与标题面包屑。"""

from dataclasses import dataclass
from pathlib import Path

from .config import CHUNK_SIZE

SKIP_DIRS = {".obsidian", ".git", "node_modules", ".trash"}


@dataclass(frozen=True)
class Chunk:
    text: str
    path: str  # 相对知识库根的路径
    heading: str  # 所在标题面包屑


def load_markdown_files(root: Path) -> list[Path]:
    return sorted(
        p for p in root.rglob("*.md")
        if not any(part in SKIP_DIRS for part in p.parts)
    )


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
    return [Chunk(text=p, path=path, heading=heading) for p in parts]
