"""入库转换层：PDF/DOCX/TXT → Markdown 落入总库，随后自动增量入库。

设计取舍：
- 转换是"提取"不是"排版复原"——目标是让内容可检索，不追求完美还原版式。
- 扫描版 PDF（图片型）提取不到文本层，明确报错提示走 OCR，不静默产出空文件。
- 输出默认进 Shared/参考资料/入库文件/，带来源 frontmatter；转换后自动跑增量索引。
"""

from datetime import date
from pathlib import Path

from .config import KB_PATH, validate_kb_path

DEFAULT_TARGET = "Shared/参考资料/入库文件"
SOURCE_NOTE = """---
type: ingested
source: {source}
ingested: {today}
---

"""


def _pdf_to_markdown(path: Path) -> str:
    import fitz  # pymupdf

    doc = fitz.open(path)
    pages = []
    for i, page in enumerate(doc, 1):
        text = page.get_text("text").strip()
        if text:
            pages.append(f"## 第 {i} 页\n\n{text}")
    doc.close()
    if not pages:
        raise ValueError("PDF 没有可提取的文本层（可能是扫描版），请先做 OCR 或换文字版")
    return "\n\n".join(pages)


def _docx_to_markdown(path: Path) -> str:
    import docx

    d = docx.Document(path)
    lines: list[str] = []
    for para in d.paragraphs:
        text = para.text.strip()
        if not text:
            continue
        style = (para.style.name or "").lower()
        if style.startswith("heading"):
            try:
                level = int(style.replace("heading", "").strip() or "1")
            except ValueError:
                level = 2
            lines.append(f"{'#' * min(level + 1, 6)} {text}")
        else:
            lines.append(text)
    if not lines:
        raise ValueError("DOCX 没有可提取的正文")
    return "\n\n".join(lines)


CONVERTERS = {
    ".pdf": _pdf_to_markdown,
    ".docx": _docx_to_markdown,
    ".txt": lambda p: p.read_text(encoding="utf-8", errors="ignore"),
    ".md": lambda p: p.read_text(encoding="utf-8", errors="ignore"),
}


def ingest_file(source: str, target_dir: str | None = None) -> Path:
    """转换单个文件 → md 落入总库，返回输出路径。不自动建索引（由 CLI 决定）。"""
    src = Path(source).expanduser().resolve()
    if not src.is_file():
        raise SystemExit(f"文件不存在: {src}")
    converter = CONVERTERS.get(src.suffix.lower())
    if converter is None:
        raise SystemExit(f"不支持的格式: {src.suffix}（当前支持 pdf/docx/txt/md）")
    validate_kb_path()

    body = converter(src)
    out_dir = KB_PATH / (target_dir or DEFAULT_TARGET)
    out_dir.mkdir(parents=True, exist_ok=True)
    out = out_dir / f"{src.stem}.md"
    front = SOURCE_NOTE.format(source=src.as_posix(), today=date.today().isoformat())
    if body.lstrip().startswith("---"):  # 已有 frontmatter 则不再包一层
        front = ""
    out.write_text(front + body + "\n", encoding="utf-8")
    return out
