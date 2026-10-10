"""入库转换层：PDF/Office/HTML/EPUB/图片 → Markdown 落入总库，随后自动增量入库。

转换后端是 docling（IBM 开源、LF AI & Data 基金会托管，MIT 许可，68k+ stars），
版面模型负责标题层级、表格、双栏阅读顺序，扫描件/图片自带 OCR——不自研解析。

设计取舍：
- 转换目标是"内容可检索"且结构保真（表格转 md 表、标题转 md 标题），不追求像素级版式复原。
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

# docling 原生解析格式；txt/md 无需解析直接读取
DOCLING_EXTS = {
    ".pdf", ".docx", ".pptx", ".xlsx",
    ".html", ".htm", ".xhtml", ".epub",
    ".png", ".jpg", ".jpeg", ".tif", ".tiff", ".webp", ".bmp",
}
PLAIN_EXTS = {".txt", ".md"}
SUPPORTED_EXTS = DOCLING_EXTS | PLAIN_EXTS

_converter = None


def _docling():
    global _converter
    if _converter is None:
        from docling.document_converter import DocumentConverter

        _converter = DocumentConverter()
    return _converter


def _to_markdown(path: Path) -> str:
    result = _docling().convert(str(path))
    md = result.document.export_to_markdown().strip()
    if not md:
        raise ValueError(f"转换结果为空: {path.name}")
    return md


def ingest_file(source: str, target_dir: str | None = None) -> Path:
    """转换单个文件 → md 落入总库，返回输出路径。不自动建索引（由 CLI 决定）。"""
    src = Path(source).expanduser().resolve()
    if not src.is_file():
        raise SystemExit(f"文件不存在: {src}")
    if src.suffix.lower() not in SUPPORTED_EXTS:
        raise SystemExit(
            f"不支持的格式: {src.suffix}（当前支持 pdf/docx/pptx/xlsx/html/epub/图片/txt/md）"
        )
    validate_kb_path()

    if src.suffix.lower() in PLAIN_EXTS:
        body = src.read_text(encoding="utf-8", errors="ignore")
    else:
        body = _to_markdown(src)
    out_dir = KB_PATH / (target_dir or DEFAULT_TARGET)
    out_dir.mkdir(parents=True, exist_ok=True)
    out = out_dir / f"{src.stem}.md"
    front = SOURCE_NOTE.format(source=src.as_posix(), today=date.today().isoformat())
    if body.lstrip().startswith("---"):  # 已有 frontmatter 则不再包一层
        front = ""
    out.write_text(front + body + "\n", encoding="utf-8")
    return out
