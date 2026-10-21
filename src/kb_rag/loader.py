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
# 非 md 文件的额外排除：运行记录、审计回执、数据快照——它们是过程/数据，不是知识。
# md 不受此清单影响（知识库正文全收，权威性交给加权层）。
NON_MD_SKIP_DIRS = {
    "work", "outputs", "logs", "delivery", "archive", "归档", "predictions",
    "backups", "state", "releases", "history", "数据区", "原始快照",
}
TEXT_EXTS = {
    ".md",
    ".py", ".ts", ".tsx", ".js", ".mjs", ".json",
    ".toml", ".yaml", ".yml", ".sh", ".sql",
}
MAX_FILE_CHARS = 200_000  # md 超限截断；非 md 超限整文件跳过
MAX_NON_MD_BYTES = 100_000  # 非 md 源码/配置上限（vendor 压缩库远超此值）
# 整目录排除（md 与非 md 都不收）：
# ① 项目源码区=项目管理，由 git 与项目 README 承担，知识库只收知识沉淀（项目状态/）——
#    公司智能体仍在构建期、源码+上游文档近万个文件占全库 87%，完整入库拖垮重建与检索
#    （2026-10-21 用户指示：单独拎出暂不入库）；
# ② 整体废弃的项目（2026-10-21 用户指示：带货剪辑与图片转 3D 两个方向彻底废弃，项目目录
#    已删除）。名字保留为墓碑：若日后从备份恢复出同名目录，不会悄悄重新入库。
# 注意：这里只管检索语料与摘要树节点，不删任何磁盘文件。
EXCLUDED_SUBTREES = {
    "公司智能体/项目文件",
    "抖音带货视频剪辑工作台",
    "商品图转 3D 工作台",
}
# 单块硬上限：一行超长文件（内嵌源码的字符串常量、压缩 JSON）段落和行都切不开，
# 曾产出 42782 字块 → 嵌入一次 ~3 万 token 推理抢 8-10GB 被 jetsam 杀
# （2026-10-19 段 72000-76000 七连死真凶）。6000 = 已嵌块实际最大 5765 之上，
# 兜底线上移不拆任何既有块，保证已嵌的 72000 块清单逐字节不变、断点不作废
MAX_CHUNK_CHARS = 6000


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
        if p.is_symlink():
            continue  # symlink 不进语料：FS 级投毒可借他人合法写入把越权内容灌进索引
        if not p.is_file():
            continue
        suffix = p.suffix.lower()
        if suffix not in TEXT_EXTS:
            continue
        rel_posix = p.relative_to(root).as_posix()
        if any(rel_posix == t or rel_posix.startswith(t + "/") for t in EXCLUDED_SUBTREES):
            continue
        rel_parts = p.relative_to(root).parts
        if any(part in SKIP_DIRS or part.startswith(".") for part in rel_parts):
            continue
        if suffix != ".md":
            if "项目文件" not in rel_parts and rel_parts[0] != "Shared":
                continue
            if any(part in NON_MD_SKIP_DIRS for part in rel_parts):
                continue
            if ".min." in p.name:
                continue
            if p.stat().st_size > MAX_NON_MD_BYTES:
                continue
        out.append((p, rel_posix))
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
    """按二级标题切块；单块超限再按段落二次切。

    frontmatter 块（首个 "---" 块）不单独成块——它是元数据不是正文，
    纯 frontmatter 块进索引只会污染检索结果（实测 query top5 曾被其占据 3 席）；
    其内容并入后续首块由嵌入层自然弱化。
    """
    lines = text.splitlines()
    if text.startswith("---"):
        end = next((i for i in range(1, len(lines)) if lines[i].strip() == "---"), None)
        if end is not None:
            lines = lines[end + 1:]
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

    for line in lines:
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
    # 单行超长兜底：行都切不开时按定长窗口硬切（>6000 字的块向量必然稀释，检索不可用）
    windowed: list[str] = []
    for part in final:
        if len(part) <= MAX_CHUNK_CHARS:
            windowed.append(part)
            continue
        windowed.extend(part[i:i + MAX_CHUNK_CHARS]
                        for i in range(0, len(part), MAX_CHUNK_CHARS))
    return [Chunk(text=p, path=path, heading=heading) for p in windowed]
