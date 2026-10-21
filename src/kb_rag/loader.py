"""加载与分块：Markdown 感知切分（语料）+ 源码文件整载（项目文件区），保留路径与标题面包屑。

语料范围：各项目 项目状态/ 与 Shared/ 的 md 为主体（知识沉淀层）；项目文件/ 源码区按项目
整树排除（见 EXCLUDED_SUBTREES，kb-rag 自身源码作为框架演示例外保留）。
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
# 整目录排除（md 与非 md 都不收）。准入判据：知识库只收"知识沉淀"——回答"为什么/怎么做/
# 是什么"、保质期长、独特的一份；不收"项目管理"——源码、逐稿过程产物、数据快照、现势状态
# （状态页由结构层查询时现读）。完整论证：项目状态/决策区/2026-10-21 - 语料准入判据.md。
# ① 项目源码区 项目文件/ ：归 git 与项目 README。公司智能体 2026-10-21 用户指示先行排除
#    （近万文件占 87%）；小红书宠物/个人创作台同日按判据排除（逐稿提示词链约 200 万字符是
#    模板副本海，会稀释检索；web 源码 245 万字符且项目已停服）。kb-rag 自身源码是"企业级
#    框架本体"的演示例外，不列入。
# ② 整体废弃的项目（2026-10-21 用户指示：带货剪辑与图片转 3D 两个方向彻底废弃，项目目录
#    已删除）。名字保留为墓碑：若日后从备份恢复出同名目录，不会悄悄重新入库。
# ③ 数据区：马来西亚候选池爬虫快照与候选商品原始数据（371 篇 md / 70 万字符）是数据查询
#    对象（走 Base/脚本），不是问答对象。
# 注意：这里只管检索语料与摘要树节点，不删任何磁盘文件。
EXCLUDED_SUBTREES = {
    "公司智能体/项目文件",
    "小红书宠物/项目文件",
    "个人创作台/项目文件",
    "个人画像/项目文件",
    "马来西亚 TikTok Shop 真实爆品选品/项目状态/数据区",
    "抖音带货视频剪辑工作台",
    "商品图转 3D 工作台",
}
# 排除树里明确保留的子树（先于排除判断）：小红书 predictions 是"预测→结果"校准账本——带诚实的
# "非盲预测/已删除"标注、全库独一份，是过程迭代里含机制的那种沉淀（2026-10-21 用户裁定保留）。
# 逐稿提示词链则只搬了 4 份代表样例到 小红书宠物/项目状态/参考资料/提示词样例/，原件不进语料。
KEPT_SUBTREES = {
    "小红书宠物/项目文件/小红书宠物/predictions",
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


def _under(rel_posix: str, trees: set[str]) -> bool:
    return any(rel_posix == t or rel_posix.startswith(t + "/") for t in trees)


def is_excluded(rel_posix: str) -> bool:
    """语料准入：保留子树优先于排除子树；两者都是严格的路径段前缀匹配（不是子串）。"""
    return not _under(rel_posix, KEPT_SUBTREES) and _under(rel_posix, EXCLUDED_SUBTREES)


def load_text_files(root: Path) -> list[tuple[Path, str]]:
    """返回 (文件, 相对路径)。

    语料规则：md 按准入规则收（EXCLUDED/KEPT_SUBTREES，知识沉淀层全收）；非 md（源码/配置）
    只收 `项目文件/` 仓库区和 `Shared/` 工具脚本——项目状态区里的 json 等是
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
        if is_excluded(rel_posix):
            continue
        rel_parts = p.relative_to(root).parts
        if any(part in SKIP_DIRS or part.startswith(".") for part in rel_parts):
            continue
        # 索引产物目录按前缀整体跳过：index_qwen/（重建草稿）、index_backup_*/（旧索引备份）
        # 里的 meta/bm25/wikitree json 是可重建缓存，混进语料会被当"知识"检索到
        if any(part.startswith("index_") for part in rel_parts[:-1]):
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
