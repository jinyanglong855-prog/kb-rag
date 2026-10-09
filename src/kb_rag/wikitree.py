"""Wiki 化结构层（阶段 2）：按 Obsidian 目录结构生成层级摘要树（PageIndex/RAPTOR 思路）。

为什么需要这一层：切块检索不理解"项目状态"该看哪一页——STATUS 长文切块后，
最新卡点所在块可能永远进不了 top-k（忠实但召回不全）。结构层把"该看哪份文件"
变成一等检索对象：节点摘要可被向量命中，命中后整份状态文件进入问答上下文。

设计取舍：
- 节点=目录（深度 ≤2），本项目知识库本身就有 README→STATUS 层级，天然适合
  "目录树+节点摘要"；不为叶子文件建节点，状态路由只需要项目粒度。
- 摘要由 LLM 生成（RAPTOR 机制），但按内容指纹缓存——状态文件没改就不重花钱；
  无 API Key 或调用失败时回退到规则摘要（frontmatter+正文开头），构建永不离线失败。
- 状态文件全文在查询时现读，不落索引——内容永远是最新的，索引只承载"路由"。
"""

import hashlib
import json
import re
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path

import numpy as np

from .config import INDEX_DIR, validate_kb_path
from .embedder import embed

MAX_SUMMARY_INPUT = 3000  # 送 LLM 的目录资料上限（字符）
FILE_TEXT_CHARS = 4000  # 命中节点后带入问答上下文的状态文件上限
TREE_JSON = "wikitree.json"
TREE_VECTORS = "wiki_vectors.npy"

SUMMARY_PROMPT = """你是知识库摘要器。根据以下目录资料写一段不超过3句的中文摘要，覆盖：当前阶段/进展、卡点或等待点、下一步。
只依据资料，不编造；资料不足就少说。

目录：{path}（{title}）
资料：
{body}

直接输出摘要正文，不要任何前后缀。"""


@dataclass
class WikiNode:
    path: str  # 相对知识库根的目录路径，根为 ""
    title: str
    summary: str
    status_file: str | None  # 权威状态文件（STATUS.md 优先，否则 README.md）
    children: list[str] = field(default_factory=list)
    summary_kind: str = "rule"  # llm | rule
    fingerprint: str = ""


def _frontmatter(text: str) -> dict[str, str]:
    if not text.startswith("---"):
        return {}
    out = {}
    for line in text.split("---", 2)[1].splitlines():
        if ":" in line:
            k, v = line.split(":", 1)
            out[k.strip()] = v.strip()
    return out


def _section(text: str, name: str) -> str:
    m = re.search(rf"^## {re.escape(name)}.*?$(.*?)(?=^## |\Z)", text, re.M | re.S)
    return m.group(1).strip() if m else ""


def _status_files(dir_path: Path) -> list[str]:
    """目录对应的权威状态文件（相对该目录）：STATUS.md 优先，其次 README.md。"""
    out = []
    for name in ("STATUS.md", "README.md"):
        if (dir_path / name).exists():
            out.append(name)
    return out


def _project_status_file(dir_path: Path) -> str | None:
    """项目节点（含 项目状态/ 子目录）的状态文件，带 项目状态/ 前缀。"""
    ps = dir_path / "项目状态"
    if not ps.is_dir():
        return None
    for name in ("STATUS.md", "README.md"):
        if (ps / name).exists():
            return f"项目状态/{name}"
    return None


def _summary_input(root: Path, node: WikiNode) -> str:
    """目录→摘要输入：有状态/主页文件的目录吃文件内容，纯文件目录吃文件清单。"""
    d = root / node.path if node.path else root
    parts: list[str] = []
    # 项目节点的 status_file 带 项目状态/ 前缀；区域/根节点直接用目录内文件
    for name in ([node.status_file] if node.status_file else []):
        f = d / name
        text = f.read_text(encoding="utf-8", errors="ignore")
        meta = _frontmatter(text)
        parts.append(f"{name} 元信息: " + "，".join(f"{k}={v}" for k, v in meta.items() if k in ("status", "updated")))
        body = text.split("---", 2)[-1]
        if name.endswith("STATUS.md"):
            parts.append(body[:1500])
        else:
            sec = _section(body, "当前状态") or _section(body, "当前阶段") or body[:800]
            parts.append(sec[:1200])
    if not parts:  # 无 README/STATUS 的目录（如 Shared 子区）：文件清单即结构事实
        names = sorted(p.relative_to(root).as_posix() for p in d.iterdir() if not p.name.startswith("."))
        parts.append("包含文件: " + "；".join(names[:15]))
    return "\n".join(parts)[:MAX_SUMMARY_INPUT]


def _rule_summary(root: Path, node: WikiNode) -> str:
    d = root / node.path if node.path else root
    if node.status_file:
        text = (d / node.status_file).read_text(encoding="utf-8", errors="ignore")
        meta = _frontmatter(text)
        head = f"[{meta.get('status', '')}/{meta.get('updated', '')}] " if meta else ""
        return head + text.split("---", 2)[-1].strip()[:200]
    names = sorted(p.name for p in d.iterdir() if not p.name.startswith("."))[:8]
    return "包含：" + "、".join(names)


def _llm_summary(node: WikiNode, body: str) -> str | None:
    try:
        from openai import OpenAI

        from .config import LLM_API_KEY, LLM_BASE_URL, LLM_MODEL

        if not LLM_API_KEY:
            return None
        client = OpenAI(api_key=LLM_API_KEY, base_url=LLM_BASE_URL)
        resp = client.chat.completions.create(
            model=LLM_MODEL,
            messages=[{"role": "user", "content": SUMMARY_PROMPT.format(path=node.path or "知识库根", title=node.title, body=body)}],
            temperature=0.2,
        )
        return resp.choices[0].message.content.strip() or None
    except Exception:
        return None  # 摘要失败不阻塞索引，回退规则摘要


def _discover(root: Path) -> list[WikiNode]:
    """两级结构：根 + 项目节点（含 项目状态/，状态文件从项目状态区取）+ 区域节点（如 Shared，展开一层子目录）。"""
    nodes = [WikiNode(path="", title="知识库根", summary="", status_file=None)]
    skip = {".obsidian", ".git", ".trash", "node_modules"}
    for name in _status_files(root):  # 根节点状态文件（STATUS.md 优先）
        nodes[0].status_file = name
        break
    for depth1 in sorted(root.iterdir()):
        if not depth1.is_dir() or depth1.name in skip or depth1.name.startswith(".") or depth1.name.startswith("_"):
            continue
        rel1 = depth1.name
        n = WikiNode(path=rel1, title=rel1, summary="", status_file=None)
        project_status = _project_status_file(depth1)
        if project_status:
            n.status_file = project_status  # 项目节点：不展开 项目文件/ 仓库区
        else:
            for name in _status_files(depth1):
                n.status_file = name
                break
            for depth2 in sorted(depth1.iterdir()):
                if not depth2.is_dir() or depth2.name.startswith(".") or depth2.name.startswith("_"):
                    continue
                m = WikiNode(path=f"{rel1}/{depth2.name}", title=depth2.name, summary="", status_file=None)
                for name in _status_files(depth2):
                    m.status_file = name
                    break
                nodes.append(m)
        nodes.append(n)
    for n in nodes:
        n.children = sorted(
            m.path for m in nodes
            if m.path.startswith(n.path + "/") and m.path.count("/") == n.path.count("/") + 1
        ) if n.path else []
    return nodes


def build_wiki(use_llm: bool = True) -> int:
    root = validate_kb_path()
    t0 = time.time()
    nodes = _discover(root)
    old = {}
    tree_path = Path(INDEX_DIR) / TREE_JSON
    if tree_path.exists():
        old = {n["path"]: n for n in json.loads(tree_path.read_text(encoding="utf-8"))["nodes"]}

    llm_used = 0
    for n in nodes:
        body = _summary_input(root, n)
        fp = hashlib.sha1(f"{n.title}|{body}".encode()).hexdigest()[:12]
        n.fingerprint = fp
        if n.path in old and old[n.path].get("fingerprint") == fp and old[n.path].get("summary"):
            if not (use_llm and old[n.path].get("summary_kind") == "rule"):
                n.summary = old[n.path]["summary"]  # 内容没变，复用摘要不重花钱
                n.summary_kind = old[n.path].get("summary_kind", "rule")
                continue
            # 规则回退产物在额度可用时重试 LLM（如 429 余额不足期的降级摘要）
        n.summary = _llm_summary(n, body) if use_llm else None
        if n.summary:
            n.summary_kind = "llm"
            llm_used += 1
        else:
            n.summary = _rule_summary(root, n)

    texts = [f"{n.path} {n.title}\n{n.summary}" for n in nodes]
    vectors = np.array(embed(texts), dtype=np.float32)
    vectors /= np.maximum(np.linalg.norm(vectors, axis=1, keepdims=True), 1e-10)

    INDEX_DIR.mkdir(exist_ok=True)
    tree_path.write_text(json.dumps({"nodes": [asdict(n) for n in nodes]}, ensure_ascii=False, indent=1), encoding="utf-8")
    np.save(INDEX_DIR / TREE_VECTORS, vectors)
    print(f"Wiki 摘要树完成: {len(nodes)} 节点 / LLM 摘要 {llm_used} 个（其余缓存或规则回退）/ {time.time() - t0:.1f}s")
    return len(nodes)


def _load() -> tuple[list[WikiNode], np.ndarray]:
    data = json.loads((Path(INDEX_DIR) / TREE_JSON).read_text(encoding="utf-8"))
    nodes = [WikiNode(**n) for n in data["nodes"]]
    return nodes, np.load(Path(INDEX_DIR) / TREE_VECTORS)


def wiki_query(question: str, k: int = 3) -> list[dict]:
    """问题→目录节点：节点摘要余弦检索；命中后现读整份状态文件作上下文。"""
    root = validate_kb_path()
    nodes, vectors = _load()
    q = np.array(embed([question])[0], dtype=np.float32)
    q /= max(np.linalg.norm(q), 1e-10)
    scores = vectors @ q
    out = []
    for i in np.argsort(-scores)[:k]:
        n = nodes[int(i)]
        file_text = ""
        if n.status_file:
            f = (root / n.path / n.status_file) if n.path else (root / n.status_file)
            if f.exists():
                file_text = f.read_text(encoding="utf-8", errors="ignore")[:FILE_TEXT_CHARS]
        out.append({"path": n.path or "（根目录）", "title": n.title, "summary": n.summary,
                    "status_file": n.status_file, "file_text": file_text, "score": round(float(scores[i]), 4)})
    return out


def show_tree() -> None:
    nodes, _ = _load()
    for n in sorted(nodes, key=lambda x: x.path.count("/")):
        print(f"\n== {n.path or '（根）'} [{n.summary_kind}] -> {n.status_file or '无状态文件'}")
        print(f"   {n.summary[:120]}")
