"""问答生成层：检索 top-k → LLM 生成带引用的回答（RAG 闭环最后一环）。

引用格式强制：回答中的每个关键论断标注 [n]，n 对应检索来源路径——
让"AI 说的"可以追到"库里写的"。

阶段 2 路由：状态类问题（进度/卡点/下一步）先走结构层——目录摘要树命中
项目节点后，整份状态文件进入上下文，解决"STATUS 长文切块召回不全"。
路由是显式关键词规则（与权威性加权同一哲学：可解释，不做黑盒意图分类）。

插件自描述（面向任意宿主 agent / 多用户）：
- 别名消解在库内完成：口语简称（"创作台"/"TK"）→ 项目状态页直接入上下文，
  不依赖宿主 agent 记得查别名索引。
- L2 新鲜度兜底：回答前对比文件指纹，索引过期先增量补齐（本地嵌入，无
  LLM 开销）——"拿旧知识回答"从静默错误变成被修复的前置步骤。
"""

import re

from openai import OpenAI

from . import rewrite, scopes
from .aliases import resolve_aliases
from .config import KB_PATH, LLM_API_KEY, LLM_BASE_URL, LLM_MODEL
from .hybrid import hybrid_query
from .search import staleness, sync_incremental
from .wikitree import FILE_TEXT_CHARS, wiki_query

PROMPT = """你是本地知识库的问答助手。只依据提供的资料回答，不编造。
资料中每条带编号；若包含"状态页（整文件）"，它反映该项目最新状态，优先采信。
回答要求：
1. 关键论断后标注来源编号，如 [1]
2. 资料不足以回答时，明确说"资料中没有提到"，不要猜
3. 中文回答，简洁

资料：
{context}

问题：{question}
"""

# 显式状态意图规则：宁多勿漏——误命中只多带几条状态页上下文，漏命中才是真召回缺失。
# 除关键词外加问句结构信号（"还X吗/要不要/是否"），实测"闲鱼兜底还继续发吗"曾被关键词漏判
STATUS_INTENT = re.compile(
    r"状态|进度|卡|下一步|接下来|进展|做到哪|停在哪|在等|等待|怎么样|还.{0,6}吗|要不要|是否"
    r"|现在|目前|最近|计划|打算|安排|暂停|停止|停了|停掉|取消|放弃|继续"
)

ALIAS_ENTRY_CHARS = 1200  # 非状态问题里，别名命中项目的状态页只带入入口摘要


def _is_status_question(q: str) -> bool:
    return bool(STATUS_INTENT.search(q))


def _project_status_rel(project: str) -> str:
    """项目（"" 为总库根）的权威状态文件相对路径；无则空串。"""
    base = KB_PATH / project if project else KB_PATH
    if project:
        for name in ("STATUS.md", "README.md"):
            if (base / "项目状态" / name).exists():
                return f"{project}/项目状态/{name}"
        return ""
    for name in ("STATUS.md", "README.md"):
        if (base / name).exists():
            return name
    return ""


def _alias_sources(question: str, is_status: bool) -> dict[str, tuple[str, str]]:
    """别名命中项目的状态页 → (状态文件相对路径 -> (来源标签, 内容))。"""
    out: dict[str, tuple[str, str]] = {}
    for hit in resolve_aliases(question):
        rel = _project_status_rel(hit.project)
        if not rel or rel in out:
            continue
        text = (KB_PATH / rel).read_text(encoding="utf-8", errors="ignore")
        limit = FILE_TEXT_CHARS if is_status else ALIAS_ENTRY_CHARS
        name = "总库根" if not hit.project else hit.project
        out[rel] = (f"{rel}（别名“{hit.alias}”→{name}·状态页）", text[:limit])
    return out


def ask(question: str, k: int = 5, caller: str = "local") -> str:
    from .audit import log_read

    if not LLM_API_KEY:
        raise SystemExit("请在 .env 里填 LLM_API_KEY（智谱/DeepSeek 均可，接口兼容 OpenAI SDK）")

    allowed = scopes.resolve(caller, scopes.READ)
    restricted = "*" not in allowed
    log_read(caller, "ask", {"question": question, "k": k})  # 查询审计：问答是最常用的读通道，不能无痕

    # L2 新鲜度兜底：索引过期先增量补齐再回答，保证引用反映库内当前内容。
    # 受限角色跳过自动补齐——只读角色一次提问不该有能力触发全量重嵌（资源型 DoS 面）。
    # 索引缺失不再自动全量构建：换模型后全库重建是小时级任务，问答路径里静默跑会像卡死，
    # 改为给出明确指令（kb-rag index 可断点续传，中断后重跑同一命令接着算）。
    notice = ""
    st = staleness()
    if st["mode"] == "missing":
        return ("索引不存在：请先运行 uv run kb-rag index 建立索引"
                "（全量构建耗时较长，可断点续传）。")
    if st["mode"] != "clean" and not restricted:
        from .indexer import IndexNotQueryable

        try:
            r = sync_incremental()
        except IndexNotQueryable as e:  # 错位索引：拒绝原因作为回答，不裸崩
            return str(e)
        if r["mode"] == "mismatch":
            return (f"向量空间不一致，无法回答：索引由 [{r['index_identity']}] 生成，"
                    f"当前嵌入配置是 [{r['config_identity']}]。"
                    "请先运行 uv run kb-rag index 全量重建，或恢复 .env 的 EMBED_* 配置。")
        if r["mode"] == "locked":
            notice = f"⚠ 另一进程（{r['holder']}）正在重建索引，本次按现有索引回答。\n\n"
        elif r["mode"] == "delta":
            notice = (f"⚠ 索引过期（变更 {st['changed']} / 删除 {st['deleted']} 个文件），"
                      f"已自动增量入库（+{r['added']} 块 / {r['seconds']:.1f}s）后回答。\n\n")
    elif st["mode"] != "clean":
        notice = "⚠ 索引可能过期（受限角色不触发自动重建，请联系管理员跑 kb-rag add）。\n\n"

    from .indexer import IndexNotQueryable, ensure_queryable

    try:
        ensure_queryable()  # 空间门禁：配置变了但文件没变时 staleness 探不到，这里兜住
    except IndexNotQueryable as e:  # 拒绝原因作为回答返回：CLI/MCP 两侧都不裸崩
        return str(e)

    is_status = _is_status_question(question)
    alias_srcs = _alias_sources(question, is_status)
    if restricted:  # 读范围过滤：别名命中的状态页同样不许越权
        alias_srcs = {rel: v for rel, v in alias_srcs.items() if scopes.in_scope(rel, allowed)}

    sources: list[tuple[str, str]] = []  # (来源标签, 内容)
    added_rels: set[str] = set()
    for rel, (label, text) in alias_srcs.items():  # 别名命中优先，引用编号靠前
        sources.append((label, text))
        added_rels.add(rel)
    if is_status:
        for w in wiki_query(question, k=3):
            if w["status_file"]:
                rel = f"{w['path']}/{w['status_file']}" if w["path"] != "（根目录）" else w["status_file"]
                if rel in added_rels or (restricted and not scopes.in_scope(rel, allowed)):
                    continue  # 别名已带入同一份状态页不重复；越权节点不进上下文
                sources.append((f"{rel}（{w['title']}·状态页整文件）",
                                f"节点摘要：{w['summary']}\n{w['file_text']}"))
                added_rels.add(rel)
    fetch = k * 3 if restricted else k
    if rewrite.enabled():
        # 查询改写融合检索（Multi-Query + HyDE；SHA-1 缓存保证同问题同改写，
        # 检索逐位可复现）+ 精排——多路候选先融合再由交叉编码器择优
        from .rerank import rerank

        hits = [h for h in rewrite.fused_query(question, k=fetch)
                if scopes.in_scope(h["path"], allowed)]
        hits = rerank(question, hits)[:k]
    else:
        hits = [h for h in hybrid_query(question, k=fetch)
                if scopes.in_scope(h["path"], allowed)][:k]
    if not hits and not sources:
        return notice + "知识库中没有检索到相关内容（或内容不在调用者的读范围内）。"
    for h in hits:
        sources.append((f"{h['path']} #{h['heading']}", h["text"][:800]))

    context = "\n\n".join(f"[{i}] 来源：{label}\n{text}" for i, (label, text) in enumerate(sources, 1))
    client = OpenAI(api_key=LLM_API_KEY, base_url=LLM_BASE_URL)
    resp = client.chat.completions.create(
        model=LLM_MODEL,
        messages=[{"role": "user", "content": PROMPT.format(context=context, question=question)}],
        temperature=0.2,
    )
    answer = resp.choices[0].message.content
    footer = "\n".join(f"[{i}] {label}" for i, (label, _) in enumerate(sources, 1))
    return f"{notice}{answer}\n\n来源：\n{footer}"
