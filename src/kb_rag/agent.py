"""阶段 4 Agentic RAG：五层检索做成 tools，LLM 自主调度（参考调度循环）。

分工边界：生产调度器是公司智能体项目选定的 DeepSeek Harness（一切皆插件）。
本模块只做两件事——
1. 把各检索层封装成 schema 规范的 tools（公司智能体的知识插件直接复用这份定义）；
2. 用 DeepSeek 原生 function calling 跑通参考调度循环，证明"检索策略自适应"可行：
   由模型按问题类型选层（状态类→结构层、多跳关系→图、精确词→hybrid、模糊语义→向量），
   而不是人工 if-else 路由。

不引入 agent 框架依赖：调度循环是薄胶水，真正的框架整合发生在 DSH 侧。
"""

import json
from pathlib import Path

from openai import OpenAI

from .config import INDEX_DIR, KB_PATH, LLM_API_KEY, LLM_BASE_URL, LLM_MODEL, validate_kb_path
from .hybrid import hybrid_query
from .search import query as vector_query
from .wikitree import wiki_query

MAX_STEPS = 6

SYSTEM_PROMPT = """你是本地知识库的智能检索助手，自主决定用哪些检索工具回答问题。

调度规则：
- 问某项目"现状/进度/卡点/下一步"→ 先 status_lookup
- 问跨项目的关系、对比、多跳关联（"A 和 B 有什么关系"、"哪些项目都涉及 X"）→ graph_query
- 问题里有明确的项目名/人名/专有名词 → hybrid_search
- 语义模糊、说不太清关键词 → vector_search
- 工具结果知道该读哪个文件但内容被截断 → read_file 读整文件
- 可以组合多个工具；结果足够就停止调用，直接回答。

回答要求：关键论断标注来源（文件路径或工具名）；资料不足就明说，不编造。中文回答。"""

TOOLS_SPEC = [
    {"type": "function", "function": {
        "name": "vector_search",
        "description": "向量语义检索：问题语义模糊、没有精确关键词时用",
        "parameters": {"type": "object", "properties": {
            "question": {"type": "string"}, "k": {"type": "integer", "default": 5}},
            "required": ["question"]}}},
    {"type": "function", "function": {
        "name": "hybrid_search",
        "description": "BM25+向量混合检索：问题含项目名/人名/精确专有名词时用",
        "parameters": {"type": "object", "properties": {
            "question": {"type": "string"}, "k": {"type": "integer", "default": 8}},
            "required": ["question"]}}},
    {"type": "function", "function": {
        "name": "status_lookup",
        "description": "结构层目录路由：问某项目的现状/进度/卡点/下一步时用，返回状态页整文件",
        "parameters": {"type": "object", "properties": {"question": {"type": "string"}},
                       "required": ["question"]}}},
    {"type": "function", "function": {
        "name": "graph_query",
        "description": "知识图谱多跳检索：问跨项目关系/对比/多跳关联时用",
        "parameters": {"type": "object", "properties": {
            "question": {"type": "string"},
            "mode": {"type": "string", "enum": ["mix", "local", "global", "hybrid"], "default": "mix"}},
            "required": ["question"]}}},
    {"type": "function", "function": {
        "name": "read_file",
        "description": "按库内相对路径读整个文件（前面工具结果被截断、需要全文时用）",
        "parameters": {"type": "object", "properties": {"path": {"type": "string"}},
                       "required": ["path"]}}},
]


def _fmt_hits(hits: list[dict]) -> str:
    return "\n\n".join(
        f"[{h['path']} #{h['heading']}]（score={h['score']}）\n{h['text'][:600]}"
        for h in hits
    ) or "无结果"


def _tool_vector_search(question: str, k: int = 5) -> str:
    return _fmt_hits(vector_query(question, k))


def _tool_hybrid_search(question: str, k: int = 8) -> str:
    return _fmt_hits(hybrid_query(question, k))


def _tool_status_lookup(question: str) -> str:
    out = []
    for w in wiki_query(question, k=3):
        node = f"节点：{w['title']}（{w['path']}）\n摘要：{w['summary']}"
        if w["status_file"]:
            node += f"\n状态页整文件 {w['status_file']}：\n{w['file_text'][:3000]}"
        out.append(node)
    return "\n\n---\n\n".join(out) or "无命中节点"


def _tool_graph_query(question: str, mode: str = "mix") -> str:
    if not (Path(INDEX_DIR) / "lightrag" / "graph_chunk_entity_relation.graphml").exists():
        return "知识图谱未建立（先跑 kb-rag graph-build），请改用其他检索工具"
    from .graph import query_graph

    return query_graph(question, mode)


def _tool_read_file(path: str) -> str:
    p = (KB_PATH / path).resolve()
    if not p.is_file() or not p.is_relative_to(KB_PATH.resolve()):
        return f"文件不存在或越界: {path}"
    return p.read_text(encoding="utf-8", errors="ignore")[:8000]


_DISPATCH = {
    "vector_search": _tool_vector_search,
    "hybrid_search": _tool_hybrid_search,
    "status_lookup": _tool_status_lookup,
    "graph_query": _tool_graph_query,
    "read_file": _tool_read_file,
}


def agent(question: str, max_steps: int = MAX_STEPS) -> str:
    """LLM 自主调度五层检索工具直到给出最终回答，返回 (回答, 调用轨迹)。"""
    validate_kb_path()
    if not LLM_API_KEY:
        raise SystemExit("请在 .env 里填 LLM_API_KEY")
    client = OpenAI(api_key=LLM_API_KEY, base_url=LLM_BASE_URL)
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": question},
    ]
    trace: list[str] = []

    for _ in range(max_steps):
        resp = client.chat.completions.create(
            model=LLM_MODEL, messages=messages, tools=TOOLS_SPEC, temperature=0.2,
        )
        msg = resp.choices[0].message
        messages.append(msg)
        if not msg.tool_calls:
            return msg.content or "（无回答）", trace
        for tc in msg.tool_calls:
            name = tc.function.name
            args = json.loads(tc.function.arguments or "{}")
            trace.append(f"{name}({args})")
            result = _DISPATCH[name](**args)
            messages.append({"role": "tool", "tool_call_id": tc.id, "content": result[:6000]})
    return "达到最大工具调用轮数仍未收敛，以上为最后尝试。", trace
