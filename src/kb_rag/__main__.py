"""CLI: uv run kb-rag index | add | query "问题" | ask "问题" | wiki ["问题"]"""

import argparse

from .hybrid import build_bm25_index, hybrid_query
from .rerank import rerank
from .ask import ask
from .search import build_index, query
from .store import build_chroma, chroma_query
from .wikitree import build_wiki, show_tree, wiki_query


def main() -> None:
    parser = argparse.ArgumentParser(prog="kb-rag")
    sub = parser.add_subparsers(dest="cmd", required=True)
    idx = sub.add_parser("index", help="全量重建索引（结构大改/首次/规则变更后）")
    idx.add_argument("--bm25", action="store_true", help="仅重建 BM25 索引（复用已有向量块缓存）")
    idx.add_argument("--chroma", action="store_true", help="从向量缓存导入 Chroma 持久库")
    idx.add_argument("--wiki", action="store_true", help="仅重建 Wiki 摘要树（内容未变走缓存）")
    sub.add_parser("add", help="增量入库：文件放进总库后跑此命令，秒级可检索（按文件指纹只重嵌变更）")
    ing = sub.add_parser("ingest", help="PDF/Word/PPT/Excel/HTML/EPUB/图片 → md 落入总库并增量入库（docling 后端）")
    ing.add_argument("file", help="源文件路径")
    ing.add_argument("--to", dest="target", default=None, help="库内目标目录（默认 Shared/参考资料/入库文件）")
    ing.add_argument("--no-index", action="store_true", help="只转换不建索引")
    q = sub.add_parser("query", help="语义检索知识库")
    q.add_argument("question")
    q.add_argument("-k", type=int, default=5)
    q.add_argument("--mode", choices=["vector", "hybrid", "chroma"], default="vector")
    q.add_argument("--rerank", action="store_true", help="交叉编码器精排（首次运行需下载模型）")
    a = sub.add_parser("ask", help="检索+生成：带引用的回答（需 LLM_API_KEY）")
    a.add_argument("question")
    a.add_argument("-k", type=int, default=5)
    w = sub.add_parser("wiki", help="结构层：无参数=查看摘要树；带问题=看目录路由命中")
    w.add_argument("question", nargs="?", default=None)
    w.add_argument("-k", type=int, default=3)
    gb = sub.add_parser("graph-build", help="阶段3：LightRAG 增量建图（--full 全量重建；--scope 只建某子树）")
    gb.add_argument("--full", action="store_true")
    gb.add_argument("--scope", default=None, help="相对路径前缀，如 kb-rag/")
    gq = sub.add_parser("graph", help="阶段3：图检索问答（跨项目多跳）")
    gq.add_argument("question")
    gq.add_argument("--mode", choices=["mix", "local", "global", "hybrid", "naive"], default="mix")
    gv = sub.add_parser("graph-viz", help="阶段3：知识图谱导出 pyvis 交互 HTML（默认度数前300实体）")
    gv.add_argument("--out", default=None)
    gv.add_argument("--top", type=int, default=300)
    ag = sub.add_parser("agent", help="阶段4：Agentic 检索——LLM 自主调度五层工具回答")
    ag.add_argument("question")
    ag.add_argument("--max-steps", type=int, default=6)
    ev = sub.add_parser("evals", help="阶段5：跑评测集，五层检索匹配率对比（--skip-graph 跳过图层）")
    ev.add_argument("--skip-graph", action="store_true")
    args = parser.parse_args()

    if args.cmd == "ask":
        print(ask(args.question, args.k))
        return
    if args.cmd == "wiki":
        if args.question:
            for i, r in enumerate(wiki_query(args.question, args.k), 1):
                print(f"\n[{i}] {r['score']}  {r['path']} -> {r['status_file']}")
                print(f"    摘要: {r['summary'][:100]}")
                print(f"    整文件带入 {len(r['file_text'])} 字符")
        else:
            show_tree()
        return
    if args.cmd == "index":
        if args.bm25:
            build_bm25_index()
        elif args.chroma:
            build_chroma()
        elif args.wiki:
            build_wiki()
        else:
            build_index()
            build_bm25_index()
            build_chroma()
            build_wiki()
        return
    if args.cmd == "add":
        from .search import incremental_index

        r = incremental_index()
        if r["mode"] == "clean":
            print(f"语料无变化，共 {r['total']} 块 / {r['seconds']:.1f}s")
            return
        if r["mode"] == "full":
            print("首次增量：已自动执行全量索引（生成文件指纹缓存，此后 add 为秒级）")
            return
        build_bm25_index()
        build_chroma()
        print(f"增量入库完成: +{r['added']} 块 / -{r['removed']} 块 / 总 {r['total']} 块 / {r['seconds']:.1f}s")
        return
    if args.cmd == "graph-build":
        from .graph import build_graph

        r = build_graph(full=args.full, scope=args.scope)
        print(f"建图完成: +{r['added']} / -{r['removed']} 文件 / 图谱语料 {r['total']} 文件 / {r['seconds']:.1f}s")
        return
    if args.cmd == "graph":
        from .graph import query_graph

        print(query_graph(args.question, args.mode))
        return
    if args.cmd == "graph-viz":
        from .graph import export_viz

        print(f"图谱可视化: {export_viz(args.out, args.top)}")
        return
    if args.cmd == "agent":
        from .agent import agent

        answer, trace = agent(args.question, args.max_steps)
        if trace:
            print("== 工具轨迹 ==")
            for i, t in enumerate(trace, 1):
                print(f"[{i}] {t}")
            print()
        print(answer)
        return
    if args.cmd == "evals":
        from .evals import run_evals

        run_evals(skip_graph=args.skip_graph)
        return
    if args.cmd == "ingest":
        from .ingest import ingest_file
        from .search import incremental_index

        out = ingest_file(args.file, args.target)
        print(f"已转换: {out}")
        if not args.no_index:
            r = incremental_index()
            if r["mode"] in ("delta", "full"):
                build_bm25_index()
                build_chroma()
                print(f"增量入库完成: +{r['added']} 块 / 总 {r['total']} 块 / {r['seconds']:.1f}s")
            else:
                print("增量入库：语料无变化")
        return
    else:
        if args.mode == "hybrid":
            hits = hybrid_query(args.question, args.k)
        elif args.mode == "chroma":
            hits = chroma_query(args.question, args.k)
        else:
            hits = query(args.question, args.k)
        if args.rerank:
            hits = rerank(args.question, hits)
        label = {"vector": "vector(numpy)", "hybrid": "hybrid(BM25+向量 RRF)", "chroma": "chroma"}[args.mode]
        if args.rerank:
            label += " + rerank"
        print(f"== 检索模式: {label} ==")
        for i, r in enumerate(hits, 1):
            print(f"\n[{i}] {r['score']}  {r['path']}  # {r['heading']}")
            print(r["text"][:300].replace("\n", " "))
