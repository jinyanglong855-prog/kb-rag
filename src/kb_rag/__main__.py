"""CLI: uv run kb-rag index | query "问题" | ask "问题" | wiki ["问题"]"""

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
    idx = sub.add_parser("index", help="对知识库建索引")
    idx.add_argument("--bm25", action="store_true", help="仅重建 BM25 索引（复用已有向量块缓存）")
    idx.add_argument("--chroma", action="store_true", help="从向量缓存导入 Chroma 持久库")
    idx.add_argument("--wiki", action="store_true", help="仅重建 Wiki 摘要树（内容未变走缓存）")
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
