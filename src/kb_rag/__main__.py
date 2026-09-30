"""CLI: uv run kb-rag index [--bm25] | uv run kb-rag query "问题" [-k 5] [--mode hybrid]"""

import argparse

from .hybrid import build_bm25_index, hybrid_query
from .search import build_index, query


def main() -> None:
    parser = argparse.ArgumentParser(prog="kb-rag")
    sub = parser.add_subparsers(dest="cmd", required=True)
    idx = sub.add_parser("index", help="对知识库建索引")
    idx.add_argument("--bm25", action="store_true", help="仅重建 BM25 索引（复用已有向量块缓存）")
    q = sub.add_parser("query", help="语义检索知识库")
    q.add_argument("question")
    q.add_argument("-k", type=int, default=5)
    q.add_argument("--mode", choices=["vector", "hybrid"], default="vector")
    args = parser.parse_args()

    if args.cmd == "index":
        if args.bm25:
            build_bm25_index()
        else:
            build_index()
            build_bm25_index()
    else:
        hits = hybrid_query(args.question, args.k) if args.mode == "hybrid" else query(args.question, args.k)
        label = "hybrid(BM25+向量 RRF)" if args.mode == "hybrid" else "vector"
        print(f"== 检索模式: {label} ==")
        for i, r in enumerate(hits, 1):
            print(f"\n[{i}] {r['score']}  {r['path']}  # {r['heading']}")
            print(r["text"][:300].replace("\n", " "))
