"""CLI: uv run kb-rag index | uv run kb-rag query "问题" [-k 5]"""

import argparse

from .search import build_index, query


def main() -> None:
    parser = argparse.ArgumentParser(prog="kb-rag")
    sub = parser.add_subparsers(dest="cmd", required=True)
    sub.add_parser("index", help="对知识库建向量索引")
    q = sub.add_parser("query", help="语义检索知识库")
    q.add_argument("question")
    q.add_argument("-k", type=int, default=5)
    args = parser.parse_args()

    if args.cmd == "index":
        build_index()
    else:
        for i, r in enumerate(query(args.question, args.k), 1):
            print(f"\n[{i}] {r['score']}  {r['path']}  # {r['heading']}")
            print(r["text"][:300].replace("\n", " "))
