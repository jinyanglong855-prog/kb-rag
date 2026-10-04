"""集中配置：全部来自 .env，不写死本机路径。"""

import os
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

KB_PATH = Path(os.environ.get("KB_PATH", ""))
INDEX_DIR = Path(os.environ.get("INDEX_DIR", "index"))
EMBED_MODEL = os.environ.get("EMBED_MODEL", "BAAI/bge-small-zh-v1.5")
CHUNK_SIZE = int(os.environ.get("CHUNK_SIZE", "1200"))
RERANK_MODEL = os.environ.get("RERANK_MODEL", "BAAI/bge-reranker-base")


def validate_kb_path() -> Path:
    if not KB_PATH.is_dir():
        raise SystemExit(f"KB_PATH 无效: {KB_PATH}（请检查 .env）")
    return KB_PATH
