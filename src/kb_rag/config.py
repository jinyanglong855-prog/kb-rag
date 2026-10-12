"""集中配置：全部来自 .env，不写死本机路径。"""

import os
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()
# MCP/服务形态从任意 cwd 启动时也能找到项目 .env（load_dotenv 不覆盖已有环境变量）
load_dotenv(Path(__file__).resolve().parents[2] / ".env")

KB_PATH = Path(os.environ.get("KB_PATH", ""))
# 仓库根：INDEX_DIR/scopes.json 等运行配置的锚点。INDEX_DIR 不跟随 cwd——
# 测试/服务进程换 KB_PATH 或 cwd 时，索引产物不允许落到"当前目录的 index/"。
REPO_ROOT = Path(__file__).resolve().parents[2]
_raw_index_dir = os.environ.get("INDEX_DIR", "index")
INDEX_DIR = Path(_raw_index_dir) if Path(_raw_index_dir).is_absolute() else REPO_ROOT / _raw_index_dir
EMBED_MODEL = os.environ.get("EMBED_MODEL", "BAAI/bge-small-zh-v1.5")
CHUNK_SIZE = int(os.environ.get("CHUNK_SIZE", "1200"))
RERANK_MODEL = os.environ.get("RERANK_MODEL", "BAAI/bge-reranker-base")
LLM_API_KEY = os.environ.get("LLM_API_KEY") or os.environ.get("DEEPSEEK_API_KEY", "")
LLM_BASE_URL = os.environ.get("LLM_BASE_URL", "https://open.bigmodel.cn/api/paas/v4")
LLM_MODEL = os.environ.get("LLM_MODEL", "glm-4.5-flash")
HYBRID_DEFAULT = True  # ask 层默认走混合检索


def validate_kb_path() -> Path:
    if not KB_PATH.is_dir():
        raise SystemExit(f"KB_PATH 无效: {KB_PATH}（请检查 .env）")
    return KB_PATH
