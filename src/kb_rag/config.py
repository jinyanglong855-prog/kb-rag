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
# 嵌入模型（2026-10-21 换代完成）：Qwen3-Embedding-0.6B-Q 本地 ONNX。
# 考卷实测 v@5：bge 40% → 0.6B 78%（8B+指令 87% 为天花板，API 路线经 EMBED_BACKEND=openai 可随时切换）
EMBED_MODEL = os.environ.get("EMBED_MODEL", "Qwen/Qwen3-Embedding-0.6B-Q")
# 嵌入后端：fastembed=本地 ONNX（零出域，默认）；openai=任意 OpenAI 兼容 HTTP 接口
# （百炼/硅基流动/llama-server/Ollama/vLLM 同一套代码）。企业落地按隐私边界选，
# 换后端=换向量空间，索引必须整体重建（indexer 的一致性门禁会拦住静默混用）。
EMBED_BACKEND = os.environ.get("EMBED_BACKEND", "fastembed")
EMBED_API_BASE = os.environ.get("EMBED_API_BASE", "")
EMBED_API_KEY = os.environ.get("EMBED_API_KEY", "")
EMBED_API_BATCH = int(os.environ.get("EMBED_API_BATCH", "10"))  # 百炼 v4 单请求上限 10 条
EMBED_DIMS = int(os.environ.get("EMBED_DIMS", "0")) or None  # API 侧截断维度（MRL），0=模型默认
# 查询侧指令前缀（文档侧永远裸文本）。考卷实测：Qwen3 4B/8B 加指令 +9 分，0.6B 反降，
# 故默认空；用 {q} 占位，如 "Instruct: Given a web search query, retrieve relevant passages that answer the query\nQuery:{q}"
EMBED_QUERY_INSTRUCT = os.environ.get("EMBED_QUERY_INSTRUCT", "")
# 全量/增量嵌入的断点粒度（块）：每段落盘一次，进程被杀最多重做一段
EMBED_SEGMENT = int(os.environ.get("EMBED_SEGMENT", "4000"))
CHUNK_SIZE = int(os.environ.get("CHUNK_SIZE", "1200"))
# 父子分块（small-to-big）：检索与排序仍按子块（现切块），ask 装配层把命中块
# 膨胀为父块（同文件同标题的连续块串=章节/整文件）再进上下文——命中精确、
# 回答带上下文。KB_PARENT_RETRIEVE=0 一键回退；预算=单父块进上下文的目标上限
# （软上限：命中块自身超预算时按原块完整返回，不丢命中）。
PARENT_RETRIEVE = os.environ.get("KB_PARENT_RETRIEVE", "1") != "0"
PARENT_MAX_CHARS = int(os.environ.get("KB_PARENT_CHARS", "3000"))
RERANK_MODEL = os.environ.get("RERANK_MODEL", "BAAI/bge-reranker-base")
LLM_API_KEY = os.environ.get("LLM_API_KEY") or os.environ.get("DEEPSEEK_API_KEY", "")
LLM_BASE_URL = os.environ.get("LLM_BASE_URL", "https://open.bigmodel.cn/api/paas/v4")
LLM_MODEL = os.environ.get("LLM_MODEL", "glm-4.5-flash")
HYBRID_DEFAULT = True  # ask 层默认走混合检索


def validate_kb_path() -> Path:
    if not KB_PATH.is_dir():
        raise SystemExit(f"KB_PATH 无效: {KB_PATH}（请检查 .env）")
    return KB_PATH
