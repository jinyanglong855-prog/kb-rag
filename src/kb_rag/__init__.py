"""kb-rag: 本地知识库检索，从朴素 RAG 到 Agentic GraphRAG 的六阶段演进。"""

import os
from importlib.metadata import PackageNotFoundError, version as _pkg_version

# BLAS/ONNX 钉单线程：多线程浮点归算顺序随进程内存状态变化，同一查询跨进程
# 向量有微小差异；93k 块近重复语料会把差异放大成排名翻转（评测不可复现）。
# 必须在 numpy 首次导入前设置，故放包 __init__——任何子模块都晚于它执行。
for _var in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
             "NUMEXPR_NUM_THREADS"):
    os.environ.setdefault(_var, "1")

# 嵌入模型缓存钉到仓库内：fastembed 缺省放系统临时目录，MCP 宿主（mcp SDK
# 默认环境白名单无 TMPDIR）冷启动会定位到 /tmp 触发模型重下载，内网镜像
# 401 直接让检索工具崩。钉住后一次下载永久可用（.cache/ 已 gitignore）。
_repo_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
os.environ.setdefault("FASTEMBED_CACHE_PATH", os.path.join(_repo_root, ".cache", "fastembed"))

try:
    __version__ = _pkg_version("kb-rag")
except PackageNotFoundError:  # 直接以源码树运行（未安装）时的回退
    __version__ = "0.0.0+dev"
