"""kb-rag: 本地知识库检索，从朴素 RAG 到 Agentic GraphRAG 的六阶段演进。"""

import os
from importlib.metadata import PackageNotFoundError, version as _pkg_version

# BLAS/ONNX 钉单线程：多线程浮点归约顺序随进程内存状态变化，同一查询跨进程
# 向量有微小差异；93k 块近重复语料会把差异放大成排名翻转（评测不可复现）。
# 必须在 numpy 首次导入前设置，故放包 __init__——任何子模块都晚于它执行。
for _var in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
             "NUMEXPR_NUM_THREADS"):
    os.environ.setdefault(_var, "1")

try:
    __version__ = _pkg_version("kb-rag")
except PackageNotFoundError:  # 直接以源码树运行（未安装）时的回退
    __version__ = "0.0.0+dev"
