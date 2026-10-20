"""嵌入层：fastembed 本地 ONNX 推理，零 API 成本。

threads=8（2026-10-17 实测更新）：Qwen3-Embedding-0.6B-Q 在 8 线程下三个
独立进程的查询向量逐位一致（旧 bge-small-zh 时代多线程跨进程有浮点差、
93k 近重复语料放大成排名翻转，才钉死单线程；换模型后以新实测为准）。
单线程在本模型要 3.7s/条、全库重建 95 小时，不可行；8 线程 0.1s/条、
重建 2.7 小时。若未来再现实测不稳，EMBED_THREADS=1 可回退。

⚠️ 重建时长勘误（2026-10-21 干净复测 3072 块）：上段"0.1s/条、2.7 小时"
是短样本外推的失实值。实测稳定 ~1.8 块/s ≈ 1230 字符/s；全库 7810 万字符
（+路径标题前缀 ≈ 8600 万）折算 **约 17-20 小时**，须用 work/embed_resume.py
断点续传跑。日常增量不受影响（500 块 ≈ 6 分钟）。旧 bge-small-zh 的
"约 12 分钟"量级不可比（模型小 10 倍 + 512 token 截断）。
"""

import os
from functools import lru_cache

from fastembed import TextEmbedding

from .config import EMBED_MODEL


@lru_cache(maxsize=1)
def get_model() -> TextEmbedding:
    threads = int(os.environ.get("EMBED_THREADS", "8"))
    return TextEmbedding(model_name=EMBED_MODEL, threads=threads)


def embed(texts: list[str]) -> list[list[float]]:
    # batch_size=4（2026-10-17 实测，真实语料块最长 1195 字）：Qwen3 是解码器
    # 架构，fastembed 默认批 256 在长文本下 ONNX 单次推理就把 RSS 顶到 ~7.9GB
    # 被 macOS jetsam 杀（重建三次 EXIT=137 的真凶）；批 16 更糟（9.6GB）。
    # 批 4 峰值 3.4GB 且反而最快（3 条/s vs 默认 1 条/s），内存速度双优。
    return list(get_model().embed(texts, batch_size=4))
