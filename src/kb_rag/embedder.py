"""嵌入层：可插拔后端，同一接口切本地/API。

- fastembed：本地 ONNX，零出域（默认）。threads=8（2026-10-17 实测 Qwen3-0.6B
  三进程查询向量逐位一致；旧 bge 时代多线程跨进程有浮点差才钉单线程，EMBED_THREADS=1
  可回退）；batch_size=4 钉死——Qwen3 是解码器架构，fastembed 默认批 256 在
  真实长块上单次推理 RSS ~7.9GB 被 macOS jetsam 杀，批 4 峰值 3.4GB 且反而最快。
  全库重建吞吐实测 ~1.8 块/s ≈ 1230 字符/s → 9.3 万块约 17-20 小时（须走
  indexer 的分段续传）；日常增量 500 块 ≈ 6 分钟。
- openai：任意 OpenAI 兼容 /v1/embeddings（百炼 compatible-mode、硅基流动、
  llama-server、Ollama、vLLM），企业落地按隐私边界选服务商或自托管。

文档侧与查询侧不对称是设计：文档永远裸文本（含路径标题前缀，见 indexer），
查询可加指令前缀（EMBED_QUERY_INSTRUCT）。向量空间由 embed_identity() 唯一
标识，索引 meta 记录它，换后端/模型/维度/指令任一项都是另一个空间。
"""

import os
from functools import lru_cache

import numpy as np

from .config import (EMBED_API_BASE, EMBED_API_BATCH, EMBED_API_KEY, EMBED_BACKEND,
                     EMBED_DIMS, EMBED_MODEL, EMBED_QUERY_INSTRUCT)

# 文档侧输入格式的版本标签：改 indexer.embed_input 的拼法必须同步改它（空间随之变化）。
# 字面值沿用旧 meta.json 的 doc_prefix 取值，现役索引无需重建即可被新引擎接管。
DOC_PREFIX_VERSION = "path+heading"


class FastEmbedBackend:
    kind = "fastembed"
    # ONNX arena 随进程内累计处理量增长（8000 块段实测峰值 6GB），多段任务须换进程
    isolate_segments = True

    def __init__(self, model: str):
        self.model = model
        self.dims = None
        self._engine = None

    def _get(self):
        if self._engine is None:
            from fastembed import TextEmbedding

            threads = int(os.environ.get("EMBED_THREADS", "8"))
            self._engine = TextEmbedding(model_name=self.model, threads=threads)
        return self._engine

    @property
    def dim(self) -> int:
        from fastembed import TextEmbedding

        for d in TextEmbedding.list_supported_models():
            if d["model"] == self.model:
                return int(d["dim"])
        return len(self.embed_documents(["dim probe"])[0])

    def embed_documents(self, texts: list[str]) -> list[np.ndarray]:
        return list(self._get().embed(texts, batch_size=4))


class OpenAICompatBackend:
    kind = "openai"
    isolate_segments = False  # 推理在远端，本进程无内存累积

    def __init__(self, base_url: str, api_key: str, model: str, batch: int, dims: int | None):
        if not base_url:
            raise ValueError("EMBED_BACKEND=openai 需要 EMBED_API_BASE（如 https://dashscope.aliyuncs.com/compatible-mode/v1）")
        self.base_url, self.api_key, self.model, self.batch, self.dims = base_url, api_key, model, batch, dims
        self._client = None
        self._dim: int | None = None

    def _get(self):
        if self._client is None:
            from openai import OpenAI

            # 本地 llama-server/Ollama 不校验 key，但客户端要求非空
            self._client = OpenAI(base_url=self.base_url, api_key=self.api_key or "none",
                                  timeout=600, max_retries=2)
        return self._client

    @property
    def dim(self) -> int:
        if self._dim is None:  # 探测结果缓存在实例上：未设 EMBED_DIMS 时不能每次查询多打一次 API
            self._dim = self.dims or len(self.embed_documents(["dim probe"])[0])
        return self._dim

    def embed_documents(self, texts: list[str]) -> list[np.ndarray]:
        out: list[np.ndarray] = []
        for i in range(0, len(texts), self.batch):
            kw: dict = {"model": self.model, "input": texts[i:i + self.batch]}
            if self.dims:
                kw["dimensions"] = self.dims
            resp = self._get().embeddings.create(**kw)
            got = sorted(resp.data, key=lambda d: d.index)
            if len(got) != len(kw["input"]):
                # 异构服务端静默丢条目会让后续所有行错位——整批零向量/错配，必须响
                raise RuntimeError(f"嵌入服务返回条数不符：请求 {len(kw['input'])} 条，返回 {len(got)} 条"
                                   f"（{self.base_url} / {self.model}）")
            out.extend(np.asarray(d.embedding, dtype=np.float32) for d in got)
        return out


@lru_cache(maxsize=1)
def get_backend():
    if EMBED_BACKEND == "fastembed":
        if EMBED_DIMS:
            raise ValueError("EMBED_DIMS 仅 openai 后端支持（API 侧 MRL 截断）；fastembed 本地模型维度固定，"
                             "设了也会被静默忽略，故显式拒绝")
        return FastEmbedBackend(EMBED_MODEL)
    if EMBED_BACKEND == "openai":
        return OpenAICompatBackend(EMBED_API_BASE, EMBED_API_KEY, EMBED_MODEL, EMBED_API_BATCH, EMBED_DIMS)
    raise ValueError(f"未知 EMBED_BACKEND={EMBED_BACKEND!r}（可选 fastembed / openai）")


def doc_identity() -> str:
    """文档侧向量空间指纹（缓存键）：文档向量与查询侧指令无关，不含 inst/raw 位——
    切换 EMBED_QUERY_INSTRUCT 调试查询侧时，全部文档向量缓存照常复用。"""
    b = get_backend()
    dims = f"@{b.dims}" if b.dims else ""
    return f"{b.kind}:{b.model}{dims}:{DOC_PREFIX_VERSION}"


def embed_identity() -> str:
    """完整空间指纹（门禁键）= doc_identity + 查询侧指令位。
    索引 meta 存它，查询/增量前比对——不一致即查询两侧不同空间，禁止检索。"""
    q = "inst" if EMBED_QUERY_INSTRUCT else "raw"
    return f"{doc_identity()}:{q}"


def embed(texts: list[str]) -> list[np.ndarray]:
    """文档侧嵌入（裸文本）。"""
    return get_backend().embed_documents(texts)


def embed_queries(texts: list[str]) -> list[np.ndarray]:
    """查询侧嵌入：按配置加指令前缀；用 replace 不用 format，问题里的花括号不触发占位解析。"""
    if EMBED_QUERY_INSTRUCT:
        texts = [EMBED_QUERY_INSTRUCT.replace("{q}", t) for t in texts]
    return get_backend().embed_documents(texts)


def embed_dim() -> int:
    return get_backend().dim
