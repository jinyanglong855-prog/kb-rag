"""Chroma 持久向量库（阶段 1c）。

设计说明（面试故事点）：4793 块时 numpy 暴力检索毫秒级即可用；引入 Chroma
不是为了性能，而是为了边界——当语料增长、需要元数据过滤和增量更新时，
向量库成为必要组件。切换通过 --store 选择，numpy 路径保留作对照。
"""

import json
from pathlib import Path

import chromadb
import numpy as np

from .config import INDEX_DIR
from .embedder import embed


def _collection():
    client = chromadb.PersistentClient(path=str(Path(INDEX_DIR) / "chroma"))
    return client.get_or_create_collection("kb_chunks", metadata={"hnsw:space": "cosine"})


def build_chroma() -> int:
    chunks = json.loads((Path(INDEX_DIR) / "chunks.json").read_text(encoding="utf-8"))
    vectors = np.load(Path(INDEX_DIR) / "vectors.npy")
    col = _collection()
    if col.count():
        col.delete(ids=[f"kb{i}" for i in range(col.count())])
    # Chroma 单批上限约 5461（语料增长后分批写入）
    batch = 5000
    for start in range(0, len(chunks), batch):
        end = min(start + batch, len(chunks))
        col.add(
            ids=[f"kb{i}" for i in range(start, end)],
            embeddings=vectors[start:end].tolist(),
            documents=[c["text"] for c in chunks[start:end]],
            metadatas=[{"path": c["path"], "heading": c["heading"]} for c in chunks[start:end]],
        )
    print(f"Chroma 索引完成: {col.count()} 块")
    return col.count()


def chroma_query(question: str, k: int = 5) -> list[dict]:
    col = _collection()
    q = embed([question])[0]
    res = col.query(query_embeddings=[q], n_results=k)
    out = []
    for i in range(len(res["ids"][0])):
        meta = res["metadatas"][0][i]
        out.append({
            "path": meta["path"],
            "heading": meta["heading"],
            "text": res["documents"][0][i],
            "score": round(1.0 - float(res["distances"][0][i]), 4),  # 余弦距离转相似度
        })
    return out
