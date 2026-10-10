"""建索引与检索：numpy 余弦即可覆盖万级以下语料，向量库留到阶段 1。

入库两种方式：
- `index` 全量重建（结构大改/首次/规则变更后）
- `add` 增量入库（日常）：按文件 SHA-1 只重嵌新增/变更文件，秒级可检索
"""

import hashlib
import json
import time
from pathlib import Path

import numpy as np

from .authority import apply as authority_apply
from .config import INDEX_DIR, KB_PATH, validate_kb_path
from .embedder import embed
from .loader import chunk_file, load_text_files

META_JSON = "meta.json"


def _file_hash(p: Path) -> str:
    return hashlib.sha1(p.read_bytes()).hexdigest()


def _scan_hashes(root: Path) -> dict[str, str]:
    return {rel: _file_hash(f) for f, rel in load_text_files(root)}


def _chunk_files(root: Path, rels: set[str]) -> tuple[list[dict], list[str]]:
    """切指定文件集合，返回 (chunk dicts, 对应文本列表)。"""
    chunks: list[dict] = []
    for f, rel in load_text_files(root):
        if rel not in rels:
            continue
        text = f.read_text(encoding="utf-8", errors="ignore")
        for c in chunk_file(text, rel, is_markdown=f.suffix.lower() == ".md"):
            chunks.append({"path": c.path, "heading": c.heading, "text": c.text})
    return chunks, [c["text"] for c in chunks]


def _embed_normalized(texts: list[str]) -> np.ndarray:
    vectors = np.array(embed(texts), dtype=np.float32)
    vectors /= np.maximum(np.linalg.norm(vectors, axis=1, keepdims=True), 1e-10)
    return vectors


def _save(chunks: list[dict], vectors: np.ndarray, hashes: dict[str, str]) -> None:
    INDEX_DIR.mkdir(exist_ok=True)
    np.save(INDEX_DIR / "vectors.npy", vectors)
    (INDEX_DIR / "chunks.json").write_text(
        json.dumps(chunks, ensure_ascii=False), encoding="utf-8")
    (INDEX_DIR / META_JSON).write_text(
        json.dumps({"hashes": hashes}, ensure_ascii=False), encoding="utf-8")


def build_index() -> int:
    root = validate_kb_path()
    t0 = time.time()
    chunks: list[dict] = []
    for f, rel in load_text_files(root):
        text = f.read_text(encoding="utf-8", errors="ignore")
        for c in chunk_file(text, rel, is_markdown=f.suffix.lower() == ".md"):
            chunks.append({"path": c.path, "heading": c.heading, "text": c.text})
    vectors = _embed_normalized([c["text"] for c in chunks])
    _save(chunks, vectors, _scan_hashes(root))
    print(f"全量索引完成: {len(chunks)} 块 / {vectors.shape[1]} 维 / {time.time() - t0:.1f}s")
    return len(chunks)


def incremental_index() -> dict:
    """增量入库：按文件哈希只重嵌新增/变更部分；无 meta 时回退全量。"""
    root = validate_kb_path()
    t0 = time.time()
    meta_path = Path(INDEX_DIR) / META_JSON
    if not meta_path.exists():
        build_index()
        from .hybrid import build_bm25_index
        from .store import build_chroma
        build_bm25_index()
        build_chroma()
        return {"mode": "full", "added": 0, "removed": 0, "total": None, "seconds": time.time() - t0}

    chunks = json.loads((Path(INDEX_DIR) / "chunks.json").read_text(encoding="utf-8"))
    vectors = np.load(Path(INDEX_DIR) / "vectors.npy")
    old_hashes = json.loads(meta_path.read_text(encoding="utf-8"))["hashes"]
    new_hashes = _scan_hashes(root)

    changed = {rel for rel, h in new_hashes.items() if old_hashes.get(rel) != h}
    deleted = set(old_hashes) - set(new_hashes)
    if not changed and not deleted:
        return {"mode": "clean", "added": 0, "removed": 0,
                "total": len(chunks), "seconds": time.time() - t0}

    drop = changed | deleted
    keep_idx = [i for i, c in enumerate(chunks) if c["path"] not in drop]
    kept_chunks = [chunks[i] for i in keep_idx]
    kept_vecs = vectors[keep_idx] if keep_idx else np.zeros((0, vectors.shape[1]), dtype=np.float32)

    added_chunks, added_texts = _chunk_files(root, changed)
    added_vecs = _embed_normalized(added_texts) if added_texts else np.zeros((0, vectors.shape[1]), dtype=np.float32)

    all_chunks = kept_chunks + added_chunks
    all_vecs = np.vstack([kept_vecs, added_vecs]) if (len(kept_vecs) or len(added_vecs)) else kept_vecs
    _save(all_chunks, all_vecs, new_hashes)
    return {"mode": "delta", "added": len(added_chunks), "removed": len(chunks) - len(kept_chunks),
            "total": len(all_chunks), "seconds": time.time() - t0}


def staleness() -> dict:
    """索引新鲜度：对比磁盘文件指纹与 meta.json 已索引指纹（ask 的 L2 查询兜底）。

    mode=missing 表示索引尚未建立；stale 表示有文件新增/变更/删除未入库。
    只核对切块主索引；wiki 摘要树的路由新鲜度由 index --wiki 维护。
    """
    root = validate_kb_path()
    meta_path = Path(INDEX_DIR) / META_JSON
    if not meta_path.exists():
        return {"mode": "missing", "changed": 0, "deleted": 0, "total": 0}
    old = json.loads(meta_path.read_text(encoding="utf-8"))["hashes"]
    new = _scan_hashes(root)
    changed = sum(1 for rel, h in new.items() if old.get(rel) != h)
    deleted = len(set(old) - set(new))
    return {"mode": "stale" if (changed or deleted) else "clean",
            "changed": changed, "deleted": deleted, "total": len(new)}


def sync_incremental() -> dict:
    """增量入库并同步 BM25；Chroma 打过期标记延迟同步。

    ask/hybrid 的默认检索路径只依赖 BM25+numpy（秒级），Chroma 全量重灌
    随语料线性变慢，不阻塞收口——下次 --mode chroma 查询时按标记重建。
    `add` 命令与 ask 的过期自动补齐共用此入口——保证两条路径行为一致。
    """
    r = incremental_index()
    if r["mode"] in ("delta", "full"):
        from .hybrid import build_bm25_index
        from .store import mark_stale

        build_bm25_index()
        mark_stale()
    return r


def query(question: str, k: int = 5) -> list[dict]:
    index_dir = Path(INDEX_DIR)
    vectors = np.load(index_dir / "vectors.npy")
    chunks = json.loads((index_dir / "chunks.json").read_text(encoding="utf-8"))
    q = np.array(embed([question])[0], dtype=np.float32)
    q /= max(np.linalg.norm(q), 1e-10)
    scores = vectors @ q
    return [{**chunks[i], "score": round(float(scores[i]), 4)}
            for i in np.argsort(-scores)[:k]]
