"""建索引与检索：numpy 余弦即可覆盖万级以下语料，向量库留到阶段 1。"""

import json
import time
from pathlib import Path

import numpy as np

from .authority import apply as authority_apply
from .config import INDEX_DIR, KB_PATH, validate_kb_path
from .embedder import embed
from .loader import chunk_file, load_text_files


def build_index() -> int:
    root = validate_kb_path()
    t0 = time.time()
    chunks = []
    for f, rel in load_text_files(root):
        text = f.read_text(encoding="utf-8", errors="ignore")
        chunks.extend(chunk_file(text, rel, is_markdown=f.suffix.lower() == ".md"))
    vectors = np.array(embed([c.text for c in chunks]), dtype=np.float32)
    norms = np.linalg.norm(vectors, axis=1, keepdims=True)
    vectors = vectors / np.maximum(norms, 1e-10)

    INDEX_DIR.mkdir(exist_ok=True)
    np.save(INDEX_DIR / "vectors.npy", vectors)
    (INDEX_DIR / "chunks.json").write_text(
        json.dumps([{"path": c.path, "heading": c.heading, "text": c.text} for c in chunks],
                   ensure_ascii=False),
        encoding="utf-8",
    )
    print(f"索引完成: {len(chunks)} 块 / {vectors.shape[1]} 维 / {time.time() - t0:.1f}s")
    return len(chunks)


def query(question: str, k: int = 5) -> list[dict]:
    index_dir = Path(INDEX_DIR)
    vectors = np.load(index_dir / "vectors.npy")
    chunks = json.loads((index_dir / "chunks.json").read_text(encoding="utf-8"))
    q = np.array(embed([question])[0], dtype=np.float32)
    q /= max(np.linalg.norm(q), 1e-10)
    scores = vectors @ q
    return [{**chunks[i], "score": round(float(scores[i]), 4)}
            for i in np.argsort(-scores)[:k]]
