"""索引引擎：内容寻址增量 + 分段断点续传，`index`（全量）与 `add`（增量）共用一条流水线。

核心事实：同一向量空间里，同一段文本的嵌入向量是确定的——所以向量可以按
「空间指纹 + 嵌入输入」的 sha1 寻址复用：
- add：文件指纹未变的整文件直接沿用；变更文件重新切块，块哈希命中缓存则零成本
  沿用，只嵌真缺的块（改一段 ≠ 重嵌整个文件）。
- index：无视文件指纹全部重切（切块器改了也能跑），块哈希命中旧索引或上次中断
  的半成品仍沿用；--force 才全部重嵌。成本 ∝ 真正变化的块，不 ∝ 库大小。
- 续传：每 EMBED_SEGMENT 块写一个自带哈希清单的 part（npz，内容寻址自证），
  重跑自动捡起；本地 fastembed 后端每段换子进程跑，重置 ONNX 累积内存（jetsam 防御）。
- 门禁：meta.json 记 embed_identity()，查询/增量前比对——空间不一致就明确拒绝，
  绝不混空间（查询向量×异空间文档向量=乱序），也绝不静默触发小时级重建（旧 add 行为）。
"""

import fcntl
import hashlib
import json
import multiprocessing
import os
import shutil
import sys
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np

from .config import (EMBED_DIMS, EMBED_MODEL, EMBED_QUERY_INSTRUCT, EMBED_SEGMENT,
                     INDEX_DIR, validate_kb_path)
from .embedder import (DOC_PREFIX_VERSION, doc_identity, embed, embed_dim, embed_identity,
                       get_backend)
from .loader import chunk_file, load_text_files

META_JSON = "meta.json"


class IndexNotQueryable(RuntimeError):
    """索引缺失或与当前嵌入配置不同空间——message 即给用户的行动指令。

    必须是普通异常：agent 工具审计与 MCP 服务端靠 except Exception 兜底，
    SystemExit 属 BaseException 会穿透它们（留无痕探测通道 / 杀掉服务）。"""


def embed_input(c: dict) -> str:
    """文档侧嵌入输入 = 路径+标题链+正文；查询侧保持裸问题（两侧不对称是设计）。

    BM25 语料一直含路径标题而向量路没有——拼进嵌入是 Anthropic 上下文检索的
    零 LLM 降本版，让块自带文档级语境。拼法版本钉在 embedder.DOC_PREFIX_VERSION，
    改拼法必须同步改标签，否则旧向量会被错误复用。
    """
    return f"{c['path']} {c['heading']}\n{c['text']}"


def chunk_hash(doc_space: str, text: str) -> str:
    """块内容寻址：**文档侧**空间指纹参与哈希——换模型后旧哈希自然全部失效，
    缓存不会跨空间误用；查询侧指令位不参与（文档向量与它无关，见 embedder.doc_identity）。"""
    return hashlib.sha1(f"{doc_space}\x00{text}".encode("utf-8")).hexdigest()


def scan_file_hashes(root: Path) -> dict[str, str]:
    return {rel: hashlib.sha1(f.read_bytes()).hexdigest() for f, rel in load_text_files(root)}


def _build_dir() -> Path:
    return Path(INDEX_DIR) / ".build"


def _read_meta() -> dict | None:
    p = Path(INDEX_DIR) / META_JSON
    return json.loads(p.read_text(encoding="utf-8")) if p.exists() else None


def _legacy_identity_ok(meta: dict) -> bool:
    """中间代旧索引（embed 字段自述 model+doc_prefix）：按其自述与当前配置逐项比对。
    最老的裸 meta（无 embed 字段）不在此放行——文档侧格式无从自证（2026-10-21 向量实测：
    生产库为无路径前缀的裸正文格式，前缀从未随重建上线），增量必须全量重建。"""
    e = meta.get("embed") or {}
    return (get_backend().kind == "fastembed" and not EMBED_QUERY_INSTRUCT and not EMBED_DIMS
            and e.get("model") == EMBED_MODEL and e.get("doc_prefix") == DOC_PREFIX_VERSION)


def _bare_legacy_queryable(meta: dict) -> bool:
    """初代裸 meta（只有 hashes）的**查询侧**接管判定：查询向量空间可由「模型+维度」
    唯一确定——历史上只发布过 bge-small-zh 一代裸索引，且要求当前就是该模型+裸查询。
    文档侧格式无证，增量/重建仍走 mismatch（见 rebuild）。"""
    if not (get_backend().kind == "fastembed" and not EMBED_QUERY_INSTRUCT and not EMBED_DIMS
            and not (meta.get("embed")) and EMBED_MODEL == "BAAI/bge-small-zh-v1.5"):
        return False
    try:
        return int(np.load(Path(INDEX_DIR) / "vectors.npy", mmap_mode="r").shape[1]) \
            == get_backend().dim
    except Exception:
        return False


def _mismatch_msg(index_id: str, config_id: str) -> str:
    return (f"向量空间不一致，已拒绝：索引由 [{index_id}] 生成，当前嵌入配置是 [{config_id}]。"
            "不同模型的向量不可混用（相似度没有意义）。请运行 uv run kb-rag index 全量重建"
            "（引擎复用所有可自证同空间的向量，其余重嵌；旧格式库无法自证的部分会整体重嵌），"
            "或把 .env 的 EMBED_* 配置改回原样。")


def ensure_queryable() -> None:
    """查询/增量前的空间门禁：索引缺失、空间不一致、文件错位，抛 IndexNotQueryable。"""
    meta = _read_meta()
    if meta is None:
        raise IndexNotQueryable("索引不存在：先运行 uv run kb-rag index 建立索引（可断点续传）")
    cur = embed_identity()
    idx = (meta.get("embed") or {}).get("identity")
    if idx is None:
        if not (_legacy_identity_ok(meta) or _bare_legacy_queryable(meta)):
            raise IndexNotQueryable(_mismatch_msg("<旧版索引（无空间指纹）>", cur))
        # 旧索引接管：中间代完全放行；裸 meta 仅查询侧（增量由 rebuild 的 mismatch 拦）
    elif idx != cur:
        raise IndexNotQueryable(_mismatch_msg(idx, cur))
    dim = meta.get("embed", {}).get("dim")
    if dim is not None and dim != embed_dim():  # 兜底：指纹一致但维度对不上（异常情形）
        raise IndexNotQueryable(_mismatch_msg(f"{idx} 维度{dim}", f"{cur} 维度{embed_dim()}"))
    residue = _save_residue()
    if residue:  # 落盘残件：同块数错位 meta 校验查不出，残件是唯一目击者（复验 P1-1）
        raise IndexNotQueryable(_residue_msg(residue))
    total = meta.get("total_chunks")
    if total is not None:  # 落盘中断防御：meta 是最后写的，旧 meta+新向量必在此现形
        try:
            n = int(np.load(Path(INDEX_DIR) / "vectors.npy", mmap_mode="r").shape[0])
        except (FileNotFoundError, ValueError) as e:
            raise IndexNotQueryable(f"vectors.npy 缺失或不可读（{e}）：运行 uv run kb-rag index 重建。") from e
        if n != total:
            raise IndexNotQueryable(_misaligned_msg(expected=total, actual=n))


def _save_residue() -> list[str]:
    """上次落盘被杀的残件（正常完成时 tmp 已被 replace 消失）。"""
    try:
        return sorted(f.name for f in Path(INDEX_DIR).glob(".*.tmp"))
    except OSError:
        return []


def _residue_msg(residue: list[str]) -> str:
    return (f"检测到落盘残件 {residue}：上次写入被中断，索引可能处于新旧文件错位状态，拒绝检索。"
            "若索引正在构建中（有 kb-rag index/add 在跑）请稍后重试；"
            "否则运行 uv run kb-rag index 自愈（引擎会先隔离残件与旧索引再续传重建）。")


def _misaligned_msg(expected: int, actual: int) -> str:
    return (f"索引文件错位（meta 记 {expected} 块，vectors.npy 实有 {actual} 行，"
            "上次落盘中断或并发写入）：本次已拒绝检索。"
            "运行 uv run kb-rag index 重建（引擎会先把错位文件移入 index/.corrupt-* 再续传重建，"
            ".build 半成品不受影响）。")


def _load_old_index(meta: dict) -> dict:
    """读旧索引三件套；行数错位（上次落盘中断）时把四件套整体移入 .corrupt-<时间戳>
    再报错——修复指令必须可自愈，不能让人对着"重跑 index"的提示撞同一堵墙（审查 P0-1）。
    .build 半成品不受影响，重跑自动续传。"""
    vec_path = Path(INDEX_DIR) / "vectors.npy"
    chunks_path = Path(INDEX_DIR) / "chunks.json"
    hp = Path(INDEX_DIR) / "chunk_hashes.json"
    residue = _save_residue()
    chunks = json.loads(chunks_path.read_text(encoding="utf-8"))
    vecs = np.load(vec_path, mmap_mode="r")  # 只映射不驻留，按行取
    hashes = json.loads(hp.read_text(encoding="utf-8")) if hp.exists() else None
    misaligned = len(chunks) != vecs.shape[0] or (hashes is not None and len(hashes) != len(chunks))
    if misaligned or residue:
        # 错位或落盘残件：四件套整体隔离（连同残件）。同块数错位 meta 校验查不出，
        # 残件是唯一目击者——两者都宁可重建也不带病续跑（审查 P0-1 / 复验 P1-1）
        stamp = time.strftime("%Y%m%d-%H%M%S")
        bad = Path(INDEX_DIR) / f".corrupt-{stamp}"
        bad.mkdir(parents=True, exist_ok=True)
        files = [vec_path, chunks_path, hp, Path(INDEX_DIR) / META_JSON]
        files += [Path(INDEX_DIR) / name for name in residue]
        for f in files:
            if f.exists():
                os.replace(f, bad / f.name)
            reason = "落盘残件存在" if residue else \
                f"chunks {len(chunks)} 行 ≠ vectors {vecs.shape[0]} 行"
        raise IndexNotQueryable(
            f"索引完整性异常（{reason}，上次落盘中断）：已把旧索引移入 index/.corrupt-{stamp}，"
            "请重跑本次命令完成重建（.build 半成品仍有效，可续传）。")
    return {"chunks": chunks, "chunk_hashes": hashes, "vectors": vecs}


def embed_texts_normalized(texts: list[str], batch: int = 256) -> np.ndarray:
    """批量嵌入并 L2 归一化；外层分批+预分配，避免整库 Python 浮点列表滞留内存。

    内存大户在 embedder 内部——fastembed 已钉 batch_size=4（默认批 256 在真实长块上
    单次推理 RSS ~7.9GB 被 jetsam 杀）；本层批大小只影响 numpy 转换的瞬时占用。
    按长度排序分批试过（+14% 吞吐）但在长块密集区四连死（rc=137），已回退。
    """
    out: np.ndarray | None = None
    for i in range(0, len(texts), batch):
        vecs = np.array(embed(texts[i:i + batch]), dtype=np.float32)
        vecs /= np.maximum(np.linalg.norm(vecs, axis=1, keepdims=True), 1e-10)
        if out is None:
            out = np.zeros((len(texts), vecs.shape[1]), dtype=np.float32)
        out[i:i + len(vecs)] = vecs
    return out if out is not None else np.zeros((0, embed_dim()), dtype=np.float32)


def _embed_segment_worker(texts: list[str]) -> np.ndarray:
    """子进程入口：fresh 进程 = fresh ONNX arena，段间不累积内存。"""
    return embed_texts_normalized(texts)


def _embed_in_subprocess(texts: list[str]) -> np.ndarray:
    ctx = multiprocessing.get_context("spawn")
    with ProcessPoolExecutor(max_workers=1, mp_context=ctx) as ex:  # 每段新建→用完即销毁
        return ex.submit(_embed_segment_worker, texts).result()


def _salvage_parts(doc_space: str) -> dict[str, tuple[np.ndarray, int]]:
    """回收上次中断的半成品：part 里存的就是文档侧向量，用 doc_identity 判空间一致
    （part 自带哈希清单自证内容有效性）。内存上界 ≈ 全部残件行数×维度×4B
    （93k×1024 约 380MB）+ 最终 vstack 同量级，16GB 机器实测口径 ~1.2GB 成立；
    调大 EMBED_SEGMENT 只影响单 part 大小，不改变总上界。"""
    hits: dict[str, tuple[np.ndarray, int]] = {}
    bdir = _build_dir()
    plan = bdir / "plan.json"
    if not plan.exists():
        return hits
    try:
        if json.loads(plan.read_text(encoding="utf-8")).get("identity") != doc_space:
            return {}  # 换了空间：半成品全部作废
    except json.JSONDecodeError:
        return {}  # plan 残件不可信：放弃回收（parts 会在新计划里被清掉）
    for part in sorted((bdir / "parts").glob("part_*.npz")):
        try:
            data = np.load(part, allow_pickle=False)
            hashes, vecs = data["hashes"], data["vecs"]
        except Exception:
            continue  # 写入中断的半个 part：残件丢弃，该段重嵌
        if len(hashes) != vecs.shape[0]:
            continue
        for i, h in enumerate(hashes):
            hits[str(h)] = (vecs, i)
    return hits


def _rechunk_files(root: Path, rels: set[str]) -> list[dict]:
    out: list[dict] = []
    for f, rel in load_text_files(root):
        if rel not in rels:
            continue
        text = f.read_text(encoding="utf-8", errors="ignore")
        for c in chunk_file(text, rel, is_markdown=f.suffix.lower() == ".md"):
            out.append({"path": c.path, "heading": c.heading, "text": c.text})
    return out


def _save_index(chunks: list[dict], vecs: np.ndarray, chunk_hashes: list[str],
                file_hashes: dict[str, str], identity: str) -> None:
    """原子性尽量高：逐文件写临时+replace；跨文件仍非原子，行数错位由 _load_old_index 兜底。"""
    INDEX_DIR.mkdir(parents=True, exist_ok=True)
    meta = {"hashes": file_hashes, "total_chunks": len(chunks),
            "embed": {"identity": identity, "model": EMBED_MODEL,
                      "backend": get_backend().kind,
                      "dim": int(vecs.shape[1]) if vecs.size else embed_dim(),
                      "query": "inst" if EMBED_QUERY_INSTRUCT else "raw",
                      "doc_prefix": DOC_PREFIX_VERSION}}

    def put(name: str, write) -> None:
        tmp = INDEX_DIR / f".{name}.tmp"
        with open(tmp, "wb") as f:
            write(f)
        os.replace(tmp, INDEX_DIR / name)

    put("vectors.npy", lambda f: np.save(f, vecs))
    put("chunks.json", lambda f: f.write(json.dumps(chunks, ensure_ascii=False).encode("utf-8")))
    put("chunk_hashes.json", lambda f: f.write(json.dumps(chunk_hashes).encode("utf-8")))
    put("meta.json", lambda f: f.write(json.dumps(meta, ensure_ascii=False).encode("utf-8")))


def _acquire_build_lock() -> tuple[object | None, str]:
    """构建独占锁（flock，非阻塞）：并发 add/index 会互拆 .build 续传现场并交叉落盘
    （审查 P1-2）。返回 (锁句柄或 None, 持有者信息)；None=已有构建在跑。"""
    Path(INDEX_DIR).mkdir(parents=True, exist_ok=True)
    path = Path(INDEX_DIR) / ".lock"
    # 不加 O_TRUNC：后到竞争者若截断文件会把持有者 PID 抹掉（复验 P2-3）
    f = os.fdopen(os.open(path, os.O_CREAT | os.O_WRONLY, 0o644), "w")
    try:
        fcntl.flock(f, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError:
        try:
            holder = path.read_text(encoding="utf-8").strip()
        except OSError:
            holder = ""
        return None, holder or "?"
    f.write(str(os.getpid()))
    f.flush()
    return f, f"PID {os.getpid()}"


def rebuild(mode: str = "delta", force: bool = False, quiet: bool = False) -> dict:
    """统一构建入口。mode=delta 按文件指纹增量；mode=full 重切全部（仍按块哈希复用）；
    force=True 无视缓存全部重嵌。返回统计 dict，mode 可为 full/delta/clean/mismatch/locked。"""
    t0 = time.time()
    lock_fd, holder = _acquire_build_lock()
    if lock_fd is None:
        return {"mode": "locked", "holder": holder, "seconds": time.time() - t0}
    try:
        return _rebuild_locked(mode, force, quiet, t0)
    finally:
        lock_fd.close()  # 关闭即释放 flock


def _rebuild_locked(mode: str, force: bool, quiet: bool, t0: float) -> dict:
    root = validate_kb_path()
    identity = embed_identity()       # 门禁键：含查询侧指令位
    doc_space = doc_identity()        # 缓存键：文档向量与查询侧指令无关
    meta = _read_meta()
    new_file_hashes = scan_file_hashes(root)

    if meta is not None and mode == "delta":
        old_id = meta.get("embed", {}).get("identity")
        ok = (old_id == identity) if old_id is not None else _legacy_identity_ok(meta)
        if not ok:
            return {"mode": "mismatch", "index_identity": old_id or "<旧版索引>",
                    "config_identity": identity, "seconds": time.time() - t0}
        old_fh = meta.get("hashes", {})
        changed = {rel for rel, h in new_file_hashes.items() if old_fh.get(rel) != h}
        deleted = set(old_fh) - set(new_file_hashes)
        if not changed and not deleted:
            total = meta.get("total_chunks")
            if total is None:  # 旧版 meta 无总数：只读 npy 头取行数，不解析 102MB chunks.json
                total = int(np.load(Path(INDEX_DIR) / "vectors.npy", mmap_mode="r").shape[0])
            return {"mode": "clean", "added": 0, "reused": 0, "embedded": 0,
                    "removed": 0, "total": total, "seconds": time.time() - t0}

    try:
        old = _load_old_index(meta) if meta is not None else None
    except IndexNotQueryable:
        # 完整性异常已就地隔离（.corrupt-*）：状态等价于首次构建，本调用直接续跑
        # （.build 半成品不受隔离影响，仍可捡回）；不在中间态抛给用户多跑一次
        old = None
        meta = None
    # 缓存可用性按**文档侧**指纹判：查询指令位换挡时门禁照拦（delta mismatch），
    # 但文档向量本就与指令无关，full 重建必须照常复用（审查 P1-5）
    stored_gate = meta.get("embed", {}).get("identity") if meta else None
    doc_match = (stored_gate.rsplit(":", 1)[0] == doc_space) if stored_gate \
        else _legacy_identity_ok(meta or {})
    old_usable = old is not None and doc_match
    if old is not None and old_usable and old["chunk_hashes"] is None:
        # 旧版索引无块哈希清单：按当前文档侧身份回填（93k 块 sha1 秒级），此后永续复用
        old["chunk_hashes"] = [chunk_hash(doc_space, embed_input(c)) for c in old["chunks"]]

    hash_to_src: dict[str, tuple] = {}
    if not force and old is not None and old_usable and old["chunk_hashes"] is not None:
        hash_to_src = {h: (old["vectors"], i) for i, h in enumerate(old["chunk_hashes"])}
    if not force:
        for h, src in _salvage_parts(doc_space).items():
            hash_to_src.setdefault(h, src)

    old_fh = meta.get("hashes", {}) if meta else {}
    rechunk = set(new_file_hashes) if mode == "full" else {
        rel for rel, h in new_file_hashes.items() if old_fh.get(rel) != h}
    deleted = set(old_fh) - set(new_file_hashes)

    old_total = len(old["chunks"]) if old else 0
    kept_chunks: list[dict] = []
    kept_hashes: list[str] = []
    kept_vecs = np.zeros((0, old["vectors"].shape[1] if old else embed_dim()), dtype=np.float32)
    if old is not None and mode == "delta":
        drop = rechunk | deleted
        keep_idx = [i for i, c in enumerate(old["chunks"]) if c["path"] not in drop]
        kept_chunks = [old["chunks"][i] for i in keep_idx]
        kept_hashes = [old["chunk_hashes"][i] for i in keep_idx]
        if keep_idx:
            kept_vecs = np.asarray(old["vectors"][keep_idx], dtype=np.float32)

    new_chunks = _rechunk_files(root, rechunk)
    reuse_vecs: list[np.ndarray] = []
    pending: list[tuple[str, str]] = []
    new_h: list[str] = []
    for c in new_chunks:
        text = embed_input(c)
        h = chunk_hash(doc_space, text)
        new_h.append(h)
        src = hash_to_src.get(h)
        if src is None:
            pending.append((h, text))
        else:
            arr, row = src
            reuse_vecs.append(np.asarray(arr[row], dtype=np.float32))
    del hash_to_src  # 释放对旧 mmap 的行引用，落盘覆写前解除映射

    embedded: list[np.ndarray] = []
    if pending:
        bdir = _build_dir()
        shutil.rmtree(bdir, ignore_errors=True)
        (bdir / "parts").mkdir(parents=True)
        (bdir / "plan.json").write_text(json.dumps(
            {"identity": doc_space, "pending": len(pending),
             "segment": max(1, EMBED_SEGMENT), "created": time.strftime("%Y-%m-%d %H:%M:%S")},
            ensure_ascii=False), encoding="utf-8")
        isolate = get_backend().isolate_segments
        seg = max(1, EMBED_SEGMENT)
        t_emb = time.time()
        for start in range(0, len(pending), seg):
            group = pending[start:start + seg]
            texts = [t for _, t in group]
            # 单段任务不值得付子进程冷启动（模型加载 ~2-3s）；多段才隔离
            vecs = (_embed_in_subprocess(texts) if isolate and len(pending) > seg
                    else embed_texts_normalized(texts))
            np.savez(bdir / "parts" / f"part_{start:08d}.npz",
                     hashes=np.array([h for h, _ in group]), vecs=vecs)
            embedded.append(vecs)
            if not quiet:
                done = start + len(group)
                el = time.time() - t_emb
                print(f"[indexer] 嵌入 {done}/{len(pending)} 块 "
                      f"({done * 100 // len(pending)}%) / 已用 {el:.0f}s"
                      f" / 预计还需 {el / done * (len(pending) - done):.0f}s",
                      file=sys.stderr, flush=True)

    all_chunks = kept_chunks + new_chunks
    parts = [a for a in (kept_vecs, *reuse_vecs, *embedded) if a.shape[0]]
    # 空数组要滤掉：零行 kept_vecs 的维度可能属于旧空间（换空间全量重建时），vstack 会因维度不合报错
    all_vecs = (np.vstack(parts) if parts else np.zeros((0, embed_dim()), dtype=np.float32))
    _save_index(all_chunks, all_vecs, kept_hashes + new_h, new_file_hashes, identity)
    shutil.rmtree(_build_dir(), ignore_errors=True)
    return {"mode": "full" if (mode == "full" or meta is None) else "delta",
            "added": len(new_chunks), "reused": len(reuse_vecs), "embedded": len(pending),
            "removed": old_total - len(kept_chunks), "total": len(all_chunks),
            "seconds": time.time() - t0}
