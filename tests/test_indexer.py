"""索引引擎回归：内容寻址复用、断点续传、空间门禁、旧索引接管、归一化位置保持。

不加载真实模型：FakeBackend 用文本 sha1 派生确定性 8 维向量（同文本必同向量），
"复用是否真的发生"由假后端见过哪些文本直接断言。语料与索引目录全部指向
tmp_path，永不触碰真实总库与真实 index/（conftest 已隔离，这里再隔离一层）。
"""

import hashlib
import json
from types import SimpleNamespace

import numpy as np
import pytest

import kb_rag.indexer as indexer
import kb_rag.search as search


class FakeBackend:
    kind = "fake"
    isolate_segments = False
    model = "fake-model"
    dims = None
    dim = 8

    def __init__(self):
        self.seen: list[str] = []
        self.calls = 0
        self.fail_after_calls: int | None = None  # 模拟进程被杀：第 N 次调用后抛错

    def embed_documents(self, texts):
        self.calls += 1
        if self.fail_after_calls is not None and self.calls > self.fail_after_calls:
            raise RuntimeError("simulated kill")
        self.seen.extend(texts)
        out = []
        for t in texts:
            v = np.frombuffer(hashlib.sha1(t.encode()).digest()[:8], dtype=np.uint8)
            out.append(v.astype(np.float32) + 1.0)  # +1 防零向量
        return out


def _write(kb, rel, text):
    p = kb / rel
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(text, encoding="utf-8")


TWO_SECTION = "# 标题\n\n## 甲\n\n" + "甲段落。\n\n" * 20 + "\n## 乙\n\n" + "乙段落。\n\n" * 20


@pytest.fixture
def env(tmp_path, monkeypatch):
    import kb_rag.config as config

    kb = tmp_path / "kb"
    kb.mkdir()
    idx = tmp_path / "index"
    for mod in (config, indexer, search):
        monkeypatch.setattr(mod, "INDEX_DIR", idx)
    monkeypatch.setattr(config, "KB_PATH", kb)
    fake = FakeBackend()
    # 真实格式 = f"{doc_identity}:{raw|inst}"（指令位在最后，rsplit 一次拆出 doc 段）
    identity = "fake:fake-model:path+heading:raw"
    monkeypatch.setattr(indexer, "get_backend", lambda: fake)
    monkeypatch.setattr(indexer, "embed_identity", lambda: identity)
    # 生产中 doc_identity 与 embed_identity 同源于配置；测试里两者都要钉到 fake 空间
    monkeypatch.setattr(indexer, "doc_identity", lambda: "fake:fake-model:path+heading")
    monkeypatch.setattr(indexer, "embed", fake.embed_documents)
    monkeypatch.setattr(indexer, "embed_dim", lambda: 8)
    monkeypatch.setattr(search, "embed_queries", fake.embed_documents)
    return SimpleNamespace(kb=kb, idx=idx, fake=fake, identity=identity)


def test_full_then_clean_delta(env):
    _write(env.kb, "a.md", TWO_SECTION)
    _write(env.kb, "b.md", "## 唯一\n\n正文。\n")
    r = indexer.rebuild("full")
    assert r["mode"] == "full" and r["total"] == 3 and r["embedded"] == 3
    meta = json.loads((env.idx / "meta.json").read_text())
    assert meta["embed"]["identity"] == env.identity
    assert meta["total_chunks"] == 3
    assert len(json.loads((env.idx / "chunk_hashes.json").read_text())) == 3
    r2 = indexer.rebuild("delta")
    assert r2["mode"] == "clean" and r2["embedded"] == 0


def test_delta_reuses_unchanged_chunk_and_embeds_only_changed(env):
    _write(env.kb, "a.md", TWO_SECTION)
    indexer.rebuild("full")
    env.fake.seen.clear()
    _write(env.kb, "a.md", TWO_SECTION.replace("乙段落。", "乙改了。"))
    r = indexer.rebuild("delta")
    # 甲节块的 (路径+标题+正文) 未变 → 哈希命中零成本沿用；乙节块真重嵌
    assert (r["added"], r["reused"], r["embedded"], r["removed"], r["total"]) == (2, 1, 1, 2, 2)
    assert len(env.fake.seen) == 1 and "乙改了。" in env.fake.seen[0]


def test_identity_mismatch_refuses_delta_and_query(env, monkeypatch):
    _write(env.kb, "a.md", TWO_SECTION)
    indexer.rebuild("full")
    monkeypatch.setattr(indexer, "embed_identity", lambda: env.identity + "x")  # 模拟换嵌入配置
    r = indexer.rebuild("delta")
    assert r["mode"] == "mismatch" and r["index_identity"] == env.identity
    with pytest.raises(indexer.IndexNotQueryable, match="向量空间不一致"):
        search.query("任何问题")


def test_resume_after_kill_never_reembeds_saved_segment(env, monkeypatch):
    monkeypatch.setattr(indexer, "EMBED_SEGMENT", 2)
    for i in range(5):
        _write(env.kb, f"f{i}.md", f"## 节{i}\n\n内容{i}。\n")
    env.fake.fail_after_calls = 1  # 第 1 段（2 块）落盘后被"杀"
    with pytest.raises(RuntimeError, match="simulated kill"):
        indexer.rebuild("full")
    parts = list((env.idx / ".build" / "parts").glob("part_*.npz"))
    assert len(parts) == 1
    first_two = list(env.fake.seen)

    env.fake.fail_after_calls = None
    env.fake.seen.clear()
    r = indexer.rebuild("full")
    assert r["mode"] == "full" and r["total"] == 5
    assert r["embedded"] == 3  # 只嵌缺的 3 块
    assert set(env.fake.seen).isdisjoint(first_two)  # 已落盘的 2 块零重复
    assert not (env.idx / ".build").exists()  # 完成后半成品清理


def test_force_ignores_cache(env):
    _write(env.kb, "a.md", TWO_SECTION)
    indexer.rebuild("full")
    env.fake.seen.clear()
    r = indexer.rebuild("full", force=True)
    assert r["reused"] == 0 and r["embedded"] == r["total"] == 2


def test_query_instruct_toggle_keeps_doc_cache(env, monkeypatch):
    """审查 P1-5：文档向量与查询侧指令无关——切指令位门禁要拦查询，但全量重建时
    文档向量必须 100% 复用，而不是白重嵌 17 小时得到逐位相同的结果。"""
    _write(env.kb, "a.md", TWO_SECTION)
    indexer.rebuild("full")
    monkeypatch.setattr(indexer, "embed_identity",
                        lambda: "fake:fake-model:path+heading:inst")  # 模拟开指令前缀
    assert indexer.rebuild("delta")["mode"] == "mismatch"  # 查询侧换空间：增量拒绝
    r = indexer.rebuild("full")
    assert r["reused"] == r["total"] == 2 and r["embedded"] == 0  # 文档缓存原封不动


def test_build_lock_blocks_second_builder(env, monkeypatch):
    import fcntl

    env.idx.mkdir(parents=True, exist_ok=True)
    holder = open(env.idx / ".lock", "w")
    fcntl.flock(holder, fcntl.LOCK_EX)  # 模拟另一进程持锁构建
    try:
        r = indexer.rebuild("delta")
        assert r["mode"] == "locked" and r["holder"]
    finally:
        holder.close()
    assert indexer.rebuild("delta")["mode"] in ("full", "delta", "clean")  # 释放后可进


def test_legacy_index_taken_over_without_rebuild(env, monkeypatch):
    _write(env.kb, "a.md", TWO_SECTION)
    _write(env.kb, "keep.md", "## 留\n\n原样。\n")
    indexer.rebuild("full")
    # 把新式索引退回旧版形态：meta 无 identity、删 chunk_hashes.json——模拟现役 bge 索引
    meta = json.loads((env.idx / "meta.json").read_text())
    legacy = {"hashes": meta["hashes"],
              "embed": {"model": "fake-model", "doc_prefix": "path+heading"}}
    (env.idx / "meta.json").write_text(json.dumps(legacy))
    (env.idx / "chunk_hashes.json").unlink()
    env.fake.kind = "fastembed"  # _legacy_identity_ok 只认 fastembed 旧索引
    monkeypatch.setattr(indexer, "EMBED_MODEL", "fake-model")

    env.fake.seen.clear()
    # 关键场景：变更文件里未变的块要经旧索引回填的哈希零成本复用（新增一节、原节不动）
    _write(env.kb, "keep.md", "## 留\n\n原样。\n\n## 新增\n\n新内容。\n")
    r = indexer.rebuild("delta")
    assert r["mode"] == "delta" and r["reused"] == 1 and r["embedded"] == 1
    assert indexer.ensure_queryable() is None  # 接管后空间门禁放行
    assert len(json.loads((env.idx / "chunk_hashes.json").read_text())) == r["total"]


def test_bare_legacy_meta_query_only(env, monkeypatch):
    """初代裸 meta（无 embed 字段，= 现役生产形态）：查询侧接管放行，增量拒绝并指重建。
    生产向量实测为无前缀裸正文格式，文档侧格式无证，混格增量会污染库。"""
    _write(env.kb, "a.md", TWO_SECTION)
    indexer.rebuild("full")
    (env.idx / "meta.json").write_text(json.dumps(
        {"hashes": json.loads((env.idx / "meta.json").read_text())["hashes"]}))
    (env.idx / "chunk_hashes.json").unlink()
    env.fake.kind = "fastembed"  # 历史形态只有 fastembed 一代
    monkeypatch.setattr(indexer, "EMBED_MODEL", "BAAI/bge-small-zh-v1.5")  # 裸 meta 钉历史模型
    # 维度自证：裸 meta 靠「历史模型 + 向量维度一致」放行查询侧（本库 dim=8 与后端一致）
    assert indexer.ensure_queryable() is None
    r = indexer.rebuild("delta")
    assert r["mode"] == "mismatch"  # 文档格式无证：增量拒绝


def test_save_residue_blocks_query_and_self_heals(env, monkeypatch):
    """复验 P1-1：同块数落盘中断 meta 校验查不出，tmp 残件是唯一目击者——
    查询必须拒绝；重建入口把残件与旧索引一并隔离，重跑可自愈续传。"""
    _write(env.kb, "a.md", TWO_SECTION)
    indexer.rebuild("full")
    (env.idx / ".vectors.npy.tmp").write_bytes(b"half-written")  # 模拟被杀于 replace 之间
    with pytest.raises(indexer.IndexNotQueryable, match="落盘残件"):
        indexer.ensure_queryable()
    r = indexer.rebuild("full")  # 重建：隔离 + 重新嵌入（无 .build 可续传时诚实全嵌）
    assert r["mode"] == "full" and r["total"] == 2
    assert not list(env.idx.glob(".*.tmp")) and indexer.ensure_queryable() is None


def test_embed_texts_normalized_keeps_positions(monkeypatch):
    def fake_embed(texts):
        return [[float(len(t)), 1.0] for t in texts]  # 第一维 = 文本长度，可反查来源

    monkeypatch.setattr(indexer, "embed", fake_embed)
    texts = ["aaaaa", "b", "ccc", "dddddddd", "ee"]
    out = indexer.embed_texts_normalized(texts, batch=2)
    recovered = out[:, 0] / out[:, 1]
    assert np.allclose(recovered, [5, 1, 3, 8, 2])  # 位置 i 的向量来自 texts[i]
    assert np.allclose(np.linalg.norm(out, axis=1), 1.0)
    monkeypatch.setattr(indexer, "embed_dim", lambda: 8)
    assert indexer.embed_texts_normalized([], batch=2).shape == (0, 8)
