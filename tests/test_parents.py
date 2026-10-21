"""父块扩展回归：连续同键分组（含同名标题分节）、整段/中心窗口、同父去重、
无行号透传、行号越界防御；混合检索行号携带用 FakeBackend 真建索引+BM25 集成验证。

不加载真实模型：索引测试复用 test_indexer 的 FakeBackend（sha1 派生确定性向量），
父块单元测试只依赖手写的 chunks.json 夹具。语料与索引目录全部指向 tmp_path。
"""

import hashlib
import json
from types import SimpleNamespace

import numpy as np
import pytest

import kb_rag.indexer as indexer
import kb_rag.parents as parents
import kb_rag.search as search


class FakeBackend:
    kind = "fake"
    isolate_segments = False
    model = "fake-model"
    dims = None
    dim = 8

    def embed_documents(self, texts):
        out = []
        for t in texts:
            v = np.frombuffer(hashlib.sha1(t.encode()).digest()[:8], dtype=np.uint8)
            out.append(v.astype(np.float32) + 1.0)
        return out


def _mk(path, heading, i, size):
    """块文本带唯一标记前缀：断言窗口内容时用 startswith/in 判定成员归属。"""
    return {"path": path, "heading": heading, "text": f"{path}#{heading}#{i}:" + "字" * size}


def _fixture_chunks():
    # a.md：引言(1块) / 用法(3块) / 备注(2块) / 用法·重复标题(2块)；b.py：源码(3块)
    return [
        _mk("a.md", "引言", 0, 100),
        _mk("a.md", "用法", 1, 200), _mk("a.md", "用法", 2, 200), _mk("a.md", "用法", 3, 200),
        _mk("a.md", "备注", 4, 200), _mk("a.md", "备注", 5, 200),
        _mk("a.md", "用法", 6, 200), _mk("a.md", "用法", 7, 200),
        _mk("b.py", "源码", 8, 400), _mk("b.py", "源码", 9, 400), _mk("b.py", "源码", 10, 400),
    ]


@pytest.fixture
def idx(tmp_path, monkeypatch):
    import kb_rag.config as config

    d = tmp_path / "index"
    d.mkdir(parents=True)
    monkeypatch.setattr(config, "INDEX_DIR", d)
    monkeypatch.setattr(parents, "INDEX_DIR", d)
    (d / "chunks.json").write_text(
        json.dumps(_fixture_chunks(), ensure_ascii=False), encoding="utf-8")
    return d


def test_runs_grouping_and_repeated_heading(idx):
    # 同名标题在不同位置 = 两个父块（按连续同键分组，不按键全库合并）
    assert parents.parent_runs(json.loads((idx / "chunks.json").read_text())) == \
        [0, 1, 1, 1, 2, 2, 3, 3, 4, 4, 4]


def test_expand_returns_full_parent_within_budget(idx):
    hits = [{"path": "a.md", "heading": "用法", "text": _mk("a.md", "用法", 1, 200)["text"],
             "idx": 1, "score": 1.0}]
    out = parents.expand(hits, budget=1000)
    assert len(out) == 1 and out[0]["parent"] is True
    assert out[0]["text"] == "\n\n".join(
        _mk("a.md", "用法", i, 200)["text"] for i in (1, 2, 3))
    assert out[0]["path"] == "a.md" and out[0]["score"] == 1.0  # 其余字段透传


def test_expand_window_centered_on_hit_over_budget(idx):
    # b.py 父块 1200 字 > 预算 1000：命中中间块(9) → 中心+左邻(8,9)=800，右邻放不下
    hits = [{"path": "b.py", "heading": "源码", "text": _mk("b.py", "源码", 9, 400)["text"],
             "idx": 9}]
    out = parents.expand(hits, budget=1000)
    text = out[0]["text"]
    assert len(text) <= 1000
    assert _mk("b.py", "源码", 9, 400)["text"] in text  # 命中块必在窗口内
    assert _mk("b.py", "源码", 8, 400)["text"] in text
    assert _mk("b.py", "源码", 10, 400)["text"] not in text


def test_expand_dedup_same_parent(idx):
    two_same = [{"path": "a.md", "heading": "用法", "text": "x", "idx": 1},
                {"path": "a.md", "heading": "用法", "text": "y", "idx": 2}]
    assert len(parents.expand(two_same, budget=1000)) == 1
    cross = [{"path": "a.md", "heading": "用法", "text": "x", "idx": 1},
             {"path": "a.md", "heading": "备注", "text": "y", "idx": 4}]
    assert len(parents.expand(cross, budget=1000)) == 2


def test_expand_first_hit_defines_window(idx):
    # 同父块两命中去重时，窗口由更靠前（更相关）的命中定义
    hits = [{"path": "a.md", "heading": "用法", "text": "b", "idx": 2},
            {"path": "a.md", "heading": "用法", "text": "a", "idx": 1}]
    out = parents.expand(hits, budget=500)
    assert len(out) == 1
    assert _mk("a.md", "用法", 1, 200)["text"] in out[0]["text"]  # 中心(2)+左邻(1)=400
    assert _mk("a.md", "用法", 3, 200)["text"] not in out[0]["text"]


def test_expand_passthrough_without_idx(idx):
    hits = [{"path": "x.md", "heading": "h", "text": "t"}]
    assert parents.expand(hits) == hits


def test_expand_out_of_range_idx(idx):
    hits = [{"path": "x.md", "heading": "h", "text": "t", "idx": 99},
            {"path": "x.md", "heading": "h", "text": "t", "idx": "bad"}]
    assert parents.expand(hits) == hits
    assert parents.expand([]) == []


def test_expand_rejects_stale_idx(idx):
    # 行号指向的块与命中块不同文件/不同节（并发增量重建换序）：原样透传不展开——
    # scope 只在命中时按 path 判过，展开成无关文件内容即越权面（对抗审查 P1-1）
    hits = [{"path": "a.md", "heading": "用法", "text": "命中原文", "idx": 9}]
    out = parents.expand(hits, budget=1000)
    assert out == hits  # b.py 的块 9 与命中 (a.md, 用法) 不匹配：不展开、不改写


TWO_SECTION = "# 标题\n\n## 甲\n\n" + "甲段落。\n\n" * 400 + "\n## 乙\n\n" + "乙段落。\n\n" * 400


@pytest.fixture
def env(tmp_path, monkeypatch):
    """与 test_indexer.env 同款：FakeBackend 建真索引；另补 hybrid/parents 的 INDEX_DIR。"""
    import kb_rag.config as config
    import kb_rag.hybrid as hybrid

    kb = tmp_path / "kb"
    kb.mkdir()
    d = tmp_path / "index"
    for mod in (config, indexer, search, hybrid, parents):
        monkeypatch.setattr(mod, "INDEX_DIR", d)
    monkeypatch.setattr(config, "KB_PATH", kb)
    fake = FakeBackend()
    identity = "fake:fake-model:path+heading:raw"
    monkeypatch.setattr(indexer, "get_backend", lambda: fake)
    monkeypatch.setattr(indexer, "embed_identity", lambda: identity)
    monkeypatch.setattr(indexer, "doc_identity", lambda: "fake:fake-model:path+heading")
    monkeypatch.setattr(indexer, "embed", fake.embed_documents)
    monkeypatch.setattr(indexer, "embed_dim", lambda: 8)
    monkeypatch.setattr(search, "embed_queries", fake.embed_documents)
    return SimpleNamespace(kb=kb, idx=d, fake=fake, identity=identity)


def test_hybrid_hits_carry_idx_and_expand_works(env):
    (env.kb / "a.md").write_text(TWO_SECTION, encoding="utf-8")
    r = indexer.rebuild("full")
    assert r["mode"] == "full"
    from kb_rag.hybrid import build_bm25_index, hybrid_query

    build_bm25_index()
    hits = hybrid_query("甲段落", k=3)
    assert hits and all(isinstance(h["idx"], int) and 0 <= h["idx"] < r["total"] for h in hits)
    out = parents.expand(hits)
    assert all(h.get("parent") for h in out)
    # 假向量是随机噪声，不预设哪节排第一；只断言结构：每条命中的原文都在其
    # 父块文本里，且父块含兄弟块（严格长于命中块）；同父块重复命中被去重
    by_idx = {h["idx"]: h for h in out}
    assert len(out) <= len(hits)
    for h in hits:
        p = by_idx.get(h["idx"])
        if p is None:  # 同父块被更靠前的命中去重
            continue
        assert h["text"] in p["text"] and len(p["text"]) > len(h["text"])


def test_runs_stay_per_file_after_delta_rebuild(env):
    # 增量重建 = kept_chunks + new_chunks 拼接（新/变更文件整体后移）：逐文件仍连续，
    # 父块分组不得跨文件（对抗审查 P1 验证面）
    (env.kb / "a.md").write_text("# A\n\n## 甲\n\n甲内容。\n\n## 乙\n\n乙内容。\n", encoding="utf-8")
    (env.kb / "b.md").write_text("# B\n\n## 丙\n\n丙内容。\n", encoding="utf-8")
    assert indexer.rebuild("full")["mode"] == "full"
    (env.kb / "b.md").write_text("# B\n\n## 丙\n\n丙内容改。\n\n## 丁\n\n丁内容。\n", encoding="utf-8")
    assert indexer.rebuild("delta")["mode"] == "delta"
    chunks = json.loads((env.idx / "chunks.json").read_text())
    runs = parents.parent_runs(chunks)
    for r in set(runs):
        members = [i for i, x in enumerate(runs) if x == r]
        assert len({chunks[i]["path"] for i in members}) == 1  # 父块不跨文件
        assert members == list(range(members[0], members[-1] + 1))  # 行号连续
