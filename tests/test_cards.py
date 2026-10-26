"""分析卡层回归：试点选取确定性、文件指纹缓存、增量不重复不重嵌、API 失败保留旧卡、
装配层原文回读与让位、开关默认关。

不加载真实模型/真实 LLM：cards._llm 打桩为"从提示词提取路径+内容前缀"的确定性
文本；嵌入用 FakeBackend（同 test_indexer，sha1 派生 8 维确定性向量）。语料与
索引目录全部指向 tmp_path，永不触碰真实总库与真实 index/。
"""

import json
from types import SimpleNamespace

import numpy as np
import pytest

import kb_rag.cards as cards
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

    def embed_documents(self, texts):
        self.seen.extend(texts)
        out = []
        for t in texts:
            v = np.frombuffer(__import__("hashlib").sha1(t.encode()).digest()[:8], dtype=np.uint8)
            out.append(v.astype(np.float32) + 1.0)
        return out


def _write(kb, rel, text):
    p = kb / rel
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(text, encoding="utf-8")


def _select(kb, rels):
    data = cards._load()
    data.setdefault("meta", {})["selection"] = rels
    cards._save(data)


@pytest.fixture
def env(tmp_path, monkeypatch):
    import kb_rag.config as config

    kb = tmp_path / "kb"
    (kb / "kb-rag/项目状态/参考资料").mkdir(parents=True)  # 容纳评测集路径
    idx = tmp_path / "index"
    for mod in (config, indexer, search, cards):
        monkeypatch.setattr(mod, "INDEX_DIR", idx)
    monkeypatch.setattr(config, "KB_PATH", kb)
    monkeypatch.setattr(cards, "KB_PATH", kb)
    monkeypatch.setenv("KB_CARDS", "1")
    fake = FakeBackend()
    monkeypatch.setattr(indexer, "get_backend", lambda: fake)
    monkeypatch.setattr(indexer, "embed_identity", lambda: "fake:fake-model:path+heading:raw")
    monkeypatch.setattr(indexer, "doc_identity", lambda: "fake:fake-model:path+heading")
    monkeypatch.setattr(indexer, "embed", fake.embed_documents)
    monkeypatch.setattr(indexer, "embed_dim", lambda: 8)
    monkeypatch.setattr(search, "embed_queries", fake.embed_documents)

    calls: list[str] = []

    def fake_llm(prompt: str) -> str:
        calls.append(prompt)
        path = prompt.split("文档路径：")[1].split("\n")[0]
        content = prompt.split("文档内容：\n")[1]
        return f"【分析卡】{path}：{content[:20]}"

    monkeypatch.setattr(cards, "_llm", fake_llm)
    return SimpleNamespace(kb=kb, idx=idx, fake=fake, calls=calls)


def test_selection_golden_first_then_project_roundrobin(env):
    (env.kb / "kb-rag/项目状态/参考资料/eval_set.json").write_text(json.dumps(
        {"cases": [{"id": "t", "type": "single_hop", "question": "q",
                    "expected_any": ["g.md"]}]}, ensure_ascii=False), encoding="utf-8")
    _write(env.kb, "g.md", "金标文件。\n")
    for i in range(4):
        _write(env.kb, f"甲项目/f{i}.md", f"内容{i}。" + "x" * (100 - i))  # 项目内体积递减
        _write(env.kb, f"乙项目/f{i}.md", f"内容{i}。" + "y" * (100 - i))
    sel1 = cards.pilot_selection(env.kb, n=5)
    sel2 = cards.pilot_selection(env.kb, n=5)
    assert sel1 == sel2  # 确定性：同语料永远同一份清单
    assert sel1[0] == "g.md"  # 评测金标优先（保证试点效果可被评测集测出）
    proj = [p.split("/")[0] for p in sel1[1:]]
    assert proj == ["乙项目", "甲项目"] * 2  # 项目轮询、项目内按体积降序


def test_cache_no_double_llm_and_chunk_shape(env):
    _write(env.kb, "a.md", "# 标题\n\n正文内容。\n")
    hashes = indexer.scan_file_hashes(env.kb)
    chunks, stats = cards.ensure_cards(env.kb, hashes)
    assert stats["generated"] == 0 and chunks == []  # 无 selection：零 LLM 零卡片
    _select(env.kb, ["a.md"])
    chunks, stats = cards.ensure_cards(env.kb, hashes)
    assert stats["generated"] == 1 and len(env.calls) == 1
    c = chunks[0]
    assert c["path"] == "a.md" and c["heading"] == cards.CARD_HEADING and c["card"] == 1
    assert c["text"].startswith("【分析卡】a.md")
    chunks2, stats2 = cards.ensure_cards(env.kb, hashes)
    assert stats2["generated"] == 0 and chunks2 == chunks  # 指纹未变不重生成
    assert len(env.calls) == 1


def test_rebuild_with_cards_then_delta_no_duplicate_no_reembed(env):
    _write(env.kb, "a.md", "## 甲\n\n甲内容。\n")
    _write(env.kb, "b.md", "## 乙\n\n乙内容。\n")
    _select(env.kb, ["a.md", "b.md"])
    r = indexer.rebuild("full")
    assert r["total"] == 4 and r["cards"] == 2  # 2 文件块 + 2 卡片块
    env.fake.seen.clear()
    assert indexer.rebuild("delta")["mode"] == "clean"  # 语料与卡片都没变：clean 短路
    r3 = indexer.rebuild("full")  # 全量重切：卡片弃旧换新但哈希命中，零重嵌零翻倍
    assert r3["embedded"] == 0 and r3["reused"] == 4 and r3["total"] == 4
    chunks = json.loads((env.idx / "chunks.json").read_text())
    assert len([c for c in chunks if c.get("card")]) == 2


def test_stale_card_regenerated_on_delta(env):
    _write(env.kb, "a.md", "## 甲\n\n甲内容旧。\n")
    _select(env.kb, ["a.md"])
    indexer.rebuild("full")
    env.calls.clear()
    _write(env.kb, "a.md", "## 甲\n\n甲内容新。\n")
    r = indexer.rebuild("delta")
    assert r["mode"] == "delta" and r["total"] == 2  # 文件块 + 新卡
    assert len(env.calls) == 1  # 过期卡恰好重生成一次
    card_chunks = [c for c in json.loads((env.idx / "chunks.json").read_text()) if c.get("card")]
    assert len(card_chunks) == 1 and "新" in card_chunks[0]["text"]  # 旧卡不残留


def test_api_failure_keeps_old_card_and_builds(env, monkeypatch):
    _write(env.kb, "a.md", "## 甲\n\n甲内容旧。\n")
    _select(env.kb, ["a.md"])
    indexer.rebuild("full")

    def boom(prompt):
        raise RuntimeError("API down")

    monkeypatch.setattr(cards, "_llm", boom)
    _write(env.kb, "a.md", "## 甲\n\n甲内容改。\n")
    r = indexer.rebuild("delta")  # 生成失败不阻塞建索引
    assert r["total"] == 2
    card = [c for c in json.loads((env.idx / "chunks.json").read_text()) if c.get("card")][0]
    assert "旧" in card["text"]  # 宁旧勿缺：保留过期旧卡，不带失败内容进索引


def test_deleted_file_drops_card(env):
    _write(env.kb, "a.md", "## 甲\n\n甲内容。\n")
    _select(env.kb, ["a.md"])
    indexer.rebuild("full")
    (env.kb / "a.md").unlink()
    r = indexer.rebuild("delta")
    chunks = json.loads((env.idx / "chunks.json").read_text())
    assert r["total"] == 0 and not [c for c in chunks if c.get("card")]  # 卡随文件清除


def test_disabled_by_default(env, monkeypatch):
    monkeypatch.setenv("KB_CARDS", "0")
    _write(env.kb, "a.md", "## 甲\n\n甲内容。\n")
    _select(env.kb, ["a.md"])
    r = indexer.rebuild("full")
    assert r["cards"] == 0 and r["total"] == 1  # 开关关：卡片不进索引，也不花 LLM
    assert env.calls == []


def test_pending_gate_blocks_clean_shortcircuit(env):
    _write(env.kb, "a.md", "## 甲\n\n甲内容。\n")
    hashes = indexer.scan_file_hashes(env.kb)
    assert not cards.pending(hashes)  # 无 selection：不触发
    _select(env.kb, ["a.md"])
    assert cards.pending(hashes)  # 缺卡
    indexer.rebuild("full")  # 建索引时顺带把卡生成了
    assert not cards.pending(hashes) and indexer.rebuild("delta")["mode"] == "clean"
    data = cards._load()
    data["cards"]["a.md"]["card_hash"] = "stale"  # 双指纹语义：卡与内容的对应关系失效即过期
    cards._save(data)
    env.calls.clear()
    r = indexer.rebuild("delta")  # 卡片过期：clean 短路让位，增量继续并重生成
    assert r["mode"] == "delta" and len(env.calls) == 1


def test_resolve_hits_dedup_readback_and_passthrough(env):
    _write(env.kb, "doc.md", "frontmatter 占位\n\n原文正文第一行。\n")
    orig = {"path": "doc.md", "heading": "节", "text": "原文块", "idx": 0}
    card = {"path": "doc.md", "heading": cards.CARD_HEADING, "text": "卡片",
            "card": 1, "idx": 9}
    assert cards.resolve_hits([card, orig]) == [orig]  # 同文件原文已在：卡片让位
    out = cards.resolve_hits([card])  # 仅卡片命中：回读原文
    assert len(out) == 1 and out[0]["text"].startswith("frontmatter 占位")
    assert out[0]["card_readback"] is True
    assert cards.resolve_hits([orig]) == [orig]  # 无卡片：原样返回零开销
    (env.kb / "doc.md").unlink()
    assert cards.resolve_hits([card]) == []  # 原文已删：命中丢弃，不假答


def test_corrupt_cache_falls_back_to_bak_never_overwritten(env):
    _write(env.kb, "a.md", "## 甲\n\n甲内容。\n")
    _select(env.kb, ["a.md"])            # save #1
    _select(env.kb, ["a.md", "b.md"])    # save #2：bak = 上一次完好缓存
    (env.idx / "cards.json").write_text("{corrupt", encoding="utf-8")
    assert cards._load()["meta"]["selection"] == ["a.md"]  # 主文件损坏 → 回退 bak
    (env.idx / "cards.json.bak").write_text("also bad", encoding="utf-8")
    before = (env.idx / "cards.json").read_text()
    chunks, _stats = cards.ensure_cards(env.kb, indexer.scan_file_hashes(env.kb))
    assert chunks == []  # 双损坏：按无卡构建
    assert (env.idx / "cards.json").read_text() == before  # 不用空缓存覆盖销毁现场


def test_orphan_cards_cleaned_on_ensure(env):
    _write(env.kb, "a.md", "## 甲\n\n甲内容。\n")
    _select(env.kb, ["a.md"])
    data = cards._load()
    data["cards"]["orphan.md"] = {"file_hash": "x", "card": "孤儿卡"}
    cards._save(data)
    chunks, _stats = cards.ensure_cards(env.kb, indexer.scan_file_hashes(env.kb))
    assert [c["path"] for c in chunks] == ["a.md"]  # 掉出 selection 的孤儿卡不进索引
    assert "orphan.md" not in cards._load()["cards"]  # 缓存同步清理，不残留


def test_persistent_failure_backs_off_after_three(env, monkeypatch):
    _write(env.kb, "a.md", "## 甲\n\n甲内容。\n")
    _select(env.kb, ["a.md"])
    hashes = indexer.scan_file_hashes(env.kb)

    def boom(prompt):
        raise RuntimeError("API down")

    monkeypatch.setattr(cards, "_llm", boom)
    for _ in range(3):
        cards.ensure_cards(env.kb, hashes)  # 三次全失败：errors 计到 3
    env.calls.clear()
    cards.ensure_cards(env.kb, hashes)  # 第 4 次：退避跳过，不再烧 LLM
    assert env.calls == []
    entry = cards._load()["cards"]["a.md"]
    assert entry["errors"] == 3 and not entry.get("card")
    assert not cards.pending(hashes)  # 退避中的文件不触发 clean 短路失效


def test_retry_after_transient_failure(env, monkeypatch):
    """复验 P1 回归：文件变更后一次 API 抖动不得把过期旧卡伪装成"已同步"——
    API 恢复后的下一次构建必须立即重试并换上新卡。"""
    _write(env.kb, "a.md", "## 甲\n\n甲内容旧。\n")
    _select(env.kb, ["a.md"])
    indexer.rebuild("full")  # 卡=旧内容
    _write(env.kb, "a.md", "## 甲\n\n甲内容新。\n")

    def boom(prompt):
        raise RuntimeError("API 抖动")

    monkeypatch.setattr(cards, "_llm", boom)
    indexer.rebuild("delta")  # 生成失败：过期旧卡顶上（宁旧勿缺）
    card = [c for c in json.loads((env.idx / "chunks.json").read_text()) if c.get("card")][0]
    assert "旧" in card["text"]
    entry = cards._load()["cards"]["a.md"]
    assert entry["errors"] == 1 and entry["card_hash"] != entry["file_hash"]  # 未伪装成已同步

    monkeypatch.setattr(cards, "_llm", _deterministic_llm(env))
    env.calls.clear()
    r = indexer.rebuild("delta")  # API 恢复：立即重试
    assert r["mode"] == "delta" and len(env.calls) == 1
    cards_now = [c for c in json.loads((env.idx / "chunks.json").read_text()) if c.get("card")]
    assert len(cards_now) == 1 and "新" in cards_now[0]["text"]  # 旧卡被新卡替换
    assert not cards.pending(indexer.scan_file_hashes(env.kb))


def _deterministic_llm(env):
    """env fixture 同款确定性 LLM 桩（monkeypatch 撤销后无法直接取回原桩，重建一份）。"""
    def fake_llm(prompt: str) -> str:
        env.calls.append(prompt)
        path = prompt.split("文档路径：")[1].split("\n")[0]
        content = prompt.split("文档内容：\n")[1]
        return f"【分析卡】{path}：{content[:20]}"
    return fake_llm


def test_run_pilot_reselects_when_n_changes(env):
    _write(env.kb, "a.md", "## 甲\n\n甲内容。\n")
    _write(env.kb, "b.md", "## 乙\n\n乙内容。\n")
    _select(env.kb, ["a.md"])  # 旧清单 1 个文件
    out = cards.run_pilot(2)
    assert "已选取 2 个试点文件" in out  # N 变化 = 明确重选，不静默沿用旧清单
    assert cards._load()["meta"]["selection"] == ["a.md", "b.md"]


def test_hit_via_marks_card_only_hits():
    import kb_rag.evals as evals

    assert evals._hit_via(["d.md"], [{"path": "d.md", "card": 1}]) == "card"
    assert evals._hit_via(["d.md"], [{"path": "d.md"}, {"path": "d.md", "card": 1}]) == "orig"
    assert evals._hit_via(["d.md"], [{"path": "x.md"}]) == "-"


def test_enabled_by_default_after_rollout(monkeypatch):
    monkeypatch.delenv("KB_CARDS", raising=False)
    assert cards.enabled() is True  # 2026-10-26 全库铺开后默认开
    monkeypatch.setenv("KB_CARDS", "0")
    assert cards.enabled() is False


def test_all_mode_follows_corpus_and_new_files(env):
    """全库模式（增量恢复的关键）：清单动态跟随语料——新增文件自动建卡、删除自动清卡。"""
    _write(env.kb, "a.md", "## 甲\n\n甲内容。\n")
    _write(env.kb, "b.md", "## 乙\n\n乙内容。\n")
    data = cards._load()
    data["meta"]["selection_mode"] = "all"
    cards._save(data)
    chunks, stats = cards.ensure_cards(env.kb, indexer.scan_file_hashes(env.kb))
    assert stats["covered"] == 2
    _write(env.kb, "c.md", "## 丙\n\n丙内容。\n")  # 新文件入库
    hashes = indexer.scan_file_hashes(env.kb)
    assert cards.pending(hashes)  # add 时会自动触发建卡
    chunks2, stats2 = cards.ensure_cards(env.kb, hashes)
    assert stats2["generated"] == 1 and stats2["covered"] == 3
    (env.kb / "b.md").unlink()  # 文件删除
    chunks3, _ = cards.ensure_cards(env.kb, indexer.scan_file_hashes(env.kb))
    assert {c["path"] for c in chunks3} == {"a.md", "c.md"}
    assert "b.md" not in cards._load()["cards"]
