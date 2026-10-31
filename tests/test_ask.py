"""ask 装配层回归：索引过期且增量被拒（旧格式索引）时照常回答带提示，真正不可查时才拒答。

不打真实 LLM/检索：全部 monkeypatch，秒级完成。2026-10-21 教训固化——"不能增量"曾被
错写成"不能查询"，生产 ask 因索引长期过期而整体哑火。
"""

from types import SimpleNamespace

import pytest

import kb_rag.ask as ask_mod
import kb_rag.indexer as indexer


@pytest.fixture
def wired(monkeypatch):
    monkeypatch.setattr(ask_mod, "LLM_API_KEY", "fake-key")
    monkeypatch.setattr(ask_mod.scopes, "resolve", lambda caller, kind: {"*"})
    monkeypatch.setattr(ask_mod, "staleness",
                        lambda: {"mode": "stale", "changed": 1, "deleted": 2, "total": 3})
    monkeypatch.setattr(ask_mod, "sync_incremental",
                        lambda: {"mode": "mismatch", "index_identity": "<旧版索引>",
                                 "config_identity": "cfg"})
    monkeypatch.setattr(ask_mod, "_alias_sources", lambda q, s: {})
    monkeypatch.setattr(ask_mod.rewrite, "enabled", lambda: False)
    monkeypatch.setattr(ask_mod, "PARENT_RETRIEVE", False)
    monkeypatch.setattr(ask_mod, "hybrid_query",
                        lambda q, k: [{"path": "a.md", "heading": "h", "text": "正文", "score": 1.0}])

    class FakeClient:
        def __init__(self, **kw):
            self.chat = SimpleNamespace(completions=SimpleNamespace(create=self._create))

        def _create(self, **kw):
            return SimpleNamespace(choices=[SimpleNamespace(
                message=SimpleNamespace(content="回答内容 [1]"))])

    monkeypatch.setattr(ask_mod, "OpenAI", FakeClient)


def test_delta_refused_but_queryable_answers_with_notice(wired, monkeypatch):
    monkeypatch.setattr(indexer, "ensure_queryable", lambda: None)  # 旧索引仅查询接管放行
    synced = []
    monkeypatch.setattr(ask_mod, "sync_incremental",
                        lambda: synced.append(1) or {"mode": "mismatch", "index_identity": "<旧版索引>",
                                                     "config_identity": "cfg"})
    out = ask_mod.ask("这个问法不含状态词")
    assert synced, "过期且未暂停时必须尝试增量补齐（此断言防测试走进暂停分支空转）"
    assert "本次按现有索引回答" in out and "回答内容 [1]" in out
    assert "a.md #h" in out  # 来源脚注照常


def test_truly_unqueryable_still_refuses(wired, monkeypatch):
    def boom():
        raise indexer.IndexNotQueryable("向量空间不一致，已拒绝")

    monkeypatch.setattr(indexer, "ensure_queryable", boom)
    assert ask_mod.ask("这个问法不含状态词") == "向量空间不一致，已拒绝"


def test_ingest_paused_never_auto_syncs(wired, monkeypatch):
    """入库暂停（KB_INGEST_PAUSED=1）：索引过期也绝不自动注入语料，只带提示照常回答。

    2026-10-28 教训固化——暂停若只写在操作纪律里，ask 的 L2 兜底会无视它
    （实测暂停期间一次生产冒烟即自动入库 +589 块）。
    """
    synced = []
    monkeypatch.setattr(ask_mod, "sync_incremental",
                        lambda: synced.append(1) or {"mode": "delta", "added": 1, "seconds": 0.1})
    monkeypatch.setattr(ask_mod, "INGEST_PAUSED", True)
    monkeypatch.setattr(indexer, "ensure_queryable", lambda: None)
    out = ask_mod.ask("这个问法不含状态词")
    assert synced == []  # 暂停期间零注入
    assert "增量入库处于暂停" in out and "回答内容 [1]" in out  # 回答不被阻塞
