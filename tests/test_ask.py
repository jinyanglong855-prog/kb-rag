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
    out = ask_mod.ask("这个问法不含状态词")
    assert "本次按现有索引回答" in out and "回答内容 [1]" in out
    assert "a.md #h" in out  # 来源脚注照常


def test_truly_unqueryable_still_refuses(wired, monkeypatch):
    def boom():
        raise indexer.IndexNotQueryable("向量空间不一致，已拒绝")

    monkeypatch.setattr(indexer, "ensure_queryable", boom)
    assert ask_mod.ask("这个问法不含状态词") == "向量空间不一致，已拒绝"
