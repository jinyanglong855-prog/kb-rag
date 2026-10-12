"""scopes 权限解析单元测试：resolve / in_scope / effective / KB_WRITE_SCOPE 交集。"""

import json

import pytest

from kb_rag import scopes

RULES = {
    "default": {"read": ["*"], "write": ["*"]},
    "callers": {"finance": {"read": ["enterprise-agent-platform"], "write": ["enterprise-agent-platform"]}},
}


@pytest.fixture(autouse=True)
def _rules_file(tmp_path, monkeypatch):
    p = tmp_path / "scopes.json"
    p.write_text(json.dumps(RULES, ensure_ascii=False), encoding="utf-8")
    monkeypatch.setenv("KB_SCOPES", str(p))
    monkeypatch.delenv("KB_WRITE_SCOPE", raising=False)
    scopes._load.cache_clear()
    yield
    scopes._load.cache_clear()


def test_unknown_caller_falls_to_default():
    assert scopes.resolve("nobody", scopes.READ) == {"*"}
    assert scopes.resolve("nobody", scopes.WRITE) == {"*"}  # 单人现状：缺省全放开


def test_registered_caller_is_scoped():
    assert scopes.resolve("finance", scopes.READ) == {"enterprise-agent-platform"}
    assert scopes.resolve("finance", scopes.WRITE) == {"enterprise-agent-platform"}


def test_write_cap_intersects(monkeypatch):
    monkeypatch.setenv("KB_WRITE_SCOPE", "enterprise-agent-platform,私有区")
    # default 原本 "*"：与全局上限取交集后被收紧
    assert scopes.resolve("nobody", scopes.WRITE) == {"enterprise-agent-platform", "私有区"}
    # finance 已是具体集合：交集不变
    assert scopes.resolve("finance", scopes.WRITE) == {"enterprise-agent-platform"}
    # 上限不影响读
    assert scopes.resolve("finance", scopes.READ) == {"enterprise-agent-platform"}


def test_write_cap_cannot_expand(monkeypatch):
    monkeypatch.setenv("KB_WRITE_SCOPE", "私有区")
    assert scopes.resolve("finance", scopes.WRITE) == {"__none__"}  # 交集为空=无可写


def test_in_scope_top_dir_matching():
    assert scopes.in_scope("enterprise-agent-platform/项目状态/a.md", {"enterprise-agent-platform"})
    assert not scopes.in_scope("私有区/画像.md", {"enterprise-agent-platform"})
    assert scopes.in_scope("任何/路径.md", {"*"})


def test_in_scope_rejects_root_file_unless_star():
    assert not scopes.in_scope("STATUS.md", {"enterprise-agent-platform"})
    assert scopes.in_scope("STATUS.md", {"*"})


def test_effective_caller(monkeypatch):
    monkeypatch.delenv("KB_CALLER", raising=False)
    assert scopes.effective() == "local"
    assert scopes.effective("dsh") == "dsh"
    monkeypatch.setenv("KB_CALLER", " finance ")
    assert scopes.effective() == "finance"  # 去空白，容忍配置手误
